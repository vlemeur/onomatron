"""Optuna hyperparameter tuning: one persistent study per model (SQLite, so a
run can be resumed), TPE sampler, ASHA pruning on the val NLL reported at
every evaluation. Every trial is also one MLflow run (params + val_nll/step +
train_loss/step + wall time + peak memory); heavy artifacts (checkpoint,
samples) are logged only when the trial improves on the best so far.

Objective: minimize the best val NLL (the test split is never touched here).
"""

import logging
import time
from math import inf
from typing import Any

import optuna
import torch

from onomatron.benchmark.config import BenchmarkConfig
from onomatron.benchmark.dataset import build_model, load_benchmark_data
from onomatron.benchmark.progress import BenchProgress
from onomatron.benchmark.spaces import default_params, search_space
from onomatron.benchmark.tracking import (
    RunLogger,
    get_device,
    init_mlflow,
    measure_peak_mb,
    model_size_mb,
    n_parameters,
    time_sampling,
)
from onomatron.utils import train

logger = logging.getLogger(__name__)

TUNING_EXPERIMENT = "onomatron-tuning"


def run_tuning(
    models: list[str], cfg: BenchmarkConfig, progress: BenchProgress | None = None
) -> dict[str, dict[str, Any]]:
    """Tune every model in `models`; return {model: best_params}.

    Progress goes to `progress` if given, else a local single bar is created.
    """
    init_mlflow(cfg)
    device = get_device(cfg.device)
    logger.info(
        "tuning %d models on device=%s (%d trials each)",
        len(models),
        device,
        cfg.n_trials,
    )
    if progress is None:
        with BenchProgress(len(models) * cfg.n_trials) as progress:
            return _tune_all(models, cfg, device, progress)
    return _tune_all(models, cfg, device, progress)


def _tune_all(
    models: list[str],
    cfg: BenchmarkConfig,
    device: torch.device,
    progress: BenchProgress,
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for model in models:
        progress.start_model(model, cfg.n_trials, cfg.steps)
        results[model] = _tune_model(model, cfg, device, progress)
    return results


def _tune_model(
    model: str,
    cfg: BenchmarkConfig,
    device: torch.device,
    progress: BenchProgress,
) -> dict[str, Any]:
    """Run (or resume) the Optuna study for one model; return its best params."""
    study = optuna.create_study(
        study_name=model,
        storage=cfg.optuna_storage,
        direction="minimize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=cfg.seed),
        pruner=optuna.pruners.SuccessiveHalvingPruner(
            min_resource=cfg.eval_every, reduction_factor=cfg.reduction_factor
        ),
    )
    best_so_far = [inf]  # mutable cell shared by every trial of this study

    def objective(trial: optuna.Trial) -> float:
        params = search_space(model, trial, cfg)
        seed = cfg.seed + trial.number  # reproducible init per trial
        result: float | None = None  # None = pruned by ASHA mid-training
        with RunLogger(
            cfg,
            TUNING_EXPERIMENT,
            run_name=f"{model}-trial-{trial.number}",
            tags={
                "model": model,
                "kind": "tuning",
                "optuna_study": model,
                "optuna_trial_number": trial.number,
                "device": str(device),
            },
        ) as run:
            run.params(
                {
                    **params,
                    "model": model,
                    "seed": seed,
                    "steps": cfg.steps,
                    "patience": cfg.patience,
                    "eval_every": cfg.eval_every,
                }
            )
            data = load_benchmark_data(model, cfg, params.get("block_size", 2), device)
            net = build_model(model, data.vocab, params, seed, device, data.names_train)

            if model == "ngram":
                # Count-based: no training loop, no pruning — val NLL directly.
                val_nll = net.nll(data.names_val)
                run.metric("val_nll", val_nll)
                run.metric("n_parameters", 0)
                run.metric("model_size_mb", model_size_mb(net))
                logger.debug(
                    "ngram trial %d: n=%d smoothing=%.3g -> val_nll=%.4f",
                    trial.number,
                    params["n"],
                    params["smoothing"],
                    val_nll,
                )
                return val_nll  # no training loop: nothing to prune, exit now

            run.metric("n_parameters", n_parameters(net))
            opt = torch.optim.AdamW(
                net.parameters(),
                lr=params["lr"],
                weight_decay=params["weight_decay"],
            )

            def on_eval(step: int, val_nll: float) -> None:
                """ASHA: report the intermediate value; raise to abort training.
                The raise is caught in do_train so the MLflow run below can
                close cleanly (FINISHED + tag status=pruned) before Optuna
                gets its TrialPruned."""
                trial.report(val_nll, step)
                run.metric("val_nll", val_nll, step=step)
                best = best_so_far[0] if best_so_far[0] < inf else None
                progress.set_step(step, best)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            def do_train() -> Any:
                try:
                    return train(
                        net,
                        data.x_train,
                        data.y_train,
                        opt,
                        steps=cfg.steps,
                        batch_size=params["batch_size"],
                        x_val=data.x_val,
                        y_val=data.y_val,
                        eval_every=cfg.eval_every,
                        clip_grad=params["clip_grad"],
                        patience=cfg.patience,
                        seed=seed,
                        on_eval=on_eval,
                        progress=True,  # per-trial train bar, erased at trial end
                        bar_leave=False,
                    )
                except optuna.TrialPruned:
                    return None  # pruned mid-training

            t0 = time.perf_counter()
            history, peak_mb = measure_peak_mb(device, do_train)
            wall = time.perf_counter() - t0

            if history is None:
                run.tag("status", "pruned")
                logger.debug("%s trial %d: pruned by ASHA", model, trial.number)
            else:
                run.tag("status", "complete")
                for i in range(0, len(history.train_losses), cfg.eval_every):
                    run.metric("train_loss", history.train_losses[i], step=i)
                run.metrics(
                    {
                        "train_wall_time_s": wall,
                        "train_peak_mem_mb": peak_mb,
                        "best_step": history.best[0],
                        "best_val_nll": history.best[1],
                    }
                )
                if history.best[1] < best_so_far[0]:
                    best_so_far[0] = history.best[1]
                    _log_model_artifacts(run, model, trial.number, net, seed)
                logger.debug(
                    "%s trial %d: best val_nll=%.4f at step %d (%.1fs, peak %.1f MB)",
                    model,
                    trial.number,
                    history.best[1],
                    history.best[0],
                    wall,
                    peak_mb,
                )
                result = history.best[1]

        if result is None:
            raise optuna.TrialPruned()  # after the run closed cleanly
        return result

    def on_trial_done(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        """Advance the global bar after every trial (complete or pruned)."""
        complete = [
            t.value for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ]
        progress.trial_done(float(min(complete)) if complete else None)

    study.optimize(objective, n_trials=cfg.n_trials, callbacks=[on_trial_done])
    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not complete:
        # e.g. --trials 1 with the single trial pruned by ASHA: best_value /
        # best_params would raise — fall back to constructor defaults.
        logger.warning(
            "study %s: %d trials, none complete — using constructor defaults",
            model,
            len(study.trials),
        )
        return default_params(model)
    logger.info(
        "study %s: %d trials, best val_nll=%.4f, params=%s",
        model,
        len(study.trials),
        study.best_value,
        study.best_params,
    )
    return dict(study.best_params)


def _log_model_artifacts(
    run: RunLogger, model: str, trial_number: int, net: Any, seed: int
) -> None:
    """Log the checkpoint + a sample of generated names for a trial that is
    the best so far (heavy artifacts are not logged for every trial)."""
    if isinstance(net, torch.nn.Module):
        ckpt = run.tmp_path("model.pt")
        torch.save(net.state_dict(), ckpt)
        run.artifact(ckpt)
    samples = time_sampling(net, n=30, seed=seed, max_len=40)["samples"]
    run.text("samples.txt", "\n".join(samples) + "\n")
