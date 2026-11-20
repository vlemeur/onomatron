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
from tqdm import tqdm

from onomatron.benchmark.config import BenchmarkConfig
from onomatron.benchmark.dataset import build_model, load_benchmark_data
from onomatron.benchmark.spaces import search_space
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


def run_tuning(models: list[str], cfg: BenchmarkConfig) -> dict[str, dict[str, Any]]:
    """Tune every model in `models`; return {model: best_params}."""
    init_mlflow(cfg)
    device = get_device(cfg.device)
    logger.info(
        "tuning %d models on device=%s (%d trials each)",
        len(models),
        device,
        cfg.n_trials,
    )
    results: dict[str, dict[str, Any]] = {}
    for model in tqdm(models, desc="models", unit="model"):
        results[model] = _tune_model(model, cfg, device)
    return results


def _tune_model(
    model: str, cfg: BenchmarkConfig, device: torch.device
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
                logger.info(
                    "ngram trial %d: n=%d smoothing=%.3g -> val_nll=%.4f",
                    trial.number,
                    params["n"],
                    params["smoothing"],
                    val_nll,
                )
                result = val_nll

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
                    )
                except optuna.TrialPruned:
                    return None  # pruned mid-training

            t0 = time.perf_counter()
            history, peak_mb = measure_peak_mb(device, do_train)
            wall = time.perf_counter() - t0

            if history is None:
                run.tag("status", "pruned")
                logger.info("%s trial %d: pruned by ASHA", model, trial.number)
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
                logger.info(
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

    # One progress bar per study: the callback fires after every trial
    # (complete or pruned), the inner train() bar shows the steps of the
    # running trial.
    trials_bar = tqdm(total=cfg.n_trials, desc=f"trials {model}", unit="trial")

    def on_trial_done(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        state = trial.state.name.lower()
        best = f"{study.best_value:.4f}" if study.best_value is not None else "—"
        trials_bar.set_postfix_str(f"last={state}, best val_nll={best}")
        trials_bar.update(1)

    study.optimize(objective, n_trials=cfg.n_trials, callbacks=[on_trial_done])
    trials_bar.close()
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
