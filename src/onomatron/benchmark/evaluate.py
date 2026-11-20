"""Final evaluation (benchmark): retrain each model with its tuned best params
(or the study's best, or constructor defaults) for `final_steps`, then log the
full metric set to one MLflow run per model in the "onomatron-benchmark"
experiment: val/test NLL, size, train wall time + peak memory, and inference
speed/memory. Heavy artifacts (checkpoint, samples, loss curves) are logged
here and only here — the test split is touched only at this stage.
"""

import logging
import time
from typing import Any

import optuna
import plotly.graph_objects as go
import torch

from onomatron.benchmark.config import BenchmarkConfig
from onomatron.benchmark.dataset import build_model, load_benchmark_data
from onomatron.benchmark.progress import BenchProgress
from onomatron.benchmark.spaces import default_params
from onomatron.benchmark.tracking import (
    RunLogger,
    get_device,
    init_mlflow,
    measure_peak_mb,
    model_size_mb,
    n_parameters,
    time_sampling,
)
from onomatron.utils import TrainHistory, train

logger = logging.getLogger(__name__)

EVAL_EXPERIMENT = "onomatron-benchmark"


def run_evaluation(
    models: list[str],
    cfg: BenchmarkConfig,
    params_map: dict[str, dict[str, Any]] | None = None,
    progress: BenchProgress | None = None,
) -> dict[str, dict[str, float]]:
    """Evaluate every model in `models`; return {model: key metrics}.

    Params resolution order: explicit `params_map` (from a tuning run) >
    best params of the persisted Optuna study > constructor defaults.
    Progress goes to `progress` if given, else a local single bar is created.
    """
    init_mlflow(cfg)
    device = get_device(cfg.device)
    logger.info("evaluating %d models on device=%s", len(models), device)
    if progress is None:
        with BenchProgress(len(models), desc="eval") as progress:
            return _eval_all(models, cfg, device, params_map, progress)
    return _eval_all(models, cfg, device, params_map, progress)


def _eval_all(
    models: list[str],
    cfg: BenchmarkConfig,
    device: torch.device,
    params_map: dict[str, dict[str, Any]] | None,
    progress: BenchProgress,
) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    for model in models:
        progress.start_model(model, 1, cfg.final_steps)
        params = _resolve_params(model, cfg, params_map)
        results[model] = _evaluate_model(model, cfg, device, params, progress)
        progress.trial_done()
    return results


def _resolve_params(
    model: str,
    cfg: BenchmarkConfig,
    params_map: dict[str, dict[str, Any]] | None,
) -> dict[str, Any]:
    if params_map and model in params_map:
        return params_map[model]
    try:
        study = optuna.load_study(study_name=model, storage=cfg.optuna_storage)
        return dict(study.best_params)
    except KeyError, ValueError, RuntimeError:
        # KeyError: no such study; ValueError/RuntimeError: study exists but has
        # no COMPLETE trial (all pruned) — best_params would raise.
        logger.info("no usable optuna study for %s, using constructor defaults", model)
        return default_params(model)


def _evaluate_model(
    model: str,
    cfg: BenchmarkConfig,
    device: torch.device,
    params: dict[str, Any],
    progress: BenchProgress,
) -> dict[str, float]:
    seed = cfg.seed
    with RunLogger(
        cfg,
        EVAL_EXPERIMENT,
        run_name=f"eval-{model}",
        tags={"model": model, "kind": "eval", "device": str(device)},
    ) as run:
        run.params(
            {
                **params,
                "model": model,
                "seed": seed,
                "final_steps": cfg.final_steps,
                "patience": cfg.patience,
                "eval_every": cfg.eval_every,
            }
        )
        data = load_benchmark_data(model, cfg, params.get("block_size", 2), device)
        t0 = time.perf_counter()
        net = build_model(model, data.vocab, params, seed, device, data.names_train)
        run.metric("n_parameters", n_parameters(net))
        run.metric("model_size_mb", model_size_mb(net))

        if model == "ngram":
            # "training" a count model = fit (in build_model) + val NLL
            val_nll = net.nll(data.names_val)
            wall = time.perf_counter() - t0
            test_nll = net.nll(data.names_test)
            inf = time_sampling(
                net, cfg.n_inference_samples, seed, cfg.inference_max_len
            )
            run.metrics(
                {
                    "val_nll": val_nll,
                    "test_nll": test_nll,
                    "train_wall_time_s": wall,
                    "inference_names_per_sec": inf["inference_names_per_sec"],
                    "inference_chars_per_sec": inf["inference_chars_per_sec"],
                    "inference_peak_rss_mb": inf["inference_peak_rss_mb"],
                }
            )
            run.text("samples.txt", "\n".join(inf["samples"]) + "\n")
            logger.info("eval %s: val=%.4f test=%.4f", model, val_nll, test_nll)
            return {
                "val_nll": val_nll,
                "test_nll": test_nll,
                "train_wall_time_s": wall,
                "inference_names_per_sec": inf["inference_names_per_sec"],
            }

        opt = torch.optim.AdamW(
            net.parameters(), lr=params["lr"], weight_decay=params["weight_decay"]
        )

        def do_train() -> TrainHistory:
            return train(
                net,
                data.x_train,
                data.y_train,
                opt,
                steps=cfg.final_steps,
                batch_size=params["batch_size"],
                x_val=data.x_val,
                y_val=data.y_val,
                eval_every=cfg.eval_every,
                clip_grad=params["clip_grad"],
                patience=cfg.patience,
                seed=seed,
                on_eval=lambda step, _v: progress.set_step(step),
                progress=True,  # per-model train bar, erased at run end
                bar_leave=False,
            )

        t0 = time.perf_counter()  # reset: wall time covers the training loop only
        history, train_peak_mb = measure_peak_mb(device, do_train)
        wall = time.perf_counter() - t0
        _, val_fwd_peak_mb = measure_peak_mb(
            device, lambda: net.nll(data.x_val, data.y_val)
        )
        val_nll = history.best[1]
        test_nll = net.nll(data.x_test, data.y_test)
        steps_done = len(history.train_losses)
        examples_per_sec = steps_done * params["batch_size"] / wall
        inf = time_sampling(net, cfg.n_inference_samples, seed, cfg.inference_max_len)

        for i in range(0, len(history.train_losses), cfg.eval_every):
            run.metric("train_loss", history.train_losses[i], step=i)
        for step, v in history.val:
            run.metric("val_nll", v, step=step)
        run.metrics(
            {
                "val_nll": val_nll,
                "test_nll": test_nll,
                "best_step": history.best[0],
                "train_wall_time_s": wall,
                "train_peak_mem_mb": train_peak_mb,
                "val_forward_peak_mem_mb": val_fwd_peak_mb,
                "examples_per_sec": examples_per_sec,
                "inference_names_per_sec": inf["inference_names_per_sec"],
                "inference_chars_per_sec": inf["inference_chars_per_sec"],
                "inference_peak_rss_mb": inf["inference_peak_rss_mb"],
            }
        )

        ckpt = run.tmp_path("model.pt")
        torch.save(net.state_dict(), ckpt)
        run.artifact(ckpt)
        run.text("samples.txt", "\n".join(inf["samples"]) + "\n")
        curves = run.tmp_path("curves.html")
        _write_curves_html(history, curves)
        run.artifact(curves)

        logger.info(
            "eval %s: val=%.4f test=%.4f "
            "(%.1fs, %.1f MB peak, %.0f ex/s, %.0f names/s)",
            model,
            val_nll,
            test_nll,
            wall,
            train_peak_mb,
            examples_per_sec,
            inf["inference_names_per_sec"],
        )
        return {
            "val_nll": val_nll,
            "test_nll": test_nll,
            "train_wall_time_s": wall,
            "inference_names_per_sec": inf["inference_names_per_sec"],
        }


def _write_curves_html(history: TrainHistory, path: Any) -> None:
    """Train loss curve + val NLL markers as a standalone HTML artifact."""
    fig = go.Figure()
    fig.add_scatter(
        x=list(range(len(history.train_losses))),
        y=history.train_losses,
        mode="lines",
        name="train",
    )
    if history.val:
        steps, nlls = zip(*history.val, strict=True)
        fig.add_scatter(x=list(steps), y=list(nlls), mode="markers", name="val")
    fig.update_layout(xaxis_title="step", yaxis_title="loss / NLL")
    fig.write_html(str(path))
