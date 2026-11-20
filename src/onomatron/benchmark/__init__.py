"""Benchmark and hyperparameter tuning for the name models.

Optuna (SQLite studies, TPE + ASHA, objective = val NLL) drives the search,
MLflow (SQLite backend) tracks every trial and every final evaluation. All
configuration defaults live in `BenchmarkConfig`; the runner scripts are thin
CLIs over `run()`.
"""

from typing import Any

from onomatron.benchmark.config import GRADIENT_MODELS, MODELS, BenchmarkConfig
from onomatron.benchmark.evaluate import run_evaluation
from onomatron.benchmark.progress import BenchProgress
from onomatron.benchmark.tune import run_tuning

__all__ = [
    "GRADIENT_MODELS",
    "MODELS",
    "BenchmarkConfig",
    "run",
    "run_evaluation",
    "run_tuning",
]


def run(
    mode: str = "all",
    models: list[str] | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    """Entry point used by scripts/benchmark.py.

    mode="tune": Optuna studies only; mode="eval": final benchmark runs only
    (params from `overrides`-free studies if present, else defaults);
    mode="all": tune then evaluate with the tuned best params.
    `overrides` are BenchmarkConfig fields (n_trials, steps, device…).
    """
    cfg = BenchmarkConfig(**overrides)
    models = list(models) if models else list(MODELS)
    # One bar for the whole run: n_trials per model in tuning + 1 eval per model.
    n_tuning = len(models) * cfg.n_trials if mode in ("tune", "all") else 0
    n_eval = len(models) if mode in ("eval", "all") else 0
    results: dict[str, Any] = {}
    with BenchProgress(total=n_tuning + n_eval) as progress:
        if mode in ("tune", "all"):
            results["tuning"] = run_tuning(models, cfg, progress)
        if mode in ("eval", "all"):
            params_map = results.get("tuning") if mode == "all" else None
            results["eval"] = run_evaluation(models, cfg, params_map, progress)
    return results
