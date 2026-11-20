"""MLflow tracking helpers + device-agnostic performance metrics.

MLflow runs on a SQLite backend (one file under data/benchmark/), artifacts in
data/benchmark/artifacts/<experiment>/. No print here: stdlib logging only,
basicConfig lives in the runner scripts.

Memory/speed metrics are device-agnostic:
- CUDA: native peak via torch.cuda.reset_peak_memory_stats/max_memory_allocated;
- MPS: no native peak counter, so a daemon thread polls
  torch.mps.current_allocated_memory() while the measured callable runs;
- CPU: process peak RSS via resource.getrusage (bytes on macOS, KiB on Linux).
Sampling speed is always measured on CPU because every model's sample() builds
its context tensors on CPU; a deepcopy of the model is moved there for the
measurement so the trained weights are untouched.
"""

import copy
import logging
import resource
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import mlflow
import torch
from torch import nn

from onomatron.benchmark.config import BenchmarkConfig

logger = logging.getLogger(__name__)


def get_device(name: str) -> torch.device:
    """Resolve a device name; "auto" picks cuda > mps > cpu."""
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def init_mlflow(cfg: BenchmarkConfig) -> None:
    """Point MLflow at the SQLite tracking URI and create the local dirs."""
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)
    cfg.tmp_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    logger.info("mlflow tracking uri: %s", cfg.mlflow_tracking_uri)


class RunLogger:
    """Thin context manager over mlflow.start_run with typed helpers.

    Each Optuna trial opens one run (nested inside nothing — the study itself
    lives in optuna.db); correlation is via the optuna_trial_number tag and
    the run name f"{model}-trial-{n}".
    """

    def __init__(
        self,
        cfg: BenchmarkConfig,
        experiment_name: str,
        run_name: str,
        tags: dict[str, Any] | None = None,
    ) -> None:
        self.cfg = cfg
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.tags = tags or {}
        self._cm: Any = None

    def __enter__(self) -> RunLogger:
        mlflow.set_tracking_uri(self.cfg.mlflow_tracking_uri)
        if mlflow.get_experiment_by_name(self.experiment_name) is None:
            mlflow.create_experiment(
                self.experiment_name,
                artifact_location=(self.cfg.artifacts_dir / self.experiment_name)
                .resolve()
                .as_uri(),
            )
        mlflow.set_experiment(self.experiment_name)
        self._cm = mlflow.start_run(run_name=self.run_name)
        self._cm.__enter__()
        if self.tags:
            mlflow.set_tags({k: str(v) for k, v in self.tags.items()})
        return self

    def __exit__(self, *exc: Any) -> Any:
        return self._cm.__exit__(*exc)

    def params(self, values: dict[str, Any]) -> None:
        mlflow.log_params({k: str(v) for k, v in values.items()})

    def metric(self, key: str, value: float, step: int | None = None) -> None:
        mlflow.log_metric(key, float(value), step=step)

    def metrics(self, values: dict[str, float], step: int | None = None) -> None:
        mlflow.log_metrics({k: float(v) for k, v in values.items()}, step=step)

    def tag(self, key: str, value: Any) -> None:
        mlflow.set_tag(key, str(value))

    def tmp_path(self, name: str) -> Path:
        """A scratch file inside the run's tmp dir (for artifacts to log)."""
        d = self.cfg.tmp_dir / self.run_name
        d.mkdir(parents=True, exist_ok=True)
        return d / name

    def artifact(self, path: Path, artifact_path: str | None = None) -> None:
        mlflow.log_artifact(str(path), artifact_path=artifact_path)

    def text(self, name: str, content: str) -> None:
        mlflow.log_text(content, name)


def _rss_peak_mb() -> float:
    """Peak RSS of this process in MB (ru_maxrss: bytes on macOS, KiB on Linux)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1e6 if sys.platform == "darwin" else peak / 1e3


def measure_peak_mb[T](device: torch.device, fn: Callable[[], T]) -> tuple[T, float]:
    """Run `fn` and return (result, peak memory in MB) on `device`.

    CUDA uses the native peak counter; MPS is polled by a daemon thread (no
    native peak counter exists); CPU falls back to the process peak RSS.
    """
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        result = fn()
        return result, torch.cuda.max_memory_allocated(device) / 1e6
    if device.type == "mps":
        peak = [0.0]
        stop = threading.Event()

        def poll() -> None:
            while not stop.is_set():
                peak[0] = max(peak[0], torch.mps.current_allocated_memory())
                stop.wait(0.005)

        thread = threading.Thread(target=poll, daemon=True)
        thread.start()
        try:
            result = fn()
        finally:
            stop.set()
            thread.join()
        return result, peak[0] / 1e6
    result = fn()
    return result, _rss_peak_mb()


def time_sampling(model: Any, n: int, seed: int, max_len: int) -> dict[str, Any]:
    """Sample `n` names on a CPU copy of `model` (sample() builds its context
    tensors on CPU) and return speed metrics + the sampled names.

    A short warmup sample runs first so one-time costs (imports, allocator
    growth) don't pollute the measurement.
    """
    m = copy.deepcopy(model).to("cpu") if isinstance(model, nn.Module) else model
    # Positional-embedding models (transformer) crash when the sampling
    # context grows past block_size (pos_emb has block_size rows).
    block = getattr(m, "block_size", None)
    if block is not None:
        max_len = min(max_len, block - 1)
    m.sample(n=8, seed=seed, max_len=max_len)  # warmup
    t0 = time.perf_counter()
    samples = m.sample(n=n, seed=seed, max_len=max_len)
    dt = time.perf_counter() - t0
    chars = sum(len(s) for s in samples)
    return {
        "samples": samples,
        "inference_names_per_sec": n / dt,
        "inference_chars_per_sec": chars / dt,
        "inference_peak_rss_mb": _rss_peak_mb(),
    }


def model_size_mb(model: Any) -> float:
    """Size in MB of the model itself (weights for nn.Modules, count tensor
    for the ngram model)."""
    if isinstance(model, nn.Module):
        return sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6
    return model.counts.numel() * model.counts.element_size() / 1e6


def n_parameters(model: Any) -> int:
    """Trainable parameter count (0 for the count-based ngram model)."""
    if isinstance(model, nn.Module):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return 0
