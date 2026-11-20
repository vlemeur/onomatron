"""Configuration of the benchmark / hyperparameter-tuning pipeline.

`BenchmarkConfig` is the single source of truth for every default (budget,
pruning, device, paths…). The runner scripts stay thin: they only forward a
few CLI overrides into here. All experiment state lives under `data_dir`
(data/benchmark/), which is gitignored and regenerable.
"""

from pathlib import Path

from pydantic import BaseModel, ConfigDict

from onomatron.data import DEFAULT_NAMES_PATH

# Every model of onomatron.models, in increasing order of complexity.
MODELS: tuple[str, ...] = (
    "ngram",
    "linear",
    "mlp",
    "wavenet",
    "dilated_wavenet",
    "rnn",
    "lstm",
    "gru",
    "transformer",
    "rwkv",
    "mamba",
)

# Models trained by gradient descent; ngram is count-based (no optimizer).
GRADIENT_MODELS: tuple[str, ...] = tuple(m for m in MODELS if m != "ngram")


class BenchmarkConfig(BaseModel):
    """All knobs of the benchmark/tuning pipeline (pydantic, strict)."""

    model_config = ConfigDict(extra="forbid")

    # Tuning budget (per model)
    n_trials: int = 30
    steps: int = 5000  # max training steps per trial
    final_steps: int = 10000  # steps of the final evaluation run
    patience: int = 20  # early stopping: consecutive val evals without improvement
    eval_every: int = 100  # val NLL evaluation period (= ASHA min_resource)
    reduction_factor: int = 3  # ASHA reduction factor
    # Reproducibility / hardware
    seed: int = 42
    device: str = "auto"  # auto | cpu | mps | cuda
    # Shared training search space (AdamW itself is fixed, not suggested)
    batch_sizes: tuple[int, ...] = (64, 128, 256)
    lr_low: float = 1e-4
    lr_high: float = 1e-1
    weight_decays: tuple[float, ...] = (0.0, 1e-4, 1e-3)
    clip_grads: tuple[float | None, ...] = (1.0, 5.0, None)
    # Inference benchmark
    n_inference_samples: int = 200
    inference_max_len: int = 40
    # Data
    split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)
    names_path: Path = DEFAULT_NAMES_PATH
    # Experiment state (gitignored, regenerable)
    data_dir: Path = Path("data/benchmark")

    @property
    def optuna_storage(self) -> str:
        """Optuna storage URL (SQLite: resumable studies, one file)."""
        return f"sqlite:///{self.data_dir / 'optuna.db'}"

    @property
    def mlflow_tracking_uri(self) -> str:
        """MLflow tracking URI (SQLite backend, one file)."""
        return f"sqlite:///{self.data_dir / 'mlflow.db'}"

    @property
    def artifacts_dir(self) -> Path:
        """Root of the MLflow artifact store (one subdir per experiment)."""
        return self.data_dir / "artifacts"

    @property
    def tmp_dir(self) -> Path:
        """Scratch dir for artifacts before they are logged to MLflow."""
        return self.data_dir / "tmp"
