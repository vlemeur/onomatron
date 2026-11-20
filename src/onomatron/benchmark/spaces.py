"""Optuna search spaces: one architecture space per model + a shared training
space. Constraints of each architecture (power-of-2 block size, receptive
field, head divisibility) are encoded directly in the suggestions, so the
sampler only ever proposes valid configurations.
"""

from typing import Any

import optuna

from onomatron.benchmark.config import BenchmarkConfig

# Keys of the shared training space, filtered out when building the model.
TRAIN_KEYS: tuple[str, ...] = ("lr", "batch_size", "weight_decay", "clip_grad")


def model_params(model: str, trial: optuna.Trial) -> dict[str, Any]:
    """Architecture hyperparameters suggested for `model` on `trial`."""
    if model == "ngram":
        return {
            # NGramModel stores a DENSE count tensor of shape (vocab,)*n:
            # with vocab~38 that is 38**n int64 — n=6 is already ~24 GB.
            # Cap n at 5 (~635 MB) to keep trials safe on a laptop.
            "n": trial.suggest_int("n", 2, 5),
            "smoothing": trial.suggest_float("smoothing", 1e-2, 10.0, log=True),
        }
    if model == "linear":
        return {}  # no architecture hyperparameter (one weight matrix)
    if model == "mlp":
        return {
            "block_size": trial.suggest_int("block_size", 2, 8),
            "emb_dim": trial.suggest_int("emb_dim", 4, 32, log=True),
            "h_dim": trial.suggest_int("h_dim", 32, 512, log=True),
            "batch_norm": trial.suggest_categorical("batch_norm", [True, False]),
        }
    if model == "wavenet":
        return {
            # block_size must be a power of 2: each level halves the time axis
            "block_size": trial.suggest_categorical("block_size", [2, 4, 8, 16]),
            "emb_dim": trial.suggest_int("emb_dim", 8, 64, log=True),
            "h_dim": trial.suggest_int("h_dim", 32, 512, log=True),
        }
    if model == "dilated_wavenet":
        n_layers = trial.suggest_int("n_layers", 3, 6)
        return {
            # receptive field is 2**n_layers positions: it must cover block_size
            "n_layers": n_layers,
            "block_size": trial.suggest_int("block_size", 2, min(16, 2**n_layers)),
            "emb_dim": trial.suggest_int("emb_dim", 8, 64, log=True),
        }
    if model in ("rnn", "lstm", "gru"):
        return {
            "emb_dim": trial.suggest_int("emb_dim", 4, 32, log=True),
            "h_dim": trial.suggest_int("h_dim", 32, 512, log=True),
        }
    if model == "transformer":
        n_emb = trial.suggest_categorical("n_emb", [32, 64, 128, 256])
        heads = [h for h in (2, 4, 8, 16) if n_emb % h == 0]  # d_model % nhead == 0
        return {
            "block_size": trial.suggest_categorical("block_size", [4, 8, 16]),
            "n_emb": n_emb,
            "n_head": trial.suggest_categorical("n_head", heads),
            "n_layer": trial.suggest_int("n_layer", 2, 6),
        }
    if model == "rwkv":
        return {
            "n_emb": trial.suggest_categorical("n_emb", [32, 64, 128, 256]),
            "n_layer": trial.suggest_int("n_layer", 2, 6),
        }
    if model == "mamba":
        return {
            "n_emb": trial.suggest_categorical("n_emb", [32, 64, 128, 256]),
            "n_layer": trial.suggest_int("n_layer", 2, 6),
            "d_state": trial.suggest_categorical("d_state", [16, 32, 64]),
        }
    raise ValueError(f"unknown model: {model}")


def train_params(trial: optuna.Trial, cfg: BenchmarkConfig) -> dict[str, Any]:
    """Shared training hyperparameters (optimizer itself is fixed to AdamW)."""
    return {
        "lr": trial.suggest_float("lr", cfg.lr_low, cfg.lr_high, log=True),
        "batch_size": trial.suggest_categorical("batch_size", list(cfg.batch_sizes)),
        "weight_decay": trial.suggest_categorical(
            "weight_decay", list(cfg.weight_decays)
        ),
        "clip_grad": trial.suggest_categorical("clip_grad", list(cfg.clip_grads)),
    }


def search_space(
    model: str, trial: optuna.Trial, cfg: BenchmarkConfig
) -> dict[str, Any]:
    """Full parameter dict for one trial: architecture + training space."""
    params = model_params(model, trial)
    if model != "ngram":  # ngram has no gradient training
        params.update(train_params(trial, cfg))
    return params


def default_params(model: str) -> dict[str, Any]:
    """Constructor defaults + default training params, used by the final
    evaluation when no tuned study is available for `model`."""
    architectures: dict[str, dict[str, Any]] = {
        "ngram": {"n": 2, "smoothing": 1.0},
        "linear": {},
        "mlp": {"block_size": 3, "emb_dim": 2, "h_dim": 100, "batch_norm": True},
        "wavenet": {"block_size": 4, "emb_dim": 2, "h_dim": 100},
        "dilated_wavenet": {"block_size": 8, "emb_dim": 64, "n_layers": 4},
        "rnn": {"emb_dim": 2, "h_dim": 100},
        "lstm": {"emb_dim": 2, "h_dim": 100},
        "gru": {"emb_dim": 2, "h_dim": 100},
        "transformer": {"block_size": 8, "n_emb": 64, "n_head": 4, "n_layer": 2},
        "rwkv": {"n_emb": 64, "n_layer": 2},
        "mamba": {"n_emb": 64, "n_layer": 2, "d_state": 16},
    }
    train_defaults: dict[str, Any] = {
        "lr": 1e-3,
        "batch_size": 128,
        "weight_decay": 0.0,
        "clip_grad": None,
    }
    params = dict(architectures[model])
    if model != "ngram":
        params.update(train_defaults)
    return params
