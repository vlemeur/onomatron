"""Self-contained model bundles, for publishing (Hugging Face Hub) or local use.

A bundle is a directory holding exactly two files:

- ``config.json``: format tag, model name, class name, architecture params,
  seed, the vocab (``stoi`` mapping) and optional training metrics;
- ``model.pt``: the weights — a ``state_dict`` for ``nn.Module`` models, a
  small dict (``counts``/``n``/``smoothing``) for the count-based ngram model.

``load_bundle`` rebuilds the exact model from a bundle directory, and
``load_bundle_from_hub`` downloads one from a HF model repo. Only torch and
this package are needed — no optuna/mlflow/benchmark imports, so the same
code runs inside a published HF model repo with a minimal vendored copy of
``onomatron`` (see ``onomatron.publish.export.vendor_onomatron``).
"""

import importlib
import json
import logging
import shutil
from pathlib import Path
from typing import Any

import torch
from torch import nn

from onomatron.data import SENTINEL, Vocab

logger = logging.getLogger(__name__)

BUNDLE_FORMAT = "onomatron-model/1"
CONFIG_NAME = "config.json"
WEIGHTS_NAME = "model.pt"


def make_config(
    model: str,
    cls_name: str,
    params: dict[str, Any],
    vocab: Vocab,
    seed: int | None,
    metrics: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Build the config.json content of a bundle."""
    config: dict[str, Any] = {
        "format": BUNDLE_FORMAT,
        "model": model,
        "class": cls_name,
        "params": params,
        "seed": seed,
        "vocab": {"sentinel": SENTINEL, "stoi": vocab.stoi},
    }
    if metrics:
        config["metrics"] = metrics
    return config


def write_bundle(
    out_dir: str | Path, *, config: dict[str, Any], weights_path: str | Path
) -> Path:
    """Assemble a bundle from a config dict and an existing weights file."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / CONFIG_NAME).write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    shutil.copy(weights_path, out / WEIGHTS_NAME)
    logger.info("wrote bundle %s (%s + %s)", out, CONFIG_NAME, WEIGHTS_NAME)
    return out


def save_bundle(
    out_dir: str | Path,
    *,
    model: str,
    net: Any,
    params: dict[str, Any],
    vocab: Vocab,
    seed: int | None,
    metrics: dict[str, float] | None = None,
) -> Path:
    """Save a live model as a bundle (weights are serialized from `net`)."""
    config = make_config(
        model, type(net).__name__, params, vocab, seed, metrics=metrics
    )
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / CONFIG_NAME).write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if isinstance(net, nn.Module):
        torch.save(net.state_dict(), out / WEIGHTS_NAME)
    else:  # count-based ngram model: no state_dict, persist the counts
        torch.save(
            {"counts": net.counts, "n": net.n, "smoothing": net.smoothing},
            out / WEIGHTS_NAME,
        )
    logger.info("saved bundle %s (model=%s)", out, model)
    return out


def load_bundle(path: str | Path) -> tuple[Any, dict[str, Any]]:
    """Rebuild (model, config) from a bundle directory; the model is on CPU
    and in eval mode."""
    bundle = Path(path)
    config = json.loads((bundle / CONFIG_NAME).read_text(encoding="utf-8"))
    if config.get("format") != BUNDLE_FORMAT:
        raise ValueError(
            f"unsupported bundle format {config.get('format')!r} in {bundle} "
            f"(expected {BUNDLE_FORMAT!r})"
        )
    vocab = Vocab.from_stoi(config["vocab"]["stoi"])
    module = importlib.import_module(f"onomatron.models.{config['model']}")
    cls = getattr(module, config["class"])
    state = torch.load(bundle / WEIGHTS_NAME, map_location="cpu", weights_only=True)
    if issubclass(cls, nn.Module):
        net = cls(vocab, seed=config.get("seed"), **config["params"])
        net.load_state_dict(state)
        net.eval()
    else:  # count-based ngram model
        net = cls(vocab, **config["params"])
        net.counts = state["counts"]
    logger.info("loaded bundle from %s (model=%s)", bundle, config["model"])
    return net, config


def load_bundle_from_hub(
    repo_id: str,
    *,
    revision: str | None = None,
    token: str | None = None,
    cache_dir: str | Path | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Download config.json + model.pt from a HF model repo and load them."""
    from huggingface_hub import snapshot_download

    local = snapshot_download(
        repo_id=repo_id,
        allow_patterns=[CONFIG_NAME, WEIGHTS_NAME],
        revision=revision,
        token=token,
        cache_dir=cache_dir,
    )
    return load_bundle(local)
