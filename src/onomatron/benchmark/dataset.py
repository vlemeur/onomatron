"""Dataset and model construction for the benchmark.

Three data families, matching the model interfaces:
- ngram: no tensors — fit() counts n-grams, nll() takes raw name lists;
- context models (linear, mlp, wavenet, dilated_wavenet): (N, block_size)
  index rows -> next-char target (N,); linear takes a one-hot row instead;
- sequence models (rnn, lstm, gru, transformer, rwkv, mamba): full sequences
  (N, T) from make_sequences; the transformer additionally truncates to its
  block_size (positional embeddings cover block_size positions only).
"""

import logging
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from onomatron.benchmark.config import BenchmarkConfig
from onomatron.benchmark.spaces import TRAIN_KEYS
from onomatron.data import (
    Vocab,
    load_names,
    make_ngram_pairs,
    make_sequences,
    split_names,
)
from onomatron.models.dilated_wavenet import DilatedWaveNet
from onomatron.models.gru import GRUName
from onomatron.models.linear import LinearBigram
from onomatron.models.lstm import LSTMName
from onomatron.models.mamba import MambaName
from onomatron.models.mlp import MLP
from onomatron.models.ngram import NGramModel
from onomatron.models.rnn import RNNName
from onomatron.models.rwkv import RWKVName
from onomatron.models.transformer import TransformerName
from onomatron.models.wavenet import WaveNet

logger = logging.getLogger(__name__)

CONTEXT_MODELS: tuple[str, ...] = ("linear", "mlp", "wavenet", "dilated_wavenet")
SEQUENCE_MODELS: tuple[str, ...] = (
    "rnn",
    "lstm",
    "gru",
    "transformer",
    "rwkv",
    "mamba",
)

MODEL_CLASSES: dict[str, type[nn.Module]] = {
    "linear": LinearBigram,
    "mlp": MLP,
    "wavenet": WaveNet,
    "dilated_wavenet": DilatedWaveNet,
    "rnn": RNNName,
    "lstm": LSTMName,
    "gru": GRUName,
    "transformer": TransformerName,
    "rwkv": RWKVName,
    "mamba": MambaName,
}


@dataclass
class BenchmarkData:
    """Train/val/test tensors (None for ngram) + raw name splits + vocab."""

    vocab: Vocab
    names_train: list[str]
    names_val: list[str]
    names_test: list[str]
    x_train: torch.Tensor | None = None
    y_train: torch.Tensor | None = None
    x_val: torch.Tensor | None = None
    y_val: torch.Tensor | None = None
    x_test: torch.Tensor | None = None
    y_test: torch.Tensor | None = None


def load_benchmark_data(
    model: str, cfg: BenchmarkConfig, block_size: int, device: torch.device
) -> BenchmarkData:
    """Load names, split them (seeded), build the vocab and the (x, y) tensors
    matching `model`'s input family, all on `device` (except ngram: CPU)."""
    names = load_names(cfg.names_path)
    names_train, names_val, names_test = split_names(names, cfg.split_ratios, cfg.seed)
    vocab = Vocab(names)
    data = BenchmarkData(vocab, names_train, names_val, names_test)
    if model == "ngram":
        return data  # count-based: no tensors, nll() takes name lists

    if model in CONTEXT_MODELS:
        # make_ngram_pairs returns (N, n) with the LAST column as the target,
        # so a block_size context needs n = block_size + 1 (notebook convention);
        # linear consumes a single context char directly (n=2).
        n = 2 if model == "linear" else block_size + 1

        def make(ns: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
            pairs = make_ngram_pairs(ns, vocab, n)
            if model == "linear":
                x = F.one_hot(pairs[:, 0], num_classes=len(vocab)).float()
            else:
                x = pairs[:, :-1]
            return x.to(device), pairs[:, -1].to(device)

    else:  # SEQUENCE_MODELS

        def make(ns: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
            x, y = make_sequences(ns, vocab)
            if model == "transformer" and x.shape[1] > block_size:
                x, y = x[:, :block_size], y[:, :block_size]
            return x.to(device), y.to(device)

    data.x_train, data.y_train = make(names_train)
    data.x_val, data.y_val = make(names_val)
    data.x_test, data.y_test = make(names_test)
    logger.info(
        "benchmark data for %s: train %s, val %s, test %s (device=%s)",
        model,
        data.x_train.shape,
        data.x_val.shape,
        data.x_test.shape,
        device,
    )
    return data


def build_model(
    model: str,
    vocab: Vocab,
    params: dict[str, Any],
    seed: int,
    device: torch.device,
    names_train: list[str],
) -> nn.Module | NGramModel:
    """Instantiate `model` with `params` (architecture keys only — training
    keys are filtered out) and `seed`; move nn.Modules to `device`. The ngram
    model is fitted on the training names directly (it has no seed/device)."""
    if model == "ngram":
        return NGramModel.fit(
            names_train, vocab, n=params["n"], smoothing=params["smoothing"]
        )
    kwargs = {k: v for k, v in params.items() if k not in TRAIN_KEYS}
    net = MODEL_CLASSES[model](vocab, seed=seed, **kwargs)
    return net.to(device)
