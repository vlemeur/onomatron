"""Shared fixtures: a tiny French-name dataset, its vocab, and the (x, y)
tensors in the two shapes the model families expect — context windows for
mlp/wavenet/linear, full sequences for rnn/lstm/gru."""

import pytest
import torch
import torch.nn.functional as F

from onomatron.data import Vocab, make_ngram_pairs, make_sequences

NAMES = [
    "cosette",
    "javert",
    "valjean",
    "marius",
    "gavroche",
    "fantine",
    "éponine",
    "thénardier",
]

# Context width shared by the context-window models (wavenet wants a power
# of 2, dilated_wavenet wants 2**n_layers >= block_size).
BLOCK_SIZE = 4


@pytest.fixture(scope="session")
def names() -> list[str]:
    return list(NAMES)


@pytest.fixture(scope="session")
def vocab(names: list[str]) -> Vocab:
    return Vocab(names)


@pytest.fixture(scope="session")
def ctx_xy(names: list[str], vocab: Vocab) -> tuple[torch.Tensor, torch.Tensor]:
    """Context windows: x (N, BLOCK_SIZE) of char indices -> y (N,) target."""
    pairs = make_ngram_pairs(names, vocab, BLOCK_SIZE + 1)
    return pairs[:, :-1], pairs[:, -1]


@pytest.fixture(scope="session")
def onehot_xy(names: list[str], vocab: Vocab) -> tuple[torch.Tensor, torch.Tensor]:
    """One-hot bigrams for LinearBigram: x (N, V) -> y (N,)."""
    pairs = make_ngram_pairs(names, vocab, 2)
    x = F.one_hot(pairs[:, 0], num_classes=len(vocab)).float()
    return x, pairs[:, 1]


@pytest.fixture(scope="session")
def seq_xy(names: list[str], vocab: Vocab) -> tuple[torch.Tensor, torch.Tensor]:
    """Full sequences (N, T) for the recurrent models (rnn/lstm/gru)."""
    return make_sequences(names, vocab)
