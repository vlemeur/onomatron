"""Regression tests for TransformerName.sample.

The sampling context cannot grow past block_size (pos_emb has block_size
rows): sample() must cap max_len at block_size - 1 instead of crashing —
the old default max_len=40 crashed as soon as a sampled name exceeded
block_size chars.
"""

import pytest

from onomatron.data import Vocab
from onomatron.models.transformer import TransformerName


@pytest.fixture
def vocab() -> Vocab:
    return Vocab(["cosette", "javert", "marius", "gavroche"])


def make_model(vocab: Vocab, block_size: int = 4) -> TransformerName:
    return TransformerName(
        vocab, block_size=block_size, n_emb=8, n_head=2, n_layer=1, seed=0
    )


def test_sample_default_max_len_does_not_crash(vocab: Vocab) -> None:
    model = make_model(vocab, block_size=4)
    samples = model.sample(n=5, min_len=2, seed=0)  # max_len=None -> capped
    assert len(samples) == 5
    assert all(2 <= len(s) <= 3 for s in samples)  # max_len = block_size - 1


def test_sample_explicit_max_len_beyond_block_size_is_capped(vocab: Vocab) -> None:
    model = make_model(vocab, block_size=4)
    samples = model.sample(n=5, min_len=2, seed=0, max_len=40)
    assert all(len(s) <= 3 for s in samples)


def test_sample_is_seeded(vocab: Vocab) -> None:
    model = make_model(vocab)
    # eval mode, like every real consumer (published bundles are loaded in
    # eval): in train mode the embedding dropout makes sampling stochastic
    model.eval()
    assert model.sample(n=3, min_len=2, seed=42) == model.sample(
        n=3, min_len=2, seed=42
    )


def test_probs_shape_and_normalization(vocab: Vocab) -> None:
    import torch

    model = make_model(vocab)
    x = torch.tensor([[model.eos, model.eos, model.eos, model.eos]])
    p = model.probs(x)
    assert p.shape == (1, 4, len(vocab))
    assert torch.allclose(p.sum(dim=-1), torch.ones(1, 4), atol=1e-5)
