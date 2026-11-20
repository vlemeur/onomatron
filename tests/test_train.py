"""Tests for the shared training loop (utils.train), on a tiny LinearBigram
so they run in milliseconds."""

import pytest
import torch
import torch.nn.functional as F

from onomatron.data import Vocab, make_ngram_pairs
from onomatron.models.linear import LinearBigram
from onomatron.utils import train

NAMES = ["cosette", "javert", "valjean", "marius", "gavroche", "fantine"]


@pytest.fixture
def vocab() -> Vocab:
    return Vocab(NAMES)


@pytest.fixture
def bigram(vocab: Vocab) -> tuple[torch.Tensor, torch.Tensor]:
    pairs = make_ngram_pairs(NAMES, vocab, 2)
    x = F.one_hot(pairs[:, 0], num_classes=len(vocab)).float()
    return x, pairs[:, 1]


def test_one_loss_per_step_and_no_val_by_default(
    vocab: Vocab, bigram: tuple[torch.Tensor, torch.Tensor]
) -> None:
    model = LinearBigram(vocab, seed=0)
    x, y = bigram
    opt = torch.optim.SGD(model.parameters(), lr=0.5)
    history = train(model, x, y, opt, steps=50, progress=False)
    assert len(history.train_losses) == 50
    assert history.val == []
    assert history.best is None


def test_training_decreases_the_loss(
    vocab: Vocab, bigram: tuple[torch.Tensor, torch.Tensor]
) -> None:
    model = LinearBigram(vocab, seed=0)
    x, y = bigram
    opt = torch.optim.SGD(model.parameters(), lr=0.5)
    history = train(model, x, y, opt, steps=200, progress=False)
    losses = history.train_losses
    assert sum(losses[-20:]) / 20 < sum(losses[:20]) / 20


def test_validation_records_every_eval_and_restores_best(
    vocab: Vocab, bigram: tuple[torch.Tensor, torch.Tensor]
) -> None:
    model = LinearBigram(vocab, seed=0)
    x, y = bigram
    x_val, y_val = x[:8], y[:8]
    opt = torch.optim.SGD(model.parameters(), lr=0.5)
    history = train(
        model,
        x,
        y,
        opt,
        steps=100,
        x_val=x_val,
        y_val=y_val,
        eval_every=10,
        patience=5,
        progress=False,
    )
    assert history.val, "validation NLL should have been recorded"
    assert {step for step, _ in history.val} <= set(range(0, 100, 10)) | {99}
    assert history.best == min(history.val, key=lambda t: t[1])
    # the best checkpoint was restored into the model before returning
    assert model.nll(x_val, y_val) == pytest.approx(history.best[1])


def test_on_eval_exception_propagates(
    vocab: Vocab, bigram: tuple[torch.Tensor, torch.Tensor]
) -> None:
    model = LinearBigram(vocab, seed=0)
    x, y = bigram
    opt = torch.optim.SGD(model.parameters(), lr=0.5)

    def prune(_step: int, _nll: float) -> None:
        raise RuntimeError("pruned")

    with pytest.raises(RuntimeError, match="pruned"):
        train(
            model,
            x,
            y,
            opt,
            steps=100,
            x_val=x,
            y_val=y,
            eval_every=1,
            on_eval=prune,
            progress=False,
        )
