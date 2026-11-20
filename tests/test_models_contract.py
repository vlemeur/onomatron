"""Contract tests shared by every name model.

The models/ package promises one interface — loss(x, y), nll(x, y),
probs(x), sample(...), n_parameters — whatever the architecture. These
tests pin that contract for each model on a tiny fixed dataset:

- probs are a proper distribution over the vocab;
- nll equals the loss AND the manual -log P(target) from probs;
- sample is deterministic under a seed and respects its constraints;
- the seed makes the init reproducible;
- a short SGD run actually decreases the loss (learning smoke test).

Two input shapes coexist: context windows x (N, block) -> y (N,) for
mlp/wavenet/dilated_wavenet (and one-hot bigrams for linear), full
sequences x, y (N, T) for the recurrent models. NGramModel is count-based:
no loss(x, y), its probs is a property and its nll takes names.
"""

from collections.abc import Iterator
from typing import Protocol, Self

import pytest
import torch
from torch.nn import Parameter

from onomatron.data import SENTINEL, Vocab
from onomatron.models.dilated_wavenet import DilatedWaveNet
from onomatron.models.gru import GRUName
from onomatron.models.linear import LinearBigram
from onomatron.models.lstm import LSTMName
from onomatron.models.mlp import MLP
from onomatron.models.ngram import NGramModel
from onomatron.models.rnn import RNNName
from onomatron.models.wavenet import WaveNet

SEED = 0
CONTEXT_MODELS = ["mlp", "wavenet", "dilated_wavenet"]
SEQ_MODELS = ["rnn", "lstm", "gru"]


class Trainable(Protocol):
    """The minimal surface the training loop needs."""

    def loss(self, x: torch.Tensor, ys: torch.Tensor, /) -> torch.Tensor: ...
    def parameters(self) -> Iterator[Parameter]: ...


class ContextModel(Trainable, Protocol):
    """The interface of the context-window models: x (N, block) -> y (N,)."""

    @property
    def n_parameters(self) -> int: ...

    def nll(self, x: torch.Tensor, ys: torch.Tensor, /) -> float: ...
    def probs(self, x: torch.Tensor, /) -> torch.Tensor: ...
    def sample(
        self, n: int = 10, min_len: int = 3, seed: int | None = None, max_len: int = 40
    ) -> list[str]: ...
    def state_dict(self) -> dict[str, torch.Tensor]: ...
    def eval(self) -> Self: ...


class SeqModel(Trainable, Protocol):
    """The interface of the recurrent models: x, y (N, T); probs also
    returns the new hidden state."""

    @property
    def n_parameters(self) -> int: ...

    def nll(self, x: torch.Tensor, ys: torch.Tensor, /) -> float: ...
    def probs(self, x: torch.Tensor, /) -> tuple[torch.Tensor, object]: ...
    def sample(
        self, n: int = 10, min_len: int = 3, seed: int | None = None, max_len: int = 40
    ) -> list[str]: ...
    def state_dict(self) -> dict[str, torch.Tensor]: ...


def make_context_model(name: str, vocab: Vocab) -> ContextModel:
    if name == "mlp":
        return MLP(vocab, block_size=4, emb_dim=4, h_dim=16, seed=SEED)
    if name == "wavenet":
        return WaveNet(vocab, block_size=4, emb_dim=4, h_dim=16, seed=SEED)
    if name == "dilated_wavenet":
        # 2**n_layers = 4 must cover block_size=4
        return DilatedWaveNet(vocab, block_size=4, emb_dim=4, n_layers=2, seed=SEED)
    raise ValueError(name)


def make_seq_model(name: str, vocab: Vocab) -> SeqModel:
    cls = {"rnn": RNNName, "lstm": LSTMName, "gru": GRUName}[name]
    return cls(vocab, emb_dim=4, h_dim=16, seed=SEED)


def assert_sample_constraints(
    samples: list[str], vocab: Vocab, n: int, min_len: int, max_len: int
) -> None:
    assert len(samples) == n
    allowed = set(vocab.stoi) - {SENTINEL}
    for s in samples:
        assert min_len <= len(s) <= max_len
        assert set(s) <= allowed


def assert_sgd_learns(
    model: Trainable,
    x: torch.Tensor,
    y: torch.Tensor,
    lr: float = 0.05,
    steps: int = 300,
) -> None:
    """Full-batch SGD must decrease the loss: mean of the last 20 steps
    below the mean of the first 20 (robust to a noisy single last step)."""
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    losses = []
    for _ in range(steps):
        loss = model.loss(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    head = sum(losses[:20]) / 20
    tail = sum(losses[-20:]) / 20
    assert tail < head, f"loss did not decrease: {head:.4f} -> {tail:.4f}"


@pytest.mark.parametrize("name", CONTEXT_MODELS)
class TestContextModels:
    """mlp / wavenet / dilated_wavenet: x (N, block) -> y (N,)."""

    def test_probs_are_a_distribution(
        self, name: str, vocab: Vocab, ctx_xy: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        model = make_context_model(name, vocab)
        x, _ = ctx_xy
        p = model.probs(x)
        assert p.shape == (x.shape[0], len(vocab))
        assert torch.allclose(p.sum(dim=1), torch.ones(x.shape[0]), atol=1e-5)

    def test_nll_equals_loss_and_manual_nll(
        self, name: str, vocab: Vocab, ctx_xy: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        model = make_context_model(name, vocab)
        model.eval()  # BatchNorm: train-mode loss uses batch stats, eval doesn't
        x, y = ctx_xy
        nll = model.nll(x, y)
        assert nll == pytest.approx(model.loss(x, y).item())
        p = model.probs(x)
        manual = -torch.log(p[torch.arange(len(y)), y]).mean()
        assert nll == pytest.approx(manual.item(), rel=1e-5)

    def test_n_parameters_counts_trainable_params(
        self, name: str, vocab: Vocab
    ) -> None:
        model = make_context_model(name, vocab)
        expected = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert model.n_parameters == expected > 0

    def test_seed_makes_init_reproducible(self, name: str, vocab: Vocab) -> None:
        a = make_context_model(name, vocab)
        b = make_context_model(name, vocab)
        for pa, pb in zip(
            a.state_dict().values(), b.state_dict().values(), strict=False
        ):
            assert torch.equal(pa, pb)

    def test_sample_is_seeded_and_constrained(self, name: str, vocab: Vocab) -> None:
        model = make_context_model(name, vocab)
        s1 = model.sample(n=5, min_len=3, seed=42)
        s2 = model.sample(n=5, min_len=3, seed=42)
        assert s1 == s2
        assert_sample_constraints(s1, vocab, n=5, min_len=3, max_len=40)

    def test_sgd_decreases_loss(
        self, name: str, vocab: Vocab, ctx_xy: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        model = make_context_model(name, vocab)
        x, y = ctx_xy
        assert_sgd_learns(model, x, y)


@pytest.mark.parametrize("name", SEQ_MODELS)
class TestSeqModels:
    """rnn / lstm / gru: x, y (N, T); probs returns (probs, new state)."""

    def test_probs_are_a_distribution_per_position(
        self, name: str, vocab: Vocab, seq_xy: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        model = make_seq_model(name, vocab)
        x, _ = seq_xy
        p, _h = model.probs(x)
        assert p.shape == (x.shape[0], x.shape[1], len(vocab))
        sums = p.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_nll_equals_loss_and_ignores_padding(
        self, name: str, vocab: Vocab, seq_xy: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        model = make_seq_model(name, vocab)
        x, y = seq_xy
        nll = model.nll(x, y)
        assert nll == pytest.approx(model.loss(x, y).item())
        p, _h = model.probs(x)
        mask = y != -100
        picked = p[mask].gather(1, y[mask].unsqueeze(1)).squeeze(1)
        assert nll == pytest.approx((-torch.log(picked)).mean().item(), rel=1e-5)

    def test_n_parameters_counts_trainable_params(
        self, name: str, vocab: Vocab
    ) -> None:
        model = make_seq_model(name, vocab)
        expected = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert model.n_parameters == expected > 0

    def test_seed_makes_init_reproducible(self, name: str, vocab: Vocab) -> None:
        a = make_seq_model(name, vocab)
        b = make_seq_model(name, vocab)
        for pa, pb in zip(
            a.state_dict().values(), b.state_dict().values(), strict=False
        ):
            assert torch.equal(pa, pb)

    def test_sample_is_seeded_and_constrained(self, name: str, vocab: Vocab) -> None:
        model = make_seq_model(name, vocab)
        s1 = model.sample(n=5, min_len=3, seed=42)
        s2 = model.sample(n=5, min_len=3, seed=42)
        assert s1 == s2
        assert_sample_constraints(s1, vocab, n=5, min_len=3, max_len=40)

    def test_sgd_decreases_loss(
        self, name: str, vocab: Vocab, seq_xy: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        model = make_seq_model(name, vocab)
        x, y = seq_xy
        assert_sgd_learns(model, x, y, lr=0.02)


class TestLinearBigram:
    """LinearBigram takes one-hot bigrams: x (N, V) -> y (N,)."""

    def test_probs_are_a_distribution(
        self, vocab: Vocab, onehot_xy: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        model = LinearBigram(vocab, seed=SEED)
        x, _ = onehot_xy
        p = model.probs(x)
        assert p.shape == (x.shape[0], len(vocab))
        assert torch.allclose(p.sum(dim=1), torch.ones(x.shape[0]), atol=1e-5)

    def test_nll_equals_loss_and_manual_nll(
        self, vocab: Vocab, onehot_xy: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        model = LinearBigram(vocab, seed=SEED)
        x, y = onehot_xy
        nll = model.nll(x, y)
        assert nll == pytest.approx(model.loss(x, y).item())
        p = model.probs(x)
        manual = -torch.log(p[torch.arange(len(y)), y]).mean()
        assert nll == pytest.approx(manual.item(), rel=1e-5)

    def test_n_parameters_is_the_weight_matrix(self, vocab: Vocab) -> None:
        model = LinearBigram(vocab, seed=SEED)
        assert model.n_parameters == len(vocab) ** 2

    def test_sample_is_seeded_and_constrained(self, vocab: Vocab) -> None:
        model = LinearBigram(vocab, seed=SEED)
        s1 = model.sample(n=5, min_len=3, seed=42)
        s2 = model.sample(n=5, min_len=3, seed=42)
        assert s1 == s2
        assert_sample_constraints(s1, vocab, n=5, min_len=3, max_len=40)

    def test_sgd_decreases_loss(
        self, vocab: Vocab, onehot_xy: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        model = LinearBigram(vocab, seed=SEED)
        x, y = onehot_xy
        assert_sgd_learns(model, x, y, lr=0.5, steps=200)


class TestNGramModel:
    """Count-based model: probs is a property, nll takes raw names."""

    def test_probs_are_a_distribution(self, names: list[str], vocab: Vocab) -> None:
        model = NGramModel.fit(names, vocab, n=3, smoothing=1.0)
        p = model.probs
        sums = p.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_zero_smoothing_gives_certain_transitions(self) -> None:
        vocab = Vocab(["a"])
        model = NGramModel.fit(["a"], vocab, n=2, smoothing=0.0)
        a, dot = vocab.stoi["a"], vocab.stoi[SENTINEL]
        assert model.probs[dot, a].item() == pytest.approx(1.0)
        assert model.probs[a, dot].item() == pytest.approx(1.0)
        assert model.probs[a, a].item() == pytest.approx(0.0)
        # a perfectly predicted name has zero NLL
        assert model.nll(["a"]) == pytest.approx(0.0, abs=1e-6)

    def test_nll_is_positive_with_smoothing(
        self, names: list[str], vocab: Vocab
    ) -> None:
        model = NGramModel.fit(names, vocab, n=2, smoothing=1.0)
        assert model.nll(names) > 0

    def test_n_below_two_raises(self, vocab: Vocab) -> None:
        with pytest.raises(ValueError, match="n must be"):
            NGramModel(vocab, n=1)

    def test_sample_is_seeded_and_constrained(
        self, names: list[str], vocab: Vocab
    ) -> None:
        model = NGramModel.fit(names, vocab, n=2)
        s1 = model.sample(n=5, min_len=3, seed=42)
        s2 = model.sample(n=5, min_len=3, seed=42)
        assert s1 == s2
        assert_sample_constraints(s1, vocab, n=5, min_len=3, max_len=40)

    def test_learning_by_counting_beats_uniform(
        self, names: list[str], vocab: Vocab
    ) -> None:
        # the fitted bigram must assign higher NLL to a name made of the
        # vocab's rarest transitions than to a training name on average:
        # simply check the training NLL is below the uniform baseline
        model = NGramModel.fit(names, vocab, n=2, smoothing=1.0)
        uniform = torch.log(torch.tensor(float(len(vocab))))
        assert model.nll(names) < uniform.item()
