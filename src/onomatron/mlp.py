"""Multi-layer perceptron character model — the NPLM of Bengio et al. (2003).

Same spirit as LinearBigram but with a hidden layer: an embedding table maps
each context char to a dense vector, the `block_size` embeddings are
concatenated, and an optional batch-norm + tanh hidden layer feeds a linear
output over the vocab.

Initialization relies on PyTorch's defaults (Kaiming-uniform for nn.Linear,
N(0,1) for nn.Embedding): `seed` only fixes the global RNG so those defaults
are reproducible. Overriding every parameter with raw randn (std=1) would
saturate tanh at step 0 and pin the loss on a high plateau.
"""

import logging
from collections.abc import Generator
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from torch import nn

from onomatron.data import SENTINEL, Vocab

logger = logging.getLogger(__name__)


class MLP(nn.Module):
    """logits = W2 @ tanh(BN(W1 @ concat(C[x]) + b1)) + b2 (BN optional)."""

    def __init__(
        self,
        vocab: Vocab,
        block_size: int,
        emb_dim: int = 2,
        h_dim: int = 100,
        batch_norm: bool = True,
        seed: int | None = None,
    ) -> None:
        if seed is not None:
            # Seed the global RNG BEFORE building the layers, so their own
            # default init (Kaiming-uniform for Linear, N(0,1) for Embedding)
            # is reproducible. Do not overwrite parameters with raw randn.
            torch.manual_seed(seed)
        super().__init__()
        self.vocab = vocab
        self.block_size = block_size
        self.n_vocab = len(vocab)
        self.batch_norm = batch_norm
        self.embedding = nn.Embedding(self.n_vocab, emb_dim)
        self.hidden = nn.Linear(block_size * emb_dim, h_dim)
        self.bn = nn.BatchNorm1d(h_dim) if batch_norm else None
        self.output = nn.Linear(h_dim, self.n_vocab)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Context indices (N, block_size) -> logits (N, n_vocab)."""
        emb = self.embedding(x)  # (N, block_size, emb_dim)
        pre = self.hidden(emb.flatten(1))  # (N, h_dim)
        if self.bn is not None:
            pre = self.bn(pre)
        h = torch.tanh(pre)  # (N, h_dim)
        return self.output(h)  # (N, n_vocab)

    @contextmanager
    def _eval_mode(self) -> Generator[None]:
        """Temporarily switch to eval mode: with BatchNorm, train mode uses
        minibatch statistics and rejects batch size 1, which sample() makes."""
        was_training = self.training
        self.eval()
        try:
            yield
        finally:
            self.train(was_training)

    def loss(self, x: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
        """Cross-entropy against the next-char targets."""
        return F.cross_entropy(self(x), ys)

    @torch.no_grad()
    def nll(self, x: torch.Tensor, ys: torch.Tensor) -> float:
        """Mean negative log-likelihood per example, directly comparable to
        NGramModel.nll."""
        with self._eval_mode():
            return float(self.loss(x, ys))

    @torch.no_grad()
    def probs(self, x: torch.Tensor) -> torch.Tensor:
        """P(next | context)"""
        with self._eval_mode():
            return F.softmax(self(x), dim=1)

    @torch.no_grad()
    def sample(
        self, n: int = 10, seed: int | None = None, max_len: int = 40
    ) -> list[str]:
        """Sample n names with a rolling context of n-1 chars, padded with
        the sentinel at the start; draw until the sentinel comes back
        (max_len as a runaway guard)."""
        gen = torch.Generator().manual_seed(seed) if seed is not None else None

        eos = self.vocab.stoi[SENTINEL]

        out: list[str] = []

        with self._eval_mode():  # sample one name at a time (batch size 1)
            for _ in range(n):
                chars: list[str] = []
                context = torch.tensor([eos] * self.block_size).view(
                    (1, self.block_size)
                )
                while len(chars) < max_len:
                    row = self.probs(context)
                    ix = int(
                        torch.multinomial(
                            row,
                            num_samples=1,
                            replacement=True,
                            generator=gen,
                        ).item()
                    )
                    if ix == eos:
                        break
                    chars.append(self.vocab.itos[ix])
                    context = torch.cat([context[:, 1:], torch.tensor([[ix]])], dim=1)
                out.append("".join(chars))
        return out
