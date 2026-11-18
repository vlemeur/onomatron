"""Wavenet class implementing inspired by paper
WAVENET: A GENERATIVE MODEL FOR RAW AUDIO
"""

import logging

import torch
import torch.nn.functional as F
from torch import nn

from onomatron.data import SENTINEL, Vocab

logger = logging.getLogger(__name__)


class WaveNet(nn.Module):
    """TO BE DONE"""

    def __init__(
        self,
        vocab: Vocab,
        block_size: int,
        emb_dim: int = 2,
        h_dim: int = 100,
        seed: int | None = None,
    ) -> None:
        # block size must be power of 2 (and not 1)
        if block_size < 2 or (block_size & (block_size - 1)) != 0:
            raise ValueError(
                f"block_size must be of power of 2, current is {block_size}"
            )
        if seed is not None:
            torch.manual_seed(seed)
        super().__init__()
        self.vocab, self.block_size = vocab, block_size
        self.emb_dim, self.h_dim = emb_dim, h_dim
        self.n_vocab = len(vocab)

        self.embedding = nn.Embedding(self.n_vocab, self.emb_dim)

        self.levels = nn.ModuleList()
        t, c = self.block_size, self.emb_dim
        while t > 1:
            seq: list[nn.Module] = [
                nn.Linear(c, self.h_dim),
                nn.LayerNorm(self.h_dim),
                nn.Tanh(),
                nn.Flatten(1, 2),
                nn.Unflatten(1, (t // 2, 2 * self.h_dim)),
            ]
            self.levels.append(nn.Sequential(*seq))
            t //= 2
            c = 2 * self.h_dim
        self.output = nn.Linear(c, self.n_vocab)

    @property
    def n_parameters(self) -> int:
        """Number of trainable parameters (sum of numel over all parameters)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """TO BE DONE"""
        h = self.embedding(x)
        for level in self.levels:
            h = level(h)
        return self.output(h.flatten(1))

    def loss(self, x: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
        """Cross-entropy against the next-char targets."""
        return F.cross_entropy(self(x), ys)

    @torch.no_grad()
    def nll(self, x: torch.Tensor, ys: torch.Tensor) -> float:
        """Mean negative log-likelihood per example, directly comparable to
        NGramModel.nll."""
        return float(self.loss(x, ys))

    @torch.no_grad()
    def probs(self, x: torch.Tensor) -> torch.Tensor:
        """P(next | context)"""
        return F.softmax(self(x), dim=1)

    @torch.no_grad()
    def sample(
        self, n: int = 10, min_len: int = 3, seed: int | None = None, max_len: int = 40
    ) -> list[str]:
        """Sample n names with a rolling context, padded with the sentinel;
        draw until the sentinel comes back (max_len as a runaway guard). The
        sentinel is rejected until the name reaches `min_len` chars."""
        gen = torch.Generator().manual_seed(seed) if seed is not None else None
        eos = self.vocab.stoi[SENTINEL]
        out: list[str] = []
        for _ in range(n):
            chars: list[str] = []
            context = torch.tensor([eos] * self.block_size).view((1, self.block_size))
            while len(chars) < max_len:
                row = self.probs(context)
                ix = int(
                    torch.multinomial(
                        row, num_samples=1, replacement=True, generator=gen
                    ).item()
                )
                if ix == eos:
                    if len(chars) >= min_len:
                        break
                    continue  # too short: reject eos, redraw same context
                chars.append(self.vocab.itos[ix])
                context = torch.cat([context[:, 1:], torch.tensor([[ix]])], dim=1)
            out.append("".join(chars))
        return out
