"""LTSM inspired by Generating Sequences With
Recurrent Neural Networks Alex Graves 2014."""

import logging

import torch
import torch.nn.functional as F
from torch import nn

from onomatron.data import SENTINEL, Vocab

logger = logging.getLogger(__name__)


class LSTMName(nn.Module):
    """TOBEDONE"""

    def __init__(
        self,
        vocab: Vocab,
        emb_dim: int = 2,
        h_dim: int = 100,
        seed: int | None = None,
    ) -> None:

        if seed is not None:
            torch.manual_seed(seed)
        super().__init__()
        self.vocab = vocab
        self.emb_dim, self.h_dim = emb_dim, h_dim
        self.n_vocab = len(vocab)
        self.eos = self.vocab.stoi[SENTINEL]

        self.embedding = nn.Embedding(self.n_vocab, self.emb_dim)
        self.lstm = nn.LSTM(
            input_size=self.emb_dim, hidden_size=self.h_dim, batch_first=True
        )

        self.output = nn.Linear(in_features=self.h_dim, out_features=self.n_vocab)

    @property
    def n_parameters(self) -> int:
        """Number of trainable parameters (sum of numel over all parameters)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self, x: torch.Tensor, hc: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Char indices (N, T) -> (logits (N, T, n_vocab), (h_n, c_n)).

        `hc` is the (h, c) state to start from; None means zeros."""
        emb = self.embedding(x)
        out, hc_new = self.lstm(emb, hc)
        return self.output(out), hc_new

    def loss(self, x: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
        """Cross-entropy over all position, ignoring the sentinel padding."""
        logits, _ = self(x)
        return F.cross_entropy(
            logits.view(-1, self.n_vocab),
            ys.reshape(-1),
        )

    @torch.no_grad()
    def nll(self, x: torch.Tensor, ys: torch.Tensor) -> float:
        """Mean negative log-likelihood per example, directly comparable to
        NGramModel.nll."""
        return float(self.loss(x, ys))

    @torch.no_grad()
    def probs(
        self, x: torch.Tensor, hc: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """P(next | prefix) at every position: (N, T, n_vocab).

        Also returns the new (h, c) state, so sample() can chain calls:
        p, hc_new = model.probs(tok, hc)."""
        logits, hc_new = self(x, hc)
        return F.softmax(logits, dim=-1), hc_new

    @torch.no_grad()
    def sample(
        self, n: int = 10, min_len: int = 3, seed: int | None = None, max_len: int = 40
    ) -> list[str]:
        """Sample n names"""
        gen = torch.Generator().manual_seed(seed) if seed is not None else None
        out: list[str] = []
        for _ in range(n):
            chars: list[str] = []
            tok = torch.tensor([[self.eos]])
            hc = None
            while len(chars) < max_len:
                row, hc = self.probs(tok, hc)
                row = row[0, -1]
                ix = int(
                    torch.multinomial(
                        row, num_samples=1, replacement=True, generator=gen
                    ).item()
                )
                if ix == self.eos:
                    if len(chars) >= min_len:
                        break
                    continue  # too short : reject eos, redraw same context

                chars.append(self.vocab.itos[ix])
                tok = torch.tensor([[ix]])
            out.append("".join(chars))
        return out
