"""RNN inspired by Mikolov et al. 2010"""

import logging

import torch
import torch.nn.functional as F
from torch import nn

from onomatron.data import SENTINEL, Vocab

logger = logging.getLogger(__name__)


class RNNName(nn.Module):
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
        self.rnn = nn.RNN(
            input_size=self.emb_dim, hidden_size=self.h_dim, batch_first=True
        )

        self.output = nn.Linear(in_features=self.h_dim, out_features=self.n_vocab)

    @property
    def n_parameters(self) -> int:
        """Number of trainable parameters (sum of numel over all parameters)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: torch.Tensor, h: torch.Tensor | None = None) -> torch.Tensor:
        """"""
        emb = self.embedding(x)
        # RNN output and hidden state
        out, h_rnn = self.rnn(emb, h)
        return self.output(out), h_rnn

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
    def probs(self, x: torch.Tensor, h: torch.Tensor | None = None) -> torch.Tensor:
        """P(next | prefix) at every position: (N, T, n_vocab).

        Also returns the final hidden state, so sample() can chain calls:
        p, h_new = model.probs(tok, h)."""
        logits, h_rnn = self(x, h)
        return F.softmax(logits, dim=-1), h_rnn

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
            h = None
            while len(chars) < max_len:
                row, h = self.probs(tok, h)
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
