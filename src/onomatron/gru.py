"""Character-level GRU model, inspired by Cho et al. 2014,
"Learning Phrase Representations using RNN Encoder–Decoder
for Statistical Machine Translation"."""

import logging

import torch
import torch.nn.functional as F
from torch import nn

from onomatron.data import SENTINEL, Vocab

logger = logging.getLogger(__name__)


class GRUName(nn.Module):
    """Embedding -> nn.GRU -> Linear over the vocab. Predicts the next char at
    every position of the sequence (loss over all positions); the final hidden
    state h is chained from one char to the next in sample()."""

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
        self.rnn = nn.GRU(
            input_size=self.emb_dim, hidden_size=self.h_dim, batch_first=True
        )

        self.output = nn.Linear(in_features=self.h_dim, out_features=self.n_vocab)

    @property
    def n_parameters(self) -> int:
        """Number of trainable parameters (sum of numel over all parameters)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self, x: torch.Tensor, h: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Char indices (N, T) -> (logits (N, T, n_vocab), new hidden state h).

        `h` is the hidden state to start from; None starts from zeros."""
        emb = self.embedding(x)
        out, h_new = self.rnn(emb, h)
        return self.output(out), h_new

    def loss(self, x: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
        """Cross-entropy over all positions. Padding targets are -100 (ignored
        via cross_entropy's default ignore_index); the real end-of-name
        sentinel (index 0) still counts."""
        logits, _ = self(x)
        return F.cross_entropy(
            logits.view(-1, self.n_vocab),
            ys.reshape(-1),
        )

    @torch.no_grad()
    def nll(self, x: torch.Tensor, ys: torch.Tensor) -> float:
        """Mean NLL per non-padding token, directly comparable to NGramModel.nll."""
        return float(self.loss(x, ys))

    @torch.no_grad()
    def probs(
        self, x: torch.Tensor, h: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """P(next | prefix) at every position, plus the new hidden state:
        p, h_new = model.probs(tok, h)  # p: (N, T, n_vocab)."""
        logits, h_new = self(x, h)
        return F.softmax(logits, dim=-1), h_new

    @torch.no_grad()
    def sample(
        self, n: int = 10, min_len: int = 3, seed: int | None = None, max_len: int = 40
    ) -> list[str]:
        """Sample n names one char at a time, chaining the hidden state h.

        Starts from the sentinel and draws until it comes back (max_len as a
        runaway guard); the sentinel is rejected until the name reaches
        `min_len` chars."""
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
