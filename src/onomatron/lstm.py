"""Character-level LSTM language model, inspired by Graves et al. 2014,
"Generating Sequences With Recurrent Neural Networks"."""

import logging

import torch
import torch.nn.functional as F
from torch import nn

from onomatron.data import SENTINEL, Vocab

logger = logging.getLogger(__name__)

# (h, c) pair chained between steps during sampling; None = start from zeros.
HCState = tuple[torch.Tensor, torch.Tensor]


class LSTMName(nn.Module):
    """Embedding -> nn.LSTM -> Linear over the vocab. Predicts the next char at
    every position of the sequence (loss over all positions); the final (h, c)
    state is chained from one char to the next in sample()."""

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
        self, x: torch.Tensor, hc: HCState | None = None
    ) -> tuple[torch.Tensor, HCState]:
        """Char indices (N, T) -> (logits (N, T, n_vocab), new (h, c) state).

        `hc` is the (h, c) state to start from; None starts from zeros."""
        emb = self.embedding(x)
        out, hc_new = self.lstm(emb, hc)
        return self.output(out), hc_new

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
        self, x: torch.Tensor, hc: HCState | None = None
    ) -> tuple[torch.Tensor, HCState]:
        """P(next | prefix) at every position, plus the new (h, c) state:
        p, hc_new = model.probs(tok, hc)  # p: (N, T, n_vocab)."""
        logits, hc_new = self(x, hc)
        return F.softmax(logits, dim=-1), hc_new

    @torch.no_grad()
    def sample(
        self, n: int = 10, min_len: int = 3, seed: int | None = None, max_len: int = 40
    ) -> list[str]:
        """Sample n names one char at a time, chaining the (h, c) state.

        Starts from the sentinel and draws until it comes back (max_len as a
        runaway guard); the sentinel is rejected until the name reaches
        `min_len` chars."""
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
