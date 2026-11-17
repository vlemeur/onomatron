"""Single-layer linear character model — the "neural" bigram of makemore part 1.

Unlike NGramModel, which stores smoothed counts, this model learns a weight
matrix by gradient descent: logits = one_hot(context) @ W, trained with
cross-entropy on the next character.
"""

import logging

import torch
import torch.nn.functional as F
from torch import nn

from onomatron.data import SENTINEL, Vocab

logger = logging.getLogger(__name__)


class LinearBigram(nn.Module):
    """One linear layer over one-hot contexts: logits = xenc @ W (no bias)."""

    def __init__(self, vocab: Vocab, seed: int | None = None) -> None:
        super().__init__()
        self.vocab = vocab
        self.n_vocab = len(vocab)
        self.linear = nn.Linear(self.n_vocab, self.n_vocab, bias=False)
        if seed is not None:
            generator = torch.Generator().manual_seed(seed)
            with torch.no_grad():
                self.linear.weight.copy_(
                    torch.randn((self.n_vocab, self.n_vocab), generator=generator)
                )

    def forward(self, xenc: torch.Tensor) -> torch.Tensor:
        """One-hot context rows -> logits over the vocab."""
        return self.linear(xenc)

    def loss(self, xenc: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
        """Cross-entropy (log_softmax + NLL) against the next-char targets."""
        return F.cross_entropy(self(xenc), ys)

    @torch.no_grad()
    def nll(self, xenc: torch.Tensor, ys: torch.Tensor) -> float:
        """Mean negative log-likelihood per bigram, directly comparable to
        NGramModel.nll."""
        return float(self.loss(xenc, ys))

    @torch.no_grad()
    def probs(self, xenc: torch.Tensor) -> torch.Tensor:
        """P(next | context)"""
        return F.softmax(self(xenc), dim=1)

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
        for _ in range(n):
            chars: list[str] = []
            ix = eos
            while len(chars) < max_len:
                x = F.one_hot(torch.tensor([ix]), num_classes=self.n_vocab).float()
                row = self.probs(x)[0]
                ix = int(
                    torch.multinomial(
                        row, num_samples=1, replacement=True, generator=gen
                    ).item()
                )
                if ix == eos:
                    break
                chars.append(self.vocab.itos[ix])
            out.append("".join(chars))
        return out
