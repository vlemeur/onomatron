"""Count-based bigram language model over characters.

No training: the model IS the smoothed count matrix P(next | current).
torch is used for tensors and multinomial sampling only — there are no
parameters and no backward pass.
"""

import logging

import torch

from onomatron.data import SENTINEL, Vocab, make_bigram_pairs

logger = logging.getLogger(__name__)


class BigramModel:
    """Bigram model over a Vocab: raw counts -> smoothed probabilities."""

    def __init__(self, vocab: Vocab, smoothing: float = 1.0) -> None:
        self.vocab = vocab
        self.smoothing = smoothing
        self.counts = torch.zeros((len(vocab), len(vocab)), dtype=torch.int64)

    @classmethod
    def fit(cls, names: list[str], vocab: Vocab, smoothing: float = 1.0) -> BigramModel:
        """Count all sentinel-padded bigrams of names into the matrix."""
        model = cls(vocab, smoothing)
        pairs = make_bigram_pairs(names, vocab)
        model.counts.index_put_(
            (pairs[:, 0], pairs[:, 1]),
            torch.ones(len(pairs), dtype=torch.int64),
            accumulate=True,
        )
        logger.info(
            "fitted bigram model: %d names, %d bigram tokens, vocab %d",
            len(names),
            len(pairs),
            len(vocab),
        )
        return model

    @property
    def probs(self) -> torch.Tensor:
        """Smoothed, row-normalized transition matrix P(next | current)."""
        c = self.counts.float() + self.smoothing
        return c / c.sum(dim=1, keepdim=True)

    def nll(self, names: list[str]) -> float:
        """Mean negative log-likelihood per bigram over names."""
        pairs = make_bigram_pairs(names, self.vocab)
        p = self.probs[pairs[:, 0], pairs[:, 1]]
        return float(-torch.log(p).mean())

    @torch.no_grad()
    def sample(
        self, n: int = 10, seed: int | None = None, max_len: int = 40
    ) -> list[str]:
        """Sample n names: start at the sentinel, draw next characters until
        the sentinel comes back (max_len as a runaway guard)."""
        gen = torch.Generator().manual_seed(seed) if seed is not None else None
        probs = self.probs
        eos = self.vocab.stoi[SENTINEL]
        out: list[str] = []
        for _ in range(n):
            chars: list[str] = []
            ix = eos
            while len(chars) < max_len:
                ix = int(
                    torch.multinomial(
                        probs[ix], num_samples=1, replacement=True, generator=gen
                    ).item()
                )
                if ix == eos:
                    break
                chars.append(self.vocab.itos[ix])
            out.append("".join(chars))
        return out
