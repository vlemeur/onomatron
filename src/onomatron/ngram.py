"""Count-based n-gram language model over characters (n=2 = plain bigram).

No training: the model IS the smoothed count tensor P(next | context of n-1
chars). torch is used for tensors and multinomial sampling only — there are
no parameters and no backward pass.
"""

import logging

import torch

from onomatron.data import SENTINEL, Vocab, make_ngram_pairs

logger = logging.getLogger(__name__)


class NGramModel:
    """N-gram model over a Vocab: raw counts -> smoothed probabilities."""

    def __init__(self, vocab: Vocab, n: int = 2, smoothing: float = 1.0) -> None:
        if n < 2:
            raise ValueError("n must be >= 2")
        self.vocab = vocab
        self.n = n
        self.smoothing = smoothing
        self.counts = torch.zeros((len(vocab),) * n, dtype=torch.int64)

    @classmethod
    def fit(
        cls, names: list[str], vocab: Vocab, n: int = 2, smoothing: float = 1.0
    ) -> NGramModel:
        """Count all sentinel-padded n-grams of names into the tensor."""
        model = cls(vocab, n, smoothing)
        pairs = make_ngram_pairs(names, vocab, n)
        ctx, tgt = pairs[:, :-1], pairs[:, -1]
        model.counts.index_put_(
            (*ctx.unbind(dim=1), tgt),
            torch.ones(len(pairs), dtype=torch.int64),
            accumulate=True,
        )
        logger.info(
            "fitted %d-gram model: %d names, %d tokens, vocab %d",
            n,
            len(names),
            len(pairs),
            len(vocab),
        )
        return model

    @property
    def probs(self) -> torch.Tensor:
        """Smoothed transition tensor, normalized over the last dim:
        P(next | context)."""
        c = self.counts.float() + self.smoothing
        return c / c.sum(dim=-1, keepdim=True)

    def nll(self, names: list[str]) -> float:
        """Mean negative log-likelihood per n-gram over names."""
        pairs = make_ngram_pairs(names, self.vocab, self.n)
        p = self.probs[(*pairs[:, :-1].unbind(dim=1), pairs[:, -1])]
        return float(-torch.log(p).mean())

    @torch.no_grad()
    def sample(
        self, n: int = 10, min_len: int = 3, seed: int | None = None, max_len: int = 40
    ) -> list[str]:
        """Sample n names with a rolling context of n-1 chars, padded with
        the sentinel at the start; draw until the sentinel comes back
        (max_len as a runaway guard). The sentinel is rejected until the
        name reaches `min_len` chars."""
        gen = torch.Generator().manual_seed(seed) if seed is not None else None
        probs = self.probs
        eos = self.vocab.stoi[SENTINEL]
        out: list[str] = []
        for _ in range(n):
            context = [eos] * (self.n - 1)
            chars: list[str] = []
            while len(chars) < max_len:
                row = probs[tuple(context)]
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
                context = context[1:] + [ix]
            out.append("".join(chars))
        return out
