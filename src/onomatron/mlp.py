"""Multi-layer perceptron character model — the NPLM of Bengio et al. (2003).

Same spirit as LinearBigram but with a hidden layer: an embedding table maps
each context char to a dense vector, the `block_size` embeddings are
concatenated, and a tanh hidden layer feeds a linear output over the vocab.
"""

import logging

import torch
import torch.nn.functional as F
from torch import nn

from onomatron.data import Vocab

logger = logging.getLogger(__name__)


class MLP(nn.Module):
    """logits = W2 @ tanh(W1 @ concat(C[x]) + b1) + b2."""

    def __init__(
        self,
        vocab: Vocab,
        block_size: int,
        emb_dim: int = 2,
        h_dim: int = 100,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.vocab = vocab
        self.block_size = block_size
        self.n_vocab = len(vocab)
        self.embedding = nn.Embedding(self.n_vocab, emb_dim)
        self.hidden = nn.Linear(block_size * emb_dim, h_dim)
        self.output = nn.Linear(h_dim, self.n_vocab)
        if seed is not None:
            generator = torch.Generator().manual_seed(seed)
            with torch.no_grad():
                for p in self.parameters():
                    p.copy_(torch.randn(p.shape, generator=generator))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Context indices (N, block_size) -> logits (N, n_vocab)."""
        emb = self.embedding(x)  # (N, block_size, emb_dim)
        h = torch.tanh(self.hidden(emb.flatten(1)))  # (N, h_dim)
        return self.output(h)  # (N, n_vocab)

    def loss(self, x: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
        """Cross-entropy against the next-char targets."""
        return F.cross_entropy(self(x), ys)

    @torch.no_grad()
    def nll(self, x: torch.Tensor, ys: torch.Tensor) -> float:
        """Mean negative log-likelihood per example, directly comparable to
        NGramModel.nll."""
        return float(self.loss(x, ys))
