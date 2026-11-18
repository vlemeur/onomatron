"""Dilated causal WaveNet — van den Oord et al. (2016), "WaveNet: A
Generative Model for Raw Audio", adapted to character-level names.

Unlike the hierarchical WaveNet in wavenet.py (which folds the sequence in
half at each level and predicts from a single vector), every layer here keeps
the full sequence length: causal convolutions with exponentially growing
dilation (1, 2, 4, ...) give the last position a receptive field of
2**n_layers, and every position predicts the next character.
"""

import logging

import torch
import torch.nn.functional as F
from torch import nn

from onomatron.data import SENTINEL, Vocab

logger = logging.getLogger(__name__)


class CausalDilatedConv(nn.Module):
    """Causal 1D convolution: kernel_size=2 with left-only padding of
    `dilation`, so the output has the same length as the input and position
    t only sees positions <= t."""

    def __init__(self, c_in: int, c_out: int, dilation: int) -> None:
        super().__init__()
        self.dilation = dilation
        self.conv = nn.Conv1d(c_in, c_out, kernel_size=2, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(N, c_in, T) -> (N, c_out, T)."""
        return self.conv(F.pad(x, (self.dilation, 0)))


class WaveBlock(nn.Module):
    """One gated dilated block: h = tanh(filter(x)) * sigmoid(gate(x)).

    Returns (residual, skip): residual = x + W_r @ h feeds the next block,
    skip = W_s @ h is summed into the output head.
    """

    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.filter = CausalDilatedConv(channels, channels, dilation)
        self.gate = CausalDilatedConv(channels, channels, dilation)
        self.residual = nn.Conv1d(channels, channels, kernel_size=1)
        self.skip = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(N, C, T) -> (residual (N, C, T), skip (N, C, T))."""
        h = torch.tanh(self.filter(x)) * torch.sigmoid(self.gate(x))
        return x + self.residual(h), self.skip(h)


class DilatedWaveNet(nn.Module):
    """logits for every context position: (N, block_size) -> (N, block_size,
    n_vocab), one next-char distribution per position."""

    def __init__(
        self,
        vocab: Vocab,
        block_size: int,
        emb_dim: int = 64,
        n_layers: int = 4,
        seed: int | None = None,
    ) -> None:
        if seed is not None:
            torch.manual_seed(seed)
        super().__init__()
        # Receptive field of the last position with kernel 2 and dilations
        # 1..2**(n_layers-1): 2**n_layers positions. It must cover the context.
        if 2**n_layers < block_size:
            raise ValueError(
                f"n_layers={n_layers} gives a receptive field of {2**n_layers} "
                f"positions, smaller than block_size={block_size}"
            )
        self.vocab, self.block_size = vocab, block_size
        self.n_vocab = len(vocab)
        self.embedding = nn.Embedding(self.n_vocab, emb_dim)
        self.blocks = nn.ModuleList(
            WaveBlock(emb_dim, dilation=2**i) for i in range(n_layers)
        )
        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(emb_dim, emb_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(emb_dim, self.n_vocab, kernel_size=1),
        )

    @property
    def n_parameters(self) -> int:
        """Number of trainable parameters (sum of numel over all parameters)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Context indices (N, block_size) -> logits (N, block_size, n_vocab)."""
        h = self.embedding(x).transpose(1, 2)  # (N, emb_dim, T)
        skip = torch.zeros_like(h)
        for block in self.blocks:
            h, s = block(h)
            skip = skip + s
        return self.head(skip).transpose(1, 2)  # (N, T, n_vocab)

    def loss(self, x: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
        """Cross-entropy against the next-char targets.

        ys (N,): predict only the last position — same interface as MLP and
        WaveNet, NLL directly comparable across models. ys (N, T): predict
        the next char at every position (the original WaveNet training
        regime: one forward supervises T predictions).
        """
        logits = self(x)  # (N, T, V)
        if ys.dim() == 1:
            return F.cross_entropy(logits[:, -1], ys)
        return F.cross_entropy(logits.transpose(1, 2), ys)

    @torch.no_grad()
    def nll(self, x: torch.Tensor, ys: torch.Tensor) -> float:
        """Mean negative log-likelihood per example; with ys (N,) it is the
        last-position NLL, directly comparable to NGramModel/MLP/WaveNet."""
        return float(self.loss(x, ys))

    @torch.no_grad()
    def probs(self, x: torch.Tensor) -> torch.Tensor:
        """P(next | context) from the last position: (N, T) -> (N, n_vocab)."""
        return F.softmax(self(x)[:, -1], dim=1)

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
