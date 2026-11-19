"""Character-level Transformer model, inspired by Vaswany et al. 2017,
"Attention Is All You Need"."""

import logging

import torch
import torch.nn.functional as F
from torch import nn

from onomatron.data import SENTINEL, Vocab

logger = logging.getLogger(__name__)


class TransformerName(nn.Module):
    """Embedding -> nn.GRU -> Linear over the vocab. Predicts the next char at
    every position of the sequence (loss over all positions); the final hidden
    state h is chained from one char to the next in sample()."""

    def __init__(
        self,
        vocab: Vocab,
        block_size: int,  # max number of position
        n_emb: int = 64,  # embeding dimension
        n_head: int = 4,  # number of heads for multi head attention
        n_layer: int = 2,  # number of transformer encoder layers
        seed: int | None = None,
    ) -> None:

        if seed is not None:
            torch.manual_seed(seed)
        super().__init__()
        self.vocab = vocab
        self.n_emb, self.n_head = n_emb, n_head
        self.n_layer, self.block_size = n_layer, block_size
        self.n_vocab = len(vocab)
        self.eos = self.vocab.stoi[SENTINEL]

        self.tok_emb = nn.Embedding(self.n_vocab, self.n_emb)
        self.pos_emb = nn.Embedding(self.block_size, self.n_emb)
        layer = nn.TransformerEncoderLayer(
            d_model=self.n_emb,
            nhead=self.n_head,
            dim_feedforward=4 * self.n_emb,
            dropout=0.0,  # No dropout on MPS
            batch_first=True,
        )
        self.emb_drop = nn.Dropout(0.1)
        self.encoder = nn.TransformerEncoder(layer, num_layers=self.n_layer)

        self.ln_f = nn.LayerNorm(self.n_emb)
        self.head = nn.Linear(self.n_emb, self.n_vocab)

    @property
    def n_parameters(self) -> int:
        """Number of trainable parameters (sum of numel over all parameters)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """"""
        T = x.shape[1]
        pos = torch.arange(T, device=x.device)
        h = self.emb_drop(self.tok_emb(x) + self.pos_emb(pos))
        # To avoid looking into the future
        mask = torch.triu(
            torch.full((T, T), float("-inf"), device=x.device), diagonal=1
        )
        h = self.encoder(h, mask=mask)
        return self.head(self.ln_f(h))

    def loss(self, x: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
        """Cross-entropy over all positions. Padding targets are -100 (ignored
        via cross_entropy's default ignore_index); the real end-of-name
        sentinel (index 0) still counts."""
        logits = self(x)
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
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """P(next | prefix) at every position, plus the new hidden state:
        p, h_new = model.probs(tok, h)  # p: (N, T, n_vocab)."""
        logits = self(x)
        return F.softmax(logits, dim=-1)

    @torch.no_grad()
    def sample(
        self, n: int = 10, min_len: int = 3, seed: int | None = None, max_len: int = 40
    ) -> list[str]:
        """Sample n names one char at a time, chaining the hidden state h.

        Starts from the sentinel and draws until it comes back (max_len as a
        runaway guard); the sentinel is rejected until the name reaches
        `min_len` chars."""
        if max_len is None:
            max_len = self.block_size - 1

        gen = torch.Generator().manual_seed(seed) if seed is not None else None
        out: list[str] = []
        for _ in range(n):
            chars: list[str] = []
            context = torch.tensor([[self.eos]])
            while len(chars) < max_len:
                row = self.probs(context)[0, -1]
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
                context = torch.cat((context, torch.tensor([[ix]])), dim=1)
            out.append("".join(chars))
        return out
