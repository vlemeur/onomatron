"""Character-level RWKV model, inspired by Peng et al. 2023, "RWKV:
Reinventing RNNs for the Transformer Era" (v4 "Eagle" blocks, simplified).

Each block alternates:
- time-mixing: a WKV "linear attention" over time — a running weighted average
  of past values with weights exp(k_j - k_t);
- channel-mixing: a per-position MLP (ReLU^2), like the transformer FFN.

The time-mix has two equivalent forms: a vectorized one (cumsum/cummax over T,
used for training) and a recurrent one (state carried step by step, used for
sampling — O(1) per generated char, like an RNN). No positional embedding:
the recurrence itself carries time.
"""

import logging
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import nn

from onomatron.data import SENTINEL, Vocab

logger = logging.getLogger(__name__)


class LayerState(NamedTuple):
    """Recurrent state of one RWKV block, all (N, C): the shifted input x_prev,
    and the WKV accumulators (a, b, m) so that a/b = weighted average of past
    values. m is the running max of k for numerical stability."""

    x_prev: torch.Tensor
    a: torch.Tensor
    b: torch.Tensor
    m: torch.Tensor


# Model state = one LayerState per block; None = use the vectorized training path.
RWKVState = list[LayerState]


def _shift(x: torch.Tensor) -> torch.Tensor:
    """Shift along time by one: x_prev[:, 0] = 0, x_prev[:, t] = x[:, t-1]."""
    return torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)


class RWKVBlock(nn.Module):
    """One RWKV block: time-mix (WKV over time) then channel-mix (per-position
    MLP), each with residual + LayerNorm."""

    def __init__(self, n_emb: int) -> None:
        super().__init__()
        self.n_emb = n_emb
        self.ln1 = nn.LayerNorm(n_emb)
        self.ln2 = nn.LayerNorm(n_emb)
        # time-mix
        self.time_r = nn.Linear(n_emb, n_emb)
        self.time_rx = nn.Linear(n_emb, n_emb, bias=False)
        self.time_k = nn.Linear(n_emb, n_emb, bias=False)
        self.time_v = nn.Linear(n_emb, n_emb, bias=False)
        # channel-mix
        self.chan_r = nn.Linear(n_emb, n_emb)
        self.chan_rx = nn.Linear(n_emb, n_emb, bias=False)
        self.chan_k = nn.Linear(n_emb, 4 * n_emb)
        self.chan_v = nn.Linear(4 * n_emb, n_emb, bias=False)

    def initial_state(self, n: int, device: torch.device) -> LayerState:
        z = torch.zeros(n, self.n_emb, device=device)
        return LayerState(z, z.clone(), z.clone(), torch.full_like(z, float("-inf")))

    def _time_mix(self, x: torch.Tensor, x_prev: torch.Tensor, a, b, m):
        """Shared by both paths. x, x_prev: (..., C); a, b, m: (..., C) or None
        for the vectorized path. Returns (h, a, b, m)."""
        r = torch.sigmoid(self.time_r(x) + self.time_rx(x_prev))
        x_mix = x + (r - 1.0) * x_prev
        k = self.time_k(x_mix)
        v = self.time_v(x_mix)
        if a is None:  # vectorized: global max over time + cumsum (m cancels in a/b)
            m = k.max(dim=1, keepdim=True).values
            e = torch.exp(k - m)
            a = torch.cumsum(e * v, dim=1)
            b = torch.cumsum(e, dim=1)
        else:  # recurrent: running max, rescale old accumulators to the new max
            m_new = torch.maximum(m, k)
            rescale = torch.exp(m - m_new)
            e = torch.exp(k - m_new)
            a = a * rescale + e * v
            b = b * rescale + e
            m = m_new
        wkv = a / (b + 1e-8)
        return self.ln1(x + r * wkv), a, b, m

    def forward_seq(self, x: torch.Tensor) -> torch.Tensor:
        """Vectorized over time (training path). x: (N, T, C) -> (N, T, C)."""
        h, _, _, _ = self._time_mix(x, _shift(x), None, None, None)
        r2 = torch.sigmoid(self.chan_r(h) + self.chan_rx(_shift(x)))
        k2 = F.relu(self.chan_k(h)) ** 2
        return self.ln2(h + r2 * self.chan_v(k2))

    def step(
        self, x_t: torch.Tensor, state: LayerState
    ) -> tuple[torch.Tensor, LayerState]:
        """One time step (sampling path). x_t: (N, C) -> (out (N, C), new state)."""
        x_prev, a, b, m = state
        h, a, b, m = self._time_mix(x_t, x_prev, a, b, m)
        r2 = torch.sigmoid(self.chan_r(h) + self.chan_rx(x_prev))
        k2 = F.relu(self.chan_k(h)) ** 2
        out = self.ln2(h + r2 * self.chan_v(k2))
        return out, LayerState(x_t, a, b, m)


class RWKVName(nn.Module):
    """Embedding -> RWKV blocks -> LayerNorm -> Linear over the vocab. Predicts
    the next char at every position; sample() chains the recurrent state from
    one char to the next, like an RNN."""

    def __init__(
        self,
        vocab: Vocab,
        n_emb: int = 64,
        n_layer: int = 2,
        seed: int | None = None,
    ) -> None:
        if seed is not None:
            torch.manual_seed(seed)
        super().__init__()
        self.vocab = vocab
        self.n_emb, self.n_layer = n_emb, n_layer
        self.n_vocab = len(vocab)
        self.eos = vocab.stoi[SENTINEL]

        self.tok_emb = nn.Embedding(self.n_vocab, n_emb)
        self.blocks = nn.ModuleList(RWKVBlock(n_emb) for _ in range(n_layer))
        self.ln_f = nn.LayerNorm(n_emb)
        self.head = nn.Linear(n_emb, self.n_vocab)

    @property
    def n_parameters(self) -> int:
        """Number of trainable parameters (sum of numel over all parameters)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self, x: torch.Tensor, state: RWKVState | None = None
    ) -> tuple[torch.Tensor, RWKVState | None]:
        """Char indices (N, T) -> (logits (N, T, n_vocab), state).

        state=None: vectorized training path (no state returned). state given:
        recurrent path, one step at a time, returning the new state (sampling)."""
        h = self.tok_emb(x)  # (N, T, C)
        if state is None:
            for block in self.blocks:
                h = block.forward_seq(h)
            return self.head(self.ln_f(h)), None
        new_state: RWKVState = []
        for block, st in zip(self.blocks, state, strict=True):
            outs = []
            for i in range(x.shape[1]):
                o, st = block.step(h[:, i], st)
                outs.append(o)
            h = torch.stack(outs, dim=1)
            new_state.append(st)
        return self.head(self.ln_f(h)), new_state

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
        self, x: torch.Tensor, state: RWKVState | None = None
    ) -> tuple[torch.Tensor, RWKVState | None]:
        """P(next | prefix) at every position, plus the new state when a state
        was given: p, state_new = model.probs(tok, state)  # p: (N, T, n_vocab)."""
        logits, state_new = self(x, state)
        return F.softmax(logits, dim=-1), state_new

    @torch.no_grad()
    def sample(
        self, n: int = 10, min_len: int = 3, seed: int | None = None, max_len: int = 40
    ) -> list[str]:
        """Sample n names one char at a time, chaining the recurrent state.

        Starts from the sentinel and draws until it comes back (max_len as a
        runaway guard); the sentinel is rejected until the name reaches
        `min_len` chars."""
        gen = torch.Generator().manual_seed(seed) if seed is not None else None
        out: list[str] = []
        for _ in range(n):
            chars: list[str] = []
            tok = torch.tensor([[self.eos]])
            state = [b.initial_state(1, tok.device) for b in self.blocks]
            while len(chars) < max_len:
                row, state = self.probs(tok, state)
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
