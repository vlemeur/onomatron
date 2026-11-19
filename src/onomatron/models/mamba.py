"""Character-level Mamba model, inspired by Gu & Dao 2023, "Mamba: Linear-Time
Sequence Modeling with Selective State Spaces" (simplified S6 block).

Selective SSM: the state update h_t = Abar_t * h_{t-1} + Bbar_t * x_t uses
discretized A, B, dt that DEPEND on the input x_t (the "selection" mechanism) —
unlike RWKV's input-independent WKV weights. A causal depthwise conv gives local
context before the scan.

The scan has two equivalent forms: a vectorized one (log-space cumsum over T,
used for training: h_t = e^{L_t} * cumsum(e^{-L_j} v_j) with L = cumsum(dt*A))
and a recurrent one (state carried step by step, used for sampling — O(1) per
generated char, like an RNN).
"""

import logging
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import nn

from onomatron.data import SENTINEL, Vocab

logger = logging.getLogger(__name__)


class MambaState(NamedTuple):
    """Recurrent state of one Mamba block: the SSM state h (N, C, S) and the
    depthwise-conv buffer (N, C, d_conv-1) for causality at sampling time."""

    h: torch.Tensor
    conv_buf: torch.Tensor


# Model state = one MambaState per block; None = use the vectorized training path.
MambaModelState = list[MambaState]


class MambaBlock(nn.Module):
    """One Mamba block: in_proj -> causal depthwise conv + SiLU -> selective
    SSM scan (A, B, dt from x) -> gate by z -> out_proj, residual + LayerNorm."""

    def __init__(self, n_emb: int, d_state: int = 16, d_conv: int = 4) -> None:
        super().__init__()
        self.n_emb = n_emb
        self.d_state = d_state
        self.d_conv = d_conv
        self.in_proj = nn.Linear(n_emb, 2 * n_emb)  # -> (u, z)
        # depthwise causal conv, padding done manually (left) so seq/step match
        self.conv = nn.Conv1d(n_emb, n_emb, kernel_size=d_conv, groups=n_emb)
        self.x_proj = nn.Linear(n_emb, n_emb + 2 * d_state)  # -> (dt_raw, B, C)
        self.dt_bias = nn.Parameter(torch.zeros(n_emb))
        self.A_log = nn.Parameter(torch.zeros(d_state))  # A = -exp(A_log) < 0
        self.out_proj = nn.Linear(n_emb, n_emb)
        self.ln = nn.LayerNorm(n_emb)

    def initial_state(self, n: int, device: torch.device) -> MambaState:
        h = torch.zeros(n, self.n_emb, self.d_state, device=device)
        buf = torch.zeros(n, self.n_emb, self.d_conv - 1, device=device)
        return MambaState(h, buf)

    def _selective_params(self, u: torch.Tensor):
        """From input frames u (..., C): dt (..., C), logAbar (..., C, S),
        Bbar (..., C, S), C (..., S). A, B, dt all depend on u: the selection."""
        dt_raw, B, C = self.x_proj(u).split(
            [self.n_emb, self.d_state, self.d_state], dim=-1
        )
        dt = F.softplus(dt_raw + self.dt_bias)  # (..., C)
        A = -torch.exp(self.A_log)  # (S,)
        logAbar = dt.unsqueeze(-1) * A  # (..., C, S)
        Bbar = dt.unsqueeze(-1) * B.unsqueeze(-2)  # (..., C, S)
        return logAbar, Bbar, C

    def _conv(self, u: torch.Tensor) -> torch.Tensor:
        """Causal depthwise conv + SiLU. u: (..., T, C) -> (..., T, C)."""
        leading = u.shape[:-1]
        x = u.transpose(-1, -2)  # (..., C, T)
        x = F.pad(x, (self.d_conv - 1, 0))  # left pad = causal
        x = self.conv(x)  # (..., C, T)
        return F.silu(x.transpose(-1, -2)).reshape(*leading, -1)

    def forward_seq(self, x: torch.Tensor) -> torch.Tensor:
        """Vectorized over time (training path). x: (N, T, C) -> (N, T, C)."""
        u, z = self.in_proj(x).chunk(2, dim=-1)
        u = self._conv(u)  # (N, T, C)
        logAbar, Bbar, C = self._selective_params(u)  # (N,T,C,S), (N,T,C,S), (N,T,S)
        v = Bbar * u.unsqueeze(-1)  # (N, T, C, S)
        L = torch.cumsum(logAbar, dim=1)  # (N, T, C, S); logAbar < 0 so L decreases
        m = L.max(dim=1, keepdim=True).values  # stability; cancels in the ratio
        h = torch.exp(L - m) * torch.cumsum(torch.exp(m - L) * v, dim=1)
        y = (h * C.unsqueeze(-2)).sum(-1)  # (N, T, C)
        return self.ln(x + self.out_proj(y) * F.silu(z))

    def step(
        self, x_t: torch.Tensor, state: MambaState
    ) -> tuple[torch.Tensor, MambaState]:
        """One time step (sampling path). x_t: (N, C) -> (out (N, C), new state)."""
        u, z = self.in_proj(x_t).chunk(2, dim=-1)
        conv_in = torch.cat([state.conv_buf, u.unsqueeze(-1)], dim=-1)  # (N, C, d_conv)
        u = F.silu(self.conv(conv_in).squeeze(-1))  # (N, C)
        new_buf = conv_in[..., 1:]
        logAbar, Bbar, C = self._selective_params(u)  # (N,C,S), (N,C,S), (N,S)
        h = torch.exp(logAbar) * state.h + Bbar * u.unsqueeze(-1)  # (N, C, S)
        y = (h * C.unsqueeze(1)).sum(-1)  # (N, C)
        out = self.ln(x_t + self.out_proj(y) * F.silu(z))
        return out, MambaState(h, new_buf)


class MambaName(nn.Module):
    """Embedding -> Mamba blocks -> LayerNorm -> Linear over the vocab. Predicts
    the next char at every position; sample() chains the recurrent state from
    one char to the next, like an RNN."""

    def __init__(
        self,
        vocab: Vocab,
        n_emb: int = 64,
        n_layer: int = 2,
        d_state: int = 16,
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
        self.blocks = nn.ModuleList(MambaBlock(n_emb, d_state) for _ in range(n_layer))
        self.ln_f = nn.LayerNorm(n_emb)
        self.head = nn.Linear(n_emb, self.n_vocab)

    @property
    def n_parameters(self) -> int:
        """Number of trainable parameters (sum of numel over all parameters)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self, x: torch.Tensor, state: MambaModelState | None = None
    ) -> tuple[torch.Tensor, MambaModelState | None]:
        """Char indices (N, T) -> (logits (N, T, n_vocab), state).

        state=None: vectorized training path (no state returned). state given:
        recurrent path, one step at a time, returning the new state (sampling)."""
        h = self.tok_emb(x)  # (N, T, C)
        if state is None:
            for block in self.blocks:
                h = block.forward_seq(h)
            return self.head(self.ln_f(h)), None
        new_state: MambaModelState = []
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
        self, x: torch.Tensor, state: MambaModelState | None = None
    ) -> tuple[torch.Tensor, MambaModelState | None]:
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
