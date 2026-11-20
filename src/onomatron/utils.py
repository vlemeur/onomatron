"""Shared training helpers for the notebooks (lr range test, training loop)."""

import copy
import logging
from collections.abc import Callable
from typing import NamedTuple

import plotly.express as px
import plotly.graph_objects as go
import torch
from torch import nn
from torch.optim.lr_scheduler import LRScheduler
from tqdm import tqdm

logger = logging.getLogger(__name__)


class TrainHistory(NamedTuple):
    """What `train` returns: the per-step training loss, plus (step, val NLL)
    pairs when validation tensors were provided (empty otherwise), plus
    `best` = (step, val NLL) of the best checkpoint (None without val)."""

    train_losses: list[float]
    val: list[tuple[int, float]]
    best: tuple[int, float] | None = None


def lr_range_test(
    model_fn: Callable[[int], nn.Module],
    x: torch.Tensor,
    y: torch.Tensor,
    lrs: list[float],
    steps: int = 100,
    batch_size: int | None = None,
    seed: int = 42,
) -> list[float]:
    """Final loss after `steps` SGD steps at each lr.

    Full-batch if `batch_size` is None, else random minibatches (drawn with a
    generator seeded by `seed`) — prefer a minibatch here: it matches the real
    training regime, and a full-batch test tends to overestimate the usable lr.
    `model_fn(seed)` must return a fresh model exposing `.loss(x, y)` and
    `.parameters()` (e.g. LinearBigram, MLP, WaveNet), so every lr starts from
    the same initialization.
    """
    gen = torch.Generator().manual_seed(seed) if seed is not None else None
    losses: list[float] = []
    for lr in tqdm(lrs, desc="lr range test", unit="lr"):
        model = model_fn(seed)
        opt = torch.optim.SGD(model.parameters(), lr=lr)
        for _ in range(steps):
            if batch_size is None:
                loss = model.loss(x, y)
            else:
                ix = torch.randint(0, x.shape[0], (batch_size,), generator=gen)
                loss = model.loss(x[ix], y[ix])
            opt.zero_grad()
            loss.backward()
            opt.step()
        losses.append(loss.item())
    return losses


def plot_lr_range(lrs: list[float], losses: list[float], steps: int = 100) -> float:
    """Plot loss vs learning rate (log scale); return the lr at the steepest
    descent of the curve (Smith 2015), NOT divided by ~10 — the caller does.

    More robust than the raw argmin when the curve gets noisy at high lr: past
    a certain lr the loss after `steps` depends on minibatch luck, so the
    global min can be a fluke that diverges in real training. The steepest
    slope is measured between the means of two adjacent blocks of ~10% of the
    curve (noise cancels in each mean, the log-lr denominator is ~constant).
    The winning segment is drawn on the figure and the selected lr is marked
    with a black dashed vertical line.
    """
    lrs_t = torch.tensor(lrs, dtype=torch.float64)
    loss_t = torch.tensor(losses, dtype=torch.float64)
    n = loss_t.numel()
    k = max(2, n // 10)
    log_lrs = torch.log10(lrs_t)
    best_i, best_slope = 0, float("inf")
    for i in range(0, n - 2 * k + 1, max(1, k // 2)):
        s = (loss_t[i + k : i + 2 * k].mean() - loss_t[i : i + k].mean()) / (
            log_lrs[i + k : i + 2 * k].mean() - log_lrs[i : i + k].mean()
        )
        if float(s) < best_slope:
            best_slope, best_i = float(s), i
    best_lr = float(lrs_t[best_i + k // 2])

    fig = go.Figure(go.Scatter(x=lrs, y=losses, mode="lines", name="loss"))
    # steepest segment: straight line between the means of the two blocks
    fig.add_scatter(
        x=[
            float(lrs_t[best_i : best_i + k].mean()),
            float(lrs_t[best_i + k : best_i + 2 * k].mean()),
        ],
        y=[
            float(loss_t[best_i : best_i + k].mean()),
            float(loss_t[best_i + k : best_i + 2 * k].mean()),
        ],
        mode="lines+markers",
        name="steepest segment",
        line_color=px.colors.qualitative.Set2[1],
    )
    # selected lr: black dashed vertical line + marker on the curve
    fig.add_vline(x=best_lr, line=dict(color="black", dash="dash"))
    idx = int(torch.argmin((log_lrs - torch.log10(torch.tensor(best_lr))).abs()))
    fig.add_scatter(
        x=[lrs[idx]],
        y=[losses[idx]],
        mode="markers",
        name=f"selected lr = {best_lr:.3g}",
        marker=dict(color="black", size=12, symbol="x"),
    )
    fig.update_layout(
        xaxis_type="log",
        xaxis_title="learning rate",
        yaxis_title=f"loss after {steps} steps",
    )
    fig.show()
    return best_lr


def train(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    opt: torch.optim.Optimizer,
    steps: int,
    batch_size: int | None = None,
    scheduler: LRScheduler | None = None,
    seed: int | None = None,
    x_val: torch.Tensor | None = None,
    y_val: torch.Tensor | None = None,
    eval_every: int | None = None,
    clip_grad: float | None = None,
    patience: int | None = None,
    on_eval: Callable[[int, float], None] | None = None,
) -> TrainHistory:
    """Train `model` on (x, y); return a TrainHistory.

    Full-batch if `batch_size` is None, else random minibatches drawn with a
    generator seeded by `seed`. `scheduler` must wrap `opt` (e.g. OneCycleLR
    with total_steps=steps) and is stepped after every update, so the lr
    varies during training. If `x_val`/`y_val` are given, `model.nll` on the
    validation set is recorded every `eval_every` steps (default: ~100 points
    spread over the run) and at the last step; `model` must expose
    `.nll(x, y)` (LinearBigram and MLP both do) and is responsible for
    switching itself to eval mode there (MLP does, for its BatchNorm).
    If `clip_grad` is set, the global gradient norm is clipped to that value
    after each backward — a guard against loss spikes from noisy minibatches
    at high lr (`torch.nn.utils.clip_grad_norm_`).
    With `patience` set and validation tensors given, training stops after
    `patience` consecutive val evaluations without improvement, and the best
    weights (lowest val NLL) are restored into `model` before returning:
    early stopping + best-checkpoint restore. `history.best` then holds
    (step, val NLL) of that checkpoint. `on_eval`, when given, is called with
    (step, val NLL) after every validation evaluation — a hook for external
    consumers such as an Optuna pruner, which may raise (e.g. TrialPruned) to
    abort the run early; the exception propagates out of `train`.
    """
    gen = torch.Generator().manual_seed(seed) if seed is not None else None
    model.train()  # minibatch statistics for any BatchNorm (no-op otherwise)
    if x_val is not None and y_val is not None and eval_every is None:
        eval_every = max(1, steps // 100)
    train_losses: list[float] = []
    val: list[tuple[int, float]] = []
    best: tuple[int, float] | None = None
    best_state: dict[str, torch.Tensor] | None = None
    bad_evals = 0
    for step in tqdm(range(steps), desc="train", unit="step"):
        if batch_size is None:
            loss = model.loss(x, y)
        else:
            # indices are drawn on CPU (generator lives there) but must sit on
            # the same device as x for the indexing to work (MPS/CUDA)
            ix = torch.randint(0, x.shape[0], (batch_size,), generator=gen).to(x.device)
            loss = model.loss(x[ix], y[ix])
        opt.zero_grad()
        loss.backward()
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        opt.step()
        if scheduler is not None:
            scheduler.step()
        train_losses.append(loss.item())
        if (
            eval_every
            and x_val is not None
            and y_val is not None
            and (step % eval_every == 0 or step == steps - 1)
        ):
            val_nll = model.nll(x_val, y_val)
            val.append((step, val_nll))
            if on_eval is not None:
                on_eval(step, val_nll)  # may raise (e.g. optuna.TrialPruned)
            if best is None or val_nll < best[1]:
                best = (step, val_nll)
                best_state = copy.deepcopy(model.state_dict())
                bad_evals = 0
            else:
                bad_evals += 1
                if patience is not None and bad_evals >= patience:
                    logger.info(
                        "early stopping at step %d (best val NLL %.4f at step %d)",
                        step,
                        best[1],
                        best[0],
                    )
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    return TrainHistory(train_losses, val, best)


def plot_history(history: TrainHistory, smooth: int | None = None) -> None:
    """Plot the training loss curve (blue), with the validation NLL overlaid
    as orange markers when the model was trained with validation tensors.

    With `smooth` > 1, the per-step losses are replaced by a rolling mean
    over `smooth` steps — the raw minibatch curve is noisy.
    """
    fig = go.Figure()
    losses = history.train_losses
    if smooth is not None and 1 < smooth < len(losses):
        window = (
            torch.tensor(losses, dtype=torch.float32).unfold(0, smooth, 1).mean(dim=1)
        )
        centers = torch.arange(len(window)) + (smooth - 1) / 2
        fig.add_scatter(
            x=centers.tolist(),
            y=window.tolist(),
            mode="lines",
            name=f"train (rolling mean, w={smooth})",
        )
    else:
        fig.add_scatter(
            x=list(range(len(losses))),
            y=losses,
            mode="lines",
            name="train",
        )
    if history.val:
        steps, nlls = zip(*history.val, strict=True)
        fig.add_scatter(
            x=list(steps),
            y=list(nlls),
            mode="markers",
            name="val",
        )
    fig.update_layout(xaxis_title="step", yaxis_title="loss / NLL")
    fig.show()
