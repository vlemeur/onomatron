"""Shared training helpers for the notebooks (lr range test, training loop)."""

from collections.abc import Callable
from typing import NamedTuple

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.optim.lr_scheduler import LRScheduler
from tqdm import tqdm


class TrainHistory(NamedTuple):
    """What `train` returns: the per-step training loss, plus (step, val NLL)
    pairs when validation tensors were provided (empty otherwise)."""

    train_losses: list[float]
    val: list[tuple[int, float]]


def lr_range_test(
    model_fn: Callable[[int], nn.Module],
    x: torch.Tensor,
    y: torch.Tensor,
    lrs: list[float],
    steps: int = 100,
    seed: int = 42,
) -> list[float]:
    """Final loss after `steps` full-batch SGD steps at each lr.

    `model_fn(seed)` must return a fresh model exposing `.loss(x, y)` and
    `.parameters()` (e.g. LinearBigram, or an MLP class with the same
    interface), so every lr starts from the same initialization.
    """
    losses: list[float] = []
    for lr in tqdm(lrs, desc="lr range test", unit="lr"):
        model = model_fn(seed)
        opt = torch.optim.SGD(model.parameters(), lr=lr)
        for _ in range(steps):
            loss = model.loss(x, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        losses.append(loss.item())
    return losses


def plot_lr_range(lrs: list[float], losses: list[float], steps: int = 100) -> float:
    """Plot loss vs learning rate on a log scale; return the lr at the minimum."""
    plt.plot(lrs, losses)
    plt.xscale("log")
    plt.xlabel("learning rate")
    plt.ylabel(f"loss after {steps} steps")
    return lrs[int(torch.tensor(losses).argmin())]


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
    """
    gen = torch.Generator().manual_seed(seed) if seed is not None else None
    model.train()  # minibatch statistics for any BatchNorm (no-op otherwise)
    if x_val is not None and y_val is not None and eval_every is None:
        eval_every = max(1, steps // 100)
    train_losses: list[float] = []
    val: list[tuple[int, float]] = []
    for step in tqdm(range(steps), desc="train", unit="step"):
        if batch_size is None:
            loss = model.loss(x, y)
        else:
            ix = torch.randint(0, x.shape[0], (batch_size,), generator=gen)
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
            val.append((step, model.nll(x_val, y_val)))
    return TrainHistory(train_losses, val)


def plot_history(history: TrainHistory, smooth: int | None = None) -> None:
    """Plot the training loss curve (blue), with the validation NLL overlaid
    as orange markers when the model was trained with validation tensors.

    With `smooth` > 1, the per-step losses are replaced by a rolling mean
    over `smooth` steps — the raw minibatch curve is noisy.
    """
    losses = history.train_losses
    if smooth is not None and 1 < smooth < len(losses):
        window = (
            torch.tensor(losses, dtype=torch.float32).unfold(0, smooth, 1).mean(dim=1)
        )
        centers = torch.arange(len(window)) + (smooth - 1) / 2
        plt.plot(
            centers.tolist(),
            window.tolist(),
            color="tab:blue",
            label=f"train (rolling mean, w={smooth})",
        )
    else:
        plt.plot(losses, color="tab:blue", label="train")
    if history.val:
        steps, nlls = zip(*history.val, strict=True)
        plt.plot(steps, nlls, marker="o", color="tab:orange", label="val")
    plt.xlabel("step")
    plt.ylabel("loss / NLL")
    plt.legend()
