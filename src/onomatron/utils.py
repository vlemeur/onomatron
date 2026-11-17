"""Shared training helpers for the notebooks (lr range test, training loop)."""

from collections.abc import Callable

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.optim.lr_scheduler import LRScheduler
from tqdm import tqdm


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
) -> list[float]:
    """Train `model` on (x, y); return the loss at each step.

    Full-batch if `batch_size` is None, else random minibatches drawn with a
    generator seeded by `seed`. `scheduler` must wrap `opt` (e.g. OneCycleLR
    with total_steps=steps) and is stepped after every update, so the lr
    varies during training.
    """
    gen = torch.Generator().manual_seed(seed) if seed is not None else None
    losses: list[float] = []
    for _ in tqdm(range(steps), desc="train", unit="step"):
        if batch_size is None:
            loss = model.loss(x, y)
        else:
            ix = torch.randint(0, x.shape[0], (batch_size,), generator=gen)
            loss = model.loss(x[ix], y[ix])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if scheduler is not None:
            scheduler.step()
        losses.append(loss.item())
    return losses
