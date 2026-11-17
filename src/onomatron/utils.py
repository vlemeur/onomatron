"""Shared training helpers for the notebooks (learning-rate range test)."""

from collections.abc import Callable

import matplotlib.pyplot as plt
import torch
from torch import nn


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
    for lr in lrs:
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
