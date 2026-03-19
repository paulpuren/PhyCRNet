"""Shared utilities for training, evaluation, and experiment I/O."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch


def set_random_seed(seed: int) -> None:
    """Set NumPy and PyTorch seeds for reproducibility."""

    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_directory(path: Path) -> None:
    """Create a directory tree if it does not already exist."""

    path.mkdir(parents=True, exist_ok=True)


def save_checkpoint(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        save_path: Path,
    ) -> None:
    """Persist model and optimizer state."""

    ensure_directory(save_path.parent)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        },
        save_path,
    )


def load_checkpoint(
        model: torch.nn.Module,
        save_path: Path,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        map_location: Optional[torch.device] = None,
    ) -> Tuple[torch.nn.Module, Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler._LRScheduler]]:
    """Load model and optional optimizer state from a checkpoint."""

    checkpoint = torch.load(save_path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return model, optimizer, scheduler


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters."""

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def frobenius_norm(array: np.ndarray) -> float:
    """Compute the Frobenius norm of a NumPy array."""

    return float(np.sqrt(np.sum(array**2)))


def plot_training_loss(train_loss: Iterable[float], save_path: Path) -> None:
    """Save the training-loss curve."""

    ensure_directory(save_path.parent)
    plt.figure()
    plt.plot(list(train_loss), label="train loss")
    plt.yscale("log")
    plt.legend()
    plt.savefig(save_path, dpi=300)
    plt.close("all")


def plot_time_trace(
        predicted: np.ndarray,
        truth: np.ndarray,
        t_pred: np.ndarray,
        t_true: np.ndarray,
        save_path: Path,
        x_index: int = 32,
        y_index: int = 32,
    ) -> None:
    """Plot a time trace for a fixed grid location."""

    ensure_directory(save_path.parent)
    plt.figure()
    plt.plot(t_pred, predicted[:, y_index, x_index], label=f"x={x_index}, y={y_index}, CRL")
    plt.plot(t_true, truth[:, y_index, x_index], "--", label=f"x={x_index}, y={y_index}, Ref.")
    plt.xlabel("t")
    plt.ylabel("u")
    plt.xlim(0, 2)
    plt.legend()
    plt.savefig(save_path)
    plt.close("all")


def post_process_comparison(
    output: torch.Tensor,
    truth: np.ndarray,
    axis_limits: Tuple[float, float, float, float],
    uv_limits: Tuple[float, float, float, float],
    step_index: int,
    save_path: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Render prediction/reference comparisons for a single time step."""

    xmin, xmax, ymin, ymax = axis_limits
    u_min, u_max, v_min, v_max = uv_limits

    resolution = truth.shape[-1] - 2
    x = np.linspace(xmin, xmax, resolution + 1)[:-1]
    x_star, y_star = np.meshgrid(x, x)

    u_true = truth[step_index, 0, 1:-1, 1:-1]
    u_pred = output[step_index, 0, 1:-1, 1:-1].detach().cpu().numpy()
    v_true = truth[step_index, 1, 1:-1, 1:-1]
    v_pred = output[step_index, 1, 1:-1, 1:-1].detach().cpu().numpy()

    ensure_directory(save_path.parent)
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(7, 7))
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    plots = (
        (axes[0, 0], u_pred, "u-RCNN", u_min, u_max),
        (axes[0, 1], u_true, "u-Ref.", u_min, u_max),
        (axes[1, 0], v_pred, "v-RCNN", v_min, v_max),
        (axes[1, 1], v_true, "v-Ref.", v_min, v_max),
    )

    for axis, values, title, vmin, vmax in plots:
        color_mesh = axis.scatter(
            x_star,
            y_star,
            c=values,
            alpha=0.9,
            edgecolors="none",
            cmap="RdYlBu",
            marker="s",
            s=4,
            vmin=vmin,
            vmax=vmax,
        )
        axis.axis("square")
        axis.set_xlim([xmin, xmax])
        axis.set_ylim([ymin, ymax])
        axis.set_title(title)
        fig.colorbar(color_mesh, ax=axis)

    fig.savefig(save_path)
    plt.close("all")
    return u_true, u_pred, v_true, v_pred
