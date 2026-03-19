"""High-order finite-difference solver for the 2D FitzHugh-Nagumo equation.

Spatial derivatives use a fourth-order Laplacian with periodic boundary
conditions. Time integration uses classical RK4.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.io


np.random.seed(66)


Array2D = np.ndarray

LAPLACIAN_STENCIL: Tuple[Tuple[float, Tuple[int, int]], ...] = (
    (4.0 / 3.0, (-1, 0)),
    (4.0 / 3.0, (0, -1)),
    (4.0 / 3.0, (0, 1)),
    (4.0 / 3.0, (1, 0)),
    (-1.0 / 12.0, (-2, 0)),
    (-1.0 / 12.0, (0, -2)),
    (-1.0 / 12.0, (0, 2)),
    (-1.0 / 12.0, (2, 0)),
)


@dataclass(frozen=True)
class SimulationConfig:
    """Configuration for FitzHugh-Nagumo data generation."""

    diffusion_u: float = 1.0
    diffusion_v: float = 100.0
    alpha: float = 0.01
    beta: float = 0.25
    grid_height: int = 128
    grid_width: int = 128
    dt: float = 0.0020
    dx: float = 1.0
    n_simulation_steps: int = 10000
    save_every: int = 10
    preview_frames: int = 81
    preview_stride: int = 10
    figure_dir: Path = Path("./figures/2dFN")
    data_dir: Path = Path("./data/2dFN")
    output_name: str = "FN_1001x2x128x128.mat"
    initial_condition_std: float = 0.05
    random_seed: int = 66


def apply_periodic_stencil(
        mat: Array2D,
        stencil: Sequence[Tuple[float, Tuple[int, int]]],
        scale: float,
        center_weight: float = 0.0,
    ) -> Array2D:
    """Apply a weighted stencil under periodic boundary conditions."""

    result = center_weight * mat.copy()
    for weight, shift in stencil:
        result += weight * np.roll(mat, shift=shift, axis=(0, 1))
    return result / scale


def apply_laplacian(mat: Array2D, dx: float = 1.0) -> Array2D:
    """Return the fourth-order discrete Laplacian."""

    return apply_periodic_stencil(
        mat=mat,
        stencil=LAPLACIAN_STENCIL,
        scale=dx**2,
        center_weight=-5.0,
    )


def get_temporal_diff(
        u: Array2D,
        v: Array2D,
        diffusion_u: float,
        diffusion_v: float,
        alpha: float,
        beta: float,
        dx: float,
    ) -> Tuple[Array2D, Array2D]:
    """Evaluate the FitzHugh-Nagumo PDE right-hand side."""

    laplace_u = apply_laplacian(u, dx)
    laplace_v = apply_laplacian(v, dx)

    u_t = diffusion_u * laplace_u + u - u**3 - v + alpha
    v_t = diffusion_v * laplace_v + (u - v) * beta
    return u_t, v_t


def update_rk4(
        u0: Array2D,
        v0: Array2D,
        diffusion_u: float = 1.0,
        diffusion_v: float = 100.0,
        alpha: float = 0.01,
        beta: float = 0.25,
        dt: float = 0.05,
        dx: float = 1.0,
    ) -> Tuple[Array2D, Array2D]:
    """Advance one time step using RK4."""

    k1_u, k1_v = get_temporal_diff(u0, v0, diffusion_u, diffusion_v, alpha, beta, dx)

    u1 = u0 + 0.5 * dt * k1_u
    v1 = v0 + 0.5 * dt * k1_v
    k2_u, k2_v = get_temporal_diff(u1, v1, diffusion_u, diffusion_v, alpha, beta, dx)

    u2 = u0 + 0.5 * dt * k2_u
    v2 = v0 + 0.5 * dt * k2_v
    k3_u, k3_v = get_temporal_diff(u2, v2, diffusion_u, diffusion_v, alpha, beta, dx)

    u3 = u0 + dt * k3_u
    v3 = v0 + dt * k3_v
    k4_u, k4_v = get_temporal_diff(u3, v3, diffusion_u, diffusion_v, alpha, beta, dx)

    u = u0 + (dt / 6.0) * (k1_u + 2.0 * k2_u + 2.0 * k3_u + k4_u)
    v = v0 + (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)
    return u, v


def sample_initial_conditions(config: SimulationConfig) -> Tuple[Array2D, Array2D]:
    """Sample the initial chemical concentrations."""

    shape = (config.grid_height, config.grid_width)
    u = np.random.normal(scale=config.initial_condition_std, size=shape)
    v = np.random.normal(scale=config.initial_condition_std, size=shape)
    return u, v


def build_plot_grid(resolution: int) -> Tuple[Array2D, Array2D]:
    """Build a cell-centered plotting grid."""

    coordinates = np.arange(resolution)
    return np.meshgrid(coordinates, coordinates)


def post_process(
    output: np.ndarray,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    step_index: int,
    fig_save_dir: Path,
) -> None:
    """Render the saved u/v fields for a specific frame."""

    resolution = output.shape[-1]
    x_star, y_star = build_plot_grid(resolution)
    u = output[step_index, 0, :, :]
    v = output[step_index, 1, :, :]

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(7, 7))
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for axis, field, title in zip(axes, (u, v), ("u-FDM", "v-FDM")):
        color_mesh = axis.scatter(
            x_star,
            y_star,
            c=field,
            alpha=0.95,
            edgecolors="none",
            cmap="RdYlBu",
            marker="s",
            s=3,
        )
        axis.axis("square")
        axis.set_xlim([xmin, xmax])
        axis.set_ylim([ymin, ymax])
        color_mesh.cmap.set_under("black")
        color_mesh.cmap.set_over("whitesmoke")
        axis.set_title(title)
        fig.colorbar(color_mesh, ax=axis, fraction=0.046, pad=0.04)

    fig.savefig(fig_save_dir / f"uv_[i={step_index}].png")
    plt.close(fig)


def run_simulation(config: SimulationConfig) -> np.ndarray:
    """Run the FN simulation and return a tensor shaped as (t, c, h, w)."""

    u, v = sample_initial_conditions(config)
    u_records = [u.copy()]
    v_records = [v.copy()]

    for step in range(config.n_simulation_steps):
        u, v = update_rk4(
            u,
            v,
            diffusion_u=config.diffusion_u,
            diffusion_v=config.diffusion_v,
            alpha=config.alpha,
            beta=config.beta,
            dt=config.dt,
            dx=config.dx,
        )

        if (step + 1) % config.save_every == 0:
            print(f"Completed step {step + 1}/{config.n_simulation_steps}")
            u_records.append(u.copy())
            v_records.append(v.copy())

    uv = np.stack((np.stack(u_records, axis=0), np.stack(v_records, axis=0)), axis=0)
    return np.transpose(uv, (1, 0, 2, 3))


def create_preview_images(output: np.ndarray, config: SimulationConfig) -> None:
    """Generate preview figures for a subset of saved frames."""

    config.figure_dir.mkdir(parents=True, exist_ok=True)
    for frame_index in range(config.preview_frames):
        saved_step = frame_index * config.preview_stride
        if saved_step >= output.shape[0]:
            break
        post_process(
            output=output,
            xmin=0,
            xmax=config.grid_width,
            ymin=0,
            ymax=config.grid_height,
            step_index=saved_step,
            fig_save_dir=config.figure_dir,
        )


def save_dataset(output: np.ndarray, config: SimulationConfig) -> Path:
    """Persist the generated trajectory as a MAT file."""

    config.data_dir.mkdir(parents=True, exist_ok=True)
    output_path = config.data_dir / config.output_name
    scipy.io.savemat(output_path, {"uv": output})
    return output_path


def main() -> None:
    """Generate FN simulation data and preview figures."""

    config = SimulationConfig()
    output = run_simulation(config)
    create_preview_images(output, config)
    save_dataset(output, config)


if __name__ == "__main__":
    main()
