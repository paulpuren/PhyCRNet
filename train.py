"""Training and evaluation entrypoint for the Burgers PhyCRNet model."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import scipy.io as scio
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from Models.PhyCRNet_burgers import (
    PhyCRNet,
    PhyCRNetConfig,
    PhysicsLossGenerator,
    build_model,
    compute_physics_loss,
)
from utils import (
    count_parameters,
    ensure_directory,
    frobenius_norm,
    load_checkpoint,
    plot_time_trace,
    plot_training_loss,
    post_process_comparison,
    save_checkpoint,
    set_random_seed,
)


torch.set_default_dtype(torch.float32)


@dataclass(frozen=True)
class TrainingConfig:
    """Training and evaluation configuration for Burgers experiments."""

    data_path: Path = Path("./data/burgers_1501x2x128x128.mat")
    figure_dir: Path = Path("./figures")
    model_dir: Path = Path("./model")
    pretrained_checkpoint: Path = Path("./model/checkpoint500.pt")
    checkpoint_path: Path = Path("./model/checkpoint1000.pt")
    train_loss_path: Path = Path("./model/train_loss.npy")
    train_loss_plot_path: Path = Path("./figures/train_loss.png")
    trace_plot_path: Path = Path("./figures/x=32,y=32.png")
    learning_rate: float = 1e-4
    scheduler_step_size: int = 100
    scheduler_gamma: float = 0.97
    num_iterations: int = 2000
    time_batch_size: int = 1000
    time_steps: int = 1001
    dt: float = 0.002
    dx: float = 1.0 / 128.0
    reynolds_number: float = 200.0
    random_seed: int = 66
    grid_size: int = 128
    axis_limits: Tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0)
    uv_limits: Tuple[float, float, float, float] = (-0.7, 0.7, -1.0, 1.0)
    evaluation_frames: int = 50
    evaluation_stride: int = 20

    @property
    def num_time_batch(self) -> int:
        return int(self.time_steps / self.time_batch_size)

    @property
    def steps(self) -> int:
        return self.time_batch_size + 1


def get_device() -> torch.device:
    """Select a default execution device."""

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_dataset(
        data_path: Path, 
        device: torch.device
    ) -> Tuple[np.ndarray, torch.Tensor]:
    """Load the Burgers trajectory and return NumPy and tensor views."""

    data = scio.loadmat(data_path)
    uv = data["uv"]
    initial_condition = torch.tensor(
        uv[0:1, ...], 
        dtype=torch.float32, 
        device=device
    )
    return uv, initial_condition


def create_initial_states(
        config: TrainingConfig,
        model_config: PhyCRNetConfig,
        device: torch.device,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Create initial ConvLSTM hidden states based on encoder downsampling."""

    spatial_size = config.grid_size
    for stride in model_config.input_stride[: model_config.num_layers[0]]:
        spatial_size //= stride

    hidden_channels = model_config.hidden_channels[model_config.num_layers[0]]
    state_shape = (1, hidden_channels, spatial_size, spatial_size)
    states = []
    for _ in range(model_config.num_layers[1]):
        hidden = torch.randn(*state_shape, device=device)
        cell = torch.randn(*state_shape, device=device)
        states.append((hidden, cell))
    return states


def detach_state(
        state: Sequence[Tuple[torch.Tensor, torch.Tensor]]
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Detach recurrent states from the current graph."""

    return [(hidden.detach(), cell.detach()) for hidden, cell in state]


def maybe_restore_checkpoint(
        model: PhyCRNet,
        optimizer: optim.Optimizer,
        scheduler: StepLR,
        checkpoint_path: Path,
        device: torch.device,
    ) -> None:
    """Restore a checkpoint if it exists."""

    if checkpoint_path.exists():
        load_checkpoint(
            model, 
            checkpoint_path, 
            optimizer=optimizer, 
            scheduler=scheduler, 
            map_location=device
        )
        print(f"Loaded checkpoint: {checkpoint_path}")
    else:
        print(f"Checkpoint not found, starting fresh: {checkpoint_path}")


def train_model(
        model: PhyCRNet,
        inputs: torch.Tensor,
        initial_state: Sequence[Tuple[torch.Tensor, torch.Tensor]],
        config: TrainingConfig,
        device: torch.device,
    ) -> List[float]:
    """Run physics-informed training."""

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = StepLR(
        optimizer, 
        step_size=config.scheduler_step_size, 
        gamma=config.scheduler_gamma
    )
    maybe_restore_checkpoint(
        model, 
        optimizer, 
        scheduler, 
        config.pretrained_checkpoint, 
        device
    )

    for param_group in optimizer.param_groups:
        print(f"Learning rate: {param_group['lr']}")

    loss_generator = PhysicsLossGenerator(
        config.dt, 
        config.dx, 
        config.reynolds_number
    ).to(device)
    train_loss_history: List[float] = []
    best_loss = float("inf")

    for epoch in range(config.num_iterations):
        optimizer.zero_grad()
        batch_loss = 0.0
        previous_output = None
        detached_state = list(initial_state)

        for time_batch_id in range(config.num_time_batch):
            if time_batch_id == 0:
                hidden_state = initial_state
                u0 = inputs
            else:
                hidden_state = detached_state
                assert previous_output is not None
                u0 = previous_output[-2:-1].detach()

            output, second_last_state = model(hidden_state, u0)
            output = torch.cat(tuple(output), dim=0)
            output = torch.cat((u0, output), dim=0)

            loss = compute_physics_loss(output, loss_generator)
            loss.backward(retain_graph=True)
            batch_loss += loss.item()

            previous_output = output
            detached_state = detach_state(second_last_state)

        optimizer.step()
        scheduler.step()

        progress = 100.0 * (epoch + 1) / config.num_iterations
        print(
            f"[{epoch + 1}/{config.num_iterations} {progress:.0f}%] loss: {batch_loss:.10f}"
        )
        train_loss_history.append(batch_loss)

        if batch_loss < best_loss:
            save_checkpoint(model, optimizer, scheduler, config.checkpoint_path)
            best_loss = batch_loss

    return train_loss_history


def pad_periodic_boundaries(tensor: torch.Tensor) -> torch.Tensor:
    """Pad the spatial dimensions for periodic-boundary visualization."""

    tensor = torch.cat((tensor[:, :, :, -1:], tensor, tensor[:, :, :, 0:2]), dim=3)
    tensor = torch.cat((tensor[:, :, -1:, :], tensor, tensor[:, :, 0:2, :]), dim=2)
    return tensor


def pad_truth_boundaries(array: np.ndarray) -> np.ndarray:
    """Pad NumPy truth data for periodic-boundary visualization."""

    array = np.concatenate((array[:, :, :, -1:], array, array[:, :, :, 0:2]), axis=3)
    array = np.concatenate((array[:, :, -1:, :], array, array[:, :, 0:2, :]), axis=2)
    return array


def evaluate_model(
        model: PhyCRNet,
        inputs: torch.Tensor,
        initial_state: Sequence[Tuple[torch.Tensor, torch.Tensor]],
        uv: np.ndarray,
        config: TrainingConfig,
        device: torch.device,
    ) -> float:
    """Run inference, generate figures, and report relative error."""

    load_checkpoint(model, config.checkpoint_path, map_location=device)
    model.eval()

    with torch.no_grad():
        output, _ = model(initial_state, inputs)
        output = torch.cat(tuple(output), dim=0)
        output = torch.cat((inputs, output), dim=0)
        output = pad_periodic_boundaries(output)

    truth = pad_truth_boundaries(uv[0 : config.time_steps, :, :, :])

    ten_true = []
    ten_pred = []
    for frame_index in range(config.evaluation_frames):
        step_index = config.evaluation_stride * frame_index
        u_true, u_pred, v_true, v_pred = post_process_comparison(
            output,
            truth,
            config.axis_limits,
            config.uv_limits,
            step_index=step_index,
            save_path=config.figure_dir / f"uv_comparison_{str(step_index).zfill(3)}.png",
        )
        ten_true.append([u_true, v_true])
        ten_pred.append([u_pred, v_pred])

    relative_error = frobenius_norm(np.array(ten_pred) - np.array(ten_true)) / frobenius_norm(np.array(ten_true))
    print(f"The predicted error is: {relative_error}")

    u_pred = output[:-1, 0, :, :].detach().cpu().numpy()
    u_pred = np.swapaxes(u_pred, 1, 2)
    u_true = truth[:, 0, :, :]
    t_true = np.linspace(0, 2, config.time_steps)
    t_pred = np.linspace(0, 2, config.time_steps)
    plot_time_trace(u_pred, u_true, t_pred, t_true, config.trace_plot_path)

    return float(relative_error)


def build_model_config(config: TrainingConfig) -> PhyCRNetConfig:
    """Create the architecture config used for both training and evaluation."""

    return PhyCRNetConfig(
        dt=config.dt,
        step=config.steps,
        effective_step=tuple(range(config.steps)),
    )


def run_training(config: TrainingConfig, device: torch.device) -> List[float]:
    """Load data, build the model, and execute training."""

    uv, inputs = load_dataset(config.data_path, device)
    model_config = build_model_config(config)
    model = build_model(model_config).to(device)
    initial_state = create_initial_states(config, model_config, device)

    print(f"Trainable parameters: {count_parameters(model)}")
    start_time = time.time()
    train_loss = train_model(model, inputs, initial_state, config, device)
    elapsed = time.time() - start_time

    ensure_directory(config.model_dir)
    np.save(config.train_loss_path, np.array(train_loss))
    plot_training_loss(train_loss, config.train_loss_plot_path)
    print(f"The training time is: {elapsed}")
    return train_loss


def run_evaluation(config: TrainingConfig, device: torch.device) -> float:
    """Load data and run evaluation from the saved checkpoint."""

    uv, inputs = load_dataset(config.data_path, device)
    model_config = build_model_config(config)
    model = build_model(model_config).to(device)
    initial_state = create_initial_states(config, model_config, device)
    return evaluate_model(model, inputs, initial_state, uv, config, device)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Train or evaluate PhyCRNet on the Burgers dataset."
    )
    parser.add_argument(
        "--mode",
        choices=("train", "eval", "all"),
        default="all",
        help="Whether to run training, evaluation, or both.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point."""

    args = parse_args()
    config = TrainingConfig()
    device = get_device()
    set_random_seed(config.random_seed)

    ensure_directory(config.figure_dir)
    ensure_directory(config.model_dir)

    if args.mode in ("train", "all"):
        run_training(config, device)
    if args.mode in ("eval", "all"):
        run_evaluation(config, device)


if __name__ == "__main__":
    main()
