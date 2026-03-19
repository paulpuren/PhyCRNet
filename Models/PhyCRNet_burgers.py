"""PhyCRNet model components for the 2D Burgers equation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


TensorPair = Tuple[torch.Tensor, torch.Tensor]

LAPLACIAN_KERNEL = [[[[0, 0, -1 / 12, 0, 0],
                      [0, 0, 4 / 3, 0, 0],
                      [-1 / 12, 4 / 3, -5, 4 / 3, -1 / 12],
                      [0, 0, 4 / 3, 0, 0],
                      [0, 0, -1 / 12, 0, 0]]]]

PARTIAL_Y_KERNEL = [[[[0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0],
                      [1 / 12, -8 / 12, 0, 8 / 12, -1 / 12],
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0]]]]

PARTIAL_X_KERNEL = [[[[0, 0, 1 / 12, 0, 0],
                      [0, 0, -8 / 12, 0, 0],
                      [0, 0, 0, 0, 0],
                      [0, 0, 8 / 12, 0, 0],
                      [0, 0, -1 / 12, 0, 0]]]]


@dataclass(frozen=True)
class PhyCRNetConfig:
    """Configuration for the Burgers PhyCRNet architecture."""

    input_channels: int = 2
    hidden_channels: Tuple[int, ...] = (8, 32, 128, 128)
    input_kernel_size: Tuple[int, ...] = (4, 4, 4, 3)
    input_stride: Tuple[int, ...] = (2, 2, 2, 1)
    input_padding: Tuple[int, ...] = (1, 1, 1, 1)
    dt: float = 0.002
    num_layers: Tuple[int, int] = (3, 1)
    upscale_factor: int = 8
    step: int = 1001
    effective_step: Optional[Tuple[int, ...]] = None

    def resolved_effective_step(self) -> List[int]:
        if self.effective_step is None:
            return list(range(self.step))
        return list(self.effective_step)


def initialize_weights(module: nn.Module) -> None:
    """Initialize learnable layers with the original small-weight scheme."""

    if isinstance(module, nn.Conv2d):
        scale = 0.1
        bound = scale * np.sqrt(1 / np.prod(module.weight.shape[:-1]))
        module.weight.data.uniform_(-bound, bound)
    elif isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


class ConvLSTMCell(nn.Module):
    """Convolutional LSTM cell with circular padding."""

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        input_kernel_size: int,
        input_stride: int,
        input_padding: int,
    ) -> None:
        super().__init__()

        hidden_kernel_size = 3
        self.hidden_channels = hidden_channels

        self.Wxi = nn.Conv2d(
            input_channels,
            hidden_channels,
            input_kernel_size,
            input_stride,
            input_padding,
            bias=True,
            padding_mode="circular",
        )
        self.Whi = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            hidden_kernel_size,
            1,
            padding=1,
            bias=False,
            padding_mode="circular",
        )

        self.Wxf = nn.Conv2d(
            input_channels,
            hidden_channels,
            input_kernel_size,
            input_stride,
            input_padding,
            bias=True,
            padding_mode="circular",
        )
        self.Whf = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            hidden_kernel_size,
            1,
            padding=1,
            bias=False,
            padding_mode="circular",
        )

        self.Wxc = nn.Conv2d(
            input_channels,
            hidden_channels,
            input_kernel_size,
            input_stride,
            input_padding,
            bias=True,
            padding_mode="circular",
        )
        self.Whc = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            hidden_kernel_size,
            1,
            padding=1,
            bias=False,
            padding_mode="circular",
        )

        self.Wxo = nn.Conv2d(
            input_channels,
            hidden_channels,
            input_kernel_size,
            input_stride,
            input_padding,
            bias=True,
            padding_mode="circular",
        )
        self.Who = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            hidden_kernel_size,
            1,
            padding=1,
            bias=False,
            padding_mode="circular",
        )

        nn.init.zeros_(self.Wxi.bias)
        nn.init.zeros_(self.Wxf.bias)
        nn.init.zeros_(self.Wxc.bias)
        self.Wxo.bias.data.fill_(1.0)

    def forward(
            self, 
            x: torch.Tensor, 
            h: torch.Tensor, 
            c: torch.Tensor
        ) -> TensorPair:
        input_gate = torch.sigmoid(self.Wxi(x) + self.Whi(h))
        forget_gate = torch.sigmoid(self.Wxf(x) + self.Whf(h))
        cell_state = forget_gate * c + input_gate * torch.tanh(self.Wxc(x) + self.Whc(h))
        output_gate = torch.sigmoid(self.Wxo(x) + self.Who(h))
        hidden_state = output_gate * torch.tanh(cell_state)
        return hidden_state, cell_state

    def init_hidden_tensor(
            self, 
            prev_state: TensorPair, 
            device: torch.device
        ) -> TensorPair:
        return prev_state[0].to(device), prev_state[1].to(device)


class EncoderBlock(nn.Module):
    """Encoder block used for spatial downsampling."""

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        input_kernel_size: int,
        input_stride: int,
        input_padding: int,
    ) -> None:
        super().__init__()
        self.conv = weight_norm(
            nn.Conv2d(
                input_channels,
                hidden_channels,
                input_kernel_size,
                input_stride,
                input_padding,
                bias=True,
                padding_mode="circular",
            )
        )
        self.activation = nn.ReLU()
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.conv(x))


class PhyCRNet(nn.Module):
    """Physics-informed convolutional recurrent neural network."""

    def __init__(
        self,
        input_channels: int,
        hidden_channels: Sequence[int],
        input_kernel_size: Sequence[int],
        input_stride: Sequence[int],
        input_padding: Sequence[int],
        dt: float,
        num_layers: Sequence[int],
        upscale_factor: int,
        step: int = 1,
        effective_step: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__()

        self.input_channels = [input_channels] + list(hidden_channels)
        self.hidden_channels = list(hidden_channels)
        self.input_kernel_size = list(input_kernel_size)
        self.input_stride = list(input_stride)
        self.input_padding = list(input_padding)
        self.step = step
        self.effective_step = list(effective_step) if effective_step is not None else list(range(step))
        self.dt = dt
        self.upscale_factor = upscale_factor
        self.num_encoder = int(num_layers[0])
        self.num_convlstm = int(num_layers[1])

        self.encoders = nn.ModuleList()
        for index in range(self.num_encoder):
            self.encoders.append(
                EncoderBlock(
                    input_channels=self.input_channels[index],
                    hidden_channels=self.hidden_channels[index],
                    input_kernel_size=self.input_kernel_size[index],
                    input_stride=self.input_stride[index],
                    input_padding=self.input_padding[index],
                )
            )

        self.convlstm_layers = nn.ModuleList()
        for index in range(self.num_encoder, self.num_encoder + self.num_convlstm):
            self.convlstm_layers.append(
                ConvLSTMCell(
                    input_channels=self.input_channels[index],
                    hidden_channels=self.hidden_channels[index],
                    input_kernel_size=self.input_kernel_size[index],
                    input_stride=self.input_stride[index],
                    input_padding=self.input_padding[index],
                )
            )

        self.output_layer = nn.Conv2d(
            2, 
            2, 
            kernel_size=5, 
            stride=1, 
            padding=2, 
            padding_mode="circular"
        )
        self.pixelshuffle = nn.PixelShuffle(self.upscale_factor)

        self.apply(initialize_weights)
        nn.init.zeros_(self.output_layer.bias)

    def forward(
        self,
        initial_state: Sequence[TensorPair],
        x: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], List[TensorPair]]:
        internal_state: List[TensorPair] = []
        outputs: List[torch.Tensor] = []
        second_last_state: List[TensorPair] = []
        device = x.device

        for step in range(self.step):
            residual = x
            encoded = x

            for encoder in self.encoders:
                encoded = encoder(encoded)

            lstm_output = encoded
            for layer_index, convlstm in enumerate(self.convlstm_layers):
                if step == 0:
                    internal_state.append(
                        convlstm.init_hidden_tensor(
                            initial_state[layer_index], 
                            device
                        )
                    )
                h, c = internal_state[layer_index]
                lstm_output, new_c = convlstm(lstm_output, h, c)
                internal_state[layer_index] = (lstm_output, new_c)

            decoded = self.pixelshuffle(lstm_output)
            decoded = self.output_layer(decoded)
            x = residual + self.dt * decoded

            if step == self.step - 2:
                second_last_state = [(h.clone(), c.clone()) for h, c in internal_state]
            if step in self.effective_step:
                outputs.append(x)

        return outputs, second_last_state


class Conv2dDerivative(nn.Module):
    """Fixed 2D finite-difference operator."""

    def __init__(
            self, 
            derivative_filter: Sequence[Sequence[Sequence[Sequence[float]]]], 
            resol: float, 
            kernel_size: int = 3
        ) -> None:
        super().__init__()
        self.resol = resol
        self.filter = nn.Conv2d(1, 1, kernel_size, 1, padding=0, bias=False)
        self.filter.weight = nn.Parameter(
            torch.FloatTensor(derivative_filter), 
            requires_grad=False
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.filter(inputs) / self.resol


class Conv1dDerivative(nn.Module):
    """Fixed 1D finite-difference operator."""

    def __init__(
            self, 
            derivative_filter: Sequence[Sequence[Sequence[float]]], 
            resol: float, 
            kernel_size: int = 3
        ) -> None:
        super().__init__()
        self.resol = resol
        self.filter = nn.Conv1d(1, 1, kernel_size, 1, padding=0, bias=False)
        self.filter.weight = nn.Parameter(
            torch.FloatTensor(derivative_filter), 
            requires_grad=False
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.filter(inputs) / self.resol


class PhysicsLossGenerator(nn.Module):
    """Construct Burgers-equation residuals for physics-informed training."""

    def __init__(
            self, 
            dt: float, 
            dx: float, 
            reynolds_number: float = 200.0
        ) -> None:
        super().__init__()
        self.reynolds_number = reynolds_number
        self.laplace = Conv2dDerivative(
            LAPLACIAN_KERNEL, 
            resol=dx**2, 
            kernel_size=5
        )
        self.dx = Conv2dDerivative(PARTIAL_X_KERNEL, resol=dx, kernel_size=5)
        self.dy = Conv2dDerivative(PARTIAL_Y_KERNEL, resol=dx, kernel_size=5)
        self.dt = Conv1dDerivative([[[-1, 0, 1]]], resol=dt * 2, kernel_size=3)

    def get_phy_loss(
            self, 
            output: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        laplace_u = self.laplace(output[1:-1, 0:1, :, :])
        laplace_v = self.laplace(output[1:-1, 1:2, :, :])

        u_x = self.dx(output[1:-1, 0:1, :, :])
        u_y = self.dy(output[1:-1, 0:1, :, :])
        v_x = self.dx(output[1:-1, 1:2, :, :])
        v_y = self.dy(output[1:-1, 1:2, :, :])

        u_t = self._temporal_derivative(output[:, 0:1, 2:-2, 2:-2])
        v_t = self._temporal_derivative(output[:, 1:2, 2:-2, 2:-2])

        u = output[1:-1, 0:1, 2:-2, 2:-2]
        v = output[1:-1, 1:2, 2:-2, 2:-2]

        residual_u = u_t + u * u_x + v * u_y - (1.0 / self.reynolds_number) * laplace_u
        residual_v = v_t + u * v_x + v * v_y - (1.0 / self.reynolds_number) * laplace_v
        return residual_u, residual_v

    def _temporal_derivative(self, values: torch.Tensor) -> torch.Tensor:
        time_length = values.shape[0]
        width = values.shape[3]
        height = values.shape[2]

        conv_input = values.permute(2, 3, 1, 0).reshape(width * height, 1, time_length)
        derivative = self.dt(conv_input)
        derivative = derivative.reshape(height, width, 1, time_length - 2)
        return derivative.permute(3, 2, 0, 1)


def compute_physics_loss(
        output: torch.Tensor, 
        loss_generator: PhysicsLossGenerator
    ) -> torch.Tensor:
    """Compute the Burgers residual loss with periodic padding."""

    output = torch.cat(
        (
            output[:, :, :, -2:], 
            output, 
            output[:, :, :, 0:3]
        ), 
        dim=3
    )
    output = torch.cat(
        (
            output[:, :, -2:, :], 
            output, 
            output[:, :, 0:3, :]
        ), 
        dim=2
    )

    residual_u, residual_v = loss_generator.get_phy_loss(output)
    target_u = torch.zeros_like(residual_u, device=output.device)
    target_v = torch.zeros_like(residual_v, device=output.device)
    mse_loss = nn.MSELoss()
    return mse_loss(residual_u, target_u) + mse_loss(residual_v, target_v)


def build_model(config: PhyCRNetConfig) -> PhyCRNet:
    """Factory helper for constructing the Burgers PhyCRNet model."""

    return PhyCRNet(
        input_channels=config.input_channels,
        hidden_channels=config.hidden_channels,
        input_kernel_size=config.input_kernel_size,
        input_stride=config.input_stride,
        input_padding=config.input_padding,
        dt=config.dt,
        num_layers=config.num_layers,
        upscale_factor=config.upscale_factor,
        step=config.step,
        effective_step=config.resolved_effective_step(),
    )
