"""Gaussian random field sampling utilities."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch


class GaussianRF:
    """Sample Gaussian random fields using a spectral representation."""

    def __init__(
        self,
        dim: int,
        size: int,
        alpha: float = 2,
        tau: float = 3,
        sigma: Optional[float] = None,
        boundary: str = "periodic",
        device: Optional[torch.device] = None,
    ) -> None:
        if dim not in (1, 2, 3):
            raise ValueError(f"Unsupported dimension: {dim}. Expected 1, 2, or 3.")
        if boundary != "periodic":
            raise ValueError(f"Unsupported boundary condition: {boundary}.")

        self.dim = dim
        self.size = tuple(size for _ in range(dim))
        self.device = device
        self.boundary = boundary
        self.alpha = alpha
        self.tau = tau
        self.sigma = sigma if sigma is not None else tau ** (0.5 * (2 * alpha - dim))
        self.sqrt_eig = self._build_sqrt_eigenvalues(size)

    def _build_frequency_axis(self, size: int) -> torch.Tensor:
        """Construct the symmetric frequency axis used in the spectrum."""

        k_max = size // 2
        positive = torch.arange(start=0, end=k_max, step=1, device=self.device)
        negative = torch.arange(start=-k_max, end=0, step=1, device=self.device)
        return torch.cat((positive, negative), dim=0)

    def _build_wave_numbers(self, size: int) -> Tuple[torch.Tensor, ...]:
        """Construct wave-number grids for the configured dimension."""

        frequency_axis = self._build_frequency_axis(size)

        if self.dim == 1:
            return (frequency_axis,)

        if self.dim == 2:
            wave_numbers = frequency_axis.repeat(size, 1)
            return wave_numbers.transpose(0, 1), wave_numbers

        wave_numbers = frequency_axis.repeat(size, size, 1)
        return (
            wave_numbers.transpose(1, 2),
            wave_numbers,
            wave_numbers.transpose(0, 2),
        )

    def _build_sqrt_eigenvalues(self, size: int) -> torch.Tensor:
        """Construct the square root of the covariance eigenvalues."""

        wave_numbers = self._build_wave_numbers(size)
        squared_norm = sum(component**2 for component in wave_numbers)

        scaling = (size**self.dim) * math.sqrt(2.0) * self.sigma
        sqrt_eig = scaling * ((4 * (math.pi**2) * squared_norm + self.tau**2) ** (-self.alpha / 2.0))
        sqrt_eig[(0,) * self.dim] = 0.0
        return sqrt_eig

    def sample(self, num_samples: int) -> torch.Tensor:
        """Draw samples with shape ``(num_samples, *self.size)``."""

        coeff = torch.randn(num_samples, *self.size, 2, device=self.device)
        coeff[..., 0] = self.sqrt_eig * coeff[..., 0]
        coeff[..., 1] = self.sqrt_eig * coeff[..., 1]

        samples = torch.ifft(coeff, self.dim, normalized=False)
        return samples[..., 0]
