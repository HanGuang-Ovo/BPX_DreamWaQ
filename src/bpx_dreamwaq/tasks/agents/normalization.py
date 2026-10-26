"""Explicitly updated running moments; inference never changes statistics."""

import torch
from torch import nn


class ObservationNormalizer(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.register_buffer("mean", torch.zeros(width))
        self.register_buffer("variance", torch.ones(width))
        self.register_buffer("count", torch.tensor(0.0, dtype=torch.float64))

    @torch.no_grad()
    def update(self, values: torch.Tensor):
        values = values.reshape(-1, self.mean.numel())
        if not torch.isfinite(values).all():
            raise ValueError("Non-finite observations cannot update normalization")
        n = values.shape[0]
        if n == 0:
            return
        batch_mean = values.mean(0)
        batch_var = values.var(0, unbiased=False)
        total = self.count + n
        fraction = (n / total).to(values.dtype)
        delta = batch_mean - self.mean
        self.variance.copy_(
            (1 - fraction) * self.variance + fraction * batch_var
            + delta.square() * fraction * (1 - fraction)
        )
        self.mean.add_(delta * fraction)
        self.count.copy_(total)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return (values - self.mean) / self.variance.clamp_min(1.0e-4).sqrt()
