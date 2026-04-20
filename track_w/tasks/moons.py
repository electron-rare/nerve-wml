"""MoonsTask — a 2-class two-moons task, second non-linear distribution.

Added in v1.1.4 to close §13.1 debt #14 (all v1.1 measurements done
on HardFlowProxyTask alone). The Moons topology is structurally
different from HardFlowProxyTask (XOR-on-noise with 12 classes) —
it's a 2-class interleaved-semicircle decision boundary, another
canonical non-linear benchmark.

Kept deterministic via a local torch.Generator (no global RNG
mutation). 16-dim input matches the other tasks' convention: the
two "moon" features occupy dims 0 and 1, the rest are Gaussian noise.
"""
from __future__ import annotations

import math

import torch


class MoonsTask:
    """Two interleaving half-moons + noise dims.

    sample(batch) returns (x, y) with x[:, 0:2] the moon coordinates
    and x[:, 2:] Gaussian noise. Labels y ∈ {0, 1}.
    """

    def __init__(
        self,
        dim:       int   = 16,
        n_classes: int   = 2,
        noise:     float = 0.2,
        *,
        seed:      int   = 0,
    ) -> None:
        if dim < 2:
            raise ValueError(f"dim must be >= 2, got {dim}")
        if n_classes != 2:
            raise ValueError("MoonsTask is a 2-class problem")
        self.dim       = dim
        self.n_classes = n_classes
        self.noise     = noise
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)

    def sample(self, batch: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (x ∈ R^batch×dim, y ∈ {0,1}^batch)."""
        labels = torch.randint(0, 2, (batch,), generator=self.generator)
        angles = torch.rand(batch, generator=self.generator) * math.pi
        # Upper moon (label 0): x = cos(a), y = sin(a).
        # Lower moon (label 1): x = 1 - cos(a), y = -sin(a) + 0.5.
        coord_x = torch.where(labels == 0, torch.cos(angles), 1.0 - torch.cos(angles))
        coord_y = torch.where(labels == 0, torch.sin(angles), -torch.sin(angles) + 0.5)
        coord = torch.stack([coord_x, coord_y], dim=1)
        coord = coord + self.noise * torch.randn(batch, 2, generator=self.generator)

        noise_tail = torch.randn(batch, self.dim - 2, generator=self.generator)
        x = torch.cat([coord, noise_tail], dim=1)
        return x, labels.long()
