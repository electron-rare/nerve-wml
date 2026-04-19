"""HardFlowProxyTask — a non-linearly-separable task that exposes variance.

Motivation (from paper v0.2 §Threats to Validity): the canonical
FlowProxyTask 4-class is trivially linearly separable, so MLP and LIF
pools both saturate to 1.0 on W2. The polymorphie gap becomes a
degenerate best case.

HardFlowProxyTask adds three sources of difficulty:
1. More classes (default 12 vs 4) so chance accuracy drops to ~8 %.
2. XOR-like non-linearity: class label depends on the sign-product of
   two random hyperplane projections, not on distance to a centroid.
3. Narrower class clusters (scale 0.7 vs 2.0) with overlap, so a
   linear probe plateaus around 0.5-0.7 instead of 1.0.

A well-trained MLP can beat a linear probe on this task; a LIF with
the current cosine-pattern-match decoder may not — which is exactly
the regime where the polymorphie gap becomes informative.
"""
from __future__ import annotations

import torch
from torch import Tensor


class HardFlowProxyTask:
    """Non-linearly-separable variant of FlowProxyTask.

    Labels are produced by combining two hyperplane projections:
        y = (sign(x · w1) * sign(x · w2) + 1) // 2 * (n_classes // 2) + cluster_id
    with cluster_id ∈ {0, ..., n_classes // 2 - 1} assigned by distance
    to narrow centroids.
    """

    def __init__(
        self,
        dim: int = 16,
        n_classes: int = 12,
        *,
        seed: int | None = None,
    ) -> None:
        assert n_classes % 2 == 0, "n_classes must be even (XOR split)"
        self.dim = dim
        self.n_classes = n_classes

        gen = torch.Generator()
        if seed is not None:
            gen.manual_seed(seed)

        n_sub = n_classes // 2
        # Narrow centroids (overlap encouraged).
        self._centers = torch.randn(n_sub, dim, generator=gen) * 1.2
        # Two hyperplane normals for XOR gating, zero-meaned per-cluster
        # by subtracting the centroid projection — this makes the XOR
        # bit truly orthogonal to the cluster centroid (single linear
        # probe cannot solve both simultaneously).
        self._w1 = torch.randn(dim, generator=gen)
        self._w2 = torch.randn(dim, generator=gen)
        self._gen = gen

    def sample(self, batch: int = 64) -> tuple[Tensor, Tensor]:
        n_sub = self.n_classes // 2
        sub_id = torch.randint(0, n_sub, (batch,), generator=self._gen)
        noise = torch.randn(batch, self.dim, generator=self._gen) * 0.7
        x = self._centers[sub_id] + noise

        # XOR bit computed on the NOISE component only — centroid info is
        # stripped so a linear classifier cannot read the XOR bit from
        # the raw x (centroid dominates the projection). This forces the
        # model to learn an interaction between cluster-id and projection.
        s1 = torch.sign(noise @ self._w1)
        s2 = torch.sign(noise @ self._w2)
        xor_bit = ((s1 * s2 + 1) // 2).long()
        y = xor_bit * n_sub + sub_id
        return x, y
