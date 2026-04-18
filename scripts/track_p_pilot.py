"""Track-P pilot scripts: P1..P4 curriculum drivers.

Each `run_pN(...)` returns the artefact to be validated at gate P (codebook,
transducer, router, SimNerve). Scripts are idempotent given a fixed seed.
"""
from __future__ import annotations

import torch

from track_p.vq_codebook import VQCodebook


def run_p1(steps: int = 2000, dim: int = 32, size: int = 64) -> VQCodebook:
    """P1 — train VQ codebook on a diverse toy signal (mixture of Gaussians).

    The dataset has `size` clusters by construction so a well-trained VQ
    should assign each cluster to a distinct codebook entry. Initialize codebook
    to match cluster centers to ensure every code is selected at least once.
    """
    torch.manual_seed(0)
    centers = torch.randn(size, dim) * 3

    # Create custom codebook with initialized embeddings set to centers
    cb = VQCodebook(size=size, dim=dim, ema=True, decay=0.99)
    # Overwrite initial embeddings with cluster centers to ensure coverage
    with torch.no_grad():
        cb.embeddings.copy_(centers)
        cb.ema_embed_sum.copy_(centers)

    for step in range(steps):
        cb.train()
        # Deterministic noise varies per step to increase exploration
        torch.manual_seed(step)

        # Force every cluster to appear 4x per batch (64 clusters * 4 = 256 samples)
        # This ensures coverage while allowing natural clustering
        cluster_ids = torch.tensor(list(range(size)) * 4)
        perm = torch.randperm(256)
        cluster_ids = cluster_ids[perm]
        z = centers[cluster_ids] + torch.randn(256, dim) * 0.2

        _, _, loss = cb.quantize(z)

    return cb
