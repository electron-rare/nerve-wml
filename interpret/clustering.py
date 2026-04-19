"""Torch k-means for grouping code activation centroids into semantic clusters.

~50 lines, no sklearn. Seeded via a local torch.Generator so calls never
pollute the global RNG (consistent with Plan 1 tech-debt resolution).
"""
from __future__ import annotations

import torch
from torch import Tensor


def cluster_codes_by_activation(
    centroids: Tensor,
    *,
    n_clusters: int = 8,
    max_iter: int = 10,
    seed: int = 0,
) -> Tensor:
    """k-means on code centroids.

    Args:
        centroids:  [n_codes, dim] — one activation centroid per code.
        n_clusters: number of clusters (default 8, a reasonable split of 64).
        max_iter:   k-means iterations (default 10).
        seed:       local Generator seed.

    Returns:
        labels [n_codes] long tensor in [0, n_clusters).
    """
    n_codes = centroids.shape[0]
    assert 1 <= n_clusters <= n_codes, (
        f"n_clusters {n_clusters} must be in [1, {n_codes}]"
    )

    gen = torch.Generator()
    gen.manual_seed(seed)

    # Initialise centres by sampling random code rows.
    init_idx = torch.randperm(n_codes, generator=gen)[:n_clusters]
    centres = centroids[init_idx].clone()

    labels = torch.zeros(n_codes, dtype=torch.long)
    for _ in range(max_iter):
        # Assign each code to the nearest centre (L2 distance).
        dists = torch.cdist(centroids, centres)    # [n_codes, n_clusters]
        new_labels = dists.argmin(dim=-1)

        if torch.equal(new_labels, labels):
            break
        labels = new_labels

        # Recompute centres. Empty clusters keep their old centre.
        for c in range(n_clusters):
            mask = labels == c
            if mask.any():
                centres[c] = centroids[mask].mean(dim=0)

    return labels
