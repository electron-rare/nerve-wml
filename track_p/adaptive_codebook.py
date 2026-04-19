"""AdaptiveCodebook — per-WML alphabet that can shrink/grow.

Wraps a fixed 64-slot VQCodebook with a boolean active_mask so resizing
is logical (mask update) rather than physical (tensor reshape). Existing
gate tests keep running — they just see the full 64 alphabet when no
resize has been called yet.

Spec §13 Q1: "could the 64-code alphabet grow or shrink adaptively
per WML once the system is stable?"

Plan 8 Tasks 1-3.
"""
from __future__ import annotations

import torch
from torch import Tensor

from track_p.vq_codebook import VQCodebook


class AdaptiveCodebook:
    """VQCodebook wrapper with a logical active_mask.

    - current_size() reports the count of active slots.
    - shrink(min_usage_frac) retires slots below a usage threshold.
    - grow(top_k_to_split) seeds new slots from perturbed parents.

    Underlying storage stays at 64 slots; mask controls which rows the
    public API reports as "live".
    """

    def __init__(self, size: int = 64, dim: int = 128, *, ema: bool = True,
                 seed: int = 0) -> None:
        self.storage = VQCodebook(size=size, dim=dim, ema=ema)
        self.active_mask = torch.ones(size, dtype=torch.bool)
        self._seed = seed
        self._initial_size = size

    def current_size(self) -> int:
        """Number of active slots (logically: the effective alphabet size)."""
        return int(self.active_mask.sum().item())

    def active_indices(self) -> Tensor:
        """LongTensor of indices where active_mask is True."""
        return torch.nonzero(self.active_mask, as_tuple=False).squeeze(-1)

    def active_embeddings(self) -> Tensor:
        """Return only the rows marked active."""
        return self.storage.embeddings[self.active_mask]

    def shrink(self, min_usage_frac: float = 0.01, *, min_codes: int = 8) -> list[int]:
        """Retire codes whose usage fraction is below min_usage_frac.

        Never shrinks below `min_codes` to keep the alphabet functional.
        Returns the list of storage indices that remain active after
        the shrink (for bookkeeping / transducer resize callers).
        """
        counts = self.storage.usage_counter.float()
        total = counts.sum().item()
        if total == 0:
            # No usage yet; nothing to shrink.
            return self.active_indices().tolist()

        usage_frac = counts / total
        # Candidates for retirement: currently active AND below threshold.
        to_retire = self.active_mask & (usage_frac < min_usage_frac)
        n_retire = int(to_retire.sum().item())
        n_active = self.current_size()

        # Honour the min_codes floor.
        if n_active - n_retire < min_codes:
            # Retire only the least-used ones down to the floor.
            allowed = max(0, n_active - min_codes)
            if allowed == 0:
                return self.active_indices().tolist()
            # Sort by usage (ascending) among retirement candidates.
            candidate_idx = torch.nonzero(to_retire, as_tuple=False).squeeze(-1)
            candidate_counts = counts[candidate_idx]
            order = candidate_counts.argsort()
            chosen = candidate_idx[order[:allowed]]
            to_retire = torch.zeros_like(to_retire)
            to_retire[chosen] = True

        # Apply.
        self.active_mask &= ~to_retire
        return self.active_indices().tolist()

    def grow(
        self,
        top_k_to_split: int = 4,
        *,
        perturb_scale: float = 0.01,
        seed: int = 0,
    ) -> list[int]:
        """Split the top-K most-used codes by seeding vacant slots with
        perturbed copies of their parents.

        If there are fewer vacant slots than top_k_to_split, uses as many
        as available. Returns the list of newly-activated storage indices.
        """
        vacant = torch.nonzero(~self.active_mask, as_tuple=False).squeeze(-1)
        if vacant.numel() == 0:
            return []

        counts = self.storage.usage_counter.float()
        # Only consider currently-active codes as parents.
        active_counts = counts.clone()
        active_counts[~self.active_mask] = -1.0  # exclude inactive
        n_split = min(top_k_to_split, vacant.numel())

        top_parents = active_counts.topk(n_split).indices  # active storage idx
        chosen_vacant = vacant[:n_split]

        gen = torch.Generator()
        gen.manual_seed(seed)
        with torch.no_grad():
            for parent_idx, new_idx in zip(
                top_parents.tolist(), chosen_vacant.tolist(), strict=True,
            ):
                parent_emb = self.storage.embeddings[parent_idx]
                perturbation = torch.randn(
                    parent_emb.shape, generator=gen,
                ) * perturb_scale
                self.storage.embeddings[new_idx] = parent_emb + perturbation
                if self.storage.ema:
                    self.storage.ema_embed_sum[new_idx] = (
                        parent_emb + perturbation
                    )
                    self.storage.ema_cluster_size[new_idx] = 1.0
                self.storage.usage_counter[new_idx] = 0
                self.active_mask[new_idx] = True

        return chosen_vacant.tolist()

    def quantize_active(self, z: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Quantize z but only consider active rows.

        Returns (active_indices, quantized, loss).
        active_indices are in [0, size) — the public alphabet space the
        caller sees. Internally we map these back to storage row indices.
        """
        active_rows = self.active_indices()
        sub_embed = self.storage.embeddings[active_rows]
        dist = torch.cdist(z, sub_embed)
        sub_idx = dist.argmin(dim=-1)
        active_idx = active_rows[sub_idx]                       # storage-space
        quantized = self.storage.embeddings[active_idx]

        # Straight-through + simple commitment loss (mirrors VQCodebook).
        commit_loss = 0.25 * ((z - quantized.detach()) ** 2).mean()
        codebook_loss = ((quantized - z.detach()) ** 2).mean()
        loss = commit_loss + codebook_loss
        quantized_st = z + (quantized - z).detach()
        return active_idx, quantized_st, loss
