"""Composite loss for Track-W training.

L_total = L_task + λ_vq · L_vq + λ_sep · L_role_sep + λ_surprise · L_surprise

See spec §7.1. W1-W3 only use the first two terms; sep and surprise appear in
W3 and W4 as the WMLs learn to distinguish π from ε.
"""
from __future__ import annotations

from torch import Tensor


def composite_loss(
    *,
    task_loss: Tensor,
    vq_loss:   Tensor,
    sep_loss:       Tensor | None = None,
    surprise_loss:  Tensor | None = None,
    lam_vq:       float = 0.25,
    lam_sep:      float = 0.05,
    lam_surprise: float = 0.10,
) -> Tensor:
    total = task_loss + lam_vq * vq_loss
    if sep_loss is not None:
        total = total + lam_sep * sep_loss
    if surprise_loss is not None:
        total = total + lam_surprise * surprise_loss
    return total
