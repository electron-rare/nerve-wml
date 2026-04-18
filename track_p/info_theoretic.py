"""L2 info-theoretic metrics for Gate P validation.

See spec §8.2.
"""
from __future__ import annotations

from torch import Tensor

from .vq_codebook import VQCodebook


def empirical_capacity_bps(code_rate_hz: float, code_histogram: Tensor) -> float:
    """Shannon entropy of the code distribution × code rate = bits/s throughput."""
    p = code_histogram / (code_histogram.sum() + 1e-9)
    ent_bits = -(p * (p + 1e-9).log2()).sum().item()
    return code_rate_hz * ent_bits


def dead_code_fraction(cb: VQCodebook) -> float:
    """Fraction of codes never assigned during usage tracking."""
    dead = (cb.usage_counter == 0)
    if isinstance(dead, bool):
        return float(dead)
    return dead.float().mean().item()


def kl_divergence(p: Tensor, q: Tensor) -> Tensor:
    """KL(p ‖ q) in bits — used for π/ε disambiguation."""
    p = p / (p.sum() + 1e-9)
    q = q / (q.sum() + 1e-9)
    return (p * ((p + 1e-9).log2() - (q + 1e-9).log2())).sum()
