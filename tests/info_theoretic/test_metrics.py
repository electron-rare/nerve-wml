import math
import torch

from track_p.info_theoretic import (
    empirical_capacity_bps,
    dead_code_fraction,
    kl_divergence,
)
from track_p.vq_codebook import VQCodebook


def test_capacity_lower_bound_on_uniform_stream():
    # 46 Hz × 6 bits = 276 bits/s when uniform
    capacity = empirical_capacity_bps(
        code_rate_hz=46.0,
        code_histogram=torch.ones(64) / 64,
    )
    assert capacity > 200
    assert capacity <= 46.0 * math.log2(64) + 1e-6


def test_dead_code_fraction_detects_unused_codes():
    cb = VQCodebook(size=64, dim=32, ema=False)
    # Send only the first 10 codes (hand-craft by directly bumping usage).
    cb.usage_counter[:10] = 100
    cb.usage_counter[10:] = 0
    frac = dead_code_fraction(cb)
    assert math.isclose(frac, 54 / 64, abs_tol=1e-6)


def test_kl_divergence_between_distinct_distributions():
    p = torch.tensor([0.9, 0.1])
    q = torch.tensor([0.1, 0.9])
    kl = kl_divergence(p, q)
    assert kl.item() > 1.0
