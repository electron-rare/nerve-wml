"""Unit tests for nerve_wml.methodology.mi_mine_estimator.

MINE training is stochastic; we use small arrays (N=1000, d=2),
short runs (n_epochs=100), and generous tolerances to keep the
test suite deterministic and fast (<10s total on CPU).
"""
from __future__ import annotations

import numpy as np
import pytest

from nerve_wml.methodology.mi_mine_estimator import mi_mine


def _correlated_gaussian_pair(
    n: int, rho: float, seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    cov = np.array([[1.0, rho], [rho, 1.0]])
    samples = rng.multivariate_normal([0.0, 0.0], cov, size=n)
    return samples[:, :1], samples[:, 1:]


def test_mine_independent_gaussians_near_zero() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((1000, 2)).astype(np.float32)
    y = rng.standard_normal((1000, 2)).astype(np.float32)
    mi = mi_mine(
        x, y, hidden=32, n_epochs=150, batch_size=128, seed=0, tail_average=30,
    )
    assert mi < 0.20


def test_mine_correlated_gaussians_recovers_sign() -> None:
    """Strong correlation should yield MI > 0.3 nats; tolerance wide."""
    x, y = _correlated_gaussian_pair(n=1500, rho=0.8, seed=0)
    mi = mi_mine(
        x, y, hidden=64, n_epochs=300, batch_size=256, seed=0, tail_average=50,
    )
    assert mi > 0.25


def test_mine_strong_vs_weak_correlation_monotone() -> None:
    x_w, y_w = _correlated_gaussian_pair(n=1000, rho=0.3, seed=0)
    x_s, y_s = _correlated_gaussian_pair(n=1000, rho=0.9, seed=0)
    mi_w = mi_mine(
        x_w, y_w, hidden=32, n_epochs=150, batch_size=128, seed=0,
        tail_average=30,
    )
    mi_s = mi_mine(
        x_s, y_s, hidden=32, n_epochs=150, batch_size=128, seed=0,
        tail_average=30,
    )
    assert mi_s > mi_w


def test_mine_shape_mismatch_raises() -> None:
    x = np.zeros((100, 2), dtype=np.float32)
    y = np.zeros((50, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="shape"):
        mi_mine(x, y, n_epochs=10, batch_size=32)


def test_mine_too_few_samples_raises() -> None:
    x = np.zeros((20, 2), dtype=np.float32)
    y = np.zeros((20, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="samples"):
        mi_mine(x, y, batch_size=128)


def test_mine_reproducible_with_seed() -> None:
    rng = np.random.default_rng(42)
    x = rng.standard_normal((500, 2)).astype(np.float32)
    y = rng.standard_normal((500, 2)).astype(np.float32)
    mi1 = mi_mine(
        x, y, hidden=32, n_epochs=100, batch_size=128, seed=7, tail_average=25,
    )
    mi2 = mi_mine(
        x, y, hidden=32, n_epochs=100, batch_size=128, seed=7, tail_average=25,
    )
    assert abs(mi1 - mi2) < 1e-9
