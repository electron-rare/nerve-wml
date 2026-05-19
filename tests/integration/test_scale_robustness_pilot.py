import pytest

from scripts.scale_robustness_pilot import run_scale_robustness


@pytest.mark.slow
def test_run_scale_robustness_returns_rows_per_size():
    rows = run_scale_robustness(
        sizes=(64, 128, 256), seed=0
    )
    assert [r["n"] for r in rows] == [64, 128, 256]
    for r in rows:
        assert "hsic" in r and "cknna" in r
        assert 0.0 <= r["cknna"] <= 1.0


@pytest.mark.slow
def test_scale_robustness_hsic_finite():
    rows = run_scale_robustness(sizes=(64, 128), seed=1)
    import math

    assert all(math.isfinite(r["hsic"]) for r in rows)
