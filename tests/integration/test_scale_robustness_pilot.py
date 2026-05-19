import pytest

from scripts.scale_robustness_pilot import run_scale_robustness


@pytest.mark.slow
def test_run_scale_robustness_returns_rows_per_size():
    out = run_scale_robustness(
        sizes=(64, 128, 256), seed=0
    )
    rows = out["rows"]
    assert [r["n"] for r in rows] == [64, 128, 256]
    for r in rows:
        assert "hsic" in r and "cknna" in r
        assert 0.0 <= r["cknna"] <= 1.0


@pytest.mark.slow
def test_scale_robustness_hsic_finite():
    out = run_scale_robustness(sizes=(64, 128), seed=1)
    rows = out["rows"]
    import math

    assert all(math.isfinite(r["hsic"]) for r in rows)


@pytest.mark.slow
def test_null_rows_below_real():
    out = run_scale_robustness(sizes=(64, 128), seed=0)
    assert "rows" in out and "null_rows" in out
    real_cknna = [r["cknna"] for r in out["rows"]]
    null_cknna = [r["cknna"] for r in out["null_rows"]]
    assert max(null_cknna) < max(real_cknna)
