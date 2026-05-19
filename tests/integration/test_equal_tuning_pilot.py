"""Integration test for the equal-tuning pilot."""
from __future__ import annotations

import pytest

from scripts.equal_tuning_pilot import run_equal_tuning


@pytest.mark.slow
def test_equal_tuning_shape() -> None:
    out = run_equal_tuning(budget=2, seeds_per_trial=(0, 1))
    assert set(out) >= {"learned", "vec2vec", "relrep", "procrustes"}
    for _method, payload in out.items():
        assert "trials" in payload
        assert "best" in payload
        assert isinstance(payload["trials"], list)
        assert len(payload["trials"]) >= 1
        best = payload["best"]
        assert "params" in best
        assert "mi_mean" in best
        assert "mi_ci95_low" in best
        assert "mi_ci95_high" in best
