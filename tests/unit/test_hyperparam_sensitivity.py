from scripts.hyperparam_sensitivity import _one_config


def test_one_config_returns_valid_metrics() -> None:
    """Smoke test for scripts/hyperparam_sensitivity._one_config.

    Ensures the per-config routine that backs Test (11) in paper 1
    is importable and returns the three expected metrics in
    plausible ranges. Uses a minimal budget (steps=20) to keep the
    test fast; the released artifact uses steps=400.
    """
    result = _one_config(d_hidden=8, lr=1e-2, seed=0, steps=20)

    assert set(result.keys()) == {"acc_mlp", "acc_lif", "gap"}

    acc_mlp = result["acc_mlp"]
    acc_lif = result["acc_lif"]
    gap = result["gap"]

    assert isinstance(acc_mlp, float)
    assert isinstance(acc_lif, float)
    assert isinstance(gap, float)

    assert 0.0 <= acc_mlp <= 1.0
    assert 0.0 <= acc_lif <= 1.0
    # gap = abs(acc_mlp - acc_lif) / max(acc_mlp, 1e-6): non-negative
    # but can exceed 1.0 in pathological cases (near-zero acc_mlp).
    assert gap >= 0.0
