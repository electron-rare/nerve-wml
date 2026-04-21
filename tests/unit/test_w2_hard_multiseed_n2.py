from scripts.track_w_pilot import run_w2_hard, run_w2_hard_multiseed


def test_run_w2_hard_seed_0_is_deterministic() -> None:
    """Pin the load-bearing invariant: seed=0 reproduces bit-for-bit.

    The v1.6.0 10.71% N=2 anchor is cited in paper Claim A scaling
    law. A silent regression where a later refactor breaks seeding
    would invalidate that paper number without CI catching it.
    """
    a = run_w2_hard(steps=50, seed=0)
    b = run_w2_hard(steps=50, seed=0)
    assert a["gap"] == b["gap"]
    assert a["acc_mlp"] == b["acc_mlp"]
    assert a["acc_lif"] == b["acc_lif"]


def test_run_w2_hard_multiseed_returns_5_seed_stats() -> None:
    result = run_w2_hard_multiseed(seeds=list(range(5)), steps=200)
    assert isinstance(result, dict)
    assert "seeds" in result
    assert result["seeds"] == [0, 1, 2, 3, 4]
    assert "gaps" in result
    assert len(result["gaps"]) == 5
    assert "median_gap" in result
    assert "p25_gap" in result
    assert "p75_gap" in result
    assert "mean_acc_mlp" in result
    assert "mean_acc_lif" in result


def test_run_w2_hard_multiseed_direction_stability() -> None:
    result = run_w2_hard_multiseed(seeds=list(range(5)), steps=200)
    lif_ge_mlp = sum(
        1 for i in range(5)
        if result["accs_lif"][i] >= result["accs_mlp"][i]
    )
    assert lif_ge_mlp >= 3, f"expected LIF>=MLP in >=3 of 5 seeds, got {lif_ge_mlp}"
