import pytest

from scripts.transducer_baselines_pilot import (
    _build_task,
    _mi_entropy_bits,
    _train_learned,
    run_transducer_benchmark,
)


@pytest.mark.slow
def test_run_transducer_benchmark_reports_all_methods():
    results = run_transducer_benchmark(steps=300, seed=0)
    assert {
        "learned",
        "procrustes",
        "relative_rep",
        "vec2vec",
        "null",
    } <= set(results)
    for name, row in results.items():
        # MI is non-negative and <= log2(64) = 6 bits.
        assert 0.0 <= row["mi_bits"] <= 6.01, name
        # Entropy of the dst-code distribution, also <= 6 bits.
        assert 0.0 <= row["entropy_bits"] <= 6.01, name


@pytest.mark.slow
def test_learned_transducer_beats_random_floor():
    results = run_transducer_benchmark(steps=300, seed=0)
    # The learned transducer must transmit more than the ~0-bit floor.
    assert results["learned"]["mi_bits"] > 0.5


@pytest.mark.slow
def test_null_arm_below_learned():
    res = run_transducer_benchmark(steps=300, seed=0)
    assert "null" in res
    assert "mi_bits" in res["null"]
    assert res["null"]["mi_bits"] < res["learned"]["mi_bits"] - 0.3


@pytest.mark.slow
def test_learned_lr_changes_mi() -> None:
    src, dst, *_ = _build_task(0)
    learned_a = _train_learned(src, dst, 200, lr=1e-2)
    learned_b = _train_learned(src, dst, 200, lr=5e-2)
    mi_a = _mi_entropy_bits(learned_a.forward(src, hard=True), dst)["mi_bits"]
    mi_b = _mi_entropy_bits(learned_b.forward(src, hard=True), dst)["mi_bits"]
    assert mi_a != mi_b
