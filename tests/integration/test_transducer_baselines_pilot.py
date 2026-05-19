import pytest

from scripts.transducer_baselines_pilot import run_transducer_benchmark


@pytest.mark.slow
def test_run_transducer_benchmark_reports_all_methods():
    results = run_transducer_benchmark(steps=300, seed=0)
    assert set(results) == {
        "learned",
        "procrustes",
        "relative_rep",
        "vec2vec",
    }
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
