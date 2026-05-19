import numpy as np

from nerve_wml.methodology.hsic_cknna import cknna, hsic_debiased


def test_hsic_debiased_zero_for_independent():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((400, 8))
    y = rng.standard_normal((400, 8))
    val = hsic_debiased(x, y)
    # Debiased HSIC is unbiased under independence: ~0, can be slightly negative.
    assert abs(val) < 0.05


def test_hsic_debiased_positive_for_dependent():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((400, 8))
    y = x + 0.1 * rng.standard_normal((400, 8))
    assert hsic_debiased(x, y) > 0.1


def test_hsic_debiased_symmetric():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((200, 4))
    y = rng.standard_normal((200, 4))
    a = hsic_debiased(x, y)
    b = hsic_debiased(y, x)
    assert abs(a - b) < 1e-9


def test_cknna_one_for_identical():
    rng = np.random.default_rng(3)
    x = rng.standard_normal((120, 16))
    # CKNNA is scale-invariant: a global rescale must not change the score.
    assert abs(cknna(x, 3.0 * x, k=10) - 1.0) < 1e-9


def test_cknna_low_for_independent():
    rng = np.random.default_rng(4)
    x = rng.standard_normal((200, 16))
    y = rng.standard_normal((200, 16))
    # Random neighborhoods overlap at chance ~ k / N.
    assert cknna(x, y, k=10) < 0.2


def test_cknna_high_for_aligned():
    rng = np.random.default_rng(5)
    x = rng.standard_normal((200, 16))
    y = x + 0.05 * rng.standard_normal((200, 16))
    assert cknna(x, y, k=10) > 0.6
