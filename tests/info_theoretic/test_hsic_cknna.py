import numpy as np

from nerve_wml.methodology.hsic_cknna import hsic_debiased


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
