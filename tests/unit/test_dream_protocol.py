"""Tests for bridge.dream_protocol lazy loader + MockConsolidator."""
import numpy as np
import pytest

from bridge.dream_protocol import assert_protocol_surface, load_dream_module
from bridge.mock_consolidator import MockConsolidator


def test_load_dream_module_returns_none_when_absent():
    """kiki_oniric is not installed by default — must return None."""
    mod = load_dream_module("nonexistent_module_xyz")
    assert mod is None


def test_load_dream_module_loads_installed_package():
    """Sanity check on a module we know exists."""
    import math as m
    loaded = load_dream_module("math")
    assert loaded is m


def test_mock_consolidator_returns_zero_delta_of_right_shape():
    trace = np.random.rand(100, 4).astype(np.float32)
    delta = MockConsolidator.consolidate(trace, n_transducers=12, alphabet_size=64)
    assert delta.shape == (12, 64, 64)
    assert delta.dtype == np.float32
    assert (delta == 0).all()


def test_assert_protocol_surface_accepts_mock():
    """MockConsolidator satisfies the surface check."""
    assert_protocol_surface(MockConsolidator)


def test_assert_protocol_surface_rejects_bad_module():
    class BadModule:
        pass

    with pytest.raises(AssertionError, match="consolidate"):
        assert_protocol_surface(BadModule)
