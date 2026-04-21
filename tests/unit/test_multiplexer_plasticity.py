"""Tests for the optional plasticity_schedule / constellation_lock kwargs.

These tests only exercise the plasticity controller; the canonical
multiplexer contract is still pinned by `test_multiplexer.py`. The
default behaviour of `GammaThetaMultiplexer()` must remain bit-identical
to v1.3.0, so existing consumers (bouba_sens v0.3 grids, in particular)
reproduce byte-for-byte.
"""

from __future__ import annotations

import torch

from track_p.multiplexer import GammaThetaConfig, GammaThetaMultiplexer


def test_defaults_preserve_v130_behaviour() -> None:
    """No kwargs = constellation remains a free nn.Parameter."""
    mux = GammaThetaMultiplexer(seed=0)
    assert mux.constellation.requires_grad is True
    assert mux.plasticity_step == 0
