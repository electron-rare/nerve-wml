"""Biological-culture client layer for the BioWML substrate.

A BioCultureClient sends a discrete stimulus (a small list of
alphabet codes, 0..63) to a neural culture and reads back the
resulting multi-channel spiking activity. The substrate
(track_w.bio_wml.BioWML) is provider-agnostic: it talks only to
this Protocol. Three implementations live here:

  - MockBioCultureClient — offline numpy spike simulation with
    realistic latency, jitter, and additive noise (Task 2).
  - CL1Adapter           — Cortical Labs CL1, env-gated (Task 5).
  - FinalSparkAdapter    — FinalSpark Neuroplatform, env-gated.

Plan C (bio substrate). See docs/superpowers/plans/
2026-05-19-bio-substrate-wml.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

ALPHABET_SIZE = 64


class BioApiKeyMissingError(RuntimeError):
    """Raised when a real adapter is built without NERVE_WML_BIO_API_KEY."""


@dataclass(frozen=True)
class StimulusFrame:
    """A stimulus delivered to a culture.

    codes:    the alphabet codes (0..63) being stimulated this tick.
    channels: float32 [n_codes, n_stim_channels] electrode pattern,
              one row per code. Values in [0, 1] = stimulation
              amplitude per channel.
    """

    codes: tuple[int, ...]
    channels: np.ndarray


@dataclass(frozen=True)
class ActivityFrame:
    """Spiking activity read back from a culture.

    spikes:     float32 [n_read_channels, n_bins] spike-count
                raster over a short post-stimulus window.
    latency_ms: wall-clock round-trip latency for this exchange.
    """

    spikes: np.ndarray
    latency_ms: float


@runtime_checkable
class BioCultureClient(Protocol):
    """Provider-agnostic contract for a biological-culture backend."""

    n_stim_channels: int
    n_read_channels: int
    n_bins: int

    def encode_stimulus(self, codes: list[int]) -> StimulusFrame:
        """Map alphabet codes to an electrode stimulation pattern."""
        ...

    def decode_activity(self, frame: ActivityFrame) -> list[int]:
        """Map read-back spiking activity to alphabet codes."""
        ...

    def roundtrip(self, codes: list[int]) -> ActivityFrame:
        """Stimulate with `codes`, read back, return the activity."""
        ...

    def close(self) -> None:
        """Release any underlying connection. Safe to call twice."""
        ...
