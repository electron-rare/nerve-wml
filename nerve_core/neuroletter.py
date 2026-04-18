"""Neuroletter — the elementary message on a nerve.

See spec §4.1. Alphabet size is fixed at 64 (codon-like, ~6 bits).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Role(Enum):
    PREDICTION = 0  # π — descending, continuous (γ phase)
    ERROR      = 1  # ε — ascending, bursty    (θ phase)


class Phase(Enum):
    GAMMA = 0  # 40 Hz — carries π
    THETA = 1  #  6 Hz — carries ε


@dataclass(frozen=True)
class Neuroletter:
    code:      int    # 0..63, on-wire shared integer index
    role:      Role
    phase:     Phase  # in strict mode, derived from role (N-3)
    src:       int
    dst:       int
    timestamp: float
