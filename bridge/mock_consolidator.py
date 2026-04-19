"""MockConsolidator — zero-delta stub matching the kiki_oniric surface.

Used by CI so nerve-wml tests never depend on dream-of-kiki being
installed. Returns an all-zero delta shaped like the real output.

Plan 7 Task 1.
"""
from __future__ import annotations

import numpy as np


class MockConsolidator:
    """Matches the expected kiki_oniric.consolidate signature."""

    @staticmethod
    def consolidate(
        trace: np.ndarray,
        *,
        profile: str = "P_equ",
        n_transducers: int = 12,
        alphabet_size: int = 64,
    ) -> np.ndarray:
        """Return a zero delta — shape [n_transducers, alphabet_size, alphabet_size]."""
        del trace, profile  # unused in mock
        return np.zeros((n_transducers, alphabet_size, alphabet_size), dtype=np.float32)
