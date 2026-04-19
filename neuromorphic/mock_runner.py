"""MockNeuromorphicRunner — pure-numpy LIF simulation from an exported artefact.

Replays the exported INT8 artefact (Plan 6 Task 4) without any vendor SDK.
CI uses this to assert software↔neuromorphic accuracy delta < 2 % without
needing Loihi or Akida hardware.

Plan 6 Task 5.
"""
from __future__ import annotations

import numpy as np


class MockNeuromorphicRunner:
    """Pure-numpy LIF integration + cosine pattern-match decoder."""

    def __init__(self, artefact: dict) -> None:
        self.artefact = artefact
        # Dequantize once at init for speed.
        self.codebook = (
            artefact["codebook_int8"].astype(np.float32)
            * float(artefact["codebook_scale"])
        )
        self.input_proj = (
            artefact["input_proj_int8"].astype(np.float32)
            * float(artefact["input_proj_scale"])
        )
        self.input_proj_bias = artefact["input_proj_bias"].astype(np.float32)
        self.v_thr = float(artefact["v_thr"])
        self.tau_mem = float(artefact["tau_mem"])
        self.n_neurons = int(artefact["n_neurons"])
        self.alphabet_size = int(artefact["alphabet_size"])

    def forward(self, x: np.ndarray, *, dt: float = 1e-3) -> np.ndarray:
        """Single-step forward: float input → argmax code index.

        Args:
            x: [batch, n_neurons] float input current
            dt: integration timestep

        Returns:
            [batch] int array of decoded code indices.
        """
        # Project.
        i_in = x @ self.input_proj.T + self.input_proj_bias  # [B, n_neurons]

        # LIF integration (single tick from rest).
        v_mem = np.zeros_like(i_in)
        v_mem = v_mem + dt / self.tau_mem * (-v_mem + i_in)
        spikes = (v_mem > self.v_thr).astype(np.float32)

        # Cosine similarity vs codebook.
        norms_cb = np.linalg.norm(self.codebook, axis=-1) + 1e-6
        norms_sp = np.linalg.norm(spikes, axis=-1, keepdims=True) + 1e-6
        sims = (spikes @ self.codebook.T) / (norms_cb * norms_sp)

        return sims.argmax(axis=-1)
