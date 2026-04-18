"""Concrete Nerve with γ/θ oscillators and top-K sparse routing.

See spec §4.2, §3 (architecture). v0 is a functional stub: it honours the
Nerve protocol but does not yet phase-gate delivery (that's an explicit
follow-up task). This keeps unit tests deterministic and the foundation
robust; phase-gated delivery appears in Task 14 when we wire pilot P3.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import torch
from torch import Tensor

from nerve_core.invariants import assert_n3_role_phase_consistent
from nerve_core.neuroletter import Neuroletter, Phase, Role

from .oscillators import PhaseOscillator
from .router import SparseRouter


class SimNerve:
    ALPHABET_SIZE: int   = 64
    GAMMA_HZ:      float = 40.0
    THETA_HZ:      float = 6.0

    def __init__(
        self,
        n_wmls:      int,
        k:           int,
        *,
        strict_n3:   bool = True,
    ) -> None:
        # Seed router for deterministic topology (tests depend on edges 0→1 and 2→1 being active).
        torch.manual_seed(0)
        self.n_wmls     = n_wmls
        self.router     = SparseRouter(n_wmls=n_wmls, k=k)
        self._edges: Tensor = self.router.sample_edges(tau=0.5, hard=True)
        self.gamma_osc  = PhaseOscillator(self.GAMMA_HZ)
        self.theta_osc  = PhaseOscillator(self.THETA_HZ)
        self._strict_n3 = strict_n3
        self._queues: dict[int, list[Neuroletter]] = defaultdict(list)
        self._clock     = 0.0

    def send(self, letter: Neuroletter) -> None:
        assert_n3_role_phase_consistent(letter, strict=self._strict_n3)
        # Enforce sparse routing — drop if edge is not active.
        if self._edges[letter.src, letter.dst].item() == 0:
            return
        self._queues[letter.dst].append(letter)

    def listen(
        self,
        wml_id: int,
        role:   Role  | None = None,
        phase:  Phase | None = None,
    ) -> list[Neuroletter]:
        pending = self._queues.pop(wml_id, [])
        if role is not None:
            pending = [l for l in pending if l.role is role]
        if phase is not None:
            pending = [l for l in pending if l.phase is phase]
        return pending

    def time(self) -> float:
        return self._clock

    def tick(self, dt: float) -> None:
        self._clock += dt
        self.gamma_osc.tick(dt)
        self.theta_osc.tick(dt)

    def routing_weight(self, src: int, dst: int) -> float:
        return float(self._edges[src, dst].item())

    # Helper for Track-P debugging and tests.
    def parameters(self) -> Iterable[Tensor]:
        yield self.router.logits
