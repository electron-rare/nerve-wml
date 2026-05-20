"""BioFieldWML — Biological-field-inspired WML substrate, Phase 1 scaffold.

Design input: Bellitto, 2024 — Wake-Sleep Continual Learning (WSCL).
See docs/superpowers/plans/2026-05-19-bio-substrate-wml.md §"Ref B-4".

This module implements Phase 1 of the BioFieldWML plan: the internal
wake/sleep scheduler.  All substantive neural dynamics are explicitly
out of scope for this phase:

  Phase 2 (deferred): Palacios SNN-PC variational message passing.
  Phase 3 (deferred): Tomé STDP triplet + heterosynaptic + inhibitory
                      plasticity.  Scoped to BioFieldWML per OQ-2.
  Phase 4 (deferred): Pignatelli IE plasticity.
  Phase 5 (deferred): Tucker–Friston E/I gain on sleep calls.

Phase 1 scope
-------------
- ``BioFieldWML`` conforms to the WML Protocol (nerve_core.protocols.WML)
  alongside ``MlpWML``, ``LifWML``, and ``TransformerWML``.
- Internal ``_call_count`` tracks how many times ``step()`` has been
  called since construction.
- Internal ``_sleep_every`` (configurable via ``sleep_every`` constructor
  argument, default 4) determines the phase ratio: a call is a *sleep*
  call when ``_call_count % _sleep_every == 0`` (after increment); all
  other calls are *wake* calls.
- Wake calls emit a placeholder Neuroletter on the nerve so the substrate
  is observable by downstream WMLs.  The placeholder content is a
  fixed ``code=0``, ``role=PREDICTION``, ``phase=GAMMA`` — substantive
  dynamics are deferred to Phase 2–5.
- Sleep calls are **silent**: no Neuroletter is sent.  This satisfies
  invariant N-1 (silence is legitimate) and encapsulates the WSCL
  consolidation pass per the Bellitto 2024 design input.

Wake/sleep convention (documented for TDD tests)
-------------------------------------------------
After ``_call_count`` is incremented at the start of ``step()``:
  - sleep  if ``_call_count % _sleep_every == 0``
  - wake   otherwise

With ``sleep_every=4`` and calls numbered 1-8:
  1→wake, 2→wake, 3→wake, 4→sleep, 5→wake, 6→wake, 7→wake, 8→sleep.
"""
from __future__ import annotations

__all__ = ["BioFieldWML"]

from collections.abc import Iterable

import torch
from torch import Tensor, nn

from nerve_core.protocols import Nerve


class BioFieldWML(nn.Module):
    """Wake/sleep-scheduled WML substrate (Bellitto WSCL, Phase 1).

    Parameters
    ----------
    id:
        Stable WML identifier.  Required by W-4.
    sleep_every:
        Number of ``step()`` calls per wake/sleep cycle.  A call is
        classified as a sleep call when ``_call_count % sleep_every == 0``
        (after increment).  Default: 4 (every 4th call is a sleep call).
    alphabet_size:
        Size of the local codebook vocabulary.  Must match the nerve's
        ``ALPHABET_SIZE``.  Default: 64.
    d_hidden:
        Dimensionality of the local codebook embeddings.  Default: 16.
    seed:
        Optional seed for the local ``torch.Generator`` used to
        initialise the codebook.  Does NOT mutate the global RNG (W-4).
    """

    def __init__(
        self,
        id:            int,
        sleep_every:   int  = 4,
        alphabet_size: int  = 64,
        d_hidden:      int  = 16,
        *,
        seed:          int | None = None,
    ) -> None:
        super().__init__()
        self.id            = id
        self.alphabet_size = alphabet_size
        self._sleep_every  = sleep_every
        self._call_count   = 0

        # Local codebook — N-5: each WML owns its vocabulary.
        gen = torch.Generator()
        if seed is not None:
            gen.manual_seed(seed)

        saved_rng = torch.get_rng_state()
        try:
            self.codebook = nn.Parameter(
                torch.randn(alphabet_size, d_hidden, generator=gen) * 0.1
            )
        finally:
            # Restore global RNG so construction does not mutate it.
            torch.set_rng_state(saved_rng)

    # ------------------------------------------------------------------
    # WML Protocol surface
    # ------------------------------------------------------------------

    def step(self, nerve: Nerve, t: float) -> None:
        """One simulation tick.

        1. Increment ``_call_count``.
        2. Determine phase:
           - SLEEP if ``_call_count % _sleep_every == 0``
           - WAKE  otherwise
        3. WAKE: emit a placeholder Neuroletter (PREDICTION / GAMMA,
           code=0) on every routed nerve edge.
           Substantive dynamics are deferred to Phase 2–5.
        4. SLEEP: emit nothing.  Consistent with N-1 (silence is
           legitimate) and encapsulates the WSCL consolidation pass.

        W-1: communicates only via ``nerve.send()`` — never touches
        another WML directly.
        """
        from nerve_core.neuroletter import Neuroletter, Phase, Role

        # Step 1 — advance call counter.
        self._call_count += 1

        # Step 2 — classify phase.
        # N-1: sleep calls are explicitly silent.  The sleep branch
        # will house WSCL consolidation (Bellitto 2024) in Phase 3+.
        is_sleep = (self._call_count % self._sleep_every == 0)

        if is_sleep:
            # SLEEP: silent — no neuroletters emitted (N-1 preserved).
            return

        # WAKE: emit a placeholder PREDICTION neuroletter on every
        # routed outgoing edge.  code=0 is a placeholder; real dynamics
        # (Palacios SNN-PC, Phase 2) will compute the true code.
        for dst in range(nerve.n_wmls):  # type: ignore[attr-defined]
            if dst == self.id:
                continue
            if nerve.routing_weight(self.id, dst) == 1.0:  # W-3 respected
                nerve.send(Neuroletter(
                    code=0,
                    role=Role.PREDICTION,
                    phase=Phase.GAMMA,
                    src=self.id,
                    dst=dst,
                    timestamp=t,
                ))

    def parameters(self, recurse: bool = True) -> Iterable[Tensor]:  # type: ignore[override]
        """W-2: parameters() includes codebook and all internal weights."""
        return super().parameters(recurse=recurse)
