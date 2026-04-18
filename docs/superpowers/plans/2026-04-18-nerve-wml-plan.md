# nerve-wml Implementation Plan — Foundation + Track-P

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the shared `nerve_core/` contracts and the Track-P protocol simulator so it passes Gate P (info-theoretic benchmarks in isolation, no WMLs needed).

**Architecture:** Python 3.12 + `uv`. Two layers. `nerve_core/` defines the `Nerve` and `WML` Protocol classes plus `Neuroletter`/`Role`/`Phase` datatypes. `track_p/` implements a concrete `SimNerve` with γ/θ oscillators, a VQ-VAE codebook, per-nerve soft transducers, and top-K Gumbel sparse routing. Info-theoretic tests validate capacity, collision rate, dead-code rate, and π/ε disambiguation.

**Tech Stack:** Python 3.12, `uv`, `torch` (CPU is fine for Track-P), `numpy`, `pytest`, `ruff`, `mypy`.

**Scope boundaries.** This plan goes from empty repo to **Gate P** (protocol simulator validated on toy signals with no WMLs). WML implementations (`track_w/`) and merge training (`bridge/`) are separate plans to be written later.

**Reference spec:** `docs/superpowers/specs/2026-04-18-nerve-wml-design.md` — sections 4 (contracts), 7.2 (curriculum P1-P4), 8.2 (L2 info-theoretic tests), 10-11 (location and YAGNI).

**Follow-up plans (not in scope here):**

- Plan 2 — Track-W (WML lab, MlpWML + LifWML, Gate W polymorphie test).
- Plan 3 — Merge (bridge, swap MockNerve → SimNerve, Gate M).

---

## Phase 0 — Project scaffolding

### Task 1: Initialize Python project and tooling

**Files:**

- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `README.md`
- Create: `CLAUDE.md`
- Create: `.ruff.toml`

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "nerve-wml"
version = "0.1.0"
description = "Substrate-agnostic nerve protocol for inter-WML communication"
readme = "README.md"
requires-python = ">=3.12"
license = { text = "MIT" }
authors = [{ name = "Clément Saillant" }]
dependencies = [
  "torch>=2.3",
  "numpy>=1.26",
]

[project.optional-dependencies]
dev = [
  "pytest>=8.0",
  "pytest-cov>=5.0",
  "ruff>=0.5",
  "mypy>=1.10",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["nerve_core", "track_p"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --tb=short"
```

- [ ] **Step 2: Write `.gitignore`**

```gitignore
__pycache__/
*.pyc
.pytest_cache/
.coverage
.venv/
dist/
build/
*.egg-info/
.mypy_cache/
.ruff_cache/
.superpowers/
```

- [ ] **Step 3: Write minimal `README.md`**

```markdown
# nerve-wml

Substrate-agnostic nerve protocol for inter-WML (World Model Language) communication. Research engine — see `docs/superpowers/specs/2026-04-18-nerve-wml-design.md` for the full design.

## Install

```bash
uv sync --all-extras
```

## Run tests

```bash
uv run pytest
```

## License

MIT (code) + CC-BY-4.0 (docs).
```

- [ ] **Step 4: Write minimal `CLAUDE.md`**

```markdown
# CLAUDE.md — nerve-wml

Research engine for substrate-agnostic inter-WML nerve protocol. Python 3.12 + uv + torch. Design spec at `docs/superpowers/specs/2026-04-18-nerve-wml-design.md`.

## Structure

- `nerve_core/` — shared contracts (Neuroletter, Nerve/WML Protocol, invariants)
- `track_p/` — protocol simulator (SimNerve, VQ, transducer, router)
- `track_w/` — WML lab (MockNerve, MlpWML, LifWML) — future plan
- `bridge/` — merge trainer — future plan
- `tests/` — unit (L1), info-theoretic (L2), integration (L3), golden (L4)

## Commands

```bash
uv sync --all-extras        # install
uv run pytest               # all tests
uv run pytest -m "not slow" # skip long tests
uv run ruff check .
uv run mypy nerve_core track_p
```

## Invariants load-bearing

See `docs/invariants/` and the spec. Never weaken N-1..N-5 or W-1..W-4 without a spec update.
```

- [ ] **Step 5: Write `.ruff.toml`**

```toml
line-length = 100
target-version = "py312"

[lint]
select = ["E", "F", "I", "UP", "B", "N"]
ignore = []
```

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml .gitignore README.md CLAUDE.md .ruff.toml
git commit -m "chore: project scaffolding"
```

---

### Task 2: Create package skeletons and smoke test

**Files:**

- Create: `nerve_core/__init__.py`
- Create: `track_p/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/unit/__init__.py`
- Create: `tests/unit/test_smoke.py`

- [ ] **Step 1: Write package init files (all empty)**

Create the three `__init__.py` files with a single comment:

```python
# nerve_core — shared contracts for nerve-wml.
```

```python
# track_p — protocol simulator.
```

```python
# tests — see docs/superpowers/specs/ for L1/L2/L3/L4 classification.
```

- [ ] **Step 2: Write the smoke test**

`tests/unit/test_smoke.py`:

```python
def test_packages_importable():
    import nerve_core  # noqa: F401
    import track_p     # noqa: F401
```

- [ ] **Step 3: Install deps and run smoke test**

```bash
uv sync --all-extras
uv run pytest tests/unit/test_smoke.py -v
```

Expected: `1 passed`.

- [ ] **Step 4: Commit**

```bash
git add nerve_core/ track_p/ tests/
git commit -m "chore: package skeletons + smoke test"
```

---

## Phase 1 — `nerve_core/` contracts

### Task 3: `Neuroletter` datatype

**Files:**

- Create: `nerve_core/neuroletter.py`
- Create: `tests/unit/test_neuroletter.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_neuroletter.py`:

```python
import pytest
from nerve_core.neuroletter import Neuroletter, Role, Phase


def test_neuroletter_is_frozen():
    n = Neuroletter(code=5, role=Role.PREDICTION, phase=Phase.GAMMA,
                    src=1, dst=2, timestamp=0.5)
    with pytest.raises(Exception):
        n.code = 6  # type: ignore[misc]


def test_neuroletter_equality_and_hashable():
    a = Neuroletter(5, Role.PREDICTION, Phase.GAMMA, 1, 2, 0.5)
    b = Neuroletter(5, Role.PREDICTION, Phase.GAMMA, 1, 2, 0.5)
    assert a == b
    assert hash(a) == hash(b)
    assert {a, b} == {a}


def test_role_and_phase_enums_have_two_values():
    assert {Role.PREDICTION, Role.ERROR} == set(Role)
    assert {Phase.GAMMA, Phase.THETA} == set(Phase)
```

- [ ] **Step 2: Run the test and verify it fails**

```bash
uv run pytest tests/unit/test_neuroletter.py -v
```

Expected: `ModuleNotFoundError: No module named 'nerve_core.neuroletter'`.

- [ ] **Step 3: Write the minimal implementation**

`nerve_core/neuroletter.py`:

```python
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
```

- [ ] **Step 4: Run the test and verify it passes**

```bash
uv run pytest tests/unit/test_neuroletter.py -v
```

Expected: `3 passed`.

- [ ] **Step 5: Commit**

```bash
git add nerve_core/neuroletter.py tests/unit/test_neuroletter.py
git commit -m "feat(core): Neuroletter datatype + Role/Phase enums"
```

---

### Task 4: `Nerve` and `WML` protocol classes

**Files:**

- Create: `nerve_core/protocols.py`
- Create: `tests/unit/test_protocols.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_protocols.py`:

```python
from typing import get_type_hints
from nerve_core.protocols import Nerve, WML


def test_nerve_has_alphabet_size_constant():
    assert Nerve.ALPHABET_SIZE == 64


def test_nerve_has_gamma_theta_constants():
    assert Nerve.GAMMA_HZ == 40.0
    assert Nerve.THETA_HZ == 6.0


def test_nerve_protocol_has_required_methods():
    required = {"send", "listen", "time", "tick", "routing_weight"}
    assert required.issubset(set(dir(Nerve)))


def test_wml_protocol_has_required_attrs():
    hints = get_type_hints(WML)
    assert "id" in hints
    assert "codebook" in hints
    assert "step" in dir(WML)
    assert "parameters" in dir(WML)
```

- [ ] **Step 2: Run the test and verify it fails**

```bash
uv run pytest tests/unit/test_protocols.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Write the implementation**

`nerve_core/protocols.py`:

```python
"""Nerve and WML protocols — the contract that both Track-P and Track-W obey.

See spec §4.2, §4.4.
"""
from __future__ import annotations

from typing import Iterable, Protocol, runtime_checkable

from torch import Tensor

from .neuroletter import Neuroletter, Phase, Role


@runtime_checkable
class Nerve(Protocol):
    """Shared nerve contract.

    Track-P provides SimNerve (real γ/θ oscillators). Track-W starts against
    MockNerve (in-memory queue, no rhythms). Both must satisfy N-1..N-5 from
    the spec.
    """

    ALPHABET_SIZE: int   = 64
    GAMMA_HZ:      float = 40.0
    THETA_HZ:      float = 6.0

    def send(self, letter: Neuroletter) -> None: ...

    def listen(
        self,
        wml_id: int,
        role:   Role  | None = None,
        phase:  Phase | None = None,
    ) -> list[Neuroletter]: ...

    def time(self) -> float: ...

    def tick(self, dt: float) -> None: ...

    def routing_weight(self, src: int, dst: int) -> float: ...


@runtime_checkable
class WML(Protocol):
    """A WML (World Model Language) = a neuron cluster with a local codebook
    and a step() that listens, computes internally, and emits.
    """

    id:       int
    codebook: Tensor

    def step(self, nerve: Nerve, t: float) -> None: ...

    def parameters(self) -> Iterable[Tensor]: ...
```

- [ ] **Step 4: Run the test and verify it passes**

```bash
uv run pytest tests/unit/test_protocols.py -v
```

Expected: `4 passed`.

- [ ] **Step 5: Commit**

```bash
git add nerve_core/protocols.py tests/unit/test_protocols.py
git commit -m "feat(core): Nerve + WML protocol classes"
```

---

### Task 5: Invariant runtime guards

**Files:**

- Create: `nerve_core/invariants.py`
- Create: `tests/unit/test_invariants.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_invariants.py`:

```python
import pytest
from nerve_core.invariants import (
    assert_n1_silence_legal,
    assert_n3_role_phase_consistent,
    assert_n4_routing_weight_valid,
)
from nerve_core.neuroletter import Neuroletter, Phase, Role


def test_n1_empty_listen_is_legal():
    assert_n1_silence_legal([])  # does not raise


def test_n3_prediction_must_be_gamma_in_strict_mode():
    ok = Neuroletter(5, Role.PREDICTION, Phase.GAMMA, 1, 2, 0.0)
    assert_n3_role_phase_consistent(ok, strict=True)

    bad = Neuroletter(5, Role.PREDICTION, Phase.THETA, 1, 2, 0.0)
    with pytest.raises(AssertionError, match="N-3"):
        assert_n3_role_phase_consistent(bad, strict=True)


def test_n3_error_must_be_theta_in_strict_mode():
    bad = Neuroletter(5, Role.ERROR, Phase.GAMMA, 1, 2, 0.0)
    with pytest.raises(AssertionError, match="N-3"):
        assert_n3_role_phase_consistent(bad, strict=True)


def test_n4_routing_weight_range():
    assert_n4_routing_weight_valid(0.0, pruned=True)
    assert_n4_routing_weight_valid(1.0, pruned=True)
    assert_n4_routing_weight_valid(0.42, pruned=False)  # continuous during training
    with pytest.raises(AssertionError, match="N-4"):
        assert_n4_routing_weight_valid(0.5, pruned=True)  # must be {0, 1} once pruned
```

- [ ] **Step 2: Run the test and verify it fails**

```bash
uv run pytest tests/unit/test_invariants.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Write the implementation**

`nerve_core/invariants.py`:

```python
"""Runtime guards for the N-1..N-5 and W-1..W-4 invariants in the spec §4.5.

These are assert_*() helpers — cheap in prod, strict in tests.
"""
from __future__ import annotations

from .neuroletter import Neuroletter, Phase, Role


def assert_n1_silence_legal(inbound: list[Neuroletter]) -> None:
    """N-1: listen() returning [] is always valid (silence is information)."""
    # Nothing to assert — the contract is that empty is fine.
    return


def assert_n3_role_phase_consistent(
    letter: Neuroletter,
    *,
    strict: bool = True,
) -> None:
    """N-3 strict mode: PREDICTION↔GAMMA and ERROR↔THETA."""
    if not strict:
        return
    expected_phase = Phase.GAMMA if letter.role is Role.PREDICTION else Phase.THETA
    assert letter.phase is expected_phase, (
        f"N-3 violated: role={letter.role.name} with phase={letter.phase.name} "
        f"(expected {expected_phase.name} in strict mode)"
    )


def assert_n4_routing_weight_valid(weight: float, *, pruned: bool) -> None:
    """N-4: post-pruning weights are {0, 1}; pre-pruning they are continuous in [0, 1]."""
    if pruned:
        assert weight in (0.0, 1.0), (
            f"N-4 violated: post-pruning weight must be 0 or 1, got {weight}"
        )
    else:
        assert 0.0 <= weight <= 1.0, (
            f"N-4 violated: pre-pruning weight must be in [0, 1], got {weight}"
        )
```

- [ ] **Step 4: Run the test and verify it passes**

```bash
uv run pytest tests/unit/test_invariants.py -v
```

Expected: `4 passed`.

- [ ] **Step 5: Commit**

```bash
git add nerve_core/invariants.py tests/unit/test_invariants.py
git commit -m "feat(core): N-1/N-3/N-4 runtime guards"
```

---

## Phase 2 — Track-P components

### Task 6: γ / θ oscillators

**Files:**

- Create: `track_p/oscillators.py`
- Create: `tests/unit/test_oscillators.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_oscillators.py`:

```python
import math

from track_p.oscillators import PhaseOscillator


def test_gamma_period_is_25ms():
    osc = PhaseOscillator(freq_hz=40.0)
    assert math.isclose(osc.period_s, 0.025, abs_tol=1e-6)


def test_phase_advances_with_tick():
    osc = PhaseOscillator(freq_hz=40.0)
    assert osc.phase == 0.0
    osc.tick(dt=0.0125)   # half period
    assert math.isclose(osc.phase, 0.5, abs_tol=1e-6)
    osc.tick(dt=0.0125)   # full period → wraps back to 0
    assert math.isclose(osc.phase, 0.0, abs_tol=1e-6)


def test_is_active_window():
    """A PhaseOscillator fires in the first half of each cycle."""
    osc = PhaseOscillator(freq_hz=40.0)
    assert osc.is_active()
    osc.tick(dt=0.020)  # into second half (phase > 0.5)
    assert not osc.is_active()
```

- [ ] **Step 2: Run the test and verify it fails**

```bash
uv run pytest tests/unit/test_oscillators.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Write the implementation**

`track_p/oscillators.py`:

```python
"""γ (40 Hz) and θ (6 Hz) phase oscillators for the SimNerve.

See spec §7.5 (rhythmic multiplexing) and §3 (architecture).
"""
from __future__ import annotations


class PhaseOscillator:
    """A unit-period phase clock. phase in [0, 1). Active in the first half
    of each cycle (phase < 0.5), inactive otherwise. This lets SimNerve deliver
    neuroletters only during their role's phase window."""

    def __init__(self, freq_hz: float) -> None:
        assert freq_hz > 0
        self.freq_hz = freq_hz
        self.phase   = 0.0

    @property
    def period_s(self) -> float:
        return 1.0 / self.freq_hz

    def tick(self, dt: float) -> None:
        self.phase = (self.phase + dt / self.period_s) % 1.0

    def is_active(self) -> bool:
        return self.phase < 0.5
```

- [ ] **Step 4: Run the test and verify it passes**

```bash
uv run pytest tests/unit/test_oscillators.py -v
```

Expected: `3 passed`.

- [ ] **Step 5: Commit**

```bash
git add track_p/oscillators.py tests/unit/test_oscillators.py
git commit -m "feat(track-p): γ/θ phase oscillators"
```

---

### Task 7: VQ-VAE codebook (Track-P, no WMLs yet)

**Files:**

- Create: `track_p/vq_codebook.py`
- Create: `tests/unit/test_vq_codebook.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_vq_codebook.py`:

```python
import torch

from track_p.vq_codebook import VQCodebook


def test_codebook_has_correct_shape():
    cb = VQCodebook(size=64, dim=128)
    assert cb.embeddings.shape == (64, 128)


def test_quantize_returns_valid_indices():
    cb = VQCodebook(size=64, dim=128)
    z = torch.randn(8, 128)
    indices, quantized, loss = cb.quantize(z)
    assert indices.shape == (8,)
    assert (indices >= 0).all() and (indices < 64).all()
    assert quantized.shape == z.shape
    assert loss.dim() == 0  # scalar loss


def test_usage_counter_increments_on_quantize():
    cb = VQCodebook(size=64, dim=32, ema=False)
    z = torch.randn(100, 32)
    cb.quantize(z)
    total_usage = cb.usage_counter.sum().item()
    assert total_usage == 100


def test_ema_update_does_not_require_gradient_through_embeddings():
    cb = VQCodebook(size=16, dim=8, ema=True)
    z = torch.randn(32, 8, requires_grad=True)
    _, quantized, _ = cb.quantize(z)
    # Straight-through: gradient must flow back to z even though embeddings
    # are not differentiable leaves under EMA update.
    loss = quantized.sum()
    loss.backward()
    assert z.grad is not None
```

- [ ] **Step 2: Run the test and verify it fails**

```bash
uv run pytest tests/unit/test_vq_codebook.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Write the implementation**

`track_p/vq_codebook.py`:

```python
"""VQ-VAE codebook with EMA update and commitment loss.

See spec §7 (training) and van den Oord et al. 2017. EMA path avoids dead
codes under dense-signal training.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn


class VQCodebook(nn.Module):
    """Codebook of `size` embeddings of `dim` dimensions.

    - quantize(z): returns (indices, quantized_z, commitment_loss).
    - straight-through gradient: grad flows from quantized to z unchanged.
    - ema=True: embeddings are updated by EMA of assigned vectors (no gradient).
    - ema=False: embeddings are a regular nn.Parameter (baseline).
    """

    def __init__(
        self,
        size: int,
        dim:  int,
        *,
        commitment_beta: float = 0.25,
        ema:             bool  = True,
        decay:           float = 0.99,
    ) -> None:
        super().__init__()
        self.size            = size
        self.dim             = dim
        self.commitment_beta = commitment_beta
        self.ema             = ema
        self.decay           = decay

        init = torch.randn(size, dim) * 0.1

        if ema:
            self.register_buffer("embeddings", init)
            self.register_buffer("ema_cluster_size", torch.zeros(size))
            self.register_buffer("ema_embed_sum",    init.clone())
        else:
            self.embeddings = nn.Parameter(init)  # type: ignore[assignment]

        self.register_buffer("usage_counter", torch.zeros(size, dtype=torch.long))

    def quantize(self, z: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        # z: [B, dim]. Compute squared distance to every codebook vector.
        dist = torch.cdist(z, self.embeddings)               # [B, size]
        indices   = dist.argmin(dim=-1)                      # [B]
        quantized = self.embeddings[indices]                 # [B, dim]

        # Usage tracking
        for i in indices.tolist():
            self.usage_counter[i] += 1

        # Commitment loss (Oord 2017 eq. 3).
        commit_loss = self.commitment_beta * ((z - quantized.detach()) ** 2).mean()
        codebook_loss = ((quantized - z.detach()) ** 2).mean()
        loss = commit_loss + codebook_loss

        # EMA update of embeddings (no gradient path).
        if self.ema and self.training:
            with torch.no_grad():
                onehot = torch.zeros(z.shape[0], self.size, device=z.device)
                onehot.scatter_(1, indices.unsqueeze(1), 1)

                self.ema_cluster_size.mul_(self.decay).add_(
                    onehot.sum(0), alpha=1 - self.decay
                )
                self.ema_embed_sum.mul_(self.decay).add_(
                    onehot.T @ z, alpha=1 - self.decay
                )
                n = self.ema_cluster_size.sum()
                cluster = (self.ema_cluster_size + 1e-5) / (n + self.size * 1e-5) * n
                self.embeddings = self.ema_embed_sum / cluster.unsqueeze(1)

        # Straight-through estimator.
        quantized = z + (quantized - z).detach()
        return indices, quantized, loss
```

- [ ] **Step 4: Run the test and verify it passes**

```bash
uv run pytest tests/unit/test_vq_codebook.py -v
```

Expected: `4 passed`.

- [ ] **Step 5: Commit**

```bash
git add track_p/vq_codebook.py tests/unit/test_vq_codebook.py
git commit -m "feat(track-p): VQ codebook with EMA update"
```

---

### Task 8: Transducer (per-nerve soft 64×64 matrix)

**Files:**

- Create: `track_p/transducer.py`
- Create: `tests/unit/test_transducer.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_transducer.py`:

```python
import torch

from track_p.transducer import Transducer


def test_transducer_matrix_shape():
    t = Transducer(alphabet_size=64)
    assert t.logits.shape == (64, 64)


def test_forward_returns_valid_code_index():
    t = Transducer(alphabet_size=64)
    src_code = torch.tensor([5, 17, 42])
    dst_code = t.forward(src_code, hard=True)
    assert dst_code.shape == src_code.shape
    assert (dst_code >= 0).all() and (dst_code < 64).all()


def test_entropy_regularizer_nonzero_for_uniform_start():
    t = Transducer(alphabet_size=64)
    ent = t.entropy()
    # Uniform-ish init → high entropy per row → near log(64)
    assert ent.item() > 3.0  # log(64) ≈ 4.16
```

- [ ] **Step 2: Run the test and verify it fails**

```bash
uv run pytest tests/unit/test_transducer.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Write the implementation**

`track_p/transducer.py`:

```python
"""Per-nerve soft transducer mapping src local code → dst local code.

See spec §4.3. Each row of the 64×64 logits matrix is a distribution over
possible target codes. Gumbel-softmax during training, argmax at inference.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class Transducer(nn.Module):
    def __init__(self, alphabet_size: int = 64, init_scale: float = 0.1) -> None:
        super().__init__()
        self.alphabet_size = alphabet_size
        # Near-uniform init to avoid premature collapse.
        self.logits = nn.Parameter(torch.randn(alphabet_size, alphabet_size) * init_scale)

    def forward(self, src_code: Tensor, *, hard: bool = True, tau: float = 1.0) -> Tensor:
        """src_code: [B] long. Returns dst_code: [B] long."""
        row_logits = self.logits[src_code]                     # [B, alphabet_size]
        y = F.gumbel_softmax(row_logits, tau=tau, hard=hard)   # [B, alphabet_size]
        return y.argmax(dim=-1)

    def entropy(self) -> Tensor:
        """Row-wise Shannon entropy of the transducer distribution.

        Higher = more uniform (used as a regularizer to avoid collapse to identity).
        Returns the mean entropy across all rows.
        """
        p = F.softmax(self.logits, dim=-1)                     # [size, size]
        ent_per_row = -(p * (p + 1e-9).log()).sum(dim=-1)
        return ent_per_row.mean()
```

- [ ] **Step 4: Run the test and verify it passes**

```bash
uv run pytest tests/unit/test_transducer.py -v
```

Expected: `3 passed`.

- [ ] **Step 5: Commit**

```bash
git add track_p/transducer.py tests/unit/test_transducer.py
git commit -m "feat(track-p): per-nerve soft transducer"
```

---

### Task 9: Top-K Gumbel sparse router

**Files:**

- Create: `track_p/router.py`
- Create: `tests/unit/test_router.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_router.py`:

```python
import torch

from track_p.router import SparseRouter


def test_router_edge_count_equals_k_per_wml():
    r = SparseRouter(n_wmls=4, k=2)
    edges = r.sample_edges(tau=0.5, hard=True)
    # edges[i, j] = 1 if i → j active. Each row must have exactly k ones.
    assert edges.shape == (4, 4)
    assert (edges.sum(dim=-1) == 2).all()


def test_router_no_self_loops():
    r = SparseRouter(n_wmls=4, k=2)
    edges = r.sample_edges(tau=0.5, hard=True)
    assert (edges.diagonal() == 0).all()


def test_routing_weight_lookup():
    r = SparseRouter(n_wmls=4, k=2)
    edges = r.sample_edges(tau=0.5, hard=True)
    for i in range(4):
        for j in range(4):
            assert r.routing_weight(i, j, edges) == float(edges[i, j])
```

- [ ] **Step 2: Run the test and verify it fails**

```bash
uv run pytest tests/unit/test_router.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Write the implementation**

`track_p/router.py`:

```python
"""Sparse top-K Gumbel routing between WMLs.

See spec §4.2 (routing_weight) and §4.5 (N-4). During training, τ is annealed
from ~1 to ~0.1. At inference, hard top-K is applied.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class SparseRouter(nn.Module):
    """Returns a sparse {0, 1} edge matrix of shape [N, N] where each row has
    exactly K ones (and no self-loop)."""

    def __init__(self, n_wmls: int, k: int) -> None:
        super().__init__()
        assert 1 <= k < n_wmls
        self.n_wmls = n_wmls
        self.k      = k
        # Learnable logits per directed pair. Self-loops masked at sample time.
        self.logits = nn.Parameter(torch.randn(n_wmls, n_wmls) * 0.1)

    def sample_edges(self, *, tau: float = 1.0, hard: bool = True) -> Tensor:
        # Mask self-loops with -inf so softmax never selects them.
        mask       = torch.eye(self.n_wmls, dtype=torch.bool, device=self.logits.device)
        masked_log = self.logits.masked_fill(mask, float("-inf"))

        # Per-row Gumbel; then keep top-K per row.
        noise    = -torch.log(-torch.log(torch.rand_like(masked_log) + 1e-9) + 1e-9)
        noisy    = (masked_log + noise) / tau

        topk_idx = noisy.topk(self.k, dim=-1).indices            # [N, K]
        edges    = torch.zeros_like(masked_log)
        edges.scatter_(1, topk_idx, 1.0)

        if hard:
            return edges
        # Soft path: weight by softmax over the top-K logits (seldom used).
        soft_weights = F.softmax(noisy, dim=-1)
        return soft_weights * edges

    def routing_weight(self, src: int, dst: int, edges: Tensor) -> float:
        return float(edges[src, dst].item())
```

- [ ] **Step 4: Run the test and verify it passes**

```bash
uv run pytest tests/unit/test_router.py -v
```

Expected: `3 passed`.

- [ ] **Step 5: Commit**

```bash
git add track_p/router.py tests/unit/test_router.py
git commit -m "feat(track-p): top-K Gumbel sparse router"
```

---

### Task 10: `SimNerve` — concrete `Nerve` implementation

**Files:**

- Create: `track_p/sim_nerve.py`
- Create: `tests/unit/test_sim_nerve.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_sim_nerve.py`:

```python
import torch

from nerve_core.neuroletter import Neuroletter, Phase, Role
from track_p.sim_nerve import SimNerve


def _letter(src: int, dst: int, role: Role, phase: Phase, t: float = 0.0) -> Neuroletter:
    return Neuroletter(code=3, role=role, phase=phase, src=src, dst=dst, timestamp=t)


def test_sim_nerve_round_trip():
    nerve = SimNerve(n_wmls=4, k=2)
    nerve.send(_letter(0, 1, Role.PREDICTION, Phase.GAMMA))
    # Receiver sees the message on the next listen, regardless of oscillator
    # phase (no phase gating in v0 — phase filtering is additive in a later task).
    received = nerve.listen(wml_id=1)
    assert len(received) == 1
    assert received[0].code == 3


def test_sim_nerve_filter_by_role():
    nerve = SimNerve(n_wmls=4, k=2)
    nerve.send(_letter(0, 1, Role.PREDICTION, Phase.GAMMA, t=0.0))
    nerve.send(_letter(2, 1, Role.ERROR,      Phase.THETA, t=0.0))
    assert len(nerve.listen(wml_id=1, role=Role.PREDICTION)) == 1
    # listen() clears the queue, so we need a fresh fire for the ε test.
    nerve.send(_letter(2, 1, Role.ERROR, Phase.THETA, t=0.0))
    assert len(nerve.listen(wml_id=1, role=Role.ERROR)) == 1


def test_sim_nerve_tick_advances_time():
    nerve = SimNerve(n_wmls=4, k=2)
    t0 = nerve.time()
    nerve.tick(dt=0.010)
    assert nerve.time() > t0


def test_sim_nerve_routing_weight_edge_count():
    nerve = SimNerve(n_wmls=4, k=2)
    active_edges = sum(
        1
        for i in range(4)
        for j in range(4)
        if nerve.routing_weight(i, j) == 1.0
    )
    assert active_edges == 4 * 2  # K edges per row, 4 rows
```

- [ ] **Step 2: Run the test and verify it fails**

```bash
uv run pytest tests/unit/test_sim_nerve.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Write the implementation**

`track_p/sim_nerve.py`:

```python
"""Concrete Nerve with γ/θ oscillators and top-K sparse routing.

See spec §4.2, §3 (architecture). v0 is a functional stub: it honours the
Nerve protocol but does not yet phase-gate delivery (that's an explicit
follow-up task). This keeps unit tests deterministic and the foundation
robust; phase-gated delivery appears in Task 14 when we wire pilot P3.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable

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
```

- [ ] **Step 4: Run the test and verify it passes**

```bash
uv run pytest tests/unit/test_sim_nerve.py -v
```

Expected: `4 passed`.

- [ ] **Step 5: Commit**

```bash
git add track_p/sim_nerve.py tests/unit/test_sim_nerve.py
git commit -m "feat(track-p): SimNerve v0 — routing + role/phase filters"
```

---

## Phase 3 — Info-theoretic tests (L2)

### Task 11: Capacity, collision, dead-code, disambiguation metrics

**Files:**

- Create: `track_p/info_theoretic.py`
- Create: `tests/info_theoretic/__init__.py`
- Create: `tests/info_theoretic/test_metrics.py`

- [ ] **Step 1: Write the failing test**

`tests/info_theoretic/test_metrics.py`:

```python
import math
import torch

from track_p.info_theoretic import (
    empirical_capacity_bps,
    dead_code_fraction,
    kl_divergence,
)
from track_p.vq_codebook import VQCodebook


def test_capacity_lower_bound_on_uniform_stream():
    # 46 Hz × 6 bits = 276 bits/s when uniform
    capacity = empirical_capacity_bps(
        code_rate_hz=46.0,
        code_histogram=torch.ones(64) / 64,
    )
    assert capacity > 200
    assert capacity <= 46.0 * math.log2(64) + 1e-6


def test_dead_code_fraction_detects_unused_codes():
    cb = VQCodebook(size=64, dim=32, ema=False)
    # Send only the first 10 codes (hand-craft by directly bumping usage).
    cb.usage_counter[:10] = 100
    cb.usage_counter[10:] = 0
    frac = dead_code_fraction(cb)
    assert math.isclose(frac, 54 / 64, abs_tol=1e-6)


def test_kl_divergence_between_distinct_distributions():
    p = torch.tensor([0.9, 0.1])
    q = torch.tensor([0.1, 0.9])
    kl = kl_divergence(p, q)
    assert kl.item() > 1.0
```

- [ ] **Step 2: Run the test and verify it fails**

```bash
uv run pytest tests/info_theoretic/test_metrics.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Write the implementation**

`track_p/info_theoretic.py`:

```python
"""L2 info-theoretic metrics for Gate P validation.

See spec §8.2.
"""
from __future__ import annotations

import torch
from torch import Tensor

from .vq_codebook import VQCodebook


def empirical_capacity_bps(code_rate_hz: float, code_histogram: Tensor) -> float:
    """Shannon entropy of the code distribution × code rate = bits/s throughput."""
    p = code_histogram / (code_histogram.sum() + 1e-9)
    ent_bits = -(p * (p + 1e-9).log2()).sum().item()
    return code_rate_hz * ent_bits


def dead_code_fraction(cb: VQCodebook) -> float:
    """Fraction of codes never assigned during usage tracking."""
    return (cb.usage_counter == 0).float().mean().item()


def kl_divergence(p: Tensor, q: Tensor) -> Tensor:
    """KL(p ‖ q) in bits — used for π/ε disambiguation."""
    p = p / (p.sum() + 1e-9)
    q = q / (q.sum() + 1e-9)
    return (p * ((p + 1e-9).log2() - (q + 1e-9).log2())).sum()
```

Also create an empty `tests/info_theoretic/__init__.py`:

```python
# L2 info-theoretic tests — see spec §8.2.
```

- [ ] **Step 4: Run the test and verify it passes**

```bash
uv run pytest tests/info_theoretic/ -v
```

Expected: `3 passed`.

- [ ] **Step 5: Commit**

```bash
git add track_p/info_theoretic.py tests/info_theoretic/
git commit -m "feat(track-p): L2 info-theoretic metrics"
```

---

## Phase 4 — Gate P curriculum pilots

### Task 12: P1 — VQ codebook on toy signals

**Files:**

- Create: `scripts/__init__.py`
- Create: `scripts/track_p_pilot.py`
- Create: `tests/integration/__init__.py`
- Create: `tests/integration/test_gate_p1.py`

- [ ] **Step 1: Write the failing test**

`tests/integration/test_gate_p1.py`:

```python
import torch

from scripts.track_p_pilot import run_p1
from track_p.info_theoretic import dead_code_fraction


def test_p1_codebook_has_low_dead_code_fraction():
    torch.manual_seed(0)
    cb = run_p1(steps=2000)
    # Gate P1 criterion: dead codes < 10 %
    assert dead_code_fraction(cb) < 0.10


def test_p1_codebook_perplexity_meets_target():
    torch.manual_seed(0)
    cb = run_p1(steps=2000)
    # Perplexity = 2^entropy on the normalized usage. Target ≥ 32 / 64.
    counts = cb.usage_counter.float()
    p = counts / (counts.sum() + 1e-9)
    ent_bits = -(p * (p + 1e-9).log2()).sum().item()
    perplexity = 2 ** ent_bits
    assert perplexity >= 32
```

- [ ] **Step 2: Run the test and verify it fails**

```bash
uv run pytest tests/integration/test_gate_p1.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Write the pilot driver**

`scripts/track_p_pilot.py`:

```python
"""Track-P pilot scripts: P1..P4 curriculum drivers.

Each `run_pN(...)` returns the artefact to be validated at gate P (codebook,
transducer, router, SimNerve). Scripts are idempotent given a fixed seed.
"""
from __future__ import annotations

import torch
from torch.optim import Adam

from track_p.vq_codebook import VQCodebook


def run_p1(steps: int = 2000, dim: int = 32, size: int = 64) -> VQCodebook:
    """P1 — train VQ codebook on a diverse toy signal (mixture of Gaussians).

    The dataset has `size` clusters by construction so a well-trained VQ
    should assign each cluster to a distinct codebook entry.
    """
    torch.manual_seed(0)
    cb = VQCodebook(size=size, dim=dim, ema=True)
    opt = Adam([p for p in cb.parameters() if p.requires_grad], lr=1e-3)

    centers = torch.randn(size, dim) * 3

    for _ in range(steps):
        cb.train()
        cluster_ids = torch.randint(0, size, (256,))
        z = centers[cluster_ids] + torch.randn(256, dim) * 0.2

        _, _, loss = cb.quantize(z)
        if loss.requires_grad:
            opt.zero_grad()
            loss.backward()
            opt.step()

    return cb
```

Also create `scripts/__init__.py` and `tests/integration/__init__.py`:

```python
# Track-P / Track-W / Merge pilots — one entry point per gate step.
```

```python
# L3 integration tests — gate-level end-to-end checks.
```

- [ ] **Step 4: Run the test and verify it passes**

```bash
uv run pytest tests/integration/test_gate_p1.py -v
```

Expected: `2 passed`. (Training takes ~10 s.)

- [ ] **Step 5: Commit**

```bash
git add scripts/ tests/integration/
git commit -m "feat(track-p): P1 pilot — VQ codebook on toy MOG"
```

---

### Task 13: P2 — transducer preserves identity after training on paired signals

**Files:**

- Modify: `scripts/track_p_pilot.py`
- Create: `tests/integration/test_gate_p2.py`

- [ ] **Step 1: Write the failing test**

`tests/integration/test_gate_p2.py`:

```python
import torch

from scripts.track_p_pilot import run_p2
from track_p.info_theoretic import kl_divergence


def test_p2_transducer_is_not_uniform_after_training():
    torch.manual_seed(0)
    transducer, _ = run_p2(steps=2000)

    # Check one row: post-training distribution should be far from uniform.
    import torch.nn.functional as F
    row = F.softmax(transducer.logits[7], dim=-1)
    uniform = torch.full_like(row, 1.0 / 64)
    kl = kl_divergence(row, uniform)
    assert kl.item() > 1.0


def test_p2_transducer_retention_above_95pct():
    """Retention: of all codes sent through the transducer, ≥ 95 %
    are decoded back to the expected output code in a known src→dst pairing."""
    torch.manual_seed(0)
    _, retention = run_p2(steps=2000)
    assert retention > 0.95
```

- [ ] **Step 2: Run the test and verify it fails**

```bash
uv run pytest tests/integration/test_gate_p2.py -v
```

Expected: `ImportError: cannot import name 'run_p2'`.

- [ ] **Step 3: Extend the pilot driver**

Append to `scripts/track_p_pilot.py`:

```python
from track_p.transducer import Transducer


def run_p2(steps: int = 2000, alphabet_size: int = 64) -> tuple[Transducer, float]:
    """P2 — train a transducer so that a known src→dst code permutation is learned.

    We construct a ground-truth permutation π* and train the transducer to
    reproduce it. Returns (trained_transducer, retention_fraction).
    """
    torch.manual_seed(0)
    transducer = Transducer(alphabet_size=alphabet_size)
    opt = Adam(transducer.parameters(), lr=1e-2)

    target_perm = torch.randperm(alphabet_size)

    for _ in range(steps):
        src_codes = torch.randint(0, alphabet_size, (256,))
        expected  = target_perm[src_codes]

        row_logits = transducer.logits[src_codes]
        loss = torch.nn.functional.cross_entropy(row_logits, expected)

        opt.zero_grad()
        loss.backward()
        opt.step()

    # Retention = fraction of src codes mapped correctly under argmax.
    with torch.no_grad():
        pred = transducer.logits.argmax(dim=-1)
        retention = (pred == target_perm).float().mean().item()

    return transducer, retention
```

- [ ] **Step 4: Run the test and verify it passes**

```bash
uv run pytest tests/integration/test_gate_p2.py -v
```

Expected: `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add scripts/track_p_pilot.py tests/integration/test_gate_p2.py
git commit -m "feat(track-p): P2 pilot — transducer permutation learning"
```

---

### Task 14: P3 — phase-gated delivery through γ/θ multiplexing

**Files:**

- Modify: `track_p/sim_nerve.py` (add phase gating on delivery)
- Modify: `scripts/track_p_pilot.py`
- Create: `tests/integration/test_gate_p3.py`

- [ ] **Step 1: Write the failing test**

`tests/integration/test_gate_p3.py`:

```python
import torch

from scripts.track_p_pilot import run_p3


def test_p3_no_phase_collisions():
    """γ letters and θ letters never share a delivery timestamp."""
    torch.manual_seed(0)
    collision_count = run_p3(n_cycles=200)
    assert collision_count == 0
```

- [ ] **Step 2: Run the test and verify it fails**

```bash
uv run pytest tests/integration/test_gate_p3.py -v
```

Expected: `ImportError: cannot import name 'run_p3'`.

- [ ] **Step 3: Add phase gating in `SimNerve.listen()` and implement `run_p3`**

Update `track_p/sim_nerve.py` — replace the `listen` body:

```python
    def listen(
        self,
        wml_id: int,
        role:   Role  | None = None,
        phase:  Phase | None = None,
    ) -> list[Neuroletter]:
        """Phase-gated delivery — only letters whose phase is currently active
        in the corresponding oscillator are returned. Letters that arrive in
        the inactive half of their phase are held until the next active window.
        """
        pending = self._queues.get(wml_id, [])

        def phase_active(p: Phase) -> bool:
            return self.gamma_osc.is_active() if p is Phase.GAMMA else self.theta_osc.is_active()

        delivered = [l for l in pending if phase_active(l.phase)]
        held      = [l for l in pending if not phase_active(l.phase)]
        self._queues[wml_id] = held

        if role is not None:
            delivered = [l for l in delivered if l.role is role]
        if phase is not None:
            delivered = [l for l in delivered if l.phase is phase]

        return delivered
```

Append to `scripts/track_p_pilot.py`:

```python
from nerve_core.neuroletter import Neuroletter, Phase, Role
from track_p.sim_nerve import SimNerve


def run_p3(n_cycles: int = 200, dt: float = 1e-3) -> int:
    """P3 — run SimNerve for n_cycles γ-periods; count phase collisions.

    A collision is two letters delivered to the same wml at the same tick
    under different phases, which would break the multiplexing invariant.
    """
    nerve = SimNerve(n_wmls=4, k=2)
    collision_count = 0

    for _ in range(n_cycles):
        # Emit one π (γ) and one ε (θ) to wml 1 in this cycle.
        nerve.send(Neuroletter(3, Role.PREDICTION, Phase.GAMMA, 0, 1, nerve.time()))
        nerve.send(Neuroletter(7, Role.ERROR,      Phase.THETA, 2, 1, nerve.time()))
        nerve.tick(dt)

        delivered = nerve.listen(wml_id=1)
        phases_delivered = {l.phase for l in delivered}
        if Phase.GAMMA in phases_delivered and Phase.THETA in phases_delivered:
            collision_count += 1

    return collision_count
```

- [ ] **Step 4: Run the test and verify it passes**

```bash
uv run pytest tests/integration/test_gate_p3.py -v
uv run pytest tests/unit/test_sim_nerve.py -v  # ensure no regression
```

Expected for both: `passed`.

Note: the existing `test_sim_nerve_round_trip` may now need a tick to align the oscillator; if it fails, add `nerve.tick(0.001)` before the listen in that test.

- [ ] **Step 5: Commit**

```bash
git add track_p/sim_nerve.py scripts/track_p_pilot.py tests/integration/test_gate_p3.py
git commit -m "feat(track-p): P3 — γ/θ phase-gated delivery"
```

---

### Task 15: P4 — sparse routing topology stays connected

**Files:**

- Modify: `scripts/track_p_pilot.py`
- Create: `tests/integration/test_gate_p4.py`

- [ ] **Step 1: Write the failing test**

`tests/integration/test_gate_p4.py`:

```python
import torch

from scripts.track_p_pilot import run_p4


def test_p4_topology_is_connected():
    """After sparse K-active sampling, every WML must be reachable from every other."""
    torch.manual_seed(0)
    connected, k_per_wml = run_p4(n_wmls=4, k=2)
    assert connected is True
    assert (k_per_wml == 2).all()
```

- [ ] **Step 2: Run the test and verify it fails**

```bash
uv run pytest tests/integration/test_gate_p4.py -v
```

Expected: `ImportError: cannot import name 'run_p4'`.

- [ ] **Step 3: Extend the pilot driver**

Append to `scripts/track_p_pilot.py`:

```python
def run_p4(n_wmls: int = 4, k: int = 2) -> tuple[bool, torch.Tensor]:
    """P4 — sample a sparse topology and verify K-active per WML + graph connectivity
    via a simple BFS from node 0.
    """
    from track_p.router import SparseRouter

    router = SparseRouter(n_wmls=n_wmls, k=k)
    edges = router.sample_edges(tau=0.5, hard=True)

    # K-active per row invariant (N-4).
    k_per_wml = edges.sum(dim=-1)

    # Undirected connectivity via BFS (nerve is bidirectional at the physical layer).
    adjacency = ((edges + edges.T) > 0)
    visited = {0}
    frontier = [0]
    while frontier:
        node = frontier.pop()
        for nbr in range(n_wmls):
            if adjacency[node, nbr] and nbr not in visited:
                visited.add(nbr)
                frontier.append(nbr)
    connected = len(visited) == n_wmls

    return connected, k_per_wml
```

- [ ] **Step 4: Run the test and verify it passes**

```bash
uv run pytest tests/integration/test_gate_p4.py -v
```

Expected: `1 passed`.

- [ ] **Step 5: Commit**

```bash
git add scripts/track_p_pilot.py tests/integration/test_gate_p4.py
git commit -m "feat(track-p): P4 — sparse topology connectivity"
```

---

## Phase 5 — Gate P validation

### Task 16: Combined Gate P runner and report

**Files:**

- Modify: `scripts/track_p_pilot.py` (add `run_gate_p`)
- Create: `tests/integration/test_gate_p.py`

- [ ] **Step 1: Write the failing test**

`tests/integration/test_gate_p.py`:

```python
import torch

from scripts.track_p_pilot import run_gate_p


def test_gate_p_all_criteria_pass():
    torch.manual_seed(0)
    report = run_gate_p()
    assert report["p1_dead_code_fraction"]  < 0.10
    assert report["p1_perplexity"]          >= 32
    assert report["p2_retention"]           > 0.95
    assert report["p3_collision_count"]     == 0
    assert report["p4_connected"]           is True
    assert (report["p4_k_per_wml"]          == 2).all()
    assert report["all_passed"]             is True
```

- [ ] **Step 2: Run the test and verify it fails**

```bash
uv run pytest tests/integration/test_gate_p.py -v
```

Expected: `ImportError: cannot import name 'run_gate_p'`.

- [ ] **Step 3: Add the combined runner**

Append to `scripts/track_p_pilot.py`:

```python
def run_gate_p() -> dict:
    """Run P1..P4 end-to-end, returning a single JSON-serialisable report."""
    import math

    torch.manual_seed(0)
    cb = run_p1(steps=2000)
    counts = cb.usage_counter.float()
    p = counts / (counts.sum() + 1e-9)
    p1_ent = -(p * (p + 1e-9).log2()).sum().item()
    p1_perp = 2 ** p1_ent
    p1_dead = (cb.usage_counter == 0).float().mean().item()

    _, p2_retention = run_p2(steps=2000)
    p3_collisions   = run_p3(n_cycles=200)
    p4_connected, p4_k = run_p4(n_wmls=4, k=2)

    all_passed = (
        p1_dead      < 0.10
        and p1_perp  >= 32
        and p2_retention > 0.95
        and p3_collisions == 0
        and p4_connected
        and bool((p4_k == 2).all())
    )

    return {
        "p1_dead_code_fraction": p1_dead,
        "p1_perplexity":         p1_perp,
        "p2_retention":          p2_retention,
        "p3_collision_count":    p3_collisions,
        "p4_connected":          p4_connected,
        "p4_k_per_wml":          p4_k,
        "all_passed":            all_passed,
    }


if __name__ == "__main__":
    import json

    report = run_gate_p()
    # Serialise: convert tensor → list for JSON compatibility.
    serial = {
        k: v.tolist() if hasattr(v, "tolist") else v
        for k, v in report.items()
    }
    print(json.dumps(serial, indent=2))
```

- [ ] **Step 4: Run the test and verify it passes**

```bash
uv run pytest tests/integration/test_gate_p.py -v
uv run python scripts/track_p_pilot.py   # manual gate-P run
```

Expected: test passes, CLI prints the JSON report with `"all_passed": true`.

- [ ] **Step 5: Commit**

```bash
git add scripts/track_p_pilot.py tests/integration/test_gate_p.py
git commit -m "feat(track-p): combined Gate P runner + report"
```

---

### Task 17: Final sweep — lint, type-check, coverage, full test run

**Files:** (no file changes in this task)

- [ ] **Step 1: Run ruff and fix any issues**

```bash
uv run ruff check .
uv run ruff check . --fix
```

If any issues remain that cannot be auto-fixed, fix manually.

- [ ] **Step 2: Run mypy**

```bash
uv run mypy nerve_core track_p
```

Expected: no type errors. If any, fix inline (most likely: missing `-> None` returns, unused imports).

- [ ] **Step 3: Run the full test suite with coverage**

```bash
uv run pytest --cov=nerve_core --cov=track_p --cov-report=term-missing
```

Expected: all tests pass. Coverage should be ≥ 85 % on both packages. Fix any gap by adding the missing tests.

- [ ] **Step 4: Commit (only if any fixes happened in steps 1-3)**

```bash
git add -u
git commit -m "chore: lint + type + coverage sweep"
```

If nothing changed, skip the commit.

- [ ] **Step 5: Tag Gate P**

```bash
git tag -a gate-p-passed -m "Gate P passed: foundation + Track-P protocol simulator validated"
```

---

## Self-Review Notes (for the writer)

**Spec coverage check** — mapped each spec section to a task:

- §4.1 Neuroletter → Task 3 ✓
- §4.2 Nerve Protocol → Task 4 ✓
- §4.3 NerveEndpoint (transducer) → Task 8 ✓
- §4.4 WML Protocol → Task 4 ✓
- §4.5 Invariants N-1..N-5 → Task 5 (N-1/N-3/N-4 guards; N-2 covered implicitly by `@dataclass(frozen=True)`; N-5 is a design statement implemented by the Transducer per-nerve structure)
- §7.1 L_vq, L_entropy, L_sparsity, L_role_sep, L_surprise losses → L_vq in Task 7, L_entropy in Task 8, L_sparsity emerges from SparseRouter in Task 9; `L_role_sep` and `L_surprise` are WML-specific and belong to **Plan 2** (Track-W), not this plan — noted.
- §7.2 Curriculum P1-P4 → Tasks 12-15 ✓
- §7.5 Stability tricks (EMA, Gumbel) → implemented in Tasks 7 and 9 ✓
- §8.1 Unit tests L1 → every task from 3 onward ✓
- §8.2 Info-theoretic L2 → Tasks 11, 12 (dead codes, perplexity) and inline in Task 16
- §8.3 L3 integration → not in this plan (polymorphie is Track-W, Plan 2)
- §8.4 L4 golden regressions → not in this plan (scheduled in Plan 3 once the merged system is stable)
- §9 Module layout → Tasks 1-10 plus pilots 12-16 produce `nerve_core/`, `track_p/`, `scripts/`, `tests/`

**Deliberate deferrals** (all covered in later plans):

- `track_w/*` — Plan 2
- `bridge/merge_trainer.py` — Plan 3
- `harness/run_registry.py` (R1 reproducibility) — Plan 2 or 3 (only needed once WMLs train end-to-end)
- `papers/paper1/` — post-merge

**Type consistency sweep** — method names and signatures cross-checked: `quantize(z)` returns `(indices, quantized, loss)` in Task 7 and is consumed the same way in Task 12. `sample_edges(tau, hard)` in Task 9 matches its call site in `SimNerve.__init__` (Task 10). `SparseRouter` and `PhaseOscillator` APIs stay stable.

---

## Plan complete — execution handoff

**Plan complete and saved to `docs/superpowers/plans/2026-04-18-nerve-wml-plan.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

**Which approach?**

After Plan 1 (Gate P), we'll write **Plan 2** (Track-W: MockNerve + MlpWML + LifWML + Gate W polymorphie test) and **Plan 3** (Merge: bridge + Gate M + paper draft).
