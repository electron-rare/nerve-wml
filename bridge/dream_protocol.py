"""Dream-of-kiki protocol adapter.

`nerve-wml` does not take `kiki_oniric` (the dream-of-kiki runtime) as a
runtime dep. Instead this module lazily imports it at first call; if the
package is missing, None is returned and the DreamBridge falls back to a
MockConsolidator for CI.

Expected interface (when `kiki_oniric` is installed):
    consolidate(trace: np.ndarray, *, profile: str = "P_equ") -> np.ndarray

Plan 7 Task 1.
"""
from __future__ import annotations

import importlib
from typing import Any


def load_dream_module(module_name: str = "kiki_oniric") -> Any | None:
    """Try to import kiki_oniric. Return None on ImportError.

    Callers must handle the None case by falling back to MockConsolidator.
    """
    try:
        return importlib.import_module(module_name)
    except ImportError:
        return None


def assert_protocol_surface(module: Any) -> None:
    """Sanity-check that a loaded module exposes the expected callable.

    Raises AssertionError with a clear message if the module does not
    implement the `consolidate` callable with the right signature.
    """
    assert hasattr(module, "consolidate"), (
        f"dream module {module!r} does not expose consolidate() — "
        "see docs/dream/integration-notes.md for the expected surface."
    )
    assert callable(module.consolidate), (
        f"dream module {module!r}.consolidate is not callable."
    )
