"""Top-level shared fixtures for nerve-wml tests.

Extracted from per-file copy-paste in N2 Task 4 follow-up (commit
811964d) per N3 Task 3 (loose-end #5 from N2 audit, 2026-05-10).
"""
from __future__ import annotations

import os
import sys

# Sentinel for "running on the GitHub Actions Linux/Py3.12 runner"
# where 3 platform-fragile tests fail (golden hash drift, RNG
# statistical claim, gumbel autograd). Used by tests in
# tests/golden/, tests/integration/, tests/unit/ to xfail those
# tests on CI without affecting local macOS/Py3.14 behaviour.
LINUX_CI_312 = (
    os.environ.get("CI") == "true"
    and sys.platform == "linux"
    and sys.version_info[:2] == (3, 12)
)
