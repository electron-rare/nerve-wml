"""R1 reproducibility contract — bit-stable run_id.

run_id = sha256(c_version | sorted_topology | seed | commit_sha)[:16]

See spec §7.4 and §8.4.
"""
from __future__ import annotations

import hashlib
from collections.abc import Iterable


def compute_run_id(
    *,
    c_version:  str,
    topology:   Iterable[tuple[int, int]],
    seed:       int,
    commit_sha: str,
) -> str:
    """Produce a deterministic 16-hex-char run identifier.

    Topology edges are canonicalised (sorted) so edge ordering never
    changes the id — a run is identified by the *set* of active edges.
    """
    edges_sorted = sorted(tuple(sorted(e)) for e in topology)
    payload = "|".join([
        c_version,
        repr(edges_sorted),
        str(seed),
        commit_sha,
    ])
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return digest[:16]


def run_id_for_pilot(
    *,
    pilot_name: str,
    seed:       int,
    commit_sha: str = "local",
) -> str:
    """Convenience wrapper for v1.1 pilots that don't expose an explicit
    topology (e.g., run_w2_hard_n16_multiseed, run_triple_pool_hard, the
    info-transmission pilots). The pilot name acts as the c_version.

    Use only when the true topology is internal to the pilot — prefer
    compute_run_id with an explicit topology when available.
    """
    return compute_run_id(
        c_version=pilot_name,
        topology=(),  # pilot-internal topology
        seed=seed,
        commit_sha=commit_sha,
    )
