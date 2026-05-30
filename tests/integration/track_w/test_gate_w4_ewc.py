"""Integration gate: EWC on HardSplitTask — measured forgetting reported honestly."""
import torch
from scripts.track_w_pilot import run_w4_compare
from track_w.tasks.hard_split import HardSplitTask


def test_ewc_forgetting_is_reported():
    """EWC forgetting is measured and returned (honest reporting, any value)."""
    torch.manual_seed(0)
    task = HardSplitTask(seed=0)
    result = run_w4_compare(method="ewc", task=task, steps=400, seed=0, lam=1.0)
    assert "forgetting" in result
    assert isinstance(result["forgetting"], float)
    # EWC result is reported even if it does not beat rehearsal.
    # The < 0.20 threshold is a target, NOT a construction (spec §13).


def test_ewc_beats_none_baseline():
    """Soft gate: EWC forgetting < none forgetting (seed=0, lam=1.0).

    If this fails, report the honest result — do not tune lam to pass by construction.
    Threshold traces to docs/superpowers/research/2026-05-30-w4-ewc-comparison.json
    key "ewc_vs_none_seed0".
    """
    torch.manual_seed(0)
    task = HardSplitTask(seed=0)
    none_result = run_w4_compare(method="none", task=task, steps=400, seed=0)
    ewc_result  = run_w4_compare(method="ewc",  task=task, steps=400, seed=0, lam=1.0)
    assert ewc_result["forgetting"] < none_result["forgetting"], (
        f"EWC ({ewc_result['forgetting']:.3f}) did not beat none "
        f"({none_result['forgetting']:.3f}). "
        "Record this as-is in the research JSON (honest reporting)."
    )
