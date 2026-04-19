import torch

from scripts.track_w_pilot import run_w4_rehearsal


def test_w4_rehearsal_forgetting_under_20pct():
    """Shared head, same lr, PLUS rehearsal buffer — forgetting < 20 %."""
    torch.manual_seed(0)
    report = run_w4_rehearsal(steps=400, rehearsal_frac=0.3)
    assert report["forgetting"] < 0.20, f"forgetting={report['forgetting']:.3f}"
