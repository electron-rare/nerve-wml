import torch

from scripts.track_w_pilot import run_w2_n16


def test_w2_n16_relative_gap_under_10pct():
    """16-WML half-MLP/half-LIF pool: relative polymorphie gap < 10 %."""
    torch.manual_seed(0)
    report = run_w2_n16(steps=400)
    assert report["mean_acc_mlp"] > 0.6
    assert report["mean_acc_lif"] > 0.6
    gap = abs(report["mean_acc_mlp"] - report["mean_acc_lif"]) / max(
        report["mean_acc_mlp"], 1e-6
    )
    assert gap < 0.10, f"N=16 polymorphie gap {gap:.3f} exceeds 10 %"
