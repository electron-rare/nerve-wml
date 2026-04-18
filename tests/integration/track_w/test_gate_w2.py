import torch

from scripts.track_w_pilot import run_w2


def test_w2_polymorphie_gap_under_5pct():
    torch.manual_seed(0)
    report = run_w2(steps=400)
    assert report["acc_mlp"] > 0.6
    assert report["acc_lif"] > 0.6
    gap = abs(report["acc_mlp"] - report["acc_lif"]) / report["acc_mlp"]
    assert gap < 0.05, f"polymorphie broken: {gap:.3f} >= 0.05"
