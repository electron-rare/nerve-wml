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
