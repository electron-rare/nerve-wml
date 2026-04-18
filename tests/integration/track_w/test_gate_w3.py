import torch

from scripts.track_w_pilot import run_w3


def test_w3_eps_feedback_beats_baseline():
    torch.manual_seed(0)
    baseline, with_eps = run_w3(steps=400)
    # Gate W3: baseline beaten by ≥ 10 % relative.
    assert (with_eps - baseline) / baseline >= 0.10
