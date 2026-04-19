import torch

from scripts.track_p_pilot import run_p1_random_init
from track_p.info_theoretic import dead_code_fraction


def test_p1_random_init_meets_gate():
    """With rotation + longer training, random-init hits < 10 % dead-code gate."""
    torch.manual_seed(0)
    cb, dead = run_p1_random_init(steps=16000)
    assert dead < 0.10, f"dead_code_fraction={dead:.3f}"
    assert dead_code_fraction(cb) < 0.10
