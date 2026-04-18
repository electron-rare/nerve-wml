import torch

from scripts.track_p_pilot import run_p3


def test_p3_no_phase_collisions():
    """γ letters and θ letters never share a delivery timestamp."""
    torch.manual_seed(0)
    collision_count = run_p3(n_cycles=200)
    assert collision_count == 0
