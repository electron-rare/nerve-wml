import torch

from scripts.track_p_pilot import run_p3_no_priority


def test_p3_ablation_measures_positive_collision_rate():
    """Without γ-priority, γ and θ should collide somewhat. Spec §13.1 predicts ~25 %."""
    torch.manual_seed(0)
    collision_rate = run_p3_no_priority(n_cycles=1000)
    assert collision_rate > 0.05, (
        f"collision_rate={collision_rate:.3f}: without priority, γ/θ overlap "
        "should be significant. If ~0, the ablation flag isn't wired."
    )
    assert collision_rate < 0.50
