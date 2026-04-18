import torch

from scripts.track_p_pilot import run_p2
from track_p.info_theoretic import kl_divergence


def test_p2_transducer_is_not_uniform_after_training():
    torch.manual_seed(0)
    transducer, _ = run_p2(steps=2000)

    # Check one row: post-training distribution should be far from uniform.
    import torch.nn.functional as F
    row = F.softmax(transducer.logits[7], dim=-1)
    uniform = torch.full_like(row, 1.0 / 64)
    kl = kl_divergence(row, uniform)
    assert kl.item() > 1.0


def test_p2_transducer_retention_above_95pct():
    """Retention: of all codes sent through the transducer, ≥ 95 %
    are decoded back to the expected output code in a known src→dst pairing."""
    torch.manual_seed(0)
    _, retention = run_p2(steps=2000)
    assert retention > 0.95
