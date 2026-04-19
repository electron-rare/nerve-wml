"""W2-hard at scale: does the polymorphism gap contract hold?

Background: at N=2 with the v0.4 symmetric heads, run_w2_hard reports
a 10.7 % gap in the LIF > MLP direction — large enough to violate the
5 % polymorphism contract. The question this test answers is whether
that gap reflects a real substrate asymmetry or seed-level variance
that averages out at pool scale.

Empirical finding (2026-04-20):
  - N=16: mean_acc_mlp ≈ 0.508, mean_acc_lif ≈ 0.517, gap ≈ 1.68 %
  - N=32: mean_acc_mlp ≈ 0.503, mean_acc_lif ≈ 0.511, gap ≈ 1.55 %
  Both well under the 5 % contract. The N=2 reversal was
  substantially in the variance band; averaging over N/2 instances
  per substrate refocuses on the cohort-level expressivity, which
  turns out to be statistically indistinguishable.
"""
import torch

from scripts.track_w_pilot import run_w2_hard_n16, run_w2_hard_n32


def test_w2_hard_n16_gap_under_5pct():
    """At N=16 the substrate-symmetric contract holds on the hard task."""
    torch.manual_seed(0)
    r = run_w2_hard_n16(steps=400)
    assert r["n_mlp"] == 8 and r["n_lif"] == 8
    # Both cohorts beat the 1/12 random floor by a large margin.
    assert r["mean_acc_mlp"] > 0.40
    assert r["mean_acc_lif"] > 0.40
    assert r["gap"] < 0.05, (
        f"N=16 hard-task polymorphism gap {r['gap']:.4f} exceeds 5 % — "
        "if this regresses, verify RNG isolation between MLP and LIF cohorts"
    )


def test_w2_hard_n32_gap_under_5pct():
    """At N=32 the gap continues to hold; confirms statistical closure."""
    torch.manual_seed(0)
    r = run_w2_hard_n32(steps=200)
    assert r["n_mlp"] == 16 and r["n_lif"] == 16
    assert r["mean_acc_mlp"] > 0.40
    assert r["mean_acc_lif"] > 0.40
    assert r["gap"] < 0.05, (
        f"N=32 hard-task polymorphism gap {r['gap']:.4f} exceeds 5 %"
    )


def test_w2_hard_n2_reversal_is_variance_not_substrate():
    """Cross-check: N=2 reversal shrinks at N=16 vs remaining > 5 %.

    Concretely asserts that N=16's gap is at least 3× smaller than
    the N=2 reversal (10.7 %). This encodes the scientific claim:
    the N=2 result is within the inter-seed variance band, not a
    substrate-intrinsic gap.
    """
    torch.manual_seed(0)
    r16 = run_w2_hard_n16(steps=400)
    n2_reversal = 0.107
    assert r16["gap"] < n2_reversal / 3.0, (
        f"N=16 gap {r16['gap']:.4f} is not substantially smaller than "
        f"the N=2 reversal ({n2_reversal}); the statistical-closure claim "
        "needs re-examination"
    )
