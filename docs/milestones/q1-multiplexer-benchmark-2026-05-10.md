# Q1 — GammaThetaMultiplexer empirical benchmark

**Date pre-registered:** 2026-05-10 (BEFORE building benchmark scaffold)
**Spec source:** HYPNEUM-PLANS/specs/2026-05-10-three-innovation-experiments-design.md
**Plan source:** HYPNEUM-PLANS/2026-05-10-niveau8-three-experiments.md (Task 11)
**Status:** Pre-registered, not yet executed.

## H0 (to refute)

The GammaThetaMultiplexer (GTM) outperforms classical latent-space
bridge architectures on HardFlowProxyTask N=2, with statistically
significant wins on at least 2 of 3 metrics (mutual information per
unit budget H, round-trip fidelity, bandwidth efficiency) versus each
of 3 baselines (RecursiveLink-like Yang2026, MLPBridge,
CrossAttentionBridge), Bonferroni-corrected at α=0.05/9 ≈ 0.0056.

## Methodology

- **4 architectures:**
  - GammaThetaMultiplexer (existing `track_p/multiplexer.py`, 406 LOC)
  - RecursiveLink-like (port from Yang et al. arXiv:2604.25917 — 2-layer
    linear projection + cosine alignment loss)
  - MLPBridge (vanilla 2-layer MLP, same hidden_dim as GTM)
  - CrossAttentionBridge (single-head cross-attention)
- **Same hidden_dim** across all 4 (use GTM default for parity)
- **Task:** HardFlowProxyTask N=2 (existing `track_w/tasks/hard_flow_proxy.py`)
- **Seeds:** 5 per arch (0, 17, 42, 73, 101)
- **Metrics:**
  1. mi_h — mutual information per unit budget
  2. round_trip_fidelity — encode→decode reconstruction quality
  3. bandwidth_efficiency — useful bits per channel
- **Statistical test:** Welch's t-test pairwise GTM vs each baseline,
  per metric (3 × 3 = 9 comparisons)
- **Multiple-comparisons correction:** Bonferroni α = 0.05/9 ≈ 0.0056
- **GTM ablation (separate sweep):** 3 variants × 5 seeds
  - gamma-only (theta envelope frozen at 1.0)
  - theta-only (gamma carrier frozen)
  - no-plasticity (plasticity schedule constant)

## Decision criteria (pre-stated)

- **GTM-headline:** GTM wins ≥6/9 comparisons (i.e., ≥2/3 metrics
  across all 3 baselines, Bonferroni-corrected) → Paper 2 §X.Y leads
  with the benchmark, GTM positioned as state-of-the-art bridge
- **Tied:** 3 ≤ wins < 6 OR ties dominate → Paper 2 §X.Y reframes as
  "convergent evidence : GTM matches latent-space baselines while
  preserving phase-coupled biological plausibility" — claim shifts from
  performance to interpretability+biology
- **GTM-loses:** GTM wins ≤2/9 OR loses ≥6/9 → Paper 2 §X.Y reframes
  as "falsifiability-scope : GTM efficacy is task-dependent ; on
  HardFlowProxyTask N=2 latent-space methods dominate ; PAC-based
  multiplexing requires task properties [list] which this benchmark
  lacks." Negative result publishable as scope-clarification.

## Soft-gate from Q2 outcome (cross-experiment)

If Q2 finds ≥3 FPs in Conformance Criterion, Paper 2 §X.Y narrative
must additionally widen to "GTM in framework C+ (extended criterion)"
— annotated in the closeout.
