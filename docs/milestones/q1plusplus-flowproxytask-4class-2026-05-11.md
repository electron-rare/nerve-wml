# Q1++ — GammaThetaMultiplexer cross-task generalization

**Date pre-registered:** 2026-05-11 (BEFORE sweep launch on grosmac)
**Spec source:** HYPNEUM-PLANS/2026-05-11-niveau9-scaling-experiments.md
**N8 cross-reference:** Q1 verdict `tied` on HardFlowProxyTask (XOR-on-noise hard task).
**Status:** Pre-registered, not yet executed.

## H0 (to refute)

GTM's `tied` verdict on the hard XOR task generalizes to the canonical easier 4-class FlowProxyTask. Either tied verdict reproduces (GTM advantage is general), GTM saturates with all baselines (degenerate canonical task), or GTM ties only at scale + loses at canonical (advantage is regime-specific).

## Methodology

- Same 4 architectures, 5 seeds, 3 metrics, Bonferroni α=0.0056
- Task change: FlowProxyTask **4-class linearly-separable** (canonical task in `track_w/tasks/flow_proxy.py` — pre-existing nerve-wml fixture). Per nerve-wml docs §Threats to Validity, MLP and LIF saturate to 1.0 on this task ; expectation is all 4 bridges saturate on rtf and lose discrimination.
- Wallclock estimate: easier task → faster convergence ; ~10-20 min on grosmac M5 (light, CPU-bound, single-process).
- Launch on **grosmac** local (light workload per feedback_grosmac_light_only).

## Decision criteria (pre-stated)

- **All-saturate:** all 4 archs ≥0.95 rtf with no significant Bonferroni differences → degenerate task confirms task-difficulty matters, Paper 2 §7.9 cites this as positive control showing the metric framework is sensitive to task properties.
- **Tied-stable-cross-task:** 3-5 wins → GTM advantage robust across task difficulty ; strongest possible support for §7.9 convergent-evidence narrative.
- **GTM-only-wins-at-hard:** GTM ties or loses on easy 4-class while winning at hard XOR (per Q1 mi_h domination at hard) → Paper 2 §7.9 adds "PAC phase-coupling provides advantage in regime where linear methods fail (XOR-on-noise) ; saturated at canonical" — most scientifically interesting outcome.
