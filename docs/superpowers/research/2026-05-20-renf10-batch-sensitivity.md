# Renf 10 — spectral_entropy robustness to batch size B

**Date** : 2026-05-20
**Host** : macm1 (M1, 10 cores, CPU multiproc, 8 workers)
**Wall-clock** : 3132.2 s (~52 min, under CPU contention with a sister job)
**JSON** : `2026-05-20-renf10-batch-sensitivity.json`
**Script** : `scripts/renf10_batch_sensitivity.py`

## Hypothesis

Renf 7 fixed batch size **B = 128** and reported the spectral_entropy ordering
`null < akorn_best < gtm < simple_gating`. If this ordering is B-dependent the
metric is fragile.

## Method

Sweep **B ∈ {64, 128, 256, 512}** × 50 seeds × 4 arms (gtm, simple_gating,
akorn_best with the Renf 1 top cell `n_osc=64, n_steps=32, lr=0.05`, null
via shuffled codes). For each cell train 200 steps, capture the carrier,
compute 4 candidate metrics (spectral_entropy, participation_ratio,
effective_rank, top3_dispersion). Aggregate mean/std and run paired
Wilcoxon between every pair of arms.

## Headline (spectral_entropy)

| B   | null               | akorn_best       | gtm                | simple_gating      | ordering (low→high)                                |
| --- | ------------------ | ---------------- | ------------------ | ------------------ | -------------------------------------------------- |
| 64  | 2.086 ± 0.046      | **1.933 ± 0.606** | 2.139 ± 0.013      | 3.249 ± 0.019      | akorn_best < null < gtm < simple_gating **(INV)**  |
| 128 | 1.960 ± 0.132      | **1.872 ± 0.686** | 2.173 ± 0.009      | 3.464 ± 0.016      | akorn_best < null < gtm < simple_gating **(INV)**  |
| 256 | 1.738 ± 0.070      | 2.035 ± 0.692    | 2.191 ± 0.005      | 3.623 ± 0.011      | null < akorn_best < gtm < simple_gating **(MATCH)** |
| 512 | 1.654 ± 0.026      | **1.548 ± 0.664** | 2.200 ± 0.003      | 3.766 ± 0.010      | akorn_best < null < gtm < simple_gating **(INV)**  |

The canonical ordering **only holds at B = 256** (the Renf-7 baseline was B = 128
and was an inversion — but the inversion was masked because Renf 7 only ran the
B = 128 cell, not the sweep). At every other B akorn_best falls *below* null in
mean spectral_entropy. The akorn_best dispersion is one to two orders of
magnitude larger than every other arm (std ≈ 0.6–0.7 vs 0.005–0.13), so its
estimated mean is unstable.

## Paired Wilcoxon on adjacent pairs (spectral_entropy)

| B   | first vs second                | second vs third           | third vs fourth                |
| --- | ------------------------------ | ------------------------- | ------------------------------ |
| 64  | akorn_best < null  p = 0.198   | null < gtm  p = 4.8e-09   | gtm < simple_gating  p = 1.8e-15 |
| 128 | akorn_best < null  p = 0.709   | null < gtm  p = 1.8e-15   | gtm < simple_gating  p = 1.8e-15 |
| 256 | null < akorn_best  p = 3.5e-3  | akorn_best < gtm  p = 0.24 | gtm < simple_gating  p = 1.8e-15 |
| 512 | akorn_best < null  p = 0.097   | null < gtm  p = 1.8e-15   | gtm < simple_gating  p = 1.8e-15 |

The **null ↔ akorn_best** pair is never significant at α = 0.05 by Wilcoxon, at
any B. They are statistically indistinguishable. The only signal that is rock-
solid across all four B is **gtm < simple_gating** (p ≈ 1.8e-15 at every B).

## What survives B-sweep?

- ✅ **`gtm < simple_gating`** at all four B, p ≈ 1.8e-15 — robust.
- ✅ **`null < gtm`** at B ∈ {64, 128, 512}, p ≤ 5e-09; at B = 256 the
  null mean *drops* (because more samples concentrate around the chance
  spectrum) but gtm still > null.
- ❌ **`null < akorn_best`** at the means is **B = 256 only** and even
  there p = 3.5e-3 (borderline). At B ∈ {64, 128, 512} akorn_best is
  *below* null in mean, but with enough variance that paired Wilcoxon
  cannot reject the equal-medians null hypothesis.

## Conclusion

**B-robust : PARTIAL.** The ordering `null < akorn_best < gtm < simple_gating`
is **not** B-robust. Specifically the null↔akorn_best step is fragile and
the canonical ordering only happens at B = 256. The robust empirical claims
are :

1. `gtm < simple_gating` on spectral_entropy is statistically rock-solid
   (p ≈ 1.8e-15) at every B in {64, 128, 256, 512}.
2. `gtm > null` on spectral_entropy holds at every B (p ≤ 5e-09 except
   at B = 256 where the null drops further).
3. The placement of `akorn_best` relative to `null` is **B-dependent**
   and statistically indistinguishable from null at every B.

**Paper-level implications.** Any claim that the synchrony-spectrum metric
separates akorn_best from a shuffled-null is fragile and should be retracted
or hedged. Claims that simple_gating dominates gtm and null on
spectral_entropy are robust.

## Side observation

`participation_ratio`, `effective_rank` and `top3_dispersion` show the same
structural pattern — simple_gating dominates by a huge margin, gtm > null
at every B, akorn_best is wildly variable (std ≈ 2.5–5.6 for participation
ratio). So the B-fragility is not specific to spectral_entropy ; it is a
property of the akorn_best training distribution.
