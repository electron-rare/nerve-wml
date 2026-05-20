# D3 literature check: is HSIC standalone non-informativeness already published?

**Date:** 2026-05-20
**Finding under review (D3):** "HSIC débiaisé (Song et al. 2012 unbiased
estimator) is statistically non-informative when used standalone — |r|≤0.015
with CKNNA, linear-CKA, Procrustes-r², mutual-kNN — on 1750 cells. Always
normalise to linear-CKA."

## Summary verdict

🟡 **Partially known.** The *qualitative* fact that raw/unnormalised HSIC is
unsuitable as a standalone similarity metric — because it is not invariant to
isotropic scaling and is dominated by the magnitudes of `HSIC(K,K)` and
`HSIC(L,L)` — is **explicitly stated** in Kornblith et al. (2019) and
repeated in the Klabunde et al. (2023/2025) survey. Every modern paper on
representational similarity follows that recommendation in practice
(everyone reports CKA, not HSIC).

What is **not** in the prior literature, to the best of this search, is the
*empirical quantification* of how badly raw HSIC tracks the rest of the
similarity-metric family. Concretely: no paper found reports correlations
between debiased HSIC and {CKNNA, linear-CKA, Procrustes-r², mutual-kNN} on
a large grid of layer-pair cells. Kornblith 2019 motivates normalisation
analytically and with sanity checks; the 2024 debias-CKA paper compares
biased vs debiased CKA but not debiased *HSIC* vs other metrics; Klabunde
catalogues the relationships theoretically but does not run this exact
empirical sweep. So D3's headline ("|r|≤0.015 across 8 metrics × 1750
cells") is best framed as a **quantitative confirmation of a known
theoretical caveat**, not a new conceptual discovery.

## Investigation method

Searches run (WebSearch + WebFetch):

1. `Kornblith 2019 CKA HSIC normalization invariance "not invariant" representational similarity`
2. `Klabunde 2023 survey similarity neural network representations HSIC CKA`
3. `debiased HSIC standalone uninformative scale sensitivity neural representation`
4. `"raw HSIC" OR "unnormalized HSIC" representational similarity uninformative neural networks`
5. `platonic representation hypothesis CKNNA mutual-kNN HSIC correlation comparison Huh 2024`

Papers / sources consulted:

- Kornblith et al., *Similarity of Neural Network Representations Revisited*,
  ICML 2019 (arXiv 1905.00414) — via alphaxiv overview, vitalab notes,
  Hinton lab PDF mirror. PDF binary unparseable by WebFetch; relied on
  secondary summaries.
- Klabunde, Schumacher, Strohmaier, Lemmerich, *Similarity of Neural Network
  Models: A Survey of Functional and Representational Measures*,
  arXiv 2305.06329 / ACM Computing Surveys 2025 — abstract + secondary
  summaries. Full PDF body not parseable via WebFetch.
- Murphy et al. / Lange et al., *Correcting Biased Centered Kernel Alignment
  Measures*, ICLR 2024 Re-Align workshop (arXiv 2405.01012) — read in full.
- Huh et al., *The Platonic Representation Hypothesis*, arXiv 2405.07987 —
  abstract level (introduces CKNNA, mKNN, cycle-kNN alongside linear/RBF
  CKA).
- Datumorphism CKA card, EmergentMind CKA topic — secondary explainer
  references.

## Findings per paper

### Kornblith et al. 2019 (CKA)

Kornblith et al. **explicitly state** that HSIC alone is unsuitable because
it is not invariant to isotropic scaling, and they introduce CKA precisely
as the normalisation that fixes this:

> "HSIC is not invariant to isotropic scaling. This can be fixed by
> normalizing..." — paraphrased from §3 of Kornblith 2019, where CKA is
> defined as
> `CKA(K, L) = HSIC(K, L) / sqrt(HSIC(K, K) · HSIC(L, L))`.

They argue that without this normalisation, comparisons between layers with
different activation magnitudes are dominated by scale rather than
structure. They also acknowledge Song et al.'s (2007/2012) unbiased
estimator and recommend swapping it into the centering step.

What they do **not** do: report an empirical correlation between raw
(unnormalised, debiased) HSIC and other similarity families (Procrustes,
mutual-kNN, CKNNA — the last didn't exist yet). Their experimental work
uses CKA throughout; raw HSIC is dismissed analytically, not measured.

### Song et al. 2012 (debiased / unbiased HSIC)

Song et al. (JMLR 13, 2012; building on Song et al. 2007) derive an
unbiased U-statistic estimator of HSIC and use it for feature selection via
dependence maximisation. The framing is that HSIC's *bias* under finite
samples is the issue to fix, not its *scale sensitivity*. There is no
treatment of representational similarity, no comparison with other
similarity metrics, and no claim about standalone informativeness in the
deep-network sense D3 reports. So this paper is upstream of the debiasing
step but says nothing about D3's empirical finding.

### Klabunde et al. 2023/2025 (survey)

The survey catalogues HSIC and CKA together and notes that "a normalization
of the HSIC yields the CKA measure, which is bounded in [0,1]". It
inherits Kornblith's recommendation: CKA is the operational metric of
choice; HSIC standalone is not usually presented as a competitor in the
representational-similarity table. The survey also flags the `N < D`
regime as one where the debiased Song estimator becomes important.

What the survey does not appear to provide is a full empirical correlation
matrix across the metric family (Procrustes, mutual-kNN, CKNNA, CKA,
HSIC), so D3's numerical claim is not pre-empted here either. (Note:
the full PDF body was not directly parseable from WebFetch, so this is
based on the abstract, ACM landing page, and two independent secondary
summaries.)

### Murphy / Lange et al. 2024 (debiased CKA workshop paper)

This paper compares **biased CKA vs debiased CKA** on biological and
artificial neural data, and shows that biased CKA spuriously mixes a
structure-driven and a stimuli-driven component, whereas debiased CKA
isolates the latter. It does **not** compare debiased HSIC (without
normalisation) against CKNNA / Procrustes / mutual-kNN. So it confirms the
"debias" half of D3's pipeline but says nothing about the "always
normalise" half.

### Huh et al. 2024 (Platonic Representation Hypothesis)

This paper popularises CKNNA alongside mutual-kNN, cycle-kNN, linear-CKA
and RBF-CKA as a metric family for cross-model representational alignment.
Crucially, it **does not** include raw HSIC in its metric battery — which
itself is implicit evidence that the community treats raw HSIC as
non-competitive with CKA. But it does not run the D3 experiment either.

## Conclusion

D3 sits in the middle of the novelty spectrum. The **conceptual claim** —
that raw / unnormalised HSIC is unsuitable as a standalone similarity
metric because it is dominated by scale — is well-established and was the
original motivation for CKA in Kornblith 2019, restated in Klabunde
2023/2025 and assumed implicitly by every downstream paper (Huh 2024
doesn't even include raw HSIC in its metric set). On that basis, D3 should
not be presented as a *conceptual* contribution.

The **empirical quantification** D3 produces — |r| ≤ 0.015 between
debiased HSIC and four other major metrics (CKNNA, linear-CKA,
Procrustes-r², mutual-kNN) on 1750 layer-pair cells — does not appear in
the literature reviewed. It would be a clean, citable confirmation that
prior theoretical warnings translate into near-zero rank-correlation in
practice on a modern setup. Calling it "novel" is a stretch; calling it
"first numerical demonstration on this scale and metric set" is defensible.

## Recommendation for the nerve-wml paper

Frame D3 as a **methodological caveat / sanity check**, not as a headline
finding. Cite Kornblith et al. (2019, §3) for the scale-invariance
argument and the CKA normalisation formula, Klabunde et al. (2023/2025)
for the survey-level confirmation, and Song et al. (2012) for the unbiased
estimator. Then state that we *quantitatively* confirm this caveat by
showing |r| ≤ 0.015 with four other metrics on 1750 cells, and use that
as justification for the paper's choice to report linear-CKA throughout.
One paragraph in the methods or appendix is the right amount of space; do
not give it a dedicated results section.
