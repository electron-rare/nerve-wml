# Equal-tuning protocol

## Motivation

Cross-method comparisons (learned Transducer vs. vec2vec vs. relative
representations vs. Procrustes) are only credible when every tunable
arm has had a comparable opportunity to find a good hyperparameter
configuration. Otherwise the result reflects the experimenter's prior
beliefs about which method "deserves" more tuning effort.

## Protocol

1. **Equal trial budget.** Every tunable method gets the same number
   of trials (`budget`, default 8) drawn deterministically from a
   fixed, declared grid. The grid is published in
   `scripts/equal_tuning_pilot.py` (`LEARNED_GRID`, `VEC2VEC_GRID`,
   `RELREP_GRID`).
2. **Equal seed budget per trial.** Each trial is run under
   `seeds_per_trial` independent seeds (default `(0, 1, 2)`). MI is
   averaged across seeds; a non-parametric bootstrap 95% CI is
   computed from the per-seed values.
3. **Best-of-budget selection.** For each method we report the trial
   with the highest mean MI (best-of-budget). This matches the
   convention used in HP-search papers (Lucic et al. 2018,
   Henderson et al. 2018) and avoids the "first-trial-wins" bias.
4. **Procrustes parity.** Procrustes is closed-form and has no
   hyperparameters. It is reported once under the same
   multi-seed budget (`seeds_per_trial`), so its CI is computed on
   the same number of seeds as a single trial of a tunable method.
   This is documented in the output as the `procrustes.note` field.
5. **No method-specific budget inflation.** If reviewers request a
   larger budget for one method, the same larger budget MUST be
   applied to every tunable method.

## Output shape

```python
{
  "learned":   {"trials": [...8 trials...], "best": {...}},
  "vec2vec":   {"trials": [...8 trials...], "best": {...}},
  "relrep":    {"trials": [...8 trials...], "best": {...}},
  "procrustes":{"trials": [1 trial],        "best": {...}, "note": "..."},
}
```

Each trial dict has `params`, `mi_values` (one per seed), `mi_mean`,
`mi_ci95_low`, `mi_ci95_high`.

## Reporting checklist

When citing equal-tuning results in a paper or report:

- [ ] State the budget and the seed-count per trial.
- [ ] Publish the full grid (not just the winner).
- [ ] Report mean MI ± bootstrap 95% CI for the best trial of each
      method.
- [ ] Disclose that Procrustes is non-tunable and reported at parity.
- [ ] Disclose the limitations below (in this file, this pilot's HPs
      are not all wired through the underlying runners).

## Limitations of the current pilot

This pilot is honest about what it *measures* vs. what it *declares*.
Two non-trivial gaps remain between the declared grids and the
underlying runners; they are deliberately surfaced here so that
reviewers can judge equal-effort, rather than being silently glossed
over:

- **Learned `lr` is accepted-but-ignored.** The underlying
  `scripts.transducer_baselines_pilot._train_learned(src, dst, steps)`
  hard-codes `Adam(lr=0.05)`. `LEARNED_GRID` varies `(steps, lr)` for
  grid-completeness, but `_learned_runner` discards the `lr` argument
  and only `steps` and `seed` truly vary across trials. Wiring `lr`
  through `_train_learned` is future work (Plan A.3).
- **`lambda_cycle` and `n_anchors` are accepted-but-ignored.**
  `run_transducer_benchmark(steps, seed)` does not expose
  `lambda_cycle` (vec2vec) or `n_anchors` (relative representations);
  both are hard-coded inside the bundled benchmark. `VEC2VEC_GRID` and
  `RELREP_GRID` accept these parameters for grid-completeness, but
  the trial variation in this pilot reflects **seed noise** and
  (for vec2vec) `steps`, not full HP variation. Wiring these through
  the bundled benchmark is future work (Plan A.3).
- **Consequence.** The current pilot is an *equal-budget infrastructure
  test* and a *seed-noise envelope*, not yet a full HP sweep. Once
  `_train_learned` and `run_transducer_benchmark` accept the relevant
  knobs, the same `run_equal_tuning` driver will become a real equal-
  HP-effort comparison without any change to the protocol.

This disclosure is the whole point of the protocol: equal-effort
must be visible and falsifiable, not an unverifiable claim.
