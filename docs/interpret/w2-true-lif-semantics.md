# Neuroletter semantics — WML #0 after `run_w2_true_lif`

Reference extraction of the neuroletter semantics table for an `MlpWML` trained briefly via the Plan 5 `scripts/interpret_pilot.py` harness. Numbers here are **illustrative** of what the extractor produces; they will shift with any change to the training recipe, the random seed, or the task. Regenerate the fresh values from the pilot directly when quoting them.

## Regenerate

```bash
uv run python scripts/interpret_pilot.py
open reports/interp/w2_true_lif_semantics.html
```

Default parameters: `steps=200`, `n_inputs=512`, `n_clusters=8`, `wml_id=0`. The output lands at `reports/interp/w2_true_lif_semantics.html`.

## What the table reports

For each code $c \in \{0, \dots, 63\}$:

| Field | Meaning |
|---|---|
| `n_samples_mapped` | number of the 512 random inputs whose $\pi$-head argmax is $c$ |
| `top_inputs` | summaries (mean, L2 norm, argmax dim) of the top-3 mapped inputs |
| `activation_centroid` | mean hidden state over those inputs (16-dim tensor) |
| `next_codes_distribution` | softmax of $W W^\top$ row $c$ — approximates code→next-code transition |
| cluster id | k-means cluster over all 64 centroids (8 clusters, seed 0) |

## Reading the HTML report

- **Coloured dot**: cluster membership (8-colour palette). Codes sharing a colour have similar activation centroids.
- **Dimmed rows**: `n_samples_mapped == 0`. The WML emits these codes rarely on random input; they still exist in the codebook but are not active for this input distribution.
- **Next argmax** column: which code the π-head would likely emit next if this code's centroid were the hidden state — a tracing hint, not a formal transition matrix.

## Illustrative reading

On a fresh MlpWML (d_hidden=16, seed=0) trained for 200 steps, we typically observe:

- **~20 active codes** (out of 64) for a 512-input random batch. The rest are latent capacity.
- **Cluster entropy ≈ 2.0–2.8 bits** depending on init — above the `gate-interp-passed` threshold of 2 bits but below the uniform ceiling of $\log_2 8 = 3$ bits.
- Codes within the same cluster often have `next_codes_distribution` argmax pointing at a third, shared code — suggesting a "magnitude" or "valence" pathway the WML learned.

This is the kind of narrative a reader can extract from the HTML report. The `interpret/` toolchain is agnostic to the WML checkpoint, so swapping in a trained `run_w2_n16` pool member produces the same artefact structure with different semantics.

## Follow-up

1. **Hard task**: rerun on `HardFlowProxyTask` to see whether the XOR-bit structure shows up as an extra cluster separation.
2. **LIF**: extend the extractor to `LifWML` by replacing the $\pi$-head argmax with the cosine-similarity decoder output. Would show how spike-pattern codebooks cluster relative to MLP codebooks.
3. **Cross-WML semantics**: compare two WMLs from the same pool to measure whether their local codebooks agree on the "same" concept — this is the natural hook into `dream-of-kiki` consolidation (Plan 7).
