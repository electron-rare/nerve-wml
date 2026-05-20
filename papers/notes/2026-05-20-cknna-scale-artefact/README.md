# CKNNA scale-artefact note (2026-05-20 draft)

Short arXiv methodology note distilling two findings from the
nerve-wml gap-analysis work:

- **D4** CKNNA is intrinsically N-dependent at fixed signal.
- **D9** The signal-vs-null ratio (Cohen's $d_z$) is the
  cross-paper-comparable quantity.

Target audience: the broader representational-alignment community
(PRH / CKA / CKNNA papers).

## Contents

| File                          | Purpose                                  |
| ----------------------------- | ---------------------------------------- |
| `note.md`                     | The markdown draft (pandoc → PDF)        |
| `refs.bib`                    | BibTeX entries cited by `note.md`        |
| `make_figures.py`             | Regenerates the two figures from JSON    |
| `figures/fig1_cknna_decay.png`| CKNNA@10 vs N at σ=0.05 (real + null)    |
| `figures/fig2_dz_growth.png`  | $d_z$ vs N (log–log), CKNNA + HSIC       |

## Data source

All numerical content is derived from the 1750-cell evaluation
committed in the repository root at:

    docs/superpowers/research/2026-05-20-macm1-scientific-eval.json

The figure script reads that file directly; the tables in `note.md`
were transcribed from the companion summary
`docs/superpowers/research/2026-05-20-macm1-scientific-eval.md`.

## Regenerate figures

From the repository root:

```bash
uv run --with matplotlib python \
    papers/notes/2026-05-20-cknna-scale-artefact/make_figures.py
```

The script prints the headline numbers (CKNNA@10 mean ± SD per N at
σ=0.05) for cross-checking against Table 1 of the note.

## Build PDF

```bash
cd papers/notes/2026-05-20-cknna-scale-artefact
pandoc note.md \
    --bibliography=refs.bib \
    --citeproc \
    --pdf-engine=xelatex \
    -V geometry:margin=1in \
    -V mainfont="Times New Roman" \
    -o note.pdf
```

Pandoc ≥ 3 is recommended.

## Pre-submission checklist

- [x] All numbers in the note traceable to the committed JSON
- [x] Figures regenerate without error from real data
- [x] No references to nerve-wml-internal artefacts ("Renf 6",
      "Plan A.2", etc.) in the note body
- [x] Limitations section honest about substrate, noise model, host
- [x] References are arXiv-citable
- [ ] External review (target: 1–2 weeks)
- [ ] arXiv submission (cs.LG primary, stat.ML cross-list)
