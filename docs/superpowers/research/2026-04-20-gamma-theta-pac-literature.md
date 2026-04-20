# Research Memo — `GammaThetaMultiplexer` scientific foundations

**Date**: 2026-04-20
**Project**: nerve-wml (issue #1) × bouba_sens (Hypneum Lab Paper 2 candidate)
**Scope**: literature audit for the γ/θ phase-amplitude coupling DSP multiplexer design
**Coverage**: foundational literature 1929–2025 + updates Jan 2026 → April 20, 2026

---

## 1. Executive summary

The `GammaThetaMultiplexer` design choices are **empirically supported by the literature as of 2026-04-20**, with three caveats:

1. **PAC theta-gamma as a cross-modal binding mechanism** was historically a *plausible hypothesis* but not a *measured mechanism* in cross-modal perception tasks. As of 2026, two papers directly measure PAC in cross-modal contexts — but in **audiovisual learned associations** (hippocampal), not direct perception. This leaves a **positioning gap** that bouba_sens Paper 2 can claim.
2. **Tort Modulation Index** (2010) remains the standard PAC metric but has documented biases (Aru 2015, Hyafil 2015) and at least one 2026 alternative (NARX, He et al. arXiv:2603.08866). For TDD: our proxy "θ-band peak on γ-envelope spectrum" is acceptable on synthetic signals, not on biological data.
3. **Hardware substrate reality check (2026-04-20)**: Loihi 3 does *not* officially exist — "Loihi 3" articles circulating Q1 2026 are promotional content without Intel source. Loihi 2 (Hala Point / Intel NRC) and **SpiNNcloud SpiNNaker2** (commercial January 2026) are the only viable neuromorphic targets for porting a PAC encoder.

---

## 2. Design decisions × literature validation

| Decision | Status | Evidence (most recent first) |
|---|---|---|
| `symbols_per_theta = 7` (default), range `[5, 9]` | **Strengthened** | Harris & Gong 2026 (*Nat Commun*) — nested γ packets per θ cycle in mouse V1 ; Huang et al. 2026 (*PNAS*) — TG-PAC encodes sequential items during human navigation ; Pirazzini & Ursino 2022 (*Cogn Neurodyn*) — theoretical optimum 6-8 for ordered storage ; Colgin 2016 (*Nat Rev Neurosci*) — ratio 4-10 depending on γ band (slow vs fast) ; Lisman & Idiart 1995 (*Science*) — original 7±2 conjecture |
| PAC γ/θ as coding scheme | **Strengthened** | Nour et al. 2024 (*Nature*) — *dedicated* PAC neurons in human hippocampus encode WM ; bioRxiv 2026.01.20.700563 — cortico-hippocampal PAC for learned audiovisual associations ; Harris & Gong 2026 (*Nat Commun*) — spatiotemporal traveling θ/γ waves ; Heusser et al. 2016 (*PNAS*) — successive items at successive θ phases |
| Learnable constellation `nn.Parameter[64, 2]` | **Supported by analogy** | O'Shea & Hoydis 2017 (*IEEE TCCN*) — end-to-end learned constellation in physical layer ; Hoydis et al. 2019 (arXiv:1911.13055) — geometric constellation shaping ; **no published PyTorch PAC encoder/decoder** — novel contribution |
| Tort MI as validation metric (later) | **Nuanced** | Tort et al. 2010 (*J Neurophysiol*) — canonical ; Aru et al. 2015 + Hyafil 2015 — three warnings: non-stationarity, non-sinusoidal waveforms, common driver ; Cole & Voytek 2017 — 30-40% of reported PAC may be waveform artifact ; Marzulli et al. 2025 (*Front Hum Neurosci*) — MVL outperforms MI on high-SNR BCI ; He et al. 2026 (arXiv:2603.08866) — NARX alternative reduces low-freq power bias ; **recommend: use a shuffle-surrogate normalization + optionally NARX for biological validation** |
| PAC as cross-modal binding substrate | **Partially supported** | Senkowski et al. 2008 (*TINS*) — theoretical framework, correlational ; Misselhorn et al. 2019 (*eNeuro*) — causal for γ-γ synchrony (not PAC) ; Lennert et al. 2021 (*Commun Biol*) — direct PAC in audiovisual recalibration, but **alpha-gamma not theta-gamma** ; Lizarazu et al. 2023 (*HBM*) — θ/γ PAC in auditory cortex during phoneme processing (unimodal) ; bioRxiv 2026.01.20.700563 — cortico-hippocampal PAC ~8 Hz / γ in audiovisual associative memory ; **gap: no direct θ/γ PAC measured during cross-modal perception like bouba/kiki — bouba_sens positioning opportunity** |
| Bouba/kiki as evaluation benchmark | **Strengthened** | Ramachandran & Hubbard 2001 (*Proc R Soc B*) — revival ; Maurer et al. 2006 (*Dev Sci*) — present in 2.5y toddlers ; Ćwiek et al. 2022 (*Phil Trans B*) — 22/25 languages bouba, 11/25 kiki, 89% global congruence ; Peiffer-Smadja et al. 2019 (*NeuroImage*) — fMRI evidence in auditory+visual cortex ; Fort & Schwartz 2022 (*Sci Rep*) — acoustic-physical mechanism ; **Loconsole et al. 2026 (*Science*) — effect in 1–3-day-old naïve chicks, pre-wired not acoustic-learned** ; Ioannucci et al. bioRxiv 2026-01 — V1 decoding without visual input ; Zhao et al. arXiv:2603.17306 (2026-03) — LLM phoneme-meaning iconicity on 9 dimensions (baseline) |

---

## 3. Top 15 citations (prioritized for Paper 2 manuscript)

### Must-cite (core design rationale)

1. **Harris & Gong (2026-01-30)** — "Nested spatiotemporal theta–gamma waves organize hierarchical processing across the mouse visual cortex." *Nature Communications*. DOI: 10.1038/s41467-026-68893-4
2. **Huang, Bisby, Burgess & Bush (2026-02-27)** — "Human hippocampal theta–gamma coupling coordinates sequential planning during navigation." *PNAS* 123(9):e2513547123. DOI: 10.1073/pnas.2513547123
3. **Nour et al. (2024)** — "Control of working memory by phase–amplitude coupling of human hippocampal neurons." *Nature* 629:374–381. DOI: 10.1038/s41586-024-07309-z
4. **Tort, Komorowski, Eichenbaum & Kopell (2010)** — "Measuring phase-amplitude coupling between neuronal oscillations of different frequencies." *J Neurophysiol* 104:1195–1210. DOI: 10.1152/jn.00106.2010
5. **Lisman & Idiart (1995)** — "Storage of 7 ± 2 short-term memories in oscillatory subcycles." *Science* 267:1512–1515. DOI: 10.1126/science.7878473

### Cross-modal / bouba_sens rationale

6. **Loconsole, Benavides-Varela & Regolin (2026-02-19)** — "Matching sounds to shapes: Evidence of the bouba-kiki effect in naïve baby chicks." *Science* 391:836–839. DOI: 10.1126/science.adq7188
7. **Ioannucci, Jordan, Carnet, McGettigan & Vetter (2026-01-19)** — "Decoding the 'bouba-kiki' effect in early visual cortex." bioRxiv. DOI: 10.64898/2026.01.17.700088
8. **Anonymous authors (2026-01-20)** — "Cortico-hippocampal phase–amplitude coupling is a signature of learned audiovisual associations in humans." bioRxiv. DOI: 10.64898/2026.01.20.700563
9. **Ćwiek et al. (2022)** — "The bouba/kiki effect is robust across cultures and writing systems." *Phil Trans R Soc B* 377:20200390. DOI: 10.1098/rstb.2020.0390
10. **Lennert, Samiee & Baillet (2021)** — "Coupled oscillations enable rapid temporal recalibration to audiovisual asynchrony." *Commun Biol* 4:559. DOI: 10.1038/s42003-021-02087-0

### ML prior art (learnable constellation justification)

11. **O'Shea & Hoydis (2017)** — "An Introduction to Deep Learning for the Physical Layer." *IEEE Trans Cogn Commun Netw* 3:563–575. arXiv:1702.00832

### Methodological caveats

12. **Aru et al. (2015)** — "Untangling cross-frequency coupling in neuroscience." *Curr Opin Neurobiol* 31:51–61
13. **Hyafil (2015)** — "Misidentifications of specific forms of cross-frequency coupling: three warnings." *Front Neurosci* 9:370. DOI: 10.3389/fnins.2015.00370
14. **He et al. (2026-03-09)** — "A Dynamical Systems and System Identification Framework for Phase Amplitude Coupling Analysis." arXiv:2603.08866 [q-bio.NC, eess.SP]

### Hardware grounding

15. **Shrestha, Davies et al. (2023)** — "Efficient Spectral Signal Processing with Resonate-and-Fire Neurons on Intel's Loihi 2." *IEEE ICASSP 2023*. arXiv:2310.03251

### Supporting

- Jensen & Colgin 2007 (*TCS*); Canolty & Knight 2010 (*TCS*); Colgin 2016 (*NRN*); Pirazzini & Ursino 2022 (*Cogn Neurodyn*); Senkowski et al. 2008 (*TINS*); Misselhorn et al. 2019 (*eNeuro*); Keil & Senkowski 2018 (*Neuroscientist*); Lizarazu et al. 2023 (*HBM*); Fort & Schwartz 2022 (*Sci Rep*); Fries 2015 (*Neuron*); Chao et al. 2024 (*Commun Biol*); Gonzalez et al. 2024 arXiv:2401.04491 (SpiNNaker2); Moradi et al. 2025 (*Nat Commun*) NeuroScale; NeuroBench 2026 (*Nat Commun*).

---

## 4. Open empirical questions bouba_sens can answer

These questions are **not yet answered in the literature as of 2026-04-20** — they are legitimate targets for Paper 2:

- **Q1**: Does θ/γ PAC occur during direct cross-modal *perception* tasks (bouba/kiki), not only during learned association retrieval (bioRxiv 2026.01.20.700563)?
- **Q2**: If yes, what is the optimal θ phase for γ symbol placement in cross-modal binding? (Lisman-Idiart says 7 slots; does a shared cross-modal alphabet fit in 7?)
- **Q3**: Does a *learned* 64-symbol constellation (`nn.Parameter[64, 2]`) converge to the acoustic parameters identified by Fort & Schwartz 2022 (spectral balance + temporal continuity)? If so, it's a computational validation of their mechanism.
- **Q4**: Is PAC *necessary* or *incidental* for cross-modal binding? Ablation study: compare `GammaThetaMultiplexer` (PAC active) vs a CTC-only variant (γ-γ phase synchrony without θ envelope modulation — Fries 2015 framework).
- **Q5** (from nerve-wml issue #1, architectural): does `GammaThetaMultiplexer` **coexist** with `SimNerve.listen(phase=...)` boolean gating (α option), or **replace** it (β option)? The DSP paradigm and the discrete-gating paradigm are not formally equivalent.

---

## 5. Hardware feasibility snapshot (2026-04-20)

| Substrate | PAC γ/θ feasibility | Commercial availability Apr 2026 | Key reference |
|---|---|---|---|
| **Intel Loihi 2** | Indirect (RF neurons + graded spikes, no native PAC primitive) | Research only (Intel NRC) | Shrestha et al. 2023 ICASSP |
| **SpiNNaker2 / SpiNNcloud** | Indirect (full ARM programmability, jitter risk at 40 Hz) | **Commercial since Jan 2026** (UTSA, Sandia, Leipzig deployments) | Gonzalez et al. 2024 arXiv:2401.04491 |
| **IBM NorthPole** | Impossible (stateless DNN inferencer, no oscillator) | Partnership only, no public SDK | Modha et al. 2023 *Science* |
| **BrainChip Akida 2 / 3 preview** | Impossible (event-based edge, no phase-lock primitive) | Commercial, TENNs SDK | CES 2026 demo |
| **Memristor analog (Bi2Se3)** | Natural fit (analog oscillators) | Research only, horizon 2028+ | TechXplore Mar 2026 |
| **"Loihi 3"** | **DOES NOT OFFICIALLY EXIST** 2026-04-20 — circulating articles are promotional content without Intel source | N/A | Intel Newsroom CES 2026 (no neuromorphic content) |

**Recommended path**: PyTorch → lava-dl SLAYER → lava.lib.dl.netx → Loihi 2 via Intel NRC. SpiNNaker2 via sPyNNaker is the viable alternative with larger scale (5M ARM cores) but higher inter-core jitter.

---

## 6. Uncertainty ledger

- **ICLR 2026** proceedings (23-27 April 2026, Rio) not yet indexed at 2026-04-20 — 5300+ papers, manual check on OpenReview needed for late-breaking cross-modal/oscillation papers.
- **CogSci 2026** (July 2026, Rio) — submissions closed, proceedings not available.
- **bioRxiv 2026.01.20.700563** authorship not specified in available metadata — to verify before citing.
- **Tort MI 2026 alternatives** — NARX (He et al. arXiv:2603.08866) is promising but unvalidated at scale.
- **Loconsole et al. 2026 *Science*** — methodological details (n chicks, exact statistics) not verified on full PDF, only abstract + press coverage.
- **Aru et al. 2015** — primary journal not fully verified (cited via Hyafil 2015 + PMC 8004528).
- **Waveform-induced spurious PAC** (Cole & Voytek 2017) — the 30-40% figure is contested, no consensus.

---

## 7. Bibliographic references (BibTeX)

```bibtex
@article{HarrisGong2026,
  author = {Harris and Gong},
  title = {Nested spatiotemporal theta--gamma waves organize hierarchical processing across the mouse visual cortex},
  journal = {Nature Communications},
  year = {2026},
  doi = {10.1038/s41467-026-68893-4}
}

@article{Huang2026PNAS,
  author = {Huang and Bisby and Burgess and Bush},
  title = {Human hippocampal theta--gamma coupling coordinates sequential planning during navigation},
  journal = {PNAS},
  volume = {123},
  number = {9},
  pages = {e2513547123},
  year = {2026},
  doi = {10.1073/pnas.2513547123}
}

@article{Nour2024,
  author = {Nour and others},
  title = {Control of working memory by phase--amplitude coupling of human hippocampal neurons},
  journal = {Nature},
  volume = {629},
  pages = {374--381},
  year = {2024},
  doi = {10.1038/s41586-024-07309-z}
}

@article{Loconsole2026,
  author = {Loconsole and Benavides-Varela and Regolin},
  title = {Matching sounds to shapes: Evidence of the bouba-kiki effect in na\"ive baby chicks},
  journal = {Science},
  volume = {391},
  pages = {836--839},
  year = {2026},
  doi = {10.1126/science.adq7188}
}

@misc{Ioannucci2026,
  author = {Ioannucci and Jordan and Carnet and McGettigan and Vetter},
  title = {Decoding the "bouba-kiki" effect in early visual cortex},
  year = {2026},
  howpublished = {bioRxiv},
  doi = {10.64898/2026.01.17.700088}
}

@misc{PAC_audiovisual_2026,
  title = {Cortico-hippocampal phase--amplitude coupling is a signature of learned audiovisual associations in humans},
  year = {2026},
  howpublished = {bioRxiv},
  doi = {10.64898/2026.01.20.700563}
}

@article{TortEtAl2010,
  author = {Tort and Komorowski and Eichenbaum and Kopell},
  title = {Measuring phase-amplitude coupling between neuronal oscillations of different frequencies},
  journal = {Journal of Neurophysiology},
  volume = {104},
  pages = {1195--1210},
  year = {2010},
  doi = {10.1152/jn.00106.2010}
}

@article{LismanIdiart1995,
  author = {Lisman and Idiart},
  title = {Storage of 7 \pm 2 short-term memories in oscillatory subcycles},
  journal = {Science},
  volume = {267},
  pages = {1512--1515},
  year = {1995},
  doi = {10.1126/science.7878473}
}

@article{CwiekEtAl2022,
  author = {\'Cwiek and others},
  title = {The bouba/kiki effect is robust across cultures and writing systems},
  journal = {Phil Trans R Soc B},
  volume = {377},
  pages = {20200390},
  year = {2022},
  doi = {10.1098/rstb.2020.0390}
}

@article{OSheaHoydis2017,
  author = {O'Shea and Hoydis},
  title = {An Introduction to Deep Learning for the Physical Layer},
  journal = {IEEE Trans Cogn Commun Netw},
  volume = {3},
  pages = {563--575},
  year = {2017},
  doi = {10.1109/TCCN.2017.2758370},
  note = {arXiv:1702.00832}
}

@misc{HeEtAl2026,
  author = {He and others},
  title = {A Dynamical Systems and System Identification Framework for Phase Amplitude Coupling Analysis},
  year = {2026},
  howpublished = {arXiv:2603.08866}
}
```
