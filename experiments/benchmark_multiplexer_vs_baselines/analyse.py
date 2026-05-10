"""Q1 Task 15 — Welch's t-test + Bonferroni + figure rendering.

Loads results.json (T13) and optionally ablation_results.json (T14),
computes per-metric pairwise t-tests of GTM vs each baseline,
applies Bonferroni correction (alpha = 0.05/9 = 0.00556), renders
a comparison figure, prints the verdict per pre-registration.

Plan: HYPNEUM-PLANS/2026-05-10-niveau8-three-experiments.md Task 15.
Pre-reg: docs/milestones/q1-multiplexer-benchmark-2026-05-10.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st


METRICS = ["round_trip_fidelity", "mi_h", "bandwidth_efficiency"]
BASELINES = ["RecursiveLinkBridge", "MLPBridge", "CrossAttentionBridge"]
GTM_NAME = "GTMBridge"
ALPHA_RAW = 0.05
N_COMPARISONS = len(METRICS) * len(BASELINES)  # 9
ALPHA_CORRECTED = ALPHA_RAW / N_COMPARISONS    # 0.00556


def load_results(path: Path) -> dict[str, dict[str, list[float]]]:
    """Load runner results.json into {arch: {metric: [vals_per_seed]}}."""
    raw = json.loads(path.read_text())
    out: dict[str, dict[str, list[float]]] = {}
    for r in raw["results"]:
        arch = r["arch"]
        out.setdefault(arch, {m: [] for m in METRICS + ["n_params", "train_time_s"]})
        for m in METRICS:
            out[arch][m].append(r[m])
        out[arch]["n_params"].append(r["n_params"])
        out[arch]["train_time_s"].append(r["train_time_s"])
    return out


def welch_test(gtm_vals: list[float], base_vals: list[float]) -> tuple[float, float]:
    stat, p = st.ttest_ind(gtm_vals, base_vals, equal_var=False)
    return float(stat), float(p)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("experiments/benchmark_multiplexer_vs_baselines/results.json"),
    )
    parser.add_argument(
        "--ablation",
        type=Path,
        default=Path("experiments/benchmark_multiplexer_vs_baselines/ablation_results.json"),
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("papers/paper2/figures/multiplexer_benchmark.png"),
    )
    args = parser.parse_args()

    results = load_results(args.results)
    ablation = load_results(args.ablation) if args.ablation.exists() else None

    # ===== Welch's t-test pairwise =====
    print(f"=== Q1 Welch's t-test (alpha_corrected = {ALPHA_CORRECTED:.5f}) ===\n")
    wins = {b: 0 for b in BASELINES}
    losses = {b: 0 for b in BASELINES}
    ties = {b: 0 for b in BASELINES}
    test_records: list[dict] = []

    if GTM_NAME not in results:
        print(f"ERROR: {GTM_NAME} not in results", file=sys.stderr)
        return 2

    for baseline in BASELINES:
        if baseline not in results:
            print(f"WARNING: {baseline} missing from results, skipping", file=sys.stderr)
            continue
        for metric in METRICS:
            gtm_vals = results[GTM_NAME][metric]
            base_vals = results[baseline][metric]
            stat, p = welch_test(gtm_vals, base_vals)
            gtm_mean = float(np.mean(gtm_vals))
            base_mean = float(np.mean(base_vals))
            sig = p < ALPHA_CORRECTED
            outcome = "tie"
            if sig:
                outcome = "win" if gtm_mean > base_mean else "loss"
            if outcome == "win":
                wins[baseline] += 1
            elif outcome == "loss":
                losses[baseline] += 1
            else:
                ties[baseline] += 1
            test_records.append({
                "baseline": baseline, "metric": metric,
                "gtm_mean": gtm_mean, "baseline_mean": base_mean,
                "t_stat": stat, "p": p, "significant": sig,
                "outcome_for_gtm": outcome,
            })
            print(
                f"  {baseline:25s} {metric:25s} "
                f"t={stat:+6.2f} p={p:.4g} "
                f"GTM={gtm_mean:.4f} base={base_mean:.4f} -> {outcome}"
            )

    total_wins = sum(wins.values())
    total_losses = sum(losses.values())
    total_ties = sum(ties.values())
    print(f"\nGTM wins: {total_wins}/9, losses: {total_losses}/9, ties: {total_ties}/9")

    # ===== Verdict per pre-registration =====
    if total_wins >= 6:
        verdict = "GTM_headline"
        narrative = (
            "GTM headline: GTM wins >=6/9 comparisons (>=2/3 metrics across all 3 baselines). "
            "Paper 2 section X.Y leads with the benchmark; GTM positioned as state-of-the-art bridge."
        )
    elif total_losses >= 6:
        verdict = "GTM_loses"
        narrative = (
            "GTM loses: GTM loses >=6/9 comparisons. Paper 2 section X.Y reframes as "
            "falsifiability-scope: PAC multiplexing is task-dependent, latent-space methods "
            "dominate HardFlowProxyTask N=2."
        )
    else:
        verdict = "tied"
        narrative = (
            "Tied (3-5 wins or ties dominate). Paper 2 section X.Y reframes as "
            "convergent evidence: GTM matches latent-space baselines on round-trip while "
            "preserving phase-coupled biological plausibility, with stronger MI per code unit."
        )
    print(f"\nVerdict: {verdict}")
    print(narrative)

    # ===== Render figure =====
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    archs = [GTM_NAME] + BASELINES
    abl_keys: list[str] = []
    if ablation is not None:
        abl_keys = sorted(ablation.keys())

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    colors_main = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd"]
    colors_abl = ["#ff9896", "#aec7e8", "#98df8a"]

    for ax, metric in zip(axes, METRICS):
        labels = []
        means = []
        sems = []
        bar_colors = []
        for i, arch in enumerate(archs):
            if arch not in results:
                continue
            vals = results[arch][metric]
            labels.append(arch.replace("Bridge", ""))
            means.append(float(np.mean(vals)))
            sems.append(float(np.std(vals, ddof=1) / np.sqrt(len(vals))))
            bar_colors.append(colors_main[i])
        for j, abl in enumerate(abl_keys):
            if metric not in ablation[abl]:
                continue
            vals = ablation[abl][metric]
            labels.append(abl.replace("Bridge", ""))
            means.append(float(np.mean(vals)))
            sems.append(float(np.std(vals, ddof=1) / np.sqrt(len(vals))))
            bar_colors.append(colors_abl[j % len(colors_abl)])

        x = np.arange(len(labels))
        ax.bar(x, means, yerr=sems, color=bar_colors, capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_title(metric)
        ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    fig.suptitle(
        f"Q1 - GTM benchmark on HardFlowProxyTask (5 seeds, "
        f"Bonferroni alpha={ALPHA_CORRECTED:.4f})"
    )
    plt.tight_layout()
    plt.savefig(args.figure, dpi=120)
    print(f"\nWrote figure: {args.figure}")

    # ===== Persist verdict JSON for downstream Paper 2 update =====
    verdict_json = args.figure.with_name("q1_verdict.json")
    verdict_json.write_text(json.dumps({
        "verdict": verdict,
        "narrative": narrative,
        "wins": wins, "losses": losses, "ties": ties,
        "total_wins": total_wins, "total_losses": total_losses, "total_ties": total_ties,
        "alpha_corrected": ALPHA_CORRECTED,
        "tests": test_records,
        "has_ablation": ablation is not None,
    }, indent=2))
    print(f"Wrote verdict: {verdict_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
