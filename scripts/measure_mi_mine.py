"""MINE MI estimator on pre-VQ continuous embeddings.

Companion to measure_mi_multi_estimator.py: provides a fourth MI
estimator (Belghazi et al. 2018 MINE, Donsker-Varadhan bound) on
the same continuous embeddings the Kraskov KSG operates on. If MINE
and KSG agree in magnitude (within factor 2), the pre-VQ MI
measurement is robust under two independent continuous estimators
(kNN density vs variational neural network bound).

GPU optional -- MINE training is light (500 epochs x batch 256 on
d=16 critic fits in a few seconds on CPU). On kxkm-ai the RTX 4090
is fine but not required.

Usage:
    uv run python scripts/measure_mi_mine.py \\
        --codes tests/golden/codes_mlp_lif.npz \\
        --seeds 0 1 2 --n-epochs 500 --n-samples 2000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from nerve_wml.methodology.mi_estimators import entropy_discrete
from nerve_wml.methodology.mi_mine_estimator import mi_mine


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codes", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--n-epochs", type=int, default=500)
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("papers/paper1/figures/mi_mine.json"),
    )
    args = parser.parse_args()

    if not args.codes.exists():
        raise FileNotFoundError(
            f"{args.codes} not found. Produce it via "
            "scripts/save_codes_for_checks.py."
        )

    data = np.load(args.codes)
    if "mlp_embeddings" not in data or "lif_embeddings" not in data:
        raise KeyError(
            "NPZ lacks pre-VQ embeddings. Regenerate with the updated "
            "save_codes_for_checks.py."
        )
    mlp_codes = data["mlp_codes"]
    mlp_emb = data["mlp_embeddings"]
    lif_emb = data["lif_embeddings"]

    per_seed = []
    for seed_idx, s in enumerate(args.seeds):
        emb_a = mlp_emb[seed_idx].astype(np.float32)
        emb_b = lif_emb[seed_idx].astype(np.float32)
        codes_a = mlp_codes[seed_idx].astype(np.int64)

        rng = np.random.default_rng(s)
        if emb_a.shape[0] > args.n_samples:
            idx = rng.choice(
                emb_a.shape[0], size=args.n_samples, replace=False,
            )
            emb_a_sub = emb_a[idx]
            emb_b_sub = emb_b[idx]
        else:
            emb_a_sub = emb_a
            emb_b_sub = emb_b

        print(f"seed {s}: MINE training ({args.n_epochs} epochs)...")
        mi_nats = mi_mine(
            emb_a_sub, emb_b_sub,
            hidden=args.hidden,
            n_epochs=args.n_epochs,
            batch_size=args.batch_size,
            lr=1e-3,
            seed=s,
            tail_average=50,
            device=args.device,
        )
        h_a_nats = entropy_discrete(codes_a)
        mi_over_h_a = mi_nats / h_a_nats if h_a_nats > 0 else 0.0

        per_seed.append({
            "seed":            s,
            "mi_mine_nats":    mi_nats,
            "mi_mine_over_h_a": mi_over_h_a,
            "h_a_nats":         h_a_nats,
            "n_epochs":         args.n_epochs,
            "n_samples":        min(emb_a.shape[0], args.n_samples),
        })
        print(
            f"  mi_mine={mi_nats:.4f} nats, "
            f"mi_mine/H(a)={mi_over_h_a:.4f}"
        )

    mean_nats = float(np.mean([r["mi_mine_nats"] for r in per_seed]))
    mean_norm = float(np.mean([r["mi_mine_over_h_a"] for r in per_seed]))

    summary = {
        "mi_mine_nats_mean":    mean_nats,
        "mi_mine_over_h_a_mean": mean_norm,
        "n_seeds":               len(args.seeds),
        "n_epochs":              args.n_epochs,
        "n_samples":             args.n_samples,
        "hidden":                args.hidden,
        "batch_size":            args.batch_size,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"per_seed": per_seed, "summary": summary}, indent=2)
    )

    print()
    print(
        f"MINE -- {len(args.seeds)} seeds, {args.n_epochs} epochs, "
        f"hidden={args.hidden}"
    )
    print(f"  mean MI_mine (nats):     {mean_nats:.4f}")
    print(f"  mean MI_mine / H(a):     {mean_norm:.4f}")
    print()
    print(f"Output: {args.out}")


if __name__ == "__main__":
    main()
