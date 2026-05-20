"""Renf 10: spectral_entropy robustness to batch size B.

Renf 7 fixed B=128. This script sweeps B in {64,128,256,512} and checks
whether the ordering null < akorn_best < gtm < simple_gating holds at
each B. If the ordering inverts somewhere, the metric is B-fragile.

Reuses the training paths from `synchrony_replacement_eval` but
parametrises the (B, 7) codes tensor.

Output JSON shape::

    {
      "batch_sizes": [64, 128, 256, 512],
      "seeds": N,
      "steps": S,
      "per_batch": {
        "64": {
          "aggregated": {arm: {metric: {mean, std, values}}},
          "paired":     {metric: {arm_a_vs_arm_b: {p_value, median_diff}}},
          "ordering":   {metric: [low..high]},
        }, ...
      },
      "wall_clock_s": float,
    }
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from torch.nn import functional as F  # noqa: N812
from torch.optim import Adam

from scripts.synchrony_alternatives import (
    effective_rank,
    participation_ratio,
    spectral_entropy,
    top3_dispersion,
)
from track_p.multiplexer import GammaThetaConfig, GammaThetaMultiplexer
from track_p.transducer_baselines import (
    KuramotoMultiplexer,
    SimpleGatingMultiplexer,
)

_METRICS = {
    "spectral_entropy":    spectral_entropy,
    "participation_ratio": participation_ratio,
    "effective_rank":      effective_rank,
    "top3_dispersion":     top3_dispersion,
}
_ARMS = ["gtm", "simple_gating", "akorn_best", "null"]


def _train_gtm(codes, steps, seed, device):
    cfg = GammaThetaConfig(symbols_per_theta=codes.shape[-1])
    g = GammaThetaMultiplexer(cfg=cfg, seed=seed).to(device)
    opt = Adam(g.parameters(), lr=0.02)
    for _ in range(steps):
        opt.zero_grad()
        c = g.forward(codes)
        soft = g.demodulate(c, hard=False, tau=1.0)
        loss = F.cross_entropy(
            torch.log(soft + 1e-9).reshape(-1, soft.shape[-1]),
            codes.reshape(-1),
        )
        loss.backward()
        opt.step()
        g.step()
    with torch.no_grad():
        return g.forward(codes).detach().cpu()


def _train_simple(codes, steps, device):
    m = SimpleGatingMultiplexer(alphabet_size=64, n_symbols=codes.shape[-1]).to(device)
    opt = Adam(m.parameters(), lr=0.02)
    for _ in range(steps):
        opt.zero_grad()
        c = m.forward(codes)
        logits = m.demodulate_logits(c)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), codes.reshape(-1))
        loss.backward()
        opt.step()
    with torch.no_grad():
        return m.forward(codes).detach().cpu()


def _train_akorn(codes, steps, device):
    m = KuramotoMultiplexer(
        alphabet_size=64, n_symbols=codes.shape[-1],
        n_oscillators=64, n_steps=32,
    ).to(device)
    opt = Adam(m.parameters(), lr=0.05)
    for _ in range(steps):
        opt.zero_grad()
        c = m.forward(codes)
        logits = m.demodulate_logits(c)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), codes.reshape(-1))
        loss.backward()
        opt.step()
    with torch.no_grad():
        return m.forward(codes).detach().cpu()


def _train_null(codes, steps, seed, device):
    rng = torch.Generator(); rng.manual_seed(seed + 9973)
    perm = torch.randperm(codes.shape[0], generator=rng).to(device)
    shuffled = codes[perm]
    cfg = GammaThetaConfig(symbols_per_theta=codes.shape[-1])
    g = GammaThetaMultiplexer(cfg=cfg, seed=seed).to(device)
    opt = Adam(g.parameters(), lr=0.02)
    for _ in range(steps):
        opt.zero_grad()
        c = g.forward(shuffled)
        soft = g.demodulate(c, hard=False, tau=1.0)
        loss = F.cross_entropy(
            torch.log(soft + 1e-9).reshape(-1, soft.shape[-1]),
            codes.reshape(-1),
        )
        loss.backward()
        opt.step()
        g.step()
    with torch.no_grad():
        return g.forward(shuffled).detach().cpu()


def run_one(args):
    seed, B, steps, device = args
    torch.manual_seed(seed)
    codes_cpu = torch.randint(0, 64, (B, 7))
    codes = codes_cpu.to(device)
    carriers = {
        "gtm":           _train_gtm(codes, steps, seed, device),
        "simple_gating": _train_simple(codes, steps, device),
        "akorn_best":    _train_akorn(codes, steps, device),
        "null":          _train_null(codes, steps, seed, device),
    }
    return seed, B, {arm: {m: fn(c) for m, fn in _METRICS.items()}
                     for arm, c in carriers.items()}


def _aggregate(results_for_B):
    agg = {arm: {m: {"values": []} for m in _METRICS} for arm in _ARMS}
    for _, _, row in results_for_B:
        for arm in _ARMS:
            for m in _METRICS:
                agg[arm][m]["values"].append(row[arm][m])
    for arm in _ARMS:
        for m in _METRICS:
            v = np.asarray(agg[arm][m]["values"])
            agg[arm][m]["mean"] = float(v.mean())
            agg[arm][m]["std"]  = float(v.std())

    paired = {m: {} for m in _METRICS}
    for m in _METRICS:
        for a in _ARMS:
            for b in _ARMS:
                if a == b:
                    continue
                va = np.asarray(agg[a][m]["values"])
                vb = np.asarray(agg[b][m]["values"])
                if np.allclose(va, vb):
                    paired[m][f"{a}_vs_{b}"] = {"p_value": 1.0, "median_diff": 0.0}
                    continue
                try:
                    r = stats.wilcoxon(va, vb, alternative="two-sided", zero_method="wilcox")
                    pv = float(r.pvalue)
                except ValueError:
                    pv = 1.0
                paired[m][f"{a}_vs_{b}"] = {
                    "p_value": pv,
                    "median_diff": float(np.median(va - vb)),
                }
    ordering = {
        m: sorted(_ARMS, key=lambda a: agg[a][m]["mean"]) for m in _METRICS
    }
    return {"aggregated": agg, "paired": paired, "ordering": ordering}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--batch_sizes", type=int, nargs="+", default=[64, 128, 256, 512])
    p.add_argument("--seeds", type=int, default=50)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--device", default="cpu", choices=("cpu", "mps", "cuda"))
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    if args.device != "cpu":
        args.workers = 1

    t0 = time.perf_counter()
    tasks = [(s, B, args.steps, args.device)
             for B in args.batch_sizes for s in range(args.seeds)]
    workers = args.workers or min(len(tasks), mp.cpu_count())
    print(f"running {len(tasks)} jobs ({len(args.batch_sizes)} B x "
          f"{args.seeds} seeds x 4 arms) on {workers} workers")
    if workers > 1:
        with mp.Pool(workers) as pool:
            results = pool.map(run_one, tasks)
    else:
        results = [run_one(t) for t in tasks]

    per_batch = {}
    for B in args.batch_sizes:
        per_batch[str(B)] = _aggregate([r for r in results if r[1] == B])

    out = {
        "batch_sizes": args.batch_sizes,
        "seeds": args.seeds,
        "steps": args.steps,
        "per_batch": per_batch,
        "wall_clock_s": time.perf_counter() - t0,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out} in {out['wall_clock_s']:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
