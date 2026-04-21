"""Frozen-encoder baseline for nerve-wml Claim B ablation (review F3).

Isolates the contribution of the nerve-wml VQ protocol from the trivial
alignment expected of any two substrates sharing a task and an encoder.
Two substrates (MLP-head and LIF-head) are trained as linear classifiers
on top of a single FROZEN random encoder. Their output codes (argmax
over 12-class logits) are compared via plug-in MI/H(a). If the resulting
MI/H(a) is comparable to nerve-wml's Test (1) figure of 0.91-0.96, the
claim that nerve-wml's VQ protocol contributes to substrate-agnostic
transmission must be softened.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from track_w.tasks.hard_flow_proxy import HardFlowProxyTask


class _FrozenEncoder(nn.Module):
    """Random 2-layer MLP with frozen parameters, input_dim -> d_hidden."""

    def __init__(self, input_dim: int, d_hidden: int, seed: int) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.net = nn.Sequential(
            nn.Linear(input_dim, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden),
        )
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_frozen_baseline(
    seed: int = 0,
    steps: int = 800,
    d_hidden: int = 16,
    return_encoder: bool = False,
) -> dict[str, Any]:
    """Train two trainable heads on a frozen shared encoder.

    Returns a dict with acc_mlp, acc_lif, codes_mlp, codes_lif.
    If return_encoder=True, also includes encoder_initial and
    encoder_final for verifying the encoder didn't change.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    task = HardFlowProxyTask(dim=16, n_classes=12, seed=seed)
    encoder = _FrozenEncoder(input_dim=16, d_hidden=d_hidden, seed=seed + 100)
    encoder_initial = copy.deepcopy(encoder) if return_encoder else None

    head_a = nn.Linear(d_hidden, 12)
    head_b = nn.Linear(d_hidden, 12)
    opt = torch.optim.Adam(
        list(head_a.parameters()) + list(head_b.parameters()),
        lr=1e-2,
    )

    for _ in range(steps):
        x, y = task.sample(batch=64)
        with torch.no_grad():
            z = encoder(x)
        logits_a = head_a(z)
        logits_b = head_b(z)
        loss = F.cross_entropy(logits_a, y) + F.cross_entropy(logits_b, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    eval_task = HardFlowProxyTask(dim=16, n_classes=12, seed=seed)
    x_eval, y_eval = eval_task.sample(batch=1000)
    with torch.no_grad():
        z_eval = encoder(x_eval)
        pred_a = head_a(z_eval).argmax(-1)
        pred_b = head_b(z_eval).argmax(-1)
        acc_mlp = (pred_a == y_eval).float().mean().item()
        acc_lif = (pred_b == y_eval).float().mean().item()

    result: dict[str, Any] = {
        "acc_mlp":   acc_mlp,
        "acc_lif":   acc_lif,
        "codes_mlp": pred_a.cpu().numpy().astype(np.int64),
        "codes_lif": pred_b.cpu().numpy().astype(np.int64),
    }
    if return_encoder:
        result["encoder_initial"] = encoder_initial
        result["encoder_final"] = encoder
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("papers/paper1/figures/baseline_frozen_encoder.json"),
    )
    args = parser.parse_args()

    from nerve_wml.methodology import mi_plugin_discrete, null_model_mi

    per_seed = []
    for s in args.seeds:
        r = train_frozen_baseline(seed=s, steps=args.steps)
        mi = mi_plugin_discrete(r["codes_mlp"], r["codes_lif"])
        nm = null_model_mi(
            r["codes_mlp"], r["codes_lif"], n_shuffles=1000, seed=s,
        )
        per_seed.append({
            "seed":      s,
            "acc_mlp":   r["acc_mlp"],
            "acc_lif":   r["acc_lif"],
            "mi_plugin": mi,
            "null_z":    nm.z_score,
            "null_p":    nm.p_value,
        })

    summary = {
        "mi_plugin_mean": float(np.mean([r["mi_plugin"] for r in per_seed])),
        "acc_mlp_mean":   float(np.mean([r["acc_mlp"] for r in per_seed])),
        "acc_lif_mean":   float(np.mean([r["acc_lif"] for r in per_seed])),
        "nerve_wml_comparison":
            "nerve-wml Test (1) reported MI/H = 0.91-0.96 on the same task. "
            "If this frozen-encoder baseline reaches similar MI/H, the VQ "
            "protocol contribution to Claim B must be reformulated.",
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"per_seed": per_seed, "summary": summary}, indent=2),
    )
    print(f"mean MI/H: {summary['mi_plugin_mean']:.4f}")
    print(f"mean acc MLP/LIF: {summary['acc_mlp_mean']:.4f} / {summary['acc_lif_mean']:.4f}")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
