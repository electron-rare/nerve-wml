"""Scale-robustness check for the shared-info / substrate-gap measurements.

Closes gap 3 (scale arm). PRH critiques (Huh et al. 2024) show that
representation-alignment scores can degrade as sample size grows. This runner
emits the same WML codebook embeddings through two substrates (an MlpWML and a
LifWML), then recomputes debiased HSIC and CKNNA at increasing subsample sizes
via `nerve_wml.methodology.hsic_cknna.scale_robustness_sweep`. A stable
alignment shows flat curves; a biased one shows the score drifting with N.

Run directly:  uv run python -m scripts.scale_robustness_pilot
"""
from __future__ import annotations

import torch

from nerve_wml.methodology.hsic_cknna import scale_robustness_sweep
from track_w.lif_wml import LifWML
from track_w.mlp_wml import MlpWML


def _substrate_embeddings(seed: int, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Two `[n, D]` embedding clouds from an MlpWML and a LifWML codebook.

    Each row is the codebook embedding of a randomly drawn code; the two
    substrates share the code stream so the pair is aligned by construction.
    """
    torch.manual_seed(seed)
    mlp = MlpWML(id=0, alphabet_size=64, seed=seed)
    lif = LifWML(id=1, alphabet_size=64, seed=seed + 1)
    codes = torch.randint(0, 64, (n,))
    mlp_emb = mlp.codebook.detach()[codes]
    lif_emb = lif.codebook.detach()[codes]
    return mlp_emb, lif_emb


def run_scale_robustness(
    *, sizes: tuple[int, ...] = (64, 128, 256, 512), seed: int = 0
) -> list[dict[str, float]]:
    """Run the HSIC/CKNNA sweep on real substrate embeddings."""
    n_max = max(sizes)
    mlp_emb, lif_emb = _substrate_embeddings(seed, n_max)
    rows = scale_robustness_sweep(
        mlp_emb.cpu().numpy(),
        lif_emb.cpu().numpy(),
        sizes=sizes,
        k=10,
        seed=seed,
    )
    return [
        {"n": float(r.n), "hsic": r.hsic, "cknna": r.cknna} for r in rows
    ]


def main() -> None:
    rows = run_scale_robustness(sizes=(64, 128, 256, 512), seed=0)
    print(f"{'N':>8}{'HSIC':>14}{'CKNNA':>10}")
    for r in rows:
        print(f"{int(r['n']):>8}{r['hsic']:>14.5f}{r['cknna']:>10.3f}")


if __name__ == "__main__":
    main()
