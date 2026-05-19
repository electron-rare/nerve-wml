"""Comparison baselines for the learned :class:`track_p.transducer.Transducer`.

Closes gap 1 of the validation suite. The learned Transducer is a free
[64x64] logits matrix; on its own it has no point of comparison. This module
provides three alternative src->dst code maps, each consuming the two WML
codebooks (`MlpWML.codebook` etc.) and exposing a `forward(src_code) -> dst_code`
that returns a `[B]` long tensor of dst indices, so all four can be plugged
into the same code-translation benchmark (`scripts/transducer_baselines_pilot.py`).

- ProcrustesTransducer  -- orthogonal Procrustes map (Maystre et al. 2025,
  arXiv:2510.13406). Closed-form SVD solution, supervised by code index.
- RelativeRepTransducer -- anchor-based cosine encoding, zero-shot, no fit
  (Moschella et al., ICLR 2023, arXiv:2209.15430).
- Vec2VecTransducer     -- unsupervised GAN + cycle-consistency translation,
  no paired data (Jha et al. 2025, arXiv:2505.12540).
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812


class ProcrustesTransducer(nn.Module):
    """Orthogonal Procrustes src->dst code map.

    Fits the orthogonal matrix `R` minimising `||src @ R - dst||_F` over the
    paired codebook rows (pairing = shared code index), via the SVD of
    `dst^T @ src`. At inference, a src code is mapped by projecting its src
    embedding through `R` and taking the nearest dst codebook row.

    Parameters
    ----------
    src_codebook, dst_codebook
        `[alphabet_size, D]` float tensors. D must match between the two.
    """

    rotation: Tensor
    _dst_codebook: Tensor
    _src_codebook: Tensor

    def __init__(self, src_codebook: Tensor, dst_codebook: Tensor) -> None:
        super().__init__()
        if src_codebook.shape != dst_codebook.shape:
            raise ValueError(
                f"codebook shape mismatch: {src_codebook.shape} "
                f"vs {dst_codebook.shape}"
            )
        self.alphabet_size = src_codebook.shape[0]
        src = src_codebook.detach().to(torch.float64)
        dst = dst_codebook.detach().to(torch.float64)
        # Orthogonal Procrustes: R = U V^T from SVD of dst^T @ src.
        u, _, vh = torch.linalg.svd(dst.T @ src)
        rotation = (u @ vh).to(torch.float32)
        self.register_buffer("rotation", rotation)
        self.register_buffer("_dst_codebook", dst_codebook.detach().clone())
        self.register_buffer("_src_codebook", src_codebook.detach().clone())

    def forward(self, src_code: Tensor) -> Tensor:
        """Map `[B]` long src codes to `[B]` long dst codes."""
        src_emb = self._src_lookup(src_code)  # [B, D]
        projected = src_emb @ self.rotation  # [B, D]
        dist = torch.cdist(projected, self._dst_codebook)  # [B, alphabet]
        return dist.argmin(dim=-1)

    def _src_lookup(self, src_code: Tensor) -> Tensor:
        return self._src_codebook[src_code]


class RelativeRepTransducer(nn.Module):
    """Anchor-based relative-representation src->dst map (zero-shot).

    Moschella et al. (ICLR 2023): encode each codebook row by its vector of
    cosine similarities to a fixed set of `n_anchors` anchor rows. Because the
    anchors are shared by code index across src and dst, the relative encoding
    is invariant to any rotation/reflection between the two latent spaces, so
    no fitting is needed. A src code maps to the dst code whose relative
    encoding is closest (cosine).

    Parameters
    ----------
    src_codebook, dst_codebook
        `[alphabet_size, D]` float tensors, same shape.
    n_anchors
        Number of anchor codes (sampled without replacement from the alphabet).
    seed
        RNG seed for anchor selection.
    """

    src_rel: Tensor
    dst_rel: Tensor

    def __init__(
        self,
        src_codebook: Tensor,
        dst_codebook: Tensor,
        *,
        n_anchors: int = 32,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if src_codebook.shape != dst_codebook.shape:
            raise ValueError(
                f"codebook shape mismatch: {src_codebook.shape} "
                f"vs {dst_codebook.shape}"
            )
        alphabet = src_codebook.shape[0]
        if not 1 <= n_anchors <= alphabet:
            raise ValueError(f"n_anchors must be in [1, {alphabet}]")
        gen = torch.Generator()
        gen.manual_seed(seed)
        anchors = torch.randperm(alphabet, generator=gen)[:n_anchors]
        src = src_codebook.detach()
        dst = dst_codebook.detach()
        self.register_buffer("src_rel", self._relative(src, anchors))
        self.register_buffer("dst_rel", self._relative(dst, anchors))

    @staticmethod
    def _relative(codebook: Tensor, anchors: Tensor) -> Tensor:
        """`[alphabet, n_anchors]` cosine-similarity encoding."""
        normed = F.normalize(codebook, dim=-1)
        anchor_vecs = normed[anchors]  # [n_anchors, D]
        return normed @ anchor_vecs.T  # [alphabet, n_anchors]

    def forward(self, src_code: Tensor) -> Tensor:
        """Map `[B]` long src codes to `[B]` long dst codes."""
        query = F.normalize(self.src_rel[src_code], dim=-1)  # [B, n_anchors]
        keys = F.normalize(self.dst_rel, dim=-1)  # [alphabet, n_anchors]
        sim = query @ keys.T  # [B, alphabet]
        return sim.argmax(dim=-1)
