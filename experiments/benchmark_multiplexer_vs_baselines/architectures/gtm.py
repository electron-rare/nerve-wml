"""GTMBridge — wraps track_p.multiplexer.GammaThetaMultiplexer.

GTM's native forward signature operates on long-typed code tensors
[B, K] and returns a [B, T] PAC carrier waveform — it is not a
[B, dim] -> [B, code_dim] map. To bridge HardFlowProxyTask's
[B, dim] continuous tensors to GTM's symbol-code interface, this
wrapper uses learned linear projections on each side (encode_proj,
decode_proj) — the same trick RecursiveLinkBridge applies. The
underlying GammaThetaMultiplexer module is instantiated and held
on the bridge so its parameters (constellation, plasticity hooks)
are visible to optimisers ; full wire-through of the symbol-code
path is deferred to Task 13 where the runner can quantise codes
and pass them through GTM.forward proper.
"""
from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from track_p.multiplexer import GammaThetaConfig, GammaThetaMultiplexer


class GTMBridge(nn.Module):
    """GTM with input/output linear adapters for [B, dim] tensors."""

    def __init__(
        self,
        dim: int = 16,
        code_dim: int = 16,
        *,
        seed: int | None = None,
        cfg: GammaThetaConfig | None = None,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.code_dim = code_dim
        self.encode_proj = nn.Linear(dim, code_dim)
        self.decode_proj = nn.Linear(code_dim, dim)
        self.gtm = GammaThetaMultiplexer(cfg=cfg, seed=seed)

    def encode(self, x: Tensor) -> Tensor:
        return self.encode_proj(x)

    def decode(self, code: Tensor) -> Tensor:
        return self.decode_proj(code)

    def forward(self, x: Tensor) -> Tensor:
        return self.decode(self.encode(x))
