"""Surrogate-gradient spike function for LifWML.

Forward: heaviside step at v_thr.
Backward: fast-sigmoid derivative α / (π · (1 + (α·(v − v_thr))²)).

See spec §7.5 and Neftci et al. 2019.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.autograd import Function


class _SpikeFn(Function):
    @staticmethod
    def forward(ctx, v: Tensor, v_thr: float, alpha: float) -> Tensor:  # type: ignore[override]
        ctx.save_for_backward(v)
        ctx.v_thr = v_thr
        ctx.alpha = alpha
        return (v > v_thr).float()

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor, None, None]:  # type: ignore[override]
        (v,) = ctx.saved_tensors
        alpha = ctx.alpha
        deriv = alpha / (math.pi * (1 + (alpha * (v - ctx.v_thr)) ** 2))
        return grad_output * deriv, None, None


def spike_with_surrogate(v: Tensor, v_thr: float = 1.0, alpha: float = 2.0) -> Tensor:
    """Heaviside spike with a differentiable backward.

    Args:
        v:     membrane potential tensor.
        v_thr: firing threshold.
        alpha: surrogate sharpness (higher = closer to true step).
    """
    return _SpikeFn.apply(v, v_thr, alpha)  # type: ignore[no-any-return]
