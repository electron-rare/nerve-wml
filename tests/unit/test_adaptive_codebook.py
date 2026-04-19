"""Tests for AdaptiveCodebook skeleton."""
import torch

from track_p.adaptive_codebook import AdaptiveCodebook


def test_initial_size_is_full_alphabet():
    cb = AdaptiveCodebook(size=64, dim=32)
    assert cb.current_size() == 64
    assert cb.active_mask.all()


def test_active_indices_returns_long_tensor():
    cb = AdaptiveCodebook(size=16, dim=8)
    idx = cb.active_indices()
    assert idx.dtype == torch.long
    assert idx.shape == (16,)


def test_active_embeddings_matches_mask():
    cb = AdaptiveCodebook(size=16, dim=8)
    cb.active_mask[4:] = False
    emb = cb.active_embeddings()
    assert emb.shape == (4, 8)


def test_quantize_active_returns_indices_in_storage_space():
    cb = AdaptiveCodebook(size=16, dim=8)
    # Retire half the alphabet.
    cb.active_mask[8:] = False

    z = torch.randn(10, 8)
    idx, q, loss = cb.quantize_active(z)

    # Indices are in storage space: always < 16 and always in the active half.
    assert (idx < 16).all()
    assert (idx < 8).all()  # only active rows [0..7] selectable
    assert q.shape == (10, 8)
    assert loss.dim() == 0


def test_straight_through_gradient_flows_to_z():
    cb = AdaptiveCodebook(size=16, dim=8)
    z = torch.randn(8, 8, requires_grad=True)
    _, q, _ = cb.quantize_active(z)
    q.sum().backward()
    assert z.grad is not None
