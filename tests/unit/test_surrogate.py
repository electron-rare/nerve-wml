import torch

from track_w._surrogate import spike_with_surrogate


def test_spike_forward_is_step():
    v = torch.tensor([-1.0, 0.0, 0.5, 1.5])
    spikes = spike_with_surrogate(v, v_thr=1.0)
    assert torch.allclose(spikes, torch.tensor([0.0, 0.0, 0.0, 1.0]))


def test_spike_has_nonzero_gradient():
    v = torch.tensor([0.2, 0.9, 1.1, 2.0], requires_grad=True)
    spikes = spike_with_surrogate(v, v_thr=1.0)
    spikes.sum().backward()
    assert v.grad is not None
    # Fast-sigmoid surrogate has positive gradient everywhere.
    assert (v.grad > 0).all()
