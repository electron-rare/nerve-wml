import torch

from track_w.losses import composite_loss


def test_composite_loss_returns_scalar():
    task_loss = torch.tensor(1.2, requires_grad=True)
    vq_loss   = torch.tensor(0.4, requires_grad=True)
    sep_loss  = torch.tensor(0.1, requires_grad=True)

    total = composite_loss(task_loss=task_loss, vq_loss=vq_loss, sep_loss=sep_loss)
    assert total.dim() == 0
    total.backward()
    assert task_loss.grad is not None


def test_composite_loss_weights_are_applied():
    task = torch.tensor(1.0)
    vq   = torch.tensor(1.0)
    sep  = torch.tensor(1.0)
    total = composite_loss(task_loss=task, vq_loss=vq, sep_loss=sep,
                           lam_vq=0.25, lam_sep=0.05)
    assert abs(total.item() - (1.0 + 0.25 + 0.05)) < 1e-6
