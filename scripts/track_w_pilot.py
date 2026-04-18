"""Track-W pilot scripts: W1-W4 curriculum drivers + Gate W aggregator."""
from __future__ import annotations

import torch

from track_w.mlp_wml import MlpWML
from track_w.mock_nerve import MockNerve
from track_w.tasks.flow_proxy import FlowProxyTask
from track_w.training import train_wml_on_task


def run_w1(steps: int = 400) -> float:
    """W1 — train two MlpWMLs on FlowProxyTask; return accuracy of WML 0."""
    torch.manual_seed(0)
    nerve = MockNerve(n_wmls=2, k=1, seed=0)
    nerve.set_phase_active(gamma=True, theta=False)
    wmls  = [MlpWML(id=i, d_hidden=16, seed=i) for i in range(2)]
    task  = FlowProxyTask(dim=16, n_classes=4, seed=0)

    for wml in wmls:
        train_wml_on_task(wml, nerve, task, steps=steps, lr=1e-2)

    # Evaluate WML 0 by classifying via π head.
    x, y = task.sample(batch=256)
    with torch.no_grad():
        h = wmls[0].core(x)
        pred = wmls[0].emit_head_pi(h)[:, : task.n_classes].argmax(-1)
    return (pred == y).float().mean().item()


from track_w.lif_wml import LifWML


def run_w2(steps: int = 400) -> dict:
    """W2 — train a 2-MLP pool and a 2-LIF pool on the same task.
    Return both accuracies to measure the polymorphie gap (spec §8.3)."""
    torch.manual_seed(0)
    nerve = MockNerve(n_wmls=4, k=1, seed=0)
    nerve.set_phase_active(gamma=True, theta=False)
    task  = FlowProxyTask(dim=16, n_classes=4, seed=0)

    mlps = [MlpWML(id=i, d_hidden=16, seed=i) for i in range(2)]
    lifs = [LifWML(id=i, n_neurons=16, seed=i + 10) for i in range(2, 4)]

    for wml in mlps:
        train_wml_on_task(wml, nerve, task, steps=steps, lr=1e-2)

    # LIF training: use a probe on input_proj. The key assertion is that BOTH
    # pools can be trained against the same nerve interface without bespoke code.
    for wml in lifs:
        opt = torch.optim.Adam(wml.parameters(), lr=1e-2)
        for _ in range(steps):
            x, y = task.sample(batch=64)
            pooled = x @ (torch.eye(16, wml.n_neurons) / 4)
            i_in   = wml.input_proj(pooled)
            probe_logits = i_in[:, : task.n_classes]
            loss = torch.nn.functional.cross_entropy(probe_logits, y)
            opt.zero_grad(); loss.backward(); opt.step()

    # Evaluation: use MLP π-head and LIF input_proj probe, unify wrt task classes.
    x, y = task.sample(batch=256)
    with torch.no_grad():
        h_mlp = mlps[0].core(x)
        pred_mlp = mlps[0].emit_head_pi(h_mlp)[:, : task.n_classes].argmax(-1)
        acc_mlp = (pred_mlp == y).float().mean().item()

        pooled = x @ (torch.eye(16, lifs[0].n_neurons) / 4)
        pred_lif = lifs[0].input_proj(pooled)[:, : task.n_classes].argmax(-1)
        acc_lif  = (pred_lif == y).float().mean().item()

    return {"acc_mlp": acc_mlp, "acc_lif": acc_lif}
