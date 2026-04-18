import torch

from bridge.merge_trainer import MergeTrainer
from bridge.sim_nerve_adapter import SimNerveAdapter
from track_w.mlp_wml import MlpWML
from track_w.tasks.flow_proxy import FlowProxyTask


def test_merge_trainer_freezes_wml_internals():
    wmls  = [MlpWML(id=i, d_hidden=16, seed=i) for i in range(2)]
    nerve = SimNerveAdapter(n_wmls=2, k=1, seed=0)
    task  = FlowProxyTask(dim=16, n_classes=4, seed=0)

    trainer = MergeTrainer(wmls=wmls, nerve=nerve, task=task, steps=50, lr=1e-2)

    # Snapshot WML 0 core weight before training.
    snapshot = wmls[0].core[0].weight.data.clone()

    trainer.train()

    # WML 0 core should be unchanged — only transducers move.
    assert torch.allclose(wmls[0].core[0].weight.data, snapshot)


def test_merge_trainer_updates_transducers():
    wmls  = [MlpWML(id=i, d_hidden=16, seed=i) for i in range(2)]
    nerve = SimNerveAdapter(n_wmls=2, k=1, seed=0)
    task  = FlowProxyTask(dim=16, n_classes=4, seed=0)

    # Pick one transducer's logits and snapshot them.
    assert len(nerve._transducers) > 0, "need at least one active edge"
    key = next(iter(nerve._transducers.keys()))
    snap_logits = nerve._transducers[key].logits.data.clone()

    MergeTrainer(wmls=wmls, nerve=nerve, task=task, steps=50, lr=1e-2).train()

    # Transducer should have moved.
    assert not torch.allclose(nerve._transducers[key].logits.data, snap_logits)
