import torch

from nerve_core.neuroletter import Phase, Role
from track_w.mlp_wml import MlpWML
from track_w.mock_nerve import MockNerve


def test_mlp_wml_has_required_attrs():
    wml = MlpWML(id=0, d_hidden=128, seed=0)
    assert wml.id == 0
    assert wml.codebook.shape == (64, 128)
    assert hasattr(wml, "core")
    assert hasattr(wml, "emit_head_pi")
    assert hasattr(wml, "emit_head_eps")
    assert wml.threshold_eps == 0.30


def test_mlp_wml_parameters_include_codebook_and_core():
    wml = MlpWML(id=0, d_hidden=128, seed=0)
    param_ids = {id(p) for p in wml.parameters()}
    assert id(wml.codebook) in param_ids
    # At least one linear in core should be a parameter.
    core_params = [p for p in wml.core.parameters()]
    assert len(core_params) > 0
    assert all(id(p) in param_ids for p in core_params)


def test_mlp_wml_seed_is_local():
    """Constructing an MlpWML must NOT mutate the global torch RNG."""
    torch.manual_seed(42)
    expected = torch.rand(1).item()

    torch.manual_seed(42)
    _ = MlpWML(id=0, d_hidden=128, seed=99)
    observed = torch.rand(1).item()

    assert expected == observed


def test_mlp_wml_step_emits_pi_when_gamma_active():
    nerve = MockNerve(n_wmls=2, k=1, seed=0)
    nerve.set_phase_active(gamma=True, theta=False)
    wml = MlpWML(id=0, d_hidden=16, seed=0)
    wml.step(nerve, t=0.0)
    received = nerve.listen(wml_id=1, role=Role.PREDICTION)
    # At least one π should have been sent along the single active edge.
    assert len(received) >= 1
    for letter in received:
        assert letter.role is Role.PREDICTION
        assert letter.phase is Phase.GAMMA
        assert letter.src == 0


def test_mlp_wml_step_respects_sparse_routing():
    """No message should reach a WML outside the router's topology."""
    nerve = MockNerve(n_wmls=3, k=1, seed=0)
    nerve.set_phase_active(gamma=True, theta=False)
    wml = MlpWML(id=0, d_hidden=16, seed=0)
    wml.step(nerve, t=0.0)

    dsts_reachable = [j for j in range(3) if nerve.routing_weight(0, j) == 1.0]
    unreachable    = [j for j in range(3) if j != 0 and j not in dsts_reachable]

    for dst in unreachable:
        assert nerve.listen(wml_id=dst) == []


def test_mlp_wml_emits_eps_when_surprise_high_and_theta_active():
    """With a large input mismatch and θ active (γ inactive), ε should fire."""
    nerve = MockNerve(n_wmls=2, k=1, seed=0)
    nerve.set_phase_active(gamma=False, theta=True)
    wml = MlpWML(id=0, d_hidden=16, seed=0, threshold_eps=0.0)

    # Synthesise a spike-like input letter in the wml's queue before step().
    from nerve_core.neuroletter import Neuroletter, Phase, Role
    nerve._queues[0].append(
        Neuroletter(code=42, role=Role.ERROR, phase=Phase.THETA,
                    src=1, dst=0, timestamp=0.0)
    )

    wml.step(nerve, t=0.0)

    received = nerve.listen(wml_id=1, role=Role.ERROR)
    # Under θ active + γ inactive + threshold 0 + non-trivial inbound,
    # at least one ε must be emitted.
    assert len(received) >= 1
    for letter in received:
        assert letter.role is Role.ERROR
        assert letter.phase is Phase.THETA
