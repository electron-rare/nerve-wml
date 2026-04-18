import pytest
from nerve_core.neuroletter import Neuroletter, Role, Phase


def test_neuroletter_is_frozen():
    n = Neuroletter(code=5, role=Role.PREDICTION, phase=Phase.GAMMA,
                    src=1, dst=2, timestamp=0.5)
    with pytest.raises(Exception):
        n.code = 6  # type: ignore[misc]


def test_neuroletter_equality_and_hashable():
    a = Neuroletter(5, Role.PREDICTION, Phase.GAMMA, 1, 2, 0.5)
    b = Neuroletter(5, Role.PREDICTION, Phase.GAMMA, 1, 2, 0.5)
    assert a == b
    assert hash(a) == hash(b)
    assert {a, b} == {a}


def test_role_and_phase_enums_have_two_values():
    assert {Role.PREDICTION, Role.ERROR} == set(Role)
    assert {Phase.GAMMA, Phase.THETA} == set(Phase)
