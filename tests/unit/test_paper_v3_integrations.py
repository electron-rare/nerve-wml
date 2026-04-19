"""Assert paper v0.3 integrates the 4 additional gate sections."""
from pathlib import Path


def _tex() -> str:
    return Path("papers/paper1/main.tex").read_text()


def test_paper_has_integrations_section():
    text = _tex()
    assert "\\section{Integrations}" in text


def test_paper_mentions_all_eleven_gates_in_abstract():
    # LaTeX source wraps lines; normalize whitespace before matching tokens.
    text = " ".join(_tex().split())
    abstract_tokens = [
        "Gate P",
        "Gate W",
        "Gate M",
        "Gate M2",
        "Gate Scale",
        "Gate Interp",
        "Gate Neuro",
        "Gate Dream",
        "Gate Adaptive",
        "Gate LLM Advisor",
    ]
    for token in abstract_tokens:
        assert token in text, f"abstract should mention {token!r}"


def test_integrations_cites_each_pilot_module():
    text = _tex()
    for module in (
        "track\\_p.adaptive\\_codebook",
        "neuromorphic.export",
        "bridge.dream\\_bridge",
        "bridge.kiki\\_nerve\\_advisor",
    ):
        assert module in text, f"§Integrations should cite {module}"
