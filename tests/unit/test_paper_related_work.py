"""Assertions that the paper v0.2 related-work section is present and cites all 5 references."""
from pathlib import Path


def test_refs_bib_contains_soundstream_entry():
    text = Path("papers/paper1/refs.bib").read_text()
    assert "zeghidour2022soundstream" in text, \
        "refs.bib should contain a SoundStream bibentry (Zeghidour 2022)."
    assert "SoundStream" in text


def test_main_tex_related_work_cites_all_five_refs():
    text = Path("papers/paper1/main.tex").read_text()
    # Section header.
    assert "\\section{Related Work}" in text or "\\section*{Related Work}" in text
    # All 5 expected cites.
    expected = [
        "bastos2012canonical",
        "rao1999predictive",
        "vandenoord2017neural",
        "zeghidour2022soundstream",
        "neftci2019surrogate",
    ]
    for key in expected:
        assert f"\\cite{{{key}}}" in text, f"main.tex should cite {key} in related work."
