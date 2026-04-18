import torch

from scripts.track_p_pilot import run_p1
from track_p.info_theoretic import dead_code_fraction


def test_p1_codebook_has_low_dead_code_fraction():
    torch.manual_seed(0)
    cb = run_p1(steps=4000)
    # Gate P1 criterion: dead codes < 10 %
    assert dead_code_fraction(cb) < 0.10


def test_p1_codebook_perplexity_meets_target():
    torch.manual_seed(0)
    cb = run_p1(steps=4000)
    # Perplexity = 2^entropy on the normalized usage. Target ≥ 32 / 64.
    counts = cb.usage_counter.float()
    p = counts / (counts.sum() + 1e-9)
    ent_bits = -(p * (p + 1e-9).log2()).sum().item()
    perplexity = 2 ** ent_bits
    assert perplexity >= 32
