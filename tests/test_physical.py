import random
import sys
from pathlib import Path

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).parents[1]))
from talea.physical import (  # noqa: E402
    best_harmonic_comb,
    encode_atchley,
    repetita_like_features,
    scan_sequence,
    score_window,
    window_plan,
)


def harmonic_equivalent(observed, expected, tolerance=0.12):
    return any(
        abs(observed - expected * multiplier) / (expected * multiplier) <= tolerance
        for multiplier in (0.5, 1.0, 2.0)
    )


def test_atchley_rejects_ambiguous_residues_instead_of_masking_them():
    with pytest.raises(ValueError, match="noncanonical"):
        encode_atchley("ACX")


def test_repeated_motif_outscores_composition_matched_shuffle():
    motif = "EVLDTLKRHPEVVEALKEAGVEVV"
    repeated = motif * 12
    shuffled = list(repeated)
    random.Random(17).shuffle(shuffled)
    repeated_call = score_window(repeated)["physical"]
    shuffled_call = score_window("".join(shuffled))["physical"]
    assert repeated_call["score"] > shuffled_call["score"] + 1.0
    assert harmonic_equivalent(repeated_call["period"], len(motif))


def test_window_plan_includes_terminal_and_full_adaptive_windows():
    windows = window_plan(300)
    assert (0, 300) in windows
    assert (172, 300) in windows
    assert all(0 <= start < end <= 300 for start, end in windows)


def test_short_random_sequence_returns_finite_calls():
    calls = scan_sequence("ACDEFGHIKLMNPQRSTVWY" * 3)
    assert len(calls) == 1
    assert np.isfinite(calls[0]["physical"]["score"])


def test_empty_spectrum_has_no_call():
    call = best_harmonic_comb(np.zeros(5), signal_length=8)
    assert call.period == -1
    assert call.score == 0


def test_repetita_like_features_emit_all_five_scale_statistics():
    motif = "EVLDTLKRHPEVVEALKEAGVEVV"
    result = repetita_like_features(motif * 8)
    assert result["zmax"] > 0
    assert 1 <= result["factor"] <= 5
    assert result["period"] > 0
    assert "rho_2_1" in result
    assert "rho_3_4" in result
