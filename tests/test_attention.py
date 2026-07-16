import random
import sys
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).parents[1]))
from talea.attention import (  # noqa: E402
    diagonal_mean_profile,
    scan_attention_matrix,
    score_attention_window,
)


def banded_matrix(length=256, period=20):
    matrix = np.zeros((length, length), dtype=np.float64)
    for offset in range(period, length, period):
        indices = np.arange(length - offset)
        matrix[indices, indices + offset] = 1.0
        matrix[indices + offset, indices] = 1.0
    return matrix


def test_diagonal_profile_recovers_band_offsets():
    profile = diagonal_mean_profile(banded_matrix(100, 20))
    assert profile[19] == 1.0
    assert profile[39] == 1.0
    assert profile[18] == 0.0


def test_banded_attention_outscores_position_shuffled_control():
    matrix = banded_matrix()
    permutation = list(range(len(matrix)))
    random.Random(17).shuffle(permutation)
    shuffled = matrix[np.ix_(permutation, permutation)]
    banded = score_attention_window(matrix)
    control = score_attention_window(shuffled)
    assert banded["score"] > control["score"] + 1.0
    assert min(
        abs(banded["period"] - 20 * multiplier) / (20 * multiplier)
        for multiplier in (0.5, 1.0, 2.0)
    ) < 0.15


def test_attention_scan_emits_local_windows():
    calls = scan_attention_matrix(banded_matrix(300, 30))
    assert len(calls) > 1
    assert all("attention" in call for call in calls)
