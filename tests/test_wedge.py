import sys
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).parents[1]))
from talea.wedge import (  # noqa: E402
    best_dense_interval,
    best_wedge_interval,
    candidate_periods_from_windows,
    centered_moving_average,
    expand_interval_by_period,
    extend_interval_by_residue_evidence,
    fuse_diagonal_tracks,
    harmonic_residue_scores,
    robust_diagonal_tracks,
    top_wedge_intervals,
    wedge_start_scores,
)


def localized_banded_matrix(
    length: int,
    start: int,
    end: int,
    period: int,
) -> np.ndarray:
    matrix = np.zeros((length, length), dtype=np.float64)
    for offset in range(period, end - start, period):
        indices = np.arange(start, end - offset)
        matrix[indices, indices + offset] = 1.0
        matrix[indices + offset, indices] = 1.0
    return matrix


def localized_dense_matrix(length: int, start: int, end: int) -> np.ndarray:
    matrix = np.zeros((length, length), dtype=np.float64)
    matrix[start:end, start:end] = 1.0
    np.fill_diagonal(matrix, 0.0)
    return matrix


def test_wedge_scores_peak_at_true_internal_start():
    matrix = localized_banded_matrix(512, 160, 288, 20)
    tracks = robust_diagonal_tracks(matrix, max_separation=160)
    scores = wedge_start_scores(tracks, 512, period=20, width=128)
    assert abs(int(np.argmax(scores)) - 160) <= 2


def test_wedge_search_recovers_period_width_and_interval():
    matrix = localized_banded_matrix(512, 160, 288, 20)
    tracks = robust_diagonal_tracks(matrix, max_separation=192)
    call = best_wedge_interval(
        tracks,
        512,
        periods=(17, 20, 23),
        widths=(96, 128, 160),
    )
    assert call["period"] == 20
    assert call["window_size"] == 128
    assert abs(call["start"] - 160) <= 2
    assert abs(call["end"] - 288) <= 2


def test_wedge_search_handles_terminal_interval():
    matrix = localized_banded_matrix(384, 0, 144, 18)
    tracks = robust_diagonal_tracks(matrix, max_separation=160)
    call = best_wedge_interval(
        tracks,
        384,
        periods=(15, 18, 21),
        widths=(112, 144, 176),
    )
    assert call["period"] == 18
    assert call["window_size"] == 144
    assert call["start"] == 0


def test_fuse_diagonal_tracks_averages_layers():
    matrix = localized_banded_matrix(128, 32, 96, 16)
    tracks = robust_diagonal_tracks(matrix, max_separation=64)
    fused = fuse_diagonal_tracks((tracks, tracks))
    assert len(fused) == len(tracks)
    assert np.array_equal(fused[16], tracks[16])


def test_period_candidates_use_top_windows_and_harmonic_aliases():
    windows = [
        {
            "window_size": size,
            "attention_fused": {
                "score": score,
                "period": period,
                "n_harmonics": 3,
            },
        }
        for size, score, period in (
            (64, 100.0, 11.0),
            (128, 2.0, 20.2),
            (256, 1.0, 30.0),
        )
    ]
    candidates = candidate_periods_from_windows(windows, top_n=1)
    assert candidates == (10.0, 20.0, 40.5)


def test_dense_contact_control_recovers_dense_interval():
    matrix = localized_dense_matrix(384, 96, 224)
    tracks = robust_diagonal_tracks(matrix, max_separation=160)
    call = best_dense_interval(tracks, 384, widths=(96, 128, 160))
    assert call["window_size"] == 128
    assert call["start"] == 96
    assert call["end"] == 224


def test_top_wedges_recover_two_nonredundant_loci():
    first = localized_banded_matrix(640, 80, 208, 20)
    second = localized_banded_matrix(640, 400, 528, 20)
    tracks = robust_diagonal_tracks(first + second, max_separation=192)
    calls = top_wedge_intervals(
        tracks,
        640,
        periods=(20,),
        widths=(128,),
        top_k=2,
    )
    assert len(calls) == 2
    assert {call["start"] for call in calls} == {80, 400}
    assert [call["rank"] for call in calls] == [1, 2]


def test_period_expansion_preserves_center_and_bounds():
    call = {"start": 100, "end": 200, "period": 20.0}
    expanded = expand_interval_by_period(call, 400, periods_per_flank=0.5)
    assert expanded["start"] == 90
    assert expanded["end"] == 210
    terminal = expand_interval_by_period(
        {"start": 0, "end": 100, "period": 20.0},
        400,
        periods_per_flank=0.5,
    )
    assert terminal["start"] == 0
    assert terminal["end"] == 120


def test_harmonic_residue_scores_localize_pair_endpoints():
    matrix = localized_banded_matrix(384, 96, 256, 20)
    tracks = robust_diagonal_tracks(matrix, max_separation=160)
    scores = harmonic_residue_scores(tracks, 384, period=20)
    assert np.median(scores[116:236]) > np.median(scores[:80])
    assert np.median(scores[116:236]) > np.median(scores[280:])


def test_centered_moving_average_preserves_length_and_constant_values():
    values = np.full(17, 3.0)
    result = centered_moving_average(values, 6)
    assert len(result) == len(values)
    assert np.allclose(result, values)


def test_residue_evidence_extension_is_asymmetric_and_never_contracts_seed():
    call = {"start": 100, "end": 200, "period": 20.0}
    scores = np.zeros(400)
    scores[80:230] = 4.0
    result = extend_interval_by_residue_evidence(
        call,
        400,
        scores,
        smoothing_periods=0.5,
        baseline_quantile=0.25,
        threshold_fraction=0.2,
        core_reference_quantile=0.5,
    )
    assert result["seed_start"] == 90
    assert result["seed_end"] == 210
    assert result["start"] <= 90
    assert result["end"] >= 210
    assert result["start"] < 100
    assert result["end"] > 200
