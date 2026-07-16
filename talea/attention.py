"""DFT features from the diagonal Radon projection of APC attention maps."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter

from talea.physical import score_signals, window_plan


def diagonal_mean_profile(matrix: np.ndarray) -> np.ndarray:
    """Mean APC intensity at residue separations 1..L-1.

    This diagonal average is the 45-degree Radon projection of the attention
    matrix. Its one-dimensional DFT is therefore the corresponding Fourier
    slice of the two-dimensional matrix, without image rotation/interpolation.
    """
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("attention matrix must be square")
    return np.asarray(
        [float(np.mean(np.diagonal(values, offset))) for offset in range(1, len(values))],
        dtype=np.float64,
    )


def local_distance_residual(profile: np.ndarray, width: int = 31) -> np.ndarray:
    """Remove smooth attention-vs-distance decay while preserving band spacing."""
    values = np.asarray(profile, dtype=np.float64)
    if len(values) < 5:
        return values - np.mean(values) if len(values) else values.copy()
    filter_width = min(width, len(values) if len(values) % 2 else len(values) - 1)
    filter_width = max(5, filter_width)
    if filter_width % 2 == 0:
        filter_width -= 1
    baseline = median_filter(values, size=filter_width, mode="nearest")
    return values - baseline


def score_attention_window(
    matrix: np.ndarray,
    min_period: float = 5.0,
    max_period: float = 250.0,
    min_cycles: int = 3,
) -> dict:
    """Score periodic off-diagonal bands in one APC attention window."""
    profile = diagonal_mean_profile(matrix)
    residual = local_distance_residual(profile)
    effective_maximum = min(max_period, len(residual) / max(min_cycles, 1))
    call = score_signals(
        residual[:, None],
        min_period=min_period,
        max_period=effective_maximum,
        min_cycles=min_cycles,
    ).to_dict()
    center = float(np.median(profile)) if len(profile) else 0.0
    call["profile_mad"] = (
        float(np.median(np.abs(profile - center))) if len(profile) else 0.0
    )
    return call


def scan_attention_matrix(
    matrix: np.ndarray,
    min_period: float = 5.0,
    max_period: float = 250.0,
    min_cycles: int = 3,
    window_sizes: tuple[int, ...] = (128, 256, 512),
) -> list[dict]:
    """Return multiscale attention-periodicity calls for one protein."""
    length = int(np.asarray(matrix).shape[0])
    calls = []
    for start, end in window_plan(length, window_sizes=window_sizes):
        calls.append(
            {
                "start": start,
                "end": end,
                "window_size": end - start,
                "attention": score_attention_window(
                    matrix[start:end, start:end],
                    min_period=min_period,
                    max_period=max_period,
                    min_cycles=min_cycles,
                ),
            }
        )
    return calls
