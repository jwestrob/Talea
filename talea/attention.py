"""DFT features from the diagonal Radon projection of APC attention maps."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter

from talea.physical import score_signals, window_plan


# Above this side length the vectorized path's (L, L+1) working grid stops
# fitting in cache and becomes slower than the per-diagonal loop. Measured
# crossover is between 1024 and 2046; see perf/bench_optimizations.py.
DIAGONAL_FAST_PATH_MAXIMUM = 1024


def _diagonal_mean_profile_reference(matrix: np.ndarray) -> np.ndarray:
    """Original per-diagonal implementation, retained as the concordance oracle.

    Kept in the package (rather than in the tests) so the optimized path always
    has an in-tree reference to be checked against, and so the behaviour being
    reproduced is documented where it is replaced.
    """
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("attention matrix must be square")
    return np.asarray(
        [float(np.mean(np.diagonal(values, offset))) for offset in range(1, len(values))],
        dtype=np.float64,
    )


def _diagonal_mean_profile_vectorized(values: np.ndarray) -> np.ndarray:
    """Diagonal means via a shifted-reshape cumulative sum.

    Element ``(i, i + k)`` of a row-major ``L x L`` array lies at flat index
    ``i * (L + 1) + k``.  Padding the flat array to ``L * (L + 1)`` and
    reshaping to ``(L, L + 1)`` therefore places diagonal ``k`` in column
    ``k``, valid over its first ``L - k`` rows.  A single cumulative sum down
    the columns then yields every diagonal sum at once.
    """
    length = len(values)
    padded = np.empty(length * (length + 1), dtype=np.float64)
    padded[: length * length] = values.reshape(-1)
    padded[length * length :] = 0.0
    grid = padded.reshape(length, length + 1)
    np.cumsum(grid, axis=0, out=grid)
    offsets = np.arange(1, length)
    counts = length - offsets
    return grid[counts - 1, offsets] / counts


def diagonal_mean_profile(matrix: np.ndarray) -> np.ndarray:
    """Mean APC intensity at residue separations 1..L-1.

    This diagonal average is the 45-degree Radon projection of the attention
    matrix. Its one-dimensional DFT is therefore the corresponding Fourier
    slice of the two-dimensional matrix, without image rotation/interpolation.
    """
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("attention matrix must be square")
    if len(values) < 2:
        return np.empty(0, dtype=np.float64)
    if len(values) > DIAGONAL_FAST_PATH_MAXIMUM:
        return _diagonal_mean_profile_reference(values)
    return _diagonal_mean_profile_vectorized(values)


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
