"""DFT features from the diagonal Radon projection of APC attention maps."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter

from talea.physical import best_harmonic_comb, score_signals, window_plan


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
    *,
    include_diagnostics: bool = True,
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
    )
    if not include_diagnostics:
        return {"score": call.score, "period": call.period, "n_harmonics": call.n_harmonics}
    call = call.to_dict()
    center = float(np.median(profile)) if len(profile) else 0.0
    call["profile_mad"] = (
        float(np.median(np.abs(profile - center))) if len(profile) else 0.0
    )
    return call


def _batched_window_calls(profiles: np.ndarray) -> list[dict]:
    """Score rows of equal-length diagonal profiles with independent transforms.

    Reductions are along each contiguous row. Run each FFT independently to keep
    the same single-transform kernel as the scalar implementation on every CPU.
    """
    n = profiles.shape[1]
    width = min(31, n if n % 2 else n - 1)
    residual = profiles - median_filter(profiles, size=(1, width), mode="nearest")
    time = np.arange(n, dtype=np.float64) - (n - 1) / 2
    centered = residual - residual.mean(axis=1, keepdims=True)
    slopes = np.sum(centered * time, axis=1, keepdims=True) / np.dot(time, time)
    centered -= time * slopes
    deviations = centered.std(axis=1, keepdims=True)
    normalized = np.divide(centered, deviations, out=np.zeros_like(centered), where=deviations > 1e-12)
    window = np.hanning(n)
    transformed = np.stack([np.fft.rfft(row) for row in normalized * window])
    log_power = np.log1p(np.abs(transformed) ** 2 / np.sum(window**2))
    width = min(15, max(5, (log_power.shape[1] // 8) | 1))
    residual = log_power - median_filter(log_power, size=(1, width), mode="nearest")
    usable = residual[:, 2:]
    center = np.median(usable, axis=1, keepdims=True)
    scale = 1.4826 * np.median(np.abs(usable - center), axis=1, keepdims=True)
    small = scale[:, 0] < 1e-8
    scale[small, 0] = usable[small].std(axis=1)
    spectral_z = np.divide(residual - center, scale, out=np.zeros_like(residual), where=scale >= 1e-8)
    calls = []
    for spectrum in spectral_z:
        call = best_harmonic_comb(spectrum, n, max_period=min(250.0, n / 3))
        calls.append({"score": call.score, "period": call.period, "n_harmonics": call.n_harmonics})
    return calls


def _exact_diagonal_prefix(matrix: np.ndarray, width: int) -> np.ndarray | None:
    """Shared window sums for the binary16 storage grid.

    Every binary16 value is an integer multiple of 2**-24. For up to 8192
    entries of magnitude <= 65504, each possible partial sum is an exact
    float64. Subtracting two prefixes therefore gives the same bits as summing
    the subwindow directly. Arbitrary floating-point input uses the scalar
    summation order unless it meets the same exactness conditions.
    """
    length = len(matrix)
    if matrix.dtype not in (np.float16, np.float32) or length > 8192:
        return None
    if not np.max(np.abs(matrix)) <= 65504:
        return None
    if matrix.dtype == np.float32:
        scaled = np.ldexp(matrix, 24)
        if not np.array_equal(scaled, np.rint(scaled)):
            return None
    # Flat index i*(length+1)+d is the element (i, i+d). Right padding
    # keeps the final row's unused columns inside the underlying allocation.
    flat = np.pad(matrix.ravel(), (0, width - 1))
    diagonals = np.lib.stride_tricks.sliding_window_view(flat, width)[::length + 1].T
    prefix = np.empty((width, length + 1), dtype=np.float64)
    prefix[:, 0] = 0.0
    np.cumsum(diagonals, axis=1, dtype=np.float64, out=prefix[:, 1:])
    return prefix



def compact_window_calls(matrix: np.ndarray, spans: list[tuple[int, int]], whole: dict | None) -> dict:
    """Score the fields consumed by the topology calibration, batched by window length."""
    by_size: dict[int, list[tuple[int, int]]] = {}
    result = {}
    for start, end in spans:
        if whole is not None and start == 0 and end == len(matrix):
            result[(start, end)] = whole
        else:
            by_size.setdefault(end - start, []).append((start, end))
    prefix = None
    if by_size and len(spans) >= 8:
        prefix = _exact_diagonal_prefix(matrix, max(by_size))
    for width, group in by_size.items():
        if len(group) < 2 or width < 8:
            for start, end in group:
                result[(start, end)] = score_attention_window(matrix[start:end, start:end], include_diagnostics=False)
            continue
        if prefix is None:
            profiles = np.stack([diagonal_mean_profile(matrix[start:end, start:end]) for start, end in group])
        else:
            starts = np.array([start for start, end in group])[:, None]
            offsets = np.arange(1, width)
            counts = width - offsets
            profiles = (prefix[offsets, starts + counts] - prefix[offsets, starts]) / counts
        result.update(zip(group, _batched_window_calls(profiles), strict=True))
    return result


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
