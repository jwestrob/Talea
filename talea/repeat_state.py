"""Utilities for sequence-only residue-state and repeat-architecture routing.

The residue-state representation deliberately mirrors the useful geometric
part of SOLeNNoID without requiring a structure at inference.  Two APC-corrected
ESM++ attention layers are robustly normalized, split into overlapping square
tiles along the sequence diagonal, and scored only over each tile's central
residue block.  Central scoring prevents padding and tile-edge artefacts from
dominating the reconstructed residue track.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


REPEAT_ARCHITECTURE_NAMES = {
    "2": "fibrous_repeat",
    "3": "elongated_repeat",
    "4": "closed_repeat",
    "5": "beads_on_a_string",
}


def repeat_architecture_label(top_level_classes: Sequence[str]) -> str:
    """Return a stable architecture label from RepeatsDB top-level classes."""

    classes = tuple(sorted({str(value) for value in top_level_classes}))
    if not classes:
        return "non_repeat"
    if len(classes) > 1:
        return "mixed_repeat_architecture"
    return REPEAT_ARCHITECTURE_NAMES.get(classes[0], "other_repeat_architecture")


def residue_labels_from_runs(length: int, runs: Sequence[Sequence[int]]) -> np.ndarray:
    """Build a binary, zero-based half-open residue track from labelled runs."""

    if length < 1:
        raise ValueError("length must be positive")
    labels = np.zeros(length, dtype=np.float32)
    for run in runs:
        if len(run) < 2:
            raise ValueError("each labelled run requires start and end")
        start, end = int(run[0]), int(run[1])
        if not 0 <= start < end <= length:
            raise ValueError(f"labelled run [{start}, {end}) outside length {length}")
        labels[start:end] = 1.0
    return labels


NORMALIZATION_SCOPES = ("matrix", "band")


def _normalization_positive_values(
    values, scope: str, band: int
):
    """Entries used to set the normalization scale.

    ``matrix`` -- every positive entry of the L x L map (the v0/v2 behaviour).

    ``band``   -- only entries at separations the teacher can actually see.
    The teacher scores 128x128 tiles down the diagonal, so it never observes
    |i - j| >= tile_size. The near-diagonal band is O(L) entries out of O(L^2),
    so under ``matrix`` scope its share of the map falls as ~1/L (measured:
    51.6% of entries at L=200, 5.7% at L=2046). The 0.95 quantile therefore
    slides off the band into background as proteins get longer, the scale
    shrinks, and the band's normalized intensity inflates 2.5x from length
    alone. Restricting the quantile to the band removes that: measured
    inflation over the same range falls from 2.54x to 1.15x.
    See perf/WHY_LENGTH_BIAS.md.
    """
    import numpy as _np

    if scope not in NORMALIZATION_SCOPES:
        raise ValueError(f"unsupported normalization scope: {scope}")
    if scope == "matrix":
        return values[values > 0]
    if band < 2:
        raise ValueError("normalization band must be at least 2")
    length = values.shape[0]
    offsets = range(1, min(int(band), length))
    parts = [_np.diagonal(values, offset=k) for k in offsets]
    if not parts:
        return values[values > 0]
    selected = _np.concatenate(parts)
    return selected[selected > 0]


def _matrix_quantile(values: np.ndarray, quantile: float) -> float:
    """Exact quantile of positive entries, retaining NumPy's interpolation."""
    # Use a sample only to pick a cutoff. Entries below it may be replaced by
    # zeros if both order statistics for the requested quantile are above it.
    # This leaves a small tail to partition, even for a multi-million-entry map.
    if values.size >= 65536 and quantile >= 0.8:
        sample = values.flat[:: max(1, values.size // 1024)]
        sample = sample[sample > 0]
        if sample.size:
            index = int((sample.size - 1) * (quantile - 0.04))
            sample.partition(index)
            tail = values[values >= sample[index]]
            count = np.count_nonzero(values > 0)
            omitted = count - len(tail)
            # Margin covers rounding of NumPy's fractional index in float32.
            rank = (count - 1) * quantile
            margin = 1 + int(np.ceil(count * np.finfo(np.float32).eps))
            low, high = int(rank) - margin, int(rank) + margin + 1
            if omitted <= low and high < count:
                lo, hi = low - omitted, high - omitted
                tail.partition((lo, hi))
                if tail[lo] == tail[hi]:
                    return float(tail[lo])
                positive = np.empty(count, dtype=values.dtype)
                positive[:omitted] = 0
                positive[omitted:] = tail
                return float(np.quantile(positive, quantile, overwrite_input=True))
    positive = values[values > 0]
    if not positive.size:
        return 0.0
    return float(np.quantile(positive, quantile, overwrite_input=True))


def compiled_teacher_quantile():
    """Load the optional CPU histogram only when the caller requests it."""
    try:
        from numba import config
        if not config.DISABLE_JIT:
            from talea._teacher_histogram import grid_quantile
            return grid_quantile
    except ImportError:
        pass
    return None


def normalized_attention_tiles(
    matrices: Sequence[np.ndarray],
    spans: Sequence[Mapping[str, int]],
    tile_size: int,
    quantile: float,
    clip: float,
    *,
    quantile_function=None,
) -> np.ndarray:
    """Build the teacher's diagonal tiles using the full-map normalization scale."""
    if not 0.5 <= quantile < 1.0:
        raise ValueError("quantile must be in [0.5, 1)")
    if clip <= 0:
        raise ValueError("clip must be positive")
    length = matrices[0].shape[0]
    tiles = np.zeros((len(spans), len(matrices) + 1, tile_size, tile_size), np.float32)
    bounds = []
    for i, span in enumerate(spans):
        start, end = span["source_start"], span["source_end"]
        src = slice(max(0, start), min(length, end))
        dst = slice(max(0, -start), tile_size - max(0, end - length))
        bounds.append((src, dst))
        tiles[i, -1, dst, dst] = 1.0

    # On any row, the tiles covering it start at most (tile_size - 1) apart,
    # in multiples of the central step. Their union fits in this many columns.
    step = (
        spans[1]["source_start"] - spans[0]["source_start"]
        if len(spans) > 1 else tile_size
    )
    row_width = min(length, tile_size + (tile_size - 1) // step * step)
    denominator = np.log1p(clip)
    for channel, matrix in enumerate(matrices):
        scale = quantile_function(matrix, quantile) if quantile_function is not None else None
        if scale is None:
            if not np.isfinite(matrix).all():
                raise ValueError("attention matrix contains nonfinite values")
            if np.min(matrix) < 0:
                raise ValueError("APC attention matrix must be nonnegative")
            scale = _matrix_quantile(matrix, quantile)
        if scale <= 0 or not np.isfinite(scale):
            continue

        if row_width == length:
            buffer = band = matrix.copy(order="C")
        else:
            # Store (i, j) at i * row_width + j. Only the tile squares are
            # accessed: their column ranges move right with each successive
            # row, so different used rows occupy disjoint storage. Intersections
            # of consecutive tiles share the exact same cells.
            buffer = np.zeros((length - 1) * (row_width + 1) + 1, np.float32)
            band = np.ndarray(
                matrix.shape,
                dtype=np.float32,
                buffer=buffer,
                strides=(row_width * buffer.itemsize, buffer.itemsize),
            )
            for src, _ in bounds:
                band[src, src] = matrix[src, src]

        np.divide(buffer, scale, out=buffer)
        np.log1p(buffer, out=buffer)
        np.divide(np.clip(buffer, 0.0, denominator), denominator, out=buffer)
        for i, (src, dst) in enumerate(bounds):
            tiles[i, channel, dst, dst] = band[src, src]
    return tiles


def robust_attention_channel(
    matrix: np.ndarray,
    quantile: float = 0.95,
    clip: float = 8.0,
    scope: str = "matrix",
    band: int = 128,
) -> np.ndarray:
    """Normalize a nonnegative APC attention matrix without using labels.

    Positive entries are scaled by a within-protein upper quantile, transformed
    with ``log1p``, and clipped.  This removes most protein-length and raw-layer
    scale variation while retaining sparse high-attention texture.

    ``scope`` selects which entries set that quantile; see
    ``_normalization_positive_values``. ``matrix`` reproduces the frozen v2
    behaviour exactly and remains the default.
    """

    values = np.asarray(matrix, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("attention matrix must be square")
    if not np.isfinite(values).all():
        raise ValueError("attention matrix contains nonfinite values")
    if np.any(values < 0):
        raise ValueError("APC attention matrix must be nonnegative")
    if not 0.5 <= quantile < 1.0:
        raise ValueError("quantile must be in [0.5, 1)")
    if clip <= 0:
        raise ValueError("clip must be positive")
    positive = _normalization_positive_values(values, scope, band)
    if not positive.size:
        return np.zeros_like(values, dtype=np.float32)
    scale = float(np.quantile(positive, quantile))
    if not np.isfinite(scale) or scale <= 0:
        return np.zeros_like(values, dtype=np.float32)
    transformed = np.log1p(values / scale)
    denominator = np.log1p(clip)
    return (np.clip(transformed, 0.0, denominator) / denominator).astype(
        np.float32, copy=False
    )


def attention_channels(
    matrices: Sequence[np.ndarray],
    quantile: float = 0.95,
    clip: float = 8.0,
    scope: str = "matrix",
    band: int = 128,
) -> np.ndarray:
    """Stack consistently sized robust attention channels."""

    if not matrices:
        raise ValueError("at least one attention matrix is required")
    channels = [
        robust_attention_channel(
            matrix, quantile=quantile, clip=clip, scope=scope, band=band
        )
        for matrix in matrices
    ]
    shapes = {channel.shape for channel in channels}
    if len(shapes) != 1:
        raise ValueError("attention channel shapes disagree")
    return np.stack(channels, axis=0)


def central_tile_spans(
    length: int,
    tile_size: int = 128,
    central_size: int = 64,
) -> list[dict[str, int]]:
    """Describe overlapping diagonal tiles whose central blocks cover a chain."""

    if length < 1:
        raise ValueError("length must be positive")
    if tile_size < central_size or central_size < 1:
        raise ValueError("tile size must be at least the positive central size")
    flank = tile_size - central_size
    if flank % 2:
        raise ValueError("tile minus central size must be even")
    half_flank = flank // 2
    spans = []
    for central_start in range(0, length, central_size):
        central_end = min(length, central_start + central_size)
        spans.append(
            {
                "central_start": central_start,
                "central_end": central_end,
                "source_start": central_start - half_flank,
                "source_end": central_start - half_flank + tile_size,
                "central_offset": half_flank,
            }
        )
    return spans


def extract_attention_tile(
    channels: np.ndarray,
    span: Mapping[str, int],
    tile_size: int = 128,
    include_valid_pair_mask: bool = True,
) -> np.ndarray:
    """Extract one padded square diagonal tile and an optional validity channel."""

    values = np.asarray(channels, dtype=np.float32)
    if values.ndim != 3 or values.shape[1] != values.shape[2]:
        raise ValueError("channels must have shape (channels, length, length)")
    length = values.shape[1]
    source_start = int(span["source_start"])
    source_end = int(span["source_end"])
    if source_end - source_start != tile_size:
        raise ValueError("span does not match tile size")
    destination_start = max(0, -source_start)
    destination_end = tile_size - max(0, source_end - length)
    clipped_start = max(0, source_start)
    clipped_end = min(length, source_end)
    tile = np.zeros((values.shape[0], tile_size, tile_size), dtype=np.float32)
    tile[
        :,
        destination_start:destination_end,
        destination_start:destination_end,
    ] = values[:, clipped_start:clipped_end, clipped_start:clipped_end]
    if not include_valid_pair_mask:
        return tile
    mask = np.zeros((1, tile_size, tile_size), dtype=np.float32)
    mask[
        :,
        destination_start:destination_end,
        destination_start:destination_end,
    ] = 1.0
    return np.concatenate([tile, mask], axis=0)


def longest_supported_run(track: np.ndarray, threshold: float = 0.5) -> tuple[int, int]:
    """Return the half-open longest run at or above a probability threshold."""

    values = np.asarray(track, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("track must be one-dimensional")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    supported = values >= threshold
    return _longest_supported_interval(supported, np.count_nonzero(supported))


def _longest_supported_interval(supported: np.ndarray, count: int) -> tuple[int, int]:
    if count == 0:
        return 0, 0
    if count == len(supported):
        return 0, len(supported)
    padded = np.empty(len(supported) + 2, dtype=bool)
    padded[0] = padded[-1] = False
    padded[1:-1] = supported
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    # argmax resolves ties to the first run, following sequence order.
    best = int(np.argmax(edges[1::2] - edges[::2])) * 2
    return int(edges[best]), int(edges[best + 1])


def _state_quantiles(values: np.ndarray) -> tuple[float, float]:
    n = len(values)
    middle_left, middle_right = (n - 1) // 2, n // 2
    rank = (n - 1) * 0.9
    left = int(rank)
    right = min(n - 1, left + 1)
    ordered = np.partition(values, (middle_left, middle_right, left, right))
    median = float((ordered[middle_left] + ordered[middle_right]) * 0.5)
    low, high = float(ordered[left]), float(ordered[right])
    weight = rank - left
    # NumPy's linear quantile interpolates from the nearer endpoint.
    if weight >= 0.5:
        q90 = high - (high - low) * (1.0 - weight)
    else:
        q90 = low + (high - low) * weight
    if (median == 0.0 or q90 == 0.0) and np.signbit(values).any():
        # Different choices of equal elements in partition can select opposite signed zeros.
        return float(np.median(values)), float(np.quantile(values, 0.9))
    return median, q90


def summarize_state_track(
    track: np.ndarray, threshold: float = 0.5
) -> dict[str, float | int]:
    """Summarize a residue-state probability track without changing its values."""

    values = np.asarray(track, dtype=np.float64)
    if values.ndim != 1 or not len(values):
        raise ValueError("track must be a non-empty one-dimensional array")
    if not (0.0 <= values.min() and values.max() <= 1.0):
        raise ValueError("state probabilities must be finite and in [0, 1]")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    supported = values >= threshold
    count = np.count_nonzero(supported)
    start, end = _longest_supported_interval(supported, count)
    median, q90 = _state_quantiles(values)
    return {
        "mean_probability": float(np.mean(values)),
        "median_probability": median,
        "q90_probability": q90,
        "supported_fraction": float(count / len(values)),
        "longest_run_start": start,
        "longest_run_end": end,
        "longest_run_length": end - start,
        "longest_run_fraction": float((end - start) / len(values)),
    }


def summarize_teacher_prediction(
    state_track: np.ndarray,
    teacher_track: np.ndarray,
    contract: Mapping,
) -> dict:
    """Create the compact per-protein teacher output retained in discovery runs."""

    names = tuple(str(value) for value in contract["teacher"]["class_names"])
    teacher_values = np.asarray(teacher_track, dtype=np.float64)
    if teacher_values.ndim != 2 or teacher_values.shape[1] != len(names):
        raise ValueError("teacher track dimensions disagree with the class contract")
    threshold = float(contract["output"]["state_threshold"])
    return {
        **summarize_state_track(state_track, threshold=threshold),
        "predicted_teacher_mean_class_probability": {
            name: float(value)
            for name, value in zip(names, teacher_values.mean(axis=0))
        },
    }
