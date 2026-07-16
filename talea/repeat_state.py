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


def robust_attention_channel(
    matrix: np.ndarray,
    quantile: float = 0.95,
    clip: float = 8.0,
) -> np.ndarray:
    """Normalize a nonnegative APC attention matrix without using labels.

    Positive entries are scaled by a within-protein upper quantile, transformed
    with ``log1p``, and clipped.  This removes most protein-length and raw-layer
    scale variation while retaining sparse high-attention texture.
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
    positive = values[values > 0]
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
) -> np.ndarray:
    """Stack consistently sized robust attention channels."""

    if not matrices:
        raise ValueError("at least one attention matrix is required")
    channels = [
        robust_attention_channel(matrix, quantile=quantile, clip=clip)
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
    best_start = best_end = start = 0
    inside = False
    for index, supported in enumerate(values >= threshold):
        if supported and not inside:
            start = index
            inside = True
        if inside and (not supported or index == len(values) - 1):
            end = index if not supported else index + 1
            if end - start > best_end - best_start:
                best_start, best_end = start, end
            inside = False
    return best_start, best_end


def summarize_state_track(
    track: np.ndarray, threshold: float = 0.5
) -> dict[str, float | int]:
    """Summarize a residue-state probability track without changing its values."""

    values = np.asarray(track, dtype=np.float64)
    if values.ndim != 1 or not len(values):
        raise ValueError("track must be a non-empty one-dimensional array")
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("state probabilities must be finite and in [0, 1]")
    start, end = longest_supported_run(values, threshold=threshold)
    return {
        "mean_probability": float(np.mean(values)),
        "median_probability": float(np.median(values)),
        "q90_probability": float(np.quantile(values, 0.9)),
        "supported_fraction": float(np.mean(values >= threshold)),
        "longest_run_start": start,
        "longest_run_end": end,
        "longest_run_length": end - start,
        "longest_run_fraction": float((end - start) / len(values)),
    }
