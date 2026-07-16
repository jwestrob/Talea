"""Label-free periodic unit registration inside a Talea wedge.

This is a deliberately transparent first approximation to TRUST's trace and
profile refinement. A Talea wedge supplies a repeat locus and period. We
search repeat phase, favor registrations whose adjacent-unit attention exceeds
two-unit-separated attention, and extend the registered array only while the
next unit boundary retains at least a fixed fraction of the core edge support.
No class label or RepeatsDB coordinate is consumed.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def integral_image(matrix: np.ndarray) -> np.ndarray:
    """Return a padded 2-D summed-area table."""

    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("attention matrix must be square")
    if not np.isfinite(values).all():
        raise ValueError("attention matrix contains nonfinite values")
    with np.errstate(all="ignore"):
        cumulative = values.cumsum(axis=0).cumsum(axis=1)
    if not np.isfinite(cumulative).all():
        raise FloatingPointError("attention integral image contains nonfinite values")
    return np.pad(cumulative, ((1, 0), (1, 0)))


def block_mean(
    prefix: np.ndarray,
    left: tuple[int, int],
    right: tuple[int, int],
) -> float:
    """Mean matrix value in the Cartesian product of two half-open intervals."""

    left_start, left_end = left
    right_start, right_end = right
    if not (
        0 <= left_start < left_end < prefix.shape[0]
        and 0 <= right_start < right_end < prefix.shape[1]
    ):
        raise ValueError("block interval lies outside integral image")
    total = (
        prefix[left_end, right_end]
        - prefix[left_start, right_end]
        - prefix[left_end, right_start]
        + prefix[left_start, right_start]
    )
    return float(total / ((left_end - left_start) * (right_end - right_start)))


def periodic_tiles(
    sequence_length: int,
    period: float,
    offset: float,
    minimum_terminal_fraction: float = 0.45,
) -> list[tuple[int, int]]:
    """Tile a sequence at one fractional phase, retaining substantial ends."""

    if sequence_length <= 0 or period <= 0:
        raise ValueError("sequence length and period must be positive")
    if not (0.0 <= minimum_terminal_fraction <= 1.0):
        raise ValueError("minimum terminal fraction must be in [0, 1]")
    position = float(offset)
    while position > 0:
        position -= period
    edges: list[int] = []
    while position < sequence_length:
        edges.append(int(round(position)))
        position += period
    edges.append(int(round(position)))
    clipped = sorted({max(0, min(sequence_length, edge)) for edge in edges})
    minimum_width = minimum_terminal_fraction * period
    return [
        (start, end)
        for start, end in zip(clipped, clipped[1:])
        if end > start and end - start >= minimum_width
    ]


def _core_unit_indices(
    units: Sequence[tuple[int, int]], core_start: int, core_end: int
) -> np.ndarray:
    centers = np.asarray([(start + end) / 2 for start, end in units])
    indices = np.flatnonzero((centers >= core_start) & (centers <= core_end))
    if len(indices) < 3:
        closest = np.argsort(np.abs(centers - (core_start + core_end) / 2))
        indices = np.sort(closest[: min(3, len(units))])
    if len(indices) < 3 or np.any(np.diff(indices) != 1):
        return np.asarray([], dtype=int)
    return indices


def register_periodic_units(
    attention: np.ndarray,
    period: float,
    core_start: int,
    core_end: int,
    *,
    extension_fraction: float = 0.5,
    phase_step: float = 1.0,
    minimum_terminal_fraction: float = 0.45,
    edge_cv_penalty: float = 0.1,
) -> dict:
    """Register and locally extend equal-period units without labelled truth."""

    values = np.asarray(attention, dtype=np.float64)
    sequence_length = values.shape[0]
    if not (0 <= core_start < core_end <= sequence_length):
        raise ValueError("core interval lies outside the attention matrix")
    if not (0 <= extension_fraction):
        raise ValueError("extension fraction must be nonnegative")
    if phase_step <= 0:
        raise ValueError("phase step must be positive")
    prefix = integral_image(values)
    candidates: list[dict] = []
    offsets = np.arange(0.0, max(float(period), phase_step), phase_step)
    for offset in offsets:
        units = periodic_tiles(
            sequence_length,
            period,
            float(offset),
            minimum_terminal_fraction=minimum_terminal_fraction,
        )
        core_indices = _core_unit_indices(units, core_start, core_end)
        if len(core_indices) < 3:
            continue
        edges = np.asarray(
            [
                block_mean(prefix, units[index], units[index + 1])
                for index in range(len(units) - 1)
            ],
            dtype=np.float64,
        )
        core_edges = edges[core_indices[0] : core_indices[-1]]
        core_reference = float(np.median(core_edges))
        background = np.asarray(
            [
                block_mean(prefix, units[index], units[index + 2])
                for index in range(
                    max(0, int(core_indices[0]) - 1),
                    min(len(units) - 2, int(core_indices[-1])),
                )
            ],
            dtype=np.float64,
        )
        background_reference = float(np.median(background)) if len(background) else 0.0
        epsilon = max(abs(core_reference) * 1e-9, 1e-12)
        adjacent_contrast = float(
            np.log((core_reference + epsilon) / (background_reference + epsilon))
        )
        edge_cv = float(np.std(core_edges) / (np.mean(core_edges) + epsilon))
        objective = adjacent_contrast - edge_cv_penalty * edge_cv
        candidates.append(
            {
                "objective": objective,
                "offset": float(offset),
                "units": units,
                "core_indices": core_indices,
                "edge_scores": edges,
                "core_reference": core_reference,
                "background_reference": background_reference,
                "adjacent_contrast": adjacent_contrast,
                "edge_cv": edge_cv,
            }
        )
    if not candidates:
        raise ValueError("no phase produced three consecutive core units")
    candidates.sort(key=lambda row: (-row["objective"], row["offset"]))
    best = candidates[0]
    left = int(best["core_indices"][0])
    right = int(best["core_indices"][-1])
    threshold = extension_fraction * max(float(best["core_reference"]), 1e-12)
    while left > 0 and best["edge_scores"][left - 1] >= threshold:
        left -= 1
    while right < len(best["units"]) - 1 and best["edge_scores"][right] >= threshold:
        right += 1
    selected = best["units"][left : right + 1]
    second_objective = candidates[1]["objective"] if len(candidates) > 1 else None
    return {
        "period": float(period),
        "core_start": int(core_start),
        "core_end": int(core_end),
        "phase_offset": best["offset"],
        "phase_objective": best["objective"],
        "phase_second_objective": second_objective,
        "phase_ambiguity_margin": (
            None if second_objective is None else best["objective"] - second_objective
        ),
        "adjacent_contrast": best["adjacent_contrast"],
        "core_edge_cv": best["edge_cv"],
        "core_edge_reference": best["core_reference"],
        "two_unit_background_reference": best["background_reference"],
        "extension_threshold": threshold,
        "extension_fraction": float(extension_fraction),
        "units": [
            {"start": int(start), "end": int(end)} for start, end in selected
        ],
        "n_units": len(selected),
        "start": int(selected[0][0]),
        "end": int(selected[-1][1]),
    }


def registered_closure_score(attention: np.ndarray, units: Sequence[dict]) -> dict:
    """Score an open path by terminal attention relative to adjacent units."""

    intervals = [(int(unit["start"]), int(unit["end"])) for unit in units]
    if len(intervals) < 3:
        raise ValueError("closure score requires at least three units")
    prefix = integral_image(attention)
    adjacent = np.asarray(
        [
            block_mean(prefix, intervals[index], intervals[index + 1])
            for index in range(len(intervals) - 1)
        ]
    )
    adjacent_median = float(np.median(adjacent))
    closure = block_mean(prefix, intervals[-1], intervals[0])
    epsilon = max(abs(adjacent_median) * 1e-6, 1e-12)
    closure_log_ratio = float(
        np.log((closure + epsilon) / (adjacent_median + epsilon))
    )
    return {
        "closure_attention": closure,
        "adjacent_attention_median": adjacent_median,
        "closure_log_ratio": closure_log_ratio,
        "open_path_score": -closure_log_ratio,
    }


def scan_plausible_boundary_pairs(attention: np.ndarray, registration: dict) -> dict:
    """Scan every period-consistent span containing the registered wedge core.

    This converts boundary uncertainty into an explicit hypothesis set.  The
    minimum open-path score is the strongest cyclic-closure evidence among
    those hypotheses.  Candidate count is retained so downstream calibration
    can correct the extreme-value statistic rather than hiding its multiplicity.
    """

    values = np.asarray(attention, dtype=np.float64)
    units = periodic_tiles(
        values.shape[0],
        float(registration["period"]),
        float(registration["phase_offset"]),
        minimum_terminal_fraction=0.45,
    )
    core_indices = _core_unit_indices(
        units, int(registration["core_start"]), int(registration["core_end"])
    )
    if len(core_indices) < 3:
        raise ValueError("registered core does not contain three consecutive tiles")
    prefix = integral_image(values)
    edge_scores = np.asarray(
        [
            block_mean(prefix, units[index], units[index + 1])
            for index in range(len(units) - 1)
        ]
    )
    candidates = []
    false_scan_candidates = []
    for left in range(len(units)):
        for right in range(left + 2, len(units)):
            adjacent = float(np.median(edge_scores[left:right]))
            closure = block_mean(prefix, units[right], units[left])
            epsilon = max(abs(adjacent) * 1e-6, 1e-12)
            open_score = float(-np.log((closure + epsilon) / (adjacent + epsilon)))
            row = {
                "left_unit_index": left,
                "right_unit_index": right,
                "start": int(units[left][0]),
                "end": int(units[right][1]),
                "n_units": right - left + 1,
                "open_path_score": open_score,
                "closure_attention": closure,
                "adjacent_attention_median": adjacent,
            }
            if left <= core_indices[0] and right >= core_indices[-1]:
                candidates.append(row)
            else:
                false_scan_candidates.append(row)
    candidates.sort(
        key=lambda row: (
            row["open_path_score"],
            row["start"],
            row["end"],
        )
    )
    scores = np.asarray([row["open_path_score"] for row in candidates])
    false_scan_candidates.sort(
        key=lambda row: (
            row["open_path_score"],
            row["start"],
            row["end"],
        )
    )
    return {
        "candidate_count": len(candidates),
        "open_path_score_minimum": float(scores[0]),
        "open_path_score_q05": float(np.quantile(scores, 0.05)),
        "open_path_score_q10": float(np.quantile(scores, 0.10)),
        "open_path_score_median": float(np.median(scores)),
        "strongest_closure_candidate": candidates[0],
        "noncore_false_scan_candidate_count": len(false_scan_candidates),
        "noncore_false_scan_open_path_score_minimum": (
            float(false_scan_candidates[0]["open_path_score"])
            if false_scan_candidates
            else None
        ),
        "noncore_false_scan_strongest_candidate": (
            false_scan_candidates[0] if false_scan_candidates else None
        ),
        "core_first_unit_index": int(core_indices[0]),
        "core_last_unit_index": int(core_indices[-1]),
        "tiled_unit_count": len(units),
    }
