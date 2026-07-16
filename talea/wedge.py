"""Position-retaining wedge features for periodic attention maps.

The existing diagonal Radon profile averages an attention matrix over sequence
position and preserves only residue separation.  This module instead scores a
candidate sequence interval against the finite support expected along harmonic
attention diagonals.  In midpoint/separation coordinates, that support is a
wedge: pairs at separation ``d`` can contribute only where both residues lie
inside the candidate interval.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np


def candidate_periods_from_windows(
    windows: Iterable[dict],
    minimum_window_size: int = 96,
    top_n: int = 8,
    alias_multipliers: tuple[float, ...] = (0.5, 1.0, 2.0),
    minimum_period: float = 5.0,
    maximum_period: float = 80.0,
    quantization: float = 0.5,
) -> tuple[float, ...]:
    """Generate label-free wedge periods from the strongest coarse windows."""
    if minimum_window_size < 1 or top_n < 1:
        raise ValueError("window size and top_n must be positive")
    if minimum_period <= 0 or maximum_period < minimum_period:
        raise ValueError("invalid period bounds")
    if quantization <= 0:
        raise ValueError("quantization must be positive")
    eligible = sorted(
        (
            row
            for row in windows
            if int(row["window_size"]) >= minimum_window_size
            and float(row["attention_fused"]["period"]) > 0
        ),
        key=lambda row: (
            float(row["attention_fused"]["score"]),
            int(row["attention_fused"].get("n_harmonics", 0)),
            -int(row["window_size"]),
        ),
        reverse=True,
    )[:top_n]
    result = set()
    for row in eligible:
        period = float(row["attention_fused"]["period"])
        for multiplier in alias_multipliers:
            candidate = period * multiplier
            candidate = round(candidate / quantization) * quantization
            if minimum_period <= candidate <= maximum_period:
                result.add(float(candidate))
    return tuple(sorted(result))


def robust_diagonal_tracks(
    matrix: np.ndarray,
    max_separation: int,
    baseline_quantile: float = 0.25,
    clip: float = 8.0,
) -> list[np.ndarray]:
    """Return spatially normalized upper-diagonal tracks.

    Index ``d`` contains values from ``matrix[i, i + d]``.  A low spatial
    quantile estimates background so that a repeat region may occupy more than
    half, but not nearly all, of a diagonal without being centered away.  The
    robust scale falls back to the standard deviation for sparse synthetic or
    nearly constant diagonals.
    """
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("attention matrix must be square")
    if not 0.0 <= baseline_quantile < 0.5:
        raise ValueError("baseline_quantile must be in [0, 0.5)")
    if max_separation < 1:
        raise ValueError("max_separation must be positive")
    if clip <= 0:
        raise ValueError("clip must be positive")

    length = len(values)
    maximum = min(int(max_separation), length - 1)
    tracks = [np.empty(0, dtype=np.float64) for _ in range(maximum + 1)]
    for separation in range(1, maximum + 1):
        diagonal = np.asarray(
            np.diagonal(values, offset=separation), dtype=np.float64
        )
        baseline = float(np.quantile(diagonal, baseline_quantile))
        centered = diagonal - baseline
        median = float(np.median(centered))
        scale = 1.4826 * float(np.median(np.abs(centered - median)))
        if scale < 1e-12:
            scale = float(np.std(centered))
        if scale < 1e-12:
            tracks[separation] = np.zeros_like(centered)
        else:
            tracks[separation] = np.clip(centered / scale, -clip, clip)
    return tracks


def fuse_diagonal_tracks(
    track_sets: Sequence[Sequence[np.ndarray]],
) -> list[np.ndarray]:
    """Mean-fuse normalized diagonal tracks from multiple attention layers."""
    if not track_sets:
        raise ValueError("at least one track set is required")
    maximum = min(len(tracks) for tracks in track_sets)
    result = [np.empty(0, dtype=np.float64) for _ in range(maximum)]
    for separation in range(1, maximum):
        lengths = {len(tracks[separation]) for tracks in track_sets}
        if len(lengths) != 1:
            raise ValueError("layer diagonal lengths differ")
        result[separation] = np.mean(
            np.stack([tracks[separation] for tracks in track_sets]), axis=0
        )
    return result


def phase_randomize_diagonal_tracks(
    tracks: Sequence[np.ndarray], rng: np.random.Generator
) -> list[np.ndarray]:
    """Independently circular-shift each separation track.

    This preserves every diagonal's value distribution and one-dimensional
    autocorrelation while destroying shared positional phase across
    separations.  A coherent finite-width wedge depends on that shared phase.
    """
    result = [np.empty(0, dtype=np.float64) for _ in range(len(tracks))]
    for separation in range(1, len(tracks)):
        values = np.asarray(tracks[separation], dtype=np.float64)
        if not len(values):
            continue
        shift = int(rng.integers(0, len(values)))
        result[separation] = np.roll(values, shift)
    return result


def empirical_upper_tail_pvalue(observed: float, null_values: Sequence[float]) -> float:
    """Finite-sample corrected upper-tail Monte Carlo p value."""
    values = np.asarray(null_values, dtype=np.float64)
    if values.ndim != 1 or not len(values):
        raise ValueError("null_values must be a non-empty one-dimensional sequence")
    if not np.isfinite(observed) or not np.all(np.isfinite(values)):
        raise ValueError("observed and null values must be finite")
    return float((1 + np.sum(values >= observed)) / (len(values) + 1))


def _offsets(
    center: float,
    tolerance: int,
    maximum: int,
) -> tuple[int, ...]:
    rounded = int(round(center))
    return tuple(
        value
        for value in range(rounded - tolerance, rounded + tolerance + 1)
        if 1 <= value <= maximum
    )


def wedge_start_scores(
    tracks: Sequence[np.ndarray],
    protein_length: int,
    period: float,
    width: int,
    tolerance: int = 1,
    max_harmonics: int = 8,
    min_harmonics: int = 2,
    interharmonic_controls: bool = True,
) -> np.ndarray:
    """Score every start for one period/width wedge template.

    Harmonic offsets receive weights ``1/sqrt(k)``.  When enabled, half-period
    offsets are included with opposite sign to suppress generic contact-dense
    regions.  Scores are normalized by the nominal independent-pair variance;
    this is a ranking statistic rather than a calibrated z score because
    neighboring attention entries are correlated.
    """
    if protein_length < 1:
        return np.empty(0, dtype=np.float64)
    if period <= 0:
        raise ValueError("period must be positive")
    if width < 1 or width > protein_length:
        raise ValueError("width must be between one and protein length")
    if tolerance < 0:
        raise ValueError("tolerance must be nonnegative")
    if max_harmonics < 1 or min_harmonics < 1:
        raise ValueError("harmonic counts must be positive")

    n_starts = protein_length - width + 1
    scores = np.zeros(n_starts, dtype=np.float64)
    variance = np.zeros(n_starts, dtype=np.float64)
    maximum = min(len(tracks) - 1, width - 1)
    retained_harmonics = 0

    def add_offsets(offsets: Iterable[int], weight: float) -> bool:
        members = tuple(
            separation
            for separation in offsets
            if separation < width
            and separation < len(tracks)
            and len(tracks[separation]) == protein_length - separation
        )
        if not members:
            return False
        member_weight = weight / len(members)
        for separation in members:
            track = np.asarray(tracks[separation], dtype=np.float64)
            included = width - separation
            prefix = np.concatenate(([0.0], np.cumsum(track)))
            sums = prefix[included : included + n_starts] - prefix[:n_starts]
            scores[:] += member_weight * sums
            variance[:] += (member_weight**2) * included
        return True

    for harmonic in range(1, max_harmonics + 1):
        center = harmonic * period
        if center > maximum + tolerance:
            break
        weight = 1.0 / np.sqrt(harmonic)
        if add_offsets(_offsets(center, tolerance, maximum), weight):
            retained_harmonics += 1
        if interharmonic_controls:
            control_center = (harmonic + 0.5) * period
            if control_center <= maximum + tolerance:
                add_offsets(_offsets(control_center, tolerance, maximum), -weight)

    if retained_harmonics < min_harmonics:
        return np.full(n_starts, -np.inf, dtype=np.float64)
    denominator = np.sqrt(np.maximum(variance, 1e-12))
    return scores / denominator


def best_wedge_interval(
    tracks: Sequence[np.ndarray],
    protein_length: int,
    periods: Iterable[float],
    widths: Iterable[int],
    tolerance: int = 1,
    max_harmonics: int = 8,
    min_harmonics: int = 2,
    interharmonic_controls: bool = True,
) -> dict:
    """Return the highest-scoring period/interval wedge call."""
    best: dict | None = None
    for period in sorted({float(value) for value in periods if value > 0}):
        for requested_width in sorted({int(value) for value in widths if value > 0}):
            width = min(requested_width, protein_length)
            scores = wedge_start_scores(
                tracks,
                protein_length,
                period,
                width,
                tolerance=tolerance,
                max_harmonics=max_harmonics,
                min_harmonics=min_harmonics,
                interharmonic_controls=interharmonic_controls,
            )
            if not len(scores) or not np.any(np.isfinite(scores)):
                continue
            start = int(np.nanargmax(scores))
            candidate = {
                "start": start,
                "end": start + width,
                "window_size": width,
                "period": period,
                "score": float(scores[start]),
                "tolerance": tolerance,
                "max_harmonics": max_harmonics,
                "min_harmonics": min_harmonics,
                "interharmonic_controls": interharmonic_controls,
            }
            if best is None or candidate["score"] > best["score"]:
                best = candidate
    if best is None:
        return {
            "start": 0,
            "end": 0,
            "window_size": 0,
            "period": -1.0,
            "score": float("-inf"),
        }
    return best


def _interval_iou(first: dict, second: dict) -> float:
    intersection = max(
        0,
        min(int(first["end"]), int(second["end"]))
        - max(int(first["start"]), int(second["start"])),
    )
    union = (
        int(first["end"])
        - int(first["start"])
        + int(second["end"])
        - int(second["start"])
        - intersection
    )
    return intersection / union if union else 0.0


def top_wedge_intervals(
    tracks: Sequence[np.ndarray],
    protein_length: int,
    periods: Iterable[float],
    widths: Iterable[int],
    top_k: int = 5,
    starts_per_template: int = 3,
    within_template_separation_fraction: float = 0.5,
    global_nms_iou: float = 0.3,
    tolerance: int = 1,
    max_harmonics: int = 8,
    min_harmonics: int = 2,
    interharmonic_controls: bool = True,
) -> list[dict]:
    """Return nonredundant high-scoring wedges from potentially multiple loci."""
    if top_k < 1 or starts_per_template < 1:
        raise ValueError("top_k and starts_per_template must be positive")
    if not 0.0 <= within_template_separation_fraction <= 1.0:
        raise ValueError("within-template separation fraction must be in [0, 1]")
    if not 0.0 <= global_nms_iou <= 1.0:
        raise ValueError("global_nms_iou must be in [0, 1]")

    candidates = []
    for period in sorted({float(value) for value in periods if value > 0}):
        for requested_width in sorted({int(value) for value in widths if value > 0}):
            width = min(requested_width, protein_length)
            scores = wedge_start_scores(
                tracks,
                protein_length,
                period,
                width,
                tolerance=tolerance,
                max_harmonics=max_harmonics,
                min_harmonics=min_harmonics,
                interharmonic_controls=interharmonic_controls,
            )
            working = np.asarray(scores, dtype=np.float64).copy()
            minimum_separation = max(
                1, int(round(width * within_template_separation_fraction))
            )
            for _ in range(starts_per_template):
                if not len(working) or not np.any(np.isfinite(working)):
                    break
                start = int(np.nanargmax(working))
                score = float(working[start])
                candidates.append(
                    {
                        "start": start,
                        "end": start + width,
                        "window_size": width,
                        "period": period,
                        "score": score,
                        "tolerance": tolerance,
                        "max_harmonics": max_harmonics,
                        "min_harmonics": min_harmonics,
                        "interharmonic_controls": interharmonic_controls,
                    }
                )
                left = max(0, start - minimum_separation + 1)
                right = min(len(working), start + minimum_separation)
                working[left:right] = -np.inf

    selected = []
    for candidate in sorted(
        candidates,
        key=lambda value: (
            float(value["score"]),
            -int(value["window_size"]),
        ),
        reverse=True,
    ):
        if all(_interval_iou(candidate, member) <= global_nms_iou for member in selected):
            candidate = {**candidate, "rank": len(selected) + 1}
            selected.append(candidate)
            if len(selected) == top_k:
                break
    return selected


def expand_interval_by_period(
    call: dict,
    protein_length: int,
    periods_per_flank: float = 0.5,
) -> dict:
    """Expand a compact wedge core symmetrically by a period-scaled amount."""
    if protein_length < 1:
        raise ValueError("protein_length must be positive")
    if periods_per_flank < 0:
        raise ValueError("periods_per_flank must be nonnegative")
    start = int(call["start"])
    end = int(call["end"])
    width = max(0, min(protein_length, end - start))
    period = max(0.0, float(call.get("period", 0.0)))
    target_width = min(
        protein_length,
        max(1, int(round(width + 2.0 * periods_per_flank * period))),
    )
    midpoint = (start + end) / 2
    expanded_start = int(round(midpoint - target_width / 2))
    expanded_start = max(0, min(expanded_start, protein_length - target_width))
    return {
        **call,
        "start": expanded_start,
        "end": expanded_start + target_width,
        "window_size": target_width,
        "pre_expansion_start": start,
        "pre_expansion_end": end,
        "periods_expanded_per_flank": periods_per_flank,
    }


def harmonic_residue_scores(
    tracks: Sequence[np.ndarray],
    protein_length: int,
    period: float,
    tolerance: int = 1,
    max_harmonics: int = 8,
    min_harmonics: int = 2,
    interharmonic_controls: bool = True,
) -> np.ndarray:
    """Project harmonic pair evidence back onto the participating residues.

    Each normalized diagonal value describes one residue pair.  This function
    assigns that evidence to both endpoints, preserving the chosen period but
    producing a one-dimensional boundary track.  Endpoint evidence is divided
    by its nominal variance so termini and unavailable long separations do not
    receive systematically smaller scores.
    """
    if protein_length < 1:
        return np.empty(0, dtype=np.float64)
    if period <= 0:
        raise ValueError("period must be positive")
    if tolerance < 0:
        raise ValueError("tolerance must be nonnegative")
    if max_harmonics < 1 or min_harmonics < 1:
        raise ValueError("harmonic counts must be positive")

    scores = np.zeros(protein_length, dtype=np.float64)
    variance = np.zeros(protein_length, dtype=np.float64)
    maximum = len(tracks) - 1
    retained_harmonics = 0

    def add_offsets(offsets: Iterable[int], weight: float) -> bool:
        members = tuple(
            separation
            for separation in offsets
            if separation < len(tracks)
            and len(tracks[separation]) == protein_length - separation
        )
        if not members:
            return False
        member_weight = weight / len(members)
        for separation in members:
            values = np.asarray(tracks[separation], dtype=np.float64)
            scores[: protein_length - separation] += member_weight * values
            scores[separation:] += member_weight * values
            variance[: protein_length - separation] += member_weight**2
            variance[separation:] += member_weight**2
        return True

    for harmonic in range(1, max_harmonics + 1):
        center = harmonic * period
        if center > maximum + tolerance:
            break
        weight = 1.0 / np.sqrt(harmonic)
        if add_offsets(_offsets(center, tolerance, maximum), weight):
            retained_harmonics += 1
        if interharmonic_controls:
            control_center = (harmonic + 0.5) * period
            if control_center <= maximum + tolerance:
                add_offsets(
                    _offsets(control_center, tolerance, maximum), -weight
                )

    if retained_harmonics < min_harmonics:
        return np.full(protein_length, -np.inf, dtype=np.float64)
    return scores / np.sqrt(np.maximum(variance, 1e-12))


def centered_moving_average(values: np.ndarray, width: int) -> np.ndarray:
    """Return an edge-preserving centered moving average."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if width < 1:
        raise ValueError("width must be positive")
    if not len(array) or width == 1:
        return array.copy()
    width = min(int(width), len(array))
    left = width // 2
    right = width - 1 - left
    padded = np.pad(array, (left, right), mode="edge")
    return np.convolve(padded, np.ones(width) / width, mode="valid")


def extend_interval_by_residue_evidence(
    call: dict,
    protein_length: int,
    residue_scores: np.ndarray,
    minimum_periods_per_flank: float = 0.5,
    smoothing_periods: float = 1.0,
    baseline_quantile: float = 0.4,
    threshold_fraction: float = 0.2,
    core_reference_quantile: float = 0.75,
    maximum_extra_periods: float | None = None,
) -> dict:
    """Grow a high-precision wedge core using period-specific residue evidence.

    The established half-period expansion is treated as a minimum.  Beyond
    that seed, left and right boundaries are chosen independently by the best
    positive cumulative evidence above a robust, core-relative threshold.
    This removes the forced symmetry of a fixed expansion while never
    contracting the validated compact core.
    """
    if protein_length < 1:
        raise ValueError("protein_length must be positive")
    values = np.asarray(residue_scores, dtype=np.float64)
    if values.shape != (protein_length,):
        raise ValueError("residue_scores length does not match protein_length")
    if not np.all(np.isfinite(values)):
        raise ValueError("residue_scores must be finite")
    if minimum_periods_per_flank < 0 or smoothing_periods <= 0:
        raise ValueError("period expansion parameters are invalid")
    if not 0.0 <= baseline_quantile <= 1.0:
        raise ValueError("baseline_quantile must be in [0, 1]")
    if not 0.0 <= threshold_fraction <= 1.0:
        raise ValueError("threshold_fraction must be in [0, 1]")
    if not 0.0 <= core_reference_quantile <= 1.0:
        raise ValueError("core_reference_quantile must be in [0, 1]")
    if maximum_extra_periods is not None and maximum_extra_periods < 0:
        raise ValueError("maximum_extra_periods must be nonnegative")

    period = float(call.get("period", 0.0))
    if period <= 0:
        raise ValueError("call must contain a positive period")
    core_start = max(0, min(int(call["start"]), protein_length))
    core_end = max(core_start, min(int(call["end"]), protein_length))
    if core_end <= core_start:
        raise ValueError("call interval must be non-empty")

    seed = expand_interval_by_period(
        call,
        protein_length,
        periods_per_flank=minimum_periods_per_flank,
    )
    smoothed = centered_moving_average(
        values, max(1, int(round(smoothing_periods * period)))
    )
    baseline = float(np.quantile(smoothed, baseline_quantile))
    reference = float(
        np.quantile(
            smoothed[core_start:core_end], core_reference_quantile
        )
    )
    threshold = baseline + threshold_fraction * (reference - baseline)
    adjusted = smoothed - threshold

    start = int(seed["start"])
    end = int(seed["end"])
    if maximum_extra_periods is None:
        maximum_left = start
        maximum_right = protein_length - end
    else:
        maximum_extra = int(round(maximum_extra_periods * period))
        maximum_left = min(start, maximum_extra)
        maximum_right = min(protein_length - end, maximum_extra)

    if reference > baseline and maximum_left:
        cumulative = np.cumsum(
            adjusted[start - maximum_left : start][::-1]
        )
        start -= int(np.argmax(np.concatenate(([0.0], cumulative))))
    if reference > baseline and maximum_right:
        cumulative = np.cumsum(adjusted[end : end + maximum_right])
        end += int(np.argmax(np.concatenate(([0.0], cumulative))))

    return {
        **call,
        "start": start,
        "end": end,
        "window_size": end - start,
        "pre_extension_start": int(call["start"]),
        "pre_extension_end": int(call["end"]),
        "seed_start": int(seed["start"]),
        "seed_end": int(seed["end"]),
        "minimum_periods_per_flank": minimum_periods_per_flank,
        "smoothing_periods": smoothing_periods,
        "baseline_quantile": baseline_quantile,
        "threshold_fraction": threshold_fraction,
        "core_reference_quantile": core_reference_quantile,
        "maximum_extra_periods": maximum_extra_periods,
        "residue_baseline": baseline,
        "residue_core_reference": reference,
        "residue_threshold": threshold,
    }


def local_stopping_extension_steps(
    adjusted_scores: Sequence[float], patience: int
) -> tuple[int, int]:
    """Choose positive cumulative extension before sustained local loss.

    Scores are ordered outward from the seed boundary.  The scan terminates
    after ``patience`` consecutive non-positive positions and returns the
    extension at the best cumulative evidence observed before that stop,
    together with the number of positions inspected.
    """
    values = np.asarray(adjusted_scores, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("adjusted_scores must be one-dimensional")
    if patience < 1:
        raise ValueError("patience must be positive")
    if not np.all(np.isfinite(values)):
        raise ValueError("adjusted_scores must be finite")
    cumulative = 0.0
    best = 0.0
    best_steps = 0
    nonpositive_run = 0
    inspected = 0
    for inspected, value in enumerate(values, start=1):
        cumulative += float(value)
        if cumulative > best:
            best = cumulative
            best_steps = inspected
        if value > 0:
            nonpositive_run = 0
        else:
            nonpositive_run += 1
            if nonpositive_run >= patience:
                break
    return best_steps, inspected


def extend_interval_by_locally_stopping_evidence(
    call: dict,
    protein_length: int,
    residue_scores: np.ndarray,
    minimum_periods_per_flank: float = 0.5,
    smoothing_periods: float = 1.0,
    baseline_quantile: float = 0.4,
    threshold_fraction: float = 0.2,
    core_reference_quantile: float = 0.75,
    patience_periods: float = 1.0,
    maximum_extra_periods: float = 4.0,
) -> dict:
    """Grow a wedge core until each flank shows sustained local signal loss.

    Unlike unlimited cumulative extension, this rule cannot bridge a long
    annotation-dark or nonperiodic gap merely because strong evidence occurs
    farther away.  The established half-period expansion remains the minimum.
    """
    if protein_length < 1:
        raise ValueError("protein_length must be positive")
    values = np.asarray(residue_scores, dtype=np.float64)
    if values.shape != (protein_length,) or not np.all(np.isfinite(values)):
        raise ValueError("residue_scores must be finite and match protein_length")
    if smoothing_periods <= 0 or patience_periods <= 0:
        raise ValueError("smoothing and patience periods must be positive")
    if maximum_extra_periods < 0:
        raise ValueError("maximum_extra_periods must be nonnegative")
    if not 0.0 <= baseline_quantile <= 1.0:
        raise ValueError("baseline_quantile must be in [0, 1]")
    if not 0.0 <= threshold_fraction <= 1.0:
        raise ValueError("threshold_fraction must be in [0, 1]")
    if not 0.0 <= core_reference_quantile <= 1.0:
        raise ValueError("core_reference_quantile must be in [0, 1]")

    period = float(call.get("period", 0.0))
    if period <= 0:
        raise ValueError("call must contain a positive period")
    core_start = max(0, min(int(call["start"]), protein_length))
    core_end = max(core_start, min(int(call["end"]), protein_length))
    if core_end <= core_start:
        raise ValueError("call interval must be non-empty")
    seed = expand_interval_by_period(
        call, protein_length, periods_per_flank=minimum_periods_per_flank
    )
    smoothed = centered_moving_average(
        values, max(1, int(round(smoothing_periods * period)))
    )
    baseline = float(np.quantile(smoothed, baseline_quantile))
    reference = float(
        np.quantile(smoothed[core_start:core_end], core_reference_quantile)
    )
    threshold = baseline + threshold_fraction * (reference - baseline)
    adjusted = smoothed - threshold
    patience = max(1, int(round(patience_periods * period)))
    maximum_extra = max(0, int(round(maximum_extra_periods * period)))

    start = int(seed["start"])
    end = int(seed["end"])
    left_limit = min(start, maximum_extra)
    right_limit = min(protein_length - end, maximum_extra)
    left_steps = right_steps = left_inspected = right_inspected = 0
    if reference > baseline and left_limit:
        left_steps, left_inspected = local_stopping_extension_steps(
            adjusted[start - left_limit : start][::-1], patience
        )
        start -= left_steps
    if reference > baseline and right_limit:
        right_steps, right_inspected = local_stopping_extension_steps(
            adjusted[end : end + right_limit], patience
        )
        end += right_steps

    return {
        **call,
        "start": start,
        "end": end,
        "window_size": end - start,
        "pre_extension_start": int(call["start"]),
        "pre_extension_end": int(call["end"]),
        "seed_start": int(seed["start"]),
        "seed_end": int(seed["end"]),
        "minimum_periods_per_flank": minimum_periods_per_flank,
        "smoothing_periods": smoothing_periods,
        "baseline_quantile": baseline_quantile,
        "threshold_fraction": threshold_fraction,
        "core_reference_quantile": core_reference_quantile,
        "patience_periods": patience_periods,
        "patience_residues": patience,
        "maximum_extra_periods": maximum_extra_periods,
        "maximum_extra_residues": maximum_extra,
        "left_extra_residues": left_steps,
        "right_extra_residues": right_steps,
        "left_positions_inspected": left_inspected,
        "right_positions_inspected": right_inspected,
        "residue_baseline": baseline,
        "residue_core_reference": reference,
        "residue_threshold": threshold,
    }


def dense_start_scores(
    tracks: Sequence[np.ndarray],
    protein_length: int,
    width: int,
    minimum_separation: int = 6,
    maximum_separation: int | None = None,
) -> np.ndarray:
    """Generic contact-density score for every interval start.

    This deliberately ignores harmonic spacing and is an important control for
    whether wedge localization reflects periodic geometry or merely a compact,
    attention-dense structural domain.
    """
    if width < 1 or width > protein_length:
        raise ValueError("width must be between one and protein length")
    maximum = min(len(tracks) - 1, width - 1)
    if maximum_separation is not None:
        maximum = min(maximum, int(maximum_separation))
    minimum = max(1, int(minimum_separation))
    n_starts = protein_length - width + 1
    scores = np.zeros(n_starts, dtype=np.float64)
    variance = np.zeros(n_starts, dtype=np.float64)
    for separation in range(minimum, maximum + 1):
        if len(tracks[separation]) != protein_length - separation:
            continue
        included = width - separation
        track = np.asarray(tracks[separation], dtype=np.float64)
        prefix = np.concatenate(([0.0], np.cumsum(track)))
        scores += prefix[included : included + n_starts] - prefix[:n_starts]
        variance += included
    if not np.any(variance):
        return np.full(n_starts, -np.inf, dtype=np.float64)
    return scores / np.sqrt(np.maximum(variance, 1e-12))


def best_dense_interval(
    tracks: Sequence[np.ndarray],
    protein_length: int,
    widths: Iterable[int],
    minimum_separation: int = 6,
    maximum_separation: int | None = None,
) -> dict:
    """Return the highest-scoring generic contact-density interval."""
    best: dict | None = None
    for requested_width in sorted({int(value) for value in widths if value > 0}):
        width = min(requested_width, protein_length)
        scores = dense_start_scores(
            tracks,
            protein_length,
            width,
            minimum_separation=minimum_separation,
            maximum_separation=maximum_separation,
        )
        if not len(scores) or not np.any(np.isfinite(scores)):
            continue
        start = int(np.nanargmax(scores))
        candidate = {
            "start": start,
            "end": start + width,
            "window_size": width,
            "period": -1.0,
            "score": float(scores[start]),
            "minimum_separation": minimum_separation,
            "maximum_separation": maximum_separation,
        }
        if best is None or candidate["score"] > best["score"]:
            best = candidate
    if best is None:
        return {
            "start": 0,
            "end": 0,
            "window_size": 0,
            "period": -1.0,
            "score": float("-inf"),
        }
    return best
