"""Phase-aware state-space placement of repeat units.

Talea's rhythm branch estimates a repeat period ``P`` and a registered extent,
then tiles the extent with an equal-period comb at one phase offset. The period
is usually right; the offset is where most boundary error lives. This module
re-places the units inside the same extent:

1. Phase vote. Three linear probes on the layer-35 hidden state give per-residue
   helix, strand and loop probabilities. Each track is demodulated at ``P`` in a
   Hann window of +/- ``P`` residues, and one complex coefficient per channel
   maps the demodulated signals to a vote ``t(i) ~ exp(2 pi i phi_i)``, where
   ``phi_i`` is the residue's position within its unit.
2. Decode. A hidden semi-Markov model walks the extent unit by unit. The hidden
   state is the current unit length (0.6 P to 1.6 P). A unit ``[b, b + l)``
   scores the agreement of its votes with a unit starting at ``b`` plus fused
   attention stacking evidence at lag ``l``; consecutive units may change length
   by up to 0.3 P at a cost per residue; the first boundary is free. Viterbi
   returns the units.

The detection score and every topology diagnostic are computed before this step
and are unchanged by it. Parameters live in ``assets/phase_ssm.json``.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence

import numpy as np


PROBE_CLASSES = ("helix", "strand", "loop")


def probe_parameters(asset: Mapping) -> tuple[np.ndarray, np.ndarray]:
    """Return the folded probe map as ``(weights [hidden x 3], bias [3])``."""

    probes = asset["probes"]
    if tuple(probes["classes"]) != PROBE_CLASSES:
        raise ValueError("phase-SSM probe classes disagree with the runtime")
    weights = np.asarray(probes["weights"], dtype=np.float64).T
    bias = np.asarray(probes["bias"], dtype=np.float64)
    if weights.shape[1] != len(PROBE_CLASSES) or bias.shape != (len(PROBE_CLASSES),):
        raise ValueError("phase-SSM probe parameters have the wrong shape")
    return weights, bias


def runtime_config(asset: Mapping) -> dict:
    """The decoder-side subset of the asset that postprocess workers need."""

    coefficients = asset["phase_vote"]["coefficients"]
    return {
        "coefficients": [
            [float(coefficients[name]["real"]), float(coefficients[name]["imag"])]
            for name in PROBE_CLASSES
        ],
        "decoder": dict(asset["decoder"]),
    }


def demodulate(values: np.ndarray, period: float) -> np.ndarray:
    """Local Fourier coefficient at frequency ``1 / period`` for every residue.

    ``z(i) = sum_j w(j) (x(i + j) - m(i)) exp(-2 pi i j / P) / sum_j w(j)`` with a
    Hann window ``w`` over ``|j| <= round(P)`` and ``m(i)`` the window-weighted
    local mean. Windows are truncated at the sequence ends.
    """

    x = np.asarray(values, dtype=np.float64)
    half = int(round(period))
    offsets = np.arange(-half, half + 1)
    window = 0.5 * (1 + np.cos(np.pi * offsets / (half + 1)))
    kernel = window * np.exp(-2j * np.pi * offsets / period)

    def correlate(signal: np.ndarray, weights: np.ndarray) -> np.ndarray:
        # sum_j weights(j) signal(i + j), centered and cropped to the signal
        full = np.convolve(signal, weights[::-1], mode="full")
        return full[half : half + len(signal)]

    ones = np.ones_like(x)
    numerator = correlate(x, kernel)
    window_sum = correlate(ones, window)
    mean = correlate(x, window) / np.maximum(window_sum, 1e-9)
    kernel_sum = correlate(ones, kernel)
    return (numerator - mean * kernel_sum) / np.maximum(window_sum, 1e-9)


def phase_votes(
    probabilities: np.ndarray, period: float, coefficients: Sequence[Sequence[float]]
) -> np.ndarray:
    """Complex per-residue phase vote from the ``[L x 3]`` probe probabilities."""

    tracks = np.asarray(probabilities, dtype=np.float64)
    if tracks.ndim != 2 or tracks.shape[1] != len(PROBE_CLASSES):
        raise ValueError("probe probabilities must have shape [length, 3]")
    beta = np.asarray([complex(real, imag) for real, imag in coefficients])
    votes = np.zeros(len(tracks), dtype=np.complex128)
    for channel in range(tracks.shape[1]):
        x = tracks[:, channel]
        deviation = x.std()
        x = (x - x.mean()) / deviation if deviation > 1e-12 else x * 0
        votes += beta[channel] * demodulate(x, period)
    return votes


def lag_evidence(
    fused_tracks: Sequence[np.ndarray], length: int, maximum_lag: int
) -> np.ndarray:
    """``V[l, i]``: mean fused diagonal z of residue ``i`` with ``i +/- l``.

    ``fused_tracks`` is the wedge call's layer-fused robust diagonal tracks
    (``fused_tracks[l]`` has length ``length - l``). Rows beyond the available
    separations stay zero.
    """

    evidence = np.zeros((maximum_lag + 1, length), dtype=np.float64)
    top = min(maximum_lag, len(fused_tracks) - 1)
    for lag in range(1, top + 1):
        track = np.asarray(fused_tracks[lag], dtype=np.float64)
        total = np.zeros(length)
        count = np.zeros(length)
        total[: length - lag] += track
        count[: length - lag] += 1
        total[lag:] += track
        count[lag:] += 1
        evidence[lag] = np.where(count > 0, total / np.maximum(count, 1), 0.0)
    return evidence


def length_emissions(
    evidence: np.ndarray, minimum_length: int, maximum_length: int
) -> np.ndarray:
    """Per-residue stacking evidence for each unit length, background-centered.

    Row ``r`` holds the evidence at lag ``minimum_length + r`` minus the median
    over all candidate lags at that residue.
    """

    top = evidence.shape[0] - 1
    rows = [
        evidence[lag] if lag <= top else np.zeros(evidence.shape[1])
        for lag in range(minimum_length, maximum_length + 1)
    ]
    emissions = np.stack(rows)
    return emissions - np.median(emissions, axis=0, keepdims=True)


def unit_tables(
    votes: np.ndarray,
    emissions: np.ndarray,
    minimum_length: int,
    phase_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Score every candidate unit over an extent of ``n`` residues.

    Returns ``(full, lead, tail)``, each ``[(n + 1) x R]`` for unit lengths
    ``minimum_length + r``:

    * ``full[b, r]``: a complete unit ``[b, b + l)``;
    * ``lead[b, r]``: residues ``[0, b)`` scored as the last ``b`` residues of a
      unit of length ``l`` (defined for ``b < l``);
    * ``tail[b, r]``: residues ``[b, n)`` scored as the first residues of a unit
      of length ``l`` (defined for ``n - b < l``).

    A residue at position ``j`` of a unit of length ``l`` contributes
    ``phase_weight * Re(t * exp(-2 pi i j / l)) + E[r]``. Sums use cumulative
    sums of ``t(i) exp(-2 pi i i / l)``, which make every table ``O(n)`` per length.
    """

    n = len(votes)
    lengths = minimum_length + np.arange(emissions.shape[0])
    full = np.full((n + 1, len(lengths)), -np.inf)
    lead = np.full((n + 1, len(lengths)), -np.inf)
    tail = np.full((n + 1, len(lengths)), -np.inf)
    positions = np.arange(n + 1)
    for r, unit_length in enumerate(lengths):
        rotation = np.exp(-2j * np.pi * np.arange(n) / unit_length)
        phase_sum = np.concatenate(([0.0], np.cumsum(votes * rotation)))
        emission_sum = np.concatenate(([0.0], np.cumsum(emissions[r])))
        realign = np.exp(2j * np.pi * positions / unit_length)

        starts = positions[: max(0, n - unit_length + 1)]
        full[starts, r] = phase_weight * np.real(
            realign[starts] * (phase_sum[starts + unit_length] - phase_sum[starts])
        ) + (emission_sum[starts + unit_length] - emission_sum[starts])

        heads = positions[: min(unit_length, n + 1)]
        lead[heads, r] = (
            phase_weight * np.real(realign[heads] * phase_sum[heads])
            + emission_sum[heads]
        )

        ends = positions[max(0, n - unit_length + 1) :]
        tail[ends, r] = phase_weight * np.real(
            realign[ends] * (phase_sum[n] - phase_sum[ends])
        ) + (emission_sum[n] - emission_sum[ends])
    return full, lead, tail


def viterbi_units(
    full: np.ndarray,
    lead: np.ndarray,
    tail: np.ndarray,
    minimum_length: int,
    length_change_penalty: float,
    maximum_step: int,
) -> tuple[list[int], list[int]]:
    """Best segmentation of the extent into consecutive full units.

    Returns the full-unit boundaries (first unit start to last unit end) and the
    unit lengths in order. Residues before the first boundary are scored as the
    tail of a preceding unit and residues after the last as the head of a
    following one. Ties resolve to the earliest position, then the shortest
    length.
    """

    n = full.shape[0] - 1
    count = full.shape[1]
    lengths = minimum_length + np.arange(count)
    scores = np.full((n + 1, count), -np.inf)
    back = np.full((n + 1, count), -1, dtype=np.int64)
    for r, unit_length in enumerate(lengths):
        offsets = np.arange(min(unit_length, n - unit_length + 1))
        scores[offsets + unit_length, r] = lead[offsets, r] + full[offsets, r]
        back[offsets + unit_length, r] = -2
    step = np.abs(lengths[:, None] - lengths[None, :])
    penalty = np.where(
        np.abs(np.arange(count)[:, None] - np.arange(count)[None, :]) <= maximum_step,
        length_change_penalty * step,
        np.inf,
    )
    columns = np.arange(count)
    for position in range(1, n + 1):
        current = scores[position]
        if not np.isfinite(current).any():
            continue
        candidates = current[:, None] - penalty
        previous = np.argmax(candidates, axis=0)
        best = candidates[previous, columns]
        following = position + lengths
        valid = np.nonzero((following <= n) & np.isfinite(best))[0]
        if not len(valid):
            continue
        proposal = best[valid] + full[position, valid]
        targets = following[valid]
        better = proposal > scores[targets, valid]
        scores[targets[better], valid[better]] = proposal[better]
        back[targets[better], valid[better]] = previous[valid[better]]
    remainder = n - np.arange(n + 1)[:, None]
    final = np.where(
        (remainder < lengths[None, :]) & np.isfinite(scores), scores + tail, -np.inf
    )
    final[0] = -np.inf
    if not np.isfinite(final).any():
        return [], []
    end, r = np.unravel_index(int(np.argmax(final)), final.shape)
    boundaries = [int(end)]
    unit_lengths = []
    position = int(end)
    while True:
        unit_lengths.append(int(lengths[r]))
        previous = int(back[position, r])
        position -= int(lengths[r])
        boundaries.append(position)
        if previous == -2:
            break
        r = previous
    return boundaries[::-1], unit_lengths[::-1]


def decode_units(
    votes: np.ndarray,
    fused_tracks: Sequence[np.ndarray],
    period: float,
    start: int,
    end: int,
    config: Mapping,
    start_scores: np.ndarray | None = None,
    start_weight: float = 0.0,
) -> list[dict]:
    """Decode units over ``[start, end)`` from whole-protein votes and tracks.

    With ``start_scores`` (a per-residue unit-start logit) and a nonzero
    ``start_weight``, every unit start placed by the path, including the end of
    the last full unit, adds ``start_weight * start_scores[b]``.
    """

    decoder = config["decoder"]
    length = len(votes)
    minimum_length = max(
        int(decoder["minimum_length"]),
        int(math.floor(float(decoder["minimum_length_fraction"]) * period)),
    )
    maximum_length = min(
        int(decoder["maximum_length"]),
        int(math.ceil(float(decoder["maximum_length_fraction"]) * period)),
    )
    if end - start < minimum_length or maximum_length < minimum_length:
        return []
    maximum_step = max(
        int(decoder["minimum_step"]),
        int(round(float(decoder["maximum_step_fraction"]) * period)),
    )
    evidence = lag_evidence(fused_tracks, length, int(decoder["attention_lag_maximum"]))
    emissions = length_emissions(evidence, minimum_length, maximum_length)
    full, lead, tail = unit_tables(
        votes[start:end],
        emissions[:, start:end],
        minimum_length,
        float(decoder["phase_weight"]),
    )
    if start_scores is not None and start_weight:
        scores = np.asarray(start_scores, dtype=np.float64)
        closing = scores[end] if end < length else 0.0
        bonus = start_weight * np.append(scores[start:end], closing)[:, None]
        full, tail = full + bonus, tail + bonus
    boundaries, unit_lengths = viterbi_units(
        full,
        lead,
        tail,
        minimum_length,
        float(decoder["length_change_penalty"]),
        maximum_step,
    )
    if not boundaries:
        return []
    terminal = float(decoder["terminal_fraction"])
    units = [
        {"start": start + left, "end": start + right}
        for left, right in zip(boundaries[:-1], boundaries[1:])
    ]
    if boundaries[0] >= terminal * unit_lengths[0]:
        units.insert(0, {"start": start, "end": start + boundaries[0]})
    if (end - start) - boundaries[-1] >= terminal * unit_lengths[-1]:
        units.append({"start": start + boundaries[-1], "end": end})
    return units


def place_units(
    registration: Mapping,
    probabilities: np.ndarray,
    fused_tracks: Sequence[np.ndarray],
    config: Mapping,
) -> list[dict] | None:
    """Phase-SSM units for a comb registration, or ``None`` to keep the comb.

    The decode runs over the registration's extent at its period. Fewer than
    ``minimum_units`` decoded units returns ``None``.
    """

    period = float(registration["period"])
    votes = phase_votes(probabilities, period, config["coefficients"])
    units = decode_units(
        votes,
        fused_tracks,
        period,
        int(registration["start"]),
        int(registration["end"]),
        config,
    )
    if len(units) < int(config["decoder"]["minimum_units"]):
        return None
    return units


BOUNDARY_PROBE_CLASSES = ("unit_start", "repeat_region")


def boundary_probe_parameters(asset: Mapping) -> tuple[np.ndarray, np.ndarray]:
    """Return the start-trim probe map as ``(weights [hidden x 2], bias [2])``."""

    probes = asset["probes"]
    if tuple(probes["classes"]) != BOUNDARY_PROBE_CLASSES:
        raise ValueError("start-trim probe classes disagree with the runtime")
    weights = np.asarray(probes["weights"], dtype=np.float64).T
    bias = np.asarray(probes["bias"], dtype=np.float64)
    if weights.shape[1] != len(BOUNDARY_PROBE_CLASSES) or bias.shape != (2,):
        raise ValueError("start-trim probe parameters have the wrong shape")
    return weights, bias


def start_trim_config(asset: Mapping) -> dict:
    """The decoder-side subset of the start-trim asset."""

    return {"decoder": dict(asset["decoder"])}


def trim_units(
    units: list[dict], region_scores: np.ndarray, threshold: float, minimum_units: int
) -> list[dict]:
    """Drop end units whose mean repeat-region logit is below ``threshold``.

    The weaker end goes first; at least ``minimum_units`` units are kept.
    """

    scores = np.asarray(region_scores, dtype=np.float64)
    kept = list(units)
    while len(kept) > minimum_units:
        first = float(np.mean(scores[kept[0]["start"] : kept[0]["end"]]))
        last = float(np.mean(scores[kept[-1]["start"] : kept[-1]["end"]]))
        if min(first, last) >= threshold:
            break
        kept = kept[1:] if first <= last else kept[:-1]
    return kept


def place_units_start_trim(
    registration: Mapping,
    probabilities: np.ndarray,
    boundary_scores: np.ndarray,
    fused_tracks: Sequence[np.ndarray],
    phase_config: Mapping,
    trim_config: Mapping,
) -> tuple[list[dict], str]:
    """Start-trim units for a comb registration and the placement label.

    Extents whose mean helix probability reaches ``helix_threshold`` decode
    with the helix settings (by default phase votes off and the unit-start
    boundary term on); the rest keep the phase-SSM decode. A decode with fewer
    than ``minimum_units`` units falls back to the comb. The units are then
    trimmed with the repeat-region logits.
    """

    settings = trim_config["decoder"]
    period = float(registration["period"])
    start, end = int(registration["start"]), int(registration["end"])
    probabilities = np.asarray(probabilities, dtype=np.float64)
    scores = np.asarray(boundary_scores, dtype=np.float64)
    helix = float(probabilities[start:end, 0].mean()) if end > start else 0.0
    side = settings["helix" if helix >= float(settings["helix_threshold"]) else "strand"]
    config = {
        "decoder": dict(
            phase_config["decoder"], phase_weight=float(side["phase_weight"])
        )
    }
    votes = phase_votes(probabilities, period, phase_config["coefficients"])
    units = decode_units(
        votes,
        fused_tracks,
        period,
        start,
        end,
        config,
        start_scores=scores[:, 0],
        start_weight=float(side["start_weight"]),
    )
    minimum_units = int(settings["minimum_units"])
    label = "start_trim"
    if len(units) < minimum_units:
        units = [
            {"start": int(unit["start"]), "end": int(unit["end"])}
            for unit in registration["units"]
        ]
        label = "start_trim_comb"
    units = trim_units(
        units, scores[:, 1], float(side["trim_threshold"]), minimum_units
    )
    return units, label


def apply_placement(
    registration: dict, units: list[dict] | None, method: str = "phase_ssm"
) -> dict:
    """Return the registration with placed units, keeping the comb alongside."""

    placed = dict(registration)
    placed["comb_units"] = registration["units"]
    placed["comb_start"] = registration["start"]
    placed["comb_end"] = registration["end"]
    if units is None:
        placed["unit_placement"] = "comb_fallback"
        return placed
    placed["unit_placement"] = method
    placed["units"] = [
        {"start": int(unit["start"]), "end": int(unit["end"])} for unit in units
    ]
    placed["n_units"] = len(units)
    placed["start"] = int(units[0]["start"])
    placed["end"] = int(units[-1]["end"])
    placed["unit_length_median"] = float(
        statistics.median(unit["end"] - unit["start"] for unit in units)
    )
    return placed
