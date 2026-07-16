"""Harmonic spectra of physicochemical and exact-sequence signals.

This module intentionally emits continuous, uncalibrated evidence. Thresholds
belong in a benchmark-calibration layer, not in the signal extractor.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from scipy.ndimage import median_filter

from talea.sequence import CANONICAL_AMINO_ACIDS, require_canonical_sequence


AA_ORDER = CANONICAL_AMINO_ACIDS
AA_INDEX = {amino_acid: index for index, amino_acid in enumerate(AA_ORDER)}

# Atchley et al. (2005), Factors I-V. The columns span major covarying
# physicochemical attributes and are used together rather than cherry-picking
# a projection for any one solenoid family.
ATCHLEY = {
    "A": (-0.591, -1.302, -0.733, 1.570, -0.146),
    "C": (-1.343, 0.465, -0.862, -1.020, -0.255),
    "D": (-0.798, 0.493, -0.686, -0.982, 1.564),
    "E": (-0.730, 0.063, -0.267, -0.837, 1.628),
    "F": (1.889, -0.273, 1.629, 0.801, -0.542),
    "G": (-0.270, 1.330, -0.871, -0.093, 0.179),
    "H": (0.608, 0.049, -0.007, -0.163, -0.044),
    "I": (0.825, -1.628, 0.544, 0.034, -0.680),
    "K": (1.139, 1.165, 0.283, -0.121, 0.778),
    "L": (1.056, -1.242, 0.693, -0.554, -0.347),
    "M": (0.940, -0.420, -0.032, -0.417, -0.273),
    "N": (-0.375, 1.020, -0.826, -0.014, 0.082),
    "P": (0.222, 2.043, -0.521, -0.098, 0.269),
    "Q": (0.093, 0.828, -0.179, -0.776, -0.165),
    "R": (1.491, 0.841, 1.063, 0.150, 0.360),
    "S": (-0.345, 0.443, -0.922, -0.410, -0.179),
    "T": (-0.199, 0.514, -0.598, 1.142, -0.121),
    "V": (0.282, -1.437, 0.379, 0.990, -0.623),
    "W": (2.213, -0.188, 2.067, 1.338, -0.450),
    "Y": (1.391, 0.231, 1.096, 1.253, -0.475),
}
ATCHLEY_NORMS = np.sqrt(
    np.sum(np.asarray(list(ATCHLEY.values()), dtype=np.float64) ** 2, axis=0)
)


@dataclass(frozen=True)
class HarmonicCall:
    score: float
    period: float
    frequency_bin: int
    n_harmonics: int
    n_harmonics_tested: int
    harmonic_z: tuple[float, ...]
    spectral_scale: float

    def to_dict(self) -> dict:
        value = asdict(self)
        value["harmonic_z"] = list(self.harmonic_z)
        return value


def encode_atchley(sequence: str) -> np.ndarray:
    """Return an Lx5 Atchley-factor matrix for a canonical protein sequence."""
    sequence = require_canonical_sequence(sequence)
    matrix = np.zeros((len(sequence), 5), dtype=np.float64)
    for index, amino_acid in enumerate(sequence):
        matrix[index] = ATCHLEY[amino_acid]
    return matrix


def encode_normalized_atchley(sequence: str) -> np.ndarray:
    """Atchley factors normalized over the 20-residue alphabet as in REPETITA."""
    return encode_atchley(sequence) / ATCHLEY_NORMS[None, :]


def encode_onehot(sequence: str) -> np.ndarray:
    """Return an Lx20 exact-residue indicator matrix."""
    sequence = require_canonical_sequence(sequence)
    matrix = np.zeros((len(sequence), len(AA_ORDER)), dtype=np.float64)
    for index, amino_acid in enumerate(sequence):
        matrix[index, AA_INDEX[amino_acid]] = 1.0
    return matrix


def detrend_and_standardize(signals: np.ndarray) -> np.ndarray:
    """Remove column means/linear trends and scale nonconstant channels."""
    if signals.ndim == 1:
        signals = signals[:, None]
    values = np.asarray(signals, dtype=np.float64)
    n = values.shape[0]
    if n == 0:
        return values.copy()
    centered = values - values.mean(axis=0, keepdims=True)
    time = np.arange(n, dtype=np.float64) - (n - 1) / 2
    denominator = float(np.dot(time, time))
    if denominator > 0:
        # Elementwise reduction avoids a NumPy/Accelerate matmul warning seen
        # for sparse one-hot columns on Apple Silicon.
        slopes = np.sum(centered * time[:, None], axis=0) / denominator
        centered = centered - time[:, None] * slopes[None, :]
    standard_deviation = centered.std(axis=0)
    keep = standard_deviation > 1e-12
    if not np.any(keep):
        return np.zeros((n, 0), dtype=np.float64)
    return centered[:, keep] / standard_deviation[keep][None, :]


def aggregate_spectrum(signals: np.ndarray) -> tuple[np.ndarray, float]:
    """Return robust local z scores of summed channel log-power."""
    values = detrend_and_standardize(signals)
    n = values.shape[0]
    if n < 4 or values.shape[1] == 0:
        return np.zeros(n // 2 + 1, dtype=np.float64), 0.0
    window = np.hanning(n)
    transformed = np.fft.rfft(values * window[:, None], axis=0)
    normalization = max(np.sum(window**2) * values.shape[1], 1e-12)
    power = np.sum(np.abs(transformed) ** 2, axis=1) / normalization
    log_power = np.log1p(power)
    filter_width = min(15, max(5, (len(log_power) // 8) | 1))
    if filter_width % 2 == 0:
        filter_width += 1
    baseline = median_filter(log_power, size=filter_width, mode="nearest")
    residual = log_power - baseline
    usable = residual[2:] if len(residual) > 2 else residual
    center = float(np.median(usable)) if len(usable) else 0.0
    mad = float(np.median(np.abs(usable - center))) if len(usable) else 0.0
    scale = 1.4826 * mad
    if scale < 1e-8:
        scale = float(np.std(usable)) if len(usable) else 0.0
    if scale < 1e-8:
        return np.zeros_like(residual), 0.0
    return (residual - center) / scale, scale


def best_harmonic_comb(
    spectral_z: np.ndarray,
    signal_length: int,
    min_period: float = 5.0,
    max_period: float = 250.0,
    min_cycles: int = 3,
    max_harmonics: int = 8,
    tolerance_bins: int = 1,
    excess_floor: float = 1.0,
    significance_z: float = 2.5,
) -> HarmonicCall:
    """Score candidate fundamentals by repeated peaks at integer harmonics."""
    maximum_frequency_bin = len(spectral_z) - 1
    minimum_bin = max(min_cycles, int(math.ceil(signal_length / max_period)))
    maximum_bin = min(
        maximum_frequency_bin,
        int(math.floor(signal_length / min_period)),
    )
    empty = HarmonicCall(0.0, -1.0, -1, 0, 0, tuple(), 0.0)
    if maximum_bin < minimum_bin:
        return empty

    best: HarmonicCall | None = None
    for fundamental_bin in range(minimum_bin, maximum_bin + 1):
        n_tested = min(max_harmonics, maximum_frequency_bin // fundamental_bin)
        if n_tested < 2:
            continue
        harmonic_z: list[float] = []
        weights: list[float] = []
        for harmonic in range(1, n_tested + 1):
            center = harmonic * fundamental_bin
            left = max(1, center - tolerance_bins)
            right = min(maximum_frequency_bin, center + tolerance_bins)
            harmonic_z.append(float(np.max(spectral_z[left : right + 1])))
            weights.append(1.0 / math.sqrt(harmonic))
        z_values = np.asarray(harmonic_z)
        weight_values = np.asarray(weights)
        excess = np.maximum(0.0, z_values - excess_floor)
        score = float(np.dot(excess, weight_values) / np.linalg.norm(weight_values))
        n_significant = int(np.sum(z_values >= significance_z))
        candidate = HarmonicCall(
            score=score,
            period=float(signal_length / fundamental_bin),
            frequency_bin=fundamental_bin,
            n_harmonics=n_significant,
            n_harmonics_tested=n_tested,
            harmonic_z=tuple(harmonic_z),
            spectral_scale=0.0,
        )
        if best is None or (candidate.score, candidate.n_harmonics) > (
            best.score,
            best.n_harmonics,
        ):
            best = candidate
    return best if best is not None else empty


def score_signals(
    signals: np.ndarray,
    min_period: float = 5.0,
    max_period: float = 250.0,
    min_cycles: int = 3,
) -> HarmonicCall:
    spectral_z, scale = aggregate_spectrum(signals)
    call = best_harmonic_comb(
        spectral_z,
        signals.shape[0],
        min_period=min_period,
        max_period=max_period,
        min_cycles=min_cycles,
    )
    return HarmonicCall(
        score=call.score,
        period=call.period,
        frequency_bin=call.frequency_bin,
        n_harmonics=call.n_harmonics,
        n_harmonics_tested=call.n_harmonics_tested,
        harmonic_z=call.harmonic_z,
        spectral_scale=scale,
    )


def score_window(
    sequence: str,
    min_period: float = 5.0,
    max_period: float = 250.0,
    min_cycles: int = 3,
) -> dict:
    """Score one sequence window with physical and exact-residue spectra."""
    sequence = require_canonical_sequence(sequence)
    effective_maximum = min(max_period, len(sequence) / max(min_cycles, 1))
    physical_signals = encode_atchley(sequence)
    physical = score_signals(
        physical_signals, min_period, effective_maximum, min_cycles
    )
    factor_calls = [
        score_signals(
            physical_signals[:, factor_index : factor_index + 1],
            min_period,
            effective_maximum,
            min_cycles,
        )
        for factor_index in range(physical_signals.shape[1])
    ]
    best_factor_index, best_factor = max(
        enumerate(factor_calls),
        key=lambda item: (item[1].score, item[1].n_harmonics),
    )
    factor_maximum = best_factor.to_dict()
    factor_maximum["factor"] = best_factor_index + 1
    exact = score_signals(
        encode_onehot(sequence), min_period, effective_maximum, min_cycles
    )
    return {
        "physical": physical.to_dict(),
        "physical_factor_max": factor_maximum,
        "exact": exact.to_dict(),
        "repetita": repetita_like_features(
            sequence,
            min_period=min_period,
            max_period=effective_maximum,
            min_cycles=min_cycles,
        ),
    }


def repetita_like_features(
    sequence: str,
    min_period: float = 5.0,
    max_period: float = 250.0,
    min_cycles: int = 3,
    thresholds: tuple[float, ...] = (1.5, 2.1, 2.5, 3.0, 3.4),
) -> dict:
    """Direct-sequence analogue of REPETITA's zmax/rho-theta features.

    REPETITA used PSI-BLAST sequence profiles. This intentionally uses only the
    query sequence, so it is a fair sequence-only comparator and must not be
    described as a reproduction of profile-based REPETITA performance.
    """
    signals = encode_normalized_atchley(sequence)
    n = len(sequence)
    empty = {
        "zmax": 0.0,
        "period": -1.0,
        "frequency_bin": -1,
        "factor": -1,
        **{f"rho_{str(value).replace('.', '_')}": 0.0 for value in thresholds},
    }
    if n < 4:
        return empty
    amplitudes = 2.0 * np.abs(np.fft.rfft(signals, axis=0)) / n
    if len(amplitudes) <= 1:
        return empty
    non_dc = amplitudes[1:]
    means = non_dc.mean(axis=0)
    standard_deviations = non_dc.std(axis=0)
    standard_deviations[standard_deviations < 1e-12] = 1.0
    zscores = (non_dc - means[None, :]) / standard_deviations[None, :]

    minimum_bin = max(min_cycles, int(math.ceil(n / max_period)))
    maximum_bin = min(len(amplitudes) - 1, int(math.floor(n / min_period)))
    if maximum_bin < minimum_bin:
        return empty
    # zscores index zero corresponds to frequency bin one.
    band = zscores[minimum_bin - 1 : maximum_bin]
    flat_index = int(np.argmax(band))
    local_frequency_index, factor_index = np.unravel_index(flat_index, band.shape)
    frequency_bin = minimum_bin + local_frequency_index
    result = {
        "zmax": float(band[local_frequency_index, factor_index]),
        "period": float(n / frequency_bin),
        "frequency_bin": int(frequency_bin),
        "factor": int(factor_index + 1),
    }
    for threshold in thresholds:
        result[f"rho_{str(threshold).replace('.', '_')}"] = float(np.mean(band > threshold))
    return result


def window_plan(
    length: int,
    window_sizes: tuple[int, ...] = (128, 256, 512, 1024),
    stride_fraction: float = 0.25,
) -> list[tuple[int, int]]:
    """Return deterministic multiscale zero-based, end-exclusive windows."""
    if length <= 0:
        return []
    sizes = {min(length, size) for size in window_sizes}
    if length < max(window_sizes):
        sizes.add(length)
    windows: set[tuple[int, int]] = set()
    for size in sorted(sizes):
        stride = max(16, int(round(size * stride_fraction)))
        starts = list(range(0, max(1, length - size + 1), stride))
        starts.append(length - size)
        for start in starts:
            windows.add((start, start + size))
    return sorted(windows, key=lambda value: (value[1] - value[0], value[0]))


def scan_sequence(
    sequence: str,
    min_period: float = 5.0,
    max_period: float = 250.0,
    min_cycles: int = 3,
    window_sizes: tuple[int, ...] = (128, 256, 512, 1024),
) -> list[dict]:
    """Return harmonic evidence for every multiscale window in a sequence."""
    sequence = require_canonical_sequence(sequence)
    calls = []
    for start, end in window_plan(len(sequence), window_sizes=window_sizes):
        score = score_window(
            sequence[start:end],
            min_period=min_period,
            max_period=max_period,
            min_cycles=min_cycles,
        )
        calls.append(
            {
                "start": start,
                "end": end,
                "window_size": end - start,
                **score,
            }
        )
    return calls
