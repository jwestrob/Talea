"""Auxiliary attention-topology localization used by Talea inference."""

from __future__ import annotations

import math

import numpy as np

from talea.application import frozen_operational_scores, frozen_spectrum_score
from talea.attention import scan_attention_matrix, score_attention_window
from talea.unit_registration import (
    register_periodic_units,
    registered_closure_score,
    scan_plausible_boundary_pairs,
)
from talea.wedge import (
    best_wedge_interval,
    candidate_periods_from_windows,
    fuse_diagonal_tracks,
    robust_diagonal_tracks,
)


def widths_from_config(config: dict) -> tuple[int, ...]:
    values = config["interval_widths"]
    return tuple(
        range(
            int(values["start"]),
            int(values["stop_inclusive"]) + 1,
            int(values["step"]),
        )
    )


def window_zscore(row: dict, protein_length: int, calibration: dict) -> float:
    coefficients = np.asarray(calibration["coefficients"], dtype=np.float64)
    features = np.asarray(
        [1.0, math.log2(int(row["window_size"])), math.log2(protein_length)]
    )
    residual = math.log1p(float(row["attention"]["score"])) - float(
        np.dot(coefficients, features)
    )
    return (
        residual - float(calibration["weighted_residual_mean"])
    ) / float(calibration["weighted_residual_sd"])


def fused_local_windows(
    matrices: dict[int, np.ndarray],
    protein_length: int,
    calibration: dict,
    window_sizes: tuple[int, ...],
) -> list[dict]:
    layers = tuple(int(value) for value in calibration["layers_zero_based"])
    parameters = {
        int(key): value for key, value in calibration["parameters"].items()
    }
    per_layer = {
        layer: scan_attention_matrix(
            matrices[layer],
            min_period=5.0,
            max_period=250.0,
            min_cycles=3,
            window_sizes=window_sizes,
        )
        for layer in layers
    }
    grouped: dict[tuple[int, int, int], dict[int, dict]] = {}
    for layer, windows in per_layer.items():
        for window in windows:
            key = (
                int(window["start"]),
                int(window["end"]),
                int(window["window_size"]),
            )
            grouped.setdefault(key, {})[layer] = window
    fused = []
    for (start, end, window_size), by_layer in sorted(grouped.items()):
        if set(by_layer) != set(layers):
            raise ValueError("local layer windows do not align")
        zscores = {
            layer: window_zscore(by_layer[layer], protein_length, parameters[layer])
            for layer in layers
        }
        source_layer = max(layers, key=lambda layer: zscores[layer])
        source = by_layer[source_layer]["attention"]
        fused.append(
            {
                "start": start,
                "end": end,
                "window_size": window_size,
                "attention_fused": {
                    "score": float(np.mean(list(zscores.values()))),
                    "period": float(source["period"]),
                    "n_harmonics": int(source["n_harmonics"]),
                    "period_source_layer_zero_based": source_layer,
                    "layer_residual_z": {
                        str(layer): float(zscores[layer]) for layer in layers
                    },
                },
            }
        )
    return fused


def process_attention_record(task: dict) -> dict:
    """Decode one pair of attention matrices into localization diagnostics."""

    metadata = task["metadata"]
    matrices = task["matrices"]
    length = int(metadata["length"])
    protein_calibration = task["protein_calibration"]
    local_calibration = task["local_calibration"]
    wedge_config = task["wedge_config"]
    operational_config = task["operational_config"]
    spectrum_input = {
        "layer_features": [
            {"layer": layer, **score_attention_window(matrix)}
            for layer, matrix in sorted(matrices.items())
        ]
    }
    spectrum = frozen_spectrum_score(
        spectrum_input, length, protein_calibration
    )
    windows = fused_local_windows(
        matrices,
        length,
        local_calibration,
        tuple(task["window_sizes"]),
    )
    period_config = wedge_config["period_candidates"]
    periods = candidate_periods_from_windows(
        windows,
        minimum_window_size=int(period_config["minimum_window_size"]),
        top_n=int(period_config["top_n"]),
        alias_multipliers=tuple(
            float(value) for value in period_config["alias_multipliers"]
        ),
        minimum_period=float(period_config["minimum_period"]),
        maximum_period=float(period_config["maximum_period"]),
        quantization=float(period_config["quantization"]),
    )
    wedge_call = None
    if periods:
        tracks = fuse_diagonal_tracks(
            [
                robust_diagonal_tracks(
                    matrices[layer],
                    int(wedge_config["maximum_separation"]),
                    baseline_quantile=float(
                        wedge_config["diagonal_baseline_quantile"]
                    ),
                    clip=float(wedge_config["diagonal_z_clip"]),
                )
                for layer in (13, 35)
            ]
        )
        wedge = wedge_config["wedge"]
        candidate = best_wedge_interval(
            tracks,
            length,
            periods,
            widths_from_config(wedge_config),
            tolerance=int(wedge["tolerance"]),
            max_harmonics=int(wedge["max_harmonics"]),
            min_harmonics=int(wedge["min_harmonics"]),
            interharmonic_controls=bool(wedge["interharmonic_controls"]),
        )
        if math.isfinite(float(candidate["score"])):
            wedge_call = candidate
    registration = None
    closure = None
    boundary_scan = None
    if wedge_call is None:
        status = "no_wedge_call"
    else:
        registration_config = operational_config["registration"]
        attention = matrices[
            int(registration_config["attention_layer_zero_based"])
        ].astype(np.float64)
        registration = register_periodic_units(
            attention,
            period=float(wedge_call["period"]),
            core_start=int(wedge_call["start"]),
            core_end=int(wedge_call["end"]),
            extension_fraction=float(registration_config["extension_fraction"]),
            phase_step=float(registration_config["phase_step"]),
            minimum_terminal_fraction=float(
                registration_config["minimum_terminal_fraction"]
            ),
            edge_cv_penalty=float(registration_config["edge_cv_penalty"]),
        )
        closure = registered_closure_score(attention, registration["units"])
        boundary_scan = scan_plausible_boundary_pairs(attention, registration)
        status = "ok"
    operational = frozen_operational_scores(
        status,
        boundary_scan,
        float(spectrum["frozen_esmpp_score"]),
        operational_config,
    )
    return {
        "record_id": metadata["record_id"],
        "sequence_sha256": metadata["sequence_sha256"],
        "length": length,
        "status": status,
        "frozen_spectrum": spectrum,
        "candidate_periods": list(periods),
        "wedge_call": wedge_call,
        "registration": registration,
        "closure": closure,
        "boundary_scan": boundary_scan,
        **operational,
    }
