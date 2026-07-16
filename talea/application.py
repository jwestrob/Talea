"""Frozen Talea scoring helpers for unlabeled sequence applications."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np


def layer_score_map(score_row: Mapping) -> dict[int, float]:
    """Return the unique finite whole-protein score for each attention layer."""

    result: dict[int, float] = {}
    for feature in score_row.get("layer_features", []):
        layer = int(feature["layer"])
        if layer in result:
            raise ValueError(f"duplicate layer {layer}")
        score = float(feature["score"])
        if not math.isfinite(score) or score <= -1.0:
            raise ValueError(f"invalid layer score {score} at layer {layer}")
        result[layer] = score
    return result


def frozen_spectrum_score(
    score_row: Mapping,
    protein_length: int,
    calibration: Mapping,
) -> dict:
    """Apply the frozen ESM++ layer-spectrum calibration without refitting."""

    if protein_length < 1:
        raise ValueError("protein_length must be positive")
    layers = [int(value) for value in calibration["selected_layers_zero_based"]]
    parameters = {int(key): value for key, value in calibration["parameters"].items()}
    raw_scores = layer_score_map(score_row)
    missing = sorted(set(layers) - set(raw_scores))
    if missing:
        raise ValueError(f"missing selected layers: {missing}")
    zscores = []
    result = {"selected_layers_zero_based": layers, "layers": {}}
    for layer in layers:
        raw = raw_scores[layer]
        log_score = math.log1p(raw)
        values = parameters[layer]
        expected = (
            float(values["slope"]) * math.log2(protein_length)
            + float(values["intercept"])
        )
        residual = log_score - expected
        zscore = (
            residual - float(values["control_residual_mean"])
        ) / float(values["control_residual_sd_population"])
        result["layers"][str(layer)] = {
            "raw_score": raw,
            "log_score": log_score,
            "expected_control_log_score": expected,
            "length_residual": residual,
            "length_residual_z": zscore,
        }
        zscores.append(zscore)
    result["frozen_esmpp_score"] = float(np.mean(zscores))
    return result


def frozen_operational_scores(
    registration_status: str,
    boundary_scan: Mapping | None,
    frozen_pangu_score: float,
    config: Mapping,
) -> dict:
    """Apply frozen candidate correction, standardization, and score fusion."""

    topology = config["topology_score"]
    correction = topology["candidate_count_correction"]
    topology_scale = topology["development_control_standardization"]
    pangu_scale = config["frozen_pangu_standardization"]
    if registration_status == "ok":
        if boundary_scan is None:
            raise ValueError("an ok registration requires a boundary scan")
        candidate_count = int(boundary_scan["candidate_count"])
        if candidate_count < 1:
            raise ValueError("candidate_count must be positive")
        raw = float(boundary_scan["open_path_score_minimum"])
        expected = float(correction["intercept"]) + float(
            correction["coefficient"]
        ) * math.log1p(candidate_count)
        corrected = (raw - expected) / float(correction["residual_sd"])
    else:
        candidate_count = 0
        raw = None
        expected = None
        corrected = float(correction["no_call_corrected_score"])
    topology_z = (corrected - float(topology_scale["mean"])) / float(
        topology_scale["sd"]
    )
    pangu_z = (
        float(frozen_pangu_score) - float(pangu_scale["development_control_mean"])
    ) / float(pangu_scale["development_control_sd"])
    fusion = (
        float(config["fusion"]["topology_weight"]) * topology_z
        + float(config["fusion"]["frozen_pangu_weight"]) * pangu_z
    )
    return {
        "candidate_count": candidate_count,
        "boundary_scan_open_score_raw": raw,
        "candidate_count_expected_open_score": expected,
        "candidate_corrected_topology_score": corrected,
        "topology_development_control_z": topology_z,
        "frozen_pangu_development_control_z": pangu_z,
        "equal_weight_fusion_score": fusion,
    }


def rescore_operational_prediction(
    prediction: Mapping,
    protein_calibration: Mapping,
    operational_config: Mapping,
) -> dict:
    """Reapply calibrations to retained raw layer and boundary-scan statistics."""

    layers = prediction["frozen_spectrum"]["layers"]
    layer_row = {
        "layer_features": [
            {"layer": int(layer), "score": float(values["raw_score"])}
            for layer, values in sorted(layers.items(), key=lambda item: int(item[0]))
        ]
    }
    spectrum = frozen_spectrum_score(
        layer_row,
        int(prediction["length"]),
        protein_calibration,
    )
    operational = frozen_operational_scores(
        str(prediction["status"]),
        prediction.get("boundary_scan"),
        float(spectrum["frozen_esmpp_score"]),
        operational_config,
    )
    return {**prediction, "frozen_spectrum": spectrum, **operational}


def dense_ranks(values: Sequence[float]) -> list[int]:
    """Rank finite scores from largest to smallest with deterministic ties."""

    numbers = [float(value) for value in values]
    if not all(math.isfinite(value) for value in numbers):
        raise ValueError("rank values must be finite")
    unique = sorted(set(numbers), reverse=True)
    rank_by_value = {value: index + 1 for index, value in enumerate(unique)}
    return [rank_by_value[value] for value in numbers]
