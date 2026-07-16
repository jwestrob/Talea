"""Frozen production primitives for Talea sequence-only solenoid ranking.

The production scorer combines three outputs from one ESM++ forward pass:

* a residue-state probability track decoded from layers 13 and 35 attention;
* the frozen attention-topology score; and
* a compact global classifier over mean and population-standard-deviation
  pooled hidden states from the same two transformer layers.

These functions deliberately contain no biological naming logic.  Their
outputs are unverified model scores intended for ranking candidate proteins.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence

import numpy as np
import torch

from talea.architecture_router import probability_logit
from talea.repeat_state import (
    attention_channels,
    central_tile_spans,
    extract_attention_tile,
    summarize_state_track,
)


def roundtrip_attention_storage(
    matrix: np.ndarray, storage_dtype: str
) -> np.ndarray:
    """Reproduce the numeric representation used by frozen attention artifacts.

    The lockbox attention matrices were serialized as float16 and subsequently
    loaded as float32 by both the teacher and topology decoders.  Applying the
    same round trip in a streaming run makes matrix representation part of the
    deployment contract without retaining the matrix on disk.
    """

    values = np.asarray(matrix)
    if values.ndim != 2 or values.shape[0] != values.shape[1] or not values.size:
        raise ValueError("attention matrix must be non-empty and square")
    if not np.isfinite(values).all():
        raise ValueError("attention matrix contains nonfinite values")
    dtypes = {"fp16": np.float16, "fp32": np.float32}
    if storage_dtype not in dtypes:
        raise ValueError(f"unsupported attention storage dtype: {storage_dtype}")
    return values.astype(dtypes[storage_dtype]).astype(np.float32)


def pooled_hidden_features_for_record(
    hidden_states: Sequence[torch.Tensor],
    residue_indices: torch.Tensor,
    batch_position: int,
    transformer_layer_indices: Sequence[int],
) -> dict[str, np.ndarray]:
    """Pool residue hidden states exactly as in the frozen global-head fit."""

    indices = residue_indices.to(dtype=torch.long)
    if indices.ndim != 1 or not len(indices):
        raise ValueError("residue indices must be a non-empty vector")
    features: dict[str, np.ndarray] = {}
    for layer_index in transformer_layer_indices:
        layer = int(layer_index)
        hidden_state_index = layer + 1
        if hidden_state_index >= len(hidden_states):
            raise ValueError(
                f"transformer layer {layer} unavailable in "
                f"{len(hidden_states) - 1} returned layers"
            )
        state = hidden_states[hidden_state_index]
        if not 0 <= batch_position < state.shape[0]:
            raise IndexError("batch position is outside the hidden-state batch")
        residues = state[batch_position].index_select(0, indices).to(torch.float32)
        features[f"layer_{layer}_mean"] = residues.mean(dim=0).cpu().numpy()
        features[f"layer_{layer}_std"] = residues.std(
            dim=0, unbiased=False
        ).cpu().numpy()
    return features


def flatten_global_features(
    features: Mapping[str, np.ndarray], feature_keys: Sequence[str]
) -> np.ndarray:
    """Flatten a named pooled representation in its frozen feature order."""

    keys = tuple(str(value) for value in feature_keys)
    if not keys or len(set(keys)) != len(keys):
        raise ValueError("global feature keys must be unique and non-empty")
    missing = [key for key in keys if key not in features]
    if missing:
        raise KeyError(f"missing global features: {missing}")
    arrays = [np.asarray(features[key], dtype=np.float64).reshape(-1) for key in keys]
    if any(not values.size for values in arrays):
        raise ValueError("global feature vectors must be non-empty")
    values = np.concatenate(arrays)[None, :]
    if not np.isfinite(values).all():
        raise ValueError("global feature vector contains nonfinite values")
    return values


def global_head_probability(
    model_bundle: Mapping, features: Mapping[str, np.ndarray]
) -> float:
    """Apply the frozen scikit-learn global head to one pooled representation."""

    return float(global_head_probabilities(model_bundle, [features])[0])


def global_head_probabilities(
    model_bundle: Mapping, feature_rows: Sequence[Mapping[str, np.ndarray]]
) -> np.ndarray:
    """Apply the frozen global head to a batch without changing its arithmetic."""

    required = {
        "model",
        "feature_keys",
        "control_mean",
        "control_standard_deviation",
    }
    missing = sorted(required - set(model_bundle))
    if missing:
        raise KeyError(f"global model bundle lacks required fields: {missing}")
    if not feature_rows:
        raise ValueError("at least one global feature row is required")
    values = np.concatenate(
        [
            flatten_global_features(features, model_bundle["feature_keys"])
            for features in feature_rows
        ],
        axis=0,
    )
    with warnings.catch_warnings(), np.errstate(
        divide="ignore", over="ignore", invalid="ignore"
    ):
        warnings.simplefilter("ignore", RuntimeWarning)
        probability = np.asarray(
            model_bundle["model"].predict_proba(values)[:, 1], dtype=np.float64
        )
    if probability.shape != (len(feature_rows),) or not np.isfinite(
        probability
    ).all():
        raise ValueError("global model emitted invalid probabilities")
    if np.any((probability < 0.0) | (probability > 1.0)):
        raise ValueError("global probability is outside [0, 1]")
    return probability


def standardized_probability_z(
    probability: float, control_mean: float, control_standard_deviation: float
) -> float:
    """Logit-transform and standardize a model probability against controls."""

    mean = float(control_mean)
    standard_deviation = float(control_standard_deviation)
    if not math.isfinite(mean):
        raise ValueError("control mean must be finite")
    if not math.isfinite(standard_deviation) or standard_deviation <= 0:
        raise ValueError("control standard deviation must be finite and positive")
    return (probability_logit(float(probability)) - mean) / standard_deviation


@torch.inference_mode()
def teacher_tracks_from_attention(
    model: torch.nn.Module,
    matrices: Mapping[int, np.ndarray],
    contract: Mapping,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode one protein's state and teacher-class tracks from attention.

    Tiles are scored only on their central diagonal, so every residue is
    reconstructed exactly once and padding never contributes to a summary.
    """

    attention = contract["attention"]
    layers = tuple(int(value) for value in attention["layers_zero_based"])
    missing = [layer for layer in layers if layer not in matrices]
    if missing:
        raise KeyError(f"missing teacher attention layers: {missing}")
    selected = [np.asarray(matrices[layer], dtype=np.float32) for layer in layers]
    shapes = {values.shape for values in selected}
    if len(shapes) != 1:
        raise ValueError("teacher attention matrices have inconsistent shapes")
    shape = next(iter(shapes))
    if len(shape) != 2 or shape[0] != shape[1] or shape[0] < 1:
        raise ValueError("teacher attention matrices must be non-empty and square")
    length = int(shape[0])
    channels = attention_channels(
        selected,
        quantile=float(attention["normalization_quantile"]),
        clip=float(attention["normalization_clip"]),
    )
    tile_size = int(attention["tile_size"])
    central_size = int(attention["central_size"])
    spans = central_tile_spans(length, tile_size, central_size)
    teacher_classes = len(contract["teacher"]["class_names"])
    state_track = np.zeros(length, dtype=np.float32)
    teacher_track = np.zeros((length, teacher_classes), dtype=np.float32)
    observed = np.zeros(length, dtype=np.int8)
    batch_size = int(contract["model"]["batch_size"])
    if batch_size < 1:
        raise ValueError("teacher tile batch size must be positive")
    model.eval()
    offset = (tile_size - central_size) // 2
    diagonal_indices = torch.arange(
        offset, offset + central_size, device=device, dtype=torch.long
    )
    for batch_start in range(0, len(spans), batch_size):
        batch_spans = spans[batch_start : batch_start + batch_size]
        tiles = np.stack(
            [extract_attention_tile(channels, span, tile_size) for span in batch_spans]
        )
        output = model(torch.from_numpy(tiles).to(device))
        if output.shape[1] != 1 + teacher_classes:
            raise ValueError("teacher model output channels disagree with its contract")
        diagonal = output[:, :, diagonal_indices, diagonal_indices]
        state_probability = torch.sigmoid(diagonal[:, 0]).cpu().numpy()
        teacher_probability = torch.softmax(diagonal[:, 1:], dim=1)
        teacher_probability = teacher_probability.transpose(1, 2).cpu().numpy()
        for row_index, span in enumerate(batch_spans):
            start = int(span["central_start"])
            end = int(span["central_end"])
            width = end - start
            state_track[start:end] = state_probability[row_index, :width]
            teacher_track[start:end] = teacher_probability[row_index, :width]
            observed[start:end] += 1
    if not np.all(observed == 1):
        raise RuntimeError("teacher tiles did not cover every residue exactly once")
    if not np.isfinite(state_track).all() or not np.isfinite(teacher_track).all():
        raise ValueError("teacher model emitted nonfinite values")
    if not np.allclose(teacher_track.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("teacher class probabilities are not normalized")
    return state_track, teacher_track


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


def compose_talea_score(
    teacher_state_probability: float,
    topology_z: float,
    global_probability: float,
    score_contract: Mapping,
) -> dict[str, float]:
    """Compose frozen Talea benchmark and optional stable discovery scores."""

    state = score_contract["teacher_state_calibration"]
    global_calibration = score_contract["global_calibration"]
    weights = score_contract["weights"]
    topology_value = float(topology_z)
    if not math.isfinite(topology_value):
        raise ValueError("topology z score must be finite")
    teacher_z = standardized_probability_z(
        teacher_state_probability,
        state["control_logit_mean"],
        state["control_logit_standard_deviation_population"],
    )
    global_z = standardized_probability_z(
        global_probability,
        global_calibration["control_logit_mean"],
        global_calibration["control_logit_standard_deviation_population"],
    )
    state_weight = float(weights["teacher_state_within_base"])
    topology_weight = float(weights["topology_within_base"])
    base_weight = float(weights["base_in_primary"])
    global_weight = float(weights["global_in_primary"])
    for name, value in (
        ("teacher_state_within_base", state_weight),
        ("topology_within_base", topology_weight),
        ("base_in_primary", base_weight),
        ("global_in_primary", global_weight),
    ):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if not math.isclose(state_weight + topology_weight, 1.0, abs_tol=1e-12):
        raise ValueError("within-base Talea weights must sum to one")
    if not math.isclose(base_weight + global_weight, 1.0, abs_tol=1e-12):
        raise ValueError("primary Talea weights must sum to one")
    base_score = state_weight * teacher_z + topology_weight * topology_value
    primary_score = base_weight * base_score + global_weight * global_z
    result = {
        "teacher_state_probability": float(teacher_state_probability),
        "teacher_state_z": float(teacher_z),
        "topology_development_control_z": topology_value,
        "global_probability": float(global_probability),
        "global_z": float(global_z),
        "teacher_base_score": float(base_score),
        "talea_teacher_score": float(primary_score),
    }
    discovery_weights = score_contract.get("discovery_weights")
    if discovery_weights is not None:
        discovery_state_weight = float(discovery_weights["teacher_state"])
        discovery_global_weight = float(discovery_weights["global"])
        for name, value in (
            ("discovery teacher-state", discovery_state_weight),
            ("discovery global", discovery_global_weight),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} weight must be finite and nonnegative")
        if not math.isclose(
            discovery_state_weight + discovery_global_weight, 1.0, abs_tol=1e-12
        ):
            raise ValueError("discovery Talea weights must sum to one")
        result["talea_discovery_score"] = float(
            discovery_state_weight * teacher_z
            + discovery_global_weight * global_z
        )
    return result
