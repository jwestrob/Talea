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
import threading
import warnings
from collections.abc import Mapping, Sequence
from contextlib import contextmanager, nullcontext

import numpy as np
import torch

from talea.architecture_router import probability_logit
from talea.repeat_state import (
    central_tile_spans,
    normalized_attention_tiles,
    summarize_state_track,
    summarize_teacher_prediction,
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


def pooled_hidden_features_for_layers(
    hidden_states: Mapping[int, torch.Tensor],
    residue_indices: torch.Tensor,
    batch_position: int,
    transformer_layer_indices: Sequence[int],
) -> dict[str, np.ndarray]:
    """Pool from a sparse {layer: tensor} capture instead of the full tuple.

    Arithmetically identical to ``pooled_hidden_features_for_record``; it only
    changes how the layer's tensor is located. The caller is responsible for
    supplying the same tensor that ``hidden_states[layer + 1]`` would have
    yielded -- see talea.capture.resolve_hidden_state_sources.
    """

    indices = residue_indices.to(dtype=torch.long)
    if indices.ndim != 1 or not len(indices):
        raise ValueError("residue indices must be a non-empty vector")
    features: dict[str, np.ndarray] = {}
    for layer_index in transformer_layer_indices:
        layer = int(layer_index)
        if layer not in hidden_states:
            raise KeyError(f"hidden state for transformer layer {layer} not captured")
        state = hidden_states[layer]
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

    return _global_feature_matrix([features], feature_keys)


def _global_feature_matrix(
    feature_rows: Sequence[Mapping[str, np.ndarray]], feature_keys: Sequence[str]
) -> np.ndarray:
    """Place every pooled vector in one C-order float64 prediction matrix."""
    keys = tuple(str(value) for value in feature_keys)
    if not keys or len(set(keys)) != len(keys):
        raise ValueError("global feature keys must be unique and non-empty")
    vectors = []
    width = None
    for features in feature_rows:
        missing = [key for key in keys if key not in features]
        if missing:
            raise KeyError(f"missing global features: {missing}")
        row = [np.asarray(features[key]).reshape(-1) for key in keys]
        sizes = [vector.size for vector in row]
        if not all(sizes):
            raise ValueError("global feature vectors must be non-empty")
        row_width = sum(sizes)
        if width is None:
            width = row_width
        elif row_width != width:
            raise ValueError("global feature rows must have the same total width")
        vectors.extend(row)
    values = np.concatenate(vectors, dtype=np.float64, casting="unsafe").reshape(len(feature_rows), width)
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
    values = _global_feature_matrix(feature_rows, model_bundle["feature_keys"])
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


def _pack_teacher_diagonal(diagonal, teacher_classes):
    if diagonal.shape[1] != 1 + teacher_classes:
        raise ValueError("teacher model output channels disagree with its contract")
    state = torch.sigmoid(diagonal[:, 0])
    classes = torch.softmax(diagonal[:, 1:], dim=1)
    return torch.cat((state.unsqueeze(-1), classes.transpose(1, 2)), dim=2).reshape(-1, 1 + teacher_classes)


class CudaTeacherGraphs:
    """Replay the frozen U-Net at its original tile-batch sizes on one stream."""

    @torch.inference_mode()
    def __init__(self, model, contract, device, pack_tracks=False):
        self.pack_tracks = pack_tracks
        self.track_outputs = {}
        self.stream = torch.cuda.current_stream(device)
        self.capture_thread = threading.current_thread()
        self.graphs = {}
        self._locks = {}
        tile_size = int(contract["attention"]["tile_size"])
        channels = int(contract["input_channels"])
        # CUDA requires a non-default capture stream. If the caller already
        # owns one (as in forward prefetch), capture and replay there.
        side = self.stream
        if side == torch.cuda.default_stream(device):
            side = torch.cuda.Stream(device=device)
            side.wait_stream(self.stream)
        with torch.cuda.device(device), torch.cuda.stream(side):
            indices = prepare_teacher_readout(model, contract, device) if pack_tracks else None
            for size in range(1, int(contract["model"]["batch_size"]) + 1):
                static = torch.zeros((size, channels, tile_size, tile_size),
                                     dtype=torch.float32, device=device)
                # Pre-initialize cuDNN on the capture stream for this exact shape.
                for _ in range(3):
                    model(static)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=side):
                    output = model(static)
                    if pack_tracks:
                        diagonal = output[:, :, indices, indices]
                        self.track_outputs[size] = _pack_teacher_diagonal(diagonal, len(contract["teacher"]["class_names"]))
                # Independent graph pools allow arbitrary replay order.
                self.graphs[size] = graph, static, output
                self._locks[size] = threading.RLock()
        if side != self.stream:
            self.stream.wait_stream(side)

    @contextmanager
    def input_buffer(self, size):
        """Reserve this graph's input through a direct write and the ensuing replay."""
        with self._locks[size]:
            yield self.graphs[size][1]

    def __call__(self, host_tiles):
        # The static output is borrowed until the calling thread releases the
        # shape lock in diagonal().
        graph, static, output = self.graphs[len(host_tiles)]
        # The per-protein tile tensor stays alive through the synchronous
        # state/class host reads; replay is ordered behind the H2D on this stream.
        if host_tiles is not static:
            static.copy_(host_tiles, non_blocking=True)
        graph.replay()
        return output


    def probabilities(self, host_tiles):
        size = len(host_tiles)
        with self._locks[size]:
            self(host_tiles)
            # The next replay of this shape overwrites the static track storage.
            return self.track_outputs[size].clone()

    def diagonal(self, host_tiles, indices):
        # Record workers issue work to one CUDA stream. Queue the advanced
        # indexing copy before another thread can overwrite the static output.
        # Each caller owns its diagonal until its host transfers are complete.
        with self._locks[len(host_tiles)]:
            output = self(host_tiles)
            return output[:, :, indices, indices]


def prepare_teacher_readout(
    model: torch.nn.Module, contract: Mapping, device: torch.device
) -> torch.Tensor:
    """Set the frozen teacher to evaluation mode and build its central diagonal."""
    model.eval()
    attention = contract["attention"]
    center = int(attention["central_size"])
    offset = (int(attention["tile_size"]) - center) // 2
    return torch.arange(offset, offset + center, device=device, dtype=torch.long)


def prepare_teacher_tiles(
    matrices: Mapping[int, np.ndarray], contract: Mapping, *, quantile_function=None, half_decoder=None,
) -> tuple[int, np.ndarray]:
    """Normalize one record's tiles on the host, in the model's original order."""
    attention = contract["attention"]
    layers = tuple(int(value) for value in attention["layers_zero_based"])
    missing = [layer for layer in layers if layer not in matrices]
    if missing:
        raise KeyError(f"missing teacher attention layers: {missing}")
    selected = [np.asarray(matrices[layer]) for layer in layers]
    shapes = {values.shape for values in selected}
    if len(shapes) != 1:
        raise ValueError("teacher attention matrices have inconsistent shapes")
    shape = next(iter(shapes))
    if len(shape) != 2 or shape[0] != shape[1] or shape[0] < 1:
        raise ValueError("teacher attention matrices must be non-empty and square")
    length = int(shape[0])
    tile_size = int(attention["tile_size"])
    central_size = int(attention["central_size"])
    spans = central_tile_spans(length, tile_size, central_size)
    if half_decoder is not None and all(m.dtype == np.float16 and m.flags.c_contiguous for m in selected):
        return length, half_decoder(selected, spans, tile_size, float(attention["normalization_quantile"]),
                                    float(attention["normalization_clip"]))
    tiles = normalized_attention_tiles(
        [np.asarray(m, dtype=np.float32) for m in selected],
        spans,
        tile_size,
        float(attention["normalization_quantile"]),
        float(attention["normalization_clip"]),
        quantile_function=quantile_function,
    )
    return length, tiles


@torch.inference_mode()
def teacher_tracks_from_attention(
    model: torch.nn.Module,
    matrices: Mapping[int, np.ndarray],
    contract: Mapping,
    device: torch.device,
    *,
    cuda_graphs: CudaTeacherGraphs | None = None,
    _prepared_diagonal: torch.Tensor | None = None,
    _prepared_tiles: tuple[int, np.ndarray] | None = None,
    _quantile_function=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode one protein's state and teacher-class tracks from attention.

    Tiles are scored only on their central diagonal, so every residue is
    reconstructed exactly once and padding never contributes to a summary.
    The predictor prepares its fixed diagonal and model mode once per iteration.
    """

    if _prepared_tiles is None:
        _prepared_tiles = prepare_teacher_tiles(matrices, contract, quantile_function=_quantile_function)
    packed = teacher_device_probabilities(model, _prepared_tiles, contract, device,
                                          cuda_graphs, _prepared_diagonal)
    return unpack_teacher_tracks(packed.cpu().numpy())


@torch.inference_mode()
def teacher_device_probabilities(model, prepared_tiles, contract, device, cuda_graphs, diagonal_indices):
    """Evaluate one protein in its original tile batches and retain the tracks on device."""
    length, tiles = prepared_tiles
    teacher_classes = len(contract["teacher"]["class_names"])
    batch_size = int(contract["model"]["batch_size"])
    if batch_size < 1:
        raise ValueError("teacher tile batch size must be positive")
    if diagonal_indices is None:
        diagonal_indices = prepare_teacher_readout(model, contract, device)
    probability_batches = []
    for batch_start in range(0, len(tiles), batch_size):
        size = min(batch_size, len(tiles) - batch_start)
        if isinstance(tiles, np.ndarray):
            batch = nullcontext(torch.from_numpy(tiles[batch_start : batch_start + size]))
        else:
            batch = tiles.input_batch(batch_start, size, device, cuda_graphs)
        with batch as inputs:
            if getattr(cuda_graphs, "pack_tracks", False):
                probability_batches.append(cuda_graphs.probabilities(inputs))
                continue
            if cuda_graphs is None:
                output = model(inputs.to(device))
                diagonal = output[:, :, diagonal_indices, diagonal_indices]
            else:
                diagonal = cuda_graphs.diagonal(inputs, diagonal_indices)
        if diagonal.shape[0] != size:
            raise ValueError("teacher model returned a different number of tiles")
        probability_batches.append(_pack_teacher_diagonal(diagonal, teacher_classes))
    # Central spans tile the sequence consecutively. Only the last tile is padded.
    packed = probability_batches[0]
    if len(probability_batches) > 1:
        packed = torch.cat(probability_batches)
    return packed[:length]


def unpack_teacher_tracks(host):
    """Recover and validate the state and class tracks from one protein's copied rows."""
    state_track = np.ascontiguousarray(host[:, 0], dtype=np.float32)
    teacher_track = np.ascontiguousarray(host[:, 1:], dtype=np.float32)
    if not np.isfinite(state_track).all() or not np.isfinite(teacher_track).all():
        raise ValueError("teacher model emitted nonfinite values")
    if not np.allclose(teacher_track.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("teacher class probabilities are not normalized")
    return state_track, teacher_track


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
