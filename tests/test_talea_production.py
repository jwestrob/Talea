import joblib
import numpy as np
import pytest
import torch

from talea.artifacts import load_deployment, verified_runtime_assets
from talea.model import AttentionStateTeacherUNet
from talea.production import (
    compose_talea_score,
    flatten_global_features,
    global_head_probability,
    global_head_probabilities,
    pooled_hidden_features_for_record,
    roundtrip_attention_storage,
    standardized_probability_z,
    summarize_teacher_prediction,
    teacher_tracks_from_attention,
)


def test_attention_storage_roundtrip_reproduces_frozen_float16_contract() -> None:
    matrix = np.asarray([[0.0, 0.123456], [0.123456, -0.333333]], dtype=np.float32)
    observed = roundtrip_attention_storage(matrix, "fp16")
    expected = matrix.astype(np.float16).astype(np.float32)
    assert observed.dtype == np.float32
    np.testing.assert_array_equal(observed, expected)


def test_attention_storage_roundtrip_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="square"):
        roundtrip_attention_storage(np.ones((2, 3)), "fp16")
    with pytest.raises(ValueError, match="unsupported"):
        roundtrip_attention_storage(np.eye(2), "bf16")


def test_pooled_hidden_features_use_post_layer_tuple_and_population_std() -> None:
    hidden = tuple(
        torch.arange(2 * 6 * 3, dtype=torch.float32).reshape(2, 6, 3) + 100 * index
        for index in range(4)
    )
    indices = torch.tensor([1, 3, 4])
    observed = pooled_hidden_features_for_record(hidden, indices, 1, [0, 2])
    for layer in (0, 2):
        residues = hidden[layer + 1][1].index_select(0, indices)
        np.testing.assert_allclose(
            observed[f"layer_{layer}_mean"], residues.mean(dim=0).numpy()
        )
        np.testing.assert_allclose(
            observed[f"layer_{layer}_std"],
            residues.std(dim=0, unbiased=False).numpy(),
        )


def test_flatten_global_features_honors_frozen_key_order() -> None:
    features = {"b": np.asarray([3.0, 4.0]), "a": np.asarray([1.0, 2.0])}
    observed = flatten_global_features(features, ["a", "b"])
    np.testing.assert_array_equal(observed, [[1.0, 2.0, 3.0, 4.0]])


def test_score_composition_reproduces_frozen_reference_row() -> None:
    contract = load_deployment()["score_contract"]
    observed = compose_talea_score(
        teacher_state_probability=0.04407732159566165,
        topology_z=-0.6031273763779167,
        global_probability=0.08955977693987165,
        score_contract=contract,
    )
    assert observed["teacher_state_z"] == pytest.approx(-0.43493506795904796)
    assert observed["global_z"] == pytest.approx(0.2068785172621904)
    assert observed["teacher_base_score"] == pytest.approx(-0.5190312221684823)
    assert observed["talea_teacher_score"] == pytest.approx(-0.33755378731081415)


def test_score_composition_adds_equal_weight_stable_discovery_score() -> None:
    contract = load_deployment()["score_contract"]
    observed = compose_talea_score(
        teacher_state_probability=0.04407732159566165,
        topology_z=-0.6031273763779167,
        global_probability=0.08955977693987165,
        score_contract=contract,
    )
    assert observed["talea_discovery_score"] == pytest.approx(
        0.5 * (-0.43493506795904796 + 0.2068785172621904)
    )


def test_standardized_probability_rejects_invalid_calibration() -> None:
    with pytest.raises(ValueError, match="positive"):
        standardized_probability_z(0.5, 0.0, 0.0)


def test_bundled_global_head_loads_and_batches_consistently() -> None:
    assets = verified_runtime_assets()
    bundle = joblib.load(assets.global_model)
    width = int(bundle["model"].n_features_in_) // len(bundle["feature_keys"])
    rows = [
        {
            key: np.full(width, row_index + feature_index, dtype=np.float32)
            for feature_index, key in enumerate(bundle["feature_keys"])
        }
        for row_index in range(2)
    ]
    batched = global_head_probabilities(bundle, rows)
    individual = np.asarray(
        [global_head_probability(bundle, features) for features in rows]
    )
    np.testing.assert_allclose(batched, individual, rtol=0.0, atol=1e-12)
    assert np.all((batched >= 0.0) & (batched <= 1.0))


def test_bundled_teacher_loads_and_emits_valid_tracks() -> None:
    import json

    assets = verified_runtime_assets()
    contract = json.loads(assets.teacher_contract.read_text())
    model = AttentionStateTeacherUNet(
        input_channels=int(contract["input_channels"]),
        teacher_classes=len(contract["teacher"]["class_names"]),
        base_channels=int(contract["model"]["base_channels"]),
    )
    model.load_state_dict(
        torch.load(assets.teacher_model, map_location="cpu", weights_only=True)
    )
    matrices = {
        layer: np.eye(80, dtype=np.float32) for layer in (13, 35)
    }
    state_track, teacher_track = teacher_tracks_from_attention(
        model, matrices, contract, torch.device("cpu")
    )
    summary = summarize_teacher_prediction(state_track, teacher_track, contract)
    assert state_track.shape == (80,)
    assert teacher_track.shape == (80, len(contract["teacher"]["class_names"]))
    np.testing.assert_allclose(teacher_track.sum(axis=1), 1.0, atol=1e-5)
    assert 0.0 <= summary["mean_probability"] <= 1.0
