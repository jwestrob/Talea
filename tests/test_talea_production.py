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
    prepare_teacher_tiles,
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


def test_teacher_graph_dispatch_preserves_per_protein_tile_shapes_and_tracks():
    import json
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from talea.inference import _ordered_readouts

    assets = verified_runtime_assets()
    contract = json.loads(assets.teacher_contract.read_text())
    contract["model"]["batch_size"] = 3
    model = AttentionStateTeacherUNet(
        input_channels=contract["input_channels"],
        teacher_classes=len(contract["teacher"]["class_names"]),
        base_channels=contract["model"]["base_channels"],
    )
    model.load_state_dict(torch.load(assets.teacher_model, map_location="cpu", weights_only=True))
    rng = np.random.default_rng(7783)
    observed_sizes = []
    owner = threading.current_thread()

    class ReplayOnCPU:
        def diagonal(self, host, indices):
            assert threading.current_thread() is owner
            observed_sizes.append(len(host))
            return model(host)[:, :, indices, indices]

    replay_on_cpu = ReplayOnCPU()
    inputs = [
        {layer: rng.random((length, length), dtype=np.float32) for layer in (13, 35)}
        for length in (1, 63, 64, 129, 257)
    ]
    eager = [teacher_tracks_from_attention(model, maps, contract, torch.device("cpu")) for maps in inputs]

    def prepare(index):
        assert threading.current_thread() is not owner
        return prepare_teacher_tiles(inputs[index], contract)

    with ThreadPoolExecutor(max_workers=3) as workers:
        replay_inputs = _ordered_readouts(prepare, len(inputs), workers, 3)
        for maps, tracks, tiles in zip(inputs, eager, replay_inputs, strict=True):
            observed = teacher_tracks_from_attention(
                model, maps, contract, torch.device("cpu"), cuda_graphs=replay_on_cpu, _prepared_tiles=tiles,
            )
            assert all(a.tobytes() == b.tobytes() for a, b in zip(tracks, observed, strict=True))
    assert observed_sizes == [1, 1, 1, 3, 3, 2]


def test_packed_teacher_graph_keeps_captured_index_tensor_alive(monkeypatch):
    import gc
    import weakref
    from contextlib import nullcontext
    from talea import production

    device = torch.device("cpu")
    stream = object()
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: stream)
    monkeypatch.setattr(torch.cuda, "default_stream", lambda device: None)
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    monkeypatch.setattr(torch.cuda, "CUDAGraph", object)
    monkeypatch.setattr(torch.cuda, "graph", lambda *args, **kwargs: nullcontext())

    refs = []
    def make_indices(model, contract, device):
        value = torch.tensor([1, 2], device=device)
        refs.append(weakref.ref(value))
        return value

    monkeypatch.setattr(production, "prepare_teacher_readout", make_indices)
    contract = {"attention": {"tile_size": 4}, "input_channels": 3,
                "model": {"batch_size": 2}, "teacher": {"class_names": ["a", "b"]}}
    graphs = production.CudaTeacherGraphs(torch.nn.Identity(), contract, device, pack_tracks=True)
    gc.collect()
    assert len(refs) == 1 and refs[0]() is not None
    del graphs
    gc.collect()
    assert refs[0]() is None


def test_concurrent_graph_readouts_own_independent_diagonal_results():
    from concurrent.futures import ThreadPoolExecutor
    import threading
    import time
    from talea.production import CudaTeacherGraphs

    class Replay(CudaTeacherGraphs):
        def __init__(self):
            self._locks = {1: threading.Lock()}
            self.output = torch.empty(1, 3, 8, 8)

        def __call__(self, host):
            self.output.copy_(host)
            time.sleep(0.002)  # give another worker a chance to overwrite it
            return self.output

    replay = Replay()
    indices = torch.arange(2, 6)

    def run(protein):
        host = torch.full((1, 3, 8, 8), float(protein))
        return replay.diagonal(host, indices)

    with ThreadPoolExecutor(max_workers=4) as workers:
        diagonals = list(workers.map(run, range(16)))
    for protein, values in enumerate(diagonals):
        assert torch.equal(values, torch.full((1, 3, 4), float(protein)))


def test_prepared_teacher_readout_preserves_tracks_across_partial_batches():
    from talea.production import prepare_teacher_readout

    class Teacher(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dropout = torch.nn.Dropout()
            self.outputs = []

        def forward(self, tiles):
            assert not self.dropout.training
            size = tiles.shape[-1]
            # Each tile and each class has its own distinct diagonal.
            start = sum(len(previous) for previous in self.outputs)
            count = len(tiles) * 4 * size * size
            logits = torch.arange(count).reshape(len(tiles), 4, size, size).float()
            logits = torch.sin(logits + start * 0.75)
            self.outputs.append(logits)
            return logits

    model = Teacher()
    contract = {
        "attention": {
            "layers_zero_based": [13, 35], "tile_size": 14, "central_size": 8,
            "normalization_quantile": 0.95, "normalization_clip": 8.0,
        },
        "teacher": {"class_names": ["a", "b", "c"]},
        "model": {"batch_size": 2},
    }
    device = torch.device("cpu")
    indices = prepare_teacher_readout(model, contract, device)
    for length in (1, 8, 16, 33, 40):
        matrix = np.eye(length, dtype=np.float32)
        model.outputs.clear()
        states, classes = teacher_tracks_from_attention(
            model, {13: matrix, 35: matrix}, contract, device, _prepared_diagonal=indices,
        )
        expected_states, expected_classes = [], []
        for output in model.outputs:
            diagonal = output[:, :, indices, indices]
            expected_states.extend(torch.sigmoid(diagonal[:, 0]).numpy())
            expected_classes.extend(torch.softmax(diagonal[:, 1:], dim=1).transpose(1, 2).numpy())
        assert states.tobytes() == np.concatenate(expected_states)[:length].tobytes()
        assert classes.tobytes() == np.concatenate(expected_classes)[:length].tobytes()
        assert states.flags.c_contiguous and classes.flags.c_contiguous

    # The one-off API must reapply eval even when only a child was in training mode.
    model.dropout.train()
    teacher_tracks_from_attention(model, {13: matrix, 35: matrix}, contract, device)
    assert not model.dropout.training
