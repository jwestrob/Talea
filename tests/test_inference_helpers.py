import concurrent.futures
import hashlib
import os

import numpy as np
import pytest
import torch

from talea.cli import build_parser
from talea.inference import (
    TaleaPredictor,
    _ordered_readouts,
    _readout_window,
    near_diagonal_mask,
    apc_head_mean,
    available_cpus,
    make_batches,
    postprocess_result,
    resolve_dtype,
    resolve_postprocess_workers,
)
from talea.io import SequenceRecord
from talea.model import AttentionStateTeacherUNet


def record(record_id: str, length: int) -> SequenceRecord:
    sequence = "A" * length
    return SequenceRecord(
        record_id=record_id,
        sequence=sequence,
        sequence_sha256=hashlib.sha256(sequence.encode()).hexdigest(),
    )


def test_ordered_readouts_prefetches_only_a_bounded_window():
    submitted = []
    class RecordingExecutor:
        def submit(self, fn, value):
            future = concurrent.futures.Future()
            future.set_result(fn(value))
            submitted.append(value)
            return future

    rows = _ordered_readouts(lambda i: i * 2, 10, RecordingExecutor(), width=3)
    assert next(rows) == 0
    assert submitted == [0, 1, 2, 3]
    assert list(rows) == list(range(2, 20, 2))


def test_lookahead_respects_worker_and_memory_limits():
    assert _readout_window(4, 12, False) == 4
    assert _readout_window(4, 12, True) == 8
    assert _readout_window(4, 5, True) == 5
    assert _readout_window(4, 1, True) == 1
    assert _readout_window(1, 12, True) == 1
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as workers:
        window = _readout_window(4, 12, True)
        assert list(_ordered_readouts(int, 19, workers, window)) == list(range(19))


def test_adjacent_sequence_lengths_share_mask_storage():
    small = near_diagonal_mask(257, 6, torch.device("cpu"))
    large = near_diagonal_mask(500, 6, torch.device("cpu"))
    assert small.data_ptr() == large.data_ptr()
    assert torch.equal(small, large[:257, :257])


def test_batching_obeys_square_budget_and_maximum_batch() -> None:
    records = [record("a", 8), record("b", 10), record("c", 12)]
    batches = list(make_batches(records, square_budget=300, maximum_batch=2))
    assert [[value.record_id for value in batch] for batch in batches] == [
        ["a", "b"],
        ["c"],
    ]


def test_apc_head_mean_crops_residues_and_removes_near_diagonal() -> None:
    attention = torch.rand(2, 6, 6)
    indices = torch.tensor([1, 2, 3, 4])
    observed = apc_head_mean(attention, indices, 4, minimum_separation=2)
    assert observed.shape == (4, 4)
    assert observed.dtype == np.float32
    assert np.all(np.diag(observed) == 0.0)
    assert np.all(observed[np.abs(np.subtract.outer(range(4), range(4))) < 2] == 0.0)


def test_dtype_auto_is_fp32_on_cpu() -> None:
    name, dtype = resolve_dtype("auto", torch.device("cpu"))
    assert name == "fp32"
    assert dtype is torch.float32


def test_bf16_is_rejected_on_mps_contract() -> None:
    with pytest.raises(ValueError, match="MPS"):
        resolve_dtype("bf16", torch.device("mps"))


def test_cli_defaults_to_process_postprocessing_on_every_cpu() -> None:
    args = build_parser().parse_args(["-i", "in.faa", "-o", "out.jsonl"])
    assert args.postprocess_mode == "process"
    assert args.postprocess_workers is None
    assert resolve_postprocess_workers(None, "process") == available_cpus() >= 1


def test_thread_postprocessing_defaults_to_one_worker() -> None:
    assert resolve_postprocess_workers(None, "thread") == 1
    assert resolve_postprocess_workers(3, "thread") == 3


def test_postprocess_worker_resolution_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="positive"):
        resolve_postprocess_workers(0, "process")
    with pytest.raises(ValueError, match="unsupported"):
        resolve_postprocess_workers(None, "fork")


def test_broken_worker_pool_names_the_main_guard() -> None:
    future = concurrent.futures.Future()
    future.set_exception(concurrent.futures.process.BrokenProcessPool("died"))
    with pytest.raises(RuntimeError, match="__main__"):
        postprocess_result(future)
    finished = concurrent.futures.Future()
    finished.set_result({"status": "ok"})
    assert postprocess_result(finished) == {"status": "ok"}


def bare_predictor() -> TaleaPredictor:
    """A predictor with only what the worker pool needs (no models loaded)."""
    predictor = object.__new__(TaleaPredictor)
    predictor.configs = {}
    predictor._postprocess_pool = None
    predictor._postprocess_pool_workers = 0
    return predictor


def test_process_pool_serves_later_calls_until_closed() -> None:
    with bare_predictor() as predictor:
        pool = predictor._process_pool(2)
        assert pool.submit(sum, [1, 2]).result() == 3
        assert predictor._process_pool(2) is pool
        resized = predictor._process_pool(3)
        assert resized is not pool
        with pytest.raises(RuntimeError):
            pool.submit(sum, [1])
    assert predictor._postprocess_pool is None
    with pytest.raises(RuntimeError):
        resized.submit(sum, [1])


def test_process_pool_replaces_a_broken_pool() -> None:
    with bare_predictor() as predictor:
        pool = predictor._process_pool(2)
        with pytest.raises(concurrent.futures.process.BrokenProcessPool):
            pool.submit(os._exit, 1).result()
        replacement = predictor._process_pool(2)
        assert replacement is not pool
        assert replacement.submit(sum, [4, 5]).result() == 9


def test_teacher_runtime_model_output_shape() -> None:
    model = AttentionStateTeacherUNet(
        input_channels=3,
        teacher_classes=5,
        base_channels=4,
    )
    observed = model(torch.zeros(2, 3, 32, 32))
    assert observed.shape == (2, 6, 32, 32)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("extent,phase", [("state", True), ("registered", False)])
def test_packed_readouts_equal_individual_heads(dtype, extent, phase):
    from talea.production import pooled_hidden_features_for_layers

    generator = torch.Generator().manual_seed(2711)
    predictor = object.__new__(TaleaPredictor)
    predictor.expected_layers = (13, 35)
    predictor.openness_layers = (7, 15, 23, 31)
    predictor.region_probe_layer = 35
    predictor.probe_layer = 35 if phase else None
    predictor.primary_extent = extent
    predictor._region_probe_weights = torch.randn(48, generator=generator)
    predictor._region_probe_bias = 0.031
    predictor._openness_weights = torch.randn(192, generator=generator)
    predictor._openness_bias = -0.051
    predictor.probe_weights = torch.randn(48, 3, generator=generator)
    predictor.probe_bias = torch.randn(3, generator=generator)
    hidden = {
        layer: torch.randn(2, 58, 48, generator=generator, dtype=dtype)
        for layer in (*predictor.expected_layers, *predictor.openness_layers)
    }
    for indices, span in (
        (torch.tensor([7]), slice(7, 8)),
        (torch.arange(1, 54, 2), None),
        (torch.arange(3, 55), slice(3, 55)),
    ):
        means, structure, region, gate = predictor._hidden_readouts(hidden, indices, 1, span)
        reference = pooled_hidden_features_for_layers(hidden, indices, 1, predictor.expected_layers)
        for name in reference:
            np.testing.assert_array_equal(means[name], reference[name])
        np.testing.assert_array_equal(
            structure, predictor.residue_structure_probabilities(hidden, indices, 1)
        )
        np.testing.assert_array_equal(region, predictor.region_probabilities(hidden, indices, 1))
        assert gate == predictor.openness_logit(hidden, indices, 1)
        from talea.inference import _pool_gate_layer
        for grouped in (True, False):
            compact = dict(hidden)
            for layer in predictor.openness_layers:
                compact[layer] = _pool_gate_layer(hidden[layer], [(indices, span)] * 2, grouped)
            assert predictor._hidden_readouts(compact, indices, 1, span)[3] == gate


def test_batch_token_positions_with_internal_specials_and_padding():
    from types import SimpleNamespace
    from talea.inference import _batch_residue_indices

    tokenized = {
        "special_tokens_mask": torch.tensor([[1, 0, 1, 0, 1], [1, 1, 0, 0, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]]),
    }
    records = [SimpleNamespace(length=2, record_id="A"), SimpleNamespace(length=1, record_id="B")]
    result = _batch_residue_indices(tokenized, records, "cpu")
    assert [positions.tolist() for positions, span in result] == [[1, 3], [2]]
    assert [span for positions, span in result] == [None, slice(2, 3)]
    with pytest.raises(ValueError, match="tokenized"):
        _batch_residue_indices(tokenized, records[::-1], "cpu")


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_contiguous_attention_crop_matches_gathers_bitwise(dtype):
    generator = torch.Generator().manual_seed(2711)
    for offset, length, padded in ((1, 1, 6), (2, 30, 36), (3, 63, 76)):
        attention = torch.rand(5, padded, padded, generator=generator, dtype=dtype)
        indices = torch.arange(offset, offset + length)
        reference = apc_head_mean(attention, indices, length, 3, torch.float16)
        result = apc_head_mean(attention, indices, length, 3, torch.float16,
                               residue_slice=slice(offset, offset + length))
        assert result.tobytes() == reference.tobytes()


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"
))])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_attention_staging_matches_blocking_copies_and_preserves_worker_arrays(device, dtype):
    from talea.inference import _AttentionStaging
    transfer = _AttentionStaging(pin_memory=device == "cuda")
    stored = []
    expected = []
    for length in (11, 13, 9, 37, 12):
        current = {}
        for layer in (13, 35):
            values = torch.arange(length**2, dtype=torch.float32, device=device)
            matrix = torch.sin(values + layer).reshape(length, length).to(dtype)
            current[layer] = matrix.cpu().float().numpy()
            transfer.enqueue(layer, matrix)
        # The real parent calls _hidden_readouts, whose synchronous copy releases
        # both enqueued attention maps to consumers on the same stream.
        torch.ones(1, device=device).cpu()
        expected.append(current)
        stored.append(transfer.unpack_after_readouts(length))
    for before, after in zip(expected, stored, strict=True):
        for layer in (13, 35):
            assert before[layer].tobytes() == after[layer].tobytes()


def test_record_worker_threads_own_independent_attention_staging():
    import threading
    from talea.inference import _AttentionStaging
    transfer = _AttentionStaging(pin_memory=False)
    barrier = threading.Barrier(4)

    def simulate_readout(protein):
        transfer.enqueue(13, torch.full((20, 20), protein, dtype=torch.float16))
        barrier.wait(timeout=10)
        return transfer.unpack_after_readouts(20)[13]

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as workers:
        result = list(workers.map(simulate_readout, range(4)))
    for protein, matrix in enumerate(result):
        np.testing.assert_array_equal(matrix, np.full((20, 20), protein, dtype=np.float32))


def test_teacher_graph_cache_is_opt_in_and_uses_one_stream(monkeypatch):
    import contextlib
    import threading
    from types import SimpleNamespace
    import talea.inference as inference
    predictor = object.__new__(TaleaPredictor)
    predictor.device = torch.device("cpu")
    assert predictor._get_teacher_graphs(True) is None
    predictor.device = torch.device("cuda")
    assert predictor._get_teacher_graphs(False) is None
    predictor._teacher_graphs = None
    predictor.teacher_model = predictor.teacher_contract = None
    synchronized = []
    current = [SimpleNamespace(synchronize=lambda: synchronized.append("warm-up"))]
    monkeypatch.setattr(torch.cuda, "device", lambda d: contextlib.nullcontext())
    monkeypatch.setattr(torch.cuda, "current_stream", lambda d: current[0])
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(inference, "CudaTeacherGraphs", lambda *args: SimpleNamespace(
        stream=current[0], capture_thread=threading.current_thread(),
    ))
    cached = predictor._get_teacher_graphs(True)
    assert cached is not None and predictor.teacher_cuda_graph_active
    assert predictor._get_teacher_graphs(True) is cached
    with concurrent.futures.ThreadPoolExecutor(1) as pool:
        assert pool.submit(predictor._get_teacher_graphs, True, owner_thread=True).result() is not cached
    assert synchronized == ["warm-up"]
    assert predictor._get_teacher_graphs(True, owner_thread=True).capture_thread is threading.current_thread()
    assert synchronized == ["warm-up", "warm-up"]
    synchronized.clear()
    current[0] = object()
    assert predictor._get_teacher_graphs(True) is None
    assert not predictor.teacher_cuda_graph_active
    readout_graph = predictor._get_teacher_graphs(True, replace_stream=True)
    assert synchronized == ["warm-up"]
    assert readout_graph is not cached and readout_graph.stream is current[0]
    assert predictor._get_teacher_graphs(True, replace_stream=True) is readout_graph
    assert predictor.teacher_cuda_graph_active


def test_device_autocast_uses_legacy_torch22_queries(monkeypatch):
    from talea.inference import _device_autocast_enabled

    monkeypatch.delattr(torch.amp, "is_autocast_available", raising=False)
    monkeypatch.setattr(torch, "is_autocast_enabled", lambda: False)
    monkeypatch.setattr(torch, "is_autocast_cpu_enabled", lambda: True)
    assert _device_autocast_enabled("cpu")
    assert not _device_autocast_enabled("cuda")
    assert not _device_autocast_enabled("mps")


def test_device_autocast_skips_unsupported_backend(monkeypatch):
    from talea.inference import _device_autocast_enabled

    monkeypatch.setattr(torch.amp, "is_autocast_available", lambda backend: backend == "cpu", raising=False)
    def supported_only(backend):
        assert backend == "cpu"
        return True
    monkeypatch.setattr(torch, "is_autocast_enabled", supported_only)
    assert _device_autocast_enabled("cpu")
    assert not _device_autocast_enabled("mps")


def test_device_autocast_matches_cpu_context():
    from talea.inference import _device_autocast_enabled

    with torch.autocast("cpu", enabled=False):
        assert not _device_autocast_enabled("cpu")
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert _device_autocast_enabled("cpu")


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("band_mode", ["off", "dense", "sparse"])
def test_dense_apc_grouping_preserves_reduction_bits_and_record_order(dtype, band_mode):
    from talea.inference import _apc_head_mean_tensor, _equal_apc_groups, _grouped_apc_jobs, _layer_apc
    lengths = [63] * 11 + [64] * 5 + [65]
    positions = [(torch.arange(1, n + 1), slice(1, n + 1)) for n in lengths]
    positions[9] = (torch.cat((torch.arange(1, 63), torch.tensor([64]))), None)
    groups = list(_equal_apc_groups(positions, 18))
    assert groups == [(0, 8), (8, 9), (9, 10), (10, 11), (11, 16), (16, 17)]
    generator = torch.Generator().manual_seed(78)
    maps = {k: torch.rand(len(lengths), 18, 68, 68, generator=generator, dtype=dtype) / 68 for k in (13, 35)}
    stats = {"executed_groups": 0, "executed_records": 0}
    jobs = _grouped_apc_jobs(maps, (13, 35), positions, 4, torch.float16,
                             groups=groups, stats=stats, band_mode=band_mode)
    with concurrent.futures.ThreadPoolExecutor(3) as pool:
        delivered = list(_ordered_readouts(lambda item: item, len(lengths), pool, width=3, items=jobs))
    assert [n for n, _ in delivered] == list(range(len(lengths)))
    assert stats == {"executed_groups": 2, "executed_records": 13}
    early = {k: _layer_apc(value, positions, 4, torch.float16, grouped=True, band_mode=band_mode) for k, value in maps.items()}
    for n, grouped in delivered:
        ix, span = positions[n]
        for k in maps:
            scalar = _apc_head_mean_tensor(maps[k][n], ix, len(ix), 4, torch.float16, span)
            assert torch.equal(early[k][n].view(torch.int16), scalar.view(torch.int16))
            if grouped is not None:
                assert torch.equal(grouped[k].view(torch.int16), scalar.view(torch.int16))

def test_record_workers_use_prefetch_or_ordinary_graph_stream(monkeypatch):
    from talea.inference import _record_worker_stream
    ordinary, readout = object(), object()
    calls = []

    def current(device):
        calls.append(device)
        return ordinary

    monkeypatch.setattr(torch.cuda, "current_stream", current)
    cuda = torch.device("cuda")
    assert _record_worker_stream(cuda, readout) is readout
    assert _record_worker_stream(torch.device("cpu"), None) is None
    assert not calls
    assert _record_worker_stream(cuda, None) is ordinary
    assert calls == [cuda]


def test_grouped_teacher_transfers_preserve_order_and_partial_group():
    from talea.inference import _ordered_teacher_readbacks
    from talea.production import unpack_teacher_tracks

    calls = []
    def launch(n):
        calls.append(n)
        rows = torch.zeros(n, 3)
        rows[:, 0] = torch.arange(n) * 0.125
        rows[:, 1] = 1.0
        return rows
    lengths = [3, 6, 1, 4, 2]
    result = _ordered_teacher_readbacks((((n,), n) for n in lengths), launch, None, 3)
    first = next(result)
    assert calls == lengths[:3]
    records = [first, *result]
    assert calls == lengths
    for (length, state, classes), n in zip(records, lengths, strict=True):
        assert length == n
        expected_state, expected_classes = unpack_teacher_tracks(launch(n).numpy())
        assert state.tobytes() == expected_state.tobytes()
        assert classes.tobytes() == expected_classes.tobytes()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA pinned-host transfers")
def test_async_teacher_readback_preserves_partial_group_and_order():
    from talea.inference import _ordered_teacher_readbacks

    stream = torch.cuda.Stream()
    lengths = [3, 6, 1, 4, 2, 7, 5]
    def run(asynchronous):
        calls = []
        def launch(length):
            calls.append(length)
            packed = torch.zeros((length, 3), device="cuda")
            packed[:, 0] = torch.arange(length, device="cuda") * 0.125
            packed[:, 1] = 1.0
            return packed
        rows = list(_ordered_teacher_readbacks(
            (((n,), n) for n in lengths), launch, stream, 3, asynchronous=asynchronous,
        ))
        assert calls == lengths
        return [(length, state.tobytes(), classes.tobytes()) for length, state, classes in rows]
    assert run(True) == run(False)
