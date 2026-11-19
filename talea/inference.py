"""End-to-end sequence-only Talea inference."""

from __future__ import annotations

import concurrent.futures
import contextlib
import gc
from functools import lru_cache, partial
import hashlib
import json
import multiprocessing
import os
import sys
import threading
import time
from typing import NamedTuple
from itertools import cycle
from collections import deque
from collections.abc import Iterable, Iterator, Sequence

import joblib
import numpy as np
import torch
import transformers
from packaging.version import Version
from transformers import AutoModelForMaskedLM, AutoTokenizer

from talea.artifacts import RuntimeAssets, load_deployment, verified_runtime_assets
from talea.capture import selective_capture
from talea.io import SequenceRecord
from talea.long_protein import (
    DEFAULT_WINDOW_LENGTH,
    DEFAULT_WINDOW_STRIDE,
    iter_windowed_predictions,
)
from talea.model import AttentionStateTeacherUNet
from talea.phase_ssm import probe_parameters, runtime_config
from talea.repeat_state import compiled_teacher_quantile
from talea.regions import PRIMARY_EXTENTS, finish_regions
from talea.postprocess_pool import initialize_worker, run_record, run_finalized_record
from talea.production import (
    CudaTeacherGraphs,
    compose_talea_score,
    global_head_probabilities,
    summarize_teacher_prediction,
    prepare_teacher_readout,
    prepare_teacher_tiles,
    teacher_tracks_from_attention,
    teacher_device_probabilities,
    unpack_teacher_tracks,
)
from talea.topology import process_attention_record
from talea.shared_attention import SharedAttentionArena


def automatic_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_device(requested: str | torch.device) -> torch.device:
    device = automatic_device() if str(requested) == "auto" else torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    return device


def _device_autocast_enabled(device_type: str) -> bool:
    """Query the calling thread's autocast state for this backend."""
    available = getattr(torch.amp, "is_autocast_available", None)
    if available is not None:
        # PyTorch 2.4 added the device overload, initially without MPS support.
        return available(device_type) and torch.is_autocast_enabled(device_type)
    # PyTorch 2.2/2.3 expose CUDA via the no-argument API and separate
    # backend-specific functions for CPU and other supported devices.
    if device_type == "cuda":
        return torch.is_autocast_enabled()
    query = getattr(torch, f"is_autocast_{device_type}_enabled", None)
    return query() if query is not None else False


def resolve_dtype(
    requested: str, device: torch.device
) -> tuple[str, torch.dtype]:
    name = "fp32" if requested == "auto" and device.type == "cpu" else requested
    if name == "auto":
        name = "fp16"
    if name == "bf16" and device.type == "mps":
        raise ValueError("bf16 inference is not supported on MPS")
    values = {
        "fp16": torch.float16,
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
    }
    if name not in values:
        raise ValueError(f"unsupported inference dtype: {name}")
    return name, values[name]


POSTPROCESS_MODES = ("thread", "process")
BROKEN_POOL_MESSAGE = (
    "a postprocess worker process exited abruptly. Process mode starts workers "
    "that re-import the calling script, so a script that runs Talea must keep "
    'that code under `if __name__ == "__main__":`; a worker can also be killed '
    'for lack of memory. postprocess_mode="thread" runs without worker processes.'
)
UNIT_PLACEMENTS = ("phase-ssm", "comb")


def available_cpus() -> int:
    """CPUs this process may run on.

    The scheduler affinity mask where the platform reports one, so a Slurm or
    cgroup allocation is respected; the machine's CPU count elsewhere (macOS).
    """
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def resolve_postprocess_workers(workers: int | None, mode: str) -> int:
    """Worker count for the postprocess pool; ``None`` picks the mode default.

    Process mode defaults to every available CPU. Thread mode defaults to one,
    because the GIL-bound postprocessing convoys rather than scales on threads.
    """
    if mode not in POSTPROCESS_MODES:
        raise ValueError(f"unsupported postprocess mode: {mode}")
    if workers is None:
        return available_cpus() if mode == "process" else 1
    if workers < 1:
        raise ValueError("postprocess worker count must be positive")
    return int(workers)


def postprocess_result(future: concurrent.futures.Future):
    """A postprocess future's result, explaining a broken worker pool."""

    try:
        return future.result()
    except concurrent.futures.process.BrokenProcessPool as error:
        raise RuntimeError(BROKEN_POOL_MESSAGE) from error


def enable_deterministic_convolution() -> None:
    """Pin cuDNN to deterministic convolution algorithms.

    Talea's GPU results are not reproducible between processes, and the cause is
    narrow. Measured across three separate processes on one H200 (job 1161576),
    every ESM++ stage is bit-identical -- raw attention, pooled hidden states,
    ``last_hidden_state``, and the APC matrices all compare exactly equal. The
    only stage that varies is the teacher UNet, whose ``torch.conv2d`` calls let
    cuDNN pick an algorithm per process; the state track differed by 1.4e-4 to
    3.8e-4 between runs.

    That looks negligible until it passes through ``standardized_probability_z``,
    where a logit transform divided by a control standard deviation turns ~2e-4
    of probability into ~3.8e-2 of ``teacher_state_z``.

    Enabling this made two of three test records bit-identical across processes
    and shrank the third from 2.7e-4 to 5.2e-6, so it is a large improvement
    rather than a guarantee. It is opt-in because it changes results relative to
    every run made so far -- including the frozen artifacts, which were
    themselves produced without it.
    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def postprocess_pool_context():
    """Multiprocessing context for the postprocess pool.

    ``forkserver`` where available: the server is a fresh interpreter (so it
    never inherits the parent's CUDA context) that preloads the torch-free
    worker module once, after which each worker forks from it almost for free.
    ``spawn`` elsewhere -- correct but pays the import per worker.
    """
    if "forkserver" in multiprocessing.get_all_start_methods():
        context = multiprocessing.get_context("forkserver")
        # Deliberately NOT ['__main__']: preloading the caller's main module is
        # the default, is unnecessary here, and breaks callers whose __main__ is
        # not importable (a REPL or a heredoc).
        context.set_forkserver_preload(["talea.postprocess_pool"])
        return context
    return multiprocessing.get_context("spawn")


@lru_cache(maxsize=16)
def _separation_mask_block(capacity: int, minimum_separation: int, device: torch.device):
    positions = torch.arange(capacity, device=device)
    return (positions[:, None] - positions[None, :]).abs() < minimum_separation


@lru_cache(maxsize=32)
def near_diagonal_mask(
    sequence_length: int, minimum_separation: int, device: torch.device
) -> torch.Tensor:
    """Reuse the same distance mask for all lengths in a power-of-two bucket."""
    capacity = 1 << max(0, (int(sequence_length) - 1).bit_length())
    return _separation_mask_block(capacity, minimum_separation, device)[:sequence_length, :sequence_length]


def _apc_fusion(device, dtype, mode):
    if mode not in ("off", "sym", "full"):
        raise ValueError(f"unknown APC fusion mode {mode}")
    if mode == "off" or device.type != "cuda" or dtype != torch.float16:
        return None
    if "A5000" not in torch.cuda.get_device_name(device):
        return None
    try:
        from talea._apc_cuda import FusedAPC
    except ImportError:
        return None
    return FusedAPC(mode)


def _apc_head_mean_tensor(
    layer_attention: torch.Tensor,
    indices: torch.Tensor,
    sequence_length: int,
    minimum_separation: int,
    storage_dtype: torch.dtype | None = None,
    residue_slice: slice | None = None,
    fusion=None,
) -> torch.Tensor:
    """Symmetrize, APC-correct, head-average, and crop one attention layer.

    The crop precedes the symmetrization: both operations are elementwise over
    the same index set, so the result is unchanged, but every subsequent
    reduction then runs over ``L x L`` rather than the padded ``T x T``.

    ``storage_dtype`` casts on the accelerator before the host transfer. The
    frozen contract stores these matrices as fp16, so casting first halves the
    transfer without changing the stored representation.
    """

    fused = fusion is not None and residue_slice is not None and layer_attention.dtype == torch.float16
    if fused:
        attention = fusion.symmetrize(layer_attention, residue_slice.start, sequence_length)
    else:
        if residue_slice is None:
            attention = layer_attention.index_select(1, indices).index_select(2, indices)
        else:
            # Crop both axes in one copy, retaining the contiguous layout used by
            # the reductions below (including when the batch contains padding).
            attention = layer_attention[:, residue_slice, residue_slice].contiguous()
        attention = 0.5 * (attention + attention.transpose(-1, -2))
    row = attention.sum(dim=-1, keepdim=True)
    column = attention.sum(dim=-2, keepdim=True)
    total = attention.sum(dim=(-1, -2), keepdim=True) + 1e-12
    if fused and fusion.full:
        attention = fusion.correct(attention, row, column, total).mean(dim=0)
    else:
        attention = torch.relu(attention - (row * column) / total).mean(dim=0)
    if minimum_separation > 0:
        attention = attention.masked_fill(
            near_diagonal_mask(sequence_length, minimum_separation, attention.device),
            0.0,
        )
    attention.fill_diagonal_(0.0)
    if storage_dtype is not None and attention.dtype != storage_dtype:
        attention = attention.to(storage_dtype)
    return attention


def _aligned_record(tensor):
    # CUDA may vectorize a reduction differently for a misaligned input. Each
    # reference reduction receives a fresh contiguous allocation.
    if tensor.device.type == "cuda" and tensor.data_ptr() % 512:
        return tensor.clone(memory_format=torch.contiguous_format)
    return tensor


def _equal_length_apc(attention, length, minimum_separation, storage_dtype, span, fusion=None, batched=()):
    """Batch pointwise operations; retain each original reduction's shape and strides."""
    fused = fusion is not None and attention.dtype == torch.float16
    if fused:
        symmetric = fusion.symmetrize(attention, span.start, length)
    else:
        cropped = attention[:, :, span, span].contiguous()
        symmetric = 0.5 * (cropped + cropped.transpose(-1, -2))
    if batched:
        # Each reference reduction starts from an independently aligned allocation.
        records = (tuple(_aligned_record(record) for record in symmetric.unbind())
                   if not all(which in batched for which in ("row", "column", "total")) else ())
        rows = (symmetric.sum(dim=-1, keepdim=True) if "row" in batched else
                torch.stack([record.sum(dim=-1, keepdim=True) for record in records]))
        columns = (symmetric.sum(dim=-2, keepdim=True) if "column" in batched else
                   torch.stack([record.sum(dim=-2, keepdim=True) for record in records]))
        totals = (symmetric.sum(dim=(-1, -2), keepdim=True) if "total" in batched else
                  torch.stack([record.sum(dim=(-1, -2), keepdim=True) for record in records]))
        total = totals + 1e-12
    else:
        rows, columns, totals = [], [], []
        for record in symmetric.unbind():
            record = _aligned_record(record)
            rows.append(record.sum(dim=-1, keepdim=True))
            columns.append(record.sum(dim=-2, keepdim=True))
            totals.append(record.sum(dim=(-1, -2), keepdim=True))
        total = torch.stack(totals) + 1e-12
        rows, columns = torch.stack(rows), torch.stack(columns)
    if fused and fusion.full:
        corrected = fusion.correct(symmetric, rows, columns, total)
    else:
        corrected = torch.relu(symmetric - (rows * columns) / total)
    result = (corrected.mean(dim=1) if "head" in batched else
              torch.stack([_aligned_record(record).mean(dim=0) for record in corrected.unbind()]))
    if minimum_separation > 0:
        result = result.masked_fill(near_diagonal_mask(length, minimum_separation, result.device), 0)
    result.diagonal(dim1=-2, dim2=-1).fill_(0)
    return result.to(storage_dtype) if storage_dtype is not None else result


_APC_GROUP_ELEMENTS = 4 * 1024 ** 2


def _equal_apc_groups(indices_by_record, heads, element_budget=_APC_GROUP_ELEMENTS):
    """Consecutive equal token spans, bounded independently of ESM batch size."""
    start = 0
    while start < len(indices_by_record):
        indices, span = indices_by_record[start]
        stop = start + 1
        if span is not None:
            capacity = min(8, max(1, element_budget // (heads * len(indices) ** 2)))
            while stop < min(start + capacity, len(indices_by_record)):
                if indices_by_record[stop][1] != span:
                    break
                stop += 1
        yield start, stop
        start = stop


def _grouped_apc_jobs(attentions, layers, indices_by_record, separation, dtype, stream=None, groups=None, stats=None, fusion=None, batched=(), handoff=False):
    """Yield one device map per layer and protein, or the scalar singleton path."""
    heads = attentions[layers[0]].shape[1]
    for start, stop in (_equal_apc_groups(indices_by_record, heads) if groups is None else groups):
        if stop == start + 1:
            yield start, None
            continue
        indices, span = indices_by_record[start]
        # The dispatcher must issue grouped work on the consumer's stream
        # after the wait_event() for this capture.
        with torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext():
            maps = {
                layer: _equal_length_apc(attentions[layer][start:stop], len(indices), separation, dtype, span, fusion, batched)
                for layer in layers
            }
        if stats is not None:
            stats["executed_groups"] += 1
            stats["executed_records"] += stop - start
        ready = stream.record_event() if handoff else None
        for offset in range(stop - start):
            item = start + offset, {layer: maps[layer][offset] for layer in layers}
            yield (*item, ready) if handoff else item


class _PendingCapture(NamedTuple):
    shape: torch.Size
    itemsize: int
    future: concurrent.futures.Future

    def element_size(self):
        return self.itemsize


def _layer_apc(weights, indices_by_record, separation, dtype, grouped, fusion=None, batched=()):
    """Run the existing scalar and equal-length kernels, in record order."""
    maps = []
    groups = (_equal_apc_groups(indices_by_record, weights.shape[1]) if grouped
              else ((i, i + 1) for i in range(len(indices_by_record))))
    for start, stop in groups:
        indices, span = indices_by_record[start]
        length = len(indices)
        if stop == start + 1:
            maps.append(_apc_head_mean_tensor(weights[start], indices, length, separation, dtype, span, fusion))
        else:
            maps.extend(_equal_length_apc(weights[start:stop], length, separation, dtype, span, fusion, batched).unbind())
    return maps


def _queue_capture(tensor, function, executor, stream):
    """Launch readout from a layer hook while the producer continues to encode."""
    ready = torch.cuda.current_stream(tensor.device).record_event()

    @torch.inference_mode()
    def work():
        with torch.cuda.stream(stream):
            stream.wait_event(ready)
            tensor.record_stream(stream)
            output = function(tensor)
            return output, stream.record_event()

    return _PendingCapture(tensor.shape, tensor.element_size(), executor.submit(work))


def _collect_capture(value, consumer):
    output, ready = value.future.result()
    consumer.wait_event(ready)
    if isinstance(output, torch.Tensor):
        output.record_stream(consumer)
    else:
        for tensor in output:
            tensor.record_stream(consumer)
    return output


def _checked_attention_matrix(result, sequence_length):
    if result.shape != (sequence_length, sequence_length) or not result.size:
        raise ValueError("cropped attention matrix does not match sequence length")
    if not np.isfinite(result).all():
        raise ValueError("attention matrix contains nonfinite values")
    return result


def apc_head_mean(
    layer_attention, indices, sequence_length, minimum_separation, storage_dtype=None,
    residue_slice=None,
):
    tensor = _apc_head_mean_tensor(
        layer_attention, indices, sequence_length, minimum_separation, storage_dtype,
        residue_slice,
    )
    return _checked_attention_matrix(tensor.to("cpu", dtype=torch.float32).numpy(), sequence_length)


class _AttentionStaging(threading.local):
    """Thread-local pinned scratch, reused once a protein's readouts reach the host."""

    def __init__(self, pin_memory=True):
        self._buffers = {}
        self._pin_memory = pin_memory

    def enqueue(self, layer, matrix):
        count = matrix.numel()
        buffer = self._buffers.get(layer)
        if buffer is None or buffer.numel() < count or buffer.dtype != matrix.dtype:
            # Lengths are sorted; grow geometrically instead of pinning new pages
            # for every additional residue. Total space is bounded by O(max L²).
            capacity = 1 << (count - 1).bit_length()
            buffer = torch.empty(capacity, dtype=matrix.dtype, device="cpu", pin_memory=self._pin_memory)
            self._buffers[layer] = buffer
        buffer[:count].view_as(matrix).copy_(matrix, non_blocking=True)

    def unpack_after_readouts(self, length, native_half=False):
        matrices = {}
        for layer, buffer in self._buffers.items():
            host = buffer[:length * length].reshape(length, length)
            # Workers (and possibly their IPC feeder) retain these arrays after
            # the following protein starts using our pinned storage.
            if native_half and host.dtype == torch.float16:
                # The teacher histogram validates every half word before this
                # record is submitted to a topology worker.
                matrices[layer] = host.numpy().copy()
                continue
            matrix = host.to(dtype=torch.float32, copy=True).numpy()
            matrices[layer] = _checked_attention_matrix(matrix, length)
        return matrices


def _batch_residue_indices(tokenized, records, device):
    """Resolve token positions on the host, then transfer indices once per batch."""
    residue_mask = tokenized["special_tokens_mask"] == 0
    if "attention_mask" in tokenized:
        residue_mask &= tokenized["attention_mask"].bool()
    lengths = [record.length for record in records]
    tokens = residue_mask.nonzero(as_tuple=False)
    # nonzero walks rows in order, including tokenizers with internal special tokens.
    for record, count in zip(records, residue_mask.sum(dim=1).tolist(), strict=True):
        if count != record.length:
            raise ValueError(f"tokenized residue count disagrees for {record.record_id}")
    positions = tokens[:, 1].contiguous()
    slices = []
    offset = 0
    for length in lengths:
        first = int(positions[offset])
        offset += length
        stop = int(positions[offset - 1]) + 1
        slices.append(slice(first, stop) if stop - first == length else None)
    return list(zip(positions.to(device).split(lengths), slices, strict=True))


def _record_hidden_states(state, batch_position, indices, residue_slice):
    row = state[batch_position]
    if residue_slice is None:
        return row.index_select(0, indices).float()
    # The original gather creates an aligned, contiguous tensor even for fp32.
    return row[residue_slice].to(dtype=torch.float32, memory_format=torch.contiguous_format, copy=True)


def _pool_gate_layer(state, indices_by_record, group_equal):
    """Reduce only valid residues, using their original order and fp32 accumulation."""
    means = []
    start = 0
    while start < len(indices_by_record):
        indices, span = indices_by_record[start]
        stop = start + 1
        if group_equal and span is not None:
            while stop < len(indices_by_record) and indices_by_record[stop][1] == span:
                stop += 1
        if stop == start + 1:
            residues = _record_hidden_states(state, start, indices, span)
            means.append(residues.mean(dim=0))
        else:
            residues = state[start:stop, span].to(dtype=torch.float32, memory_format=torch.contiguous_format, copy=True)
            means.extend(residues.mean(dim=1).unbind())
        start = stop
    return torch.stack(means)


def _ordered_readouts(function, count, executor, width, items=None):
    """Keep at most width records in flight and deliver results in input order."""
    positions = iter(range(count) if items is None else items)
    if executor is None or width == 1:
        yield from map(function, positions)
        return
    pending = deque()
    try:
        for _ in range(min(width, count)):
            pending.append(executor.submit(function, next(positions)))
        while pending:
            result = pending.popleft().result()
            following = next(positions, None)
            if following is not None:
                pending.append(executor.submit(function, following))
            yield result
    finally:
        for future in pending:
            future.cancel()


def _ordered_teacher_readbacks(readouts, launch, stream, depth, *, asynchronous=False):
    """Launch teacher calls in order and optionally overlap one grouped transfer."""
    def activate():
        return torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext()

    def transfer(group):
        with activate():
            tensors = [tensor for _, tensor in group]
            packed = torch.cat(tensors) if len(tensors) > 1 else tensors[0]
            event = None
            if asynchronous:
                host = torch.empty_like(packed, device="cpu", pin_memory=True)
                host.copy_(packed, non_blocking=True)
                event = torch.cuda.current_stream(packed.device).record_event()
            else:
                host = packed.cpu()
        return [(inputs, len(tensor)) for inputs, tensor in group], host, event

    def retrieve(transfer_result):
        records, host, event = transfer_result
        if event is not None:
            event.synchronize()
        array = host.numpy()
        offset = 0
        for inputs, length in records:
            end = offset + length
            yield (*inputs, *unpack_teacher_tracks(array[offset:end]))
            offset = end

    group = []
    pending = None
    for inputs, tiles in readouts:
        with activate():
            tensor = launch(tiles)
        group.append((inputs, tensor))
        if len(group) == depth:
            current = transfer(group)
            group.clear()
            if pending is not None:
                yield from retrieve(pending)
            if asynchronous:
                pending = current
            else:
                yield from retrieve(current)
            del current
    if pending is not None:
        yield from retrieve(pending)
        del pending
    if group:
        yield from retrieve(transfer(group))


def _forward_pair(batch, following, future, forward, producer, can_overlap):
    """Release the current capture to its consumer and schedule at most one successor."""
    current = forward(batch) if future is None else future.result()
    upcoming = None
    if (producer is not None and following is not None
            and can_overlap(batch, following, current)):
        upcoming = producer.submit(forward, following)
    return current, upcoming


def _record_worker_stream(device, readout_stream):
    if device.type != "cuda":
        return None
    return readout_stream if readout_stream is not None else torch.cuda.current_stream(device)


def _handoff_capture(stream, ready, attentions, hidden_states, residue_indices):
    stream.wait_event(ready)
    # Producer-owned storage must remain valid through consumer operations.
    for tensor in (*attentions.values(), *hidden_states.values()):
        tensor.record_stream(stream)
    for indices, _ in residue_indices:
        indices.record_stream(stream)


def _prefetch_memory_required(current, following, attention, hidden, width):
    """Conservative working-set estimate in addition to the live captures."""
    tokens = following[-1].length + max(2, attention.shape[-1] - current[-1].length)
    batch_size = len(following)
    # Allow six full fp32 score/softmax/capture maps plus block activations,
    # alongside current APC/teacher scratch. Original batch shapes stay fixed.
    head_bytes = attention.shape[1] * max(4, attention.element_size())
    forward = 6 * batch_size * tokens**2 * head_bytes
    forward += 24 * batch_size * tokens * hidden.shape[-1] * max(4, hidden.element_size())
    readout = 6 * width * current[-1].length**2 * head_bytes
    return forward + readout + (768 << 20)


def _prefetch_headroom(device):
    if device.type == "cuda":
        free, _ = torch.cuda.mem_get_info(device)
        # Discount cached blocks for fragmentation and leave space for the
        # runtime's allocations outside the PyTorch caching allocator.
        cached = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
        return int(.8 * (free + max(0, cached) // 2))
    if hasattr(torch.mps, "recommended_max_memory"):
        return int(.8 * (torch.mps.recommended_max_memory() - torch.mps.driver_allocated_memory()))
    return 0


def make_batches(
    records: Sequence[SequenceRecord],
    square_budget: int,
    maximum_batch: int,
) -> Iterator[list[SequenceRecord]]:
    if square_budget < 1 or maximum_batch < 1:
        raise ValueError("batch limits must be positive")
    batch: list[SequenceRecord] = []
    maximum_length = 0
    for record in records:
        candidate_maximum = max(maximum_length, record.length + 2)
        if batch and (
            len(batch) >= maximum_batch
            or candidate_maximum * candidate_maximum * (len(batch) + 1)
            > square_budget
        ):
            yield batch
            batch = []
            maximum_length = 0
        batch.append(record)
        maximum_length = max(maximum_length, record.length + 2)
    if batch:
        yield batch


def _load_teacher(
    model_path,
    contract_path,
    device: torch.device,
) -> tuple[AttentionStateTeacherUNet, dict]:
    contract = json.loads(contract_path.read_text())
    teacher = AttentionStateTeacherUNet(
        input_channels=int(contract["input_channels"]),
        teacher_classes=len(contract["teacher"]["class_names"]),
        base_channels=int(contract["model"]["base_channels"]),
    )
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    teacher.load_state_dict(state)
    teacher.eval().to(device)
    return teacher, contract


def _validate_global_bundle(bundle: dict, deployment: dict) -> None:
    contract = deployment["global_model"]
    calibration = deployment["score_contract"]["global_calibration"]
    if list(bundle["feature_keys"]) != list(contract["feature_keys"]):
        raise ValueError("global feature keys disagree with the deployment contract")
    if not np.isclose(
        float(bundle["control_mean"]),
        float(calibration["control_logit_mean"]),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("global control mean disagrees with the deployment contract")
    if not np.isclose(
        float(bundle["control_standard_deviation"]),
        float(calibration["control_logit_standard_deviation_population"]),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            "global control standard deviation disagrees with the deployment contract"
        )


class TaleaPredictor:
    """Load frozen Talea v2 once and rank one or more canonical proteins."""

    def __init__(
        self,
        *,
        device: str | torch.device = "auto",
        dtype: str = "auto",
        trust_remote_code: bool = False,
        local_files_only: bool = False,
        deterministic: bool = False,
        unit_placement: str = "phase-ssm",
        primary_extent: str = "state",
    ):
        """``unit_placement`` chooses how registered repeat units are placed.

        "phase-ssm" (default) re-places the units of every registration with the
        phase-aware state-space decoder in talea.phase_ssm; "comb" keeps the
        equal-period comb, reproducing v0.1.0 output exactly. Detection scores
        and topology diagnostics are identical under both.

        ``primary_extent`` "state" (default) cuts the primary repeat region and
        the registration to the U-Net residue-state track; "registered" keeps
        the registered extent, as v0.3.2 did (see talea.regions). Scores are
        identical under both.
        """
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        if unit_placement not in UNIT_PLACEMENTS:
            raise ValueError(f"unsupported unit placement: {unit_placement}")
        if primary_extent not in PRIMARY_EXTENTS:
            raise ValueError(f"unsupported primary extent: {primary_extent}")
        self.primary_extent = primary_extent
        self._postprocess_pool: concurrent.futures.ProcessPoolExecutor | None = None
        self._postprocess_pool_workers = 0
        self.apc_fusion_active = False
        self.apc_group_stats = None
        self._attention_arena = None
        self._teacher_graphs = None
        self._graph_readout_stream = None
        self.teacher_quantile_histogram_active = False
        self.teacher_cuda_graph_active = False
        self.cached_sdpa_bias_calls = 0
        self.fused_rotary_calls = 0
        self.early_gate_layers = 0
        self.early_apc_layers = 0
        self.unit_residual_blocks = 0
        self.arithmetic_swiglu_repairs = None
        self.fused_swiglu_calls = 0
        self.unit_placement = unit_placement
        self.deterministic = bool(deterministic)
        if self.deterministic:
            enable_deterministic_convolution()
        self.deployment = load_deployment()
        self.assets: RuntimeAssets = verified_runtime_assets(self.deployment)
        self.device = resolve_device(device)
        self.dtype_name, self.dtype = resolve_dtype(dtype, self.device)
        self.trust_remote_code = bool(trust_remote_code)
        self.local_files_only = bool(local_files_only)
        self.expected_layers = tuple(
            int(value)
            for value in self.deployment["esmplusplus"][
                "transformer_layers_zero_based"
            ]
        )
        self.configs = {
            "protein_calibration": json.loads(
                self.assets.protein_calibration.read_text()
            ),
            "local_calibration": json.loads(
                self.assets.local_calibration.read_text()
            ),
            "wedge_config": json.loads(self.assets.wedge_config.read_text()),
            "operational_config": json.loads(
                self.assets.topology_config.read_text()
            ),
        }
        self.phase_ssm = None
        if self.unit_placement == "phase-ssm":
            phase_asset = json.loads(self.assets.phase_ssm.read_text())
            self.phase_ssm = phase_asset
            self.configs["phase_ssm"] = runtime_config(phase_asset)
        region_probe = json.loads(self.assets.region_probe.read_text())
        self.region_probe_settings = region_probe["extension"]
        self.region_probe_layer = int(region_probe["hidden_state_layer_zero_based"])
        self._region_probe_weights = torch.as_tensor(
            region_probe["weights"], dtype=torch.float32, device=self.device
        )
        self._region_probe_bias = float(region_probe["bias"])
        gate = json.loads(self.assets.openness_gate.read_text())
        self.openness_layers = tuple(int(layer) for layer in gate["hidden_state_layers_zero_based"])
        self._openness_weights = torch.as_tensor(
            gate["weights"], dtype=torch.float32, device=self.device
        )
        self._openness_bias = float(gate["bias"])
        self.openness_penalty_weight = float(gate["penalty_weight"])
        self.global_bundle = joblib.load(self.assets.global_model)
        _validate_global_bundle(self.global_bundle, self.deployment)
        self.loaded_at = time.time()
        self._load_models()
        self.loaded_seconds = time.time() - self.loaded_at

    def _get_teacher_graphs(self, requested, *, replace_stream=False, owner_thread=False, pack_tracks=False, shared_pool=False):
        self.teacher_cuda_graph_active = False
        if (not requested or self.device.type != "cuda"
                or _device_autocast_enabled("cuda") or not hasattr(torch.cuda, "CUDAGraph")):
            return None
        with torch.cuda.device(self.device):
            if torch.cuda.is_current_stream_capturing():
                return None
            stream = torch.cuda.current_stream(self.device)
            cached = self._teacher_graphs
            if cached is not None:
                if cached.stream != stream and not replace_stream:
                    return None
                # cuDNN convolution plan caches are thread-local. Capture anew
                # when a caller resumes the predictor from another host thread.
                new_owner = owner_thread and cached.capture_thread is not threading.current_thread()
                if (cached.stream != stream or new_owner or getattr(cached, "pack_tracks", False) != pack_tracks
                        or getattr(cached, "shared_pool", False) != shared_pool):
                    cached.stream.synchronize()
                    self._teacher_graphs = None
                    del cached
            if self._teacher_graphs is None:
                self._teacher_graphs = CudaTeacherGraphs(self.teacher_model, self.teacher_contract, self.device, pack_tracks, shared_pool)
        self.teacher_cuda_graph_active = True
        return self._teacher_graphs

    def _process_pool(
        self, workers: int, shared_attention_mib: int = 0,
    ) -> concurrent.futures.ProcessPoolExecutor:
        """The postprocess worker pool, started once and kept for later calls.

        Each new worker re-imports the calling script, torch included when
        the script imports Talea at top level, which takes seconds on network
        storage and would dominate a call that scores a single protein. A
        call asking for a different worker count, or finding the pool broken
        by a dead worker, gets a new pool.
        """

        if shared_attention_mib < 0:
            raise ValueError("shared_attention_mib must be nonnegative")
        pool = self._postprocess_pool
        arena = getattr(self, "_attention_arena", None)
        capacity = shared_attention_mib * 1024**2
        if pool is not None and (
            workers != self._postprocess_pool_workers
            or capacity != (arena.capacity_bytes if arena is not None else 0)
            or getattr(pool, "_broken", False)
        ):
            pool.shutdown(wait=True, cancel_futures=True)
            self._postprocess_pool = self._attention_arena = pool = None
        if pool is None:
            arena = SharedAttentionArena(capacity) if capacity else None
            pool = concurrent.futures.ProcessPoolExecutor(
                max_workers=workers,
                mp_context=postprocess_pool_context(),
                initializer=initialize_worker,
                initargs=(self.configs, arena.buffer if arena is not None else None),
            )
            self._attention_arena = arena
            self._postprocess_pool = pool
            self._postprocess_pool_workers = workers
        return pool

    def close(self) -> None:
        """Stop the postprocess worker pool; a later call starts a new one."""

        self._teacher_graphs = None
        self._graph_readout_stream = None
        pool, self._postprocess_pool = self._postprocess_pool, None
        self._attention_arena = None
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)

    def __enter__(self) -> TaleaPredictor:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def _load_models(self) -> None:
        model_name = str(self.deployment["esmplusplus"]["model"])
        revision = str(self.deployment["esmplusplus"]["revision"])
        load_kwargs = {
            "local_files_only": self.local_files_only,
            "trust_remote_code": self.trust_remote_code,
            "revision": revision,
        }
        if Version(transformers.__version__) >= Version("4.56.0"):
            load_kwargs["dtype"] = self.dtype
        else:
            load_kwargs["torch_dtype"] = self.dtype
        self.esm_model = (
            AutoModelForMaskedLM.from_pretrained(model_name, **load_kwargs)
            .eval()
            .to(self.device)
        )
        self.tokenizer = getattr(self.esm_model, "tokenizer", None)
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                local_files_only=self.local_files_only,
                trust_remote_code=self.trust_remote_code,
                revision=revision,
            )
        self.teacher_model, self.teacher_contract = _load_teacher(
            self.assets.teacher_model,
            self.assets.teacher_contract,
            self.device,
        )
        observed_layers = tuple(
            int(value)
            for value in self.teacher_contract["attention"]["layers_zero_based"]
        )
        if observed_layers != self.expected_layers:
            raise ValueError(
                "teacher layers disagree with the ESM++ deployment contract"
            )
        self.probe_layer = None
        if self.phase_ssm is not None:
            weights, bias = probe_parameters(self.phase_ssm)
            self.probe_layer = int(
                self.phase_ssm["probes"]["hidden_state_layer_zero_based"]
            )
            if self.probe_layer not in self.expected_layers:
                raise ValueError("phase-SSM probe layer is not captured")
            self.probe_weights = torch.as_tensor(
                weights, dtype=torch.float32, device=self.device
            )
            self.probe_bias = torch.as_tensor(
                bias, dtype=torch.float32, device=self.device
            )

    def residue_structure_probabilities(
        self,
        hidden_states: dict[int, torch.Tensor],
        indices: torch.Tensor,
        batch_position: int,
    ) -> np.ndarray | None:
        """Per-residue helix/strand/loop probabilities from the layer-35 probes."""

        if self.probe_layer is None:
            return None
        residues = (
            hidden_states[self.probe_layer][batch_position]
            .index_select(0, indices)
            .to(torch.float32)
        )
        logits = residues @ self.probe_weights + self.probe_bias
        return torch.sigmoid(logits).cpu().numpy().astype(np.float64)

    def region_probabilities(
        self,
        hidden_states: dict[int, torch.Tensor],
        indices: torch.Tensor,
        batch_position: int,
    ) -> np.ndarray | None:
        """Per-residue probability of lying inside a repeat region (layer-35 region probe)."""

        if self.primary_extent != "state" or self.region_probe_layer not in hidden_states:
            return None
        residues = (
            hidden_states[self.region_probe_layer][batch_position]
            .index_select(0, indices)
            .to(torch.float32)
        )
        weights = torch.as_tensor(self._region_probe_weights, device=residues.device)
        logits = residues @ weights + self._region_probe_bias
        return torch.sigmoid(logits).cpu().numpy().astype(np.float64)

    def openness_logit(
        self,
        hidden_states: dict[int, torch.Tensor],
        indices: torch.Tensor,
        batch_position: int,
    ) -> float:
        """Logit of the open-versus-closed gate from residue-mean hidden states."""

        pooled = torch.cat(
            [
                hidden_states[layer][batch_position].index_select(0, indices).to(torch.float32).mean(dim=0)
                for layer in self.openness_layers
            ]
        )
        weights = torch.as_tensor(self._openness_weights, device=pooled.device)
        return float(pooled @ weights + self._openness_bias)

    def _hidden_readouts(self, hidden_states, indices, batch_position, residue_slice=None):
        """Run fixed-shape per-record reductions, then copy all small outputs together."""
        features = {}
        structure = None
        region = None
        for layer in self.expected_layers:
            residues = _record_hidden_states(hidden_states[layer], batch_position, indices, residue_slice)
            features[f"layer_{layer}_mean"] = residues.mean(dim=0)
            features[f"layer_{layer}_std"] = residues.std(dim=0, unbiased=False)
            if layer == self.probe_layer:
                structure = torch.sigmoid(residues @ self.probe_weights + self.probe_bias)
            if layer == self.region_probe_layer and self.primary_extent == "state":
                region = torch.sigmoid(residues @ self._region_probe_weights + self._region_probe_bias)

        gate_means = [
            hidden_states[layer][batch_position]
            if hidden_states[layer].ndim == 2
            else _record_hidden_states(hidden_states[layer], batch_position, indices, residue_slice).mean(dim=0)
            for layer in self.openness_layers
        ]
        gate = torch.cat(gate_means) @ self._openness_weights + self._openness_bias
        outputs = dict(features)
        if structure is not None:
            outputs["structure"] = structure
        if region is not None:
            outputs["region"] = region
        outputs["gate"] = gate
        packed = torch.cat([tensor.reshape(-1) for tensor in outputs.values()]).cpu().numpy()
        arrays = {}
        offset = 0
        for name, tensor in outputs.items():
            end = offset + tensor.numel()
            arrays[name] = packed[offset:end].reshape(tensor.shape)
            offset = end
        gate_logit = float(arrays.pop("gate"))
        probabilities = arrays.pop("structure").astype(np.float64) if structure is not None else None
        region_track = arrays.pop("region").astype(np.float64) if region is not None else None
        return arrays, probabilities, region_track, gate_logit

    @property
    def maximum_length(self) -> int:
        return int(self.deployment["input_contract"]["maximum_residues"])

    def _validate_records(self, records: Sequence[SequenceRecord]) -> None:
        identifiers: set[str] = set()
        canonical = frozenset("ACDEFGHIKLMNPQRSTVWY")
        for record in records:
            if record.record_id in identifiers:
                raise ValueError(f"duplicate record ID: {record.record_id}")
            identifiers.add(record.record_id)
            if not record.sequence:
                raise ValueError(f"empty sequence for {record.record_id}")
            unexpected = sorted(set(record.sequence) - canonical)
            if unexpected:
                raise ValueError(
                    f"noncanonical residues for {record.record_id}: "
                    f"{''.join(unexpected)}"
                )
            if record.length > self.maximum_length:
                raise ValueError(
                    f"{record.record_id} has {record.length} residues; "
                    f"frozen maximum is {self.maximum_length}"
                )
            observed_hash = hashlib.sha256(record.sequence.encode()).hexdigest()
            if observed_hash != record.sequence_sha256:
                raise ValueError(
                    f"sequence checksum mismatch for {record.record_id}"
                )

    def iter_windowed_predictions(
        self,
        records: Iterable[SequenceRecord],
        *,
        window_length: int = DEFAULT_WINDOW_LENGTH,
        window_stride: int = DEFAULT_WINDOW_STRIDE,
        **options,
    ) -> Iterator[dict]:
        """``iter_predictions`` that also accepts proteins above the maximum.

        Each longer protein is scored as overlapping windows and reported as one
        row; see talea.long_protein. ``options`` go to ``iter_predictions``.
        """

        return iter_windowed_predictions(
            self,
            records,
            window_length=window_length,
            window_stride=window_stride,
            **options,
        )

    def iter_predictions(
        self,
        records: Iterable[SequenceRecord],
        *,
        square_budget: int = 4_000_000,
        maximum_batch: int = 64,
        postprocess_workers: int | None = None,
        progress: bool = False,
        capture_mode: str = "auto",
        cache_clear_interval: int = 0,
        record_workers: int = 1,
        readout_streams: int = 0,
        teacher_owner_thread: bool = False,
        teacher_readback_records: int = 1,
        async_teacher_readback: bool = False,
        readout_square_budget: int = 1024 ** 2,
        teacher_quantile_histogram: bool = False,
        native_half_readout: bool = False,
        device_teacher_tiles: bool = False,
        worker_finalization: bool = False,
        prefetch_forward: bool = False,
        fused_rotary: bool = False,
        fused_swiglu: bool = False,
        unit_residual: bool = False,
        early_apc: bool = False,
        early_gate: str = "off",
        skip_all_valid_mask: bool = False,
        arithmetic_swiglu: bool = False,
        cache_sdpa_bias: bool = False,
        group_equal_apc: bool = False,
        fused_apc: str = "off",
        batched_apc_reductions: str = "",
        teacher_cuda_graph: bool = False,
        teacher_graph_tracks: bool = False,
        teacher_graph_pool: bool = False,
        shared_attention_mib: int = 0,
        postprocess_mode: str = "process",
    ) -> Iterator[dict]:
        """Yield predictions in length-sorted order without retaining matrices.

        ``postprocess_mode="process"`` (default) runs postprocessing in a
        worker pool, sidestepping the GIL; ``postprocess_workers=None`` then
        uses every available CPU. Workers come from a forkserver, so they never
        inherit the parent's CUDA context; the cost is a one-off worker start
        plus pickling each record's matrices (8-16 MB at maximum length,
        against seconds of compute). Output is identical to thread mode.
        The first call starts the pool and later calls reuse it, so calls that
        each score one protein pay the start once; ``close()`` (or using the
        predictor as a context manager) stops it.
        Workers re-import the calling script (Python's spawn/forkserver rule),
        so scripts must call Talea under ``if __name__ == "__main__":``.

        With record_workers > 1, threads prepare independent proteins using
        their original teacher batch shapes. Results are collected in order.

        With teacher_owner_thread enabled, record workers normalize tiles ahead
        of the ordered teacher calls on the thread that warmed the network.

        readout_square_budget bounds the concurrent per-record square matrices by
        limiting readout workers and grouped probability transfers for long inputs.

        With prefetch_forward enabled, one ESM forward can run ahead of the
        current batch's readouts when accelerator memory permits. On CUDA these
        use separate streams. MPS overlaps its GPU queue with host preparation.

        shared_attention_mib reserves this much shared process memory for APC
        inputs. When a record cannot fit, it uses the normal process queue.
        The default of zero sends all matrices through the queue.

        Batches are pipelined: a batch's rows are yielded after the next batch
        has run on the device and been submitted, so postprocessing overlaps
        the forward pass. Up to two batches of matrices wait in the pool.

        A call with a single protein always postprocesses in this process: the
        worker round trip would cost more than the work (~36 ms of pickling and
        IPC per protein on an M4 Max), and the output is identical.

        ``postprocess_mode="thread"`` runs postprocessing in-process and
        defaults to one worker: the work is dominated by many small NumPy
        calls that hold the GIL, so a thread pool convoys rather than
        parallelizes (measured 0.52x at four workers). See
        perf/bench_threads.py.

        ``capture_mode`` selects the attention-capture strategy in
        talea.capture: "sdpa" runs uncaptured layers on the fast attention
        backend; "manual" reproduces the v0.1-v0.3.4 numerics exactly; "auto"
        (default) uses sdpa on CUDA and manual on MPS and CPU, where sdpa's
        memory grows until long runs fail.

        ``cache_clear_interval`` of 0 never calls empty_cache(); calling it per
        batch returns blocks to the driver and forces later reallocation.
        """

        batched = frozenset(batched_apc_reductions.split(",")) if batched_apc_reductions else frozenset()
        if not batched <= {"row", "column", "total", "head"}:
            raise ValueError("batched_apc_reductions accepts row,column,total,head")
        apc_fusion = _apc_fusion(self.device, self.dtype, fused_apc)
        self.apc_fusion_active = apc_fusion is not None
        self.apc_group_stats = None
        if group_equal_apc:
            self.apc_group_stats = {
                "cuda_active": self.device.type == "cuda",
                "batches": 0, "model_records": 0, "eligible_groups": 0, "eligible_records": 0,
                "executed_groups": 0, "executed_records": 0, "squared_lengths": 0,
                "eligible_squared_lengths": 0, "group_size_counts": {},
                "element_budget": _APC_GROUP_ELEMENTS,
            }
        if teacher_graph_pool and self.device.type != "cuda":
            raise ValueError("a shared teacher graph pool requires CUDA")
        teacher_cuda_graph = teacher_cuda_graph or teacher_graph_tracks or teacher_graph_pool
        if early_gate not in ("off", "record", "group"):
            raise ValueError("early_gate must be off, record or group")
        early_readout = early_apc or early_gate != "off"
        if early_readout and (self.device.type != "cuda" or _device_autocast_enabled("cuda")):
            raise ValueError("early layer readouts require CUDA and an explicit model dtype")
        if teacher_readback_records < 1:
            raise ValueError("teacher_readback_records must be positive")
        if readout_square_budget < 1:
            raise ValueError("readout_square_budget must be positive")
        if async_teacher_readback and self.device.type != "cuda":
            raise ValueError("asynchronous teacher readback requires CUDA")
        teacher_owner_thread = teacher_owner_thread or teacher_readback_records > 1 or async_teacher_readback or teacher_graph_pool
        self.async_teacher_readback_batches = 0
        if shared_attention_mib < 0:
            raise ValueError("shared_attention_mib must be nonnegative")
        if not 0 <= readout_streams <= record_workers:
            raise ValueError("readout_streams must be between zero and record_workers")
        if readout_streams:
            if self.device.type != "cuda" or _device_autocast_enabled("cuda"):
                raise ValueError("independent readout streams require CUDA with an explicit model dtype")
            teacher_owner_thread = True
        if record_workers < 1:
            raise ValueError("record_workers must be positive")
        if prefetch_forward and self.device.type not in ("cuda", "mps"):
            raise ValueError("forward prefetch requires a CUDA or MPS device")
        if prefetch_forward and _device_autocast_enabled(self.device.type):
            raise ValueError("forward prefetch requires an explicit model dtype with autocast disabled")
        values = list(records)
        if not values:
            return
        self._validate_records(values)
        if capture_mode == "auto":
            # sdpa is faster and fine on CUDA; on Apple MPS its kernels grow the
            # driver's allocation over many long, variable-length batches until
            # the process runs out of memory, so MPS and CPU keep manual capture.
            capture_mode = "sdpa" if self.device.type == "cuda" else "manual"
        if postprocess_mode == "process" and len(values) == 1:
            # One protein: the worker round trip (pickling its matrices) costs
            # more than the postprocessing itself, so run it in this process.
            # Output is identical in both modes.
            postprocess_mode = "thread"
            postprocess_workers = None
        postprocess_workers = resolve_postprocess_workers(
            postprocess_workers, postprocess_mode
        )
        ordered = sorted(values, key=lambda value: (value.length, value.record_id))
        batches = list(make_batches(ordered, square_budget, maximum_batch))
        completed = 0
        window_sizes = (64, 96, 128, 192, 256, 384, 512)
        storage_dtype = self.deployment["esmplusplus"].get(
            "decoder_matrix_storage_dtype", "fp32"
        )
        if storage_dtype not in ("fp16", "fp32"):
            raise ValueError(f"unsupported attention storage dtype: {storage_dtype}")
        device_storage_dtype = torch.float16 if storage_dtype == "fp16" else None
        quantile_function = compiled_teacher_quantile() if teacher_quantile_histogram and storage_dtype == "fp16" else None
        half_decoder = None
        if (native_half_readout or device_teacher_tiles) and self.device.type == "cuda" and storage_dtype == "fp16":
            if compiled_teacher_quantile() is not None:
                from talea._teacher_histogram import native_half_tiles
                half_decoder = native_half_tiles
                teacher_owner_thread = True
        self.native_half_readout_active = half_decoder is not None
        self.device_teacher_tile_records = 0
        device_tile_plan = None
        contract_attention = self.teacher_contract["attention"]
        if device_teacher_tiles and half_decoder is not None and len(self.expected_layers) == 2:
            try:
                from talea._device_teacher_tiles import DeviceTeacherTiles as device_tile_plan
                from talea._teacher_histogram import native_half_tables
            except ImportError:
                pass
        self.teacher_quantile_histogram_active = quantile_function is not None
        minimum_separation = int(
            self.deployment["esmplusplus"]["minimum_sequence_separation"]
        )
        if postprocess_mode == "process":
            pool_context = contextlib.nullcontext(
                self._process_pool(postprocess_workers, shared_attention_mib)
            )
        else:
            pool_context = concurrent.futures.ThreadPoolExecutor(
                max_workers=postprocess_workers
            )
        worker_finalization = worker_finalization and postprocess_mode == "process"
        if worker_finalization:
            finalization_settings = (
                self.primary_extent, self.region_probe_settings,
                {"teacher": {"class_names": self.teacher_contract["teacher"]["class_names"]},
                 "output": {"state_threshold": self.teacher_contract["output"]["state_threshold"]}},
            )
        reference_maximum = int(
            self.deployment["input_contract"]["lockbox_reference_maximum_residues"]
        )

        def collect(batch_index, compact, global_probabilities):
            """Yield one batch's rows, waiting for its postprocessing."""

            nonlocal completed
            for (
                record,
                future,
                state_track,
                teacher_track,
                region_track,
                gate_logit,
            ), global_probability in zip(compact, global_probabilities, strict=True):
                if worker_finalization:
                    topology, regions, teacher_summary = postprocess_result(future)
                else:
                    teacher_summary = summarize_teacher_prediction(
                        state_track, teacher_track, self.teacher_contract,
                    )
                    topology, regions = finish_regions(
                        postprocess_result(future), state_track, region_track,
                        self.primary_extent, self.region_probe_settings,
                    )
                scores = compose_talea_score(
                    teacher_summary["mean_probability"],
                    topology["topology_development_control_z"],
                    float(global_probability),
                    self.deployment["score_contract"],
                )
                # Open-versus-closed gate: demote proteins the probe calls closed.
                penalty = self.openness_penalty_weight * min(0.0, gate_logit)
                scores = {
                    **scores,
                    "talea_discovery_score_ungated": scores["talea_discovery_score"],
                    "talea_discovery_score": scores["talea_discovery_score"] + penalty,
                    "openness_probability": float(1.0 / (1.0 + np.exp(-gate_logit))),
                    "openness_penalty": penalty,
                }
                above_reference = bool(record.length > reference_maximum)
                yield {
                    **topology,
                    **scores,
                    "repeat_regions": regions,
                    "primary_extent": self.primary_extent,
                    "primary_score_field": "talea_discovery_score",
                    "teacher_state_summary": {
                        key: value
                        for key, value in teacher_summary.items()
                        if key != "predicted_teacher_mean_class_probability"
                    },
                    "predicted_teacher_mean_class_probability": teacher_summary[
                        "predicted_teacher_mean_class_probability"
                    ],
                    "length_scope": {
                        "lockbox_reference_maximum_residues": reference_maximum,
                        "above_lockbox_reference_length": above_reference,
                    },
                    "above_lockbox_reference_length": above_reference,
                    "evidence_status": self.deployment["evidence_status"],
                    "interpretation": self.deployment["reporting_contract"][
                        "interpretation"
                    ],
                }
                completed += 1
            if progress:
                print(
                    f"Talea: batch {batch_index}/{len(batches)}; "
                    f"proteins {completed}/{len(ordered)}",
                    file=sys.stderr,
                    flush=True,
                )

        pending = None
        compact = []

        def cancel_outstanding():
            # A call stopped early leaves no queued work in the kept pool.
            for outstanding in (compact, pending[1] if pending else ()):
                for _, future, *_ in outstanding:
                    future.cancel()

        attention_staging = _AttentionStaging() if self.device.type == "cuda" else None
        with contextlib.ExitStack() as stack:
            pool = stack.enter_context(pool_context)
            submit_record = pool.submit
            if postprocess_mode == "process" and shared_attention_mib:
                submit_record = partial(self._attention_arena.submit, pool)
            overlap_enabled = prefetch_forward and len(batches) > 1
            forward_stream = readout_stream = None
            graph_prefetch = (
                prefetch_forward and teacher_cuda_graph and self.device.type == "cuda"
            )
            if overlap_enabled or graph_prefetch:
                # Also use the same stream for single-batch warm-up so its
                # static graph inputs and outputs survive into the timed call.
                if self.device.type == "cuda":
                    forward_stream = torch.cuda.current_stream(self.device)
                    if graph_prefetch:
                        if self._graph_readout_stream is None:
                            self._graph_readout_stream = torch.cuda.Stream(device=self.device)
                        readout_stream = self._graph_readout_stream
                    else:
                        readout_stream = torch.cuda.Stream(device=self.device)
                    readout_stream.wait_stream(forward_stream)
                    stack.callback(readout_stream.synchronize)
                    stack.callback(forward_stream.synchronize)
                else:
                    stack.callback(torch.mps.synchronize)
            # Build the shared diagonal on its consuming stream before
            # starting record workers, and warm any teacher graphs there.
            with torch.cuda.stream(readout_stream) if readout_stream is not None else contextlib.nullcontext():
                teacher_diagonal = prepare_teacher_readout(
                    self.teacher_model, self.teacher_contract, self.device
                )
                teacher_graphs = self._get_teacher_graphs(
                    teacher_cuda_graph, replace_stream=graph_prefetch, owner_thread=teacher_owner_thread,
                    pack_tracks=teacher_graph_tracks, shared_pool=teacher_graph_pool,
                )
            private_streams = [torch.cuda.Stream(device=self.device) for _ in range(readout_streams)]
            worker_local = threading.local()
            if private_streams:
                if readout_stream is None:
                    readout_stream = torch.cuda.current_stream(self.device)
                available_streams = cycle(private_streams)

                def bind_private_stream():
                    worker_local.stream = next(available_streams)
                    torch.cuda.set_stream(worker_local.stream)

                for private in private_streams:
                    stack.callback(private.synchronize)
            record_pool = None
            if record_workers > 1 and not _device_autocast_enabled(self.device.type):
                # Bind every worker to the graph's readout stream, including
                # when prefetch moves the encoder onto a separate stream.
                stream = _record_worker_stream(self.device, readout_stream)
                record_pool = stack.enter_context(concurrent.futures.ThreadPoolExecutor(
                    max_workers=record_workers,
                    initializer=bind_private_stream if private_streams else (torch.cuda.set_stream if stream is not None else None),
                    initargs=(stream,) if stream is not None and not private_streams else (),
                ))
            stack.callback(cancel_outstanding)
            stack.enter_context(torch.inference_mode())
            capture = stack.enter_context(
                selective_capture(
                    self.esm_model,
                    self.expected_layers,
                    tuple(sorted(set(self.expected_layers) | set(self.openness_layers))),
                    mode=capture_mode,
                )
            )
            self.unit_residual_blocks = 0
            if unit_residual:
                from talea.capture import unit_residual_context
                self.unit_residual_blocks = stack.enter_context(unit_residual_context(capture.blocks))
            self.cached_sdpa_bias_calls = 0
            if cache_sdpa_bias and self.device.type == "cuda" and capture_mode == "sdpa":
                from talea.sdpa_bias import cached_padding_bias
                bias_cache = stack.enter_context(cached_padding_bias(capture.blocks))
                stack.callback(lambda: setattr(self, "cached_sdpa_bias_calls", bias_cache.calls))
            self.fused_rotary_calls = 0
            if fused_rotary:
                from talea.rotary import fused_rotary_context
                rotary_calls = stack.enter_context(fused_rotary_context(capture.blocks, self.device))
                stack.callback(lambda: setattr(self, "fused_rotary_calls", rotary_calls[0]))
            self.fused_swiglu_calls = 0
            self.arithmetic_swiglu_repairs = None
            if fused_swiglu or arithmetic_swiglu:
                from talea.swiglu import swiglu_context
                swiglu_calls = stack.enter_context(swiglu_context(capture.blocks, self.device, arithmetic=arithmetic_swiglu))
                self.arithmetic_swiglu_repairs = swiglu_calls[1]
                stack.callback(lambda: setattr(self, "fused_swiglu_calls", swiglu_calls[0]))
            self.early_apc_layers = self.early_gate_layers = 0
            if early_readout:
                capture_stream = torch.cuda.Stream(device=self.device)
                stack.callback(capture_stream.synchronize)
                capture_executor = stack.enter_context(concurrent.futures.ThreadPoolExecutor(max_workers=1))
            producer = None
            if overlap_enabled:
                # Join the producer before restoring the ESM hooks.
                producer = stack.enter_context(concurrent.futures.ThreadPoolExecutor(
                    max_workers=1,
                    initializer=torch.cuda.set_stream if forward_stream is not None else None,
                    initargs=(forward_stream,) if forward_stream is not None else (),
                ))

            @torch.inference_mode()
            def forward(batch):
                tokenized = self.tokenizer(
                    [record.sequence for record in batch],
                    return_tensors=None,
                    padding=True,
                    return_special_tokens_mask=True,
                )
                # The fast tokenizer already pads rectangular lists. Construct
                # tensors directly to avoid Transformers flattening every list
                # in Python before handing it to torch.
                tokenized = {k: torch.from_numpy(np.asarray(v)) for k, v in tokenized.items()}
                residue_indices = _batch_residue_indices(tokenized, batch, self.device)
                model_inputs = {
                    key: value.to(self.device)
                    for key, value in tokenized.items()
                    if key != "special_tokens_mask" and isinstance(value, torch.Tensor)
                }
                if skip_all_valid_mask:
                    # Evaluate the tokenizer mask on CPU, before the ESM forward.
                    # Only a padding-only mask can be simplified this way.
                    padding_mask = tokenized.get("attention_mask")
                    capture.all_tokens_valid = (
                        "sequence_id" not in model_inputs
                        and padding_mask is not None
                        and bool(padding_mask.all())
                    )
                capture.reset()
                if early_readout:
                    for indices, _ in residue_indices:
                        indices.record_stream(capture_stream)
                if early_apc:
                    correct = partial(_layer_apc, indices_by_record=residue_indices, separation=minimum_separation,
                                      dtype=device_storage_dtype, grouped=group_equal_apc, fusion=apc_fusion, batched=batched)
                    capture.transform_attention = partial(
                        _queue_capture, function=correct, executor=capture_executor, stream=capture_stream,
                    )
                if early_gate != "off":
                    reduce_gate = partial(_pool_gate_layer, indices_by_record=residue_indices, group_equal=early_gate == "group")

                    def early_hidden(layer, value):
                        if layer in self.openness_layers and layer not in self.expected_layers:
                            return _queue_capture(value, reduce_gate, capture_executor, capture_stream)
                        return value
                    capture.transform_hidden = early_hidden
                try:
                    model_output = self.esm_model(
                        **model_inputs,
                        compute_logits=False,
                        output_attentions=False,
                        output_hidden_states=False,
                        return_dict=True,
                    )
                    capture.finalize(model_output.last_hidden_state)
                    attentions, hidden_states = capture.take()
                except BaseException:
                    capture.reset()
                    raise
                finally:
                    capture.transform_attention = capture.transform_hidden = None
                ready = None
                if readout_stream is not None:
                    ready = torch.cuda.current_stream(self.device).record_event()
                return residue_indices, attentions, hidden_states, ready

            def can_overlap(current, following, captured):
                _, attention, hidden, _ = captured
                width = min(len(current), record_workers, max(1, readout_square_budget // current[-1].length ** 2))
                required = _prefetch_memory_required(
                    current, following, next(iter(attention.values())), next(iter(hidden.values())), width,
                )
                if group_equal_apc and self.device.type == "cuda":
                    # Grouped correction temporaries can outgrow scalar scratch
                    # for a single record. Include six group-sized fp32 buffers.
                    required += 6 * 4 * _APC_GROUP_ELEMENTS
                if early_apc:
                    # The following capture can be reduced while this batch
                    # still owns encoder outputs and readout scratch.
                    heads = next(iter(attention.values())).shape[1]
                    required += 6 * 4 * max(_APC_GROUP_ELEMENTS, heads * following[-1].length**2)
                return required <= _prefetch_headroom(self.device)

            next_forward = None
            prefetched = 0

            def cancel_forward():
                if next_forward is not None:
                    next_forward.cancel()

            stack.callback(cancel_forward)
            for batch_index, batch in enumerate(batches, 1):
                following = batches[batch_index] if batch_index < len(batches) else None
                if cache_clear_interval and batch_index % cache_clear_interval == 0:
                    following = None
                captured, next_forward = _forward_pair(
                    batch, following, next_forward, forward, producer, can_overlap
                )
                if next_forward is not None:
                    prefetched += 1
                residue_indices, attentions, hidden_states, ready = captured
                del captured
                if early_readout:
                    consumer = readout_stream if readout_stream is not None else torch.cuda.current_stream(self.device)
                if early_gate != "off":
                    for layer, value in hidden_states.items():
                        if isinstance(value, _PendingCapture):
                            hidden_states[layer] = _collect_capture(value, consumer)
                            self.early_gate_layers += 1
                if ready is not None:
                    _handoff_capture(readout_stream, ready, {} if early_apc else attentions, hidden_states, residue_indices)
                if early_apc:
                    layer_maps = {layer: _collect_capture(item, consumer) for layer, item in attentions.items()}
                    self.early_apc_layers += len(layer_maps)
                if private_streams:
                    # Model capture and grouped operations belong to the owner stream.
                    # Publish the immutable separation-mask blocks there as well.
                    with torch.cuda.stream(readout_stream):
                        device = next(iter(hidden_states.values())).device
                        capacities = {1 << (record.length - 1).bit_length() for record in batch}
                        masks = [_separation_mask_block(cap, minimum_separation, device) for cap in capacities] if minimum_separation else []
                        available = readout_stream.record_event()
                    for private in private_streams:
                        _handoff_capture(private, available, {} if early_apc else attentions, hidden_states, residue_indices)
                        for mask in masks:
                            mask.record_stream(private)
                        if early_apc:
                            for maps in layer_maps.values():
                                for tensor in maps:
                                    tensor.record_stream(private)
                compact = []
                pooled_rows = []

                def finish_teacher(inputs):
                    _, matrices, _, _, _ = inputs
                    tracks = teacher_tracks_from_attention(
                        self.teacher_model, matrices, self.teacher_contract, self.device,
                        cuda_graphs=teacher_graphs, _prepared_diagonal=teacher_diagonal,
                        _quantile_function=quantile_function,
                    )
                    return (*inputs, *tracks)

                @torch.inference_mode()
                def readout(batch_position, grouped=None):
                    record = batch[batch_position]
                    indices, residue_slice = residue_indices[batch_position]
                    matrices = {}
                    device_maps = [] if device_tile_plan is not None else None
                    for layer in self.expected_layers:
                        if early_apc:
                            tensor = layer_maps[layer][batch_position]
                        elif grouped is None:
                            tensor = _apc_head_mean_tensor(
                                attentions[layer][batch_position], indices, record.length,
                                minimum_separation, device_storage_dtype, residue_slice, apc_fusion,
                            )
                        else:
                            tensor = grouped[layer]
                        if device_maps is not None:
                            device_maps.append(tensor)
                        if attention_staging is None:
                            host = tensor.to("cpu", dtype=torch.float32).numpy()
                            matrices[layer] = _checked_attention_matrix(host, record.length)
                        else:
                            attention_staging.enqueue(layer, tensor)
                    pooled, probabilities, region_track, gate_logit = self._hidden_readouts(
                        hidden_states, indices, batch_position, residue_slice
                    )
                    if attention_staging is not None:
                        # _hidden_readouts' blocking D2H copy is on the same CUDA
                        # stream as both enqueued attention copies.
                        matrices = attention_staging.unpack_after_readouts(record.length, native_half=half_decoder is not None)
                    inputs = pooled, matrices, probabilities, region_track, gate_logit
                    if device_maps is not None and all(t.dtype == torch.float16 and t.is_contiguous() for t in device_maps):
                        tables = native_half_tables(
                            tuple(matrices[layer] for layer in self.expected_layers),
                            float(contract_attention["normalization_quantile"]),
                            float(contract_attention["normalization_clip"]),
                        )
                        tiles = device_tile_plan(tuple(device_maps), tables, int(contract_attention["tile_size"]),
                                                 int(contract_attention["central_size"]))
                        return inputs, (record.length, tiles)
                    if teacher_owner_thread:
                        return inputs, prepare_teacher_tiles(matrices, self.teacher_contract, quantile_function=quantile_function, half_decoder=half_decoder)
                    return finish_teacher(inputs)

                # Bound concurrent quadratic APC scratch as records get longer.
                readout_capacity = max(1, readout_square_budget // batch[-1].length ** 2)
                width = min(record_workers, readout_capacity)

                def stream_readout(position, grouped=None, group_ready=None):
                    stream = getattr(worker_local, "stream", readout_stream) if private_streams else readout_stream
                    with torch.cuda.stream(stream):
                        if group_ready is not None:
                            stream.wait_event(group_ready)
                            for tensor in grouped.values():
                                tensor.record_stream(stream)
                        return readout(position, grouped)

                read = readout if readout_stream is None else stream_readout
                if group_equal_apc:
                    heads = attentions[self.expected_layers[0]].shape[1]
                    groups = list(_equal_apc_groups(residue_indices, heads))
                    stats = self.apc_group_stats
                    stats["batches"] += 1
                    stats["model_records"] += len(batch)
                    for start, stop in groups:
                        size = stop - start
                        squares = size * batch[start].length ** 2
                        hist = stats["group_size_counts"]
                        hist[size] = hist.get(size, 0) + 1
                        stats["squared_lengths"] += squares
                        if size > 1:
                            stats["eligible_groups"] += 1
                            stats["eligible_records"] += size
                            stats["eligible_squared_lengths"] += squares
                if early_apc and group_equal_apc:
                    for start, stop in groups:
                        if stop - start > 1:
                            stats["executed_groups"] += 1
                            stats["executed_records"] += stop - start
                if group_equal_apc and self.device.type == "cuda" and not early_apc:
                    items = _grouped_apc_jobs(
                        attentions, self.expected_layers, residue_indices, minimum_separation,
                        device_storage_dtype, readout_stream, groups, stats, apc_fusion, batched, handoff=bool(private_streams),
                    )
                    readouts = _ordered_readouts(lambda args: read(*args), len(batch), record_pool, width, items)
                else:
                    readouts = _ordered_readouts(read, len(batch), record_pool, width)
                if teacher_owner_thread:
                    # Submit teacher calls in stock order on the graph-capture thread.
                    # The same quadratic bound used for record workers limits retained maps.
                    depth = min(teacher_readback_records, readout_capacity)
                    def launch_teacher(tiles):
                        if device_tile_plan is not None and isinstance(tiles[1], device_tile_plan):
                            self.device_teacher_tile_records += 1
                        return teacher_device_probabilities(
                            self.teacher_model, tiles, self.teacher_contract, self.device,
                            teacher_graphs, teacher_diagonal,
                        )
                    # Both resident groups must fit the original matrix budget.
                    overlap_copy = async_teacher_readback and 2 * depth <= readout_capacity and len(batch) > depth
                    self.async_teacher_readback_batches += int(overlap_copy)
                    readouts = _ordered_teacher_readbacks(
                        readouts, launch_teacher, readout_stream, depth, asynchronous=overlap_copy,
                    )
                for record, outputs in zip(batch, readouts, strict=True):
                    pooled, matrices, probabilities, region_track, gate_logit, state_track, teacher_track = outputs
                    pooled_rows.append(pooled)
                    if postprocess_mode == "process":
                        payload = (
                            record.metadata(), matrices, window_sizes, probabilities, state_track,
                        )
                        if worker_finalization:
                            payload += ((teacher_track, region_track, finalization_settings),)
                            future = submit_record(run_finalized_record, payload)
                            # Workers own the remaining region and teacher reductions.
                            state_track = teacher_track = region_track = None
                        else:
                            future = submit_record(run_record, payload)
                    else:
                        future = pool.submit(
                            process_attention_record,
                            {
                                "metadata": record.metadata(),
                                "matrices": matrices,
                                "window_sizes": window_sizes,
                                "residue_structure_probabilities": probabilities,
                                "residue_state_probabilities": state_track,
                                **self.configs,
                            },
                        )
                    compact.append(
                        (record, future, state_track, teacher_track, region_track, gate_logit)
                    )
                global_probabilities = global_head_probabilities(
                    self.global_bundle, pooled_rows
                )
                if early_apc:
                    del layer_maps
                del residue_indices, attentions, hidden_states, pooled_rows
                # empty_cache() returns blocks to the driver and forces later
                # cudaMalloc; gc.collect() walks every generation. Both were
                # unconditional per batch. Interval 0 disables them entirely,
                # which is correct once the 36-layer attention tuple is gone.
                if cache_clear_interval and batch_index % cache_clear_interval == 0:
                    gc.collect()
                    if self.device.type == "mps":
                        torch.mps.empty_cache()
                    elif self.device.type == "cuda":
                        torch.cuda.empty_cache()
                # The previous batch is collected only now, so its
                # postprocessing ran in the pool during this batch's forward
                # pass and submissions.
                if pending is not None:
                    yield from collect(*pending)
                pending = (batch_index, compact, global_probabilities)
            if prefetch_forward and progress:
                print(f"Talea: prefetched {prefetched}/{len(batches) - 1} batch boundaries",
                      file=sys.stderr, flush=True)
            if pending is not None:
                yield from collect(*pending)
