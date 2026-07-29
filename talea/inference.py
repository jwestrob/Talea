"""End-to-end sequence-only Talea inference."""

from __future__ import annotations

import concurrent.futures
import gc
import hashlib
import json
import multiprocessing
import os
import sys
import time
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
from talea.model import AttentionStateTeacherUNet
from talea.postprocess_pool import initialize_worker, run_record
from talea.production import (
    compose_talea_score,
    global_head_probabilities,
    pooled_hidden_features_for_layers,
    pooled_hidden_features_for_record,
    roundtrip_attention_storage,
    summarize_teacher_prediction,
    teacher_tracks_from_attention,
)
from talea.topology import process_attention_record


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


_SEPARATION_MASK_CACHE: dict[tuple[int, int, str], torch.Tensor] = {}


def near_diagonal_mask(
    sequence_length: int, minimum_separation: int, device: torch.device
) -> torch.Tensor:
    """Cached boolean mask selecting residue pairs closer than the minimum.

    Rebuilding this per call allocated two int64 ``L x L`` temporaries -- 64 MB
    at L=2046 -- on every layer of every record. The mask depends only on
    (length, separation, device), so it is built once and reused.
    """
    key = (int(sequence_length), int(minimum_separation), str(device))
    cached = _SEPARATION_MASK_CACHE.get(key)
    if cached is None:
        positions = torch.arange(sequence_length, device=device)
        cached = (positions[:, None] - positions[None, :]).abs() < minimum_separation
        _SEPARATION_MASK_CACHE[key] = cached
    return cached


def apc_head_mean(
    layer_attention: torch.Tensor,
    indices: torch.Tensor,
    sequence_length: int,
    minimum_separation: int,
    storage_dtype: torch.dtype | None = None,
) -> np.ndarray:
    """Symmetrize, APC-correct, head-average, and crop one attention layer.

    The crop precedes the symmetrization: both operations are elementwise over
    the same index set, so the result is unchanged, but every subsequent
    reduction then runs over ``L x L`` rather than the padded ``T x T``.

    ``storage_dtype`` casts on the accelerator before the host transfer. The
    frozen contract stores these matrices as fp16, so casting first halves the
    transfer without changing the stored representation.
    """

    attention = layer_attention.index_select(1, indices).index_select(2, indices)
    attention = 0.5 * (attention + attention.transpose(-1, -2))
    row = attention.sum(dim=-1, keepdim=True)
    column = attention.sum(dim=-2, keepdim=True)
    total = attention.sum(dim=(-1, -2), keepdim=True) + 1e-12
    attention = torch.relu(attention - (row * column) / total).mean(dim=0)
    if minimum_separation > 0:
        attention = attention.masked_fill(
            near_diagonal_mask(sequence_length, minimum_separation, attention.device),
            0.0,
        )
    attention.fill_diagonal_(0.0)
    if storage_dtype is not None and attention.dtype != storage_dtype:
        attention = attention.to(storage_dtype)
    result = attention.to("cpu", dtype=torch.float32).numpy()
    if result.shape != (sequence_length, sequence_length):
        raise ValueError("cropped attention matrix does not match sequence length")
    return result


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
    ):
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
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
        self.global_bundle = joblib.load(self.assets.global_model)
        _validate_global_bundle(self.global_bundle, self.deployment)
        self.loaded_at = time.time()
        self._load_models()
        self.loaded_seconds = time.time() - self.loaded_at

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

    def iter_predictions(
        self,
        records: Iterable[SequenceRecord],
        *,
        square_budget: int = 4_000_000,
        maximum_batch: int = 64,
        postprocess_workers: int = 1,
        progress: bool = False,
        capture_mode: str = "manual",
        cache_clear_interval: int = 0,
        postprocess_mode: str = "thread",
    ) -> Iterator[dict]:
        """Yield predictions in length-sorted order without retaining matrices.

        ``postprocess_workers`` defaults to 1 because the default
        ``postprocess_mode="thread"`` cannot scale: the postprocessing is
        dominated by many small NumPy calls that hold the GIL, so a thread pool
        convoys rather than parallelizes (measured 0.52x at four workers). See
        perf/bench_threads.py.

        ``postprocess_mode="process"`` sidesteps the GIL entirely and is the
        way to use more than one core. It is spawn-based, so workers never
        inherit the parent's CUDA context; the cost is a one-off interpreter
        start per worker plus pickling each record's matrices (8-16 MB at
        maximum length, against seconds of compute).

        ``capture_mode`` selects the attention-capture strategy in
        talea.capture; "manual" preserves the frozen numerics exactly.

        ``cache_clear_interval`` of 0 never calls empty_cache(); calling it per
        batch returns blocks to the driver and forces later reallocation.
        """

        values = list(records)
        if not values:
            return
        self._validate_records(values)
        if postprocess_workers < 1:
            raise ValueError("postprocess worker count must be positive")
        if postprocess_mode not in POSTPROCESS_MODES:
            raise ValueError(f"unsupported postprocess mode: {postprocess_mode}")
        ordered = sorted(values, key=lambda value: (value.length, value.record_id))
        batches = list(make_batches(ordered, square_budget, maximum_batch))
        completed = 0
        window_sizes = (64, 96, 128, 192, 256, 384, 512)
        storage_dtype = self.deployment["esmplusplus"].get(
            "decoder_matrix_storage_dtype", "fp32"
        )
        device_storage_dtype = {
            "fp16": torch.float16,
            "fp32": None,
        }.get(storage_dtype)
        minimum_separation = int(
            self.deployment["esmplusplus"]["minimum_sequence_separation"]
        )
        if postprocess_mode == "process":
            pool_context = concurrent.futures.ProcessPoolExecutor(
                max_workers=postprocess_workers,
                mp_context=postprocess_pool_context(),
                initializer=initialize_worker,
                initargs=(self.configs,),
            )
        else:
            pool_context = concurrent.futures.ThreadPoolExecutor(
                max_workers=postprocess_workers
            )
        with pool_context as pool, torch.inference_mode(), selective_capture(
            self.esm_model,
            self.expected_layers,
            self.expected_layers,
            mode=capture_mode,
        ) as capture:
            for batch_index, batch in enumerate(batches, start=1):
                tokenized = self.tokenizer(
                    [record.sequence for record in batch],
                    return_tensors="pt",
                    padding=True,
                    return_special_tokens_mask=True,
                )
                tokenized = {
                    key: value.to(self.device)
                    for key, value in tokenized.items()
                    if isinstance(value, torch.Tensor)
                }
                special = tokenized["special_tokens_mask"]
                attention_mask = tokenized.get("attention_mask")
                model_inputs = {
                    key: value
                    for key, value in tokenized.items()
                    if key != "special_tokens_mask"
                }
                capture.reset()
                model_output = self.esm_model(
                    **model_inputs,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True,
                )
                capture.finalize(model_output.last_hidden_state)
                capture.require()
                attentions = capture.attentions
                hidden_states = capture.hidden_states
                compact = []
                pooled_rows = []
                for batch_position, record in enumerate(batch):
                    residue_mask = special[batch_position] == 0
                    if attention_mask is not None:
                        residue_mask &= attention_mask[batch_position].bool()
                    indices = torch.nonzero(
                        residue_mask, as_tuple=False
                    ).squeeze(-1)
                    if len(indices) != record.length:
                        raise ValueError(
                            "tokenized residue count disagrees for "
                            f"{record.record_id}"
                        )
                    matrices = {
                        layer: roundtrip_attention_storage(
                            apc_head_mean(
                                attentions[layer][batch_position],
                                indices,
                                record.length,
                                minimum_separation,
                                storage_dtype=device_storage_dtype,
                            ),
                            storage_dtype,
                        )
                        for layer in self.expected_layers
                    }
                    pooled_rows.append(
                        pooled_hidden_features_for_layers(
                            hidden_states,
                            indices,
                            batch_position,
                            self.expected_layers,
                        )
                    )
                    if postprocess_mode == "process":
                        future = pool.submit(
                            run_record,
                            (record.metadata(), matrices, window_sizes),
                        )
                    else:
                        future = pool.submit(
                            process_attention_record,
                            {
                                "metadata": record.metadata(),
                                "matrices": matrices,
                                "window_sizes": window_sizes,
                                **self.configs,
                            },
                        )
                    compact.append((record, matrices, future))
                global_probabilities = global_head_probabilities(
                    self.global_bundle, pooled_rows
                )
                capture.reset()
                del (
                    model_output,
                    tokenized,
                    model_inputs,
                    special,
                    attention_mask,
                    attentions,
                    hidden_states,
                )
                for (record, matrices, future), global_probability in zip(
                    compact, global_probabilities, strict=True
                ):
                    state_track, teacher_track = teacher_tracks_from_attention(
                        self.teacher_model,
                        matrices,
                        self.teacher_contract,
                        self.device,
                    )
                    teacher_summary = summarize_teacher_prediction(
                        state_track,
                        teacher_track,
                        self.teacher_contract,
                    )
                    topology = future.result()
                    scores = compose_talea_score(
                        teacher_summary["mean_probability"],
                        topology["topology_development_control_z"],
                        float(global_probability),
                        self.deployment["score_contract"],
                    )
                    reference_maximum = int(
                        self.deployment["input_contract"][
                            "lockbox_reference_maximum_residues"
                        ]
                    )
                    above_reference = bool(record.length > reference_maximum)
                    yield {
                        **topology,
                        **scores,
                        "primary_score_field": "talea_discovery_score",
                        "teacher_state_summary": {
                            key: value
                            for key, value in teacher_summary.items()
                            if key
                            != "predicted_teacher_mean_class_probability"
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
                del compact, pooled_rows, global_probabilities
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
                if progress:
                    print(
                        f"Talea: batch {batch_index}/{len(batches)}; "
                        f"proteins {completed}/{len(ordered)}",
                        file=sys.stderr,
                        flush=True,
                    )
