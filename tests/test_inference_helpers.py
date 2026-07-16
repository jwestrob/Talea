import hashlib

import numpy as np
import pytest
import torch

from talea.inference import apc_head_mean, make_batches, resolve_dtype
from talea.io import SequenceRecord
from talea.model import AttentionStateTeacherUNet


def record(record_id: str, length: int) -> SequenceRecord:
    sequence = "A" * length
    return SequenceRecord(
        record_id=record_id,
        sequence=sequence,
        sequence_sha256=hashlib.sha256(sequence.encode()).hexdigest(),
    )


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


def test_teacher_runtime_model_output_shape() -> None:
    model = AttentionStateTeacherUNet(
        input_channels=3,
        teacher_classes=5,
        base_channels=4,
    )
    observed = model(torch.zeros(2, 3, 32, 32))
    assert observed.shape == (2, 6, 32, 32)
