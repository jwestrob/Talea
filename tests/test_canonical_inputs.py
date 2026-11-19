import numpy as np
import torch

from talea._canonical_inputs import canonical_batch, canonical_token_table
from talea.io import SequenceRecord


def test_character_lookup_preserves_tensor_layout_and_position_slices():
    table = (bytes.maketrans(b"ACD", bytes([9, 7, 3])), 0, 2, 1)
    records = [SequenceRecord("first", "D", ""), SequenceRecord("second", "ACD", "")]
    inputs, indices = canonical_batch(records, table, "cpu")
    np.testing.assert_array_equal(inputs["input_ids"], [[0, 3, 2, 1, 1], [0, 9, 7, 3, 2]])
    np.testing.assert_array_equal(inputs["attention_mask"], [[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
    assert all(v.dtype == torch.int64 and v.is_contiguous() for v in inputs.values())
    for n, (positions, span) in zip((1, 3), indices):
        assert positions.tolist() == list(range(1, n + 1))
        assert span == slice(1, n + 1)


def test_unknown_tokenizers_keep_the_existing_input_path():
    assert canonical_token_table(object()) is None
