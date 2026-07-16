import numpy as np
import pytest

from talea.repeat_state import (
    attention_channels,
    central_tile_spans,
    extract_attention_tile,
    longest_supported_run,
    repeat_architecture_label,
    residue_labels_from_runs,
    robust_attention_channel,
    summarize_state_track,
)


def test_repeat_architecture_label_preserves_mixed_and_named_classes():
    assert repeat_architecture_label([]) == "non_repeat"
    assert repeat_architecture_label(["3"]) == "elongated_repeat"
    assert repeat_architecture_label(["4", "3", "4"]) == "mixed_repeat_architecture"
    assert repeat_architecture_label(["99"]) == "other_repeat_architecture"


def test_residue_labels_use_half_open_runs():
    labels = residue_labels_from_runs(8, [[1, 3, 2], [5, 8, 3]])
    np.testing.assert_array_equal(labels, [0, 1, 1, 0, 0, 1, 1, 1])
    with pytest.raises(ValueError, match="outside length"):
        residue_labels_from_runs(8, [[7, 9, 1]])


def test_robust_attention_channel_is_bounded_and_preserves_zeros():
    matrix = np.array([[0.0, 1.0, 10.0], [1.0, 0.0, 2.0], [10.0, 2.0, 0.0]])
    result = robust_attention_channel(matrix, quantile=0.5, clip=4.0)
    assert result.dtype == np.float32
    assert result.shape == matrix.shape
    assert result.min() == 0.0
    assert result.max() == 1.0
    np.testing.assert_array_equal(result[matrix == 0], 0.0)


def test_attention_channels_require_matching_shapes():
    result = attention_channels([np.eye(3), np.eye(3)])
    assert result.shape == (2, 3, 3)
    with pytest.raises(ValueError, match="shapes disagree"):
        attention_channels([np.eye(3), np.eye(4)])


def test_central_tiles_cover_every_residue_once_and_pad_symmetrically():
    spans = central_tile_spans(130, tile_size=128, central_size=64)
    assert [(row["central_start"], row["central_end"]) for row in spans] == [
        (0, 64),
        (64, 128),
        (128, 130),
    ]
    covered = [
        index
        for span in spans
        for index in range(span["central_start"], span["central_end"])
    ]
    assert covered == list(range(130))

    channels = np.ones((2, 130, 130), dtype=np.float32)
    first = extract_attention_tile(channels, spans[0])
    assert first.shape == (3, 128, 128)
    assert np.all(first[:2, :32, :] == 0)
    assert np.all(first[:2, :, :32] == 0)
    assert np.all(first[2, 32:, 32:] == 1)


def test_longest_run_and_track_summary():
    track = np.array([0.1, 0.8, 0.7, 0.2, 0.6, 0.9, 0.8])
    assert longest_supported_run(track) == (4, 7)
    summary = summarize_state_track(track)
    assert summary["longest_run_start"] == 4
    assert summary["longest_run_end"] == 7
    assert summary["longest_run_length"] == 3
    assert summary["longest_run_fraction"] == pytest.approx(3 / 7)
    assert summary["supported_fraction"] == pytest.approx(5 / 7)
