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


@pytest.mark.parametrize("length,tile,central", [
    (1, 128, 64), (96, 128, 64), (257, 128, 64), (259, 12, 4),
    (430, 31, 27), (600, 31, 31),
])
def test_normalized_diagonal_tiles_match_full_channels(length, tile, central):
    from talea.repeat_state import normalized_attention_tiles

    rng = np.random.default_rng(length)
    matrix = rng.random((length, length), dtype=np.float32)
    matrix[::2, ::7] = np.float32(-0.0)
    layers = [matrix.T, matrix.astype(np.float16).astype(np.float32)]
    spans = central_tile_spans(length, tile, central)
    for quantile, clip in ((0.5, .001), (.95, 8.0), (.999, 100.0)):
        channels = attention_channels(layers, quantile=quantile, clip=clip)
        expected = np.stack([extract_attention_tile(channels, s, tile) for s in spans])
        actual = normalized_attention_tiles(layers, spans, tile, quantile, clip)
        assert actual.tobytes() == expected.tobytes()


def test_matrix_quantile_matches_full_selection_for_unrepresentative_samples():
    from talea.repeat_state import _matrix_quantile

    matrix = np.zeros((512, 512), dtype=np.float32)
    assert _matrix_quantile(matrix, 0.95) == 0.0
    matrix.flat[::256] = 10.0
    matrix.flat[1::3] = 0.25
    for quantile in (0.5, 0.95, 0.9999):
        assert _matrix_quantile(matrix, quantile) == np.quantile(matrix[matrix > 0], quantile)


def test_shared_state_quantiles_and_vectorized_runs_are_bit_exact():
    rng = np.random.default_rng(684)
    for n in (1, 2, 6, 11, 25, 64, 811, 2046):
        cases = (
            rng.random(n),
            rng.choice([-0.0, 0.0, np.nextafter(0., 1.)], n),
            rng.choice([np.nextafter(.5, 0.), .5, np.nextafter(.5, 1.)], n),
        )
        for values in cases:
            for threshold in (0.0, 0.5, 1.0):
                result = summarize_state_track(values, threshold)
                expected = (np.mean(values), np.median(values), np.quantile(values, 0.9), np.mean(values >= threshold))
                observed = tuple(result[k] for k in ("mean_probability", "median_probability", "q90_probability", "supported_fraction"))
                assert np.array(observed).tobytes() == np.array(expected).tobytes()
                supported = values >= threshold
                runs = []
                for i, value in enumerate(supported):
                    if value and (i == 0 or not supported[i - 1]):
                        runs.append([i, i + 1])
                    elif value:
                        runs[-1][1] = i + 1
                best = max(runs, key=lambda v: v[1] - v[0], default=[0, 0])
                assert (result["longest_run_start"], result["longest_run_end"]) == tuple(best)
