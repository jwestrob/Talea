import numpy as np
import pytest

from talea.repeat_state import central_tile_spans, normalized_attention_tiles

pytest.importorskip("numba")
from talea._teacher_histogram import _half_histogram, native_half_tables, native_half_tiles


@pytest.mark.parametrize("length", [1, 25, 64, 65, 127, 159, 160, 161, 193, 271])
def test_native_teacher_half_tiles_match_full_reference(length):
    rng = np.random.default_rng(203)
    words = rng.integers(0, 0x7c00, size=(length, length), dtype=np.uint16)
    sparse = words.copy()
    sparse.ravel()[::3] = 0
    sparse.ravel()[::11] = 0x8000
    mats = [words.view(np.float16), sparse.view(np.float16)]
    spans = central_tile_spans(length)
    for q in (0.50, 0.95, 0.999):
        actual = native_half_tiles(mats, spans, 128, q, 8.0)
        expected = normalized_attention_tiles([m.astype(np.float32) for m in mats], spans, 128, q, 8.0)
        assert actual.tobytes() == expected.tobytes()
        tables = native_half_tables(mats, q, 8.0)
        restored = np.zeros_like(expected)
        for i, span in enumerate(spans):
            start = span["source_start"]
            src = slice(max(0, start), min(length, start + 128))
            dst = slice(src.start - start, src.stop - start)
            for k, matrix in enumerate(mats):
                ix = matrix[src, src].view(np.uint16)
                restored[i, k, dst, dst] = tables[k, np.where(ix == 0x8000, tables.shape[1] - 1, ix)]
            restored[i, -1, dst, dst] = 1
        assert restored.tobytes() == actual.tobytes()


@pytest.mark.parametrize("word", [0x8001, 0x7c00, 0x7e00, 0xfc00])
def test_bad_half_words_are_rejected(word):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        native_half_tiles([np.array([[word]], np.uint16).view(np.float16)], central_tile_spans(1), 128, .95, 8.0)


@pytest.mark.parametrize("size", [160 * 160 - 1, 160 * 160, 160 * 160 + 1, 2 * 0x7c00])
def test_native_half_histogram_counts_positive_words_and_both_zeros(size):
    words = (np.arange(size, dtype=np.uint32) % 0x7c00).astype(np.uint16)
    words[::7] = 0x8000
    for view in (words, words[::-1]):
        expected = np.bincount(view[(view != 0) & (view != 0x8000)], minlength=0x7c00)
        counts, count = _half_histogram(view)
        np.testing.assert_array_equal(counts, expected)
        assert count == expected.sum()
    words[::2], words[1::2] = 0, 0x8000
    counts, count = _half_histogram(words)
    assert count == 0 and not counts.any()


@pytest.mark.parametrize("word", [0x7c00, 0x7c01, 0x7fff, 0x8001, 0xfbff, 0xfc00, 0xffff])
def test_native_half_histogram_rejects_invalid_large_maps(word):
    words = np.zeros((160, 160), dtype=np.uint16)
    words[-1, -1] = word
    with pytest.raises(ValueError, match="finite and nonnegative"):
        native_half_tables([words.view(np.float16)], .95, 8.0)


@pytest.mark.parametrize("length", [95, 96])
def test_host_tile_crossover_preserves_channels_and_masks(length):
    from talea._teacher_histogram import grid_quantile
    from talea.production import prepare_teacher_tiles

    rng = np.random.default_rng(621)
    words = rng.integers(0, 0x3c00, size=(length, length), dtype=np.uint16)
    words[::3, ::2] = 0x8000
    channels = {13: words.view(np.float16), 35: np.zeros_like(words).view(np.float16)}
    contract = {"attention": {"layers_zero_based": [13, 35], "tile_size": 128, "central_size": 64,
                              "normalization_quantile": .95, "normalization_clip": 8.0}}
    for quantile in (None, grid_quantile):
        _, tiles = prepare_teacher_tiles(channels, contract, quantile_function=quantile, half_decoder=native_half_tiles)
        _, reference = prepare_teacher_tiles(channels, contract)
        assert tiles.tobytes() == reference.tobytes()
