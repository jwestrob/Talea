import numpy as np
import pytest

from talea.repeat_state import central_tile_spans, normalized_attention_tiles

pytest.importorskip("numba")
from talea._teacher_histogram import native_half_tables, native_half_tiles


@pytest.mark.parametrize("length", [1, 25, 64, 65, 127, 193, 271])
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
