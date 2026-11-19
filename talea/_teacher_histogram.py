"""Exact positive-entry quantiles for normalized attention stored on the fp16 grid."""

from functools import lru_cache

import numpy as np
from numba import njit


# Every widened, positive finite fp16 has a float32 exponent between
# 2**-24 and 2**15, and its bottom 13 float32 mantissa bits are zero.
_FIRST = 0x33800000
_LAST = 0x477FE000
_SHIFT = 13
_BINS = ((_LAST - _FIRST) >> _SHIFT) + 1


@njit(cache=True, nogil=True)
def _count_grid(words):
    histogram = np.zeros(_BINS, np.int32)
    count = 0
    used_bits = np.uint32(0)
    for word in words:
        if _FIRST <= word <= _LAST:
            histogram[(word - _FIRST) >> _SHIFT] += 1
            count += 1
            used_bits |= word
        elif word != 0 and word != 0x80000000:
            return histogram, -1
    if used_bits & ((1 << _SHIFT) - 1):
        return histogram, -1
    return histogram, count


@njit(cache=True, nogil=True)
def _neighbors(histogram, low, high):
    count = 0
    lower_bin = -1
    for i, frequency in enumerate(histogram):
        count += frequency
        if count > low and lower_bin < 0:
            lower_bin = i
        if count > high:
            return lower_bin, i
    raise ValueError("quantile rank exceeds histogram population")


@lru_cache(maxsize=1)
def _single_precision_rank():
    # NumPy releases differ in the rank precision for a Python float q and
    # float32 samples. At this q the lower-rank interpolation is zero only
    # when NumPy rounded q to single precision before computing the rank.
    values = np.array([0, 0, 1], dtype=np.float32)
    return np.quantile(values, 0.5 + 2.0 ** -27) == 0


def grid_quantile(values, quantile):
    """Return None when the array requires the ordinary NumPy path."""
    if (values.dtype != np.float32 or not values.flags.c_contiguous
            or values.size >= 2**31 or type(quantile) is not float):
        return None
    counts, size = _count_grid(values.ravel().view(np.uint32))
    if size < 0:
        return None
    if size == 0:
        return 0.0
    low_bin, high_bin, fraction = _quantile_bins(counts, size, quantile)
    words = np.array([_FIRST + (low_bin << _SHIFT), _FIRST + (high_bin << _SHIFT)], dtype=np.uint32)
    endpoints = words.view(np.float32)
    if low_bin == high_bin:
        return float(endpoints[0])
    # NumPy retains its installed version's rounding in the interpolation.
    return float(np.quantile(endpoints, fraction))

def _quantile_bins(counts, size, quantile):
    q = np.float32(quantile) if _single_precision_rank() else quantile
    rank = (size - 1) * q
    low = min(int(rank), size - 1)
    lower, upper = _neighbors(counts, low, min(low + 1, size - 1))
    return lower, upper, float(rank - low)


@lru_cache(maxsize=1)
def _half_levels():
    # Positive half words and the negative-zero word index a table directly.
    words = np.arange(0x8001, dtype=np.uint16)
    words[0x7c00:0x8000] = 0
    return words.view(np.float16).astype(np.float32)


@njit(cache=True, nogil=True)
def _half_histogram(words):
    counts = np.zeros(0x7c00, np.int32)
    total = 0
    for word in words.ravel():
        if 0 < word < 0x7c00:
            counts[word] += 1
            total += 1
        elif word != 0 and word != 0x8000:
            raise ValueError("APC attention matrix must be finite and nonnegative")
    return counts, total


@njit(cache=True, nogil=True)
def _write_half_tiles(words, tiles, table, starts):
    for i, start in enumerate(starts):
        first = max(start, 0)
        stop = min(start + tiles.shape[-1], words.shape[0])
        for y in range(first, stop):
            for x in range(first, stop):
                tiles[i, y-start, x-start] = table[words[y, x]]


def native_half_tiles(matrices, spans, tile_size, quantile, clip):
    """Normalize the exact original half words directly into the teacher tiles."""
    if not 0.5 <= quantile < 1.0 or clip <= 0:
        raise ValueError("invalid attention normalization quantile or clip")
    starts = np.asarray([span["source_start"] for span in spans], dtype=np.int32)
    tiles = np.zeros((len(starts), len(matrices)+1, tile_size, tile_size), dtype=np.float32)
    length = matrices[0].shape[0]
    for i, start in enumerate(starts):
        valid = slice(max(0, -start), min(tile_size, length-start))
        tiles[i, -1, valid, valid] = 1
    denominator = np.log1p(clip)
    levels = _half_levels()
    for channel, matrix in enumerate(matrices):
        words = matrix.view(np.uint16)
        counts, count = _half_histogram(words)
        if count == 0:
            continue
        lower, upper, fraction = _quantile_bins(counts, count, quantile)
        scale = float(levels[lower]) if lower == upper else float(np.quantile(levels[[lower,upper]], fraction))
        table = levels / scale
        np.log1p(table, out=table)
        np.divide(np.clip(table, 0., denominator), denominator, out=table)
        _write_half_tiles(words, tiles[:, channel], table, starts)
    return tiles
