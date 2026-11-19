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
    q = np.float32(quantile) if _single_precision_rank() else quantile
    rank = (size - 1) * q
    low = min(int(rank), size - 1)
    low_bin, high_bin = _neighbors(counts, low, min(low + 1, size - 1))
    words = np.array([_FIRST + (low_bin << _SHIFT), _FIRST + (high_bin << _SHIFT)], dtype=np.uint32)
    endpoints = words.view(np.float32)
    if low_bin == high_bin:
        return float(endpoints[0])
    # NumPy retains its installed version's rounding in the interpolation.
    return float(np.quantile(endpoints, float(rank - low)))
