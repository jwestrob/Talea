"""Optional compiled prefix sums for float32 attention matrices in process workers."""

import numpy as np
from numba import njit


@njit(cache=True, fastmath=False)
def integral_image_f32(values):
    """Column cumsum followed by row cumsum, in a single row-major traversal."""
    length = values.shape[0]
    prefix = np.zeros((length + 1, length + 1), dtype=np.float64)
    columns = np.empty(length, dtype=np.float64)
    for i in range(length):
        row_sum = 0.0
        for j in range(length):
            value = np.float64(values[i, j])
            if not np.isfinite(value):
                raise ValueError("attention matrix contains nonfinite values")
            columns[j] = value if i == 0 else columns[j] + value
            row_sum = columns[j] if j == 0 else row_sum + columns[j]
            prefix[i + 1, j + 1] = row_sum
    return prefix


@njit(cache=True, fastmath=False)
def diagonal_prefix_f32(matrix, width):
    """Test the fp16 grid and emit float64 diagonal prefixes with no dense scratch."""
    length = matrix.shape[0]
    flat = matrix.reshape(matrix.size)
    for value in flat:
        scaled = np.float64(value) * 16777216.0
        if abs(value) > 65504.0 or scaled != np.rint(scaled):
            return None
    prefix = np.empty((width, length + 1), dtype=np.float64)
    for separation in range(width):
        prefix[separation, 0] = 0.0
        value = np.float64(flat[separation])
        prefix[separation, 1] = value
        for i in range(1, length):
            offset = i * (length + 1) + separation
            if offset < flat.size:
                value += np.float64(flat[offset])
            prefix[separation, i + 1] = value
    return prefix


@njit(cache=True, fastmath=False)
def diagonal_means_f32(values, require_grid):
    """Accumulate upper diagonals in position order with a contiguous inner loop."""
    length = len(values)
    sums = np.empty(length - 1, dtype=np.float64)
    for separation in range(1, length):
        sums[separation - 1] = values[0, separation]
    for i in range(length):
        for separation in range(1, length - i):
            value = np.float64(values[i, i + separation])
            if require_grid:
                scaled = value * 16777216.0
                if abs(value) > 65504.0 or scaled != np.rint(scaled):
                    return None
            if i > 0:
                sums[separation - 1] += value
    for separation in range(1, length):
        if require_grid:
            sums[separation - 1] += 0.0
        sums[separation - 1] /= length - separation
    return sums


@njit(cache=True, fastmath=False)
def interval_medians(edges):
    """Medians of all adjacent-edge ranges of length >= 2, in (left, right) order."""
    n = len(edges)
    medians = np.empty(n * (n - 1) // 2, dtype=np.float64)
    ordered = np.empty(n, dtype=np.float64)
    position = 0
    for left in range(n - 1):
        ordered[0] = edges[left]
        for right in range(left + 1, n):
            value = edges[right]
            size = right - left + 1
            index = size - 1
            while index > 0 and ordered[index - 1] > value:
                ordered[index] = ordered[index - 1]
                index -= 1
            ordered[index] = value
            middle = size // 2
            median = ordered[middle] if size % 2 else (ordered[middle - 1] + ordered[middle]) / 2.0
            medians[position] = median + 0.0
            position += 1
    return medians


@njit(cache=True, fastmath=False)
def registration_geometry(prefix, offsets, period, fraction, core_start, core_end):
    """Tile all phase offsets and gather the one- and two-unit block means."""
    length = len(prefix) - 1
    capacity = int(length / period) + 3
    units = np.empty((len(offsets), capacity, 2), dtype=np.int64)
    adjacent = np.empty((len(offsets), capacity), dtype=np.float64)
    separated = np.empty((len(offsets), capacity), dtype=np.float64)
    counts = np.zeros(len(offsets), dtype=np.int64)
    core = np.full((len(offsets), 2), -1, dtype=np.int64)
    for idx, offset in enumerate(offsets):
        position = offset
        while position > 0:
            position -= period
        start = 0
        count = 0
        while position < length:
            position += period
            end = max(0, min(length, int(round(position))))
            if end - start >= fraction * period and end > start:
                units[idx, count, 0] = start
                units[idx, count, 1] = end
                if 2 * core_start <= start + end <= 2 * core_end:
                    if core[idx, 0] == -1:
                        core[idx, 0] = count
                    core[idx, 1] = count
                for distance in range(1, min(2, count) + 1):
                    left = count - distance
                    a, b = units[idx, left, 0], units[idx, left, 1]
                    total = prefix[b, end] - prefix[a, end] - prefix[b, start] + prefix[a, start]
                    mean = total / ((b - a) * (end - start))
                    if distance == 1:
                        adjacent[idx, left] = mean
                    else:
                        separated[idx, left] = mean
                count += 1
            start = end
        counts[idx] = count
    return units, adjacent, separated, counts, core
