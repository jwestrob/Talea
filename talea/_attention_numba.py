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
