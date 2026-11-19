"""Optional single-threaded compiled wedge accumulation for process workers."""

import numpy as np
from numba import njit


@njit(cache=True, fastmath=False)
def accumulate_scores(sums, indices, weights, counts, denominators):
    scores = np.zeros((len(counts), sums.shape[1]), dtype=np.float64)
    for row in range(len(counts)):
        for slot in range(counts[row]):
            values = sums[indices[row, slot]]
            weight = weights[row, slot]
            for start in range(sums.shape[1]):
                scores[row, start] += weight * values[start]
        for start in range(sums.shape[1]):
            scores[row, start] /= denominators[row]
    return scores


@njit(cache=True, fastmath=False)
def median_absolute_deviations(centered_sorted, centers, counts):
    """Median absolute deviations from two already-sorted halves of each track."""
    output = np.empty(len(counts), dtype=np.float64)
    for row in range(len(counts)):
        n = counts[row]
        center = centers[row]
        values = centered_sorted[row]
        # Left of the median the absolute deviations ascend in reverse order,
        # and to its right they ascend in forward order. Find the partition
        # of these two lists at the middle of their sorted union.
        left_size = n // 2
        right_size = n - left_size
        needed = (n + 1) // 2
        lower = max(0, needed - right_size)
        upper = min(needed, left_size)
        while lower <= upper:
            left = (lower + upper) // 2
            right = needed - left
            last_left = abs(values[left_size - left] - center) if left else -np.inf
            next_left = abs(values[left_size - left - 1] - center) if left < left_size else np.inf
            last_right = abs(values[left_size + right - 1] - center) if right else -np.inf
            next_right = abs(values[left_size + right] - center) if right < right_size else np.inf
            if last_left > next_right:
                upper = left - 1
            elif last_right > next_left:
                lower = left + 1
            else:
                middle = max(last_left, last_right)
                if n % 2 == 0:
                    middle = (middle + min(next_left, next_right)) / 2
                output[row] = middle
                break
    return output


@njit(cache=True, fastmath=False)
def track_prefixes(values, offsets, used, length):
    """Accumulate each requested diagonal in spatial order, starting at its first value."""
    prefix = np.zeros((len(offsets) - 1, length + 1), dtype=np.float64)
    for row in used:
        first = offsets[row]
        size = offsets[row + 1] - first
        if size:
            total = values[first]
            prefix[row, 1] = total
            for i in range(1, size):
                total += values[first + i]
                prefix[row, i + 1] = total
    return prefix


@njit(cache=True, fastmath=False)
def scores_from_prefixes(prefix, width, used, indices, weights, counts, denominators):
    """Compute the needed window sums directly into their shared row buffer."""
    starts = prefix.shape[1] - width
    sums = np.empty((len(used), starts), dtype=np.float64)
    for row in range(len(used)):
        separation = used[row]
        span = width - separation
        for j in range(starts):
            sums[row, j] = prefix[separation, j + span] - prefix[separation, j]
    return accumulate_scores(sums, indices, weights, counts, denominators)


@njit(cache=True, fastmath=False)
def padded_diagonals_f32(matrix, maximum):
    """Gather the spatial diagonals into the row order consumed by NumPy sort."""
    length = len(matrix)
    result = np.empty((maximum, length - 1), dtype=np.float32)
    for d in range(1, maximum + 1):
        for i in range(length - d):
            result[d - 1, i] = matrix[i, i + d]
        for i in range(length - d, length - 1):
            result[d - 1, i] = np.inf
    return result
