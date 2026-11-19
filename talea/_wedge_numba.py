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
