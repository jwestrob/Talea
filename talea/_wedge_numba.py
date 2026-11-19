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
        output[row] = _median_deviation(values, n, 0.0, center)
    return output


@njit(cache=True, fastmath=False)
def _median_deviation(values, n, baseline, center):
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
        last_left = abs((np.float64(values[left_size - left]) - baseline) - center) if left else -np.inf
        next_left = abs((np.float64(values[left_size - left - 1]) - baseline) - center) if left < left_size else np.inf
        last_right = abs((np.float64(values[left_size + right - 1]) - baseline) - center) if right else -np.inf
        next_right = abs((np.float64(values[left_size + right]) - baseline) - center) if right < right_size else np.inf
        if last_left > next_right:
            upper = left - 1
        elif last_right > next_left:
            lower = left + 1
        else:
            middle = max(last_left, last_right)
            if n % 2 == 0:
                middle = (middle + min(next_left, next_right)) / 2
            return middle
    return np.nan


@njit(cache=True, fastmath=False)
def normalize_sorted_f32(padded, ordered, baselines, counts, clip):
    """Normalize finite rows directly; send near-constant rows to NumPy std."""
    output = np.zeros(padded.shape, dtype=np.float64)
    use_std = np.zeros(len(counts), dtype=np.bool_)
    for row in range(len(counts)):
        n = counts[row]
        base = baselines[row]
        middle = n // 2
        center = np.float64(ordered[row, middle]) - base
        if n % 2 == 0:
            lower = np.float64(ordered[row, middle - 1]) - base
            center = (lower + center) / 2
        scale = 1.4826 * _median_deviation(ordered[row], n, base, center)
        if scale < 1e-12:
            use_std[row] = True
        else:
            for i in range(n):
                value = (np.float64(padded[row, i]) - base) / scale
                output[row, i] = min(clip, max(-clip, value))
    return output, use_std


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
def _window_sums(prefix, width, used):
    """Compute the needed window sums directly into their shared row buffer."""
    starts = prefix.shape[1] - width
    sums = np.empty((len(used), starts), dtype=np.float64)
    for row in range(len(used)):
        separation = used[row]
        span = width - separation
        for j in range(starts):
            sums[row, j] = prefix[separation, j + span] - prefix[separation, j]
    return sums


@njit(cache=True, fastmath=False)
def scores_from_prefixes(prefix, width, used, indices, weights, counts, denominators):
    sums = _window_sums(prefix, width, used)
    return accumulate_scores(sums, indices, weights, counts, denominators)


@njit(cache=True, fastmath=False)
def best_from_prefixes(prefix, width, used, indices, weights, counts, denominators, periods):
    """Keep only the best start and template for this width, in stock tie order."""
    sums = _window_sums(prefix, width, used)
    scores = np.empty(sums.shape[1], dtype=np.float64)
    best_row, best_start, best_score = -1, -1, -np.inf
    for row in range(len(counts)):
        scores.fill(0.0)
        for slot in range(counts[row]):
            values = sums[indices[row, slot]]
            weight = weights[row, slot]
            for start in range(len(scores)):
                scores[start] += weight * values[start]
        row_score, row_start, finite = -np.inf, 0, False
        for start in range(len(scores)):
            score = scores[start] / denominators[row]
            if np.isfinite(score):
                finite = True
            if score > row_score:
                row_score, row_start = score, start
        if finite and (best_row < 0 or row_score > best_score
                       or (row_score == best_score and periods[row] < periods[best_row])):
            best_row, best_start, best_score = row, row_start, row_score
    return best_row, best_start, best_score



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


@njit(cache=True, fastmath=False)
def suppress_packed(values, offsets, separations, start, end):
    """Set touched residue pairs in the selected diagonals to their current medians."""
    for separation in separations:
        first, last = offsets[separation], offsets[separation + 1]
        row = values[first:last]
        size = len(row)
        if size == 0:
            continue
        ordered = row.copy()
        middle = size // 2
        low, high = 0, size - 1
        while low < high:
            # Hoare partition, searching only for the upper middle element.
            pivot = ordered[(low + high) // 2]
            i, j = low, high
            while i <= j:
                while ordered[i] < pivot:
                    i += 1
                while ordered[j] > pivot:
                    j -= 1
                if i <= j:
                    ordered[i], ordered[j] = ordered[j], ordered[i]
                    i += 1
                    j -= 1
            if middle <= j:
                high = j
            elif middle >= i:
                low = i
            else:
                break
        median = ordered[middle]
        if size % 2 == 0:
            median = (np.max(ordered[:middle]) + median) / 2.0
        row[max(0, start):min(size, max(0, end))] = median
        row[max(0, start - separation):min(size, max(0, end - separation))] = median
