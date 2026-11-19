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
def _best_blocked(prefix, width, used, indices, weights, counts, denominators, periods):
    """Reuse a block of diagonal window sums across every period template."""
    starts = prefix.shape[1] - width
    block = 128
    sums = np.empty((len(used), block), dtype=np.float64)
    scores = np.empty(block, dtype=np.float64)
    maxima = np.full(len(counts), -np.inf)
    first = np.zeros(len(counts), dtype=np.int64)
    finite = np.zeros(len(counts), dtype=np.bool_)
    for base in range(0, starts, block):
        size = min(block, starts - base)
        for k, separation in enumerate(used):
            span = width - separation
            for j in range(size):
                sums[k, j] = prefix[separation, base + j + span] - prefix[separation, base + j]
        for row in range(len(counts)):
            scores[:size] = 0.0
            for slot in range(counts[row]):
                values = sums[indices[row, slot]]
                weight = weights[row, slot]
                for j in range(size):
                    scores[j] += weight * values[j]
            top, at, ok = maxima[row], first[row], finite[row]
            for j in range(size):
                score = scores[j] / denominators[row]
                if np.isfinite(score):
                    ok = True
                if score > top:
                    top, at = score, base + j
            maxima[row], first[row], finite[row] = top, at, ok
    best, start, score = -1, -1, -np.inf
    for row in range(len(counts)):
        if finite[row] and (best < 0 or maxima[row] > score
                            or (maxima[row] == score and periods[row] < periods[best])):
            best, start, score = row, first[row], maxima[row]
    return best, start, score


@njit(cache=True, fastmath=False)
def best_from_prefixes(prefix, width, used, indices, weights, counts, denominators, periods):
    """Keep only the best start and template for this width, in stock tie order."""
    # The full-width gather wins on short searches. For long searches its
    # working set exceeds cache; each block then supplies every template.
    if prefix.shape[1] - width >= 768:
        return _best_blocked(prefix, width, used, indices, weights, counts, denominators, periods)
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
def _initial_median(row):
    size = len(row)
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
    if size % 2:
        return median, True
    lower = np.max(ordered[:middle])
    return (lower + median) / 2.0, lower == median


@njit(cache=True, fastmath=False)
def _median_after_suppression(row, previous):
    # Replacing values by the previous median cannot increase the number
    # on either side. Each middle order statistic is therefore either that
    # median or the nearest remaining value on its side.
    below, above = 0, 0
    lower, upper = -np.inf, np.inf
    for value in row:
        if value < previous:
            below += 1
            lower = max(lower, value)
        elif value > previous:
            above += 1
            upper = min(upper, value)
    half = len(row) // 2
    if below < half:
        lower = previous
    if above < half:
        upper = previous
    return (lower + upper) / 2.0, lower == upper


@njit(cache=True, fastmath=False)
def suppress_packed(values, offsets, separations, start, end, median_state=None):
    """Set overlapping pairs to each track's current median; reuse known middle ranks."""
    for i, separation in enumerate(separations):
        first, last = offsets[separation], offsets[separation + 1]
        row = values[first:last]
        size = len(row)
        if size == 0:
            continue
        # State 0 is the first visit, 1 has a stable median, 2 needs the
        # neighboring ranks, and 3 permanently requests fresh partitions.
        mode = 0 if median_state is None else median_state[1][i]
        if mode == 1:
            median, stable = median_state[0][i], True
        elif mode == 2:
            median, stable = _median_after_suppression(row, median_state[0][i])
        else:
            median, stable = _initial_median(row)
        if median_state is not None:
            medians, modes = median_state
            medians[i] = median
            # A replacement can introduce a signed zero or infinity that
            # the original input guard did not see. Retain partition ties.
            if mode == 3 or not np.isfinite(median) or (median == 0 and np.signbit(median)):
                modes[i] = 3
            else:
                modes[i] = 1 if stable else 2
        row[max(0, start):min(size, max(0, end))] = median
        row[max(0, start - separation):min(size, max(0, end - separation))] = median
