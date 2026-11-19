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
