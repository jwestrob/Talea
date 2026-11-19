"""Single-threaded banded Viterbi updates for the phase decoder."""
import numpy as np
from numba import njit


@njit(cache=True, fastmath=False)
def transitions(full, lead, scores, back, minimum_length, length_change_penalty, maximum_step):
    n, count = full.shape[0] - 1, full.shape[1]
    for r in range(count):
        length = minimum_length + r
        for offset in range(min(length, n - length + 1)):
            scores[offset + length, r] = lead[offset, r] + full[offset, r]
            back[offset + length, r] = -2

    for start in range(1, n - minimum_length + 1):
        for r in range(min(count, n - start - minimum_length + 1)):
            # Only lengths within maximum_step can precede this unit. Ascending
            # predecessor order and strict comparisons retain NumPy argmax ties.
            best = -np.inf
            previous = -1
            for prior in range(max(0, r - maximum_step), min(count, r + maximum_step + 1)):
                value = scores[start, prior] - length_change_penalty * abs(prior - r)
                if value > best:
                    best = value
                    previous = prior
            end = start + minimum_length + r
            proposal = best + full[start, r]
            if np.isfinite(best) and proposal > scores[end, r]:
                scores[end, r] = proposal
                back[end, r] = previous
