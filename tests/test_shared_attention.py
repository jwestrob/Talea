from concurrent.futures import Future

import numpy as np

from talea.shared_attention import AttentionSpan, SharedAttentionArena


class DeferredPool:
    def __init__(self):
        self.tasks = []

    def submit(self, fn, payload):
        future = Future()
        self.tasks.append((payload, future))
        return future


def payload(n, value, dtype=np.float32):
    matrix = np.full((n, n), value, dtype=dtype)
    matrix[0, 0] = -0.0
    return ({"length": n}, {13: matrix.T, 35: matrix[::-1]}, (), None, None)


def test_completed_span_reuse_preserves_live_matrices_and_fallback():
    ring = SharedAttentionArena(64)
    pool = DeferredPool()
    a, b = payload(2, 7), payload(2, 11)
    f = ring.submit(pool, None, a)
    ring.submit(pool, None, b)
    ring.submit(pool, None, payload(2, 17))
    first, second, fallback = [item[0][1] for item in pool.tasks]
    assert isinstance(first, AttentionSpan) and isinstance(second, AttentionSpan)
    assert isinstance(fallback, dict)
    for layer, array in first.matrices(ring.buffer).items():
        assert array.tobytes() == a[1][layer].tobytes()
        assert not array.flags.writeable
    f.set_result(None)
    ring.submit(pool, None, payload(2, 19))
    third = pool.tasks[-1][0][1]
    assert third.offset == first.offset
    for layer, array in second.matrices(ring.buffer).items():
        assert array.tobytes() == b[1][layer].tobytes()


def test_fallback_for_non_native_dtypes_and_large_matrices():
    ring = SharedAttentionArena(64)
    pool = DeferredPool()
    for data in (payload(4, 3), payload(2, 9, np.float64)):
        ring.submit(pool, None, data)
        assert pool.tasks[-1][0] is data
    assert ring.shared_records == 0 and ring.pickled_records == 2


def test_cancel_and_submit_failure_leave_capacity_available():
    ring = SharedAttentionArena(32)
    pool = DeferredPool()
    data = payload(2, 2)
    f = ring.submit(pool, None, data)
    f.cancel()
    class Broken:
        def submit(self, fn, payload):
            raise RuntimeError("pool unavailable")
    import pytest
    with pytest.raises(RuntimeError, match="unavailable"):
        ring.submit(Broken(), None, data)
    ring.submit(pool, None, data)
    assert isinstance(pool.tasks[-1][0][1], AttentionSpan)
