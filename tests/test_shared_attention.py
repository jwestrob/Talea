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


def test_odd_half_span_preserves_bits_and_adjacent_float_span():
    ring = SharedAttentionArena(72)
    pool = DeferredPool()
    half = np.arange(9, dtype=np.float16).reshape(3, 3)
    half[1, 1] = -0.
    full = np.full((3, 3), 1.115, dtype=np.float32)
    for m in (half, full):
        ring.submit(pool, None, ({"length": 3}, {13:m}, ()))
    assert ring.shared_records == 2
    for (task, _), m in zip(pool.tasks, (half, full)):
        recovered = task[1].matrices(ring.buffer)[13]
        assert recovered.dtype == m.dtype
        assert recovered.tobytes() == m.tobytes()


def test_direct_half_out_of_order_lifetimes_and_retry():
    from concurrent.futures import ThreadPoolExecutor
    import pytest

    ring = SharedAttentionArena(32)
    pool = DeferredPool()
    layers = (13, 35)
    with ThreadPoolExecutor(2) as executor:
        first, second = list(executor.map(lambda _: ring.reserve_half(2, layers), range(2)))
    assert first.span.offset != second.span.offset
    for lease, value in ((first, 3), (second, 7)):
        for layer in lease:
            lease[layer].fill(value)
    assert ring.reserve_half(2, layers) is None
    payload_for = lambda m: ({"length": 2}, m, (), None, None)
    class Broken:
        def submit(self, fn, payload):
            raise RuntimeError("pool unavailable")
    with pytest.raises(RuntimeError):
        ring.submit(Broken(), None, payload_for(second))
    second_future = ring.submit(pool, None, payload_for(second))
    with pytest.raises(ValueError):
        ring.submit(pool, None, payload_for(second))
    first_future = ring.submit(pool, None, payload_for(first))
    del first, second
    assert ring.reserve_half(2, layers) is None
    for i, value in enumerate((7, 3)):
        received = pool.tasks[i][0][1].matrices(ring.buffer)
        assert all(a.dtype == np.float16 and np.all(a == value) for a in received.values())
    second_future.cancel()
    first_future.set_result(None)
    assert ring.reserve_half(2, layers) is not None


def test_abandoned_unsubmitted_ticket_releases_ring():
    ring = SharedAttentionArena(16)
    reserved = ring.reserve_half(2, (13, 35))
    assert reserved is not None and ring.reserve_half(2, (13, 35)) is None
    del reserved
    assert ring.reserve_half(2, (13, 35)) is not None
