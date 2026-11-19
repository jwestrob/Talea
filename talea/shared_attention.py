"""Bounded shared storage for fp16 and fp32 attention maps sent to process workers."""

from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from multiprocessing import RawArray
from threading import Lock
from typing import NamedTuple
from weakref import ref

import numpy as np


class AttentionSpan(NamedTuple):
    offset: int
    length: int
    layers: tuple[int, ...]
    half: bool = False

    def matrices(self, buffer):
        size = self.length**2
        flat = np.frombuffer(buffer, dtype=np.float16 if self.half else np.float32, count=len(self.layers) * size,
                             offset=self.offset * np.dtype(np.float32).itemsize)
        stack = flat.reshape(len(self.layers), self.length, self.length)
        stack.flags.writeable = False
        return dict(zip(self.layers, stack, strict=True))


class _ReservedMatrices(dict):
    """A writer owns these half maps until submission transfers the lease to a future."""

    def __init__(self, arena, span, values):
        super().__init__(zip(span.layers, values, strict=True))
        self.arena = arena
        self.span = span
        self.submitted = False
        self._finished = False

    def done(self):
        return self._finished

    def release(self, future):
        self._finished = True


class SharedAttentionArena:
    """A producer and its workers inherit the same buffer at process startup.

    Future completion releases each span. A full ring, an oversize record or
    a different matrix representation uses the existing pickle transport.
    """

    def __init__(self, capacity_bytes):
        if capacity_bytes < np.dtype(np.float32).itemsize:
            raise ValueError("shared attention capacity must hold at least one float32")
        self.buffer = RawArray("f", capacity_bytes // np.dtype(np.float32).itemsize)
        self._flat = np.frombuffer(self.buffer, dtype=np.float32)
        self.capacity_bytes = self._flat.nbytes
        self._head = 0
        self._live = deque()
        self._lock = Lock()
        self.shared_records = self.pickled_records = 0
        self.direct_records = 0

    def _placement(self, count):
        capacity = self._flat.size
        if not 0 < count <= capacity:
            return None
        while self._live:
            oldest = self._live[0][1]()
            if oldest is not None and not oldest.done():
                break
            self._live.popleft()
        start = self._head if self._live else 0
        offset = start % capacity
        if offset + count > capacity:
            start += capacity - offset
            offset = 0
        if self._live and start + count - self._live[0][0] > capacity:
            return None
        return start, offset

    def reserve_half(self, length, layers):
        layers = tuple(layers)
        elements = len(layers) * length**2
        count = (elements + 1) // 2
        with self._lock:
            where = self._placement(count)
            if where is None:
                return None
            start, offset = where
            span = AttentionSpan(offset, length, layers, half=True)
            values = self._flat[offset:offset+count].view(np.float16)[:elements].reshape(len(layers), length, length)
            maps = _ReservedMatrices(self, span, values)
            self._head = start + count
            self._live.append((start, ref(maps)))
            return maps

    @contextmanager
    def page_locked(self):
        """Register existing inherited storage for direct D2H as long as its readers run."""
        import torch
        runtime = torch.cuda.cudart()
        address = self._flat.ctypes.data
        status = runtime.cudaHostRegister(address, self.capacity_bytes, 0)
        # A host memory limit or unsupported registration uses the normal pinned
        # tensor staging buffer. Unexpected CUDA errors need their original report.
        if int(status) in (1, 2, 712, 801):
            yield False
            return
        torch.cuda.check_error(status)
        try:
            yield True
        finally:
            torch.cuda.check_error(runtime.cudaHostUnregister(address))


    def submit(self, pool, function, payload):
        metadata, matrices, *rest = payload
        if isinstance(matrices, _ReservedMatrices):
            with self._lock:
                if matrices.arena is not self or matrices.submitted or metadata["length"] != matrices.span.length:
                    raise ValueError("invalid or previously submitted shared attention reservation")
                matrices.submitted = True
            try:
                future = pool.submit(function, (metadata, matrices.span, *rest))
            except BaseException:
                with self._lock:
                    matrices.submitted = False
                raise
            # The callback holds the maps through asynchronous work. A completed
            # or cancelled future releases this span for the next producer.
            future.add_done_callback(matrices.release)
            with self._lock:
                self.shared_records += 1
                self.direct_records += 1
            return future
        length = metadata["length"]
        layers = tuple(matrices)
        elements = len(layers) * length**2
        half = bool(layers) and matrices[layers[0]].dtype == np.float16
        dtype = np.float16 if half else np.float32
        # Ring offsets count 4-byte words. Round odd-sized half spans up.
        count = (elements + 1) // 2 if half else elements
        capacity = self._flat.size
        suitable = 0 < count <= capacity and all(
            matrix.dtype == dtype and matrix.shape == (length, length)
            for matrix in matrices.values())
        with self._lock:
            where = self._placement(count) if suitable else None
            if where is None:
                self.pickled_records += 1
                return pool.submit(function, payload)
            start, offset = where
            dest = self._flat[offset:offset + count].view(dtype)[:elements].reshape(len(layers), length, length)
            for dest_layer, layer in zip(dest, layers, strict=True):
                np.copyto(dest_layer, matrices[layer])
            span = AttentionSpan(offset, length, layers, half)
            future = pool.submit(function, (metadata, span, *rest))
            self._head = start + count
            self._live.append((start, ref(future)))
            self.shared_records += 1
            return future
