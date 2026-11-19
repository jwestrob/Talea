"""Bounded shared storage for fp16 and fp32 attention maps sent to process workers."""

from __future__ import annotations

from collections import deque
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

    def submit(self, pool, function, payload):
        metadata, matrices, *rest = payload
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
            if not suitable or (self._live and start + count - self._live[0][0] > capacity):
                self.pickled_records += 1
                return pool.submit(function, payload)
            dest = self._flat[offset:offset + count].view(dtype)[:elements].reshape(len(layers), length, length)
            for dest_layer, layer in zip(dest, layers, strict=True):
                np.copyto(dest_layer, matrices[layer])
            span = AttentionSpan(offset, length, layers, half)
            future = pool.submit(function, (metadata, span, *rest))
            self._head = start + count
            self._live.append((start, ref(future)))
            self.shared_records += 1
            return future
