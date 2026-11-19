"""Fill the frozen U-Net's fp32 tiles directly from two on-device fp16 APC maps."""
from contextlib import contextmanager, nullcontext
import torch
import triton
import triton.language as tl


@triton.jit
def _fill(a, b, tables, output, length, start_tile, center, flank,
          TABLE, SIDE: tl.constexpr, BLOCK: tl.constexpr):
    tile = tl.program_id(1)
    pixel = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    start = (start_tile + tile) * center - flank
    y = start + pixel // SIDE
    x = start + pixel % SIDE
    valid = (pixel < SIDE * SIDE) & (x >= 0) & (x < length) & (y >= 0) & (y < length)
    index = y * length + x
    word_a = tl.load(a + index, mask=valid, other=0.).to(tl.uint16, bitcast=True).to(tl.int32)
    word_b = tl.load(b + index, mask=valid, other=0.).to(tl.uint16, bitcast=True).to(tl.int32)
    value_a = tl.load(tables + tl.where(word_a == 0x8000, TABLE - 1, word_a), mask=valid, other=0.)
    value_b = tl.load(tables + TABLE + tl.where(word_b == 0x8000, TABLE - 1, word_b), mask=valid, other=0.)
    offset = tile * 3 * SIDE * SIDE + pixel
    tl.store(output + offset, value_a, mask=pixel < SIDE * SIDE)
    tl.store(output + offset + SIDE * SIDE, value_b, mask=pixel < SIDE * SIDE)
    tl.store(output + offset + 2 * SIDE * SIDE, valid.to(tl.float32), mask=pixel < SIDE * SIDE)


def fill_tiles(maps, table, out, first, center):
    side = out.shape[-1]
    _fill[(triton.cdiv(side * side, 256), len(out))](
        *maps, table, out, maps[0].shape[0], first, center, (side - center) // 2,
        TABLE=table.shape[1], SIDE=side, BLOCK=256,
    )


class DeviceTeacherTiles:
    def __init__(self, maps, tables, side, center):
        self.maps = maps
        self.tables = tables
        self.side, self.center = side, center

        self.device_tables = None

    def __len__(self):
        return (self.maps[0].shape[0] + self.center - 1) // self.center

    @contextmanager
    def input_batch(self, start, size, device, graphs):
        if self.device_tables is None:
            # The graph owner may consume maps created on another CUDA stream.
            for tensor in self.maps:
                tensor.record_stream(torch.cuda.current_stream(device))
            self.device_tables = torch.from_numpy(self.tables).to(device, non_blocking=True)
        shape = (size, 3, self.side, self.side)
        buffer = graphs.input_buffer(size) if graphs is not None else nullcontext(
            torch.empty(shape, dtype=torch.float32, device=device)
        )
        with buffer as target, torch.cuda.device(device):
            fill_tiles(self.maps, self.device_tables, target, start, self.center)
            yield target
