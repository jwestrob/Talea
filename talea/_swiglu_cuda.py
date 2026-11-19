"""Gather the backend's exact fp16 SiLU output and multiply by the up projection."""
import torch
import triton
import triton.language as tl


@triton.jit
def _gate(x, table, out, size, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < size
    j = (i // WIDTH) * (2 * WIDTH) + i % WIDTH
    gate = tl.load(x + j, valid, 0).to(tl.uint16, bitcast=True).to(tl.int32)
    value = tl.load(x + j + WIDTH, valid, 0)
    activation = tl.load(table + gate, valid, 0)
    tl.store(out + i, activation.to(tl.float32) * value.to(tl.float32), valid)


def apply(x, table):
    width = x.shape[-1] // 2
    out = torch.empty((*x.shape[:-1], width), dtype=x.dtype, device=x.device)
    with torch.cuda.device(x.device):
        _gate[(triton.cdiv(out.numel(), 256),)](x, table, out, out.numel(), WIDTH=width, BLOCK=256)
    return out
