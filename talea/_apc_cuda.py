"""A5000 fp16 APC pointwise kernels; all reductions remain PyTorch operations."""
import torch
import triton
import triton.language as tl


@triton.jit
def _sym(source, symmetric, length, offset, heads, bs, hs, rs, cs, TILE: tl.constexpr):
    row = tl.program_id(0) * TILE + tl.arange(0, TILE)
    col = tl.program_id(1) * TILE + tl.arange(0, TILE)
    flat_head = tl.program_id(2)
    base = source + (flat_head // heads) * bs + (flat_head % heads) * hs
    valid = (row[:, None] < length) & (col[None, :] < length)
    idx = (offset + row[:, None]) * rs + (offset + col[None, :]) * cs
    # Load a contiguous tile from each side of the diagonal, then transpose
    # registers. This accepts the actual ESM attention strides and padding.
    peer = (offset + col[:, None]) * rs + (offset + row[None, :]) * cs
    a = tl.load(base + idx, valid, 0).to(tl.float32)
    b = tl.trans(tl.load(base + peer, tl.trans(valid), 0)).to(tl.float32)
    summed = (a + b).to(tl.float16).to(tl.float32)
    out = (summed * 0.5).to(tl.float16)
    tl.store(symmetric + (flat_head * length + row[:, None]) * length + col[None, :], out, valid)


@triton.jit
def _correct(symmetric, rows, cols, totals, length, TILE: tl.constexpr):
    row = tl.program_id(0) * TILE + tl.arange(0, TILE)
    col = tl.program_id(1) * TILE + tl.arange(0, TILE)
    head = tl.program_id(2)
    valid = (row[:, None] < length) & (col[None, :] < length)
    addr = symmetric + (head * length + row[:, None]) * length + col[None, :]
    x = tl.load(addr, valid, 0).to(tl.float32)
    r = tl.load(rows + head * length + row, row < length, 0).to(tl.float32)
    c = tl.load(cols + head * length + col, col < length, 0).to(tl.float32)
    total = tl.load(totals + head).to(tl.float32)
    numerator = (r[:, None] * c[None, :]).to(tl.float16).to(tl.float32)
    # c10::Half's binary CUDA division promotes to float, divides and rounds
    # to half. div_rn is IEEE fp32 division, not an approximate reciprocal.
    ratio = tl.div_rn(numerator, total).to(tl.float16).to(tl.float32)
    difference = (x - ratio).to(tl.float16)
    tl.store(addr, tl.where(difference < 0, 0., difference), valid)


class FusedAPC:
    def __init__(self, mode):
        self.full = mode == "full"

    def symmetrize(self, attention, start, length):
        x = attention.unsqueeze(0) if attention.ndim == 3 else attention
        batch, heads = x.shape[:2]
        symmetric = torch.empty((batch, heads, length, length), dtype=x.dtype, device=x.device)
        grid = (triton.cdiv(length, 32), triton.cdiv(length, 32), batch * heads)
        with torch.cuda.device(x.device):
            _sym[grid](x, symmetric, length, start, heads, *x.stride(),
                       TILE=32, enable_fp_fusion=False)
        return symmetric[0] if attention.ndim == 3 else symmetric

    def correct(self, symmetric, rows, cols, totals):
        length = symmetric.shape[-1]
        n_heads = symmetric.numel() // (length * length)
        grid = (triton.cdiv(length, 32), triton.cdiv(length, 32), n_heads)
        with torch.cuda.device(symmetric.device):
            _correct[grid](symmetric, rows, cols, totals, length, TILE=32, enable_fp_fusion=False)
        return symmetric
