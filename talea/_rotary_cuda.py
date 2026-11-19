"""Fused rotary embedding for the pinned 18-head ESM++ inference contract."""
import torch
import triton
import triton.language as tl


@triton.jit
def _rotate(q, k, cos, sin, out_q, out_k, tokens, size, BLOCK: tl.constexpr):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = idx < size
    col = idx % 64
    peer = idx + tl.where(col < 32, 32, -32)
    table = (idx // 1152 % tokens) * 32 + col % 32
    c = tl.load(cos + table, valid, 0).to(tl.float32)
    s = tl.load(sin + table, valid, 0).to(tl.float32)
    qv = tl.load(q + idx, valid, 0).to(tl.float32)
    kv = tl.load(k + idx, valid, 0).to(tl.float32)
    qp = tl.load(q + peer, valid, 0).to(tl.float32)
    kp = tl.load(k + peer, valid, 0).to(tl.float32)
    qc = (qv*c).to(tl.float16).to(tl.float32)
    kc = (kv*c).to(tl.float16).to(tl.float32)
    qs = (tl.where(col < 32, -qp, qp)*s).to(tl.float16).to(tl.float32)
    ks = (tl.where(col < 32, -kp, kp)*s).to(tl.float16).to(tl.float32)
    tl.store(out_q + idx, qc+qs, valid)
    tl.store(out_k + idx, kc+ks, valid)


def apply(q, k, cos, sin):
    out_q, out_k = torch.empty_like(q), torch.empty_like(k)
    size = q.numel()
    with torch.cuda.device(q.device):
        _rotate[(triton.cdiv(size, 256),)](q, k, cos, sin, out_q, out_k, q.shape[1], size,
                                          BLOCK=256, enable_fp_fusion=False)
    return out_q, out_k
