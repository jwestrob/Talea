"""Optional ESM++ fp16 rotary kernels retaining both intermediate product roundings."""
from contextlib import ExitStack, contextmanager
from functools import lru_cache

import torch


@lru_cache(maxsize=1)
def _metal_rotary():
    return torch.mps.compile_shader("""
#include <metal_stdlib>
using namespace metal;
kernel void rope(device const half *q, device const half *k, device const half *c, device const half *s,
                 device half *oq, device half *ok, constant uint &tokens, uint i [[thread_position_in_grid]]) {
    uint col = i % 64, peer = uint(int(i) + (col < 32 ? 32 : -32));
    uint ci = (i / (64 * 18) % tokens) * 32 + col % 32;
    float cosine = float(c[ci]), sine = float(s[ci]);
    float sign = col < 32 ? -1.0f : 1.0f;
    half qc = half(float(q[i]) * cosine), qs = half((sign * float(q[peer])) * sine);
    half kc = half(float(k[i]) * cosine), ks = half((sign * float(k[peer])) * sine);
    oq[i] = half(float(qc) + float(qs));
    ok[i] = half(float(kc) + float(ks));
}
""").rope


def _metal_apply(q, k, cos, sin):
    out_q, out_k = torch.empty_like(q), torch.empty_like(k)
    _metal_rotary()(q, k, cos, sin, out_q, out_k, q.shape[1], threads=q.numel())
    return out_q, out_k


def _rotary_forward(rotary, apply, counter):
    original = rotary.forward

    def forward(q, k):
        if (torch.is_grad_enabled() or q.dtype != torch.float16 or k.dtype != q.dtype
                or q.shape != k.shape or q.ndim != 4 or q.shape[-2:] != (18, 64)
                or not q.is_contiguous() or not k.is_contiguous()
                or q.storage_offset() or k.storage_offset() or q.device != k.device):
            return original(q, k)
        rotary._update_cos_sin_cache(q.shape[1], device=q.device, dtype=q.dtype)
        cos, sin = rotary._cos_cached, rotary._sin_cached
        if (cos.shape[-1] != 32 or sin.shape != cos.shape or cos.dtype != q.dtype or sin.dtype != q.dtype
                or not cos.is_contiguous() or not sin.is_contiguous() or cos.storage_offset() or sin.storage_offset()):
            return original(q, k)
        counter[0] += 1
        return apply(q, k, cos, sin)
    return forward


@contextmanager
def fused_rotary_context(blocks, device):
    """Restore all ESM module call paths when prediction ends or raises."""
    counter = [0]
    apply = None
    if device.type == "mps" and hasattr(torch.mps, "compile_shader"):
        apply = _metal_apply
    elif device.type == "cuda":
        try:
            from talea._rotary_cuda import apply
        except ImportError:
            pass
    with ExitStack() as restore:
        if apply is not None:
            for block in blocks:
                rotary = getattr(block.attn, "rotary", None)
                if (type(rotary).__name__ != "RotaryEmbedding" or rotary.training
                        or rotary.dim != 64 or rotary.interleaved or rotary.scale is not None
                        or "f813401638b3fddab09748aec1ad2bf537aa4208" not in type(rotary).__module__):
                    continue
                if "forward" in vars(rotary):
                    restore.callback(setattr, rotary, "forward", rotary.forward)
                else:
                    restore.callback(delattr, rotary, "forward")
                rotary.forward = _rotary_forward(rotary, apply, counter)
        yield counter
