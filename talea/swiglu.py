"""Read the frozen ESM++ SwiGLU activation from the backend's full fp16 SiLU table."""
from contextlib import ExitStack, contextmanager
from functools import lru_cache

import torch


@lru_cache(maxsize=8)
def _silu_table(device):
    # Index by the input's 16 raw bits. Torch computes every activation with
    # the same device, dtype and operator as the stock ESM++ feed-forward block.
    words = torch.arange(65536, dtype=torch.int32).to(torch.int16).to(device)
    return torch.nn.functional.silu(words.view(torch.float16))


@lru_cache(maxsize=1)
def _metal_kernel():
    return torch.mps.compile_shader("""
#include <metal_stdlib>
using namespace metal;
kernel void swiglu(device const half *x, device const half *table, device half *out,
                   constant uint &width, uint idx [[thread_position_in_grid]]) {
    uint j = 2 * (idx / width) * width + idx % width;
    half a = table[as_type<ushort>(x[j])];
    out[idx] = half(float(a) * float(x[j + width]));
}
""").swiglu


def _metal(x, table):
    width = x.shape[-1] // 2
    y = torch.empty((*x.shape[:-1], width), device=x.device, dtype=x.dtype)
    _metal_kernel()(x, table, y, width, threads=y.numel())
    return y


def _forward(original, apply, counter):
    def forward(x):
        if (torch.is_grad_enabled() or x.dtype != torch.float16 or not x.is_contiguous()
                or not x.numel() or x.storage_offset() or x.shape[-1] % 2):
            return original(x)
        counter[0] += 1
        return apply(x, _silu_table(x.device))
    return forward


@contextmanager
def swiglu_context(blocks, device, *, arithmetic=False):
    counter = [0, None]
    apply = None
    if device.type == "mps" and hasattr(torch.mps, "compile_shader"):
        apply = _metal
    elif device.type == "cuda":
        try:
            from talea._swiglu_cuda import apply
            if arithmetic:
                from talea._swiglu_cuda import calibrated_arithmetic
                with torch.cuda.device(device):
                    apply, counter[1] = calibrated_arithmetic(_silu_table(device))
        except ImportError:
            pass
    with ExitStack() as restore:
        if apply is not None:
            for block in blocks:
                for layer in block.ffn:
                    if (type(layer).__name__ != "SwiGLU" or layer.training
                            or "f813401638b3fddab09748aec1ad2bf537aa4208" not in type(layer).__module__):
                        continue
                    if "forward" in vars(layer):
                        restore.callback(setattr, layer, "forward", layer.forward)
                    else:
                        restore.callback(delattr, layer, "forward")
                    layer.forward = _forward(layer.forward, apply, counter)
        yield counter
