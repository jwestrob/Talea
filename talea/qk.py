"""Fuse ESM++'s FP16 Q/K layer normalization and rotation on validated Ampere hardware."""
import ctypes
from contextlib import ExitStack, contextmanager
from functools import lru_cache
from pathlib import Path

import torch

_REVISION = "f813401638b3fddab09748aec1ad2bf537aa4208"


@lru_cache(maxsize=1)
def _kernel():
    """Build once and atomically share the compiled library between scoring processes."""
    from talea._cuda_library import build_library
    library = build_library(Path(__file__).with_name("_qk_rope.cu"), "qk-sm86")
    call = ctypes.CDLL(str(library)).run_qk
    pointer = ctypes.c_void_p
    call.argtypes = [pointer]*7 + [ctypes.c_int,ctypes.c_int,ctypes.c_float,pointer]
    call.restype = ctypes.c_int
    return call


def _forward(attention, original, calls, kernel):
    def forward(x, attention_mask_2d=None, attention_mask_4d=None, flex_block_mask=None,
                output_attentions=False, output_s_max=False):
        kwargs = dict(attention_mask_2d=attention_mask_2d, attention_mask_4d=attention_mask_4d,
                      flex_block_mask=flex_block_mask, output_attentions=output_attentions,
                      output_s_max=output_s_max)
        if x.ndim != 3 or x.dtype != torch.float16 or torch.is_grad_enabled():
            return original(x, **kwargs)
        packed = attention.layernorm_qkv(x)
        batch, tokens, _ = packed.shape
        rotary = attention.rotary
        rotary._update_cos_sin_cache(tokens, device=x.device, dtype=x.dtype)
        cosine, sine = rotary._cos_cached, rotary._sin_cached
        qweight, kweight = attention.q_ln.weight, attention.k_ln.weight
        if not (packed.is_contiguous() and packed.shape[-1] == 3456
                and cosine.shape[-1] == 32 and sine.shape == cosine.shape
                and all(t.is_contiguous() and t.dtype == packed.dtype and t.data_ptr()%8 == 0
                        for t in (packed, cosine, sine, qweight, kweight))):
            return original(x, **kwargs)
        q = torch.empty((batch, tokens, 18, 64), dtype=x.dtype, device=x.device)
        k = torch.empty_like(q)
        pointers = [t.data_ptr() for t in (packed,qweight,kweight,cosine,sine,q,k)]
        status = kernel(*pointers, batch*tokens, tokens, attention.q_ln.eps,
                        torch.cuda.current_stream(x.device).cuda_stream)
        if status:
            raise RuntimeError(f"fused Q/K kernel returned CUDA status {status}")
        context, weights, s_max = attention._attn(q.transpose(1,2), k.transpose(1,2),
                                                  attention.reshaper(packed[...,2304:]), **kwargs)
        calls[0] += 1
        return attention.out_proj(context), weights, s_max
    return forward


@contextmanager
def fused_qk_context(blocks, device):
    calls = [0]
    supported = (device.type == "cuda" and torch.version.cuda is not None
                 and torch.__version__.split("+")[0] == "2.7.1"
                 and torch.cuda.get_device_capability(device) == (8,6))
    with ExitStack() as stack:
        if supported:
            for block in blocks:
                a = block.attn
                if (_REVISION not in type(a).__module__ or a.training or a.d_model != 1152 or a.n_heads != 18
                        or a.rotary.interleaved or a.rotary.scale is not None or a.rotary.dim != 64
                        or a.q_ln.bias is not None or a.k_ln.bias is not None or a.q_ln.eps != a.k_ln.eps
                        or a.q_ln.weight.dtype != torch.float16):
                    continue
                original = a.forward
                if "forward" in vars(a):
                    stack.callback(setattr,a,"forward",original)
                else:
                    stack.callback(delattr,a,"forward")
                a.forward = _forward(a,original,calls,_kernel())
        yield calls
