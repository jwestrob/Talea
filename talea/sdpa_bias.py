"""Reuse the additive padding bias across ESM's memory-efficient SDPA layers."""
from contextlib import ExitStack, contextmanager

import torch


def _padding_bias(mask, dtype):
    # PyTorch's efficient CUDA backend requires an eight-element row alignment.
    # Keep the original key length while backing each row with aligned storage.
    width = mask.shape[-1]
    aligned = (width + 7) // 8 * 8
    bias = torch.zeros((*mask.shape[:-1], aligned), device=mask.device, dtype=dtype)[..., :width]
    return bias.masked_fill_(~mask, float("-inf"))


class _BatchBias:
    def __init__(self):
        self.mask = None
        self.dtype = None
        self.bias = None
        self.calls = 0
        self.batches = 0

    def for_query(self, q, k, v, mask, scale):
        if mask is None or mask.dtype != torch.bool or q.device.type != "cuda" or q.dtype != torch.float16:
            return mask
        if self.mask is not mask or self.dtype != q.dtype:
            self.mask, self.dtype, self.bias = mask, q.dtype, None
            choose = getattr(torch.ops.aten, "_fused_sdp_choice", None)
            if choose is None:
                return mask
            efficient = torch.backends.cuda.SDPBackend.EFFICIENT_ATTENTION.value
            # Other backends can use different mask sentinels or math.
            if choose(q, k, v, mask, scale=scale) == efficient:
                bias = _padding_bias(mask, q.dtype)
                if choose(q, k, v, bias, scale=scale) == efficient:
                    self.bias = bias
                    self.batches += 1
        if self.bias is None:
            return mask
        self.calls += 1
        return self.bias


@contextmanager
def cached_padding_bias(blocks):
    cache = _BatchBias()

    def wrap(original, scale):
        def sdpa(q, k, v, attention_mask_4d=None):
            return original(q, k, v, cache.for_query(q, k, v, attention_mask_4d, scale))
        return sdpa

    with ExitStack() as stack:
        for block in blocks:
            attention = block.attn
            if hasattr(attention, "_sdpa_attn"):
                original = attention._sdpa_attn
                if "_sdpa_attn" in vars(attention):
                    stack.callback(setattr, attention, "_sdpa_attn", original)
                else:
                    stack.callback(delattr, attention, "_sdpa_attn")
                attention._sdpa_attn = wrap(original, attention.scale)
        yield cache
