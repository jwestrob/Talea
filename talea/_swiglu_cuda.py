"""Gather the backend's exact fp16 SiLU output and multiply by the up projection."""
from functools import lru_cache, partial

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


@triton.jit
def _silu_words(x, table, EXCEPTIONS: tl.constexpr, REPAIRS: tl.constexpr):
    word = x.to(tl.uint16, bitcast=True).to(tl.int32)
    v = x.to(tl.float32)
    result = (v / (1.0 + tl.exp(-v))).to(tl.float16).to(tl.uint16, bitcast=True).to(tl.int32)
    for index in tl.static_range(len(EXCEPTIONS)):
        result = tl.where(word == EXCEPTIONS[index], REPAIRS[index], result)
    special = (word & 0x7fff) >= 0x7c00
    original = tl.load(table + word, mask=special, other=0.).to(tl.uint16, bitcast=True).to(tl.int32)
    return tl.where(special, original, result).to(tl.uint16).to(tl.float16, bitcast=True)


@triton.jit
def _probe_silu(table, out, EXCEPTIONS: tl.constexpr, REPAIRS: tl.constexpr, BLOCK: tl.constexpr):
    word = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = word.to(tl.uint16).to(tl.float16, bitcast=True)
    tl.store(out + word, _silu_words(x, table, EXCEPTIONS, REPAIRS))


@triton.jit
def _gate_arithmetic(x, table, out, size, WIDTH: tl.constexpr, BLOCK: tl.constexpr,
                     EXCEPTIONS: tl.constexpr, REPAIRS: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < size
    j = (i // WIDTH) * (2 * WIDTH) + i % WIDTH
    gate = tl.load(x + j, valid, 0.)
    value = tl.load(x + j + WIDTH, valid, 0.)
    activation = _silu_words(gate, table, EXCEPTIONS, REPAIRS)
    tl.store(out + i, activation.to(tl.float32) * value.to(tl.float32), valid)


def _arithmetic(x, table, *, exceptions, repairs):
    width = x.shape[-1] // 2
    out = torch.empty((*x.shape[:-1], width), dtype=x.dtype, device=x.device)
    with torch.cuda.device(x.device):
        _gate_arithmetic[(triton.cdiv(out.numel(), 256),)](
            x, table, out, out.numel(), WIDTH=width, BLOCK=256, EXCEPTIONS=exceptions, REPAIRS=repairs,
        )
    return out


@lru_cache(maxsize=8)
def calibrated_arithmetic(table):
    """Calibrate once against all 65,536 native SiLU half results on this device."""
    trial = torch.empty_like(table)
    _probe_silu[(256,)](table, trial, (), (), BLOCK=256)
    target_bits = table.view(torch.int16)
    mismatches = torch.nonzero(trial.view(torch.int16) != target_bits).flatten()
    # Numerous per-element comparisons cost more than the ordinary indexed load.
    if len(mismatches) > 8:
        return apply, None
    exceptions = tuple(mismatches.tolist())
    repairs = tuple((target_bits[mismatches].int() & 0xffff).tolist())
    _probe_silu[(256,)](table, trial, exceptions, repairs, BLOCK=256)
    if not torch.equal(trial.view(torch.int16), target_bits):
        return apply, None
    return partial(_arithmetic, exceptions=exceptions, repairs=repairs), len(repairs)
