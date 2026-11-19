import numpy as np
import pytest
import torch

from talea.inference import _apc_fusion, _apc_head_mean_tensor, _equal_length_apc


def half(x):
    return np.asarray(x, dtype=np.float16).astype(np.float32)


class NumpyFusion:
    """Independent fp32 scalar arithmetic rounded at each fp16 operator boundary."""
    def __init__(self, full):
        self.full = full

    def symmetrize(self, attention, start, length):
        x = attention.numpy()[..., start:start+length, start:start+length].astype(np.float32)
        summed = half(x + x.swapaxes(-1, -2))
        return torch.from_numpy(np.ascontiguousarray(0.5 * summed, dtype=np.float16))

    def correct(self, x, rows, cols, totals):
        numerator = half(rows.numpy().astype(np.float32) * cols.numpy().astype(np.float32))
        ratio = half(numerator / totals.numpy().astype(np.float32))
        difference = half(x.numpy().astype(np.float32) - ratio)
        clamped = np.where(difference < 0, 0., difference).astype(np.float16)
        return torch.from_numpy(clamped)


def rounded_inputs(batch, length):
    gen = torch.Generator().manual_seed(length)
    x = torch.rand(batch, 18, length + 5, length + 5, generator=gen).half() / length
    # Subnormals, midpoints, asymmetric pairs and alternating exponents.
    x[..., 2, 3] = 2. ** -24
    x[..., 3, 2] = 2. ** -23
    x[..., 2, 4] = 1.
    x[..., 4, 2] = 2. ** -11
    return x


def same_bits(a, b):
    assert torch.equal(a.view(torch.int16), b.view(torch.int16))


@pytest.mark.parametrize("full", [True, False])
def test_cpu_fusion_reference_matches_scalar_and_grouped(full):
    mode = NumpyFusion(full)
    assert _apc_fusion(torch.device("cpu"), torch.float16, "full") is None
    for n in (7, 31, 32, 33, 65):
        x = rounded_inputs(3, n)
        span = slice(2, 2+n)
        ix = torch.arange(2, 2+n)
        baseline = []
        for item in x:
            target = _apc_head_mean_tensor(item, ix, n, 6, torch.float16, span)
            observed = _apc_head_mean_tensor(item, ix, n, 6, torch.float16, span, mode)
            same_bits(target, observed)
            baseline.append(target)
        actual = _equal_length_apc(x, n, 6, torch.float16, span, mode)
        same_bits(actual, torch.stack(baseline))


def test_pointwise_roundings_change_real_fp16_apc_values():
    x = rounded_inputs(1, 65)[0]
    sym = NumpyFusion(True).symmetrize(x, 2, 65)
    r, c = sym.sum(-1, keepdim=True), sym.sum(-2, keepdim=True)
    total = sym.sum((-1,-2), keepdim=True) + 1e-12
    corrected = torch.relu(sym - (r * c) / total)
    same_bits(corrected, NumpyFusion(True).correct(sym, r, c, total))
    unrounded = torch.relu(sym.float() - r.float() * c.float() / total.float()).half()
    assert torch.count_nonzero(corrected.view(torch.int16) != unrounded.view(torch.int16)) > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires an A5000")
def test_real_cuda_kernels_match_individual_torch_intermediates():
    device = torch.device("cuda")
    if "A5000" not in torch.cuda.get_device_name(device):
        pytest.skip("requires an A5000")
    mode = _apc_fusion(device, torch.float16, "full")
    assert mode is not None
    for n in (31, 32, 33, 127, 320):
        x = rounded_inputs(2, n).to(device)
        span = slice(2, 2+n)
        crop = x[..., span, span].contiguous()
        expected = 0.5 * (crop + crop.transpose(-1, -2))
        fused = mode.symmetrize(x, span.start, n)
        same_bits(fused, expected)
        rows, cols = expected.sum(-1, keepdim=True), expected.sum(-2, keepdim=True)
        total = expected.sum((-1,-2), keepdim=True) + 1e-12
        same_bits(torch.relu(expected - (rows * cols) / total), mode.correct(fused, rows, cols, total))
