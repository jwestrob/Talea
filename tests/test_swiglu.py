import torch
from talea.swiglu import _silu_table, _forward


def test_all_finite_half_activations_match_strided_reference():
    words = torch.arange(65536, dtype=torch.int32).to(torch.int16)
    x = words.view(torch.float16).reshape(16, 4096)
    finite = torch.isfinite(x)
    # Feed the same noncontiguous projection input shape that follows torch.chunk.
    gates = torch.cat((x, torch.ones_like(x)), dim=-1).chunk(2, dim=-1)[0]
    ref = torch.nn.functional.silu(gates)
    actual = _silu_table(x.device)[words.to(torch.int32).bitwise_and(65535)].reshape_as(x)
    assert torch.equal(ref[finite].view(torch.int16), actual[finite].view(torch.int16))


def test_other_dtypes_use_original_forward():
    x = torch.ones(2, 64)
    counter = [0]
    with torch.inference_mode():
        result = _forward(lambda v: v, None, counter)(x)
    assert result is x and counter == [0]
