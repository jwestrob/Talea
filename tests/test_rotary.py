from types import SimpleNamespace

import pytest
import torch

from talea.rotary import _rotary_forward, fused_rotary_context


def test_rotary_fallback_preserves_callable_for_unsupported_layout():
    q = torch.randn(1, 4, 18, 64)
    rotary = SimpleNamespace(forward=lambda q, k: (q, k))
    count = [0]
    forward = _rotary_forward(rotary, lambda *args: pytest.fail("unsupported input"), count)
    with torch.inference_mode():
        q2, k2 = forward(q, q)
        assert q2 is q and k2 is q
        x = q.half().transpose(0, 1).transpose(1, 2)
        assert forward(x, x)[0] is x
    assert count == [0]


@pytest.mark.parametrize("offset", [0, 2])
def test_rotary_uses_original_tables_and_tracks_calls(offset):
    q = torch.randn(1, 4, 18, 64).half()
    c = torch.randn(6, 32).half()
    s = torch.randn(6, 32).half()
    table = lambda x: x[offset:offset+4]
    rotary = SimpleNamespace(_cos_cached=table(c), _sin_cached=table(s),
                             _update_cos_sin_cache=lambda *args, **kwargs: None,
                             forward=lambda q, k: (q, k))
    calls = [0]
    def apply(q, k, cos, sin):
        assert cos is rotary._cos_cached and sin is rotary._sin_cached
        return q, k
    with torch.inference_mode():
        assert _rotary_forward(rotary, apply, calls)(q, q)[0] is q
    assert calls == [int(offset == 0)]


def test_cpu_context_leaves_modules_alone():
    block = SimpleNamespace(attn=SimpleNamespace())
    with fused_rotary_context([block], torch.device("cpu")) as counter:
        assert counter == [0]
