from types import SimpleNamespace

import pytest
import torch

from talea.capture import selective_capture


@pytest.mark.parametrize("mode", ["manual", "sdpa"])
def test_manual_capture_omits_only_an_all_valid_batch_mask(mode):
    calls = []

    def manual(query, key, value, mask, output_s_max):
        calls.append(mask)
        scores = query @ key.transpose(-2, -1)
        if mask is not None:
            scores = scores.masked_fill(mask.logical_not(), float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        return weights @ value, weights, None

    attention = SimpleNamespace(_attn=None, _manual_attn=manual)
    block = SimpleNamespace(attn=attention)
    model = SimpleNamespace(transformer=SimpleNamespace(blocks=[block]))
    x = torch.arange(6, dtype=torch.float32).reshape(1, 1, 3, 2)
    mask = torch.ones(1, 1, 1, 3, dtype=torch.bool)
    with selective_capture(model, (0,), (), mode=mode) as capture:
        for expected_mask in (mask, None, mask):
            if expected_mask is mask and calls:
                mask[..., -1] = False
            capture.all_tokens_valid = expected_mask is None
            expected, weights, _ = manual(x, x, x, mask, False)
            actual, _, _ = attention._attn(x, x, x, attention_mask_4d=mask)
            assert calls[-1] is expected_mask
            assert torch.equal(actual, expected)
            assert torch.equal(capture.attentions[0], weights)
