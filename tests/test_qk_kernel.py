import torch

from talea.qk import _forward, fused_qk_context


def test_cpu_does_not_install_a_cuda_qk_kernel():
    with fused_qk_context([object()], torch.device("cpu")) as count:
        assert count == [0]


def test_forward_falls_back_before_projecting_inputs_with_another_dtype():
    mask = torch.ones(1,3,dtype=torch.bool)
    inputs = torch.zeros(1,3,1152,dtype=torch.float32)
    count = [0]

    def original(x, **options):
        assert x is inputs
        assert options["attention_mask_2d"] is mask
        return "original"

    with torch.inference_mode():
        predict = _forward(None, original, count, None)
        assert predict(inputs, attention_mask_2d=mask) == "original"
    assert count == [0]
