import ctypes
from types import SimpleNamespace
from unittest.mock import Mock

import torch
import talea.tuned_gemm as tuned


def _aligned(shape):
    words = torch.empty(torch.Size(shape).numel()+127, dtype=torch.float16)
    offset = ((-words.data_ptr()) % 256) // 2
    return words[offset:offset+torch.Size(shape).numel()].reshape(shape).zero_()


def test_tuned_projection_checks_first_output_and_retires_a_late_mismatch(monkeypatch):
    stream = Mock(cuda_stream=0)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: stream)
    weight = _aligned((8,4))
    x = _aligned((2,3,4))
    library = SimpleNamespace(close_plan=Mock(), execute=Mock())
    byte = 0
    def execute(plan,index,w,inp,out,workspace,s):
        ctypes.memset(out,byte,2*2*3*8)
        return 0
    library.execute.side_effect = execute
    monkeypatch.setattr(tuned, "_library", lambda:library)
    monkeypatch.setattr(tuned, "_WORKSPACE_BYTES", 1024)
    projections = tuned._Projections(verify=True)
    monkeypatch.setattr(projections, "_new_plan", lambda *shape:(1,0))
    original = lambda value:torch.nn.functional.linear(value,weight)
    with torch.inference_mode():
        assert torch.equal(projections.apply(original,weight,x),original(x))
        byte = 99
        assert torch.equal(projections.apply(original,weight,x),original(x))
        assert torch.equal(projections.apply(original,weight,x),original(x))
    assert projections.stats["checked"] == 2
    assert projections.stats["calls"] == 1
    assert projections.stats["mismatches"] == 1
    assert projections.stats["late_mismatch_shapes"] == [(6,4,8)]
    assert library.execute.call_count == 2
    library.close_plan.assert_called_once_with(1)
    projections.close()


def test_tuned_projection_only_installs_on_validated_cuda_versions():
    with tuned.tuned_gemm_context([object()],torch.device("cpu"),"both") as stats:
        assert stats["calls"]==0
