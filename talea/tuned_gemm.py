"""Optional shape-checked cuBLASLt algorithms for two ESM++ linear projections."""
import ctypes
from collections import OrderedDict
from contextlib import ExitStack, contextmanager
from functools import lru_cache
from pathlib import Path

import torch

from talea.qk import _REVISION

_WORKSPACE_BYTES = 32 * 1024**2
# ID, tile, split-K, pipeline stages, reduction scheme, CTA swizzle, custom option.
_FFN = (5, 20, 2, 7, 1, 0, 0)
_QKV_COMPACT = (5, 20, 1, 0, 0, 1, 0)
_QKV_LARGE = (21, 23, 1, 9, 0, 0, 0)


@lru_cache(maxsize=1)
def _library():
    from talea._cuda_library import build_library
    library = ctypes.CDLL(str(build_library(Path(__file__).with_name("_gemm_lt.cpp"),
                                            "gemm-lt-sm86", ("-lcublasLt",))))
    pointer, integer = ctypes.c_void_p, ctypes.c_int
    library.setup.argtypes = [integer, integer, integer, ctypes.c_size_t, ctypes.POINTER(pointer)]
    library.details.argtypes = [pointer, integer, ctypes.POINTER(integer)]
    library.details.restype = None
    library.execute.argtypes = [pointer, integer] + [pointer] * 5
    library.close_plan.argtypes = [pointer]
    library.close_plan.restype = None
    return library


class _Projections:
    def __init__(self, verify):
        self.verify = verify
        self.plans = OrderedDict()
        self.workspaces = {}
        self.rejected_shapes = set()
        self.stats = dict(calls=0, checked=0, mismatches=0, disabled_shapes=0, enabled_shapes=0, late_mismatch_shapes=[])
        self.library = _library()

    def _new_plan(self, rows, inner, columns):
        plan = ctypes.c_void_p()
        count = self.library.setup(rows, inner, columns, _WORKSPACE_BYTES, ctypes.byref(plan))
        if inner == 3072:
            preferences = (_FFN,)
        elif rows >= 32_768:
            preferences = (_QKV_LARGE, _QKV_COMPACT)
        else:
            preferences = (_QKV_COMPACT, _QKV_LARGE)
        available = {}
        for index in range(max(0, count)):
            description = (ctypes.c_int*7)()
            self.library.details(plan, index, description)
            available[tuple(description)] = index
        for choice in preferences:
            if choice in available:
                return plan, available[choice]
        if plan.value:
            self.library.close_plan(plan)
        return None

    def apply(self, original, weight, x):
        if (torch.is_grad_enabled() or x.dtype != torch.float16 or x.ndim != 3
                or not x.is_contiguous() or not weight.is_contiguous()
                or x.data_ptr() % 256 or weight.data_ptr() % 256):
            return original(x)
        batch, tokens, inner = x.shape
        rows = batch*tokens
        columns = weight.shape[0]
        if not rows or rows > 2**31-1:
            return original(x)
        shape = (rows, inner, columns)
        if shape in self.rejected_shapes:
            return original(x)
        new = shape not in self.plans
        if new:
            # Production batches are sorted by length; avoid retaining a separate
            # vendor descriptor and handle for every geometry across an entire shard.
            if len(self.plans) >= 32:
                _, old = self.plans.popitem(last=False)
                if old is not None:
                    self.library.close_plan(old[0])
            self.plans[shape] = self._new_plan(*shape)
            if self.plans[shape] is None:
                self.stats["disabled_shapes"] += 1
        self.plans.move_to_end(shape)
        entry = self.plans[shape]
        if entry is None:
            return original(x)
        plan, index = entry
        stream = torch.cuda.current_stream(x.device)
        if stream not in self.workspaces:
            self.workspaces[stream] = torch.empty(_WORKSPACE_BYTES,dtype=torch.uint8,device=x.device)
        workspace = self.workspaces[stream]
        output = torch.empty((batch,tokens,columns),dtype=x.dtype,device=x.device)
        code = self.library.execute(plan,index,weight.data_ptr(),x.data_ptr(),output.data_ptr(),
                                    workspace.data_ptr(),stream.cuda_stream)
        if code:
            raise RuntimeError(f"tuned cuBLASLt projection returned status {code}")
        if new or self.verify:
            reference = original(x)
            self.stats["checked"] += 1
            if not torch.equal(reference.view(torch.int16),output.view(torch.int16)):
                # Keep this shape on the original call for the rest of the run.
                self.rejected_shapes.add(shape)
                self.stats["mismatches"] += 1
                if not new:
                    self.stats["late_mismatch_shapes"].append(shape)
                self.stats["disabled_shapes"] += 1
                self.plans[shape] = None
                self.library.close_plan(plan)
                return reference
            if new:
                self.stats["enabled_shapes"] += 1
        self.stats["calls"] += 1
        return output

    def close(self):
        for entry in self.plans.values():
            if entry is not None:
                self.library.close_plan(entry[0])
        self.plans.clear()
        self.workspaces.clear()


@contextmanager
def tuned_gemm_context(blocks,device,mode,verify=False):
    if mode not in ("ffn","qkv","both"):
        raise ValueError("tuned_gemm must select ffn, qkv or both")
    if (device.type != "cuda" or torch.__version__.split("+")[0] != "2.7.1"
            or torch.cuda.get_device_capability(device)!=(8,6)):
        yield dict(calls=0,checked=0,mismatches=0,disabled_shapes=0,enabled_shapes=0, late_mismatch_shapes=[])
        return
    projections = _Projections(verify)
    with ExitStack() as stack:
        stack.callback(projections.close)
        for block in blocks:
            if _REVISION not in type(block.attn).__module__ or block.training:
                continue
            modules=[]
            if mode in ("ffn","both"):
                modules.append((block.ffn[3], (1152,3072)))
            if mode in ("qkv","both"):
                modules.append((block.attn.layernorm_qkv[1], (3456,1152)))
            for module, shape in modules:
                if (type(module) is not torch.nn.Linear or module.bias is not None
                        or tuple(module.weight.shape)!=shape or module.weight.dtype != torch.float16):
                    continue
                original=module.forward
                if "forward" in vars(module):
                    stack.callback(setattr,module,"forward",original)
                else:
                    stack.callback(delattr,module,"forward")
                def linear(x, base=original, w=module.weight):
                    return projections.apply(base,w,x)
                module.forward=linear
        yield projections.stats
