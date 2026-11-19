"""Compile the optional A5000 kernels with a process-safe on-disk cache."""
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import torch


def build_library(source, name, extra_flags=()):
    import fcntl

    version = torch.version.cuda
    candidates = [os.environ.get("CUDACXX")]
    if os.environ.get("CUDA_HOME"):
        candidates.append(str(Path(os.environ["CUDA_HOME"]) / "bin" / "nvcc"))
    candidates.extend([f"/usr/local/cuda-{version}/bin/nvcc", f"/usr/local/cuda-{version.split('.')[0]}/bin/nvcc",
                       shutil.which("nvcc"), "/usr/local/cuda/bin/nvcc"])
    compiler = next((p for p in candidates if p and os.access(p, os.X_OK)), None)
    if compiler is None:
        raise RuntimeError(f"{name} requires nvcc on first use; set CUDACXX to its path")
    host = os.environ.get("CUDAHOSTCXX") or shutil.which("g++-11")
    compiler_version = subprocess.check_output([compiler, "--version"])
    key = hashlib.sha256(source.read_bytes() + compiler_version + repr((compiler, host, extra_flags)).encode()).hexdigest()[:20]
    cache_root = Path(os.environ.get("XDG_CACHE_HOME", Path.home()/".cache"))
    cache = cache_root / "talea" / name / key
    cache.mkdir(parents=True, exist_ok=True)
    library = cache / "kernel.so"
    with (cache / "build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not library.is_file():
            with tempfile.TemporaryDirectory(dir=cache) as tmp:
                output = Path(tmp)/"kernel.so"
                command = [compiler]
                if host:
                    command += ["-ccbin",host]
                command += ["-O3","-arch=sm_86","-shared","-Xcompiler=-fPIC","--cudart=shared",
                            str(source),*extra_flags,"-o",str(output)]
                subprocess.run(command,check=True)
                output.replace(library)
    return library
