"""Worker entry point for process-based postprocessing.

**This module must stay free of torch and transformers.** Postprocessing is
pure NumPy/SciPy, but the worker functions originally lived in
``talea.inference``, so every spawned worker imported the whole deep-learning
stack: measured cold-import cost 16.06 s for ``talea.inference`` versus 1.13 s
for ``talea.topology``. With a handful of records that overhead swamped the
work entirely (0.27x of serial). Keeping the worker's import graph narrow is
the difference between the process pool being a speedup and a slowdown.

Workers are created with a ``forkserver`` context that preloads this module, so
the import is paid once by the forkserver rather than per worker. The
forkserver itself is launched as a fresh interpreter and therefore never
inherits the parent's CUDA context.
"""

from __future__ import annotations

import os

from talea.topology import process_attention_record

_CONFIGS: dict | None = None


def initialize_worker(configs: dict) -> None:
    """Stash the frozen calibration configs once per worker.

    They are identical for every record, so shipping them per task would
    repickle a few hundred kilobytes per protein for nothing.
    """
    global _CONFIGS
    _CONFIGS = configs
    # Each worker is single-threaded NumPy; letting every one spin up a full
    # BLAS thread pool oversubscribes the node and undoes the parallelism.
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(variable, "1")


def run_record(payload: tuple) -> dict:
    """Postprocess ``(metadata, matrices, window_sizes[, probabilities])``."""
    metadata, matrices, window_sizes, *rest = payload
    if _CONFIGS is None:
        raise RuntimeError("postprocess worker was not initialized")
    return process_attention_record(
        {
            "metadata": metadata,
            "matrices": matrices,
            "window_sizes": window_sizes,
            "residue_structure_probabilities": rest[0] if rest else None,
            **_CONFIGS,
        }
    )
