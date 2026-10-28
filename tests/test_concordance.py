"""Concordance of the optimized implementations against the originals.

Every optimization in this branch is paired with an assertion here. Layers:

  L1  function level  -- optimized vs in-tree reference, randomized inputs
  L2  record level    -- process_attention_record, every emitted numeric field
  L3  tensor level    -- apc_head_mean crop/cast reordering
  L4  capture level   -- index resolution against the model's own tuples (GPU/
                         CPU model required; skipped when unavailable)

L4 is the only layer needing model weights; L1-L3 run anywhere.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from talea.attention import (
    DIAGONAL_FAST_PATH_MAXIMUM,
    _diagonal_mean_profile_reference,
    diagonal_mean_profile,
)
from talea.capture import resolve_hidden_state_sources
from talea.inference import apc_head_mean, near_diagonal_mask
from talea.physical import (
    _best_harmonic_comb_reference,
    aggregate_spectrum,
    best_harmonic_comb,
)
from talea.production import roundtrip_attention_storage
from talea.topology import process_attention_record, widths_from_config
from talea.wedge import (
    DiagonalWindowSums,
    _wedge_start_scores_reference,
    best_wedge_interval,
    fuse_diagonal_tracks,
    robust_diagonal_tracks,
    wedge_start_scores,
)

ASSETS = Path(__file__).resolve().parents[1] / "talea" / "assets"


# ------------------------------------------------------------------ fixtures
def synthetic_attention(length: int, period: float, seed: int) -> np.ndarray:
    """Nonnegative, symmetric, zero-near-diagonal matrix with periodic bands."""
    rng = np.random.default_rng(seed)
    index = np.arange(length)
    separation = np.abs(index[:, None] - index[None, :]).astype(np.float64)
    banded = np.exp(-separation / 200.0) * (
        0.5 + 0.5 * np.cos(2 * np.pi * separation / period)
    )
    noise = rng.gamma(shape=1.2, scale=0.05, size=(length, length))
    matrix = banded + 0.5 * (noise + noise.T)
    matrix[separation < 6] = 0.0
    np.fill_diagonal(matrix, 0.0)
    return matrix.astype(np.float16).astype(np.float32)


def build_task(length: int, seed: int = 11) -> dict:
    return {
        "metadata": {
            "record_id": f"synthetic_{length}",
            "sequence_sha256": "0" * 64,
            "length": length,
        },
        "matrices": {
            13: synthetic_attention(length, 31.5, seed),
            35: synthetic_attention(length, 31.5, seed + 1),
        },
        "window_sizes": (64, 96, 128, 192, 256, 384, 512),
        "protein_calibration": json.loads(
            (ASSETS / "protein_calibration.json").read_text()
        ),
        "local_calibration": json.loads((ASSETS / "local_calibration.json").read_text()),
        "wedge_config": json.loads((ASSETS / "wedge_config.json").read_text()),
        "operational_config": json.loads((ASSETS / "topology_config.json").read_text()),
    }


def flatten_numeric(value, prefix=""):
    out = {}
    if isinstance(value, dict):
        for key, item in value.items():
            out.update(flatten_numeric(item, f"{prefix}.{key}"))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            out.update(flatten_numeric(item, f"{prefix}[{index}]"))
    elif isinstance(value, bool):
        out[prefix] = float(value)
    elif isinstance(value, (int, float)):
        out[prefix] = float(value)
    return out


# ------------------------------------------------------- L1: diagonal profile
@pytest.mark.parametrize("length", [2, 3, 8, 64, 96, 128, 192, 256, 384, 512, 1024])
def test_diagonal_profile_is_bit_exact_on_the_fast_path(length):
    rng = np.random.default_rng(length)
    matrix = rng.gamma(1.2, 0.05, size=(length, length)).astype(np.float32)
    matrix = matrix + matrix.T
    reference = _diagonal_mean_profile_reference(matrix)
    optimized = diagonal_mean_profile(matrix)
    assert optimized.shape == reference.shape
    assert np.array_equal(optimized, reference), "fast path must be bit-exact"


def test_diagonal_profile_delegates_above_the_fast_path_threshold():
    length = DIAGONAL_FAST_PATH_MAXIMUM + 1
    rng = np.random.default_rng(0)
    matrix = rng.gamma(1.2, 0.05, size=(length, length)).astype(np.float32)
    matrix = matrix + matrix.T
    assert np.array_equal(
        diagonal_mean_profile(matrix), _diagonal_mean_profile_reference(matrix)
    )


def test_diagonal_profile_matches_reference_on_degenerate_shapes():
    for length in (0, 1):
        matrix = np.zeros((length, length), dtype=np.float32)
        reference = _diagonal_mean_profile_reference(matrix)
        optimized = diagonal_mean_profile(matrix)
        assert optimized.shape == reference.shape
        assert optimized.dtype == reference.dtype


# --------------------------------------------------------- L1: harmonic comb
def _spectra(count: int, seed: int = 3):
    rng = np.random.default_rng(seed)
    for _ in range(count):
        length = int(rng.integers(16, 1200))
        channels = int(rng.integers(1, 6))
        signal = rng.standard_normal((length, channels))
        if rng.random() < 0.7:
            period = float(rng.uniform(4, 90))
            signal = signal + rng.uniform(0.3, 4.0) * np.sin(
                2 * np.pi * np.arange(length)[:, None] / period
            )
        spectral_z, _ = aggregate_spectrum(signal)
        yield spectral_z, length


def test_harmonic_comb_selects_the_same_fundamental_everywhere():
    """The discrete outputs must agree exactly, including tie-breaks."""
    mismatches = []
    worst_score_delta = 0.0
    for spectral_z, length in _spectra(600):
        reference = _best_harmonic_comb_reference(spectral_z, length)
        optimized = best_harmonic_comb(spectral_z, length)
        if (
            reference.frequency_bin != optimized.frequency_bin
            or reference.n_harmonics != optimized.n_harmonics
            or reference.n_harmonics_tested != optimized.n_harmonics_tested
            or reference.period != optimized.period
        ):
            mismatches.append((length, reference, optimized))
        worst_score_delta = max(
            worst_score_delta, abs(reference.score - optimized.score)
        )
    assert not mismatches, f"{len(mismatches)} discrete mismatches: {mismatches[:3]}"
    # Exact, not approximate: the per-fundamental reduction keeps np.dot so the
    # score is bit-identical. A nonzero delta here means a vectorized reduction
    # crept back in and tie-breaks can flip. See the note in best_harmonic_comb.
    assert worst_score_delta == 0.0, worst_score_delta


def test_harmonic_comb_is_exact_on_tie_prone_spectra():
    """Real APC spectra are spiky and quantized, so exact ties are common.

    Smooth Gaussian test signals almost never tie; these deliberately do, which
    is the regime where a last-bit difference changes the selected period.
    """
    rng = np.random.default_rng(5)
    checked = 0
    for _ in range(300):
        length = int(rng.integers(64, 900))
        matrix = synthetic_attention(length, float(rng.uniform(8, 60)), int(rng.integers(0, 10_000)))
        profile = _diagonal_mean_profile_reference(matrix)
        # Quantize hard, so many bins carry identical values.
        spectral_z, _ = aggregate_spectrum(np.round(profile[:, None], 3))
        reference = _best_harmonic_comb_reference(spectral_z, len(profile))
        optimized = best_harmonic_comb(spectral_z, len(profile))
        assert reference == optimized, (reference, optimized)
        checked += 1
    assert checked == 300


def test_harmonic_comb_harmonic_z_agrees():
    for spectral_z, length in _spectra(200, seed=17):
        reference = _best_harmonic_comb_reference(spectral_z, length)
        optimized = best_harmonic_comb(spectral_z, length)
        assert len(reference.harmonic_z) == len(optimized.harmonic_z)
        for a, b in zip(reference.harmonic_z, optimized.harmonic_z):
            assert a == b, "tolerance-window maxima must be exact"


@pytest.mark.parametrize("tolerance", [0, 1, 2, 3])
def test_harmonic_comb_matches_across_tolerance_widths(tolerance):
    for spectral_z, length in _spectra(60, seed=100 + tolerance):
        reference = _best_harmonic_comb_reference(
            spectral_z, length, tolerance_bins=tolerance
        )
        optimized = best_harmonic_comb(spectral_z, length, tolerance_bins=tolerance)
        assert reference.frequency_bin == optimized.frequency_bin
        assert reference.harmonic_z == optimized.harmonic_z


def test_harmonic_comb_degenerate_spectra_return_the_empty_call():
    for spectral_z, length in ((np.zeros(1), 10), (np.zeros(3), 4)):
        reference = _best_harmonic_comb_reference(spectral_z, length)
        optimized = best_harmonic_comb(spectral_z, length)
        assert reference == optimized


# ------------------------------------------------------------ L1: wedge search
def _wedge_tracks(length: int, period: float, seed: int) -> list[np.ndarray]:
    return fuse_diagonal_tracks(
        [
            robust_diagonal_tracks(synthetic_attention(length, period, seed + offset), 256)
            for offset in (0, 1)
        ]
    )


@pytest.mark.parametrize("length,period", [(96, 11.0), (150, 7.5), (333, 31.5), (700, 42.0)])
def test_wedge_scores_with_shared_sums_are_bit_exact(length, period):
    """One search shares interval sums across templates; every score must be unchanged."""
    tracks = _wedge_tracks(length, period, seed=length)
    widths = widths_from_config(json.loads((ASSETS / "wedge_config.json").read_text()))
    shared = DiagonalWindowSums(tracks, length)
    for template_period in (5.0, 7.5, period / 2, period, 2 * period, 64.0):
        for requested_width in widths:
            width = min(requested_width, length)
            for controls in (True, False):
                reference = _wedge_start_scores_reference(
                    tracks, length, template_period, width, interharmonic_controls=controls
                )
                optimized = wedge_start_scores(
                    tracks,
                    length,
                    template_period,
                    width,
                    interharmonic_controls=controls,
                    diagonal_sums=shared,
                )
                assert np.array_equal(optimized, reference), (template_period, width, controls)


def test_best_wedge_interval_matches_a_reference_template_loop():
    length = 420
    tracks = _wedge_tracks(length, 23.5, seed=4)
    widths = widths_from_config(json.loads((ASSETS / "wedge_config.json").read_text()))
    periods = (11.5, 23.5, 47.0, 70.5)
    best = None
    for period in periods:
        for requested_width in widths:
            width = min(requested_width, length)
            scores = _wedge_start_scores_reference(tracks, length, period, width)
            start = int(np.nanargmax(scores))
            if best is None or float(scores[start]) > best[3]:
                best = (start, start + width, period, float(scores[start]))
    call = best_wedge_interval(tracks, length, periods, widths)
    assert (call["start"], call["end"], call["period"], call["score"]) == best


def test_shared_sums_refuse_other_tracks():
    tracks = _wedge_tracks(128, 16.0, seed=1)
    other = [np.array(track, copy=True) for track in tracks]
    with pytest.raises(ValueError, match="different tracks"):
        wedge_start_scores(
            other, 128, 16.0, 96, diagonal_sums=DiagonalWindowSums(tracks, 128)
        )


# --------------------------------------------------------------- L2: records
@pytest.mark.parametrize("length", [220, 400, 700])
def test_process_attention_record_fields_agree(length):
    """Compare against values computed with the reference kernels patched in."""
    import talea.attention as attention_module
    import talea.physical as physical_module

    optimized = process_attention_record(build_task(length))

    fast_diagonal = attention_module.score_attention_window.__globals__[
        "diagonal_mean_profile"
    ]
    fast_comb = physical_module.score_signals.__globals__["best_harmonic_comb"]
    fast_wedge = best_wedge_interval.__globals__["wedge_start_scores"]
    attention_module.score_attention_window.__globals__["diagonal_mean_profile"] = (
        _diagonal_mean_profile_reference
    )
    physical_module.score_signals.__globals__["best_harmonic_comb"] = (
        _best_harmonic_comb_reference
    )
    best_wedge_interval.__globals__["wedge_start_scores"] = (
        lambda *args, diagonal_sums=None, **kwargs: _wedge_start_scores_reference(
            *args, **kwargs
        )
    )
    try:
        reference = process_attention_record(build_task(length))
    finally:
        attention_module.score_attention_window.__globals__[
            "diagonal_mean_profile"
        ] = fast_diagonal
        physical_module.score_signals.__globals__["best_harmonic_comb"] = fast_comb
        best_wedge_interval.__globals__["wedge_start_scores"] = fast_wedge

    assert reference["status"] == optimized["status"]
    left, right = flatten_numeric(reference), flatten_numeric(optimized)
    assert set(left) == set(right), "emitted field set changed"
    worst_key, worst = "", 0.0
    for key in left:
        a, b = left[key], right[key]
        if not (math.isfinite(a) and math.isfinite(b)):
            assert a == b or (math.isnan(a) and math.isnan(b)), key
            continue
        delta = abs(a - b) / max(abs(a), 1e-12)
        if delta > worst:
            worst, worst_key = delta, key
    # Every replaced kernel is bit-exact, so the whole record must be too.
    assert worst == 0.0, f"{worst_key} drifted by {worst:.3e}"


# ---------------------------------------------------------------- L3: tensors
def _reference_apc_head_mean(layer_attention, indices, sequence_length, separation):
    """The original ordering: symmetrize the padded map, then crop."""
    attention = 0.5 * (layer_attention + layer_attention.transpose(-1, -2))
    attention = attention.index_select(1, indices).index_select(2, indices)
    row = attention.sum(dim=-1, keepdim=True)
    column = attention.sum(dim=-2, keepdim=True)
    total = attention.sum(dim=(-1, -2), keepdim=True) + 1e-12
    attention = torch.relu(attention - (row * column) / total).mean(dim=0)
    if separation > 0:
        positions = torch.arange(sequence_length, device=attention.device)
        gap = (positions[:, None] - positions[None, :]).abs()
        attention = attention.masked_fill(gap < separation, 0.0)
    attention.fill_diagonal_(0.0)
    return attention.to("cpu", dtype=torch.float32).numpy()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("length,padded", [(40, 46), (128, 130), (77, 96)])
def test_apc_head_mean_crop_reorder_is_exact(dtype, length, padded):
    torch.manual_seed(length)
    heads = 4
    raw = torch.rand(heads, padded, padded, dtype=torch.float32).to(dtype)
    indices = torch.arange(1, length + 1)
    reference = _reference_apc_head_mean(raw, indices, length, 6)
    optimized = apc_head_mean(raw, indices, length, 6, storage_dtype=None)
    assert np.array_equal(reference, optimized), "crop reorder must be bit-exact"


@pytest.mark.parametrize("length,padded", [(40, 46), (128, 130)])
def test_device_side_fp16_cast_matches_host_roundtrip(length, padded):
    """Casting to fp16 before transfer must equal the host-side round trip."""
    torch.manual_seed(length + 1)
    raw = torch.rand(4, padded, padded, dtype=torch.float32)
    indices = torch.arange(1, length + 1)
    host = roundtrip_attention_storage(
        apc_head_mean(raw, indices, length, 6, storage_dtype=None), "fp16"
    )
    device_cast = roundtrip_attention_storage(
        apc_head_mean(raw, indices, length, 6, storage_dtype=torch.float16), "fp16"
    )
    assert np.array_equal(host, device_cast)


def test_near_diagonal_mask_matches_the_inline_construction():
    for length in (5, 64, 257):
        positions = torch.arange(length)
        expected = (positions[:, None] - positions[None, :]).abs() < 6
        assert torch.equal(near_diagonal_mask(length, 6, torch.device("cpu")), expected)


def test_near_diagonal_mask_is_cached_not_rebuilt():
    first = near_diagonal_mask(32, 6, torch.device("cpu"))
    second = near_diagonal_mask(32, 6, torch.device("cpu"))
    assert first is second


# ---------------------------------------------------------------- L4: capture
def test_hidden_state_source_resolution_reproduces_the_frozen_offset():
    """layer + 1 indexing: layer 13 -> block 14, layer 35 -> last_hidden_state."""
    sources = resolve_hidden_state_sources((13, 35), n_blocks=36)
    assert sources == {13: 14, 35: None}


def test_hidden_state_source_resolution_rejects_out_of_range():
    with pytest.raises(ValueError):
        resolve_hidden_state_sources((36,), n_blocks=36)


# ------------------------------------------------------- L6: postprocess pool
def test_postprocess_worker_module_stays_free_of_torch():
    """The worker's import graph is a performance contract, not a detail.

    talea.inference costs 16.06 s to import cold (torch + transformers) against
    1.13 s for talea.topology. When the worker entry point lived in
    talea.inference, every spawned worker paid that and the process pool ran at
    0.27x of serial. Importing torch here would silently reintroduce it.
    """
    source = (
        Path(__file__).resolve().parents[1] / "talea" / "postprocess_pool.py"
    ).read_text()
    for banned in ("import torch", "import transformers", "from talea.inference"):
        assert banned not in source, f"{banned} would re-inflate worker startup"
    for banned in ("talea.production", "talea.model", "talea.artifacts"):
        assert banned not in source, f"{banned} transitively imports torch"


def test_postprocess_pool_context_never_preloads_main():
    """Preloading __main__ is multiprocessing's default and breaks REPL callers."""
    from talea.inference import postprocess_pool_context

    context = postprocess_pool_context()
    assert context.get_start_method() in ("forkserver", "spawn")
    assert "fork" != context.get_start_method(), (
        "plain fork after CUDA initialization is unsafe"
    )


def test_postprocess_worker_requires_initialization():
    import talea.postprocess_pool as pool_module

    saved = pool_module._CONFIGS
    pool_module._CONFIGS = None
    try:
        with pytest.raises(RuntimeError, match="not initialized"):
            pool_module.run_record(({}, {}, ()))
    finally:
        pool_module._CONFIGS = saved


def test_worker_entry_point_reproduces_direct_call():
    """run_record must be arithmetically identical to process_attention_record."""
    import talea.postprocess_pool as pool_module

    task = build_task(300, seed=77)
    direct = process_attention_record(task)
    configs = {
        key: task[key]
        for key in (
            "protein_calibration",
            "local_calibration",
            "wedge_config",
            "operational_config",
        )
    }
    saved = pool_module._CONFIGS
    pool_module.initialize_worker(configs)
    try:
        viaworker = pool_module.run_record(
            (task["metadata"], task["matrices"], task["window_sizes"])
        )
    finally:
        pool_module._CONFIGS = saved
    assert direct == viaworker
