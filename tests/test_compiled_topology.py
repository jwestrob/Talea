"""Exact arithmetic checks for the optional postprocess worker kernels."""

import numpy as np
import pytest

pytest.importorskip("numba")
from talea import attention, phase_ssm, unit_registration
from talea._attention_numba import diagonal_prefix_f32, integral_image_f32
from talea._phase_numba import transitions


def test_prefixes_match_numpy_including_signed_zero_and_grid_fallback(monkeypatch):
    rng = np.random.default_rng(682)
    for width in (1, 20, 35):
        values = rng.normal(size=(35, 35)).astype(np.float16).astype(np.float32)
        values[0, 0] = -0.0
        monkeypatch.setattr(attention, "_compiled_diagonal_prefix", None)
        monkeypatch.setattr(unit_registration, "_compiled_integral_image", None)
        expected = attention._exact_diagonal_prefix(values, width)
        image = unit_registration.integral_image(values)
        assert diagonal_prefix_f32(values, width).tobytes() == expected.tobytes()
        assert integral_image_f32(values).tobytes() == image.tobytes()
        values[0, 0] = 1e-19
        assert diagonal_prefix_f32(values, width) is None
        values[0, 0] = np.nan
        assert diagonal_prefix_f32(values, width) is None
        with pytest.raises(ValueError, match="nonfinite"):
            integral_image_f32(values)


def test_banded_phase_transitions_reproduce_ties_and_boundaries(monkeypatch):
    rng = np.random.default_rng(8)
    for n, count, minimum in [(2, 4, 3), (18, 1, 1), (98, 11, 8)]:
        for votes in [np.zeros(n, complex), rng.normal(size=n) + 1j * rng.normal(size=n)]:
            emissions = np.zeros((count, n))
            tables = phase_ssm.unit_tables(votes, emissions, minimum, 0.2)
            for step in (0, 1, 11):
                monkeypatch.setattr(phase_ssm, "_compiled_transitions", None)
                expected = phase_ssm.viterbi_units(*tables, minimum, 0.05, step)
                monkeypatch.setattr(phase_ssm, "_compiled_transitions", transitions)
                assert phase_ssm.viterbi_units(*tables, minimum, 0.05, step) == expected


def test_diagonal_projection_matches_both_numpy_reduction_orders(monkeypatch):
    from talea._attention_numba import diagonal_means_f32
    rng = np.random.default_rng(21)
    for length in (15, 124, 1025):
        values = rng.normal(size=(length, length)).astype(np.float16).astype(np.float32)
        values[0, -1] = -0.0
        monkeypatch.setattr(attention, "_compiled_diagonal_means", None)
        reference = attention.diagonal_mean_profile(values)
        monkeypatch.setattr(attention, "_compiled_diagonal_means", diagonal_means_f32)
        assert attention.diagonal_mean_profile(values).tobytes() == reference.tobytes()
    values[1, 2] = 2e-21
    assert diagonal_means_f32(values, True) is None


def test_registration_candidates_offsets_and_medians_match_numpy(monkeypatch):
    from talea._attention_numba import interval_medians, registration_geometry
    rng = np.random.default_rng(119)
    for period in (7, 8.5, 11.1, 21):
        values = rng.uniform(0.01, 1, size=(105, 105))
        prefix = unit_registration.integral_image(values)
        for start, end in ((0, 27), (17, 83), (62, 78)):
            args = (values, period, start, end)
            monkeypatch.setattr(unit_registration, "_compiled_registration_geometry", None)
            monkeypatch.setattr(unit_registration, "_compiled_interval_medians", None)
            reg = unit_registration.register_periodic_units(*args, phase_step=0.5, prefix=prefix)
            scan = unit_registration.scan_plausible_boundary_pairs(values, reg, prefix=prefix)
            monkeypatch.setattr(unit_registration, "_compiled_registration_geometry", registration_geometry)
            monkeypatch.setattr(unit_registration, "_compiled_interval_medians", interval_medians)
            assert unit_registration.register_periodic_units(*args, phase_step=0.5, prefix=prefix) == reg
            assert unit_registration.scan_plausible_boundary_pairs(values, reg, prefix=prefix) == scan
