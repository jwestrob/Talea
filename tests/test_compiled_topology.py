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
