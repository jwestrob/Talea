import numpy as np
import pytest

from talea.repeat_state import compiled_teacher_quantile, central_tile_spans, normalized_attention_tiles


@pytest.fixture
def histogram():
    kernel = compiled_teacher_quantile()
    if kernel is None:
        pytest.skip("requires the optional Numba kernel")
    return kernel


def test_histogram_preserves_numpy_interpolation_and_all_half_bins(histogram):
    # Exhaust the positive half grid, including subnormals. Quantiles lie on
    # and just next to boundaries where NumPy releases differ in rank precision.
    grid = np.arange(0x7C00, dtype=np.uint16).view(np.float16).astype(np.float32)
    rng = np.random.default_rng(18094)
    random = rng.choice(grid, size=2049)
    for a in (grid, grid[::-1].copy(), random, np.array([-0., 0.], np.float32),
              np.array([0., 0.5, 1.0], dtype=np.float32),
              np.tile(random, 37), np.array([1.], dtype=np.float32)):
        positive = a[a > 0]
        for q in (0.5, 0.5 + 2. ** -27, 0.811111111, 0.95, 0.999999999):
            expected = float(np.quantile(positive, q)) if positive.size else 0.
            assert histogram(a, q) == expected


def test_histogram_falls_back_for_unsupported_values_and_layouts(histogram):
    a = np.ones((18, 18), dtype=np.float32)
    assert histogram(a.T, 0.95) is None
    assert histogram(a.astype(np.float64), 0.95) is None
    assert histogram(a, np.float32(0.95)) is None
    spans = central_tile_spans(len(a), 16, 8)
    for x in (np.nextafter(np.float32(1), np.float32(2)), 2. ** -25, 1.e20):
        a[0, 0] = x
        assert histogram(a, 0.95) is None
        ref = normalized_attention_tiles([a], spans, 16, 0.95, 8)
        got = normalized_attention_tiles([a], spans, 16, 0.95, 8, quantile_function=histogram)
        assert ref.tobytes() == got.tobytes()
    for x, error in [(np.nan, "nonfinite"), (np.inf, "nonfinite"), (-1., "nonnegative")]:
        a[0, 0] = x
        assert histogram(a, 0.95) is None
        with pytest.raises(ValueError, match=error):
            normalized_attention_tiles([a], spans, 16, 0.95, 8, quantile_function=histogram)


def test_full_tiles_are_bit_identical_for_widened_apc_matrices(histogram):
    rng = np.random.default_rng(875946)
    for n in (1, 65, 132, 318):
        values = rng.random((2, n, n)).astype(np.float16).astype(np.float32)
        values[values < 0.15] = 0.
        spans = central_tile_spans(n, 128, 64)
        for q in (0.5, 0.95):
            expected = normalized_attention_tiles(values, spans, 128, q, 8)
            actual = normalized_attention_tiles(values, spans, 128, q, 8, quantile_function=histogram)
            assert expected.tobytes() == actual.tobytes()
