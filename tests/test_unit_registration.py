import numpy as np

from talea.unit_registration import (
    block_mean,
    integral_image,
    periodic_tiles,
    register_periodic_units,
    registered_closure_score,
    scan_plausible_boundary_pairs,
)


def test_integral_block_mean_matches_direct_mean():
    matrix = np.arange(100, dtype=float).reshape(10, 10)
    prefix = integral_image(matrix)
    assert block_mean(prefix, (2, 5), (6, 9)) == np.mean(matrix[2:5, 6:9])


def test_periodic_tiles_retain_substantial_terminal_units():
    assert periodic_tiles(20, period=4, offset=0) == [
        (0, 4),
        (4, 8),
        (8, 12),
        (12, 16),
        (16, 20),
    ]
    assert periodic_tiles(20, period=4, offset=3) == [
        (0, 3),
        (3, 7),
        (7, 11),
        (11, 15),
        (15, 19),
    ]


def test_registration_extends_through_supported_repeat_edges_only():
    matrix = np.zeros((40, 40), dtype=np.float32)
    true_units = [(8 + 4 * i, 12 + 4 * i) for i in range(6)]
    for left, right in zip(true_units, true_units[1:]):
        matrix[left[0] : left[1], right[0] : right[1]] = 1.0
        matrix[right[0] : right[1], left[0] : left[1]] = 1.0
    result = register_periodic_units(
        matrix,
        period=4,
        core_start=13,
        core_end=27,
        extension_fraction=0.5,
    )
    assert result["start"] == 8
    assert result["end"] == 32
    assert result["n_units"] == 6


def test_closure_score_is_lower_for_a_cyclic_unit_graph():
    units = [{"start": 4 * i, "end": 4 * (i + 1)} for i in range(5)]
    path = np.zeros((20, 20), dtype=np.float32)
    for left, right in zip(units, units[1:]):
        path[left["start"] : left["end"], right["start"] : right["end"]] = 1
        path[right["start"] : right["end"], left["start"] : left["end"]] = 1
    cycle = path.copy()
    cycle[16:20, 0:4] = 1
    cycle[0:4, 16:20] = 1
    assert registered_closure_score(path, units)["open_path_score"] > registered_closure_score(
        cycle, units
    )["open_path_score"]


def test_boundary_scan_finds_full_length_cycle_seam():
    units = [{"start": 4 * i, "end": 4 * (i + 1)} for i in range(5)]
    cycle = np.zeros((20, 20), dtype=np.float32)
    for left, right in zip(units, units[1:]):
        cycle[left["start"] : left["end"], right["start"] : right["end"]] = 1
        cycle[right["start"] : right["end"], left["start"] : left["end"]] = 1
    cycle[16:20, 0:4] = 1
    cycle[0:4, 16:20] = 1
    registration = {
        "period": 4,
        "phase_offset": 0,
        "core_start": 4,
        "core_end": 16,
    }
    result = scan_plausible_boundary_pairs(cycle, registration)
    assert result["candidate_count"] == 4
    assert result["noncore_false_scan_candidate_count"] == 2
    assert result["strongest_closure_candidate"]["start"] == 0
    assert result["strongest_closure_candidate"]["end"] == 20
