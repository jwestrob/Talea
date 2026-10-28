import sys
from pathlib import Path

import numpy as np

from talea.long_protein import combine_window_rows
from talea.regions import (
    SECONDARY_MINIMUM_STATE,
    add_region,
    repeat_regions,
    secondary_wedge_calls,
    suppress_interval,
    union_window_regions,
)
from talea.topology import process_attention_record
from talea.wedge import best_wedge_interval, robust_diagonal_tracks

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_concordance import build_task  # noqa: E402
from test_long_protein import record  # noqa: E402
from test_wedge import localized_banded_matrix  # noqa: E402

WEDGE = {"tolerance": 1, "max_harmonics": 8, "min_harmonics": 2, "interharmonic_controls": True}


def two_region_tracks(length=640):
    matrix = localized_banded_matrix(length, 40, 200, 20) + 0.7 * localized_banded_matrix(
        length, 380, 540, 32
    )
    return robust_diagonal_tracks(matrix, max_separation=256)


def test_suppression_neutralizes_exactly_the_pairs_touching_the_interval():
    rng = np.random.default_rng(0)
    tracks = [np.empty(0)] + [rng.normal(size=100 - d) for d in range(1, 10)]
    before = [track.copy() for track in tracks]
    suppress_interval(tracks, 40, 60)
    for d in range(1, 10):
        i = np.arange(100 - d)
        touch = ((i >= 40) & (i < 60)) | ((i + d >= 40) & (i + d < 60))
        assert np.all(tracks[d][touch] == np.median(before[d]))
        assert np.array_equal(tracks[d][~touch], before[d][~touch])


def test_peeling_finds_the_second_region_after_the_primary():
    tracks = two_region_tracks()
    widths = tuple(range(96, 257, 32))
    primary = best_wedge_interval(tracks, 640, (20.0, 32.0), widths, **WEDGE)
    assert primary["period"] == 20.0 and primary["start"] < 200
    calls = secondary_wedge_calls(tracks, 640, (20.0, 32.0), widths, WEDGE, primary)
    assert calls and calls[0]["rank"] == 2 and calls[0]["period"] == 32.0
    assert calls[0]["start"] >= 360 and calls[0]["end"] <= 560
    assert secondary_wedge_calls(tracks, 640, (20.0, 32.0), widths, WEDGE, None) == []


def test_weak_further_calls_are_not_reported():
    tracks = robust_diagonal_tracks(localized_banded_matrix(640, 40, 200, 20), max_separation=256)
    widths = tuple(range(96, 257, 32))
    primary = best_wedge_interval(tracks, 640, (20.0, 32.0), widths, **WEDGE)
    assert secondary_wedge_calls(tracks, 640, (20.0, 32.0), widths, WEDGE, primary) == []


def topology(secondary):
    return {
        "wedge_call": {"start": 50, "end": 150, "period": 20.0, "score": 30.0},
        "registration": {"start": 40, "end": 170, "period": 20.0},
        "secondary_wedge_calls": secondary,
    }


def test_regions_keep_the_registered_primary_and_add_supported_calls():
    state = np.full(600, 0.05)
    state[380:540] = 0.6
    regions = repeat_regions(
        topology([
            {"rank": 2, "start": 390, "end": 520, "period": 32.0, "score": 20.0},
            {"rank": 3, "start": 250, "end": 350, "period": 25.0, "score": 16.0},
        ]),
        state,
    )
    assert [(r["start"], r["end"], r["source"]) for r in regions] == [
        (40, 170, "primary"),
        (374, 536, "secondary"),
    ]
    assert regions[1]["period"] == 32.0 and regions[1]["wedge_score"] == 20.0
    assert regions[1]["mean_state_probability"] > SECONDARY_MINIMUM_STATE


def test_a_same_period_call_next_to_the_primary_merges_into_it():
    state = np.full(400, 0.6)
    regions = repeat_regions(
        topology([{"rank": 2, "start": 180, "end": 260, "period": 21.0, "score": 18.0}]),
        state,
    )
    assert len(regions) == 1
    assert (regions[0]["start"], regions[0]["end"], regions[0]["wedge_calls"]) == (40, 270, 2)


def test_an_overlapping_call_at_another_period_keeps_only_its_free_stretch():
    regions = [{"start": 40, "end": 170, "period": 20.0, "wedge_score": 30.0, "source": "primary", "wedge_calls": 1}]
    add_region(regions, {"start": 150, "end": 260, "period": 30.0, "wedge_score": 15.0, "source": "secondary", "wedge_calls": 1})
    assert [(r["start"], r["end"]) for r in regions] == [(40, 170), (170, 260)]
    add_region(regions, {"start": 160, "end": 200, "period": 40.0, "wedge_score": 13.0, "source": "secondary", "wedge_calls": 1})
    assert len(regions) == 2


def test_no_wedge_call_means_no_regions():
    assert repeat_regions({"wedge_call": None, "registration": None}, np.ones(50)) == []


def test_window_union_merges_a_region_seen_by_two_windows():
    first = [{"start": 1500, "end": 1900, "period": 24.0, "wedge_score": 40.0, "source": "primary", "wedge_calls": 1}]
    second = [
        {"start": 500, "end": 950, "period": 24.5, "wedge_score": 35.0, "source": "primary", "wedge_calls": 1},
        {"start": 1300, "end": 1700, "period": 40.0, "wedge_score": 20.0, "source": "secondary", "wedge_calls": 1},
    ]
    union = union_window_regions([(0, first), (1000, second)], best=0)
    assert [(r["start"], r["end"], r["source"], r["window_index"]) for r in union] == [
        (1500, 1950, "primary", 0),
        (2300, 2700, "secondary", 1),
    ]
    assert union[0]["wedge_calls"] == 2


def test_windowed_rows_carry_the_union_of_window_regions():
    parent = record("P", "A" * 3000)
    spans = [(0, 2000), (1000, 3000)]
    rows = []
    for index, regions in enumerate(([], [{"start": 100, "end": 400, "period": 30.0, "wedge_score": 25.0,
                                          "source": "primary", "wedge_calls": 1}])):
        rows.append({
            "record_id": f"P/{spans[index][0] + 1}-{spans[index][1]}", "sequence_sha256": "x", "length": 2000,
            "status": "ok", "talea_discovery_score": float(index), "teacher_state_z": 0.0, "global_z": 0.0,
            "wedge_call": None, "registration": None, "repeat_regions": regions,
        })
    combined = combine_window_rows(parent, spans, rows, 2000, 1000)
    assert [(r["start"], r["end"], r["source"]) for r in combined["repeat_regions"]] == [(1100, 1400, "primary")]


def test_topology_rows_list_further_calls_and_keep_the_primary_call():
    row = process_attention_record(build_task(700))
    assert isinstance(row["secondary_wedge_calls"], list)
    for call in row["secondary_wedge_calls"]:
        assert call["score"] <= row["wedge_call"]["score"]
        assert set(call) == {"rank", "start", "end", "period", "score"}
