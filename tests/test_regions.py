import sys
from pathlib import Path

import numpy as np

from talea.long_protein import combine_window_rows
from talea.long_protein import shift_coordinates
from talea.regions import (
    SECONDARY_MINIMUM_STATE,
    STATE_RUN_MINIMUM_LENGTH,
    add_region,
    max_sum_segment,
    repeat_regions,
    secondary_wedge_calls,
    state_runs,
    suppress_interval,
    trim_registration,
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
    state = np.full(400, 0.05)
    state[40:270] = 0.6
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


def test_a_state_run_the_wedge_never_proposed_becomes_a_state_region():
    state = np.full(500, 0.05)
    state[300:400] = 0.8  # e.g. a run of beta hairpins
    regions = repeat_regions(topology([]), state)
    assert [(r["start"], r["end"], r["source"]) for r in regions] == [(40, 170, "primary"), (300, 400, "state")]
    region = regions[1]
    assert region["period"] is None and region["wedge_score"] is None and region["units"] == []
    assert region["wedge_calls"] == 0 and region["mean_state_probability"] > 0.79


def test_a_state_run_keeps_only_its_free_stretch_and_only_if_it_spans_the_minimum():
    state = np.full(500, 0.05)
    state[100:250] = 0.8  # overlaps the registered primary region 40-170
    regions = repeat_regions(topology([]), state, primary_extent="registered")
    assert [(r["start"], r["end"], r["source"]) for r in regions] == [(40, 170, "primary"), (170, 250, "state")]
    state = np.full(500, 0.05)
    state[100:200] = 0.8  # free stretch 170-200 is too short
    assert [r["source"] for r in repeat_regions(topology([]), state, primary_extent="registered")] == ["primary"]


def test_state_runs_shorter_than_the_minimum_are_ignored():
    state = np.zeros(300)
    state[10:10 + STATE_RUN_MINIMUM_LENGTH - 1] = 0.9
    state[100:100 + STATE_RUN_MINIMUM_LENGTH] = 0.5
    assert state_runs(state) == [(100, 100 + STATE_RUN_MINIMUM_LENGTH)]


def test_wedge_regions_never_merge_into_a_state_region():
    regions = [{"start": 300, "end": 400, "period": None, "wedge_score": None, "source": "state", "wedge_calls": 0, "units": []}]
    add_region(regions, {"start": 280, "end": 420, "period": 30.0, "wedge_score": 15.0, "source": "secondary", "wedge_calls": 1})
    assert [(r["start"], r["end"], r["source"]) for r in regions] == [(300, 400, "state")]


def test_window_union_adds_state_regions_after_every_wedge_region():
    best = [
        {"start": 100, "end": 400, "period": 30.0, "wedge_score": 40.0, "source": "primary", "wedge_calls": 1},
        {"start": 500, "end": 600, "period": None, "wedge_score": None, "source": "state", "wedge_calls": 0, "units": []},
    ]
    other = [{"start": 0, "end": 300, "period": 22.0, "wedge_score": 25.0, "source": "secondary", "wedge_calls": 1,
              "mean_state_probability": 0.5}]
    union = union_window_regions([(0, best), (400, other)], best=0)
    assert [(r["start"], r["end"], r["source"]) for r in union] == [
        (100, 400, "primary"),
        (400, 700, "secondary"),
    ]
    other_state = [{"start": 200, "end": 300, "period": None, "wedge_score": None, "source": "state", "wedge_calls": 0,
                    "units": []}]
    union = union_window_regions([(0, best[:1]), (400, other_state)], best=0)
    assert [(r["start"], r["end"], r["source"], r["window_index"]) for r in union] == [
        (100, 400, "primary", 0),
        (600, 700, "state", 1),
    ]


def test_no_wedge_call_means_no_regions():
    assert repeat_regions({"wedge_call": None, "registration": None}, np.ones(50)) == []


def test_window_union_merges_a_region_seen_by_two_windows():
    first = [{"start": 1500, "end": 1900, "period": 24.0, "wedge_score": 40.0, "source": "primary", "wedge_calls": 1}]
    second = [
        {"start": 500, "end": 950, "period": 24.5, "wedge_score": 35.0, "source": "primary", "wedge_calls": 1,
         "mean_state_probability": 0.6},
        {"start": 1300, "end": 1700, "period": 40.0, "wedge_score": 20.0, "source": "secondary", "wedge_calls": 1,
         "mean_state_probability": 0.4},
    ]
    union = union_window_regions([(0, first), (1000, second)], best=0)
    assert [(r["start"], r["end"], r["source"], r["window_index"]) for r in union] == [
        (1500, 1950, "primary", 0),
        (2300, 2700, "secondary", 1),
    ]
    assert union[0]["wedge_calls"] == 2


def test_other_windows_add_their_primary_region_only_when_it_is_supported():
    best = [{"start": 100, "end": 400, "period": 30.0, "wedge_score": 40.0, "source": "primary", "wedge_calls": 1,
             "mean_state_probability": 0.1}]
    weak = [{"start": 800, "end": 1100, "period": 22.0, "wedge_score": 25.0, "source": "primary", "wedge_calls": 1,
             "mean_state_probability": 0.1}]
    strong = [{"start": 800, "end": 1100, "period": 22.0, "wedge_score": 25.0, "source": "primary", "wedge_calls": 1,
               "mean_state_probability": 0.5}]
    assert [r["source"] for r in union_window_regions([(0, best), (1000, weak)], best=0)] == ["primary"]
    union = union_window_regions([(0, best), (1000, strong)], best=0)
    assert [(r["start"], r["source"], r["window_index"]) for r in union] == [(100, "primary", 0), (1800, "secondary", 1)]


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
    windows = combined["windowing"]["windows"]
    assert windows[0]["repeat_regions"] == []
    assert [(r["start"], r["end"]) for r in windows[1]["repeat_regions"]] == [(1100, 1400)]


def test_topology_rows_list_further_calls_and_keep_the_primary_call():
    row = process_attention_record(build_task(700))
    assert isinstance(row["secondary_wedge_calls"], list)
    for call in row["secondary_wedge_calls"]:
        assert call["score"] <= row["wedge_call"]["score"]
        assert set(call) == {"rank", "start", "end", "period", "score", "units", "unit_placement"}


def test_further_calls_are_registered_like_the_primary_call():
    task = build_task(640)
    matrix = (localized_banded_matrix(640, 40, 200, 20) + 0.7 * localized_banded_matrix(640, 380, 540, 32))
    task["matrices"] = {13: matrix.astype(np.float32), 35: matrix.astype(np.float32)}
    row = process_attention_record(task)
    assert (row["wedge_call"]["start"], row["wedge_call"]["end"]) == (40, 200)
    (call,) = row["secondary_wedge_calls"]
    assert (call["start"], call["end"], call["period"]) == (380, 540, 32.0)
    assert call["unit_placement"] == "comb"
    lengths = [unit["end"] - unit["start"] for unit in call["units"]]
    assert len(lengths) >= 5 and set(lengths) == {32}
    state = np.full(640, 0.6)
    regions = [r for r in repeat_regions(row, state) if r["source"] != "state"]
    assert [len(region["units"]) for region in regions] == [len(row["registration"]["units"]), len(call["units"])]


def test_only_further_calls_the_state_supports_are_registered():
    task = build_task(640)
    matrix = (localized_banded_matrix(640, 40, 200, 20) + 0.7 * localized_banded_matrix(640, 380, 540, 32))
    task["matrices"] = {13: matrix.astype(np.float32), 35: matrix.astype(np.float32)}
    task["residue_state_probabilities"] = np.full(640, 0.05)
    (call,) = process_attention_record(task)["secondary_wedge_calls"]
    assert call["units"] is None and call["unit_placement"] is None
    task["residue_state_probabilities"][380:540] = 0.6
    (call,) = process_attention_record(task)["secondary_wedge_calls"]
    assert len(call["units"]) >= 5


def test_the_worker_passes_the_state_track_through():
    import talea.postprocess_pool as pool_module

    task = build_task(640)
    matrix = (localized_banded_matrix(640, 40, 200, 20) + 0.7 * localized_banded_matrix(640, 380, 540, 32))
    task["matrices"] = {13: matrix.astype(np.float32), 35: matrix.astype(np.float32)}
    state = np.full(640, 0.05)
    saved = pool_module._CONFIGS
    pool_module.initialize_worker({key: task[key] for key in (
        "protein_calibration", "local_calibration", "wedge_config", "operational_config")})
    try:
        row = pool_module.run_record((task["metadata"], task["matrices"], task["window_sizes"], None, state))
    finally:
        pool_module._CONFIGS = saved
    assert [call["units"] for call in row["secondary_wedge_calls"]] == [None]


def test_each_region_keeps_its_own_units_inside_its_extent():
    row = topology([
        {"rank": 2, "start": 390, "end": 520, "period": 32.0, "score": 20.0,
         "units": [{"start": s, "end": s + 32} for s in range(360, 560, 32)]},
    ])
    row["registration"]["units"] = [{"start": s, "end": s + 20} for s in range(40, 170, 20)]
    state = np.full(600, 0.6)
    primary, secondary = [r for r in repeat_regions(row, state) if r["source"] != "state"]
    assert primary["units"][0] == {"start": 40, "end": 60} and len(primary["units"]) == 7
    # The region spans 374-536; units whose midpoints fall outside it are dropped, the rest clipped.
    assert (secondary["start"], secondary["end"]) == (374, 536)
    assert secondary["units"][0] == {"start": 374, "end": 392}
    assert secondary["units"][-1] == {"start": 488, "end": 520}
    assert all(374 <= unit["start"] < unit["end"] <= 536 for unit in secondary["units"])


def test_a_merge_adds_only_the_units_beyond_the_region():
    regions = [{"start": 40, "end": 170, "period": 20.0, "wedge_score": 30.0, "source": "primary", "wedge_calls": 1,
                "units": [{"start": s, "end": s + 20} for s in range(40, 170, 20)]}]
    add_region(regions, {"start": 150, "end": 250, "period": 21.0, "wedge_score": 18.0, "source": "secondary",
                         "wedge_calls": 1, "units": [{"start": s, "end": s + 21} for s in range(152, 250, 21)]})
    (region,) = regions
    starts = [unit["start"] for unit in region["units"]]
    assert starts[:7] == list(range(40, 170, 20))
    # 152-173 centres inside 160-180 and is dropped; 173-194 starts where 160-180 ends.
    assert starts[7:] == [180, 194, 215, 236]
    assert all(left["end"] <= right["start"] for left, right in zip(region["units"], region["units"][1:]))
    assert region["units"][-1]["end"] <= region["end"] == 250


def test_window_union_moves_units_into_the_protein_frame():
    second = [{"start": 100, "end": 400, "period": 30.0, "wedge_score": 25.0, "source": "primary", "wedge_calls": 1,
               "mean_state_probability": 0.5, "units": [{"start": 100, "end": 130}, {"start": 130, "end": 160}]}]
    union = union_window_regions([(0, []), (1000, second)], best=0)
    assert union[0]["units"] == [{"start": 1100, "end": 1130}, {"start": 1130, "end": 1160}]


def units_from(start, end, period):
    return [{"start": first, "end": first + period} for first in range(start, end, period)]


def test_max_sum_segment_finds_the_heaviest_stretch():
    values = np.array([-1.0, 2.0, -0.5, 3.0, -4.0, 1.0])
    assert max_sum_segment(values, 0, len(values)) == (1, 4)
    assert max_sum_segment(values, 4, 6) == (5, 6)


def test_the_primary_region_is_cut_to_the_state_track():
    state = np.full(500, 0.05)
    state[45:160] = 0.8  # wedge 50-150; the registered extent 40-170 overshoots the state
    row = topology([])
    row["registration"]["units"] = units_from(40, 170, 20)
    (primary,) = repeat_regions(row, state)
    assert (primary["start"], primary["end"]) == (45, 160)
    assert (primary["registered_start"], primary["registered_end"]) == (40, 170)
    assert all(45 <= unit["start"] < unit["end"] <= 160 for unit in primary["units"])
    (kept,) = repeat_regions(row, state, primary_extent="registered")
    assert (kept["start"], kept["end"]) == (40, 170) and "registered_start" not in kept


def test_the_state_never_cuts_inside_the_wedge_interval():
    state = np.full(500, 0.05)
    state[80:120] = 0.8  # the U-Net fires on only part of the wedge's repeat (50-150)
    (primary,) = repeat_regions(topology([]), state)
    assert (primary["start"], primary["end"]) == (50, 150)


def test_a_flat_low_state_leaves_the_primary_registered():
    (primary,) = repeat_regions(topology([]), np.full(500, 0.05))
    assert (primary["start"], primary["end"]) == (40, 170) and "registered_start" not in primary


def test_the_cut_primary_and_state_regions_never_overlap():
    state = np.full(500, 0.05)
    state[100:250] = 0.8  # the state extends past the registered end (170)
    regions = repeat_regions(topology([]), state)
    assert [(r["start"], r["end"], r["source"]) for r in regions] == [(50, 235, "primary")]


def test_an_unknown_primary_extent_is_rejected():
    try:
        repeat_regions(topology([]), np.full(500, 0.05), primary_extent="wedge")
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_trim_registration_keeps_whole_units_at_least_half_inside():
    state = np.full(400, 0.05)
    state[75:150] = 0.8
    registration = {"start": 40, "end": 180, "period": 20.0, "n_units": 7, "units": units_from(40, 180, 20)}
    trimmed = trim_registration(registration, state)
    assert [(u["start"], u["end"]) for u in trimmed["units"]] == [(80, 100), (100, 120), (120, 140), (140, 160)]
    assert (trimmed["start"], trimmed["end"], trimmed["n_units"]) == (80, 160, 4)
    assert (trimmed["registered_start"], trimmed["registered_end"]) == (40, 180)
    assert trimmed["registered_units"] == registration["units"]
    assert registration["start"] == 40 and len(registration["units"]) == 7  # input untouched


def test_trim_registration_keeps_the_registration_when_fewer_than_three_units_survive():
    state = np.full(400, 0.05)
    state[80:120] = 0.8
    registration = {"start": 40, "end": 180, "period": 20.0, "n_units": 7, "units": units_from(40, 180, 20)}
    assert trim_registration(registration, state) is registration
    assert trim_registration(None, state) is None


def test_trimmed_registration_moves_into_the_protein_frame_in_window_mode():
    state = np.full(400, 0.05)
    state[75:150] = 0.8
    registration = {"start": 40, "end": 180, "period": 20.0, "n_units": 7, "units": units_from(40, 180, 20)}
    shifted = shift_coordinates(trim_registration(registration, state), 1000)
    assert (shifted["start"], shifted["registered_start"], shifted["registered_end"]) == (1080, 1040, 1180)
    assert shifted["registered_units"][0] == {"start": 1040, "end": 1060}


def test_a_further_region_gives_way_to_the_cut_primary():
    state = np.full(600, 0.05)
    state[60:260] = 0.8  # the primary's state stretch runs into the further call's interval
    state[260:400] = 0.6
    regions = repeat_regions(
        topology([{"rank": 2, "start": 200, "end": 400, "period": 32.0, "score": 20.0}]), state
    )
    primary, further = regions
    assert primary["source"] == "primary" and further["source"] == "secondary"
    assert primary["end"] <= further["start"]
    assert further["end"] - further["start"] >= 2 * 32


def test_trim_registration_keeps_the_units_the_wedge_floor_covers():
    state = np.full(400, 0.05)
    state[100:140] = 0.8
    registration = {"start": 40, "end": 180, "period": 20.0, "n_units": 7, "units": units_from(40, 180, 20)}
    assert trim_registration(registration, state) is registration  # two units only: kept as registered
    trimmed = trim_registration(registration, state, floor=(60, 160))
    assert (trimmed["start"], trimmed["end"], trimmed["n_units"]) == (60, 160, 5)
