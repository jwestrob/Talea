import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from talea.cli import build_parser, main
from talea.io import SequenceRecord
from talea.long_protein import (
    combine_window_rows,
    is_coordinate_key,
    iter_windowed_predictions,
    shift_coordinates,
    window_record,
    window_spans,
)
from talea.phase_ssm import runtime_config
from talea.topology import process_attention_record

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_concordance import ASSETS, build_task  # noqa: E402


def record(record_id: str, sequence: str) -> SequenceRecord:
    return SequenceRecord(
        record_id=record_id,
        sequence=sequence,
        sequence_sha256=hashlib.sha256(sequence.encode()).hexdigest(),
    )


@pytest.mark.parametrize("length", [2001, 2100, 2999, 3000, 5000, 5500, 85804])
def test_windows_have_equal_length_and_cover_the_protein(length) -> None:
    spans = window_spans(length, 2000, 1000)
    assert spans[0][0] == 0
    assert spans[-1][1] == length
    assert all(end - start == 2000 for start, end in spans)
    assert all(
        0 < later[0] - earlier[0] <= 1000
        for earlier, later in zip(spans, spans[1:], strict=False)
    )


def test_windows_follow_the_giant_protein_chunk_grid() -> None:
    spans = window_spans(42_335, 2000, 1000)
    assert spans[:41] == [(1000 * k, 1000 * k + 2000) for k in range(41)]
    assert spans[41:] == [(40_335, 42_335)]


def test_a_protein_within_one_window_is_scored_whole() -> None:
    assert window_spans(1500, 2000, 1000) == [(0, 1500)]
    assert window_spans(2000, 2000, 1000) == [(0, 2000)]


@pytest.mark.parametrize("stride", [0, 2001])
def test_window_stride_must_leave_no_gaps(stride) -> None:
    with pytest.raises(ValueError, match="stride"):
        window_spans(5000, 2000, stride)


def test_window_record_slices_and_names_in_one_based_positions() -> None:
    parent = record("P", "ACDEFGHIKL" * 30)
    window = window_record(parent, 100, 250)
    assert window.record_id == "P/101-250"
    assert window.sequence == parent.sequence[100:250]
    assert window.sequence_sha256 == hashlib.sha256(window.sequence.encode()).hexdigest()


def flatten(value, path=()):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from flatten(item, (*path, key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from flatten(item, (*path, index))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        yield path, value


def last_key(path):
    return next(part for part in reversed(path) if isinstance(part, str))


def test_shift_moves_every_position_of_a_real_row_and_nothing_else() -> None:
    task = build_task(700)
    rng = np.random.default_rng(3)
    task["residue_structure_probabilities"] = rng.uniform(0.0, 1.0, size=(700, 3))
    task["phase_ssm"] = runtime_config(
        json.loads((ASSETS / "phase_ssm.json").read_text())
    )
    row = process_attention_record(task)
    assert row["status"] == "ok"
    row["teacher_state_summary"] = {
        "longest_run_start": 120,
        "longest_run_end": 180,
        "mean_probability": 0.2,
    }
    shifted = shift_coordinates(row, 1000)
    before = dict(flatten(row))
    after = dict(flatten(shifted))
    assert before.keys() == after.keys()
    moved = set()
    for path, value in before.items():
        if is_coordinate_key(last_key(path)) and isinstance(path[-1], str):
            assert after[path] == value + 1000, path
            moved.add(".".join(str(part) for part in path if isinstance(part, str)))
        else:
            assert after[path] == value, path
    assert {
        "wedge_call.start",
        "wedge_call.end",
        "registration.start",
        "registration.end",
        "registration.core_start",
        "registration.comb_start",
        "registration.phase_offset",
        "registration.units.start",
        "registration.comb_units.end",
        "boundary_scan.strongest_closure_candidate.start",
        "teacher_state_summary.longest_run_end",
    } <= moved


class FakePredictor:
    """Scores a sequence by its tryptophans and calls the first W run."""

    maximum_length = 50

    def __init__(self):
        self.seen = []

    def iter_predictions(self, records, **options):
        self.options = options
        for value in sorted(records, key=lambda item: (item.length, item.record_id)):
            self.seen.append(value.record_id)
            first = value.sequence.find("W")
            wedge = None if first < 0 else {"start": first, "end": first + 5}
            yield {
                "record_id": value.record_id,
                "sequence_sha256": value.sequence_sha256,
                "length": value.length,
                "status": "ok" if wedge else "no_wedge_call",
                "talea_discovery_score": float(value.sequence.count("W")),
                "teacher_state_z": 0.0,
                "global_z": 0.0,
                "wedge_call": None if wedge is None else {**wedge, "period": 5.0, "score": 1.0},
                "registration": None,
            }


def test_windowed_iteration_merges_long_proteins_and_passes_short_ones() -> None:
    short = record("short", "A" * 30)
    long_sequence = "A" * 70 + "W" * 12 + "A" * 38  # 120 residues, W run at 70-82
    long = record("long", long_sequence)
    predictor = FakePredictor()
    rows = list(
        iter_windowed_predictions(
            predictor, [long, short], window_length=40, window_stride=20, progress=False
        )
    )
    assert predictor.options == {"progress": False}
    assert [row["record_id"] for row in rows] == ["short", "long"]
    assert rows[0] == next(
        iter(FakePredictor().iter_predictions([short]))
    ), "short proteins keep their rows unchanged"
    merged = rows[1]
    spans = window_spans(120, 40, 20)
    assert merged["length"] == 120
    assert merged["sequence_sha256"] == long.sequence_sha256
    assert merged["windowing"]["window_count"] == len(spans)
    best = merged["windowing"]["best_window_index"]
    start, end = spans[best]
    assert (start, end) == (60, 100), "the window holding the whole W run wins"
    assert merged["talea_discovery_score"] == 12.0
    assert long_sequence[merged["wedge_call"]["start"]] == "W"
    assert merged["wedge_call"]["start"] == 70
    summaries = merged["windowing"]["windows"]
    assert [(item["start"], item["end"]) for item in summaries] == spans
    assert [item["wedge_start"] for item in summaries] == [
        None,
        None,
        70,
        70,
        80,
    ]


def test_exact_score_ties_choose_the_earlier_window() -> None:
    parent = record("P", "A" * 100)
    spans = [(0, 60), (40, 100)]
    rows = [
        {"record_id": f"P/{s + 1}-{e}", "sequence_sha256": "x", "status": "ok",
         "talea_discovery_score": 1.0, "teacher_state_z": 0.0, "global_z": 0.0,
         "wedge_call": None, "registration": None}
        for s, e in spans
    ]
    merged = combine_window_rows(parent, spans, rows, 60, 40)
    assert merged["windowing"]["best_window_index"] == 0


def test_window_length_cannot_exceed_the_contract() -> None:
    with pytest.raises(ValueError, match="window length"):
        list(iter_windowed_predictions(FakePredictor(), [], window_length=51))


def test_cli_defaults_reject_long_proteins_on_the_folding_grid() -> None:
    args = build_parser().parse_args(["-i", "in.faa", "-o", "out.jsonl"])
    assert args.long_proteins == "reject"
    assert (args.window_length, args.window_stride) == (2000, 1000)


def test_cli_rejects_long_proteins_by_default_and_names_window_mode(tmp_path) -> None:
    fasta = tmp_path / "in.faa"
    fasta.write_text(">giant\n" + "A" * 2100 + "\n")
    with pytest.raises(ValueError, match="--long-proteins window"):
        main(["-i", str(fasta), "-o", str(tmp_path / "out.jsonl")])
