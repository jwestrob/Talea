"""Overlapping-window scoring for proteins longer than Talea's input contract.

ESM++ reads at most 2,046 residues, so a longer protein is scored as windows of
``window_length`` residues starting every ``window_stride`` residues, plus one
final window aligned to the C-terminus so that every window has the same
length. The default 2,000/1,000 grid is the chunking used to fold the
Omnitrophota giant proteins (chunk k spans residues 1000k to 1000k + 2000), so
a window score lines up with an existing chunk structure.

Each window is scored exactly as a standalone protein. The protein's row is the
best window's row (highest discovery score) with every residue coordinate moved
into the full-protein frame, plus a compact summary of every window under
``windowing``. Its ``repeat_regions`` gathers the supported regions of every
window, so a protein with repeat domains in several windows lists them all;
each window's own regions stay under ``windowing``. Proteins within the contract are scored whole and their rows are
unchanged.

This module stays free of torch, so the window arithmetic and the row merge can
be tested anywhere.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Iterator, Sequence

from talea.io import SequenceRecord
from talea.regions import union_window_regions

DEFAULT_WINDOW_LENGTH = 2000
DEFAULT_WINDOW_STRIDE = 1000
LONG_PROTEIN_MODES = ("reject", "window")


def window_spans(
    length: int, window_length: int, window_stride: int
) -> list[tuple[int, int]]:
    """Half-open windows of exactly ``window_length`` residues covering a protein.

    Windows start at 0, ``window_stride``, 2 x ``window_stride``, ... while they
    fit; when the grid stops short of the C-terminus, one more window ends
    exactly there. A protein no longer than one window is a single window.
    """

    if length < 1:
        raise ValueError("protein length must be positive")
    if window_length < 1 or not 1 <= window_stride <= window_length:
        raise ValueError(
            "window length must be positive and the stride between 1 and the "
            "window length"
        )
    if length <= window_length:
        return [(0, length)]
    starts = list(range(0, length - window_length + 1, window_stride))
    if starts[-1] + window_length < length:
        starts.append(length - window_length)
    return [(start, start + window_length) for start in starts]


def window_record(record: SequenceRecord, start: int, end: int) -> SequenceRecord:
    """The residues ``start:end`` of a protein, named ``record_id/start-end``.

    The name uses 1-based inclusive positions (the Pfam/Stockholm convention),
    so ``P/1001-3000`` holds residues 1001 to 3000 of ``P``.
    """

    sequence = record.sequence[start:end]
    return SequenceRecord(
        record_id=f"{record.record_id}/{start + 1}-{end}",
        sequence=sequence,
        sequence_sha256=hashlib.sha256(sequence.encode()).hexdigest(),
    )


def is_coordinate_key(key: str) -> bool:
    """Whether a row field holds a residue position (0-based, half-open)."""

    return key in ("start", "end", "phase_offset") or key.endswith(
        ("_start", "_end")
    )


def shift_coordinates(value, offset: int):
    """Copy a row with every residue position moved ``offset`` residues later.

    Positions are the numeric values of fields named ``start``, ``end``,
    ``*_start``, ``*_end`` or ``phase_offset`` (the origin of the registration
    comb), at any depth. Unit and candidate indices, periods and scores are
    left alone.
    """

    if isinstance(value, dict):
        return {
            key: (
                item + offset
                if is_coordinate_key(key)
                and isinstance(item, (int, float))
                and not isinstance(item, bool)
                else shift_coordinates(item, offset)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [shift_coordinates(item, offset) for item in value]
    return value


def window_summary(index: int, start: int, end: int, row: dict) -> dict:
    """One window's scores and calls, with positions in the full protein."""

    wedge = row.get("wedge_call")
    registration = row.get("registration")
    return {
        "index": index,
        "start": start,
        "end": end,
        "status": row["status"],
        "talea_discovery_score": row["talea_discovery_score"],
        "teacher_state_z": row["teacher_state_z"],
        "global_z": row["global_z"],
        "period": None if wedge is None else wedge["period"],
        "wedge_score": None if wedge is None else wedge["score"],
        "wedge_start": None if wedge is None else start + int(wedge["start"]),
        "wedge_end": None if wedge is None else start + int(wedge["end"]),
        "registered_start": (
            None if registration is None else start + int(registration["start"])
        ),
        "registered_end": (
            None if registration is None else start + int(registration["end"])
        ),
        "n_units": None if registration is None else int(registration["n_units"]),
        **(
            {"repeat_regions": shift_coordinates(row["repeat_regions"], start)}
            if "repeat_regions" in row
            else {}
        ),
    }


def combine_window_rows(
    record: SequenceRecord,
    spans: Sequence[tuple[int, int]],
    rows: Sequence[dict],
    window_length: int,
    window_stride: int,
) -> dict:
    """Merge one protein's window rows into its protein row.

    The best window is the one with the highest discovery score (the earlier
    window on an exact tie). Its row supplies every score and call, with
    positions moved into the full-protein frame; ``windowing`` lists all
    windows.
    """

    if len(rows) != len(spans) or not rows:
        raise ValueError("every window needs exactly one prediction row")
    best = max(
        range(len(rows)),
        key=lambda index: (float(rows[index]["talea_discovery_score"]), -index),
    )
    start, end = spans[best]
    combined = shift_coordinates(rows[best], start)
    if any("repeat_regions" in row for row in rows):
        combined["repeat_regions"] = union_window_regions(
            [
                (
                    window_start,
                    row.get("repeat_regions") or [],
                    float(row["talea_discovery_score"]),
                )
                for (window_start, _), row in zip(spans, rows, strict=True)
            ],
            best,
        )
    combined["record_id"] = record.record_id
    combined["sequence_sha256"] = record.sequence_sha256
    combined["length"] = record.length
    combined["windowing"] = {
        "window_length": int(window_length),
        "window_stride": int(window_stride),
        "window_count": len(rows),
        "best_window_index": best,
        "best_window_start": start,
        "best_window_end": end,
        "best_window_sequence_sha256": rows[best]["sequence_sha256"],
        "windows": [
            window_summary(index, window_start, window_end, row)
            for index, ((window_start, window_end), row) in enumerate(
                zip(spans, rows, strict=True)
            )
        ],
    }
    return combined


def iter_windowed_predictions(
    predictor,
    records: Iterable[SequenceRecord],
    *,
    window_length: int = DEFAULT_WINDOW_LENGTH,
    window_stride: int = DEFAULT_WINDOW_STRIDE,
    **prediction_options,
) -> Iterator[dict]:
    """Like ``predictor.iter_predictions``, but long proteins become windows.

    Proteins within the predictor's maximum length are passed through and keep
    their rows unchanged. Each longer protein yields one row once all of its
    windows are scored; rows therefore arrive in the predictor's order, with
    windowed proteins among the ``window_length`` records.
    """

    maximum = int(predictor.maximum_length)
    if not 1 <= window_length <= maximum:
        raise ValueError(
            f"window length must be between 1 and Talea's maximum of {maximum}"
        )
    scored: list[SequenceRecord] = []
    parents: dict[str, SequenceRecord] = {}
    spans_by_parent: dict[str, list[tuple[int, int]]] = {}
    owners: dict[str, tuple[str, int]] = {}
    for record in records:
        if record.length <= maximum:
            scored.append(record)
            continue
        if record.record_id in parents:
            raise ValueError(f"duplicate record ID: {record.record_id}")
        spans = window_spans(record.length, window_length, window_stride)
        parents[record.record_id] = record
        spans_by_parent[record.record_id] = spans
        for index, (start, end) in enumerate(spans):
            window = window_record(record, start, end)
            owners[window.record_id] = (record.record_id, index)
            scored.append(window)
    collected = {
        record_id: [None] * len(spans) for record_id, spans in spans_by_parent.items()
    }
    remaining = {
        record_id: len(spans) for record_id, spans in spans_by_parent.items()
    }
    for row in predictor.iter_predictions(scored, **prediction_options):
        owner = owners.get(row["record_id"])
        if owner is None:
            yield row
            continue
        record_id, index = owner
        collected[record_id][index] = row
        remaining[record_id] -= 1
        if not remaining[record_id]:
            yield combine_window_rows(
                parents[record_id],
                spans_by_parent[record_id],
                collected.pop(record_id),
                window_length,
                window_stride,
            )
