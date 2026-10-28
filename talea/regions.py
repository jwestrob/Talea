"""Every repeat region in a protein: Talea's primary call plus further wedge calls.

The wedge returns the single best period and interval, so a protein with two
repeat regions (a solenoid split by an insertion, or two repeat domains) is
reported by one of them. Here the search continues ("peeling"): the diagonal
values of every residue pair that touches the primary call's interval are set
to that diagonal's median, and the wedge is searched again. The search stops
at the first call whose wedge score falls below 12 or below half the primary
call's score (and after 20 further calls at most).

A further call becomes a region when the U-Net residue-state track averages at
least 0.2 over its interval. Each region extends half a period past its wedge
interval, as Talea's registration seed does. A region that overlaps or nearly
touches (within one period) a region of the same period (within 15%) merges
into it; otherwise it keeps its longest stretch outside the regions already
found, provided that stretch spans two periods.

The primary region starts as the registered extent and grows only by merging.
Registration, units and every score continue to describe the primary call
alone.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np

from talea.wedge import WedgePlan, best_wedge_interval

SECONDARY_MAXIMUM_CALLS = 20
SECONDARY_MINIMUM_SCORE = 12.0
SECONDARY_MINIMUM_RELATIVE_SCORE = 0.5
SECONDARY_MINIMUM_STATE = 0.2
MERGE_PERIOD_TOLERANCE = 0.15


def suppress_interval(tracks: Sequence[np.ndarray], start: int, end: int) -> None:
    """Set, in place, every diagonal value whose pair touches ``[start, end)`` to the diagonal's median."""

    for separation in range(1, len(tracks)):
        track = tracks[separation]
        size = len(track)
        if not size:
            continue
        median = float(np.median(track))
        # Index i of diagonal d is the pair (i, i + d).
        first, last = max(0, start), min(size, end)
        if last > first:
            track[first:last] = median
        first, last = max(0, start - separation), min(size, end - separation)
        if last > first:
            track[first:last] = median


def secondary_wedge_calls(
    tracks: Sequence[np.ndarray],
    length: int,
    periods: Sequence[float],
    widths: Sequence[int],
    wedge: Mapping,
    primary: Mapping | None,
) -> list[dict]:
    """Further wedge calls after the primary one, strongest first."""

    if primary is None or not periods:
        return []
    primary_score = float(primary["score"])
    minimum = max(SECONDARY_MINIMUM_SCORE, SECONDARY_MINIMUM_RELATIVE_SCORE * primary_score)
    work = [np.array(track, dtype=np.float64, copy=True) for track in tracks]
    suppress_interval(work, int(primary["start"]), int(primary["end"]))
    settings = {
        "tolerance": int(wedge["tolerance"]),
        "max_harmonics": int(wedge["max_harmonics"]),
        "min_harmonics": int(wedge["min_harmonics"]),
        "interharmonic_controls": bool(wedge["interharmonic_controls"]),
    }
    # Suppression edits values only, so every search shares one template plan.
    plan = WedgePlan([len(track) for track in work], length, periods, widths, **settings)
    calls = []
    for rank in range(2, 2 + SECONDARY_MAXIMUM_CALLS):
        call = best_wedge_interval(work, length, periods, widths, plan=plan, **settings)
        score = float(call["score"])
        if not math.isfinite(score) or score < minimum:
            break
        calls.append(
            {
                "rank": rank,
                "start": int(call["start"]),
                "end": int(call["end"]),
                "period": float(call["period"]),
                "score": score,
            }
        )
        suppress_interval(work, int(call["start"]), int(call["end"]))
    return calls


def _longest_free_stretch(start: int, end: int, regions: Sequence[Mapping]) -> tuple[int, int]:
    free = np.ones(end - start, dtype=bool)
    for region in regions:
        first, last = max(start, int(region["start"])), min(end, int(region["end"]))
        if last > first:
            free[first - start : last - start] = False
    best = (start, start)
    run_start = None
    for index, value in enumerate(np.append(free, False)):
        if value and run_start is None:
            run_start = index
        elif not value and run_start is not None:
            if index - run_start > best[1] - best[0]:
                best = (start + run_start, start + index)
            run_start = None
    return best


def add_region(regions: list[dict], candidate: Mapping) -> None:
    """Add ``candidate`` to ``regions`` in place, merging or trimming it as described above.

    A merge extends the existing region and adds the candidate's wedge calls to
    its count; every other field of the existing region is kept.
    """

    start, end = int(candidate["start"]), int(candidate["end"])
    period = float(candidate["period"])
    target = next(
        (
            region
            for region in regions
            if abs(period / region["period"] - 1.0) <= MERGE_PERIOD_TOLERANCE
            and start <= region["end"] + min(period, region["period"])
            and end >= region["start"] - min(period, region["period"])
        ),
        None,
    )
    if target is not None:
        target["start"] = min(target["start"], start)
        target["end"] = max(target["end"], end)
        target["wedge_calls"] += int(candidate.get("wedge_calls", 1))
        return
    start, end = _longest_free_stretch(start, end, regions)
    if end - start >= 2 * period:
        regions.append({**candidate, "start": start, "end": end})


def repeat_regions(topology: Mapping, state_track: np.ndarray) -> list[dict]:
    """Primary region plus accepted further calls, sorted by position.

    ``topology`` is a ``process_attention_record`` row; ``state_track`` is the
    U-Net residue-state probability for the same protein.
    """

    wedge = topology.get("wedge_call")
    registration = topology.get("registration")
    if wedge is None or registration is None:
        return []
    state = np.asarray(state_track, dtype=np.float64)
    length = len(state)
    regions = [
        {
            "start": int(registration["start"]),
            "end": int(registration["end"]),
            "period": float(registration["period"]),
            "wedge_score": float(wedge["score"]),
            "source": "primary",
            "wedge_calls": 1,
        }
    ]
    for call in topology.get("secondary_wedge_calls") or []:
        start, end, period = int(call["start"]), int(call["end"]), float(call["period"])
        if float(np.mean(state[start:end])) < SECONDARY_MINIMUM_STATE:
            continue
        add_region(
            regions,
            {
                "start": max(0, int(round(start - 0.5 * period))),
                "end": min(length, int(round(end + 0.5 * period))),
                "period": period,
                "wedge_score": float(call["score"]),
                "source": "secondary",
                "wedge_calls": 1,
            },
        )
    for region in regions:
        region["mean_state_probability"] = float(np.mean(state[region["start"] : region["end"]]))
    return sorted(regions, key=lambda region: region["start"])


def union_window_regions(windows: Sequence[tuple[int, Sequence[Mapping]]], best: int) -> list[dict]:
    """Every window's supported regions in the full-protein frame, as one list.

    ``windows`` holds each window's ``(start, repeat_regions)`` with window
    positions. The best window's regions enter first (its primary region stays
    ``primary``). The other windows' regions enter strongest wedge score first,
    labelled ``secondary``, and merge or trim exactly as within one protein;
    their primary regions enter only when they pass the gate a further call
    passes (mean residue-state probability >= 0.2, wedge score >= 12), since
    every window has a primary region whether or not it holds a repeat. Each
    region records the window it came from.
    """

    candidates = []
    for index, (offset, regions) in enumerate(windows):
        for region in regions:
            if (
                index != best
                and region["source"] == "primary"
                and not (
                    float(region.get("mean_state_probability", 0.0)) >= SECONDARY_MINIMUM_STATE
                    and float(region["wedge_score"]) >= SECONDARY_MINIMUM_SCORE
                )
            ):
                continue
            shifted = {
                **region,
                "start": int(region["start"]) + int(offset),
                "end": int(region["end"]) + int(offset),
                "window_index": index,
            }
            if index != best:
                shifted["source"] = "secondary"
            candidates.append(shifted)
    candidates.sort(
        key=lambda region: (
            region["window_index"] != best,
            region["source"] != "primary",
            -float(region["wedge_score"]),
        )
    )
    merged: list[dict] = []
    for candidate in candidates:
        add_region(merged, candidate)
    return sorted(merged, key=lambda region: region["start"])
