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

The registered extent overshoots the curated repeat by most of a unit on each
side, so by default ("state" primary extent) the primary region is then cut to
the U-Net residue-state track: its extent becomes the maximum-sum stretch of
(state - 0.2) inside the region widened by max(2 periods, half its length),
widened again to cover the wedge call's interval, and its units are those whose
midpoints fall inside, clipped to it. The state therefore trims only what the
registration adds around the wedge's own evidence; where the U-Net fires on
part of a repeat it has not seen, the wedge interval is kept. The cut follows
the further calls' merges and precedes the state regions; a further region then
keeps only its longest stretch outside the primary region, provided that stretch
spans two periods. 0.2 is the same threshold that admits a further call.
Then, when a further wedge region carries more than twice the primary region's
residue-state mass (the sum of state - 0.2 over its own state extent), it becomes
the primary region (``choose_primary_by_state``) and the registration takes its
units and period (``registration_for_primary``); the wedge call's registration
stays under ``wedge_call_registration``, and ``wedge_call`` and every score still
describe the strongest wedge call. The registration is cut the same way
(``trim_registration``, with the same wedge floor): it keeps the units at least
half inside that stretch,
whole, and spans them; the registered extent and units stay under
``registered_start``, ``registered_end`` and ``registered_units``. A stretch
shorter than 8 residues leaves the region unchanged, and one that keeps fewer
than three units leaves the registration unchanged. "registered" keeps v0.3.2's
regions and registration.

The residue-state track adds regions of its own ("state" regions). The wedge
proposes regions only where attention forms stripes parallel to the diagonal at
multiples of one period, so a stretch the U-Net flags can go unproposed (a run
of beta hairpins, whose contacts cross the diagonal, for one). Every stretch
where the track stays at or above 0.5 for at least 40 residues becomes a region
over its longest part outside the wedge regions, provided that part spans 40
residues. A state region has no period, wedge score or units; wedge regions
never merge into it.

Each region carries units: the primary region the registration's units, a
further region the units of its own call, registered and placed exactly as the
primary call's are. A merge adds the incoming units whose midpoints fall
outside the region's units, and a trimmed region keeps the units whose
midpoints fall inside it; units are clipped where they meet and to their
region's extent.
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
STATE_RUN_MINIMUM_PROBABILITY = 0.5
STATE_RUN_MINIMUM_LENGTH = 40
PRIMARY_EXTENTS = ("state", "registered")
PRIMARY_STATE_THRESHOLD = SECONDARY_MINIMUM_STATE
PRIMARY_MINIMUM_STRETCH = 8
TRIMMED_MINIMUM_UNITS = 3
STATE_CHOICE_RATIO = 2.0


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


def state_runs(
    state: np.ndarray,
    minimum_probability: float = STATE_RUN_MINIMUM_PROBABILITY,
    minimum_length: int = STATE_RUN_MINIMUM_LENGTH,
) -> list[tuple[int, int]]:
    """Maximal ``[start, end)`` stretches where ``state >= minimum_probability``, at least ``minimum_length`` long."""

    above = np.concatenate([[False], np.asarray(state) >= minimum_probability, [False]])
    edges = np.flatnonzero(np.diff(above.astype(np.int8)))
    return [
        (int(start), int(end))
        for start, end in zip(edges[::2], edges[1::2])
        if end - start >= minimum_length
    ]


def _units_inside(units: Sequence[Mapping], start: int, end: int) -> list[dict]:
    """Units whose midpoints lie in ``[start, end)``, clipped to it."""

    inside = []
    for unit in units:
        first, last = int(unit["start"]), int(unit["end"])
        if start <= (first + last) / 2 < end:
            first, last = max(first, start), min(last, end)
            if last > first:
                inside.append({"start": first, "end": last})
    return inside


def _merge_units(existing: Sequence[Mapping], incoming: Sequence[Mapping]) -> list[dict]:
    """Existing units plus incoming ones whose midpoints fall outside them, clipped where they meet."""

    units = [{"start": int(unit["start"]), "end": int(unit["end"])} for unit in existing]
    for unit in incoming:
        first, last = int(unit["start"]), int(unit["end"])
        middle = (first + last) / 2
        if not any(other["start"] <= middle < other["end"] for other in units):
            units.append({"start": first, "end": last})
    units.sort(key=lambda unit: (unit["start"], unit["end"]))
    resolved: list[dict] = []
    for unit in units:
        if resolved and unit["start"] < resolved[-1]["end"]:
            unit = {"start": resolved[-1]["end"], "end": unit["end"]}
            if unit["end"] <= unit["start"]:
                continue
        resolved.append(unit)
    return resolved


def add_region(regions: list[dict], candidate: Mapping) -> None:
    """Add ``candidate`` to ``regions`` in place, merging or trimming it as described above.

    A merge extends the existing region, adds the candidate's wedge calls to
    its count and its units to the region's; every other field of the
    existing region is kept. A candidate without a period (a state region)
    never merges: it keeps its longest free stretch if that spans 40 residues.
    """

    start, end = int(candidate["start"]), int(candidate["end"])
    if candidate.get("period") is None:
        start, end = _longest_free_stretch(start, end, regions)
        if end - start >= STATE_RUN_MINIMUM_LENGTH:
            regions.append({**candidate, "start": start, "end": end, "units": []})
        return
    period = float(candidate["period"])
    target = next(
        (
            region
            for region in regions
            if region.get("period") is not None
            and abs(period / region["period"] - 1.0) <= MERGE_PERIOD_TOLERANCE
            and start <= region["end"] + min(period, region["period"])
            and end >= region["start"] - min(period, region["period"])
        ),
        None,
    )
    if target is not None:
        target["start"] = min(target["start"], start)
        target["end"] = max(target["end"], end)
        target["wedge_calls"] += int(candidate.get("wedge_calls", 1))
        target["units"] = _units_inside(
            _merge_units(target.get("units", []), candidate.get("units", [])),
            target["start"],
            target["end"],
        )
        return
    start, end = _longest_free_stretch(start, end, regions)
    if end - start >= 2 * period:
        units = _units_inside(candidate.get("units", []), start, end)
        regions.append({**candidate, "start": start, "end": end, "units": units})


def max_sum_segment(values: np.ndarray, start: int, end: int) -> tuple[int, int]:
    """The ``[first, last)`` stretch of ``values[start:end]`` with the largest sum (earliest on ties)."""

    best, best_first, best_last = -math.inf, start, start
    running, running_first = 0.0, start
    for index in range(start, end):
        value = float(values[index])
        if running <= 0.0:
            running, running_first = value, index
        else:
            running += value
        if running > best:
            best, best_first, best_last = running, running_first, index + 1
    return best_first, best_last


def state_extent(
    state: np.ndarray,
    start: int,
    end: int,
    period: float,
    floor: tuple[int, int] | None = None,
) -> tuple[int, int] | None:
    """Maximum-sum stretch of (state - 0.2) inside ``[start, end)`` widened by max(2 periods, half its length).

    The stretch is widened to cover ``floor`` (the wedge call's interval), so
    the state never cuts into the wedge's own repeat evidence. None when the
    stretch is shorter than 8 residues.
    """

    values = np.asarray(state, dtype=np.float64)
    pad = int(max(2.0 * float(period), 0.5 * (end - start)))
    first, last = max_sum_segment(
        values - PRIMARY_STATE_THRESHOLD, max(0, start - pad), min(len(values), end + pad)
    )
    if last - first < PRIMARY_MINIMUM_STRETCH:
        return None
    if floor is not None:
        first, last = min(first, int(floor[0])), max(last, int(floor[1]))
    return first, last


def trim_registration(
    registration: Mapping | None,
    state_track: np.ndarray,
    floor: tuple[int, int] | None = None,
) -> Mapping | None:
    """The registration cut to the residue-state track (see the module docstring).

    ``floor`` is the wedge call's interval, which the cut never enters.
    """

    if registration is None or not registration.get("units"):
        return registration
    start, end = int(registration["start"]), int(registration["end"])
    extent = state_extent(state_track, start, end, float(registration["period"]), floor)
    if extent is None:
        return registration
    first, last = extent
    kept = [
        {"start": int(unit["start"]), "end": int(unit["end"])}
        for unit in registration["units"]
        if min(last, int(unit["end"])) - max(first, int(unit["start"]))
        >= 0.5 * (int(unit["end"]) - int(unit["start"]))
    ]
    if len(kept) < TRIMMED_MINIMUM_UNITS:
        return registration
    return {
        **registration,
        "registered_start": start,
        "registered_end": end,
        "registered_units": [
            {"start": int(unit["start"]), "end": int(unit["end"])} for unit in registration["units"]
        ],
        "start": kept[0]["start"],
        "end": kept[-1]["end"],
        "units": kept,
        "n_units": len(kept),
    }


def cut_primary_to_state(
    regions: list[dict], state: np.ndarray, floor: tuple[int, int] | None = None
) -> None:
    """Cut ``regions[0]`` (the primary region) in place to its state extent (never inside ``floor``).

    The registered extent is kept under ``registered_start`` and
    ``registered_end``. Every further region then keeps only its longest
    stretch outside the primary region, with the units inside it, and is
    dropped unless that stretch spans two of its periods.
    """

    primary = regions[0]
    extent = state_extent(state, primary["start"], primary["end"], primary["period"], floor)
    if extent is None:
        return
    primary["registered_start"], primary["registered_end"] = primary["start"], primary["end"]
    primary["start"], primary["end"] = extent
    primary["units"] = _units_inside(primary["units"], *extent)
    kept = [primary]
    for region in regions[1:]:
        start, end = _longest_free_stretch(region["start"], region["end"], [primary])
        if end - start >= 2 * float(region["period"]):
            region["start"], region["end"] = start, end
            region["units"] = _units_inside(region.get("units", []), start, end)
            kept.append(region)
    regions[:] = kept


def _state_mass(state: np.ndarray, start: int, end: int) -> float:
    return float(np.sum(np.asarray(state[start:end], dtype=np.float64) - PRIMARY_STATE_THRESHOLD))


def choose_primary_by_state(regions: list[dict], state: np.ndarray) -> None:
    """Make a further region the primary one, in place, when it carries much more residue-state evidence.

    Each further region's state extent is its maximum-sum stretch of (state - 0.2), never cut inside its
    wedge interval (its extent less half a period on each side); its state mass is the sum of (state - 0.2)
    over that extent, and the primary region's mass is taken over its own (already cut) extent. When the
    region with the most mass is a further region with at least three units and its mass exceeds twice the
    primary region's, it becomes the primary region (``selected_by: state_mass``) over its state extent with
    its units inside, the former primary region becomes a further region, and every other region keeps only
    its longest stretch outside the new primary region, if that stretch spans two of its periods. The new
    primary region carries ``registration_units`` (its whole units at least half inside its extent) for
    ``registration_for_primary``.
    """

    primary = regions[0]
    best, best_mass, best_extent = primary, _state_mass(state, primary["start"], primary["end"]), None
    primary_mass = best_mass
    for region in regions[1:]:
        if region.get("period") is None:
            continue
        period = float(region["period"])
        floor = (int(round(region["start"] + 0.5 * period)), int(round(region["end"] - 0.5 * period)))
        extent = state_extent(state, int(region["start"]), int(region["end"]), period, floor) or (
            int(region["start"]),
            int(region["end"]),
        )
        mass = _state_mass(state, *extent)
        if mass > best_mass:
            best, best_mass, best_extent = region, mass, extent
    if (
        best is primary
        or len(best.get("units") or []) < TRIMMED_MINIMUM_UNITS
        or best_mass <= STATE_CHOICE_RATIO * max(primary_mass, 1e-9)
    ):
        return
    first, last = best_extent
    whole = [
        {"start": int(unit["start"]), "end": int(unit["end"])}
        for unit in best["units"]
        if min(last, int(unit["end"])) - max(first, int(unit["start"]))
        >= 0.5 * (int(unit["end"]) - int(unit["start"]))
    ]
    primary["source"] = "secondary"
    best.update(
        source="primary",
        selected_by="state_mass",
        start=first,
        end=last,
        units=_units_inside(best["units"], first, last),
        registration_units=whole,
    )
    kept = [best]
    for region in regions:
        if region is best:
            continue
        start, end = _longest_free_stretch(region["start"], region["end"], [best])
        if end - start >= 2 * float(region["period"]):
            region["start"], region["end"] = start, end
            region["units"] = _units_inside(region.get("units", []), start, end)
            kept.append(region)
    regions[:] = kept


def registration_for_primary(registration: Mapping | None, regions: Sequence[dict]) -> Mapping | None:
    """The registration of a primary region chosen by state mass (unchanged otherwise).

    Takes the new primary region's whole units at least half inside its extent, its period and their span;
    the wedge call's registration stays under ``wedge_call_registration``. Fewer than three such units leave
    the registration unchanged. Removes ``registration_units`` from the region either way.
    """

    primary = next((region for region in regions if region.get("source") == "primary"), None)
    units = primary.pop("registration_units", None) if primary is not None else None
    if registration is None or not units or len(units) < TRIMMED_MINIMUM_UNITS:
        return registration
    return {
        **registration,
        "wedge_call_registration": {
            key: registration[key] for key in ("start", "end", "period", "units", "n_units") if key in registration
        },
        "selected_region": "state_mass",
        "start": units[0]["start"],
        "end": units[-1]["end"],
        "period": float(primary["period"]),
        "units": units,
        "n_units": len(units),
    }


def repeat_regions(
    topology: Mapping, state_track: np.ndarray, primary_extent: str = "state"
) -> list[dict]:
    """Primary region, accepted further calls and state regions, sorted by position.

    ``topology`` is a ``process_attention_record`` row; ``state_track`` is the
    U-Net residue-state probability for the same protein. ``primary_extent``
    "state" cuts the primary region to the state track, "registered" keeps the
    registered extent (see the module docstring).
    """

    if primary_extent not in PRIMARY_EXTENTS:
        raise ValueError(f"unsupported primary extent: {primary_extent}")

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
            "units": [
                {"start": int(unit["start"]), "end": int(unit["end"])}
                for unit in registration.get("units") or []
            ],
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
                "units": call.get("units") or [],
            },
        )
    if primary_extent == "state":
        cut_primary_to_state(regions, state, (int(wedge["start"]), int(wedge["end"])))
        choose_primary_by_state(regions, state)
    for start, end in state_runs(state):
        add_region(
            regions,
            {
                "start": start,
                "end": end,
                "period": None,
                "wedge_score": None,
                "source": "state",
                "wedge_calls": 0,
                "units": [],
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
    every window has a primary region whether or not it holds a repeat.
    State regions enter after every wedge region. Each region records the
    window it came from.
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
                "units": [
                    {"start": int(unit["start"]) + int(offset), "end": int(unit["end"]) + int(offset)}
                    for unit in region.get("units", [])
                ],
                "window_index": index,
            }
            if index != best and region["source"] != "state":
                shifted["source"] = "secondary"
            candidates.append(shifted)
    candidates.sort(
        key=lambda region: (
            region["source"] == "state",
            region["window_index"] != best,
            region["source"] != "primary",
            -float(region["wedge_score"] or 0.0),
        )
    )
    merged: list[dict] = []
    for candidate in candidates:
        add_region(merged, candidate)
    return sorted(merged, key=lambda region: region["start"])
