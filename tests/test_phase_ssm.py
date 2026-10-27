import json
import math

import numpy as np
import pytest

from talea.artifacts import ASSET_ROOT
from talea.phase_ssm import (
    PROBE_CLASSES,
    apply_placement,
    decode_units,
    demodulate,
    lag_evidence,
    length_emissions,
    place_units,
    probe_parameters,
    runtime_config,
    unit_tables,
)


def _asset() -> dict:
    return json.loads((ASSET_ROOT / "phase_ssm.json").read_text())


def _votes_for_units(length: int, boundaries: list[int], lead: int, tail: int):
    """Unit-phase votes exp(2 pi i phi) for full units between ``boundaries``."""
    phase = np.zeros(length)
    first = boundaries[1] - boundaries[0]
    phase[: boundaries[0]] = (lead - boundaries[0] + np.arange(boundaries[0])) / lead
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        phase[left:right] = (np.arange(right - left)) / (right - left)
    phase[boundaries[-1] :] = (np.arange(length - boundaries[-1])) / tail
    assert first > 0
    return np.exp(2j * np.pi * phase)


def test_demodulation_reads_the_phase_of_a_periodic_signal():
    period, offset = 17.5, 0.8
    positions = np.arange(200)
    signal = np.cos(2 * np.pi * positions / period + offset)
    angle = np.angle(demodulate(signal, period))
    expected = 2 * np.pi * positions / period + offset
    interior = slice(30, 170)
    error = np.angle(np.exp(1j * (angle[interior] - expected[interior])))
    assert np.max(np.abs(error)) < 0.1


def test_unit_tables_match_direct_sums():
    rng = np.random.default_rng(7)
    n, minimum_length, count, weight = 37, 6, 5, 3.0
    votes = rng.normal(size=n) + 1j * rng.normal(size=n)
    emissions = rng.normal(size=(count, n))
    full, lead, tail = unit_tables(votes, emissions, minimum_length, weight)

    def score(position: int, offset: int, unit_length: int, r: int) -> float:
        rotation = np.exp(-2j * np.pi * offset / unit_length)
        phase = weight * float(np.real(votes[position] * rotation))
        return phase + emissions[r, position]

    for r in range(count):
        unit_length = minimum_length + r
        for start in range(n + 1):
            if start + unit_length <= n:
                direct = sum(
                    score(start + j, j, unit_length, r) for j in range(unit_length)
                )
                assert full[start, r] == pytest.approx(direct, abs=1e-9)
            else:
                assert full[start, r] == -np.inf
            if start < unit_length:
                direct = sum(
                    score(i, unit_length - start + i, unit_length, r)
                    for i in range(start)
                )
                assert lead[start, r] == pytest.approx(direct, abs=1e-9)
            if n - start < unit_length:
                direct = sum(
                    score(i, i - start, unit_length, r) for i in range(start, n)
                )
                assert tail[start, r] == pytest.approx(direct, abs=1e-9)


def test_decoder_recovers_uneven_units_and_a_substantial_terminal_unit():
    boundaries = [6, 26, 48, 69, 89, 110]
    votes = _votes_for_units(122, boundaries, lead=20, tail=21)
    units = decode_units(votes, [], 21.0, 0, 122, runtime_config(_asset()))
    # the 6 leading residues fall short of the 0.45 terminal fraction; the 12
    # trailing residues clear it
    assert [(unit["start"], unit["end"]) for unit in units] == [
        (6, 26),
        (26, 48),
        (48, 69),
        (69, 89),
        (89, 110),
        (110, 122),
    ]


def test_decoder_works_inside_an_extent_and_keeps_protein_coordinates():
    boundaries = [40, 60, 80, 100, 120]
    votes = _votes_for_units(160, boundaries, lead=20, tail=20)
    units = decode_units(votes, [], 20.0, 40, 120, runtime_config(_asset()))
    assert [(unit["start"], unit["end"]) for unit in units] == [
        (40, 60),
        (60, 80),
        (80, 100),
        (100, 120),
    ]


def test_placement_falls_back_when_fewer_than_three_units_decode():
    registration = {"period": 20.0, "start": 0, "end": 30, "units": []}
    probabilities = np.full((30, len(PROBE_CLASSES)), 0.5)
    config = runtime_config(_asset())
    assert place_units(registration, probabilities, [], config) is None


def test_apply_placement_keeps_the_comb_alongside():
    comb = {
        "period": 10.0,
        "start": 0,
        "end": 30,
        "n_units": 3,
        "units": [
            {"start": 0, "end": 10},
            {"start": 10, "end": 20},
            {"start": 20, "end": 30},
        ],
    }
    decoded = [
        {"start": 2, "end": 11},
        {"start": 11, "end": 21},
        {"start": 21, "end": 29},
    ]
    placed = apply_placement(comb, decoded)
    assert placed["unit_placement"] == "phase_ssm"
    assert (placed["start"], placed["end"], placed["n_units"]) == (2, 29, 3)
    assert placed["comb_units"] == comb["units"]
    assert placed["unit_length_median"] == 9.0
    fallback = apply_placement(comb, None)
    assert fallback["unit_placement"] == "comb_fallback"
    assert fallback["units"] == comb["units"]


def test_lag_evidence_averages_both_partners():
    tracks = [np.empty(0), np.array([1.0, 2.0, 3.0])]
    evidence = lag_evidence(tracks, 4, 2)
    np.testing.assert_allclose(evidence[1], [1.0, 1.5, 2.5, 3.0])
    assert not evidence[2].any()
    emissions = length_emissions(evidence, 1, 2)
    np.testing.assert_allclose(np.median(emissions, axis=0), 0.0)


def test_phase_asset_matches_the_runtime_contract():
    asset = _asset()
    weights, bias = probe_parameters(asset)
    assert weights.shape == (1152, 3)
    assert bias.shape == (3,)
    assert asset["probes"]["hidden_state_layer_zero_based"] == 35
    config = runtime_config(asset)
    assert len(config["coefficients"]) == 3
    decoder = config["decoder"]
    assert decoder["phase_weight"] == 256.0
    assert decoder["length_change_penalty"] == 8.0
    assert math.isclose(decoder["terminal_fraction"], 0.45)
