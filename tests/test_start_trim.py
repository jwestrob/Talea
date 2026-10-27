import json

import numpy as np

from talea.artifacts import ASSET_ROOT
from talea.phase_ssm import (
    boundary_probe_parameters,
    decode_units,
    phase_votes,
    place_units_start_trim,
    runtime_config,
    start_trim_config,
    trim_units,
)


def _phase_config() -> dict:
    return runtime_config(json.loads((ASSET_ROOT / "phase_ssm.json").read_text()))


def _trim_config() -> dict:
    return start_trim_config(json.loads((ASSET_ROOT / "start_trim.json").read_text()))


def _units(boundaries: list[int]) -> list[dict]:
    return [
        {"start": left, "end": right}
        for left, right in zip(boundaries[:-1], boundaries[1:])
    ]


def test_start_scores_place_units_without_phase_votes():
    length, boundaries = 140, [10, 32, 51, 75, 96, 118]
    scores = np.full(length + 1, -4.0)
    scores[boundaries] = 4.0
    config = {"decoder": dict(_phase_config()["decoder"], phase_weight=0.0)}
    units = decode_units(
        np.zeros(length, dtype=complex),
        [],
        21.0,
        10,
        118,
        config,
        start_scores=scores[:length],
        start_weight=32.0,
    )
    assert units == _units(boundaries)


def test_zero_start_weight_reproduces_the_phase_ssm_decode():
    rng = np.random.default_rng(3)
    votes = rng.normal(size=150) + 1j * rng.normal(size=150)
    config = _phase_config()
    plain = decode_units(votes, [], 18.0, 12, 140, config)
    weighted = decode_units(
        votes, [], 18.0, 12, 140, config, start_scores=rng.normal(size=150)
    )
    assert plain == weighted


def test_trim_drops_the_weaker_end_first_and_keeps_three_units():
    units = _units([0, 10, 20, 30, 40, 50])
    region = np.zeros(50)
    region[0:10] = -3.0
    region[40:50] = -1.0
    assert trim_units(units, region, -0.5, 3) == _units([10, 20, 30, 40])
    assert trim_units(units, region, -2.0, 3) == _units([10, 20, 30, 40, 50])
    # ties drop the first unit
    assert trim_units(units, np.full(50, -9.0), 0.0, 3) == units[2:]


def test_strand_extents_keep_the_phase_ssm_decode_before_trimming():
    rng = np.random.default_rng(11)
    length, period = 160, 20.0
    registration = {
        "period": period,
        "start": 10,
        "end": 150,
        "units": _units(list(range(10, 151, 20))),
    }
    probabilities = np.column_stack(
        [np.full(length, 0.05), rng.uniform(size=length), rng.uniform(size=length)]
    )
    scores = np.column_stack([rng.normal(size=length), np.full(length, 3.0)])
    scores[10:40, 1] = -3.0
    phase_config = _phase_config()
    votes = phase_votes(probabilities, period, phase_config["coefficients"])
    expected = trim_units(
        decode_units(votes, [], period, 10, 150, phase_config), scores[:, 1], 0.5, 3
    )
    units, label = place_units_start_trim(
        registration, probabilities, scores, [], phase_config, _trim_config()
    )
    assert label == "start_trim"
    assert units == expected
    scores[:, 0] = rng.normal(size=length) * 10
    again, _ = place_units_start_trim(
        registration, probabilities, scores, [], phase_config, _trim_config()
    )
    assert again == units


def test_start_trim_asset_matches_the_runtime_contract():
    asset = json.loads((ASSET_ROOT / "start_trim.json").read_text())
    weights, bias = boundary_probe_parameters(asset)
    assert weights.shape == (1152, 2)
    assert bias.shape == (2,)
    assert asset["probes"]["hidden_state_layer_zero_based"] == 35
    decoder = _trim_config()["decoder"]
    assert decoder["helix_threshold"] == 0.2
    assert decoder["helix"] == {
        "phase_weight": 0.0,
        "start_weight": 32.0,
        "trim_threshold": -0.5,
    }
    assert decoder["strand"]["phase_weight"] == 256.0
    assert decoder["strand"]["trim_threshold"] == 0.5
