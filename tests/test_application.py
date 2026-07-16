import math

import pytest

from talea.application import (
    dense_ranks,
    frozen_operational_scores,
    frozen_spectrum_score,
)


def test_frozen_spectrum_score_applies_layer_calibration():
    row = {
        "layer_features": [
            {"layer": 13, "score": math.e**2 - 1},
            {"layer": 35, "score": math.e**4 - 1},
        ]
    }
    calibration = {
        "selected_layers_zero_based": [13, 35],
        "parameters": {
            "13": {
                "slope": 0.0,
                "intercept": 1.0,
                "control_residual_mean": 0.0,
                "control_residual_sd_population": 2.0,
            },
            "35": {
                "slope": 0.0,
                "intercept": 2.0,
                "control_residual_mean": 0.0,
                "control_residual_sd_population": 1.0,
            },
        },
    }
    result = frozen_spectrum_score(row, 256, calibration)
    assert result["layers"]["13"]["length_residual_z"] == pytest.approx(0.5)
    assert result["layers"]["35"]["length_residual_z"] == pytest.approx(2.0)
    assert result["frozen_esmpp_score"] == pytest.approx(1.25)


def test_frozen_operational_scores_and_no_call_floor():
    config = {
        "topology_score": {
            "candidate_count_correction": {
                "coefficient": -0.5,
                "intercept": 2.0,
                "residual_sd": 2.0,
                "no_call_corrected_score": -3.0,
            },
            "development_control_standardization": {"mean": 0.0, "sd": 2.0},
        },
        "frozen_pangu_standardization": {
            "development_control_mean": 1.0,
            "development_control_sd": 2.0,
        },
        "fusion": {
            "topology_weight": 0.5,
            "frozen_pangu_weight": 0.5,
        },
    }
    observed = frozen_operational_scores(
        "ok", {"candidate_count": 3, "open_path_score_minimum": 4.0}, 3.0, config
    )
    expected = 2.0 - 0.5 * math.log1p(3)
    assert observed["candidate_corrected_topology_score"] == pytest.approx(
        (4.0 - expected) / 2.0
    )
    missing = frozen_operational_scores("no_wedge_call", None, 3.0, config)
    assert missing["candidate_corrected_topology_score"] == -3.0
    assert missing["candidate_count"] == 0


def test_dense_ranks_are_descending_and_tie_aware():
    assert dense_ranks([2.0, 1.0, 2.0, -1.0]) == [1, 2, 1, 3]
    with pytest.raises(ValueError, match="finite"):
        dense_ranks([1.0, math.inf])
