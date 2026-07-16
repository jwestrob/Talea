"""Apply Talea's conditional repeat-architecture router.

The router is deliberately conditional: its probabilities distinguish among
elongated/open, closed/compact, and beads-on-string repeat architectures after
a sequence has been scored by Talea.  They are model outputs, not curated
structure annotations, and must not be interpreted as a broad repeat-versus-
nonrepeat decision.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np


ARCHITECTURE_CATEGORIES = (
    "elongated_open",
    "closed_compact",
    "beads_on_string",
)
ROUTER_FEATURE_NAMES = (
    "state_mean_logit",
    "state_q90_logit",
    "topology_control_z",
    "state_mean_logit_x_topology_control_z",
)


def probability_logit(value: float, epsilon: float = 1e-5) -> float:
    """Return a finite logit for a probability-like score."""

    number = float(value)
    if not math.isfinite(number):
        raise ValueError("probability must be finite")
    if not 0.0 <= number <= 1.0:
        raise ValueError("probability must be in [0, 1]")
    if not 0.0 < epsilon < 0.5:
        raise ValueError("epsilon must be in (0, 0.5)")
    clipped = min(max(number, epsilon), 1.0 - epsilon)
    return math.log(clipped / (1.0 - clipped))


def topology_control_z(row: Mapping) -> float:
    """Read either supported spelling of the frozen topology control z score."""

    for field in ("topology_control_z", "topology_development_control_z"):
        if row.get(field) is not None:
            value = float(row[field])
            if not math.isfinite(value):
                raise ValueError(f"{field} must be finite")
            return value
    record_id = row.get("record_id", "<unknown>")
    raise ValueError(f"topology prediction lacks a control z score: {record_id}")


def router_feature_vector(state: Mapping, topology: Mapping) -> np.ndarray:
    """Construct the router's frozen four-feature vector."""

    mean = probability_logit(float(state["mean_probability"]))
    q90 = probability_logit(float(state["q90_probability"]))
    topology_score = topology_control_z(topology)
    values = np.asarray(
        [mean, q90, topology_score, mean * topology_score], dtype=np.float64
    )
    if not np.isfinite(values).all():
        record_id = state.get("record_id", "<unknown>")
        raise ValueError(f"nonfinite router feature for {record_id}")
    return values


def _validated_model_arrays(
    model: Mapping,
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    feature_names = tuple(str(value) for value in model["feature_names"])
    if feature_names != ROUTER_FEATURE_NAMES:
        raise ValueError(
            "router feature contract disagrees with this Talea implementation"
        )
    categories = tuple(str(value) for value in model["categories_in_estimator_order"])
    if len(categories) != len(ARCHITECTURE_CATEGORIES) or set(categories) != set(
        ARCHITECTURE_CATEGORIES
    ):
        raise ValueError("router category contract is incomplete or unexpected")
    mean = np.asarray(model["scaler_mean"], dtype=np.float64)
    scale = np.asarray(model["scaler_scale"], dtype=np.float64)
    coefficients = np.asarray(model["coefficients"], dtype=np.float64)
    intercepts = np.asarray(model["intercepts"], dtype=np.float64)
    expected_features = len(ROUTER_FEATURE_NAMES)
    if mean.shape != (expected_features,) or scale.shape != (expected_features,):
        raise ValueError("router scaler dimensions disagree with its feature contract")
    if coefficients.shape != (len(categories), expected_features):
        raise ValueError("router coefficient dimensions disagree with its contract")
    if intercepts.shape != (len(categories),):
        raise ValueError("router intercept dimensions disagree with its contract")
    arrays = (mean, scale, coefficients, intercepts)
    if not all(np.isfinite(values).all() for values in arrays):
        raise ValueError("router model contains nonfinite parameters")
    if np.any(scale <= 0):
        raise ValueError("router scaler values must be positive")
    return categories, mean, scale, coefficients, intercepts


def _sigmoid(values: np.ndarray) -> np.ndarray:
    result = np.empty_like(values, dtype=np.float64)
    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def router_probabilities(
    feature_values: Sequence[float], model: Mapping
) -> dict[str, float]:
    """Reproduce scikit-learn's normalized one-vs-rest probabilities."""

    categories, mean, scale, coefficients, intercepts = _validated_model_arrays(model)
    values = np.asarray(feature_values, dtype=np.float64)
    if values.shape != (len(ROUTER_FEATURE_NAMES),) or not np.isfinite(values).all():
        raise ValueError("router feature vector has the wrong shape or nonfinite values")
    standardized = (values - mean) / scale
    independent = _sigmoid(coefficients @ standardized + intercepts)
    denominator = float(np.sum(independent))
    if not math.isfinite(denominator) or denominator <= 0:
        raise ValueError("router probability normalization failed")
    estimator_probabilities = independent / denominator
    by_category = {
        category: float(estimator_probabilities[index])
        for index, category in enumerate(categories)
    }
    return {category: by_category[category] for category in ARCHITECTURE_CATEGORIES}


def conditional_architecture_prediction(
    state: Mapping,
    topology: Mapping,
    model: Mapping,
    threshold: float = 0.5,
) -> dict:
    """Return conditional architecture probabilities and an optional hard call."""

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("category threshold must be in [0, 1]")
    probabilities = router_probabilities(
        router_feature_vector(state, topology), model
    )
    category = max(
        ARCHITECTURE_CATEGORIES,
        key=lambda name: (probabilities[name], -ARCHITECTURE_CATEGORIES.index(name)),
    )
    confidence = probabilities[category]
    return {
        "conditional_category_probabilities": probabilities,
        "conditional_category_confidence": confidence,
        "conditional_category_call": (
            category if confidence >= threshold else "repeat_architecture_unresolved"
        ),
        "category_call_threshold": float(threshold),
        "evidence_status": "UNVERIFIED model output; not a curated structure annotation",
        "scope": (
            "conditional architecture routing after Talea scoring; not a broad "
            "repeat-versus-nonrepeat call"
        ),
    }
