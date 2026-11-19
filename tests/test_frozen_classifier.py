import joblib
import numpy as np
import pytest

from talea import _frozen_classifier as classifier
from talea.artifacts import verified_runtime_assets
from talea.inference import TaleaPredictor
from talea.production import global_head_probabilities


@pytest.fixture(scope="module")
def fitted():
    assets = verified_runtime_assets()
    return joblib.load(assets.global_model), assets.sha256["global_model"]


def test_frozen_classifier_retains_input_validation_and_array_ownership(fitted):
    bundle, digest = fitted
    private = classifier.prepare_frozen_classifier(bundle, digest)
    if private is bundle:
        pytest.skip("runtime uses the ordinary classifier")
    frozen = private["model"]
    original = bundle["model"]
    assert original.named_steps["standardscaler"].copy
    assert not frozen.named_steps["standardscaler"].copy

    rng = np.random.default_rng(71)
    for width in (1, 8, 24, 96):
        rows = [{k: rng.standard_normal(1152).astype(np.float32) for k in bundle["feature_keys"]} for _ in range(width)]
        before = [[v.tobytes() for v in row.values()] for row in rows]
        assert global_head_probabilities(private, rows).tobytes() == global_head_probabilities(bundle, rows).tobytes()
        assert [[v.tobytes() for v in row.values()] for row in rows] == before
    for pipeline in (original, frozen):
        with pytest.raises(ValueError, match="features"):
            pipeline.predict_proba(np.zeros((1, 7)))
        with pytest.raises(ValueError, match="NaN"):
            pipeline.named_steps["pca"].transform(np.full((1, 4608), np.nan))


def test_unknown_model_or_runtime_uses_unmodified_classifier(fitted, monkeypatch):
    bundle, digest = fitted
    assert classifier.prepare_frozen_classifier(bundle, "other") is bundle
    monkeypatch.setattr(classifier.sklearn, "__version__", "unknown")
    assert classifier.prepare_frozen_classifier(bundle, digest) is bundle


def test_access_or_replacement_uses_mutable_public_model(fitted):
    bundle, digest = fitted
    predictor = TaleaPredictor.__new__(TaleaPredictor)
    predictor.global_bundle = bundle
    predictor._prediction_global_bundle = classifier.prepare_frozen_classifier(bundle, digest)
    assert predictor.global_bundle is predictor._prediction_global_bundle is bundle
    replacement = dict(bundle)
    predictor.global_bundle = replacement
    assert predictor._prediction_global_bundle is replacement
