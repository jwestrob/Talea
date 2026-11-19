"""Private sklearn pipeline for predictions on Talea-owned feature matrices."""
from copy import deepcopy

import numpy as np
import sklearn
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


_GLOBAL_MODEL_SHA256 = "8701dad34e3392a0f84884e88d0bf3e9741a75903882ec7658bbb8b2fb2161de"


class _FrozenPCA(PCA):
    def _transform(self, X, xp, x_is_centered=False):
        # PCA.transform still validates X. Keep sklearn 1.6.1's order of
        # projection, centering and whitening, including its scale clipping.
        projected = X @ self.components_.T
        if not x_is_centered:
            projected -= self._projected_mean
        if self.whiten:
            scale = xp.sqrt(self.explained_variance_)
            minimum = xp.finfo(scale.dtype).eps
            scale[scale < minimum] = minimum
            projected /= scale
        return projected


def prepare_frozen_classifier(bundle, artifact_sha256):
    """Specialize only the verified model and tested sklearn/NumPy arithmetic."""
    if (artifact_sha256 != _GLOBAL_MODEL_SHA256 or sklearn.__version__ != "1.6.1"
            or np.__version__ not in ("2.1.3", "2.4.6")):
        return bundle
    model = bundle["model"]
    steps = (("standardscaler", StandardScaler), ("pca", PCA), ("logisticregression", LogisticRegression))
    if type(model) is not Pipeline or tuple((name, type(step)) for name, step in model.steps) != steps:
        return bundle

    pipeline = deepcopy(model)
    pipeline.named_steps["standardscaler"].set_params(copy=False)
    pca = pipeline.named_steps["pca"]
    frozen = _FrozenPCA(**pca.get_params(deep=False))
    frozen.__dict__.update(vars(pca))
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        frozen._projected_mean = np.reshape(frozen.mean_, (1, -1)) @ frozen.components_.T
    pipeline.set_params(pca=frozen)
    return {**bundle, "model": pipeline}
