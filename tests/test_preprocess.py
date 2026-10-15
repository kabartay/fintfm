"""Direct tests of the feature conditioner in ``fintfm.inference.preprocess``.

A financial ratio with a near-zero denominator produces a genuine ``+-inf``, so every
transform must survive one rather than crash the run over a single cell.
"""

import numpy as np
import pytest

from fintfm.inference.preprocess import FeatureConditioner


def _with_infinities(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(200, 3)).astype(np.float32)
    X[5, 0] = np.inf
    X[6, 0] = -np.inf
    X[7, 1] = np.nan
    return X


@pytest.mark.parametrize("kind", ["rank", "power"])
def test_infinities_are_clamped_to_the_finite_extremes(kind):
    """``+-inf`` maps where the column's finite max/min map, and NaN stays NaN.

    Audit AU1.3: the default ``"rank"`` transform passed infinities to
    ``QuantileTransformer``, which raises, while ``"power"`` already clamped them.
    """
    X = _with_infinities()
    cond = FeatureConditioner(kind).fit(X)
    out = cond.transform(X)
    assert np.isnan(out[7, 1])
    assert np.isfinite(np.delete(out.ravel(), 7 * 3 + 1)).all()
    finite = np.where(np.isfinite(X[:, 0]), X[:, 0], np.nan)
    hi, lo = np.nanargmax(finite), np.nanargmin(finite)
    assert out[5, 0] == pytest.approx(out[hi, 0])
    assert out[6, 0] == pytest.approx(out[lo, 0])
    # unseen rows are clamped with the bounds fitted on the training rows
    query = np.array([[np.inf, 0.0, 0.0], [-np.inf, 0.0, 0.0]], dtype=np.float32)
    got = cond.transform(query)
    assert got[0, 0] == pytest.approx(out[hi, 0])
    assert got[1, 0] == pytest.approx(out[lo, 0])
