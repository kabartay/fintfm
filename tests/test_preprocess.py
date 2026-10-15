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


# --- AU5.12: the "squash" transform and SVD augmentation, used only by ensemble members ---


def test_squash_is_bounded_monotone_and_linear_near_the_centre():
    """Robust scaling then ``z / sqrt(1 + (z/3)^2)``: order kept, tails bounded, centre intact."""
    from fintfm.inference.preprocess import SQUASH_SCALE

    rng = np.random.default_rng(0)
    X = rng.standard_cauchy(size=(500, 2)).astype(np.float32)  # heavy tails on purpose
    X[3, 1] = np.nan
    out = FeatureConditioner("squash").fit(X).transform(X)
    assert np.isnan(out[3, 1]) and np.isfinite(np.delete(out.ravel(), 3 * 2 + 1)).all()
    assert np.abs(out[np.isfinite(out)]).max() < SQUASH_SCALE
    order = np.argsort(X[:, 0])
    assert np.all(np.diff(out[order, 0]) >= 0)  # monotone: ranking preserved
    # by hand, from the formula: median-centred, IQR-scaled, soft-clipped
    col = X[:, 0]
    z = (col - np.median(col)) / (np.quantile(col, 0.75) - np.quantile(col, 0.25))
    np.testing.assert_allclose(out[:, 0], z / np.sqrt(1 + (z / 3) ** 2), rtol=1e-4, atol=1e-5)
    # near the median the map is close to the identity on the robust scale
    near = np.abs(z) < 0.3
    np.testing.assert_allclose(out[near, 0], z[near], atol=0.01)


def test_squash_falls_back_to_std_for_a_zero_iqr_column_and_uses_training_rows_only():
    X = np.zeros((100, 2), dtype=np.float32)
    X[:3, 0] = [5.0, -5.0, 10.0]  # IQR 0, but not constant
    X[:, 1] = 7.0  # constant: neither IQR nor std, left merely centred
    cond = FeatureConditioner("squash").fit(X)
    out = cond.transform(X)
    assert np.isfinite(out).all()
    assert out[2, 0] > out[0, 0] > 0 > out[1, 0]
    np.testing.assert_array_equal(out[:, 1], 0.0)
    # a query is scaled by the training rows' statistics, not its own
    q = cond.transform(np.array([[5.0, 8.0]], dtype=np.float32))
    assert q[0, 0] == pytest.approx(out[0, 0])
    assert q[0, 1] == pytest.approx(1.0 / np.sqrt(1 + (1.0 / 3) ** 2))


def test_squash_clamps_infinities_like_the_other_transforms():
    X = _with_infinities()
    out = FeatureConditioner("squash").fit(X).transform(X)
    finite = np.where(np.isfinite(X[:, 0]), X[:, 0], np.nan)
    assert out[5, 0] == pytest.approx(out[np.nanargmax(finite), 0])
    assert out[6, 0] == pytest.approx(out[np.nanargmin(finite), 0])


def test_svd_augmenter_appends_the_context_projections_and_passes_nan_through():
    from fintfm.inference.preprocess import SVDAugmenter

    rng = np.random.default_rng(0)
    ctx = rng.normal(size=(50, 5)).astype(np.float32)
    ctx[4, 2] = np.nan
    aug = SVDAugmenter(2).fit(ctx)
    q = rng.normal(size=(7, 5)).astype(np.float32)
    q[1, 0] = np.nan
    out = aug.transform(q)
    assert out.shape == (7, 7)
    np.testing.assert_array_equal(out[:, :5], q)  # originals untouched, NaN included
    # the appended columns are projections on the context's leading right singular vectors
    _, _, vt = np.linalg.svd(np.nan_to_num(ctx).astype(np.float64), full_matrices=False)
    for j in range(2):
        proj = np.nan_to_num(q) @ vt[j]
        assert abs(abs(np.corrcoef(out[:, 5 + j], proj)[0, 1]) - 1) < 1e-5
        assert np.allclose(np.abs(out[:, 5 + j]), np.abs(proj), atol=1e-4)
    # signs fixed by the largest loading, so refitting the same context reproduces the bytes
    np.testing.assert_array_equal(SVDAugmenter(2).fit(ctx).transform(q), out)


def test_svd_augmenter_caps_components_and_zero_appends_nothing():
    from fintfm.inference.preprocess import SVDAugmenter

    ctx = np.random.default_rng(1).normal(size=(3, 4)).astype(np.float32)
    assert SVDAugmenter(10).fit(ctx).components_.shape == (3, 4)  # capped at the row count
    np.testing.assert_array_equal(SVDAugmenter(0).fit(ctx).transform(ctx), ctx)
    with pytest.raises(ValueError):
        SVDAugmenter(-1)
