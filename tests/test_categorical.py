"""Tests for out-of-fold target encoding (docs/results/FINDINGS.md §100).

The leak test is the load-bearing one. Naive target encoding fails in a way that makes the
model *worse* while leaving every aggregate score looking reasonable, so it has to be pinned
by a test that would catch its return rather than by a comment saying not to do it.
"""

from __future__ import annotations

import numpy as np
import pytest

from fintfm.inference.categorical import (
    CategoricalTargetEncoder,
    infer_categorical_features,
)


def test_out_of_fold_encoding_does_not_leak_the_row_s_own_label():
    """The failure this module exists to prevent, stated as a measurement.

    On a column where every level is unique, a row's label is the *only* evidence about its
    level. Naive encoding therefore reproduces the label almost exactly; out-of-fold encoding
    has no evidence at all and must fall back to the prior. Correlation between the encoded
    column and the label separates the two cleanly, and nothing else does.
    """
    rng = np.random.default_rng(0)
    n = 400
    X = np.arange(n, dtype=np.float64).reshape(-1, 1)  # every level unique
    y = rng.integers(0, 2, size=n)

    enc = CategoricalTargetEncoder(categorical_features=[0], random_state=0)
    oof = enc.fit_transform(X, y)[:, 0]

    # No out-of-fold evidence exists for a unique level, so every row gets the prior.
    # A constant column: correlation with the label is undefined, which is the point.
    assert np.allclose(oof, enc.prior_), "unique levels must fall back to the prior"
    assert np.ptp(oof) < 1e-6, f"the encoding must carry no row-level information: {np.ptp(oof)}"

    # The naive alternative, computed here only to show what is being avoided.
    naive = np.array([enc.maps_[0][float(v)] for v in X[:, 0]])
    assert np.corrcoef(naive, y)[0, 1] > 0.9, "naive encoding should leak, or this test is moot"


def test_encoding_recovers_a_real_level_effect():
    """Leak-freeness must not be bought by encoding nothing.

    With levels that genuinely differ in target rate and enough rows per level, the
    out-of-fold encoding has to order them correctly -- otherwise the previous test would pass
    trivially for an encoder that always returns the prior.
    """
    rng = np.random.default_rng(1)
    rates = {0: 0.1, 1: 0.5, 2: 0.9}
    levels = rng.integers(0, 3, size=3000)
    y = (rng.random(3000) < np.array([rates[v] for v in levels])).astype(int)
    X = levels.reshape(-1, 1).astype(np.float64)

    enc = CategoricalTargetEncoder(categorical_features=[0], random_state=0)
    oof = enc.fit_transform(X, y)[:, 0]

    means = [oof[levels == v].mean() for v in (0, 1, 2)]
    assert means[0] < means[1] < means[2], f"levels must be ordered by rate: {means}"
    assert abs(means[0] - 0.1) < 0.05 and abs(means[2] - 0.9) < 0.05


def test_smoothing_shrinks_a_rare_level_toward_the_prior():
    """A level seen twice must not be encoded as certainty."""
    y = np.array([1, 1] + [0] * 98)
    X = np.array([[0.0], [0.0]] + [[1.0]] * 98)

    enc = CategoricalTargetEncoder(categorical_features=[0], smoothing=10.0, random_state=0)
    enc.fit_transform(X, y)
    rare = enc.maps_[0][0.0]

    assert rare < 0.5, f"two positives out of two must not encode near 1.0, got {rare}"
    assert rare > enc.prior_, "it should still move above the prior, just not all the way"


def test_unseen_query_level_falls_back_to_the_prior():
    """High-cardinality columns guarantee unseen levels at query time."""
    y = np.array([0, 1] * 50)
    X = np.repeat(np.arange(10.0), 10).reshape(-1, 1)

    enc = CategoricalTargetEncoder(categorical_features=[0], random_state=0)
    enc.fit_transform(X, y)
    out = enc.transform(np.array([[999.0], [0.0]]))

    assert out[0, 0] == pytest.approx(enc.prior_), "an unseen level has no evidence"
    assert out[1, 0] == pytest.approx(enc.maps_[0][0.0])


def test_non_categorical_columns_pass_through_untouched():
    """An encoder that silently rewrites numeric columns would be a regression, not a fix."""
    rng = np.random.default_rng(2)
    X = rng.normal(size=(50, 4))
    X[:, 1] = rng.integers(0, 3, size=50)
    y = rng.integers(0, 2, size=50)

    out = CategoricalTargetEncoder(categorical_features=[1], random_state=0).fit_transform(X, y)

    for col in (0, 2, 3):
        assert np.allclose(out[:, col], X[:, col].astype(np.float32)), f"column {col} changed"


def test_multiclass_targets_fall_back_to_frequency_encoding():
    """Target statistics are binary-only for now; the fallback must still be ordered."""
    levels = np.array([0] * 70 + [1] * 30)
    X = levels.reshape(-1, 1).astype(np.float64)
    y = np.array([0, 1, 2] * 33 + [0])

    enc = CategoricalTargetEncoder(categorical_features=[0], random_state=0)
    out = enc.fit_transform(X, y)[:, 0]

    assert out[levels == 0].mean() > out[levels == 1].mean(), "frequency order must survive"
    assert 0.0 <= out.min() and out.max() <= 1.0


def test_infer_categorical_features_refuses_to_guess_from_level_counts():
    """A low-cardinality integer column may be genuinely ordinal; guessing would break it."""
    X = np.zeros((10, 3))
    assert infer_categorical_features(X) == []
    assert infer_categorical_features(X, dtypes=["float", "category", "object"]) == [1, 2]
    with pytest.raises(ValueError, match="2 entries for 3 columns"):
        infer_categorical_features(X, dtypes=["float", "category"])


def test_rejects_nonsense_configuration():
    with pytest.raises(ValueError, match="smoothing must be >= 0"):
        CategoricalTargetEncoder(smoothing=-1.0)
    with pytest.raises(ValueError, match="n_folds must be >= 2"):
        CategoricalTargetEncoder(n_folds=1)
    with pytest.raises(RuntimeError, match="fit_transform must be called"):
        CategoricalTargetEncoder(categorical_features=[0]).transform(np.zeros((2, 1)))


def test_continuous_target_gets_target_statistics_not_frequency():
    # 19 of TabArena's 51 datasets are regression. Frequency-encoding them would discard the
    # label information on the larger half of the suite, so this asserts the branch taken.
    from fintfm.inference.categorical import _supports_target_statistics

    rng = np.random.default_rng(0)
    y = rng.normal(size=200)
    assert _supports_target_statistics(y)
    X = np.column_stack([rng.integers(0, 4, size=200).astype(float), rng.normal(size=200)])
    enc = CategoricalTargetEncoder(categorical_features=[0], random_state=0)
    enc.fit_transform(X, y)
    assert not enc._frequency


def test_integer_multiclass_codes_stay_on_frequency():
    # Averaging class codes is the §100 mistake moved to the target side: class 3 is a name,
    # not a quantity.
    from fintfm.inference.categorical import _supports_target_statistics

    rng = np.random.default_rng(1)
    y = rng.integers(0, 5, size=200)
    assert not _supports_target_statistics(y)
    assert not _supports_target_statistics(y.astype(np.float64))


def test_binary_target_still_gets_target_statistics():
    from fintfm.inference.categorical import _supports_target_statistics

    rng = np.random.default_rng(2)
    assert _supports_target_statistics(rng.integers(0, 2, size=100))
    assert _supports_target_statistics(rng.integers(0, 2, size=100).astype(np.float64))


def test_continuous_target_encoding_is_out_of_fold():
    # The leak that matters is the same one: a row's own target inside its own encoding. A
    # constant-within-level target makes it visible -- with leakage the encoding reproduces
    # the target exactly, out of fold it cannot.
    rng = np.random.default_rng(3)
    levels = rng.integers(0, 5, size=300).astype(float)
    y = levels * 10.0 + rng.normal(scale=0.01, size=300)
    X = np.column_stack([levels, rng.normal(size=300)])
    enc = CategoricalTargetEncoder(categorical_features=[0], random_state=0)
    oof = enc.fit_transform(X, y)[:, 0]
    full = enc.transform(X)[:, 0]
    # out-of-fold and full-context encodings must differ; identical means no folding happened
    assert np.abs(oof - full).max() > 1e-6
