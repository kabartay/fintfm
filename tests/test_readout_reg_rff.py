from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fintfm.readout.rff import RFFRidgeCV

ALPHAS = np.logspace(-3, 5, 17)


def nonlinear(n: int, seed: int):
    """A target with a sine and an interaction that no linear map captures."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    y = np.sin(2.0 * X[:, 0]) + X[:, 1] * X[:, 2] + 0.5 * X[:, 3] + rng.normal(scale=0.05, size=n)
    return X, y


def rmse(a, b) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def test_rff_beats_ridge_on_nonlinear_target():
    X, y = nonlinear(1500, 0)
    Xt, yt = nonlinear(500, 1)
    ridge = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS)).fit(X, y)
    rff = make_pipeline(
        StandardScaler(), RFFRidgeCV(alphas=ALPHAS, n_features=400, random_state=0)
    ).fit(X, y)
    assert rmse(rff.predict(Xt), yt) < 0.6 * rmse(ridge.predict(Xt), yt)


def test_rff_not_worse_on_linear_target():
    rng = np.random.default_rng(2)
    coef = np.array([1.0, -2.0, 0.5, 0.0, 1.5])
    X = rng.normal(size=(600, 5))
    y = X @ coef + rng.normal(scale=0.1, size=600)
    Xt = rng.normal(size=(300, 5))
    yt = Xt @ coef
    ridge = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS)).fit(X, y)
    rff = make_pipeline(
        StandardScaler(), RFFRidgeCV(alphas=ALPHAS, n_features=200, random_state=0)
    ).fit(X, y)
    assert rmse(rff.predict(Xt), yt) < 1.5 * rmse(ridge.predict(Xt), yt) + 0.02


def test_rff_respects_seed():
    X, y = nonlinear(400, 3)
    a = RFFRidgeCV(alphas=ALPHAS, n_features=100, random_state=5).fit(X, y).predict(X)
    b = RFFRidgeCV(alphas=ALPHAS, n_features=100, random_state=5).fit(X, y).predict(X)
    c = RFFRidgeCV(alphas=ALPHAS, n_features=100, random_state=6).fit(X, y).predict(X)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_rff_caps_features_by_rows():
    X, y = nonlinear(40, 4)
    est = RFFRidgeCV(alphas=ALPHAS, n_features=1000, random_state=0).fit(X, y)
    assert est._d_rff == 20
    assert np.isfinite(est.predict(X)).all()


def test_reg_readout_validation():
    pytest.importorskip("fintfm")
    from fintfm.readout.entry import check_reg_readout

    assert check_reg_readout(None, source="x") == "ridge"
    assert check_reg_readout("rff", source="x") == "rff"
    with pytest.raises(ValueError, match="reg_readout"):
        check_reg_readout("svm", source="reg_readout")
