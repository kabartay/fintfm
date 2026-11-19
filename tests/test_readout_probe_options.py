"""The ``probe`` setting configures the linear probe's readout.

The options exist to improve FinTFM's multiclass and regression entries on TabArena-Lite
(fintfm DR7.6, 2026-11-11) without changing the binary entry. These tests stand in for the network
with a fixed representation.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

pytest.importorskip("fintfm")

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.pipeline import Pipeline

from fintfm.readout import LinearReadoutModel, entry_defaults


class _Network:
    """A stand-in network whose representation is two fixed columns of the row."""

    def transform_representation(self, X: np.ndarray, stage: str) -> np.ndarray:
        """Return a two-column representation of ``X``.

        Args:
            X: Query rows.
            stage: Ignored; accepted for signature compatibility.

        Returns:
            ``(n, 2)``: the first feature and a constant.
        """
        return np.column_stack([X[:, 0], np.ones(len(X))])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return a uniform three-class distribution, the head's stand-in output.

        Args:
            X: Query rows.

        Returns:
            ``(n, 3)`` probabilities.
        """
        return np.full((len(X), 3), 1.0 / 3.0)


def _fitted(
    monkeypatch: pytest.MonkeyPatch, probe: dict | None, *, splines: bool = False
) -> LinearReadoutModel:
    """A readout fitted on three features with the network fit stubbed.

    Cross-fitting is off (``probe_crossfit_folds=0``): these tests cover the readout options on a
    fixed representation, and the stand-in network has no context to restrict. The stubbed network fit
    resolves the entry settings as the real one does. Spline features are off unless ``splines`` is set,
    so the ``raw`` block's width is the feature count; the multiclass entry turns them on (fintfm
    FINDINGS §293). ``probe`` of ``None`` keeps the multiclass entry's own readout.
    """
    settings = entry_defaults("multiclass")
    settings.pop("checkpoint")
    settings.update({"probe_crossfit_folds": 0, "partition": None})
    if not splines:
        settings["probe_spline"] = None
    if probe is not None:
        settings["probe"] = probe
    model = LinearReadoutModel("multiclass", "/unused.pt", **settings)

    def stub_fit_network(self, X, y) -> None:
        self._resolve_entry(copy.deepcopy(self.settings))
        self.model = _Network()

    monkeypatch.setattr(LinearReadoutModel, "_fit_network", stub_fit_network)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 3)).astype(np.float32)
    X[::7, 2] = np.nan
    y = np.digitize(X[:, 1], [-0.5, 0.5])
    return model.fit(X, y)


def test_empty_object_is_the_plain_logistic_regression(monkeypatch: pytest.MonkeyPatch) -> None:
    """``probe={}`` is a bare ``C=1`` logistic regression on the representation."""
    model = _fitted(monkeypatch, {})
    assert type(model._linear_probe) is LogisticRegression
    assert model._linear_probe.n_features_in_ == 2


def test_unset_takes_the_entry_default_for_the_problem_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without a ``probe`` the multiclass entry's own readout applies (scale, cv and raw).

    With splines off the probe sees the two representation columns and the three raw features;
    with the entry's splines on (at most 10,000 rows) the raw block is expanded, so it is wider.
    """
    model = _fitted(monkeypatch, None)
    assert model._probe_opts == {"scale": True, "cv": True, "raw": True}
    assert isinstance(model._linear_probe, Pipeline)
    assert model._linear_probe.n_features_in_ == 5
    splined = _fitted(monkeypatch, None, splines=True)
    assert splined._raw_spline is not None
    assert splined._linear_probe.n_features_in_ > 5


def test_raw_appends_the_features_and_scale_wraps_a_scaler(monkeypatch: pytest.MonkeyPatch) -> None:
    """``raw`` widens the probe's input by the three features, missing values included."""
    model = _fitted(monkeypatch, {"scale": True, "cv": True, "raw": True})
    assert isinstance(model._linear_probe, Pipeline)
    assert model._linear_probe.n_features_in_ == 5
    X = np.random.default_rng(1).normal(size=(10, 3)).astype(np.float32)
    X[0, 2] = np.nan
    proba = model._linear_probe.predict_proba(model._probe_input(model.model, X))
    assert np.isfinite(proba).all()


def test_head_appends_the_head_log_probabilities(monkeypatch: pytest.MonkeyPatch) -> None:
    """``head`` widens the probe's input by one column per class."""
    model = _fitted(monkeypatch, {"scale": True, "head": True})
    assert model._linear_probe.n_features_in_ == 5


def test_unknown_key_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A misspelled option fails the fit rather than running the default readout."""
    with pytest.raises(ValueError, match="unknown key"):
        _fitted(monkeypatch, {"sclae": True})


def _probe_for(y: np.ndarray, opts: dict):
    """The unfitted readout the model builds for labels ``y`` under probe options ``opts``."""
    model = LinearReadoutModel("multiclass", "/unused.pt")
    model._probe_opts = opts
    return model._new_classifier_probe(y)


def test_cv_keeps_three_folds_when_every_class_has_three_rows() -> None:
    """On data with at least three rows per class the cross-validated readout is as configured."""
    probe = _probe_for(np.array([0, 0, 0, 1, 1, 1, 1]), {"cv": True})
    assert isinstance(probe, LogisticRegressionCV)
    assert probe.cv == 3


def test_cv_folds_shrink_to_the_rarest_class() -> None:
    """A class with two rows leaves room for two stratified folds, not three."""
    probe = _probe_for(np.array([0, 0, 0, 0, 1, 1]), {"cv": True})
    assert isinstance(probe, LogisticRegressionCV)
    assert probe.cv == 2


def test_cv_falls_back_to_the_fixed_readout_with_a_single_row_class() -> None:
    """With a one-row class there is nothing to select ``C`` on, so the readout is plain and fits."""
    y = np.array([0, 0, 0, 1])
    probe = _probe_for(y, {"cv": True, "scale": True})
    assert isinstance(probe, Pipeline)
    assert type(probe[-1]) is LogisticRegression
    probe.fit(np.arange(8, dtype=float).reshape(4, 2), y)
    assert probe.predict_proba(np.zeros((1, 2))).shape == (1, 2)
