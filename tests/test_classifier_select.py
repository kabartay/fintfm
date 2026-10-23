"""``max_features_strategy="select"``: supervised column selection for wide tables (DR8.3).

A table wider than the checkpoint is cut, at ``fit`` and on training rows only, to the
columns :class:`fintfm.inference.preprocess.ColumnSelector` ranks highest, and the same
indices are applied at predict time. These tests pin the contract: the selection never sees
a query row, is reproducible at a seed, changes nothing on a table within the cap, works
for binary and multiclass labels, and on a wide task whose signal sits in a few columns it
keeps them and beats the round-robin ``"subsample"`` strategy.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from sklearn.metrics import roc_auc_score

from fintfm.inference.classifier import FinancialTFMClassifier
from fintfm.inference.preprocess import ColumnSelector
from fintfm.modeling.model import FinancialTFM, ModelConfig

#: Width the test checkpoints accept; every "wide" table below is wider.
CAP = 8


def _capped_model(max_classes: int = 3) -> FinancialTFM:
    """An untrained network accepting at most :data:`CAP` features.

    Args:
        max_classes: Classes its head represents.

    Returns:
        The network, initialised at a fixed seed.
    """
    torch.manual_seed(0)
    return FinancialTFM(
        ModelConfig(max_features=CAP, max_classes=max_classes, d_model=16, d_cell=8,
                    n_heads=2, n_layers=1, n_col_layers=1, d_ff=32)
    )


def _signal_task(
    n: int = 600, n_feat: int = 600, n_classes: int = 2, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A wide task whose label depends on five columns only, the rest pure noise.

    Args:
        n: Rows, split evenly into training and query halves.
        n_feat: Columns.
        n_classes: Classes, cut from a linear score of the five signal columns at its
            quantiles, so the classes are balanced.
        seed: Seed for the data and the signal columns' positions.

    Returns:
        ``(X_train, y_train, X_query, y_query, signal_columns)``.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_feat)).astype(np.float32)
    signal = np.sort(rng.choice(n_feat, size=5, replace=False))
    score = X[:, signal] @ rng.choice([-1.0, 1.0], size=5)
    cuts = np.quantile(score, np.linspace(0, 1, n_classes + 1)[1:-1])
    y = np.searchsorted(cuts, score).astype(np.int64)
    k = n // 2
    return X[:k], y[:k], X[k:], y[k:], signal


def _centroid_logits(
    self: FinancialTFM, Xc: np.ndarray, yc: np.ndarray, Xq: np.ndarray, **kwargs: object
) -> torch.Tensor:
    """A stand-in for a trained checkpoint: nearest-centroid logits from the context.

    The untrained test network has no in-context skill to compare strategies with, so the
    comparison test replaces :meth:`FinancialTFM.predict_context` with this, which does
    read its context: each query's logit for class ``c`` is minus its mean squared distance
    to the context rows of class ``c``, over whichever columns the strategy handed it.

    Args:
        self: The network, unused.
        Xc: ``(m, f)`` conditioned context rows.
        yc: ``(m,)`` their label codes.
        Xq: ``(q, f)`` conditioned query rows.
        **kwargs: ``n_classes`` (read) and the real method's other keywords (ignored).

    Returns:
        ``(q, n_classes)`` float32 logits.
    """
    n_classes = int(kwargs.get("n_classes") or int(yc.max()) + 1)
    Xc, Xq = np.nan_to_num(Xc), np.nan_to_num(Xq)
    cents = np.stack([Xc[yc == c].mean(axis=0) for c in range(n_classes)])
    d = ((Xq[:, None, :] - cents[None]) ** 2).mean(axis=2)
    return torch.from_numpy(-4.0 * d).float()


def test_select_keeps_the_signal_columns_and_beats_subsample(monkeypatch):
    """Five signal columns in 600 through an 8-feature model. ``select`` must keep all five,
    and score clearly above ``subsample``, whose 75 members each see 8 columns and so mostly
    vote on noise."""
    monkeypatch.setattr(FinancialTFM, "predict_context", _centroid_logits)
    Xtr, ytr, Xq, yq, signal = _signal_task()
    auc = {}
    for strategy in ("select", "subsample"):
        clf = FinancialTFMClassifier(
            _capped_model(2), max_context=300, max_features_strategy=strategy
        ).fit(Xtr, ytr)
        auc[strategy] = roc_auc_score(yq, clf.predict_proba(Xq)[:, 1])
        if strategy == "select":
            assert clf.selected_columns_.size == CAP
            assert set(signal) <= set(clf.selected_columns_.tolist())
            assert clf.n_members_ == 1
    assert auc["select"] > 0.9, auc
    assert auc["select"] - auc["subsample"] > 0.15, auc


@pytest.mark.parametrize("n_classes", [2, 3])
@pytest.mark.parametrize("method", ["extra_trees", "mutual_info"])
def test_select_works_for_binary_and_multiclass(n_classes, method):
    Xtr, ytr, Xq, _, signal = _signal_task(n=400, n_feat=60, n_classes=n_classes)
    clf = FinancialTFMClassifier(
        _capped_model(3), max_context=200, max_features_strategy="select",
        select_method=method,
    ).fit(Xtr, ytr)
    assert clf.n_features_in_ == 60
    assert clf.selected_columns_.size == CAP
    assert np.all(np.diff(clf.selected_columns_) > 0)  # sorted, original order kept
    assert np.isin(signal, clf.selected_columns_).sum() >= 4
    p = clf.predict_proba(Xq)
    assert p.shape == (len(Xq), n_classes) and np.isfinite(p).all()
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-5)
    assert set(np.unique(clf.predict(Xq))) <= set(range(n_classes))


def test_select_is_fitted_on_training_rows_only():
    """Different query rows never move the selection, and the query is scored on exactly the
    training-fitted columns: the same as a plain classifier fitted on those columns."""
    Xtr, ytr, Xq, _, _ = _signal_task(n=400, n_feat=60)
    model = _capped_model(2)
    clf = FinancialTFMClassifier(model, max_context=100, max_features_strategy="select")
    clf.fit(Xtr, ytr)
    cols = clf.selected_columns_.copy()
    np.testing.assert_array_equal(cols, ColumnSelector(CAP).fit(Xtr, ytr).columns_)
    rng = np.random.default_rng(7)
    other = (rng.normal(size=Xq.shape) * 50).astype(np.float32)
    p_a = clf.predict_proba(Xq)
    clf.predict_proba(other)
    p_b = clf.predict_proba(np.vstack([Xq[:10], other[:30]]))
    np.testing.assert_array_equal(clf.selected_columns_, cols)
    np.testing.assert_allclose(p_b[:10], p_a[:10], atol=1e-6)  # float noise of batch size
    plain = FinancialTFMClassifier(model, max_context=100).fit(Xtr[:, cols], ytr)
    np.testing.assert_array_equal(p_a, plain.predict_proba(Xq[:, cols]))
    with pytest.raises(ValueError, match="feature width"):
        clf.predict_proba(Xq[:, cols])


def test_select_is_deterministic_under_random_state():
    Xtr, ytr, Xq, _, _ = _signal_task(n=400, n_feat=60, seed=3)
    model = _capped_model(2)

    def run(seed: int) -> tuple[np.ndarray, np.ndarray]:
        """Columns and predictions of a ``select`` fit at ``seed``."""
        clf = FinancialTFMClassifier(
            model, max_context=100, max_features_strategy="select", random_state=seed,
            n_ensemble=2, select_subsample=150,
        ).fit(Xtr, ytr)
        return clf.selected_columns_, clf.predict_proba(Xq)

    cols_a, p_a = run(5)
    cols_b, p_b = run(5)
    np.testing.assert_array_equal(cols_a, cols_b)
    assert p_a.tobytes() == p_b.tobytes()


@pytest.mark.parametrize("n_ensemble", [1, 3])
@pytest.mark.parametrize("width", [CAP - 3, CAP])
def test_select_on_a_table_within_the_cap_is_the_default_bit_for_bit(n_ensemble, width):
    Xtr, ytr, Xq, _, _ = _signal_task(n=300, n_feat=width)
    model = _capped_model(2)
    default = FinancialTFMClassifier(model, max_context=100, n_ensemble=n_ensemble)
    select = FinancialTFMClassifier(
        model, max_context=100, n_ensemble=n_ensemble, max_features_strategy="select"
    )
    p_default = default.fit(Xtr, ytr).predict_proba(Xq)
    p_select = select.fit(Xtr, ytr).predict_proba(Xq)
    assert select.selected_columns_ is None
    assert p_select.tobytes() == p_default.tobytes()


def test_select_combines_with_an_ensemble_and_the_whole_row_methods():
    """The narrowed table is an ordinary one: members share the selected columns, and the
    methods ``subsample`` refuses (representation) work on it."""
    Xtr, ytr, Xq, _, _ = _signal_task(n=300, n_feat=40)
    clf = FinancialTFMClassifier(
        _capped_model(2), max_context=100, max_features_strategy="select", n_ensemble=3
    ).fit(Xtr, ytr)
    assert clf.n_members_ == 3
    cols = clf.selected_columns_
    by_hand = [
        FinancialTFMClassifier(clf.model_, max_context=100, random_state=i)
        .fit(Xtr[:, cols], ytr).predict_proba(Xq[:, cols])
        for i in range(3)
    ]
    np.testing.assert_array_equal(clf.predict_proba(Xq), np.mean(by_hand, axis=0))
    assert clf.transform_representation(Xq).shape == (len(Xq), 16)


def test_select_drops_constant_columns_before_ranking():
    """Twenty columns but only six vary: those six are kept, and nothing is ranked."""
    rng = np.random.default_rng(0)
    X = np.zeros((200, 20), dtype=np.float32)
    X[:, 5] = np.nan
    X[:, 9] = np.inf
    varying = np.array([1, 4, 8, 11, 15, 19])
    X[:, varying] = rng.normal(size=(200, varying.size))
    y = (X[:, 4] > 0).astype(np.int64)
    sel = ColumnSelector(CAP).fit(X, y)
    np.testing.assert_array_equal(sel.columns_, varying)
    X[:, 2] = rng.normal(size=200)  # a seventh and eighth varying column, then a ninth
    X[:, 3] = rng.normal(size=200)
    X[:, 6] = rng.normal(size=200)
    sel = ColumnSelector(CAP).fit(X, y)
    assert sel.columns_.size == CAP and 4 in sel.columns_
    assert not np.isin([0, 5, 7, 9], sel.columns_).any()
    clf = FinancialTFMClassifier(_capped_model(2), max_context=100,
                                 max_features_strategy="select")
    with pytest.raises(ValueError, match="every column constant"):
        clf.fit(np.ones((50, 20), dtype=np.float32), y[:50])


def test_select_refuses_unknown_settings():
    Xtr, ytr, _, _, _ = _signal_task(n=100, n_feat=20)
    with pytest.raises(ValueError, match="select_method"):
        FinancialTFMClassifier(
            _capped_model(2), max_features_strategy="select", select_method="lightgbm"
        ).fit(Xtr, ytr)
    with pytest.raises(ValueError, match="select_subsample"):
        FinancialTFMClassifier(
            _capped_model(2), max_features_strategy="select", select_subsample=0
        ).fit(Xtr, ytr)
