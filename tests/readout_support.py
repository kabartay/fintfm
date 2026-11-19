"""Shared fixtures and helpers of the ``test_readout_*`` tests.

``Harness`` is a thin test convenience over :class:`fintfm.readout.LinearReadoutModel`: it takes pandas
frames (categorical columns become label codes, as the TabArena adapter produces them) and, for
binary problems, returns the positive-class column of ``predict_proba`` as AutoGluon does. Everything
else is the library class unchanged.
"""

from __future__ import annotations

import copy
import re

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from fintfm.inference.classifier import FinancialTFMClassifier
from fintfm.modeling.model import FinancialTFM, ModelConfig
from fintfm.readout import LinearReadoutModel, entry_defaults

N_TRAIN = 240
N_TEST = 40
MAX_CONTEXT = 60
PARTITION = {
    "n_bags": 2,
    "max_depth": 1,
    "min_samples_leaf": 0.1,
    "min_leaf_rows": 30,
    "min_class_rows": 2,
}
PROBLEM_TYPES = ("binary", "multiclass", "regression")
_PER_TYPE = re.compile(r"^(checkpoint|partition|probe)_(binary|multiclass|regression)$")


def frame_to_matrix(X: pd.DataFrame) -> tuple[np.ndarray, list[int]]:
    """The float32 matrix and categorical positions the TabArena adapter hands the library.

    Args:
        X: A frame whose categorical columns have ``category`` dtype.

    Returns:
        ``(matrix, categorical_features)``; category codes are ``-1`` for a missing value.
    """
    cols, cats = [], []
    for j, name in enumerate(X.columns):
        if isinstance(X[name].dtype, pd.CategoricalDtype):
            cols.append(X[name].cat.codes.to_numpy(dtype=np.float32))
            cats.append(j)
        else:
            cols.append(X[name].to_numpy(dtype=np.float32, na_value=np.nan))
    return np.column_stack(cols).astype(np.float32), cats


class Harness(LinearReadoutModel):
    """The library estimator with frame in, AutoGluon-style probabilities out; for tests only."""

    def _get_model_params(self) -> dict:
        """A copy of the settings, as the TabArena adapter's ``_get_model_params`` returns them."""
        return copy.deepcopy(self.settings)

    def fit(self, X=None, y=None, categorical_features=None):
        names = None
        if isinstance(X, pd.DataFrame):
            names = list(X.columns)
            X, categorical_features = frame_to_matrix(X)
        return super().fit(X, y, categorical_features or (), feature_names=names)

    def preprocess(self, X: pd.DataFrame, is_train: bool = False) -> np.ndarray:
        """The matrix the network reads for ``X``, through the fitted encoder."""
        return self._encode(frame_to_matrix(X)[0], is_train=is_train)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.problem_type == "regression":  # AutoGluon reads the predictions from predict_proba
            return self.predict(X)
        proba = super().predict_proba(frame_to_matrix(X)[0])
        return (
            proba[:, 1]
            if self.problem_type == "binary" and proba.ndim == 2 and proba.shape[1] > 1
            else proba
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return super().predict(frame_to_matrix(X)[0])


def make(
    problem_type: str, tmp_path=None, checkpoint: str | None = None, **hyperparameters
) -> Harness:
    """An unfitted estimator at the measured entry, with ``hyperparameters`` overriding it.

    Per-problem-type names (``partition_binary``, ``probe_regression``, ``checkpoint_multiclass``) set
    the plain setting when they name ``problem_type`` and are ignored otherwise. With a ``checkpoint``
    path the tiny-network settings of the fast tests (``max_context`` 60) apply.

    Args:
        problem_type: ``binary``, ``multiclass`` or ``regression``.
        tmp_path: Unused; kept so the call reads like the adapter's.
        checkpoint: A local checkpoint path, or ``None`` for the published reference.
        **hyperparameters: Settings overriding the entry.

    Returns:
        An unfitted estimator on CPU with seed 0.
    """
    settings = entry_defaults(problem_type)
    ref = settings.pop("checkpoint")
    if checkpoint is not None:
        ref = checkpoint
        settings.update(
            {
                "max_context": MAX_CONTEXT,
                "feature_transform": "squash",
                "max_features_strategy": "select",
            }
        )
    for key, value in hyperparameters.items():
        match = _PER_TYPE.match(key)
        if match is None:
            settings[key] = value
        elif match.group(2) == problem_type:
            if match.group(1) == "checkpoint":
                ref = value
            else:
                settings[match.group(1)] = value
    return Harness(problem_type, ref, device="cpu", random_state=0, **settings)


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory) -> str:
    """An untrained network small enough to fit and score in a second."""
    torch.manual_seed(0)
    path = tmp_path_factory.mktemp("ckpt") / "tiny.pt"
    cfg = ModelConfig(
        max_features=8,
        max_classes=10,
        d_cell=8,
        d_model=16,
        n_layers=1,
        n_col_layers=1,
        n_heads=2,
        d_ff=32,
    )
    FinancialTFM(cfg).save(str(path))
    return str(path)


def data(
    problem_type: str, seed: int = 0, categorical: bool = False
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """A small table with signal in the first two columns, split into train and test rows."""
    rng = np.random.default_rng(seed)
    n = N_TRAIN + N_TEST
    X = pd.DataFrame(rng.normal(size=(n, 5)), columns=[f"f{i}" for i in range(5)])
    signal = X["f0"] + 0.5 * X["f1"] + rng.normal(scale=0.5, size=n)
    if categorical:
        X["cat"] = pd.Categorical(rng.choice(["a", "b", "c"], size=n))
    if problem_type == "binary":
        y = (signal > 0).astype(int)
    elif problem_type == "multiclass":
        y = pd.Series(np.digitize(signal, [-0.6, 0.6]))
    else:
        y = signal
    y = pd.Series(np.asarray(y))
    return (
        X.iloc[:N_TRAIN].reset_index(drop=True),
        y.iloc[:N_TRAIN].reset_index(drop=True),
        X.iloc[N_TRAIN:],
    )


class Spy:
    """Records every row that a classifier encodes, with the context it encodes it against."""

    def __init__(self, monkeypatch):
        self.calls: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        self.restricted: list[np.ndarray] = []
        original = FinancialTFMClassifier.transform_representation
        restrict = FinancialTFMClassifier.restrict_context
        spy = self

        def transform(clf, X, stage="encode_rows"):
            ctx_values = np.asarray(clf._raw_X, dtype=np.float32)[np.asarray(clf.context_indices_)]
            spy.calls.append(
                (
                    np.asarray(X, dtype=np.float32).copy(),
                    ctx_values,
                    np.asarray(clf.context_indices_),
                )
            )
            return original(clf, X, stage=stage)

        def restrict_context(clf, rows):
            spy.restricted.append(np.asarray(rows))
            return restrict(clf, rows)

        monkeypatch.setattr(FinancialTFMClassifier, "transform_representation", transform)
        monkeypatch.setattr(FinancialTFMClassifier, "restrict_context", restrict_context)

    def rows_seen_in_own_context(self) -> int:
        """Encoded rows that equal a row of the context they were encoded against."""
        n = 0
        for X, ctx_values, _ in self.calls:
            ctx = {row.tobytes() for row in ctx_values}
            n += sum(row.tobytes() in ctx for row in X)
        return n
