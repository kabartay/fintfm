# ruff: noqa: F811
"""The ``partition`` setting wraps the FinTFM classifier in tree-partitioned contexts (fintfm fix 4)."""

from __future__ import annotations

import functools

import numpy as np
import pandas as pd
import pytest
from readout_support import (
    checkpoint,  # noqa: F401
    make,
)

from fintfm.inference.large_n import PartitionedTFMClassifier
from fintfm.readout import LinearReadoutModel
from fintfm.readout.entry import validated_partition


def _partition(model: LinearReadoutModel) -> dict | None:
    """The partition setting the model resolves for its problem type."""
    return model._resolve_entry(model._get_model_params())["partition"]


def test_the_entry_partitions_every_problem_type() -> None:
    """The measured entry carries a bagged partition for each problem type, with the binary row gate."""
    assert _partition(make("binary", partition_bags_max_rows=None)) == {"n_bags": 4}
    assert _partition(make("multiclass")) == {"n_bags": 8}
    assert _partition(make("regression")) == {"n_bags": 4}


def test_a_setting_replaces_the_entry_partition() -> None:
    assert _partition(make("binary", partition={"max_depth": 3}, partition_scale_with_n=False)) == {
        "max_depth": 3
    }
    assert _partition(make("binary", partition=None)) is None
    assert _partition(make("binary", partition={}, partition_scale_with_n=False)) == {}


@pytest.mark.parametrize(
    ("value", "match"),
    [
        ({"max_detph": 2}, "max_detph"),
        ({"random_state": 3}, "random_state"),
        ({"estimator": None}, "estimator"),
        ([1, 2], "JSON object"),
    ],
)
def test_bad_values_raise(value, match: str) -> None:
    """A misspelled or reserved key raises rather than running the default under the arm's name."""
    with pytest.raises(ValueError, match=match):
        validated_partition(value, source="partition")
    with pytest.raises(ValueError, match=match):
        _partition(make("binary", partition=value))


def _fit(
    checkpoint: str, monkeypatch: pytest.MonkeyPatch, **settings
) -> tuple[LinearReadoutModel, list[dict]]:
    """Fit on a toy binary table, returning the model and the arguments of every PartitionedTFMClassifier built."""
    calls: list[dict] = []
    original = PartitionedTFMClassifier.__init__

    @functools.wraps(original)
    def spy(self, *args, **kwargs):
        calls.append(kwargs)
        original(self, *args, **kwargs)

    monkeypatch.setattr(PartitionedTFMClassifier, "__init__", spy)
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(200, 4)), columns=list("abcd"))
    y = pd.Series((X["a"] + 0.3 * rng.normal(size=200) > 0).astype(int))
    settings = {
        "partition": None,
        "probe_crossfit_folds": 0,
        "max_context": 50,
        "n_ensemble": 1,
        **settings,
    }
    model = make("binary", None, checkpoint, **settings)
    model.fit(X=X, y=y)
    return model, calls


def test_partition_setting_reaches_the_partitioned_classifier(
    checkpoint: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    model, calls = _fit(checkpoint, monkeypatch, partition={"max_depth": 1, "min_leaf_rows": 30})
    assert len(calls) == 1
    assert calls[0]["max_depth"] == 1
    assert calls[0]["min_leaf_rows"] == 30
    assert calls[0]["random_state"] == 0
    assert isinstance(model.model, PartitionedTFMClassifier)


def test_no_partition_builds_no_partitioned_classifier(
    checkpoint: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    model, calls = _fit(checkpoint, monkeypatch)
    assert calls == []
    assert not isinstance(model.model, PartitionedTFMClassifier)


class _Net:
    """Stands in for the network: records where it was moved."""

    def __init__(self) -> None:
        self.moved_to: list[str] = []

    def to(self, device: str) -> None:
        self.moved_to.append(device)


class _Clf:
    def __init__(self, net: _Net) -> None:
        self.model = net
        self.device = "mps"


class _Partitioned:
    def __init__(self, net: _Net) -> None:
        self.estimator = _Clf(net)
        self.global_estimator_ = _Clf(net)
        self.leaf_estimators_ = [{1: _Clf(net), 2: None, 3: _Clf(net)}]


def test_to_reaches_every_inner_classifier() -> None:
    """Every leaf, the fallback and the template follow a device change; a fallback leaf is skipped."""
    model = LinearReadoutModel("binary", "/unused.pt")
    net = _Net()
    model.model = _Partitioned(net)
    model.to("cpu")
    inner = model.inner_classifiers()
    assert len(inner) == 4
    assert all(clf.device == "cpu" for clf in inner)
    assert model.device == "cpu"
    assert np.unique(net.moved_to).tolist() == ["cpu"]
