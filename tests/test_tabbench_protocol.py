"""Tests for the TabBench protocol harness (task 35.4). SMOKE-TEST only: no network, tiny nets."""

import json

import numpy as np
import pandas as pd
import pytest
import torch

from fintfm.experiments import tabbench_protocol as tp
from fintfm.modeling.model import FinancialTFM, ModelConfig


def _tiny() -> FinancialTFM:
    torch.manual_seed(1)
    return FinancialTFM(
        ModelConfig(max_features=8, max_classes=2, d_model=16, n_heads=2, n_layers=1, d_ff=32)
    )


def test_folds_are_five_stratified_twenty_percent_and_deterministic():
    y = np.array([0] * 80 + [1] * 20)
    a, b = tp.split_folds(y), tp.split_folds(y)
    assert len(a) == 5
    for (tr, te), (tr2, te2) in zip(a, b, strict=True):
        np.testing.assert_array_equal(te, te2)
        assert len(te) == 20 and len(tr) == 80
        assert y[te].sum() == 4  # stratified: 20% of the 20 positives
        assert not set(tr) & set(te)


def test_a_one_row_class_stays_in_training():
    y = np.array([0] * 50 + [1] * 49 + [2])
    for tr, te in tp.split_folds(y):
        assert 99 in tr and 99 not in te
        assert len(tr) + len(te) == 100


def test_encoding_parses_dates_blanks_and_categories_without_leaking_test_codes():
    tr = pd.DataFrame(
        {
            "num": [1.0, 2.0, np.nan],
            "when": ["2011-12-31T23:00:00", "2012-01-01T00:00:00", "2012-01-02T00:00:00"],
            "blank": [" ", "3", "4"],
            "cat": ["a", "b", "a"],
        }
    )
    te = tr.iloc[:2].copy()
    te["cat"] = ["a", "zzz"]
    xa, xb, n_cat = tp.encode_features(tr, te)
    assert xa.dtype == np.float32 and xa.shape == (3, 4) and xb.shape == (2, 4)
    assert n_cat == 1
    assert xa[1, 1] - xa[0, 1] == pytest.approx(1 / 24, abs=2e-3)  # days; float32 resolves ~1e-3 d
    assert np.isnan(xa[0, 2]) and xa[1, 2] == 3  # blank string is missing
    assert xb[0, 3] == 0 and np.isnan(xb[1, 3])  # unseen category is missing


def test_eligibility_reasons():
    assert tp.eligibility(3, 5, 8) == "3 classes (binary checkpoint)"
    assert "exceed" in tp.eligibility(2, 9, 8)
    assert tp.eligibility(2, 8, 8) is None


def test_run_is_resumable_sharded_and_counts_coverage(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    tables = {}
    for did in (1, 2, 3):
        X = pd.DataFrame(rng.normal(size=(120, 4)), columns=list("abcd"))
        y = (X["a"] + rng.normal(size=120) > 0).astype(int).to_numpy()
        tables[did] = (X, y, f"toy{did}")
    tables[3] = (tables[3][0], np.arange(120) % 3, "multi")  # three classes: skipped
    calls: list[int] = []

    def fake_load(did: int):
        calls.append(did)
        return tables[did]

    monkeypatch.setattr(tp, "load_openml", fake_load)
    arms = {"checkpoint": _tiny(), "untrained_control": _tiny()}
    rec = tp.run("unused", tmp_path, [1, 2, 3], shard=(0, 2), arms=arms, max_context=60)
    assert sorted(calls) == [1, 3] and rec["pending_ids"] == [2]  # shard 0 of 2
    tp.run("unused", tmp_path, [1, 2, 3], shard=(1, 2), arms=arms, max_context=60)
    calls.clear()
    rec = tp.run("unused", tmp_path, [1, 2, 3], arms=arms, max_context=60)
    assert calls == []  # everything already done
    assert rec["pending_ids"] == []
    for arm in tp.ARMS:
        cov = rec["arms"][arm]["coverage"]
        assert (cov["n_attempted"], cov["n_scored"], cov["n_skipped"]) == (3, 2, 1)
        assert "3 classes" in cov["skipped"]["openml-3"]
        rows = rec["arms"][arm]["per_dataset"]
        assert set(rows) == {"openml-1", "openml-2"}
        assert all(0 <= v["roc_auc"] <= 1 and 0 <= v["accuracy"] <= 1 for v in rows.values())
    assert rec["paired"]["n_datasets"] == 2
    assert rec["not_listed_count"] == tp.SUITE_SIZE - 3
    assert "INTERNAL ONLY" in rec["internal_only"]
    one = json.loads((tmp_path / "datasets" / "openml-1.json").read_text())
    assert len(one["arms"]["checkpoint"]["folds"]) == 5
    assert "mean_predicted" in one["arms"]["checkpoint"]["folds"][0]
    assert "INTERNAL ONLY" in tp.summarise(rec)


def test_a_load_failure_is_a_skip_with_a_reason(tmp_path, monkeypatch):
    def boom(did: int):
        raise OSError("offline")

    monkeypatch.setattr(tp, "load_openml", boom)
    arms = {"checkpoint": _tiny(), "untrained_control": _tiny()}
    rec = tp.run("unused", tmp_path, [7], arms=arms)
    cov = rec["arms"]["checkpoint"]["coverage"]
    assert cov["n_scored"] == 0 and "offline" in cov["skipped"]["openml-7"]
    assert rec["paired"] is None
