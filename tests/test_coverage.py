"""Openspec 36.4: every multi-task harness reports attempted = scored + skipped, with reasons.

Synthetic inputs only. Each harness that scores several tasks, datasets or splits is driven
with at least one unit that cannot be scored (a degenerate single-class split, an unloadable
dataset, a checkpoint narrower than the data) and must count it as skipped with a reason.
"""

import dataclasses
import json

import numpy as np
import pandas as pd
import pytest

from fintfm.evaluation import bench
from fintfm.evaluation import credit_suite as cs
from fintfm.evaluation import proxy_suite as ps
from fintfm.evaluation.coverage import Coverage
from fintfm.experiments import openml_breadth, prior_score
from fintfm.prior.base import Task


def _balanced(rec: dict) -> None:
    assert rec["n_attempted"] == rec["n_scored"] + rec["n_skipped"]
    assert len(rec["skipped"]) == rec["n_skipped"]
    assert all(reason for reason in rec["skipped"].values())


def test_an_unresolved_unit_is_an_error_not_a_quiet_drop():
    cov = Coverage()
    cov.attempt("a")
    cov.attempt("b")
    cov.scored("a")
    with pytest.raises(ValueError, match="unresolved"):
        cov.to_dict()
    cov.skip("b", "one class")
    rec = cov.to_dict()
    _balanced(rec)
    assert rec["coverage"] == 0.5
    assert "1/2 tasks scored, 1 skipped" in cov.summary()


def test_a_skip_needs_a_reason_and_a_prior_attempt_and_happens_once():
    cov = Coverage()
    with pytest.raises(ValueError, match="before it was attempted"):
        cov.scored("x")
    cov.attempt("x")
    with pytest.raises(ValueError, match="without a reason"):
        cov.skip("x", " ")
    cov.scored("x")
    with pytest.raises(ValueError, match="twice"):
        cov.skip("x", "late")
    with pytest.raises(ValueError, match="attempted twice"):
        cov.attempt("x")
    assert Coverage().to_dict()["coverage"] is None


def test_bench_synthetic_counts_a_single_class_split_as_skipped(monkeypatch, capsys):
    class _Model:
        cfg = type("C", (), {"max_features": 8, "max_classes": 2})()

    monkeypatch.setattr(bench.FinancialTFM, "load", staticmethod(lambda path: _Model()))
    calls = iter(
        [
            {"fintfm": 0.9, "logreg": 0.8},
            {"fintfm": float("nan"), "logreg": float("nan")},  # single-class test split
            {"fintfm": 0.7, "logreg": 0.8},
        ]
    )
    monkeypatch.setattr(bench, "run_one", lambda *a, **k: next(calls))
    rec = bench.run_synthetic("unused.pt", n_tasks=3, n_rows=60)
    _balanced(rec)
    assert (rec["n_attempted"], rec["n_scored"], rec["n_skipped"]) == (3, 2, 1)
    assert "single class" in rec["skipped"]["synthetic-1"]
    assert rec["mean_auc"]["fintfm"] == pytest.approx(0.8)
    assert "coverage: 2/3 synthetic tasks scored, 1 skipped" in capsys.readouterr().out
    json.dumps(rec)


def _sampler(rng: np.random.Generator, n_rows: int) -> Task:
    X = rng.normal(size=(n_rows, 4)).astype(np.float32)
    if rng.random() < 0.5:
        y = np.zeros(n_rows, dtype=np.int64)  # degenerate: one class throughout
    else:
        y = (X[:, 0] + 0.3 * rng.normal(size=n_rows) > 0).astype(np.int64)
    return Task(X=X, y=y, n_classes=2, is_categorical=np.zeros(4, dtype=bool), source="t")


def test_prior_score_counts_degenerate_tasks_as_skipped_with_a_reason():
    rec = prior_score.score_prior(_sampler, n_tasks=8, n_rows=200)
    _balanced(rec)
    assert rec["n_attempted"] == 8
    assert 0 < rec["n_skipped"] < 8, "the sampler must give both kinds of task"
    assert set(rec["skipped"].values()) == {"one class on a side of the split"}
    assert prior_score.summarise({"t": rec}).count("tasks scored") == 1


def test_prior_score_with_every_task_degenerate_still_reports_its_counts():
    def always_one_class(rng, n_rows):
        return Task(
            X=rng.normal(size=(n_rows, 3)).astype(np.float32),
            y=np.zeros(n_rows, dtype=np.int64),
            n_classes=2,
            is_categorical=np.zeros(3, dtype=bool),
            source="t",
        )

    rec = prior_score.score_prior(always_one_class, n_tasks=3, n_rows=100)
    _balanced(rec)
    assert (rec["n_attempted"], rec["n_scored"], rec["n_skipped"]) == (3, 0, 3)
    assert np.isnan(rec["performance"])


def test_openml_breadth_counts_an_unloadable_dataset_as_skipped(monkeypatch, capsys):
    def load(name, version, max_rows):
        if name == "gone":
            return None
        rng = np.random.default_rng(0)
        return rng.normal(size=(50, 3)), (rng.random(50) < 0.3).astype(np.int64)

    monkeypatch.setattr(openml_breadth, "load_task", load)
    monkeypatch.setattr(
        openml_breadth,
        "score_task",
        lambda X, y, model_path, seed=0: {"fintfm": 0.5, "logreg": 0.4},
    )
    cov = Coverage()
    results = openml_breadth.run("unused.pt", suite=[("a", 1), ("gone", 1), ("b", 1)], coverage=cov)
    rec = cov.to_dict()
    _balanced(rec)
    assert len(results) == rec["n_scored"] == 2
    assert (rec["n_attempted"], rec["n_skipped"]) == (3, 1)
    assert "gone" in rec["skipped"]
    assert "coverage: 2/3 datasets scored" in capsys.readouterr().out


def test_credit_suite_rows_coverage_matches_the_rows():
    rows = [
        {"dataset": "d", "split": 0, "model": "fintfm", "skipped": ""},
        {"dataset": "d", "split": 0, "model": "logreg", "skipped": ""},
        {
            "dataset": "d",
            "split": 1,
            "model": "fintfm",
            "skipped": "checkpoint takes 4 features, data has 5",
        },
    ]
    rec = cs.rows_coverage(rows).to_dict()
    _balanced(rec)
    assert (rec["n_attempted"], rec["n_scored"], rec["n_skipped"]) == (3, 2, 1)
    assert "takes 4 features" in rec["skipped"]["d/split1/fintfm"]


def test_credit_suite_run_writes_a_coverage_sidecar(tmp_path, monkeypatch):
    import torch

    from fintfm.modeling.model import FinancialTFM, ModelConfig

    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 5))
    y = (rng.random(600) < 1 / (1 + np.exp(-(2 * X[:, 0] - 1.5)))).astype(np.int64)
    panel = cs.CreditPanel(name="undated", X=X, feature_names=tuple("abcde"), y=y)
    admitted = dataclasses.replace(cs.DATASETS["ulb_fraud"].licence, include=True)
    monkeypatch.setattr(
        cs,
        "DATASETS",
        {
            "undated": cs.SuiteDataset(
                "undated", "https://example.invalid", lambda root: panel, admitted
            )
        },
    )
    torch.manual_seed(0)
    ckpt = tmp_path / "m.pt"
    FinancialTFM(
        ModelConfig(
            max_features=4,
            max_classes=2,
            d_cell=8,
            d_model=16,
            n_layers=1,
            n_col_layers=1,
            n_heads=2,
        )
    ).save(str(ckpt))
    cs.run(tmp_path / "r.csv", str(ckpt), datasets=("undated",), root=tmp_path, with_lightgbm=False)
    rec = json.loads((tmp_path / "r.coverage.json").read_text())
    _balanced(rec)
    # 3 splits x (fintfm, logistic_regression); fintfm is wider-than-checkpoint, so skipped
    assert (rec["n_attempted"], rec["n_scored"], rec["n_skipped"]) == (6, 3, 3)


def test_proxy_suite_frame_coverage_counts_a_single_class_split_as_skipped():
    frame = pd.DataFrame(
        {
            "task_id": [1, 1, 1],
            "split": [0, 1, 1],
            "method": ["lr", "lr", "fintfm"],
            "status": ["ok", "skipped: one class in the test split", "skipped: too wide"],
        }
    )
    rec = ps.frame_coverage(frame).to_dict()
    _balanced(rec)
    assert (rec["n_attempted"], rec["n_scored"], rec["n_skipped"]) == (3, 1, 2)
    assert rec["skipped"]["1/split1/lr"] == "skipped: one class in the test split"
