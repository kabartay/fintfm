"""``fintfm-bench --credit`` reports the small-n probe by default (openspec 13.6).

No real data or checkpoint: panels are synthetic, the checkpoint loader and the in-context
classifier are stubs, so the test only checks the wiring.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from fintfm.evaluation import bench
from fintfm.evaluation.datasets import CreditDataset, SurvivalDataset
from fintfm.experiments import prior_ablation


class _StubTFM:
    """Stand-in for ``FinancialTFMClassifier`` backed by logistic regression."""

    def __init__(self, model_path: str, **kwargs: object) -> None:
        """Ignore the checkpoint path and options."""
        self.lr = LogisticRegression(max_iter=200)

    def fit(self, X: np.ndarray, y: np.ndarray) -> _StubTFM:
        """Fit on NaN-free features and return self."""
        self.lr.fit(np.nan_to_num(X), y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities."""
        return self.lr.predict_proba(np.nan_to_num(X))


def _panel(horizon: int, n: int = 1500, d: int = 6) -> CreditDataset:
    """Build a small synthetic panel with a roughly 10% default rate."""
    rng = np.random.default_rng(horizon)
    X = rng.normal(size=(n, d)).astype(np.float32)
    y = (X[:, 0] + rng.normal(size=n) > 1.3).astype(int)
    return CreditDataset(X, y, f"synthetic-h{horizon}", horizon, "test", "synthetic")


@pytest.fixture
def stubbed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace data, checkpoint and per-panel benchmarking with cheap stubs."""
    cfg = SimpleNamespace(cfg=SimpleNamespace(max_features=64))
    monkeypatch.setattr(bench.FinancialTFM, "load", lambda p: cfg)
    monkeypatch.setattr(bench, "load_polish_bankruptcy", _panel)
    monkeypatch.setattr(bench, "load_taiwan_bankruptcy", lambda: _panel(9))
    monkeypatch.setattr(bench, "run_one", lambda *a, **k: {})
    monkeypatch.setattr(bench, "_compare_context_strategies", lambda *a, **k: None)
    monkeypatch.setattr(prior_ablation, "FinancialTFMClassifier", _StubTFM)


def test_credit_output_includes_the_sample_efficiency_probe(stubbed, capsys):
    out = bench.run_credit("ckpt.pt", probe_sizes=(100, 250))
    probe = out["sample_efficiency"]
    assert set(probe["sizes"]) == {"100", "250"}
    assert {"crossover_size", "leads_at", "test_rows"} <= set(probe)
    assert "fintfm" in probe["sizes"]["100"] and "gboost" in probe["sizes"]["100"]
    assert "sample-efficiency probe" in capsys.readouterr().out


def test_probe_can_be_skipped(stubbed, capsys):
    out = bench.run_credit("ckpt.pt", sample_efficiency=False)
    assert out == {}
    assert "sample-efficiency" not in capsys.readouterr().out


def test_cli_flag_skips_probe(monkeypatch):
    seen: dict = {}
    monkeypatch.setattr(bench, "run_credit", lambda m, **k: seen.update(k))
    monkeypatch.setattr("sys.argv", ["fintfm-bench", "--credit"])
    bench.main()
    assert seen == {"sample_efficiency": True}
    monkeypatch.setattr("sys.argv", ["fintfm-bench", "--credit", "--no-sample-efficiency"])
    bench.main()
    assert seen == {"sample_efficiency": False}


def _dated_panel(n: int = 400) -> SurvivalDataset:
    """Build a small synthetic dated panel with one company per row."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, 4)).astype(np.float32)
    year = np.where(np.arange(n) < n // 2, 2014, 2018)
    return SurvivalDataset(
        X,
        (X[:, 0] > 1).astype(int),
        np.full(n, -1),
        np.ones(n, dtype=int),
        1,
        year,
        "synthetic-dated",
        "test",
        "synthetic",
        groups=np.arange(n),
    )


def test_credit_output_says_time_split_is_not_available(stubbed, capsys):
    out = bench.run_credit("ckpt.pt", sample_efficiency=False)
    text = capsys.readouterr().out
    assert out == {}
    assert text.count("time-based split: NOT AVAILABLE") == 4
    assert "TIME-BASED SPLIT: NOT RUN" in text


def test_credit_prints_random_and_time_side_by_side_with_a_dated_panel(
    stubbed, monkeypatch, capsys
):
    fake = {"random": {"m": 0.9}, "time": {"m": 0.8}}
    monkeypatch.setattr(bench, "run_time_vs_random", lambda ds, path: fake)
    out = bench.run_credit("ckpt.pt", sample_efficiency=False, time_panel=_dated_panel())
    assert out == {"time_vs_random": fake}
    assert "TIME-BASED SPLIT: NOT RUN" not in capsys.readouterr().out


def test_time_vs_random_prints_both_columns(stubbed, monkeypatch, capsys):
    def fake_run_one(name, X, y, path, split=None):
        return {"m": 0.9 if "random" in name else 0.8}

    monkeypatch.setattr(bench, "run_one", fake_run_one)
    out = bench.run_time_vs_random(_dated_panel(), "ckpt.pt", train_until=2014, test_from=2018)
    text = capsys.readouterr().out
    assert out == {"random": {"m": 0.9}, "time": {"m": 0.8}}
    assert "random AUC" in text and "time AUC" in text


def test_time_split_refuses_a_panel_without_periods(stubbed):
    ds = _dated_panel()
    undated = SurvivalDataset(
        ds.X, ds.y, ds.period, ds.n_observed, 1, None, "undated", "test", "synthetic"
    )
    assert not undated.has_period_labels
    with pytest.raises(ValueError, match="no period"):
        bench.run_time_vs_random(undated, "ckpt.pt")
    assert "NOT AVAILABLE" in bench.time_split_note(undated)


def test_cli_credit_with_time_split_loads_dated_panel(monkeypatch):
    seen: dict = {}
    panel = _dated_panel()
    monkeypatch.setattr(bench, "load_v4finbench", lambda root, max_rows=None: panel)
    monkeypatch.setattr(bench, "run_credit", lambda m, **k: seen.update(k))
    monkeypatch.setattr("sys.argv", ["fintfm-bench", "--credit", "--time-split"])
    bench.main()
    assert seen["time_panel"] is panel


class _PathStub(_StubTFM):
    """Checkpoint ``b.pt`` sees only the noise column, ``a.pt`` sees every feature."""

    def __init__(self, model_path: str, **kwargs: object) -> None:
        """Remember which checkpoint this stands for."""
        super().__init__(model_path)
        self.use_all = model_path == "a.pt"

    def _cols(self, X: np.ndarray) -> np.ndarray:
        """Features this checkpoint may use."""
        return X if self.use_all else X[:, -1:]

    def fit(self, X: np.ndarray, y: np.ndarray) -> _PathStub:
        """Fit on this checkpoint's columns."""
        self.lr.fit(np.nan_to_num(self._cols(X)), y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Class probabilities from this checkpoint's columns."""
        return self.lr.predict_proba(np.nan_to_num(self._cols(X)))


@pytest.fixture
def compare_stubbed(monkeypatch: pytest.MonkeyPatch) -> list[CreditDataset]:
    """Stub the checkpoint loader and classifier; return three synthetic panels."""
    cfg = SimpleNamespace(cfg=SimpleNamespace(max_features=64))
    monkeypatch.setattr(bench.FinancialTFM, "load", lambda p: cfg)
    monkeypatch.setattr(bench, "FinancialTFMClassifier", _PathStub)
    return [_panel(1), _panel(3), _panel(5)]


def test_assign_folds_is_stratified_and_disjoint():
    y = np.r_[np.ones(25, dtype=int), np.zeros(475, dtype=int)]
    fold = bench.assign_folds(y, 5, seed=1)
    assert fold.shape == y.shape and set(fold) == set(range(5))
    for f in range(5):  # each row in exactly one fold, classes spread evenly
        assert (y[fold == f] == 1).sum() == 5 and (fold == f).sum() == 100
    with pytest.raises(ValueError):
        bench.assign_folds(y, 30)
    with pytest.raises(ValueError):
        bench.assign_folds(y, 1)


def test_compare_reports_per_fold_ap_first_and_coverage(compare_stubbed, capsys):
    rec = bench.run_credit_compare("a.pt", "b.pt", folds=5, n_boot=200, panels=compare_stubbed)
    assert rec["coverage"]["n_scored"] == 3 and rec["coverage"]["n_skipped"] == 0
    p = rec["panels"]["synthetic-h1"]
    assert len(p["per_fold"]) == 5
    assert {"ap", "roc_auc"} <= set(p["per_fold"][0]["a.pt"])
    assert p["mean"]["a.pt"]["ap"] == pytest.approx(
        np.mean([r["a.pt"]["ap"] for r in p["per_fold"]])
    )
    assert p["mean"]["a.pt"]["ap"] > p["mean"]["b.pt"]["ap"]
    assert p["difference"]["ap"]["delta"] > 0
    out = capsys.readouterr().out
    assert out.index("mean AP") < out.index("mean AUC") and "coverage: 3/3 panels" in out


def test_compare_bootstrap_uses_matched_rows(compare_stubbed, monkeypatch):
    seen: list[tuple] = []
    real = bench.paired_auc_difference

    def spy(y, a, b, *args, **kw):
        seen.append((len(y), len(a), len(b), bool(np.isnan(a).any() or np.isnan(b).any())))
        return real(y, a, b, *args, **kw)

    monkeypatch.setattr(bench, "paired_auc_difference", spy)
    bench.run_credit_compare("a.pt", "b.pt", folds=3, n_boot=150, panels=compare_stubbed)
    assert len(seen) == 6  # three panels, AP and ROC-AUC
    assert all(n == na == nb and not nan for n, na, nb, nan in seen)  # every row scored by both


def test_compare_applies_holm_across_panels(compare_stubbed):
    rec = bench.run_credit_compare("a.pt", "b.pt", folds=3, n_boot=300, panels=compare_stubbed)
    for key in ("ap", "roc_auc"):
        raw = [rec["panels"][n]["difference"][key]["p"] for n in rec["panels"]]
        holm = [rec["panels"][n]["difference"][key]["p_holm"] for n in rec["panels"]]
        assert holm == pytest.approx(bench.holm_adjusted_p(raw))
        assert all(h >= r for h, r in zip(holm, raw, strict=True))


def test_compare_skips_oversized_panel_with_reason_and_writes_json(
    compare_stubbed, monkeypatch, tmp_path
):
    monkeypatch.setattr(
        bench.FinancialTFM, "load", lambda p: SimpleNamespace(cfg=SimpleNamespace(max_features=4))
    )
    out = tmp_path / "r.json"
    rec = bench.run_credit_compare(
        "a.pt", "b.pt", folds=3, n_boot=150, panels=compare_stubbed, out=str(out)
    )
    assert rec["coverage"]["n_skipped"] == 3 and rec["panels"] == {}
    assert "features" in next(iter(rec["coverage"]["skipped"].values()))
    assert json.loads(out.read_text())["coverage"]["n_attempted"] == 3


def test_cli_compare_dispatches(monkeypatch):
    seen: dict = {}
    monkeypatch.setattr(bench, "run_credit_compare", lambda a, b, **k: seen.update(a=a, b=b, **k))
    monkeypatch.setattr(
        "sys.argv", ["fintfm-bench", "--credit", "--folds", "5", "--compare", "x.pt,y.pt"]
    )
    bench.main()
    assert (seen["a"], seen["b"], seen["folds"]) == ("x.pt", "y.pt", 5)
