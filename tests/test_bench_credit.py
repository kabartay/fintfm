"""``fintfm-bench --credit`` reports the small-n probe by default (openspec 13.6).

No real data or checkpoint: panels are synthetic, the checkpoint loader and the in-context
classifier are stubs, so the test only checks the wiring.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from fintfm.evaluation import bench
from fintfm.evaluation.datasets import CreditDataset
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
