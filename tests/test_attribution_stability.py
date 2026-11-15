"""Attribution stability harness and thresholds (openspec 12.5).

The thresholds are the constants in ``fintfm.experiments.attribution_stability``, recorded in
``openspec/changes/zero-shot-attribution/tasks.md`` before any measurement. The first two tests
are SMOKE-TEST of the harness (any attribution function, a tiny untrained model through the
public classifier). The third asserts the thresholds for permutation attribution of the
``v4-cellattn-labels`` checkpoint on a pinned prior task; it skips where the (gitignored)
checkpoint is absent.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from fintfm.experiments import attribution_stability as ast_
from fintfm.modeling.model import FinancialTFM, ModelConfig
from fintfm.prior.financial import sample_financial_task

CHECKPOINT = Path(__file__).resolve().parents[1] / "runs" / "v4-cellattn-labels.pt"
PINNED_TASK_SEED = 0


def _corr_attribution(Xc, yc, Xe, ye, seed):
    """Toy attribution: absolute correlation of each explain column with the label."""
    return np.array([abs(np.corrcoef(Xe[:, j], ye)[0, 1]) for j in range(Xe.shape[1])])


def test_harness_accepts_any_function_and_reports_stability() -> None:
    """A model-free attribution function passes the recorded thresholds on a clean signal."""
    rng = np.random.default_rng(0)
    n = 3000
    X = rng.normal(size=(n, 8))
    y = (rng.random(n) < 1 / (1 + np.exp(-(2.5 * X[:, 0] + 1.5 * X[:, 1] - 0.5 * X[:, 2])))).astype(
        int
    )
    res = ast_.stability(
        X,
        y,
        _corr_attribution,
        "toy",
        "corr",
        seeds=(0, 1),
        n_resamples=3,
        n_context=500,
        n_explain=800,
        top_k=3,
    )
    assert res.top_features[0]["index"] == 0
    assert res.resample_spearman_mean >= ast_.MIN_RESAMPLE_SPEARMAN
    assert res.seed_spearman_mean >= ast_.MIN_SEED_SPEARMAN
    assert res.seed_topk_jaccard_mean >= ast_.MIN_SEED_TOPK_JACCARD
    assert abs(sum(res.mean_importance) - 1.0) < 1e-9


def test_permutation_attribution_runs_through_public_classifier() -> None:
    """Shape, non-negativity and normalisation of the permutation method (untrained model)."""
    torch.manual_seed(0)
    cfg = ModelConfig(max_features=6, max_classes=3, d_model=16, n_heads=2, n_layers=1, d_ff=32)
    fn = ast_.permutation_attribution(FinancialTFM(cfg), n_repeats=2)
    rng = np.random.default_rng(1)
    X = rng.normal(size=(120, 5))
    y = (X[:, 0] + 0.5 * rng.normal(size=120) > 0).astype(int)
    imp = fn(X[:60], y[:60], X[60:], y[60:], 0)
    assert imp.shape == (5,) and np.all(imp >= 0)
    assert imp.sum() == pytest.approx(1.0) or imp.sum() == 0.0


@pytest.mark.skipif(
    not CHECKPOINT.exists() or not os.environ.get("FINTFM_SLOW_ATTRIBUTION"),
    reason="needs runs/v4-cellattn-labels.pt (gitignored) and FINTFM_SLOW_ATTRIBUTION=1; "
    "about 13 minutes on a loaded CPU",
)
@pytest.mark.xfail(
    strict=True,
    reason="2026-11-15 SMOKE-TEST of the recorded thresholds: resample Spearman 0.409 < 0.60 "
    "on the pinned task; the thresholds are not edited after measuring",
)
def test_permutation_attribution_meets_recorded_thresholds_on_pinned_prior_task() -> None:
    """Context-resample and seed stability of the checkpoint's permutation attribution."""
    task = sample_financial_task(np.random.default_rng(PINNED_TASK_SEED), 1500, max_features=24)
    X, y = np.asarray(task.X, dtype=float), np.asarray(task.y).astype(int)
    fn = ast_.permutation_attribution(str(CHECKPOINT), n_repeats=2, max_context=400)
    res = ast_.stability(
        X,
        y,
        fn,
        "prior-task-0",
        "permutation",
        seeds=(0, 1),
        n_resamples=3,
        n_context=400,
        n_explain=400,
        top_k=5,
    )
    assert res.resample_spearman_mean >= ast_.MIN_RESAMPLE_SPEARMAN, res.resample_spearman_mean
    assert res.seed_spearman_mean >= ast_.MIN_SEED_SPEARMAN, res.seed_spearman_mean
    assert res.seed_topk_jaccard_mean >= ast_.MIN_SEED_TOPK_JACCARD, res.seed_topk_jaccard_mean
