"""The §27 seed study on synthetic data: plumbing only, SMOKE-TEST, never a score."""

import numpy as np
import pytest

from fintfm.experiments.boosting_seeds import (
    format_report,
    run_seed_study,
    stratified_subsample,
)


def _panel(n: int = 1200, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    y = (X[:, 0] + rng.normal(scale=1.0, size=n) > 1.4).astype(int)
    return X, y


def _signal(Xs, ys, Xt, seed):
    return 1 / (1 + np.exp(-Xt[:, 0]))


def _noise(Xs, ys, Xt, seed):
    return np.random.default_rng(seed).uniform(size=len(Xt))


def test_subsample_holds_the_stated_positive_count() -> None:
    X, y = _panel()
    Xs, ys = stratified_subsample(X, y, 100, 0.05, np.random.default_rng(0))
    assert len(ys) == 100 and ys.sum() == 5 and Xs.shape == (100, 4)


def test_study_reports_every_seed_and_orders_the_arms() -> None:
    X, y = _panel()
    rec = run_seed_study(
        X, y, {"fintfm": _signal, "catboost": _noise}, sizes=(100, 250), seeds=(0, 1, 2), n_boot=200
    )
    assert rec["skipped"] == {}
    for size in ("100", "250"):
        cell = rec["sizes"][size]
        assert len(cell["fintfm"]["average_precision"]["per_seed"]) == 3
        ap = cell["paired"]["average_precision"]
        assert ap["delta_pooled"] > 0 and ap["seeds_model_ahead"] == 3
        assert ap["ci95"][0] > 0
        assert 0 < ap["p_holm_across_n"] <= 1
        assert cell["paired"]["roc_auc"]["p"] > 0


def test_holm_adjusts_upwards_across_sizes() -> None:
    X, y = _panel()
    rec = run_seed_study(
        X,
        y,
        {"fintfm": _signal, "catboost": _noise},
        sizes=(100, 250, 500),
        seeds=(0, 1),
        n_boot=200,
    )
    for size in ("100", "250", "500"):
        b = rec["sizes"][size]["paired"]["roc_auc"]
        assert b["p_holm_across_n"] >= b["p"]


def test_a_skipped_arm_leaves_the_paired_block_empty_not_wrong() -> None:
    X, y = _panel()
    rec = run_seed_study(
        X,
        y,
        {"fintfm": _signal, "catboost": lambda *a: None},
        sizes=(100,),
        seeds=(0, 1),
        n_boot=100,
    )
    assert rec["sizes"]["100"]["paired"]["average_precision"] is None


def test_oversized_request_is_listed_as_skipped() -> None:
    X, y = _panel(300)
    rec = run_seed_study(
        X, y, {"fintfm": _signal, "catboost": _noise}, sizes=(100, 5000), seeds=(0,), n_boot=100
    )
    assert "n5000" in rec["skipped"]


def test_missing_reference_is_an_error() -> None:
    X, y = _panel()
    with pytest.raises(ValueError, match="catboost"):
        run_seed_study(X, y, {"fintfm": _signal}, sizes=(100,), seeds=(0,))


def test_report_prints_average_precision_before_roc_auc() -> None:
    X, y = _panel()
    rec = run_seed_study(
        X, y, {"fintfm": _signal, "catboost": _noise}, sizes=(100,), seeds=(0, 1), n_boot=100
    )
    text = format_report(rec)
    assert text.index("AVERAGE PRECISION") < text.index("ROC-AUC") < text.index("ECE")
    assert "Holm across n" in text
