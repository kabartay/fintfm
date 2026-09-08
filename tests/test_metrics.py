import numpy as np
import pytest

from fintfm.evaluation.metrics import evaluate_binary, expected_calibration_error, recall_at_top_k


def test_perfect_calibration_has_near_zero_ece():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, size=200_000)
    y = (rng.uniform(size=p.shape) < p).astype(int)  # outcomes drawn at the stated rate
    ece, bins = expected_calibration_error(y, p)
    assert ece < 0.01
    assert len(bins) == 10


def test_miscalibrated_model_is_caught_despite_perfect_auc():
    """The point of tracking calibration: perfect ranking, badly wrong probabilities."""
    y = np.array([0] * 900 + [1] * 100)
    p = np.where(y == 1, 0.99, 0.5)  # ranks perfectly, wildly overstates risk
    m = evaluate_binary(y, p)
    assert m.roc_auc == 1.0
    assert m.ece > 0.4
    assert m.brier > 0.2
    assert m.mean_predicted > 5 * m.base_rate


def test_recall_at_top_k():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    p = np.array([0.9, 0.8, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    assert recall_at_top_k(y, p, 2) == 1.0
    assert recall_at_top_k(y, p, 1) == 0.5
    assert recall_at_top_k(np.zeros(5, dtype=int), p[:5], 2) == 0.0


def test_auc_is_nan_for_single_class_rather_than_invented():
    m = evaluate_binary(np.zeros(10, dtype=int), np.full(10, 0.3))
    assert np.isnan(m.roc_auc)
    assert m.n_positive == 0


def test_shape_mismatch_rejected():
    with pytest.raises(ValueError, match="shape mismatch"):
        evaluate_binary(np.array([0, 1]), np.array([0.5]))


def test_dataset_loaders_declare_period_labels_honestly():
    """Both currently loadable panels are cross-sectional; nothing may claim otherwise.

    `has_period_labels` gates time-based evaluation. If a loader wrongly claimed True, a
    random split would be reported as out-of-time validation — which is the exact failure
    a supervisory reviewer looks for. Guarded here rather than trusted.
    """
    from fintfm.evaluation.datasets import CACHE_DIR

    if not (CACHE_DIR / "taiwan_bankruptcy.zip").exists():
        pytest.skip("dataset not cached; loader test needs network")
    from fintfm.evaluation import load_taiwan_bankruptcy

    ds = load_taiwan_bankruptcy()
    assert ds.has_period_labels is False
    assert ds.X.shape == (6819, 95)
    assert ds.licence == "CC-BY-4.0"
    assert "CC BY 4.0" in ds.attribution  # attribution is a licence obligation, not a nicety
    assert 0.02 < ds.default_rate < 0.05


def test_paired_auc_difference_detects_a_real_gap_and_ignores_a_fake_one():
    """The guard against reading a win count as a result."""
    from fintfm.evaluation.metrics import paired_auc_difference

    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(900), np.ones(100)].astype(int)
    strong = rng.uniform(size=1000) * 0.3 + y * 0.6  # clearly informative
    weak = rng.uniform(size=1000)  # pure noise

    delta, (lo, hi), p = paired_auc_difference(y, strong, weak, n_boot=500)
    assert delta > 0.2
    assert lo > 0  # CI excludes zero
    assert p < 0.05

    # two draws of the same noise process: no real difference to find
    noise_a, noise_b = rng.uniform(size=1000), rng.uniform(size=1000)
    d2, (lo2, hi2), p2 = paired_auc_difference(y, noise_a, noise_b, n_boot=500)
    assert lo2 < 0 < hi2  # CI straddles zero
    assert p2 > 0.05


def test_holm_bonferroni_is_stricter_than_raw_alpha():
    from fintfm.evaluation.metrics import holm_bonferroni

    # 0.04 would pass a raw 0.05 test but must not survive 10 comparisons
    assert holm_bonferroni([0.04] * 10) == [False] * 10
    # a single very small p-value survives
    assert holm_bonferroni([0.0001, 0.9, 0.9])[0] is True
    assert holm_bonferroni([]) == []
    # NaNs are non-significant, never silently significant
    assert holm_bonferroni([float("nan")]) == [False]
