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

    delta, (lo, _hi), p = paired_auc_difference(y, strong, weak, n_boot=500)
    assert delta > 0.2
    assert lo > 0  # CI excludes zero
    assert p < 0.05

    # two draws of the same noise process: no real difference to find
    noise_a, noise_b = rng.uniform(size=1000), rng.uniform(size=1000)
    _d2, (lo2, hi2), p2 = paired_auc_difference(y, noise_a, noise_b, n_boot=500)
    assert lo2 < 0 < hi2  # CI straddles zero
    assert p2 > 0.05


def _pinned_inputs():
    """A small fixed problem whose row-bootstrap output is pinned below."""
    rng = np.random.default_rng(7)
    y = (rng.random(300) < 0.2).astype(int)
    return y, y * 0.3 + rng.random(300), rng.random(300)


def test_row_bootstrap_is_unchanged_by_cluster_support():
    """`groups=None` must stay byte-identical to the bootstrap every §144-§164 table used (AU2.1).

    Values computed with the code before `groups` existed.
    """
    from sklearn.metrics import average_precision_score

    from fintfm.evaluation.metrics import paired_auc_difference

    y, a, b = _pinned_inputs()
    assert paired_auc_difference(y, a, b, n_boot=300, seed=3, metric=average_precision_score) == (
        0.3495457401755093, (0.1996078476595747, 0.44564383730758306), 0.0033222591362126247
    )
    assert paired_auc_difference(y, a, b, n_boot=300, seed=3) == (
        0.24720380416582954, (0.11251744049050161, 0.34792782634226976), 0.0033222591362126247
    )


def test_singleton_groups_reproduce_the_row_bootstrap():
    """One row per group, in row order, is the row bootstrap draw for draw (AU2.1)."""
    from fintfm.evaluation.metrics import paired_auc_difference

    y, a, b = _pinned_inputs()
    row = paired_auc_difference(y, a, b, n_boot=300, seed=3)
    assert paired_auc_difference(y, a, b, n_boot=300, seed=3, groups=np.arange(300)) == row


def test_cluster_bootstrap_is_wider_on_clustered_data():
    """Rows of one company are not independent; resampling companies must widen the CI (AU2.1).

    Each company's rows share its label and a company-level error in each model, as V4FinBench's
    ~5 rows per company share a balance sheet. The row bootstrap counts them as independent.
    """
    from sklearn.metrics import average_precision_score

    from fintfm.evaluation.metrics import paired_auc_difference

    rng = np.random.default_rng(0)
    n_companies, rows_each = 300, 5
    company = np.repeat(np.array([f"c{i:04d}" for i in range(n_companies)]), rows_each)
    y = np.repeat((rng.random(n_companies) < 0.1).astype(int), rows_each)
    shared_a = np.repeat(rng.normal(size=n_companies), rows_each)
    shared_b = np.repeat(rng.normal(size=n_companies), rows_each)
    noise = rng.normal(scale=0.2, size=(2, len(y)))
    a = y * 1.5 + shared_a + noise[0]
    b = y * 1.5 + shared_b + noise[1]
    kw = {"n_boot": 500, "metric": average_precision_score}
    d_row, (lo_r, hi_r), _ = paired_auc_difference(y, a, b, **kw)
    d_grp, (lo_g, hi_g), _ = paired_auc_difference(y, a, b, groups=company, **kw)
    assert d_row == d_grp  # the point estimate does not depend on how we resample
    assert hi_g - lo_g > 1.5 * (hi_r - lo_r)


def test_cluster_bootstrap_refuses_misaligned_groups():
    from fintfm.evaluation.metrics import paired_auc_difference

    y, a, b = _pinned_inputs()
    with pytest.raises(ValueError, match="groups"):
        paired_auc_difference(y, a, b, groups=np.arange(299))


def test_holm_bonferroni_is_stricter_than_raw_alpha():
    from fintfm.evaluation.metrics import holm_bonferroni

    # 0.04 would pass a raw 0.05 test but must not survive 10 comparisons
    assert holm_bonferroni([0.04] * 10) == [False] * 10
    # a single very small p-value survives
    assert holm_bonferroni([0.0001, 0.9, 0.9])[0] is True
    assert holm_bonferroni([]) == []
    # NaNs are non-significant, never silently significant
    assert holm_bonferroni([float("nan")]) == [False]


def test_constant_base_rate_predictor_is_flagged_degenerate():
    """Spec E9. The failure finding 17 exposed: perfect calibration, zero content.

    A predictor returning the base rate for everyone has ECE near zero and AUC exactly 0.5.
    It must never read as a success, and it beat every real model on ECE in this project's
    own measurements before this guard existed.
    """
    y = np.r_[np.zeros(953), np.ones(47)].astype(int)
    m = evaluate_binary(y, np.full(1000, 0.047))
    assert m.roc_auc == 0.5
    assert m.ece < 0.001  # better calibrated than any trained model here
    assert m.is_degenerate, "a feature-free predictor must be flagged, not celebrated"
    assert abs(m.brier_skill) < 0.01  # ~zero skill against its own reference
    assert "DEGENERATE" in m.summary()


def test_brier_skill_is_positive_only_for_a_model_with_content():
    y = np.r_[np.zeros(953), np.ones(47)].astype(int)
    rng = np.random.default_rng(0)
    informative = np.clip(rng.uniform(size=1000) * 0.1 + y * 0.15, 0, 1)
    m = evaluate_binary(y, informative)
    assert m.brier_skill > 0.1
    assert not m.is_degenerate


def test_summary_always_shows_skill_beside_raw_brier():
    """Raw Brier is misleading on an imbalanced base rate; skill must travel with it."""
    y = np.r_[np.zeros(90), np.ones(10)].astype(int)
    s = evaluate_binary(y, np.full(100, 0.1)).summary()
    assert "skill=" in s
    assert "Brier=" in s


def test_cache_dir_is_anchored_on_the_repo_root_not_a_parent_count():
    """Regression: the package move silently duplicated 13 MB of downloads.

    `CACHE_DIR` used `parents[2]`, which pointed at the repo root while this module lived at
    `src/fintfm/data.py` and at `src/` after it moved to `src/fintfm/evaluation/datasets.py`.
    Nothing failed; datasets were simply re-downloaded to the wrong place. Anchoring on
    `pyproject.toml` cannot break that way.
    """
    from fintfm.evaluation.datasets import CACHE_DIR

    assert CACHE_DIR.parent.name == "data"
    assert (CACHE_DIR.parent.parent / "pyproject.toml").exists(), (
        f"CACHE_DIR {CACHE_DIR} is not under the repo root"
    )
    assert "src" not in CACHE_DIR.parts


def test_v4finbench_loader_joins_rather_than_stacks():
    """Regression: the horizon files have different row counts and must be joined.

    h=0 has 1,000,087 rows and h=5 has 598,832, because a five-year-ahead label needs five
    more years of data. Stacking them positionally silently misaligns companies. Missing
    horizons are administrative censoring, not survival, so `n_observed` must vary.
    """
    from fintfm.evaluation.datasets import CACHE_DIR

    if not (CACHE_DIR / "v4finbench" / "company_years_h6.parquet").exists():
        pytest.skip("V4FinBench not fetched; run `uv run fintfm-fetch v4finbench`")
    from fintfm.evaluation import load_v4finbench

    ds = load_v4finbench(max_rows=5000)
    assert ds.has_period_labels and ds.year is not None
    assert ds.licence == "CC-BY-4.0" and "CC BY 4.0" in ds.attribution
    assert ds.X.shape[0] == ds.period.shape[0] == ds.n_observed.shape[0]
    # ragged observation is the point: not every row carries all six horizons
    assert ds.n_observed.min() < ds.n_horizons, "no censoring found; the join likely failed"
    assert ds.n_observed.max() <= ds.n_horizons
    # a defaulting row must default within its observed window
    defaulted = ds.period >= 0
    assert (ds.period[defaulted] < ds.n_observed[defaulted]).all()
    assert ((ds.period >= 0) == (ds.y == 1)).all()


@pytest.mark.parametrize("model", ["lightgbm", "catboost", "xgboost"])
def test_boosting_baselines_fit_with_torch_loaded(model):
    """Regression: fitting these in a torch process segfaulted (docs/results/FINDINGS.md §25).

    PyTorch bundles its own OpenMP runtime and the boosting libraries load the system one;
    two in a process crash on macOS, and KMP_DUPLICATE_LIB_OK does not help. The subprocess
    boundary is the fix, and this test is the guard — note it imports torch deliberately.
    """
    import torch  # noqa: F401 - present precisely because it used to cause the crash

    from fintfm.evaluation.boosting import fit_predict_boosting

    rng = np.random.default_rng(0)
    X = rng.normal(size=(240, 5))
    y = (rng.random(240) < 0.25).astype(int)
    p = fit_predict_boosting(model, X[:180], y[:180], X[180:])
    if p is None:
        pytest.skip(f"{model} unavailable in this environment")
    assert p.shape == (60,)
    assert ((p >= 0) & (p <= 1)).all()


def test_unknown_boosting_baseline_is_rejected():
    from fintfm.evaluation.boosting import fit_predict_boosting

    with pytest.raises(ValueError, match="unknown baseline"):
        fit_predict_boosting("not_a_model", np.zeros((4, 2)), np.zeros(4), np.zeros((2, 2)))
