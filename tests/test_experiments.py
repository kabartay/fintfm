"""Guards for the Phase 1 ablation harness.

These run at a size that finishes in seconds. The point is that the harness is correct —
that compute really is matched and results really are re-derivable — not that a two-step
model learns anything.
"""

import json
from pathlib import Path

from fintfm.experiments.prior_ablation import VARIANTS, run_ablation, summarise
from fintfm.modeling.model import ModelConfig


def test_variants_isolate_exactly_one_variable():
    assert VARIANTS["financial"] == 1.0
    assert VARIANTS["generic"] == 0.0
    assert 0.0 < VARIANTS["mixed"] < 1.0


def test_ablation_matches_compute_and_writes_rederivable_results(tmp_path: Path):
    cfg = ModelConfig(
        max_features=8,
        max_classes=2,
        d_cell=8,
        d_model=16,
        n_heads=2,
        n_col_layers=1,
        n_layers=1,
        d_ff=16,
    )
    record = run_ablation(
        out_dir=tmp_path,
        steps=2,
        batch_size=2,
        n_rows=16,
        model_cfg=cfg,
        variants={"financial": 1.0, "generic": 0.0},
        horizons=(),  # training-harness test; real-data scoring is exercised separately
    )
    # the untrained control is added by default and is what makes a tie interpretable
    assert set(record["variants"]) == {"financial", "generic", "untrained"}
    assert record["variants"]["untrained"]["steps"] == 0
    assert record["variants"]["financial"]["steps"] == 2
    sizes = {v["n_parameters"] for v in record["variants"].values()}
    assert len(sizes) == 1, "compute must be matched across variants"

    written = json.loads((tmp_path / "results.json").read_text())
    assert written["config"]["steps"] == 2
    assert written["git_commit"]  # provenance is recorded, not just the numbers
    assert "created" in written


def test_summarise_reports_the_exit_condition():
    record = {
        "config": {"steps": 10},
        "variants": {
            "financial": {
                "n_parameters": 100,
                "metrics": {"d/balanced": {"roc_auc": 0.7, "ece": 0.01}},
            },
            "generic": {
                "n_parameters": 100,
                "metrics": {"d/balanced": {"roc_auc": 0.6, "ece": 0.02}},
            },
        },
    }
    text = summarise(record)
    assert "EXIT CONDITION" in text
    assert "+0.1000" in text  # financial ahead by 0.10 AUC


def test_summarise_says_so_rather_than_printing_nan_when_nothing_scored():
    """A skipped evaluation must read as "not evaluated", never as a nan verdict."""
    record = {
        "config": {"steps": 3},
        "variants": {
            "financial": {"n_parameters": 10, "metrics": {}},
            "generic": {"n_parameters": 10, "metrics": {}},
        },
    }
    text = summarise(record)
    assert "cannot be evaluated" in text
    # a bare "nan" substring also matches "fiNANcial"; check for a nan *value* instead
    assert "+nan" not in text
    assert "nan vs" not in text
    assert "delta" not in text


def test_untrained_control_can_be_disabled(tmp_path: Path):
    cfg = ModelConfig(
        max_features=6,
        max_classes=2,
        d_cell=8,
        d_model=16,
        n_heads=2,
        n_col_layers=1,
        n_layers=1,
        d_ff=16,
    )
    record = run_ablation(
        out_dir=tmp_path,
        steps=1,
        batch_size=2,
        n_rows=16,
        model_cfg=cfg,
        variants={"financial": 1.0},
        horizons=(),
        include_untrained_control=False,
    )
    assert set(record["variants"]) == {"financial"}


def test_sample_efficiency_sizes_are_ascending_and_documented():
    """The probe must sweep upward from genuinely small n, which is the regime under test."""
    import inspect

    from fintfm.experiments.prior_ablation import sample_efficiency_probe

    sig = inspect.signature(sample_efficiency_probe)
    sizes = sig.parameters["train_sizes"].default
    assert list(sizes) == sorted(sizes)
    assert min(sizes) <= 250, "must probe below 1000 rows, where the TFM advantage is claimed"
    assert max(sizes) >= 4000, "must reach the ~8000 crossover region to observe it"


# --- capability probes (docs/results/FINDINGS.md §42) --------------------------------------


def test_probes_have_the_ceilings_they_claim():
    """The probes are only diagnostic if their ceilings are what the docstring says.

    §42 happened because every measurement was on a benchmark where a single column scores
    0.9799, so a weak model looked competent. A probe whose ceiling is unknown repeats that.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    from fintfm.experiments.capability import make_probe

    def fit(kind, model):
        Xtr, ytr, Xte, yte = make_probe(kind, seed=0)
        return roc_auc_score(yte, model.fit(Xtr, ytr).predict_proba(Xte)[:, 1])

    # linear: a fitted linear model should essentially solve it
    assert fit("linear", LogisticRegression(max_iter=1000)) > 0.98
    # conjunction: trees represent an AND natively
    assert fit("conjunction", HistGradientBoostingClassifier(max_iter=100)) > 0.90
    # xor: no additive function separates parity, so logistic regression is pinned at chance
    assert fit("xor", LogisticRegression(max_iter=1000)) < 0.60
    assert fit("xor", HistGradientBoostingClassifier(max_iter=100)) > 0.80
    # noise: nothing can beat chance, and an arm that does is leaking
    assert fit("noise", HistGradientBoostingClassifier(max_iter=100)) < 0.60


def test_probe_base_rate_is_respected():
    from fintfm.experiments.capability import PROBES, make_probe

    for kind in PROBES:
        _, ytr, _, yte = make_probe(kind, rate=0.05, seed=1)
        for y in (ytr, yte):
            assert 0.02 < y.mean() < 0.09, (kind, y.mean())


def test_unknown_probe_is_refused():
    import pytest as _pytest

    from fintfm.experiments.capability import make_probe

    with _pytest.raises(ValueError, match="unknown probe"):
        make_probe("quadratic")


def test_run_includes_an_untrained_control(tmp_path):
    """The floor must be present automatically, not remembered."""
    from fintfm.experiments.capability import run
    from fintfm.modeling.model import FinancialTFM, ModelConfig

    ckpt = tmp_path / "m.pt"
    FinancialTFM(
        ModelConfig(
            max_features=20, d_model=16, d_cell=8, n_layers=1, n_col_layers=1, max_classes=2
        )
    ).save(str(ckpt), trained_objectives=("classification",))
    rec = run({"probe": str(ckpt)}, tmp_path, seeds=(0,), max_context=200, n_features=20)
    arms = {r["arm"] for r in rec["results"]}
    assert "untrained_control" in arms
    assert "logistic_regression" in arms


def test_untrained_control_is_reproducible(tmp_path):
    """A floor that moves is not a floor.

    Unseeded, the control scored 0.344 to 0.569 on `linear` across invocations, so whether a
    trained model "cleared the control" depended on the draw (docs/results/FINDINGS.md §47).
    """
    from fintfm.experiments.capability import run
    from fintfm.modeling.model import FinancialTFM, ModelConfig

    ckpt = tmp_path / "m.pt"
    FinancialTFM(
        ModelConfig(
            max_features=20, d_model=16, d_cell=8, n_layers=1, n_col_layers=1, max_classes=2
        )
    ).save(str(ckpt), trained_objectives=("classification",))

    def control_scores():
        rec = run({"probe": str(ckpt)}, tmp_path, seeds=(0,), max_context=200, n_features=20)
        return [r["auc_mean"] for r in rec["results"] if r["arm"] == "untrained_control"]

    first, second = control_scores(), control_scores()
    assert first == second, f"untrained control is not reproducible: {first} vs {second}"


def test_feature_sweep_reproduces_the_shape_finding_49_reports(tmp_path):
    """§49's sweep must be re-runnable, and logistic regression must stay flat.

    The flat baseline is what makes the finding interpretable: if every arm degraded with
    width, the task would be getting harder rather than the model failing to aggregate.
    """
    import numpy as np

    from fintfm.experiments.capability import feature_sweep
    from fintfm.modeling.model import FinancialTFM, ModelConfig

    ckpt = tmp_path / "m.pt"
    FinancialTFM(
        ModelConfig(
            max_features=40, d_model=16, d_cell=8, n_layers=1, n_col_layers=1, max_classes=2
        )
    ).save(str(ckpt), trained_objectives=("classification",))
    sweep = feature_sweep({"probe": str(ckpt)}, widths=(5, 40), seeds=(0,), max_context=200)

    assert sweep["widths"] == [5, 40]
    assert set(sweep["arms"]) == {"probe", "logistic_regression"}
    lr = sweep["arms"]["logistic_regression"]
    assert all(v > 0.95 for v in lr), f"the linear baseline should be flat and high: {lr}"
    assert all(np.isfinite(v) for v in sweep["arms"]["probe"])


def test_base_rate_sweep_keeps_the_linear_baseline_flat(tmp_path):
    """§51's reading only follows if the task itself does not get harder as it balances.

    Logistic regression must hold near 1.0 across rates; if it fell too, the base-rate
    decline would be a property of the task rather than of the model.
    """
    import numpy as np

    from fintfm.experiments.capability import base_rate_sweep
    from fintfm.modeling.model import FinancialTFM, ModelConfig

    ckpt = tmp_path / "m.pt"
    FinancialTFM(
        ModelConfig(max_features=8, d_model=16, d_cell=8, n_layers=1, n_col_layers=1, max_classes=2)
    ).save(str(ckpt), trained_objectives=("classification",))
    sweep = base_rate_sweep(
        {"probe": str(ckpt)}, rates=(0.05, 0.50), n_features=5, seeds=(0,), max_context=300
    )

    assert sweep["rates"] == [0.05, 0.50]
    lr = sweep["arms"]["logistic_regression"]
    assert all(v > 0.95 for v in lr), f"baseline must stay flat across rates: {lr}"
    assert all(np.isfinite(v) for v in sweep["arms"]["probe"])


def test_bayes_ceiling_closed_form_matches_empirical_bayes_optimal_auc():
    """Task 39.3's explicit verification: the closed form must match reality, not assumption.

    Before this was trusted for a real measurement (docs/results/FINDINGS.md §74), the formula
    Phi(mu/sqrt(2)) was checked against the empirical AUC of the true Bayes-optimal statistic
    (the informative dimension itself) on 200,000 rows. Kept as a permanent regression test so
    the formula cannot silently drift from what the probe actually measures.
    """
    import numpy as np
    from sklearn.metrics import roc_auc_score

    from fintfm.experiments.capability import _bayes_optimal_mu

    rng = np.random.default_rng(0)
    for target in (0.6, 0.75, 0.9, 0.99):
        mu = _bayes_optimal_mu(target)
        n = 200_000
        y = (rng.random(n) < 0.5).astype(int)
        x1 = rng.normal(size=n) + mu * y  # the Bayes-optimal statistic itself
        empirical = roc_auc_score(y, x1)
        assert abs(empirical - target) < 0.01, (target, mu, empirical)


def test_bayes_ceiling_probe_runs_and_regret_is_consistent(tmp_path):
    """Shape and consistency check, not a claim about any checkpoint's actual regret."""
    from fintfm.experiments.capability import BAYES_AUC_TARGETS, bayes_ceiling_probe
    from fintfm.modeling.model import FinancialTFM, ModelConfig

    model = FinancialTFM(
        ModelConfig(max_features=8, d_model=16, d_cell=8, n_layers=1, n_col_layers=1, max_classes=2)
    ).eval()
    targets = (0.5, 0.9, 0.999)
    result = bayes_ceiling_probe(model, targets=targets, seeds=2, n=200, n_ensemble=2)

    assert set(result) == set(targets)
    for target, (achieved, regret) in result.items():
        assert 0.0 <= achieved <= 1.0
        assert abs(regret - (target - achieved)) < 1e-9
    assert set(BAYES_AUC_TARGETS) >= {0.5, 0.9, 0.999}


def test_multiclass_probe_is_solvable_by_the_model_it_was_generated_from():
    """The ceiling must be reachable, or a shortfall says nothing about the model.

    ``make_multiclass_probe`` draws labels as ``argmax(X @ W)``, so multinomial logistic
    regression is the correctly-specified model for it. If the baseline could not reach a
    high accuracy, a low fintfm number would be a property of the task rather than of the
    architecture -- exactly the confound §42 was created to prevent.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    from fintfm.experiments.capability import make_multiclass_probe

    for n_classes in (3, 10):
        Xtr, ytr, Xte, yte = make_multiclass_probe(
            n=4000, n_features=8, n_classes=n_classes, seed=0
        )
        assert set(ytr) == set(range(n_classes)), "every class must be present in context"
        acc = accuracy_score(yte, LogisticRegression(max_iter=1000).fit(Xtr, ytr).predict(Xte))
        assert acc > 0.9, f"K={n_classes}: ceiling must be reachable, got {acc:.4f}"


def test_multiclass_sweep_skips_checkpoints_that_cannot_represent_the_task(tmp_path):
    """A binary checkpoint must be recorded as skipped, never scored.

    Scoring a ``max_classes=2`` model on a 10-class task would produce a number the
    architecture forbids and invite a comparison against the multiclass arm that means
    nothing. The reason is carried into the record so a reader of the JSON sees the gap
    rather than inferring it from a missing row.
    """
    import numpy as np

    from fintfm.experiments.capability import multiclass_sweep
    from fintfm.modeling.model import FinancialTFM, ModelConfig

    base = {"max_features": 8, "d_model": 16, "d_cell": 8, "n_layers": 1, "n_col_layers": 1}
    binary = tmp_path / "binary.pt"
    wide = tmp_path / "wide.pt"
    FinancialTFM(ModelConfig(max_classes=2, **base)).save(
        str(binary), trained_objectives=("classification",)
    )
    FinancialTFM(ModelConfig(max_classes=5, **base)).save(
        str(wide), trained_objectives=("classification",)
    )

    sweep = multiclass_sweep(
        {"wide": str(wide), "binary": str(binary)},
        class_counts=(3,),
        n_features=6,
        seeds=(0,),
        max_context=300,
    )

    assert sweep["class_counts"] == [3]
    assert any("binary@K=3" in s for s in sweep["skipped"]), sweep["skipped"]
    assert np.isnan(sweep["arms"]["binary"]["accuracy"][0])
    assert np.isfinite(sweep["arms"]["wide"]["accuracy"][0])
    # The floor is measured, not assumed: unequal argmax regions put it above 1/K.
    assert sweep["majority_rate"][0] > 1 / 3


def test_regression_sweep_scores_a_quantile_head_checkpoint(tmp_path):
    """Task 48.3's comparison point: regression_sweep must route a head_type='quantile'
    checkpoint through FinancialTFMQuantileRegressor rather than the binned-head path, and
    every per-arm metric must come back finite and in its documented range.
    """
    import numpy as np

    from fintfm.experiments.capability import regression_sweep
    from fintfm.modeling.model import FinancialTFM, ModelConfig

    ckpt = tmp_path / "quantile.pt"
    FinancialTFM(
        ModelConfig(
            max_features=20,
            max_classes=3,
            d_cell=16,
            d_model=32,
            n_heads=2,
            n_col_layers=1,
            n_layers=1,
            d_ff=32,
            head_type="quantile",
            n_quantiles=9,
        )
    ).save(str(ckpt), trained_objectives=("quantile",))

    sweep = regression_sweep(
        {"quantile_head": str(ckpt)}, shapes=("linear",), seeds=(0,), max_context=200
    )

    assert sweep["skipped"] == []
    for metric in ("nrmse", "spearman", "coverage", "outer_mass"):
        for arm in ("quantile_head", "untrained_control", "ridge_gaussian", "binning_oracle"):
            value = sweep["arms"][arm][metric][0]
            if arm == "binning_oracle" and metric == "coverage":
                assert np.isnan(value)  # the oracle is a point, not a distribution
                continue
            assert np.isfinite(value), f"{arm}/{metric} is not finite: {value}"
    for arm in ("quantile_head", "untrained_control"):
        assert 0.0 <= sweep["arms"][arm]["coverage"][0] <= 1.0
        assert 0.0 <= sweep["arms"][arm]["outer_mass"][0] <= 1.0


def test_prior_ablation_main_guard_is_the_last_statement():
    """`python -m` must reach every function the CLI flags call, so the guard goes last."""
    import ast
    from pathlib import Path

    import fintfm.experiments.prior_ablation as mod

    last = ast.parse(Path(mod.__file__).read_text()).body[-1]
    assert isinstance(last, ast.If) and "__main__" in ast.unparse(last.test)


def test_bayesian_regret_is_zero_for_true_probabilities_and_positive_for_noise():
    """Task 39.18: R = AUC_Bayes - AUC_model, ~0 at the truth, positive for a noisy model."""
    import numpy as np

    from fintfm.experiments.capability import bayesian_regret

    rng = np.random.default_rng(0)
    p_true = rng.uniform(0.05, 0.95, size=20_000)
    y = (rng.random(20_000) < p_true).astype(int)

    exact = bayesian_regret(y, p_true, p_true)
    assert not exact["skipped"]
    assert exact["regret"] == 0.0 and exact["regret_norm"] == 0.0

    noisy = bayesian_regret(y, p_true, p_true + rng.normal(0, 0.5, size=p_true.shape))
    assert noisy["regret"] > 0.05
    assert abs(noisy["regret"] - (noisy["auc_bayes"] - noisy["auc_model"])) < 1e-12
    expected_norm = noisy["regret"] / (noisy["auc_bayes"] - 0.5)
    assert abs(noisy["regret_norm"] - expected_norm) < 1e-12


def test_bayesian_regret_skips_single_class_tasks_and_aggregates_the_rest():
    """A single-class task is skipped (AUC undefined), counted, and never averaged in."""
    import numpy as np

    from fintfm.experiments.capability import aggregate_regret, bayesian_regret

    p = np.array([0.2, 0.8, 0.5, 0.9])
    one_class = bayesian_regret(np.ones(4, dtype=int), p, p)
    assert one_class["skipped"] and one_class["regret"] is None

    ok = bayesian_regret(np.array([0, 1, 0, 1]), p, p[::-1].copy())
    agg = aggregate_regret([one_class, ok])
    assert agg["n_tasks"] == 1 and agg["n_skipped"] == 1
    assert agg["regret"] == ok["regret"]
    empty = aggregate_regret([one_class])
    assert empty["regret"] is None and empty["n_tasks"] == 0

    # Ceiling at chance: R_norm is 0 / 0, reported as None rather than NaN or inf.
    chance = bayesian_regret(
        np.array([0, 1, 0, 1]), np.full(4, 0.5), np.array([0.1, 0.9, 0.2, 0.8])
    )
    assert chance["regret_norm"] is None


def test_capability_record_carries_regret_beside_raw_auc(tmp_path):
    """The capability output gains regret fields; raw auc_mean results are unchanged."""
    import json

    from fintfm.experiments.capability import run, summarise
    from fintfm.modeling.model import FinancialTFM, ModelConfig

    ckpt = tmp_path / "m.pt"
    FinancialTFM(
        ModelConfig(
            max_features=20, d_model=16, d_cell=8, n_layers=1, n_col_layers=1, max_classes=2
        )
    ).save(str(ckpt), trained_objectives=("classification",))
    rec = run(
        {"probe": str(ckpt)},
        tmp_path,
        seeds=(0,),
        max_context=200,
        n_features=20,
        bayes_targets=(0.9,),
        bayes_seeds=2,
    )
    assert all("auc_mean" in r for r in rec["results"])
    cell = rec["bayes_ceiling"]["probe"]["0.9"]
    for key in ("auc_bayes", "auc_model", "regret", "regret_norm", "n_tasks", "n_skipped"):
        assert key in cell
    assert cell["n_tasks"] == 2 and len(cell["per_task"]) == 2
    assert abs(cell["regret"] - (cell["auc_bayes"] - cell["auc_model"])) < 1e-9
    assert cell["auc_bayes"] > 0.85  # empirical Bayes AUC on 800 query rows, nominal 0.9
    saved = json.loads((tmp_path / "capability.json").read_text())
    assert "bayes_ceiling" in saved
    assert "R_norm" in summarise(rec)


# --- calibrated gradient-boosting arms (sample-efficiency-regime 13.7) -------------------


def _gbm_factory():
    """Cheap boosting pipeline for the 13.7 tests."""
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline

    return make_pipeline(
        SimpleImputer(strategy="median"),
        GradientBoostingClassifier(n_estimators=10, random_state=0),
    )


def _toy_panel(n: int = 400, seed: int = 0):
    import numpy as np

    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4)).astype("float32")
    y = (X[:, 0] + rng.normal(scale=1.0, size=n) > 1.0).astype(int)
    return X, y


def test_holdout_calibration_never_sees_test_rows():
    """13.7: calibration and fit slices partition the training rows only."""
    import numpy as np

    from fintfm.experiments.prior_ablation import calibrated_gboost_holdout_arms

    X, y = _toy_panel()
    X_tr, y_tr, X_te, y_te = X[:250], y[:250], X[250:], y[250:]
    arms, info = calibrated_gboost_holdout_arms(
        _gbm_factory, X_tr, y_tr, X_te, y_te, calibration_frac=0.3, seed=0
    )
    fit, cal = np.array(info["fit_idx"]), np.array(info["cal_idx"])
    assert np.intersect1d(fit, cal).size == 0
    assert sorted(np.concatenate([fit, cal]).tolist()) == list(range(len(y_tr)))
    assert fit.max() < len(y_tr) and cal.max() < len(y_tr)  # indices into train, not test
    assert info["calibration_frac"] == 0.3
    assert info["n_cal"] == len(cal) and info["skipped"] is None
    for key in ("gboost_platt_holdout", "gboost_isotonic_holdout"):
        m = arms[key]
        assert m is not None and 0.0 <= m.ece <= 1.0
        assert m.average_precision > 0 and m.pred_sd >= 0  # AP and spread fields present


def test_holdout_calibration_skips_with_a_reason_on_degenerate_slices():
    """13.7: one-class or unstratifiable training rows are skipped and say why."""
    import numpy as np

    from fintfm.experiments.prior_ablation import calibrated_gboost_holdout_arms

    X, y = _toy_panel(60)
    X_te, y_te = X[:30], y[:30]
    one_pos = np.zeros(30, dtype=int)
    one_pos[0] = 1
    arms, info = calibrated_gboost_holdout_arms(_gbm_factory, X[30:], one_pos, X_te, y_te)
    assert all(v is None for v in arms.values())
    assert "stratify" in info["skipped"]
    # two positives and a tiny slice: whichever way it falls, never a crash, reason recorded
    two_pos = np.zeros(30, dtype=int)
    two_pos[:2] = 1
    arms, info = calibrated_gboost_holdout_arms(
        _gbm_factory, X[30:], two_pos, X_te, y_te, calibration_frac=0.1
    )
    assert info["skipped"] is not None or all(v is not None for v in arms.values())
    if info["skipped"]:
        assert all(v is None for v in arms.values())


def test_holdout_calibration_rejects_bad_fraction():
    import pytest

    from fintfm.experiments.prior_ablation import calibrated_gboost_holdout_arms

    X, y = _toy_panel(100)
    with pytest.raises(ValueError):
        calibrated_gboost_holdout_arms(_gbm_factory, X, y, X, y, calibration_frac=1.0)


def test_probe_output_carries_the_holdout_arms(monkeypatch):
    """13.7: the probe record has the new arms beside the old ones (stub TFM, no checkpoint)."""
    import numpy as np

    from fintfm.evaluation.datasets import CreditDataset
    from fintfm.experiments import prior_ablation as pa

    class _StubTFM:
        def __init__(self, *a, **k):
            pass

        def fit(self, X, y):
            self.rate = float(np.mean(y))
            return self

        def predict_proba(self, X):
            p = np.clip(0.5 * self.rate + 0.1 * (X[:, 0] > 0), 1e-3, 0.9)
            return np.column_stack([1 - p, p])

    monkeypatch.setattr(pa, "FinancialTFMClassifier", _StubTFM)
    X, y = _toy_panel(1200)
    ds = CreditDataset(X=X, y=y, name="toy", horizon_years=3, licence="x", attribution="x")
    rec = pa.sample_efficiency_probe("stub.pt", train_sizes=(100, 300), ds=ds, calibration_frac=0.4)
    assert rec["calibration_frac"] == 0.4
    for cell in rec["sizes"].values():
        for key in ("fintfm", "gboost", "gboost_sigmoid", "gboost_isotonic"):
            assert key in cell
        assert "gboost_platt_holdout" in cell and "gboost_isotonic_holdout" in cell
        assert cell["calibration_holdout"]["calibration_frac"] == 0.4
        assert "fit_idx" not in cell["calibration_holdout"]
