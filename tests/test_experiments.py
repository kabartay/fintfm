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
    cfg = ModelConfig(max_features=8, max_classes=2, d_cell=8, d_model=16, n_heads=2,
                      n_col_layers=1, n_layers=1, d_ff=16)
    record = run_ablation(
        out_dir=tmp_path, steps=2, batch_size=2, n_rows=16, model_cfg=cfg,
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
            "financial": {"n_parameters": 100, "metrics": {"d/balanced": {"roc_auc": 0.7, "ece": 0.01}}},
            "generic": {"n_parameters": 100, "metrics": {"d/balanced": {"roc_auc": 0.6, "ece": 0.02}}},
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
    cfg = ModelConfig(max_features=6, max_classes=2, d_cell=8, d_model=16, n_heads=2,
                      n_col_layers=1, n_layers=1, d_ff=16)
    record = run_ablation(
        out_dir=tmp_path, steps=1, batch_size=2, n_rows=16, model_cfg=cfg,
        variants={"financial": 1.0}, horizons=(), include_untrained_control=False,
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


# --- capability probes (docs/FINDINGS.md §42) --------------------------------------


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
    FinancialTFM(ModelConfig(max_features=20, d_model=16, d_cell=8, n_layers=1,
                             n_col_layers=1, max_classes=2)).save(
        str(ckpt), trained_objectives=("classification",))
    rec = run({"probe": str(ckpt)}, tmp_path, seeds=(0,), max_context=200, n_features=20)
    arms = {r["arm"] for r in rec["results"]}
    assert "untrained_control" in arms
    assert "logistic_regression" in arms


def test_untrained_control_is_reproducible(tmp_path):
    """A floor that moves is not a floor.

    Unseeded, the control scored 0.344 to 0.569 on `linear` across invocations, so whether a
    trained model "cleared the control" depended on the draw (docs/FINDINGS.md §47).
    """
    from fintfm.experiments.capability import run
    from fintfm.modeling.model import FinancialTFM, ModelConfig

    ckpt = tmp_path / "m.pt"
    FinancialTFM(ModelConfig(max_features=20, d_model=16, d_cell=8, n_layers=1,
                             n_col_layers=1, max_classes=2)).save(
        str(ckpt), trained_objectives=("classification",))

    def control_scores():
        rec = run({"probe": str(ckpt)}, tmp_path, seeds=(0,), max_context=200, n_features=20)
        return [r["auc_mean"] for r in rec["results"] if r["arm"] == "untrained_control"]

    first, second = control_scores(), control_scores()
    assert first == second, f"untrained control is not reproducible: {first} vs {second}"
