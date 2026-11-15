"""33.4: coherence across horizons under V4FinBench's published protocol.

Synthetic predictions only; no real V4FinBench data and no checkpoint.
"""

import json

import numpy as np
import pytest

from fintfm.experiments.v4_protocol import (
    HORIZON_FILES,
    coherence_from_runs,
    curve_coherence,
)


def test_a_monotone_curve_set_scores_one_hundred_percent():
    c = np.array([[0.01, 0.02, 0.02, 0.05], [0.0, 0.1, 0.2, 0.9]])  # a tie is not a fall
    out = curve_coherence(c)
    assert out["fully_monotone"] == 1.0
    assert out["violations"] == 0 and out["violation_rate"] == 0.0


def test_a_crossing_set_scores_the_right_fraction():
    c = np.array(
        [
            [0.1, 0.2, 0.3],  # monotone
            [0.1, 0.3, 0.2],  # one fall
            [0.5, 0.4, 0.3],  # two falls
            [0.1, 0.1, 0.1],  # monotone
        ]
    )
    out = curve_coherence(c)
    assert out["fully_monotone"] == 0.5
    assert out["violations"] == 3 and out["n_steps"] == 8
    assert out["violation_rate"] == pytest.approx(3 / 8)


def test_curve_coherence_agrees_with_the_existing_counter_and_summariser():
    """Reuse, not redefinition: same count as hazard.py and same rates as term_structure."""
    torch = pytest.importorskip("torch")
    from fintfm.experiments.term_structure import summarise_curves
    from fintfm.modeling.hazard import coherence_violations

    c = np.random.RandomState(1).rand(200, 6)
    out = curve_coherence(c)
    assert out["violations"] == int(coherence_violations(torch.as_tensor(c)))
    ref = summarise_curves("x", c, np.zeros_like(c), 1, 0)
    assert out["violation_rate"] == pytest.approx(ref.violation_rate)
    assert out["fully_monotone"] == pytest.approx(ref.fully_monotone)


def _write_horizon_run(path, names, y, preds):
    """A minimal single-horizon run directory: one fold, the arms in ``preds``."""
    path.mkdir(parents=True)
    np.savez_compressed(
        path / "predictions_fold0.npz",
        y_true=y,
        company_name_test=np.array(names),
        **{f"pred_{a}": p for a, p in preds.items()},
    )
    arms = [{"arm": a, "folds": [{"fold": 0, "roc_auc": 0.9, "avg_precision": 0.2}]} for a in preds]
    (path / "v4_protocol.json").write_text(json.dumps({"arms": arms}))


def test_independent_per_horizon_arms_are_measured_like_a_joint_arm(tmp_path):
    """A monotone arm scores 100%; independent per-horizon arms are scored by the same rule."""
    names = ["a", "a", "b", "c"]  # company a has two rows
    y = np.array([0, 1, 0, 1])
    joint = {0: [0.1, 0.3, 0.2, 0.4], 1: [0.2, 0.4, 0.3, 0.5], 2: [0.3, 0.5, 0.4, 0.6]}
    indep_b = {0: 0.5, 1: 0.1, 2: 0.2}  # company b's independent curve crosses
    dirs = {}
    for h in (0, 1, 2):
        indep = [joint[h][0], joint[h][1], indep_b[h], joint[h][3]]
        dirs[h] = tmp_path / f"h{h}"
        _write_horizon_run(
            dirs[h], names, y, {"joint": np.array(joint[h]), "indep": np.array(indep)}
        )
    res = coherence_from_runs(dirs, folds=(0,))
    j = res["arms"]["joint"]["folds"][0]
    i = res["arms"]["indep"]["folds"][0]
    assert j["n_companies"] == 3 and j["fully_monotone"] == 1.0
    assert i["fully_monotone"] == pytest.approx(2 / 3)
    assert i["violations"] == 1 and i["violation_rate"] == pytest.approx(1 / 6)
    # level and discrimination travel with the coherence
    assert len(i["mean_predicted_by_horizon"]) == 3 and len(i["observed_rate_by_horizon"]) == 3
    assert i["roc_auc_by_horizon"] == [0.9] * 3 and i["ap_by_horizon"] == [0.2] * 3
    assert res["arms"]["indep"]["fully_monotone_mean"] == pytest.approx(2 / 3)


def test_only_companies_present_at_every_horizon_enter_the_curve(tmp_path):
    _write_horizon_run(tmp_path / "h0", ["a", "b"], np.array([0, 1]), {"m": np.array([0.1, 0.2])})
    _write_horizon_run(tmp_path / "h1", ["b", "c"], np.array([1, 0]), {"m": np.array([0.3, 0.9])})
    res = coherence_from_runs({0: tmp_path / "h0", 1: tmp_path / "h1"}, folds=(0,))
    assert res["arms"]["m"]["folds"][0]["n_companies"] == 1  # only b


def test_coherence_refuses_a_run_that_predates_the_company_names(tmp_path):
    for h in (0, 1):
        d = tmp_path / f"h{h}"
        d.mkdir()
        np.savez_compressed(d / "predictions_fold0.npz", y_true=np.zeros(2), pred_m=np.zeros(2))
        (d / "v4_protocol.json").write_text(json.dumps({"arms": []}))
    with pytest.raises(FileNotFoundError, match="company_name_test"):
        coherence_from_runs({0: tmp_path / "h0", 1: tmp_path / "h1"}, folds=(0,))
    with pytest.raises(ValueError, match="two horizons"):
        coherence_from_runs({0: tmp_path / "h0"}, folds=(0,))


def test_run_output_carries_company_names_and_the_coherence_field(tmp_path):
    """End to end on a synthetic two-horizon panel: run twice, then measure coherence."""
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    from fintfm.experiments.v4_protocol import run, summarise_coherence

    rng = np.random.RandomState(0)
    company = np.array([f"c{c}_f{k}" for c in range(4) for k in range(40) for _ in range(3)])
    for h in (0, 1):
        df = pd.DataFrame(
            {
                "company": company,
                "country": [int(c[1]) for c in company],
                "main_label": (rng.rand(len(company)) < 0.3).astype(int),
                "x1": rng.randn(len(company)),
                "x2": rng.randn(len(company)),
            }
        )
        df.to_parquet(tmp_path / HORIZON_FILES[h])
    recs = {
        h: run(None, tmp_path / f"o{h}", horizon=h, root=tmp_path, folds=(0,), with_boosting=False)
        for h in (0, 1)
    }
    assert recs[0]["coherence"]["measured"] is False
    saved = np.load(tmp_path / "o0" / "predictions_fold0.npz")
    assert saved["company_name_test"].shape == saved["y_true"].shape
    res = coherence_from_runs({0: tmp_path / "o0", 1: tmp_path / "o1"}, folds=(0,))
    row = res["arms"]["logistic_regression"]["folds"][0]
    assert row["n_companies"] > 0 and 0.0 <= row["fully_monotone"] <= 1.0
    assert "logistic_regression" in summarise_coherence(res)
