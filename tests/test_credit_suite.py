"""The DR4.3 credit suite: split protocol, top-decile AP, licence gate and an end-to-end run.

Synthetic panels only, monkeypatched in place of the real loaders -- the PPDai, Bondora and ULB
files never enter this repository. Covers what would fail silently: an out-of-time split
training on rows whose outcome falls in the test window, a borrower on both sides, a CSV that
drops a model, and a dataset scored before its licence is admitted.
"""

import csv
import dataclasses

import numpy as np
import pytest

from fintfm.evaluation import credit_suite as cs
from fintfm.prior.real_panel import EVAL_ONLY_SOURCES


def _panel(n: int = 600, dated: bool = False, grouped: bool = False, seed: int = 0,
           embargo: float = 0.0) -> cs.CreditPanel:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5))
    X[rng.random(X.shape) < 0.05] = np.nan
    logit = 2.0 * np.nan_to_num(X[:, 0]) - 1.5
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(np.int64)
    return cs.CreditPanel(
        name="synthetic", X=X, feature_names=tuple(f"f{i}" for i in range(5)), y=y,
        groups=rng.integers(0, n // 3, size=n) if grouped else None,
        time=np.sort(rng.uniform(0, 1000, size=n)) if dated else None, embargo=embargo,
    )


def test_suite_keys_are_all_evaluation_only():
    # A dataset this suite scores must be one the FinTFM-R sampler refuses to train on.
    assert set(cs.DATASETS) <= EVAL_ONLY_SOURCES
    for key, ds in cs.DATASETS.items():
        assert ds.key == key
        assert ds.licence.url.startswith("https://") and ds.licence.stated
        assert ds.licence.checked == cs.CHECKED and ds.licence.reason


def test_ap_top_decile_by_hand():
    # 20 rows, so k = 2. Ranking: pos, neg, ... -> precision@1 = 1, one hit; min(k, P=3) = 2.
    y = np.zeros(20, dtype=int)
    y[[0, 5, 9]] = 1
    p = -np.arange(20, dtype=float)
    assert cs.ap_top_decile(y, p) == pytest.approx(0.5)
    # a ranking that fills the decile with positives scores 1 whatever the base rate
    assert cs.ap_top_decile(y, y.astype(float)) == pytest.approx(1.0)
    one = np.zeros(1000, dtype=int)
    one[3] = 1
    assert cs.ap_top_decile(one, one.astype(float)) == pytest.approx(1.0)
    assert np.isnan(cs.ap_top_decile(np.zeros(10, dtype=int), np.ones(10)))


def test_out_of_time_splits_respect_the_window_the_embargo_and_the_borrower():
    panel = _panel(n=3000, dated=True, grouped=True, embargo=50.0)
    splits = cs.out_of_time_splits(panel)
    assert len(splits) == cs.N_SPLITS
    tests = np.concatenate([te for _, te in splits])
    # the test windows tile the latest half of the rows, each about a sixth, with no overlap
    assert len(np.unique(tests)) == len(tests)
    assert set(tests) == set(np.flatnonzero(panel.time >= np.quantile(panel.time, 0.5)))
    for tr, te in splits:
        assert abs(len(te) - len(panel.y) / 6) < 0.02 * len(panel.y)
        assert panel.time[tr].max() < panel.time[te].min() - panel.embargo
        assert not set(panel.groups[tr]) & set(panel.groups[te])
    # train grows split by split
    assert [len(tr) for tr, _ in splits] == sorted(len(tr) for tr, _ in splits)


def test_stratified_splits_partition_the_rows_and_keep_groups_whole():
    panel = _panel(n=900, grouped=True)
    splits = cs.stratified_splits(panel)
    tests = np.concatenate([te for _, te in splits])
    assert sorted(tests) == list(range(900))
    for tr, te in splits:
        assert not set(panel.groups[tr]) & set(panel.groups[te])
    ungrouped = cs.stratified_splits(_panel(n=900))
    rate = _panel(n=900).y.mean()
    for _, te in ungrouped:
        assert abs(_panel(n=900).y[te].mean() - rate) < 0.01


def test_a_single_class_split_is_refused_not_scored():
    panel = _panel(n=600, dated=True)
    y = panel.y.copy()
    y[panel.time >= np.quantile(panel.time, 5 / 6)] = 0
    with pytest.raises(ValueError, match="one class"):
        cs.suite_splits(dataclasses.replace(panel, y=y))


def test_an_unadmitted_licence_is_refused_before_any_file_is_read(tmp_path):
    # The packaged entries all wait on an owner decision; nothing under root exists.
    with pytest.raises(ValueError, match="licence not admitted"):
        cs.run(tmp_path / "out.csv", None, datasets=("ppdai",), root=tmp_path / "missing")
    with pytest.raises(ValueError, match="unknown dataset"):
        cs.run(tmp_path / "out.csv", None, datasets=("credit-g",), root=tmp_path)
    assert not (tmp_path / "out.csv").exists()


def _admit(monkeypatch, **loaders):
    admitted = dataclasses.replace(cs.DATASETS["ulb_fraud"].licence, include=True)
    monkeypatch.setattr(cs, "DATASETS", {
        key: cs.SuiteDataset(key, "https://example.invalid", loader, admitted)
        for key, loader in loaders.items()
    })


def test_end_to_end_run_writes_one_row_per_dataset_split_and_model(tmp_path, monkeypatch):
    import torch

    from fintfm.modeling.model import FinancialTFM, ModelConfig
    from fintfm.provenance import sha256_file

    _admit(
        monkeypatch,
        dated=lambda root: dataclasses.replace(_panel(n=900, dated=True, grouped=True,
                                                      embargo=10.0), name="dated"),
        undated=lambda root: dataclasses.replace(_panel(n=600, seed=1), name="undated"),
    )
    # a tiny LightGBM keeps the subprocess fast; the packaged value is pinned below
    monkeypatch.setattr(cs, "LIGHTGBM_PARAMS", {"n_estimators": 5})
    torch.manual_seed(0)
    ckpt = tmp_path / "m.pt"
    FinancialTFM(ModelConfig(max_features=8, max_classes=2, d_cell=8, d_model=16, n_layers=1,
                             n_col_layers=1, n_heads=2)).save(str(ckpt))
    out = tmp_path / "results.csv"
    assert cs.main(["--model", str(ckpt), "--out", str(out), "--datasets", "dated,undated",
                    "--root", str(tmp_path)]) == 0
    with out.open() as fh:
        rows = list(csv.DictReader(fh))
    assert tuple(rows[0]) == cs.COLUMNS
    models = ("fintfm", "logistic_regression", "lightgbm_1000")
    assert [(r["dataset"], r["split"], r["model"]) for r in rows] == [
        (d, str(k), m) for d in ("dated", "undated") for k in range(3) for m in models
    ]
    assert {r["split_kind"] for r in rows if r["dataset"] == "dated"} == {"out_of_time"}
    assert {r["split_kind"] for r in rows if r["dataset"] == "undated"} == {"stratified"}
    for r in rows:
        assert r["skipped"] == ""
        assert 0.0 <= float(r["average_precision"]) <= 1.0
        assert 0.0 <= float(r["roc_auc"]) <= 1.0
        assert 0.0 <= float(r["ap_top_decile"]) <= 1.0
        assert 0.0 < float(r["mean_predicted"]) < 1.0
        assert int(r["n_test_positive"]) > 0
    sha = sha256_file(ckpt)
    assert {r["checkpoint_sha256"] for r in rows if r["model"] == "fintfm"} == {sha}
    assert {r["checkpoint_sha256"] for r in rows if r["model"] != "fintfm"} == {""}
    # the learnable synthetic label is found by the baseline, so the scores are not noise
    lr = [float(r["roc_auc"]) for r in rows if r["model"] == "logistic_regression"]
    assert min(lr) > 0.7


def test_a_too_wide_dataset_skips_fintfm_and_still_scores_the_baselines(tmp_path, monkeypatch):
    import torch

    from fintfm.modeling.model import FinancialTFM, ModelConfig

    _admit(monkeypatch, undated=lambda root: _panel(n=600))
    torch.manual_seed(0)
    ckpt = tmp_path / "m.pt"
    FinancialTFM(ModelConfig(max_features=4, max_classes=2, d_cell=8, d_model=16, n_layers=1,
                             n_col_layers=1, n_heads=2)).save(str(ckpt))
    rows = cs.run(tmp_path / "r.csv", str(ckpt), datasets=("undated",), root=tmp_path,
                  with_lightgbm=False)
    fintfm = [r for r in rows if r["model"] == "fintfm"]
    assert len(fintfm) == 3 and all("takes 4 features" in r["skipped"] for r in fintfm)
    assert all(r["skipped"] == "" for r in rows if r["model"] == "logistic_regression")


def test_lightgbm_runs_at_one_thousand_trees():
    # §167: LightGBM's 100-tree default is under-trained; the suite must not fall back to it.
    assert cs.LIGHTGBM_PARAMS == {"n_estimators": 1000, "learning_rate": 0.05}


def test_bondora_frame_label_window_and_bookkeeping():
    import pandas as pd

    n = 6
    raw = pd.DataFrame({
        "ReportAsOfEOD": ["2021-07-20"] * n,
        "LoanDate": ["2019-01-01", "2019-01-01", "2019-06-01", "2020-01-01", "2020-07-21",
                     "2018-03-01"],
        # defaults inside, outside, none, inside, (dropped: window not closed), at the edge
        "DefaultDate": ["2019-06-01", "2020-03-01", None, "2020-12-31", "2021-01-01",
                        "2019-03-01"],
        "UserName": ["a", "a", "b", "c", "d", "e"],
        "NewCreditCustomer": [True, False, True, False, True, False],
        **{c: [1.0] * n for c in cs.BONDORA_NUMERIC},
        **{f: ["x"] * n for f, _ in cs.BONDORA_CATEGORICAL},
    })
    frame = cs.bondora_frame(raw)
    assert frame["label"].tolist() == [1, 0, 0, 1, 1]
    assert frame["_user"].tolist() == ["a", "a", "b", "c", "e"]
    assert list(frame.columns[-3:]) == ["label", "_user", "_loan_date"]
