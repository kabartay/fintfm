"""Tests for the schema-variation harness (task 39.12). SMOKE-TEST only."""

import json

import numpy as np
import pytest
import torch
from sklearn.metrics import roc_auc_score

from fintfm.experiments import schema_variation as sv
from fintfm.modeling.model import FinancialTFM, ModelConfig

ROWS = (64, 4000)


def _tiny(max_features: int = 12) -> FinancialTFM:
    torch.manual_seed(1)
    return FinancialTFM(
        ModelConfig(
            max_features=max_features, max_classes=2, d_model=16, n_heads=2, n_layers=1, d_ff=32
        )
    )


def test_bayes_auc_identical_across_widths_and_on_target():
    y_ref = None
    for width in (5, 10, 50, 200):
        X, y, pb = sv.make_base(0, width, 3, ROWS)
        assert roc_auc_score(y[64:], pb[64:]) == pytest.approx(sv.TARGET_AUC, abs=0.02)
        if y_ref is None:
            y_ref, pb_ref, X_ref = y, pb, X
        np.testing.assert_array_equal(y, y_ref)  # same labels
        np.testing.assert_array_equal(pb, pb_ref)  # same true probabilities
        np.testing.assert_array_equal(X[:, :5], X_ref[:, :5])  # prefix-stable columns


def test_schema_edits_never_change_the_bayes_probability():
    Xb, _, pb = sv.make_base(0, 30, 3, ROWS)
    for sc in sv.default_schemas(widths=(30,), fixed_width=30):
        if sc.n_signal != 3 or sc.width != 30:
            continue
        X, _, perm = sv.apply_schema(Xb, sc, 0)
        # true probability recomputed from the signal columns found in the transformed table
        # (undo scaling is not needed: scaled variants are checked separately below)
        if not sc.scaled:
            cols = [int(np.where(perm == j)[0][0]) for j in range(3)]
            np.testing.assert_array_equal(sv.bayes_probability(X[:, cols], pb_mu(0)), pb)


def pb_mu(_seed: int) -> float:
    from fintfm.experiments.capability import _bayes_optimal_mu

    return _bayes_optimal_mu(sv.TARGET_AUC)


def test_signal_count_variants_share_the_bayes_auc():
    aucs = []
    for s in (1, 3, 10):
        _, y, pb = sv.make_base(0, 20, s, (64, 20000))
        aucs.append(roc_auc_score(y[64:], pb[64:]))
    assert aucs == pytest.approx([sv.TARGET_AUC] * 3, abs=0.01)


def test_permutation_and_scaling_are_exact_transforms():
    Xb, _, _ = sv.make_base(0, 20, 3, (32, 32))
    X, relevant, perm = sv.apply_schema(Xb, sv.Schema("p", 20, order="shuffled"), 0)
    np.testing.assert_array_equal(X, Xb[:, perm])
    assert set(perm[relevant].tolist()) == {0, 1, 2}
    last, rel_last, _ = sv.apply_schema(Xb, sv.Schema("l", 20, order="signal_last"), 0)
    np.testing.assert_array_equal(last[:, -3:], Xb[:, :3])
    assert rel_last.tolist() == [17, 18, 19]
    sc, _, _ = sv.apply_schema(Xb, sv.Schema("s", 20, scaled=True), 0)
    assert sc.std(axis=0).max() / sc.std(axis=0).min() > 1e3
    for j in range(20):  # exact affine map per column: correlation with the base column is 1
        assert abs(np.corrcoef(sc[:, j], Xb[:, j])[0, 1]) == pytest.approx(1.0)


def test_column_edits_only_touch_noise_columns():
    Xb, _, _ = sv.make_base(0, 30, 3, (200, 200))
    sch = sv.Schema("e", 30, n_dup=4, n_const=3, n_cat=5, missing=0.4)
    X, relevant, _ = sv.apply_schema(Xb, sch, 0)
    np.testing.assert_array_equal(X[:, :3], Xb[:, :3])  # signal untouched, no NaN in it
    np.testing.assert_array_equal(X[:, 26:], Xb[:, [3, 1, 3, 0]])  # copies, filled from the right
    assert relevant.tolist() == [0, 1, 2, 27, 29]
    assert (X[:, 23:26] == 3.5).all()
    assert set(np.unique(X[:, 18:23])) <= {0.0, 1.0, 2.0, 3.0}
    assert 0.3 < np.isnan(X[:, 3:18]).mean() < 0.5
    assert not np.isnan(X[:, 18:]).any()


def test_deterministic_and_seed_sensitive():
    sch = sv.Schema("d", 20, order="shuffled", scaled=True, n_dup=2, missing=0.2)
    Xb = sv.make_base(0, 20, 3, (16, 16))[0]
    a = sv.apply_schema(Xb, sch, 3)
    b = sv.apply_schema(sv.make_base(0, 20, 3, (16, 16))[0], sch, 3)
    np.testing.assert_array_equal(a[0], b[0])
    assert not np.array_equal(a[2], sv.apply_schema(Xb, sch, 4)[2])
    with pytest.raises(ValueError, match="noise"):
        sv.apply_schema(np.zeros((4, 6)), sv.Schema("x", 6, n_dup=4), 0)


def test_recall_at_k():
    assert sv.recall_at_k(np.array([0.9, 0.8, 0.0, 0.1]), np.array([0, 1])) == 1.0
    assert sv.recall_at_k(np.array([0.9, 0.0, 0.8, 0.1]), np.array([0, 1])) == 0.5


def test_end_to_end_tiny_with_recorded_skip(tmp_path):
    schemas = [
        sv.Schema("w5", 5),
        sv.Schema("w20", 20),
        sv.Schema("w10_mixed", 10, order="shuffled", missing=0.3, n_cat=2, n_const=1),
    ]
    rec = sv.run(
        {},
        tmp_path,
        schemas=schemas,
        seeds=(0, 1),
        n_context=48,
        n_query=60,
        n_importance=40,
        models={"tiny": _tiny(12)},
    )
    assert {s["width"] for s in rec["skipped"]} == {20}
    assert {s["arm"] for s in rec["skipped"]} == {"tiny", "untrained_control"}
    got = {(r["arm"], r["schema"]) for r in rec["results"]}
    assert ("tiny", "w20") not in got and ("logistic_regression", "w20") in got
    assert ("tiny", "w5") in got and ("untrained_control", "w10_mixed") in got
    assert len({r["auc_bayes_mean"] for r in rec["results"] if r["schema"] == "w5"}) == 1
    assert all(r["n_seeds"] == 2 for r in rec["results"])
    assert all(0.0 <= r["recall_at_k_mean"] <= 1.0 for r in rec["results"])
    on_disk = json.loads((tmp_path / "schema_variation.json").read_text())
    assert on_disk["skipped"] and "SKIPPED tiny at width 20" in sv.summarise(rec)


def test_duplicate_names_refused():
    with pytest.raises(ValueError, match="unique"):
        sv.run({}, None, schemas=[sv.Schema("a", 5), sv.Schema("a", 6)], models={"t": _tiny()})
