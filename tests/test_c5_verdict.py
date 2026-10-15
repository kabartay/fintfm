"""§159's pre-registered decision rules behave exactly as written, at their boundaries.

`scripts/c5_verdict.py` is the only place those rules are applied. A rule that drifted from
§159's text -- a `>=` for a `>`, a yardstick quietly changed -- would turn a pre-registered
verdict into a post-hoc one without anything looking wrong, so the boundaries are pinned here.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from c5_verdict import YARDSTICK, fold_diffs, rule1, rule2, rule3, rule_helps


def test_yardstick_is_section_158s_measured_gap():
    assert YARDSTICK == 0.0073


def test_rule1_needs_both_seeds_and_the_yardstick():
    assert rule1(0.010, 0.010) == "helps"
    assert rule1(0.020, -0.001) == "null"  # one seed, however large, is not enough
    assert rule1(0.007, 0.007) == "null"  # same sign but within the yardstick
    assert rule1(-0.010, -0.010) == "hurts"
    assert rule1(-0.0073, -0.0073) == "null"  # the boundary itself is not past it


def test_rule2_is_sign_agreement_only():
    assert rule2(0.001, 0.020)
    assert rule2(-0.001, -0.020)
    assert not rule2(0.001, -0.001)
    assert not rule2(0.0, 0.010)


def test_rule3_needs_the_mean_and_four_negative_folds():
    assert rule3([-0.02, -0.02, -0.02, -0.02, 0.01]) == "loss"
    assert rule3([-0.04, -0.04, -0.04, 0.01, 0.01]) == "null"  # big mean, only 3 negative
    assert rule3([-0.005, -0.005, -0.005, -0.005, -0.005]) == "null"  # 5 negative, small mean


def test_helps_rule_needs_the_mean_and_four_folds_of_its_sign():
    """§163's rule, two-sided (audit AU2.9)."""
    assert rule_helps([0.01, 0.01, 0.01, 0.01, -0.001]) == "helps"
    assert rule_helps([0.04, 0.04, 0.04, -0.01, -0.01]) == "null"  # big mean, only 3 positive
    assert rule_helps([0.007, 0.007, 0.007, 0.007, 0.007]) == "null"  # 5 positive, small mean
    assert rule_helps([0.0073] * 5) == "null"  # the boundary itself is not past it
    assert rule_helps([-0.01, -0.01, -0.01, -0.01, 0.001]) == "hurts"
    assert rule_helps([-0.04, -0.04, -0.04, 0.01, 0.01]) == "null"
    assert rule_helps([-0.0073] * 5) == "null"
    assert rule_helps([0.0, 0.02, 0.02, 0.02, 0.02]) == "helps"  # a zero fold is not positive...
    assert rule_helps([0.0, 0.0, 0.03, 0.03, 0.03]) == "null"  # ...so two of them leave three


def test_helps_rule_reproduces_section_163s_verdict():
    """§163's per-fold ΔAP, as published: mean +0.0076, five of five positive -> helps."""
    assert rule_helps([0.0068, 0.0018, 0.0115, 0.0145, 0.0033]) == "helps"
    # the same folds with fold 1 lowered by 0.0020 put the mean at +0.0072, under the bar
    assert rule_helps([0.0068, -0.0002, 0.0115, 0.0145, 0.0033]) == "null"


def test_fold_diffs_refuses_mismatched_rows(tmp_path):
    rng = np.random.default_rng(0)
    ys = [(rng.random(50) < 0.3).astype(int) for _ in range(5)]
    for y in ys:
        y[0], y[1] = 1, 0

    def write(name, flip_fold=None):
        d = tmp_path / name
        d.mkdir()
        for k, y in enumerate(ys):
            y = 1 - y if k == flip_fold else y
            np.savez(d / f"predictions_fold{k}.npz", y_true=y, pred_fintfm=rng.random(50))
        return d

    control, same, different = write("c"), write("a"), write("b", flip_fold=2)
    assert len(fold_diffs(control, same)) == 5  # identical rows compare
    try:
        fold_diffs(control, different)
    except SystemExit as e:
        assert "fold 2" in str(e) and "rows differ" in str(e)
    else:
        raise AssertionError("mismatched rows were compared")


def test_fold_diffs_refuses_a_different_protocol(tmp_path):
    """Matching labels are not enough: a different row cap or horizon is a different run (AU2.6)."""
    import json

    import pytest

    rng = np.random.default_rng(0)
    ys = [(rng.random(50) < 0.3).astype(int) for _ in range(5)]
    for y in ys:
        y[0], y[1] = 1, 0

    def write(name, **config):
        d = tmp_path / name
        d.mkdir()
        for k, y in enumerate(ys):
            np.savez(d / f"predictions_fold{k}.npz", y_true=y, pred_fintfm=rng.random(50))
        base = {"horizon": 0, "max_rows": None, "n_features": 130, "fold_seed": 42, "n_splits": 5}
        rec = {"dataset": "v4finbench", "config": {**base, **config, "finetune_steps": len(name)}}
        (d / "v4_protocol.json").write_text(json.dumps(rec))
        return d

    control, arm = write("c"), write("arm")  # differ only in an inference setting: fine
    assert len(fold_diffs(control, arm)) == 5
    capped = write("capped", max_rows=200_000)
    with pytest.raises(SystemExit, match="protocol settings differ"):
        fold_diffs(control, capped)


def test_bootstrap_p_is_never_exactly_zero():
    from fintfm.evaluation.metrics import paired_auc_difference

    rng = np.random.default_rng(0)
    y = (rng.random(400) < 0.3).astype(int)
    perfect = y + rng.random(400) * 0.1
    noise = rng.random(400)
    _, _, p = paired_auc_difference(y, perfect, noise, n_boot=200)
    assert 0 < p <= 1 / 100
