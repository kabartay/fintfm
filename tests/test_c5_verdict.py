"""§159's pre-registered decision rules behave exactly as written, at their boundaries.

`scripts/c5_verdict.py` is the only place those rules are applied. A rule that drifted from
§159's text -- a `>=` for a `>`, a yardstick quietly changed -- would turn a pre-registered
verdict into a post-hoc one without anything looking wrong, so the boundaries are pinned here.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from c5_verdict import YARDSTICK, fold_diffs, rule1, rule2, rule3


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
