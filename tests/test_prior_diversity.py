"""AU5.11's three prior-diversity flags: heavy-tailed columns, uniform tree splits and
label-informative missingness (``heavy_tail_column_prob``, ``tree_split_by``,
``missing_label``).

Same two kinds of test as ``tests/test_prior_v2.py``. **Off is invisible**: with the flags at
their defaults a drawn stream is byte-identical to the one drawn by ``63500ad``, before the
flags existed. **On does what it claims**, on small cases; what a model trained on them learns
is not tested here.
"""

from __future__ import annotations

import argparse
import hashlib
import platform
import sys
from dataclasses import replace
from unittest import mock

import numpy as np
import pytest
from scipy import stats

from fintfm.config import ConfigError, load_config
from fintfm.prior.knobs import (
    TaskKnobs,
    _heavy_tail_column,
    _missing_mask,
    check_knobs,
    knobs_from,
)
from fintfm.prior.mixture import PriorConfig, sample_task
from fintfm.prior.tree import _oblivious_forest

_BANDS = ((0.5, 0.0, 0.05), (0.35, 0.05, 0.2), (0.15, 0.2, 0.5))
_BASE = {"max_features": 12, "max_classes": 5, "n_rows_choices": (40, 64), "p_tree": 0.3,
         "p_regression": 0.1, "p_task_family": 0.2, "p_financial": 0.2}
#: The DR2.3 knobs at the values `scripts/prior_stats.py` checks.
_V2 = {"feature_ratio_bands": _BANDS, "p_binary": 0.7, "imbalance_draws": True,
       "categorical_mode": "unordered", "missing_mode": "mixed", "tie_column_prob": 0.15,
       "irrelevant_frac_max": 0.2, "duplicate_frac_max": 0.1, "label_flip_max": 0.05}
#: The three AU5.11 fields spelled out at their defaults.
_OFF = {"heavy_tail_column_prob": 0.0, "tree_split_by": "std", "missing_label": False}


def _digest(cfg: PriorConfig) -> str:
    """Hash 40 tasks drawn at a fixed seed, plus the generator's next draw."""
    rng = np.random.default_rng(2026)
    h = hashlib.sha256()
    for _ in range(40):
        t = sample_task(rng, cfg)
        h.update(np.ascontiguousarray(t.X).tobytes())
        h.update(np.asarray(t.y).tobytes())
        h.update(np.asarray(t.is_categorical).tobytes())
        h.update(t.source.encode())
    h.update(str(rng.random()).encode())
    return h.hexdigest()


# --- off is invisible ----------------------------------------------------------------------


@pytest.mark.parametrize("v2_knobs", [False, True])
def test_flags_off_leave_every_drawn_task_byte_identical(v2_knobs):
    knob_kw = _V2 if v2_knobs else {}
    cfg = PriorConfig(**_BASE, **knob_kw)
    assert _digest(replace(cfg, **_OFF)) == _digest(cfg)
    if not v2_knobs:
        assert knobs_from(cfg) is None


def test_recorded_hash_of_both_configurations():
    """The digest recorded by running these draws on a checkout of ``63500ad``, before the
    flags existed: the plain and the DR2.3-knobs mixtures, hashed together. Holds only where
    it was recorded, as in ``tests/test_prior_v2.py``."""
    knob_kw = _V2
    h = hashlib.sha256()
    for cfg in (PriorConfig(**_BASE), PriorConfig(**_BASE, **knob_kw)):
        rng = np.random.default_rng(2026)
        for _ in range(40):
            t = sample_task(rng, cfg)
            h.update(np.ascontiguousarray(t.X).tobytes())
            h.update(np.asarray(t.y).tobytes())
            h.update(np.asarray(t.is_categorical).tobytes())
            h.update(t.source.encode())
        h.update(str(rng.random()).encode())
    if sys.platform == "darwin" and platform.machine() == "arm64":
        assert h.hexdigest() == (
            "45068bcb4374504bb2018148105409a2167df461582c42f24d02550e1cee1caa"
        )


def test_off_fields_keep_the_knobs_none():
    assert knobs_from(PriorConfig()) is None
    assert knobs_from(PriorConfig(**_OFF)) is None
    assert knobs_from(PriorConfig(heavy_tail_column_prob=0.1)) is not None
    assert knobs_from(PriorConfig(tree_split_by="uniform")) is not None


# --- heavy tails ---------------------------------------------------------------------------


def test_heavy_tail_map_is_monotone_and_raises_kurtosis():
    gaussian = np.random.default_rng(0).normal(size=4000)
    gains = []
    for seed in range(20):
        out, a = _heavy_tail_column(np.random.default_rng(seed), gaussian)
        assert 0.3 <= a <= 2.0
        assert stats.spearmanr(gaussian, out)[0] == pytest.approx(1.0)
        gains.append(stats.kurtosis(out) - stats.kurtosis(gaussian))
    assert min(gains) > 0.0 and np.median(gains) > 1.0


def test_heavy_tail_map_leaves_nan_ties_and_constant_columns_alone():
    col = np.array([1.0, 1.0, 2.0, np.nan, 5.0, 5.0])
    out, _ = _heavy_tail_column(np.random.default_rng(1), col)
    assert np.isnan(out[3]) and out[0] == out[1] and out[4] == out[5]
    const, _ = _heavy_tail_column(np.random.default_rng(1), np.full(8, 3.0))
    assert np.array_equal(const, np.full(8, 3.0))


def test_heavy_tail_knob_raises_kurtosis_through_apply_task_knobs():
    cfg = PriorConfig(**{**_BASE, "p_financial": 0.0, "p_tree": 0.0, "p_regression": 0.0,
                           "p_task_family": 0.0, "n_rows_choices": (256,)})

    def median_kurtosis(heavy: float) -> float:
        vals: list[float] = []
        for seed in range(30):
            rng = np.random.default_rng(seed)
            t = sample_task(rng, replace(cfg, heavy_tail_column_prob=heavy))
            num = ~np.asarray(t.is_categorical)
            X = t.X[:, num].astype(np.float64)
            vals += [stats.kurtosis(c[np.isfinite(c)]) for c in X.T if np.isfinite(c).sum() > 3]
        return float(np.median(vals))
    assert median_kurtosis(1.0) > median_kurtosis(0.0) + 1.0


# --- tree splits ---------------------------------------------------------------------------


def _split_probabilities(split_by: str) -> np.ndarray:
    """The column-choice probabilities `_oblivious_forest` passes to ``rng.choice``."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(64, 5)) * np.array([1.0, 1.0, 1.0, 1.0, 1000.0])
    spy = mock.MagicMock(wraps=rng)
    _oblivious_forest(spy, X, 3, 4, split_by)
    return np.asarray(spy.choice.call_args.kwargs["p"])


def test_uniform_split_no_longer_tracks_column_spread():
    by_std, uniform = _split_probabilities("std"), _split_probabilities("uniform")
    assert by_std[4] > 0.99  # the 1000x-scaled column takes nearly every split
    assert np.allclose(uniform, 0.2)


def test_split_by_std_is_the_default_and_unknown_values_refuse():
    X = np.random.default_rng(3).normal(size=(50, 6))
    a = _oblivious_forest(np.random.default_rng(9), X, 4, 3)
    b = _oblivious_forest(np.random.default_rng(9), X, 4, 3, "std")
    assert np.array_equal(a, b)
    with pytest.raises(ValueError, match="split_by"):
        _oblivious_forest(np.random.default_rng(9), X, 4, 3, "nope")
    with pytest.raises(ValueError, match="tree_split_by"):
        check_knobs(TaskKnobs(tree_split_by="nope"))


# --- label-informative missingness ---------------------------------------------------------


def _class_gap(mask: np.ndarray, y: np.ndarray) -> float:
    """Largest per-column absolute difference of the blanked share between the two classes."""
    blank = mask.mean(axis=0) > 0
    if not blank.any():
        return 0.0
    gap = np.abs(mask[y == 1][:, blank].mean(axis=0) - mask[y == 0][:, blank].mean(axis=0))
    return float(gap.max())


def test_mnar_label_makes_missingness_depend_on_the_label():
    n = 600
    X = np.random.default_rng(0).normal(size=(n, 6))
    y = (np.arange(n) % 2).astype(np.int64)
    gaps: dict[str, list[float]] = {"mnar_label": [], "mcar": [], "mar": [], "mnar": []}
    for seed in range(400):
        mask, rec = _missing_mask(np.random.default_rng(seed), X, y)
        if rec["mechanism"] in gaps and rec["rate"] > 0.05:
            gaps[rec["mechanism"]].append(_class_gap(mask, y))
    assert len(gaps["mnar_label"]) >= 10 and len(gaps["mcar"]) >= 10
    assert np.median(gaps["mnar_label"]) > 0.15
    # features here are independent of y, so every feature-driven mechanism leaves it alone
    assert np.median(gaps["mcar"]) < 0.1 and np.median(gaps["mnar"]) < 0.1


def test_without_y_the_mechanism_table_and_stream_are_unchanged():
    X = np.random.default_rng(0).normal(size=(80, 5))
    mechanisms = set()
    for seed in range(200):
        a = _missing_mask(np.random.default_rng(seed), X)
        b = _missing_mask(np.random.default_rng(seed), X, None)
        assert np.array_equal(a[0], b[0]) and a[1] == b[1]
        mechanisms.add(a[1]["mechanism"])
    assert mechanisms == {"none", "mcar", "mar", "mnar"}


def test_missing_label_reaches_apply_task_knobs_and_needs_mixed_missingness():
    check_knobs(TaskKnobs(missing_mode="mixed", missing_label=True))
    with pytest.raises(ValueError, match="missing_label"):
        check_knobs(TaskKnobs(missing_label=True))
    cfg = PriorConfig(**{**_BASE, "p_financial": 0.0, "p_tree": 0.0, "p_regression": 0.0,
                           "p_task_family": 0.0, "n_rows_choices": (128,)},
                      missing_mode="mixed", missing_label=True)
    seen = set()
    for seed in range(80):
        t = sample_task(np.random.default_rng(seed), cfg)
        seen.add(t.meta["knobs"]["missing"]["mechanism"])
    assert "mnar_label" in seen


# --- configuration and flags ---------------------------------------------------------------


def test_config_and_flags_reach_the_prior(tmp_path):
    from fintfm.modeling.train import build_parser, build_prior_config

    cfg = load_config()
    prior = build_prior_config(build_parser().parse_args([]), cfg)
    assert (prior.heavy_tail_column_prob, prior.tree_split_by, prior.missing_label) == (
        0.0, "std", False
    )
    args = build_parser().parse_args([
        "--heavy-tail-column-prob", "0.2", "--tree-split-by", "uniform", "--missing-label",
        "--missing-mode", "mixed",
    ])
    prior = build_prior_config(args, cfg)
    assert (prior.heavy_tail_column_prob, prior.tree_split_by, prior.missing_label) == (
        0.2, "uniform", True
    )
    f = tmp_path / "c.yaml"
    f.write_text("prior:\n  tree_split_by: uniform\n")
    assert load_config(f).prior.tree_split_by == "uniform"
    f.write_text("prior:\n  tree_split_by: sideways\n")
    with pytest.raises(ConfigError, match="tree_split_by"):
        load_config(f)
    f.write_text("prior:\n  heavy_tail_column_prob: 1.5\n")
    with pytest.raises(ConfigError, match="heavy_tail_column_prob"):
        load_config(f)
    assert isinstance(build_parser(), argparse.ArgumentParser)
