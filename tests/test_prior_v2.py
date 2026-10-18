"""Prior v2 pieces of design review DR2.2 (gated labels) and DR2.3 (knob distributions).

Same two kinds of test as ``tests/test_prior_levers.py``. **Off is invisible**: with every new
field at its default the random stream and the batches are the ones recorded before these
fields existed. **On does what it claims**, on small cases. Neither says anything about what a
model trained on them learns; ``scripts/prior_stats.py --checks`` measures the distributions
at 300 tasks per source.
"""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import replace

import numpy as np
import pytest
import torch

from fintfm.config import ConfigError, load_config
from fintfm.inference.preprocess import FeatureConditioner
from fintfm.modeling.model import normalize_features
from fintfm.prior.financial import (
    _GATE_KINDS,
    _accounts,
    _is_ratio_column,
    sample_financial_task,
)
from fintfm.prior.gated import sample_gated_task
from fintfm.prior.knobs import (
    MIN_ROWS_PER_CLASS,
    TaskKnobs,
    _missing_mask,
    apply_task_knobs,
    cut_classes,
    knobs_from,
    parse_feature_ratio_bands,
)
from fintfm.prior.mixture import PriorConfig, sample_batch, sample_task

#: ``tests/test_prior_levers.py``'s mixture: every synthetic source plus regression.
_MIX = PriorConfig(
    max_features=12, max_classes=5, n_rows_choices=(40, 64), p_tree=0.15, p_regression=0.1,
    p_trivial=0.1, p_crossed=0.1, p_task_family=0.15, p_financial=0.5,
)

#: Every DR2.2/DR2.3 field spelled out at its default.
_OFF = {
    "financial_gate_prob": 0.0, "financial_sign_flip_prob": 0.0, "feature_ratio_bands": (),
    "p_binary": None,
    "imbalance_draws": False, "categorical_mode": "ordered", "missing_mode": "mcar",
    "tie_column_prob": 0.0, "irrelevant_frac_max": 0.0, "duplicate_frac_max": 0.0,
    "label_flip_max": 0.0,
}

#: Every knob on, at the values ``scripts/prior_stats.py`` checks.
_ON = {
    "feature_ratio_bands": ((0.5, 0.0, 0.05), (0.35, 0.05, 0.2), (0.15, 0.2, 0.5)),
    "p_binary": 0.7, "imbalance_draws": True, "categorical_mode": "unordered",
    "missing_mode": "mixed", "tie_column_prob": 0.15, "irrelevant_frac_max": 0.2,
    "duplicate_frac_max": 0.1, "label_flip_max": 0.05,
}


# --- off is invisible ----------------------------------------------------------------------


def test_every_new_field_at_its_default_keeps_the_recorded_stream():
    # The split points, class counts and generator position recorded at 8b30a75, before any
    # lever existed (tests/test_prior_levers.py); unchanged with each DR2.2/DR2.3 field stated
    # at its default, which is the guarantee a knob may not move the reference recipe.
    assert knobs_from(PriorConfig()) is None
    for cfg in (_MIX, replace(_MIX, **_OFF)):
        rng = np.random.default_rng(2026)
        got = [sample_batch(rng, cfg, 4) for _ in range(4)]
        assert [(b.n_ctx, b.X.shape[1], b.n_classes.tolist()) for b in got] == [
            (52, 64, [2, 2, 3, 5]), (26, 40, [2, 2, 2, 2]), (33, 40, [3, 2, 2, 3]),
            (31, 64, [2, 5, 5, 2]),
        ]
        assert rng.random() == 0.21560154636155637


def test_cut_classes_without_label_knobs_is_the_inline_code_it_replaced():
    # The SCM and tree priors carried this cut inline; with neither p_binary nor
    # imbalance_draws set it must make the same draws and labels whether knobs is None or a
    # TaskKnobs with only column-side knobs on.
    t = np.random.default_rng(0).normal(size=200)
    for seed in range(20):
        a = cut_classes(np.random.default_rng(seed), t, 6, None)
        b_rng = np.random.default_rng(seed)
        b = cut_classes(b_rng, t, 6, TaskKnobs(tie_column_prob=0.5))
        assert a[1] == b[1] and np.array_equal(a[0], b[0])


# --- DR2.2: the gated source ---------------------------------------------------------------


def _inside(X: np.ndarray, cols: list[int], cuts: list[float], directions: list[int]) -> np.ndarray:
    """Recompute a gated task's gate from its recorded columns, cuts and directions."""
    inside = np.ones(X.shape[0], dtype=bool)
    for c, cut, d in zip(cols, cuts, directions):
        inside &= d * X[:, c] > d * cut
    return inside


def test_gated_task_puts_its_positives_behind_the_gate():
    rng = np.random.default_rng(3)
    inside_pos = total_pos = 0
    sides = []
    first_cols = set()
    for _ in range(60):
        t = sample_gated_task(rng, 400, max_features=10)
        g = t.meta["gate"]
        ok = np.isfinite(t.X[:, g["columns"]]).all(axis=1)
        inside = _inside(np.nan_to_num(t.X), g["columns"], g["cuts"], g["directions"])
        pos = (t.y == 1) & ok
        inside_pos += int((pos & inside).sum())
        total_pos += int(pos.sum())
        assert t.n_classes == 2 and t.y.sum() >= 2 and t.source == "gated"
        assert abs(inside[ok].mean() - g["share"]) < 0.08
        sides += g["directions"]
        first_cols.add(g["columns"][0])
    # the drawn leak share is LogU(0.002, 0.05), so about 2% of positives sit outside
    assert inside_pos / total_pos > 0.93
    assert 0.35 < np.mean(np.asarray(sides) > 0) < 0.65  # the gate's side is a coin
    assert len(first_cols) > 5  # and its column is not fixed


def test_gated_source_is_reachable_through_source_weights_and_refused_with_horizons():
    cfg = PriorConfig(max_features=8, source_weights={"gated": 1.0})
    t = sample_task(np.random.default_rng(0), cfg, n_rows=64)
    assert t.source == "gated"
    with pytest.raises(ValueError, match="survival label"):
        sample_batch(np.random.default_rng(0), replace(cfg, n_horizons=4), 2)


# --- DR2.2: the gated financial label ------------------------------------------------------


def test_insolvent_share_builds_negative_equity_and_keeps_the_identities():
    rng = np.random.default_rng(1)
    macro = {"rate": 0.05, "cycle": 0.0}
    sector = rng.integers(0, 3, size=500)
    par = {"margin": np.full(3, 0.1), "leverage": np.full(3, 0.4), "cyclicality": np.ones(3)}
    plain = _accounts(np.random.default_rng(2), 500, macro, sector, par)
    acc = _accounts(np.random.default_rng(2), 500, macro, sector, par, insolvent_share=0.1)
    assert (plain["equity"] > 0).all()  # leverage is capped at 0.98 otherwise
    assert np.isclose((acc["equity"] < 0).mean(), 0.1)
    np.testing.assert_allclose(acc["total_assets"], acc["total_liabilities"] + acc["equity"])
    np.testing.assert_allclose(
        acc["total_liabilities"], acc["short_term_liabilities"] + acc["long_term_debt"]
    )


def test_gated_financial_label_sits_behind_an_exposed_gate():
    rng = np.random.default_rng(5)
    kinds = set()
    inside_pos = total_pos = 0
    for _ in range(80):
        t = sample_financial_task(rng, 512, max_features=40, gate_prob=1.0)
        g = t.meta["gate"]
        kinds.add(g["kind"])
        col = t.meta["columns"].index(f"gate:{g['ratio']}")
        x = t.X[:, col].astype(np.float64)
        ok = np.isfinite(x)
        inside = x < g["cut"] if g["side"] == "low" else x > g["cut"]
        pos = (t.y == 1) & ok
        inside_pos += int((pos & inside).sum())
        total_pos += int(pos.sum())
        assert abs(inside[ok].mean() - g["share"]) < 0.05
        assert len(t.meta["columns"]) == t.n_features
    assert kinds == set(_GATE_KINDS)
    assert inside_pos / total_pos > 0.93


def test_financial_gate_off_attaches_nothing_and_horizons_are_refused():
    t = sample_financial_task(np.random.default_rng(0), 128)
    assert t.meta is None
    with pytest.raises(ValueError, match="n_horizons"):
        sample_financial_task(np.random.default_rng(0), 128, n_horizons=4, gate_prob=0.5)
    cfg = PriorConfig(p_financial=1.0, n_horizons=4, financial_gate_prob=0.5)
    with pytest.raises(ValueError, match="n_horizons"):
        sample_batch(np.random.default_rng(0), cfg, 2)
    with pytest.raises(ValueError, match="financial_gate_prob"):
        sample_batch(np.random.default_rng(0), PriorConfig(financial_gate_prob=1.5), 2)


def test_sign_flip_off_keeps_the_recorded_gated_financial_stream():
    # SHA-256 over X, y, n_classes and n_ctx of four gated financial batches, and the
    # generator's position after them, recorded at e1bc433 before the flip existed. The
    # default mixture's stream is pinned by the first test in this file.
    cfg = PriorConfig(max_features=40, max_classes=5, n_rows_choices=(128, 256),
                      source_weights={"financial": 1.0}, financial_gate_prob=1.0)
    for c in (cfg, replace(cfg, financial_sign_flip_prob=0.0)):
        rng = np.random.default_rng(2026)
        h = hashlib.sha256()
        for _ in range(4):
            b = sample_batch(rng, c, 4)
            h.update(np.ascontiguousarray(b.X.numpy()).tobytes())
            h.update(b.y.numpy().tobytes())
            h.update(b.n_classes.numpy().tobytes())
            h.update(str(b.n_ctx).encode())
        assert h.hexdigest() == "388ee2ea6327ba36e65e7b1af707f5bc9ef51beafcbaa40db7c0ef4c36337899"
        assert rng.random() == 0.6906481870401622


@pytest.mark.parametrize("gate_prob", [0.0, 1.0])
def test_sign_flip_negates_ratio_columns_and_leaves_the_rest_of_the_task(gate_prob):
    # The flip is drawn last, so at a fixed seed the flipped task is the unflipped one with
    # some ratio columns negated: same label, gate, column names and NaN pattern.
    flipped = ratios = 0
    for seed in range(30):
        base = sample_financial_task(np.random.default_rng(seed), 256, max_features=40,
                                     gate_prob=gate_prob)
        for p in (0.5, 1.0):
            t = sample_financial_task(np.random.default_rng(seed), 256, max_features=40,
                                      gate_prob=gate_prob, sign_flip_prob=p)
            assert np.array_equal(t.y, base.y)
            if base.meta is not None:  # with the gate off the unflipped task records nothing
                assert t.meta["columns"] == base.meta["columns"]
            assert t.meta["gate"] == (base.meta or {}).get("gate")
            np.testing.assert_array_equal(np.isnan(t.X), np.isnan(base.X))
            is_ratio = np.array([_is_ratio_column(c) for c in t.meta["columns"]])
            flip = np.array(t.meta["flipped"])
            assert not (flip & ~is_ratio).any()  # levels, sector and noise never flip
            if p == 1.0:
                assert np.array_equal(flip, is_ratio)
            else:
                flipped += int(flip.sum())
                ratios += int(is_ratio.sum())
            want = np.where(flip[None, :], -base.X, base.X)
            np.testing.assert_array_equal(t.X, want)
    assert 0.4 < flipped / ratios < 0.6
    assert _is_ratio_column("redundant:gate:long_term_debt/equity")
    assert _is_ratio_column("leverage") and not _is_ratio_column("equity")
    assert not _is_ratio_column("sector") and not _is_ratio_column("noise")


def test_sign_flip_reaches_the_financial_source_and_is_range_checked():
    cfg = PriorConfig(max_features=30, n_rows_choices=(64,), source_weights={"financial": 1.0},
                      financial_sign_flip_prob=1.0)
    t = sample_task(np.random.default_rng(0), cfg, 64)
    assert t.meta["flipped"] == [_is_ratio_column(c) for c in t.meta["columns"]]
    with pytest.raises(ValueError, match="financial_sign_flip_prob"):
        sample_batch(np.random.default_rng(0), PriorConfig(financial_sign_flip_prob=1.5), 2)


def test_rank_conditioner_and_normalisation_treat_a_negated_column_as_its_mirror():
    # What makes the flip a pure change of direction for the model: the inference "rank"
    # conditioner fitted on a negated column returns the negated output (to float32
    # rounding in the normal tail), with the same NaNs, and the model's context z-score is
    # exactly odd. ("power" is not: Yeo-Johnson fits its lambda per sign, measured off by
    # up to 1.0 on a few cells.)
    for seed in range(6):
        t = sample_financial_task(np.random.default_rng(seed), 512, max_features=40,
                                  gate_prob=1.0)
        n_ctx = 358
        ratio = np.array([_is_ratio_column(c) for c in t.meta["columns"]])
        X = t.X[:, ratio]
        a = FeatureConditioner("rank").fit(X[:n_ctx]).transform(X)
        b = FeatureConditioner("rank").fit(-X[:n_ctx]).transform(-X)
        np.testing.assert_array_equal(np.isnan(a), np.isnan(b))
        np.testing.assert_allclose(a, -b, atol=1e-3)
        Xt = torch.from_numpy(t.X)[None]
        za, ma, pa = normalize_features(Xt, n_ctx)
        zb, mb, pb = normalize_features(-Xt, n_ctx)
        assert torch.equal(za, -zb) and torch.equal(ma, mb) and torch.equal(pa, pb)


# --- DR2.3: the knobs ----------------------------------------------------------------------


def _draw(src: str, n_tasks: int, n_rows: int = 256, **knobs: object) -> list:
    """Draw ``n_tasks`` tasks of one source under ``knobs``."""
    cfg = PriorConfig(max_features=40, max_classes=6, source_weights={src: 1.0}, **knobs)
    rng = np.random.default_rng(11)
    return [sample_task(rng, cfg, n_rows=n_rows) for _ in range(n_tasks)]


def test_p_binary_sets_the_binary_share():
    for src in ("scm", "tree"):
        assert all(t.n_classes == 2 for t in _draw(src, 15, p_binary=1.0))
        assert all(t.n_classes >= 3 for t in _draw(src, 15, p_binary=0.0, imbalance_draws=True))


def test_imbalance_draws_keep_every_class_at_four_rows_and_the_minority_under_half():
    for src in ("scm", "tree"):
        for t in _draw(src, 30, n_rows=200, imbalance_draws=True):
            counts = np.bincount(t.y, minlength=t.n_classes)
            assert counts.min() >= MIN_ROWS_PER_CLASS, counts
            if t.n_classes == 2:
                assert counts.min() <= 100


def test_unordered_categoricals_replace_the_ordered_codes():
    tasks = _draw("scm", 40, n_rows=512, categorical_mode="unordered")
    cards = [
        len(np.unique(t.X[np.isfinite(t.X[:, j]), j]))
        for t in tasks for j in np.flatnonzero(t.is_categorical)
    ]
    assert cards and max(cards) <= 64 and max(cards) > 7  # the ordered codes stop at 7
    assert all("categorical_levels" in t.meta["knobs"] for t in tasks)


def test_mixed_missingness_draws_each_mechanism_and_mnar_follows_the_value():
    x = np.random.default_rng(0).normal(size=(4000, 3))
    seen = {}
    for seed in range(60):
        mask, rec = _missing_mask(np.random.default_rng(seed), x)
        seen.setdefault(rec["mechanism"], []).append((mask, rec))
    assert set(seen) == {"none", "mcar", "mar", "mnar"}
    for mask, rec in seen["mnar"]:
        for j in np.flatnonzero(mask.any(axis=0)):
            gap = abs(x[mask[:, j], j].mean() - x[~mask[:, j], j].mean())
            assert gap > 0.3  # missingness depends on the value itself
            assert abs(mask[:, j].mean() - rec["rate"]) < max(0.02, 0.5 * rec["rate"])
    for mask, _ in seen["mcar"]:
        for j in np.flatnonzero(mask.any(axis=0)):
            assert abs(x[mask[:, j], j].mean() - x[~mask[:, j], j].mean()) < 0.3


def test_ties_irrelevant_duplicates_and_flips_on_one_task():
    rng = np.random.default_rng(0)
    base = sample_gated_task(rng, 300, n_features=6)
    knobs = TaskKnobs(tie_column_prob=1.0, irrelevant_frac_max=1.0, duplicate_frac_max=1.0,
                      label_flip_max=0.2)
    t = apply_task_knobs(np.random.default_rng(1), base, knobs, max_features=20)
    rec = t.meta["knobs"]
    assert t.n_features == 6 + rec["n_irrelevant"] + rec["n_duplicate"] <= 20
    assert rec["n_irrelevant"] + rec["n_duplicate"] > 0
    assert len(rec["tie_kinds"]) == rec["numeric_columns"] == t.n_features
    for j in range(t.n_features):  # every column now repeats a value
        col = t.X[np.isfinite(t.X[:, j]), j]
        assert len(np.unique(col)) < len(col)
    assert 0 < rec["label_flip_rate"] <= 0.3 and set(np.unique(t.y)) == {0, 1}
    assert t.meta["gate"] == base.meta["gate"]  # the source's own record survives


def test_feature_ratio_bands_set_the_width():
    bands = ((1.0, 0.1, 0.2),)
    for src in ("scm", "tree", "task_family", "gated"):
        for t in _draw(src, 6, n_rows=200, feature_ratio_bands=bands):
            assert 20 - 1 <= t.n_features <= 40, (src, t.n_features)
            assert t.meta["width_band"] == 0


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"categorical_mode": "sorted"}, "categorical_mode"),
        ({"missing_mode": "mnar"}, "missing_mode"),
        ({"p_binary": 1.5}, r"\[0, 1\]"),
        ({"feature_ratio_bands": ((-1.0, 0.0, 0.1),)}, "weight"),
        ({"feature_ratio_bands": ((0.0, 0.0, 0.1),)}, "sum to zero"),
    ],
)
def test_malformed_knobs_are_refused(overrides, match):
    with pytest.raises(ValueError, match=match):
        sample_batch(np.random.default_rng(0), PriorConfig(**overrides), 2)


# --- configuration and flags ---------------------------------------------------------------


def test_config_and_flags_reach_the_prior(tmp_path):
    from fintfm.modeling.train import build_parser, build_prior_config

    f = tmp_path / "c.yaml"
    f.write_text("prior:\n  feature_ratio_bands: [[0.5, 0, 0.05], [0.5, 0.05, 0.2]]\n"
                 "  categorical_mode: unordered\n  financial_gate_prob: 0.25\n")
    cfg = load_config(f)
    prior = build_prior_config(build_parser().parse_args([]), cfg)
    assert prior.feature_ratio_bands == ((0.5, 0.0, 0.05), (0.5, 0.05, 0.2))
    assert prior.categorical_mode == "unordered" and prior.financial_gate_prob == 0.25
    args = build_parser().parse_args([
        "--feature-ratio-bands", "1:0.1:0.2", "--p-binary", "0.7", "--imbalance-draws",
        "--missing-mode", "mixed", "--tie-column-prob", "0.1", "--irrelevant-frac-max", "0.2",
        "--duplicate-frac-max", "0.1", "--label-flip-max", "0.05",
        "--financial-gate-prob", "0.5", "--financial-sign-flip-prob", "0.25",
    ])
    prior = build_prior_config(args, cfg)
    assert prior.feature_ratio_bands == ((1.0, 0.1, 0.2),)
    assert (prior.p_binary, prior.imbalance_draws, prior.missing_mode) == (0.7, True, "mixed")
    assert (prior.tie_column_prob, prior.irrelevant_frac_max, prior.duplicate_frac_max,
            prior.label_flip_max, prior.financial_gate_prob) == (0.1, 0.2, 0.1, 0.05, 0.5)
    assert prior.financial_sign_flip_prob == 0.25
    assert build_prior_config(build_parser().parse_args([]), cfg).financial_sign_flip_prob == 0.0
    assert parse_feature_ratio_bands("") == ()
    with pytest.raises(ValueError, match="weight:lo:hi"):
        parse_feature_ratio_bands("1:2")
    assert isinstance(build_parser(), argparse.ArgumentParser)


@pytest.mark.parametrize(
    "text, match",
    [
        ("prior:\n  categorical_mode: sorted\n", "categorical_mode"),
        ("prior:\n  missing_mode: mnar\n", "missing_mode"),
        ("prior:\n  feature_ratio_bands: [[1, 0]]\n", "feature_ratio_bands"),
        ("prior:\n  financial_gate_prob: 2\n", "financial_gate_prob"),
        ("prior:\n  financial_sign_flip_prob: -0.1\n", "financial_sign_flip_prob"),
    ],
)
def test_config_refuses_malformed_knobs(tmp_path, text, match):
    f = tmp_path / "c.yaml"
    f.write_text(text)
    with pytest.raises(ConfigError, match=match):
        load_config(f)


def test_every_knob_on_still_collates_a_batch():
    cfg = PriorConfig(
        max_features=30, max_classes=5, n_rows_choices=(64, 128), scm_reuse_graph=3,
        source_weights={"scm": 0.3, "tree": 0.2, "task_family": 0.1, "gated": 0.1,
                        "financial": 0.2, "regression": 0.1},
        financial_gate_prob=0.5, **_ON,
    )
    rng = np.random.default_rng(0)
    for _ in range(3):
        b = sample_batch(rng, cfg, 8)
        assert b.X.shape[0] == 8 and b.X.shape[2] == 30
        assert (b.n_classes >= 2).all()
