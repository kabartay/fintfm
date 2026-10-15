"""Audit Phase 5 pretraining levers (AU5.4-AU5.6) and the task dump (AU5.15 prerequisite).

Two kinds of test per lever. **Off is invisible**: with every lever at its default, the random
stream and the batches are what they were before the levers existed, because a knob that
moves the default path silently changes what "the reference recipe" means for every later
comparison. **On does what it claims**, on a tiny case. Neither says anything about accuracy.
"""

from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np
import pytest
import torch

from fintfm.inference.preprocess import FeatureConditioner
from fintfm.modeling.hazard import CENSORED
from fintfm.prior import mixture
from fintfm.prior.base import (
    Task,
    TaskBatch,
    condition_task_features,
    permute_task_classes,
    truncate_query,
)
from fintfm.prior.dump import TaskDump, open_task_dump, write_task_dump
from fintfm.prior.mixture import PriorConfig, sample_batch

#: Every synthetic classification source plus regression, so the default-path fingerprint
#: passes through every branch the levers sit behind.
_MIX = PriorConfig(
    max_features=12, max_classes=5, n_rows_choices=(40, 64), p_tree=0.15, p_regression=0.1,
    p_trivial=0.1, p_crossed=0.1, p_task_family=0.15, p_financial=0.5,
)


def _assert_batches_equal(a: TaskBatch, b: TaskBatch) -> None:
    """Fail unless two batches are identical, treating NaN as equal to NaN."""
    assert a.n_ctx == b.n_ctx
    np.testing.assert_array_equal(a.X.numpy(), b.X.numpy())  # NaN-aware
    assert a.X.shape == b.X.shape and a.X.dtype == b.X.dtype
    assert torch.equal(a.y, b.y)
    assert torch.equal(a.n_classes, b.n_classes)
    assert (a.period is None) == (b.period is None)
    if a.period is not None:
        assert torch.equal(a.period, b.period)


# --- off is invisible ----------------------------------------------------------------------


def test_default_stream_is_pinned_and_explicitly_off_levers_change_nothing():
    # Recorded from the code at 8b30a75, before any lever existed: the split points, class
    # counts and the generator's position after four batches. The position pins how many
    # draws were consumed, which is what byte-identity of every later batch rests on. (Run
    # locally, a SHA-256 over X, y, n_classes and n_ctx of these four batches was also
    # unchanged: ac1021f4...42f17ce.)
    rng = np.random.default_rng(2026)
    got = [sample_batch(rng, _MIX, 4) for _ in range(4)]
    assert [(b.n_ctx, b.X.shape[1], b.n_classes.tolist()) for b in got] == [
        (52, 64, [2, 2, 3, 5]), (26, 40, [2, 2, 2, 2]), (33, 40, [3, 2, 2, 3]),
        (31, 64, [2, 5, 5, 2]),
    ]
    assert rng.random() == 0.21560154636155637

    for off in (
        replace(_MIX, permute_classes=False, feature_transforms=(), max_query_rows=None),
        replace(_MIX, feature_transforms=("none",)),
    ):
        rng = np.random.default_rng(2026)
        for want in got:
            _assert_batches_equal(sample_batch(rng, off, 4), want)
        assert rng.random() == 0.21560154636155637


# --- AU5.5: class permutation --------------------------------------------------------------


def test_permutation_is_one_bijection_per_task_shared_by_context_and_query():
    y = np.array([0, 1, 2, 2, 1, 0, 2, 1, 0, 2], dtype=np.int64)
    task = Task(X=np.zeros((10, 2), np.float32), y=y, n_classes=4,
                is_categorical=np.zeros(2, bool))
    out = permute_task_classes(np.random.default_rng(1), task)
    assert out.n_classes == 4 and out.y.dtype == y.dtype
    assert set(out.y.tolist()) <= set(range(4))  # n_classes masking still valid
    mapping = {}
    for old, new in zip(y.tolist(), out.y.tolist()):
        assert mapping.setdefault(old, new) == new  # same map everywhere, context and query
    assert len(set(mapping.values())) == len(mapping)  # a bijection
    for old, new in mapping.items():
        assert (out.y == new).sum() == (y == old).sum()  # class balance moves with the name


def test_permutation_leaves_regression_alone_and_refuses_survival():
    reg = Task(X=np.zeros((6, 1), np.float32), y=np.zeros(6, np.int64), n_classes=3,
               is_categorical=np.zeros(1, bool), y_continuous=np.arange(6.0))
    rng = np.random.default_rng(0)
    assert permute_task_classes(rng, reg) is reg
    assert rng.random() == np.random.default_rng(0).random()  # and consumed no draw
    period = np.array([0, CENSORED, 1, CENSORED])
    surv = Task(X=np.zeros((4, 1), np.float32), y=(period != CENSORED).astype(np.int64),
                n_classes=2, is_categorical=np.zeros(1, bool), period=period, n_horizons=2)
    with pytest.raises(ValueError, match="survival"):
        permute_task_classes(rng, surv)


def test_sample_batch_permutes_classes_per_task():
    # The financial prior always codes the rare default as 1, so without the lever class 1 is
    # the minority in every task; with it, in about half of them. Features and the split are
    # untouched, and each task's labels are a relabelling of the unpermuted ones.
    cfg = PriorConfig(max_features=10, max_classes=2, n_rows=64, p_financial=1.0)
    rare_is_one = {}
    for on in (False, True):
        rng = np.random.default_rng(4)
        flags = []
        for _ in range(10):
            b = sample_batch(rng, replace(cfg, permute_classes=on), 8)
            for i in range(8):
                counts = np.bincount(b.y[i].numpy(), minlength=2)
                if counts[0] != counts[1]:
                    flags.append(bool(counts[1] < counts[0]))
        rare_is_one[on] = np.mean(flags)
    assert rare_is_one[False] == 1.0
    assert 0.3 < rare_is_one[True] < 0.7

    # the same draws, relabelled: rerun one batch with a generator that skips nothing else
    a = sample_batch(np.random.default_rng(9), cfg, 8)
    b = sample_batch(np.random.default_rng(9), replace(cfg, permute_classes=True), 8)
    np.testing.assert_array_equal(a.X.numpy(), b.X.numpy())
    assert a.n_ctx == b.n_ctx and torch.equal(a.n_classes, b.n_classes)
    for i in range(8):
        pairs = set(zip(a.y[i].tolist(), b.y[i].tolist()))
        assert len(pairs) == len({p[0] for p in pairs}) == len({p[1] for p in pairs})
    assert not torch.equal(a.y, b.y)


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"permute_classes": True, "n_horizons": 3, "p_financial": 1.0}, "n_horizons"),
        ({"permute_classes": True, "head_type": "quantile", "p_regression": 1.0}, "inert"),
        ({"feature_transforms": ("rank", "quantile")}, "unknown feature_transforms"),
        ({"max_query_rows": 0}, "max_query_rows"),
    ],
)
def test_malformed_or_inert_lever_settings_are_refused(change, match):
    with pytest.raises(ValueError, match=match):
        sample_batch(np.random.default_rng(0), replace(_MIX, **change), 2)


# --- AU5.6: pretraining preprocessing ------------------------------------------------------


@pytest.mark.parametrize("kind", ["rank", "power", "winsor"])
def test_feature_transform_is_fitted_on_context_rows_only(kind):
    rng = np.random.default_rng(0)
    X = rng.lognormal(size=(60, 4)).astype(np.float32)
    X[rng.random(X.shape) < 0.1] = np.nan
    task = Task(X=X, y=np.zeros(60, np.int64), n_classes=2, is_categorical=np.zeros(4, bool))
    n_ctx = 40
    out = condition_task_features(task, n_ctx, kind)
    expected = FeatureConditioner(kind).fit(X[:n_ctx]).transform(X)  # type: ignore[arg-type]
    np.testing.assert_array_equal(out.X, expected)
    np.testing.assert_array_equal(np.isnan(out.X), np.isnan(X))  # missingness preserved
    assert not np.allclose(np.nan_to_num(out.X), np.nan_to_num(X))  # it did something

    # wreck the query rows: no context value may move, so nothing leaked from the query
    X2 = X.copy()
    X2[n_ctx:] = X2[n_ctx:] * 1e4 - 7.0
    out2 = condition_task_features(replace(task, X=X2), n_ctx, kind)
    np.testing.assert_array_equal(out2.X[:n_ctx], out.X[:n_ctx])


@pytest.mark.parametrize("kind", ["rank", "power", "winsor"])
def test_a_column_with_no_finite_context_value_is_left_untransformed(kind):
    # FeatureConditioner("power") raises on such a column; a small context over a sparse
    # column produces one, and a pretraining step must not die on it.
    rng = np.random.default_rng(1)
    X = rng.normal(size=(20, 3)).astype(np.float32)
    X[:8, 1] = np.nan
    task = Task(X=X, y=np.zeros(20, np.int64), n_classes=2, is_categorical=np.zeros(3, bool))
    out = condition_task_features(task, 8, kind)
    np.testing.assert_array_equal(out.X[:, 1], X[:, 1])
    keep = [0, 2]
    expected = FeatureConditioner(kind).fit(X[:8, keep]).transform(X[:, keep])  # type: ignore[arg-type]
    np.testing.assert_array_equal(out.X[:, keep], expected)


def test_feature_transform_none_is_the_identity():
    task = Task(X=np.ones((5, 2), np.float32), y=np.zeros(5, np.int64), n_classes=2,
                is_categorical=np.zeros(2, bool))
    assert condition_task_features(task, 3, "none") is task


def test_sample_batch_draws_every_listed_transform_on_the_batch_context(monkeypatch):
    seen: list[tuple[str, int]] = []
    real = mixture.condition_task_features

    def spy(task: Task, n_ctx: int, kind: str) -> Task:
        seen.append((kind, n_ctx))
        return real(task, n_ctx, kind)

    monkeypatch.setattr(mixture, "condition_task_features", spy)
    cfg = PriorConfig(max_features=6, max_classes=3, n_rows=32,
                      feature_transforms=("none", "rank", "power", "winsor"))
    rng = np.random.default_rng(1)
    for _ in range(8):
        b = sample_batch(rng, cfg, 4)
        assert {n for _, n in seen[-4:]} == {b.n_ctx}  # fitted at the batch's own split
    counts = {k: sum(kind == k for kind, _ in seen) for k in cfg.feature_transforms}
    assert all(c > 0 for c in counts.values()), counts


# --- AU5.4: long contexts without long queries ---------------------------------------------


def test_max_query_rows_truncates_only_the_query_and_draws_nothing():
    cfg = PriorConfig(max_features=6, max_classes=3, n_rows_choices=(48, 64), p_financial=0.5)
    for seed in range(3):
        full = sample_batch(np.random.default_rng(seed), cfg, 3)
        rng = np.random.default_rng(seed)
        cut = sample_batch(rng, replace(cfg, max_query_rows=5), 3)
        assert cut.n_ctx == full.n_ctx
        assert cut.X.shape[1] == full.n_ctx + min(5, full.X.shape[1] - full.n_ctx)
        np.testing.assert_array_equal(cut.X.numpy(), full.X.numpy()[:, : cut.X.shape[1]])
        assert torch.equal(cut.y, full.y[:, : cut.X.shape[1]])
        # same generator position as the untruncated draw
        ref = np.random.default_rng(seed)
        sample_batch(ref, cfg, 3)
        assert rng.random() == ref.random()


def test_truncate_query_keeps_periods_and_continuous_targets_aligned():
    period = np.arange(10) % 3 - 1
    task = Task(X=np.arange(20, dtype=np.float32).reshape(10, 2), y=np.zeros(10, np.int64),
                n_classes=2, is_categorical=np.zeros(2, bool), period=period,
                y_continuous=np.arange(10.0))
    out = truncate_query(task, n_ctx=4, max_query_rows=3)
    assert out.n_rows == 7
    np.testing.assert_array_equal(out.period, period[:7])
    np.testing.assert_array_equal(out.y_continuous, np.arange(7.0))
    assert truncate_query(task, n_ctx=4, max_query_rows=6) is task


# --- AU5.15 prerequisite: the task dump ----------------------------------------------------


_DUMP_CFG = PriorConfig(
    max_features=10, max_classes=4, n_rows_choices=(24, 40), p_financial=0.4, p_tree=0.2,
    p_regression=0.1, permute_classes=True, feature_transforms=("none", "rank"),
)


def test_sequential_dump_replays_sample_batch_exactly(tmp_path):
    write_task_dump(tmp_path / "d", _DUMP_CFG, seed=5, batch_size=3, n_batches=5, shard_size=2)
    dump = TaskDump.open(tmp_path / "d")
    assert len(dump) == 5 and len(dump.manifest["shards"]) == 3
    rng = np.random.default_rng(5)
    for i in range(5):
        _assert_batches_equal(dump.batch(i), sample_batch(rng, _DUMP_CFG, 3))
    _assert_batches_equal(dump.batch(1), dump.batch(1))  # re-read after a shard switch
    with pytest.raises(IndexError):
        dump.batch(5)


def test_per_shard_dump_draws_each_shard_from_its_own_generator(tmp_path):
    write_task_dump(tmp_path / "d", _DUMP_CFG, seed=2, batch_size=2, n_batches=3, shard_size=2,
                    stream="per_shard")
    dump = TaskDump.open(tmp_path / "d")
    for shard, indices in ((0, (0, 1)), (1, (2,))):
        rng = np.random.default_rng([2, shard])
        for i in indices:
            _assert_batches_equal(dump.batch(i), sample_batch(rng, _DUMP_CFG, 2))


def test_survival_dump_carries_periods(tmp_path):
    cfg = PriorConfig(max_features=8, max_classes=2, n_rows=24, p_financial=1.0, n_horizons=3)
    write_task_dump(tmp_path / "d", cfg, seed=0, batch_size=2, n_batches=2)
    rng = np.random.default_rng(0)
    for i in range(2):
        got = TaskDump.open(tmp_path / "d").batch(i)
        assert got.period is not None
        _assert_batches_equal(got, sample_batch(rng, cfg, 2))


def test_dump_refuses_a_different_prior_batch_size_or_length(tmp_path):
    write_task_dump(tmp_path / "d", _DUMP_CFG, seed=0, batch_size=2, n_batches=3)
    open_task_dump(tmp_path / "d", _DUMP_CFG, batch_size=2, steps=3)
    with pytest.raises(ValueError, match="permute_classes"):
        open_task_dump(tmp_path / "d", replace(_DUMP_CFG, permute_classes=False), 2, 3)
    with pytest.raises(ValueError, match="batch-size"):
        open_task_dump(tmp_path / "d", _DUMP_CFG, batch_size=4, steps=3)
    with pytest.raises(ValueError, match="fewer than --steps"):
        open_task_dump(tmp_path / "d", _DUMP_CFG, batch_size=2, steps=4)
    with pytest.raises(FileExistsError):
        write_task_dump(tmp_path / "d", _DUMP_CFG, seed=0, batch_size=2, n_batches=3)
    with pytest.raises(ValueError, match="workers"):
        write_task_dump(tmp_path / "e", _DUMP_CFG, seed=0, batch_size=2, n_batches=3, workers=2)


def test_training_from_a_sequential_dump_matches_live_sampling(tmp_path):
    # The training loop's generator feeds only sample_batch (audit AU1.18), so a sequential
    # dump at the run's seed must give the same weights as sampling live.
    from fintfm.modeling.model import ModelConfig
    from fintfm.modeling.train import TrainConfig, train

    model_cfg = ModelConfig(max_features=8, max_classes=2, d_cell=8, d_model=16, n_layers=1,
                            n_col_layers=1, n_heads=2)
    prior_cfg = PriorConfig(max_features=8, max_classes=2, n_rows=32)
    common = {"steps": 4, "batch_size": 2, "eval_every": 1000, "log_every": 1000,
              "device": "cpu", "seed": 3}
    write_task_dump(tmp_path / "d", prior_cfg, seed=3, batch_size=2, n_batches=4)
    live = train(model_cfg, prior_cfg, TrainConfig(**common), str(tmp_path / "live.pt"))
    dumped = train(model_cfg, prior_cfg, TrainConfig(**common, task_dump=str(tmp_path / "d")),
                   str(tmp_path / "dumped.pt"))
    for (name, a), (_, b) in zip(live.state_dict().items(), dumped.state_dict().items()):
        assert torch.equal(a, b), f"{name} differs between live and dumped batches"


def test_lever_flags_resolve_over_the_config_section():
    from fintfm.config import PriorConfigDefaults
    from fintfm.prior.levers import add_prior_lever_arguments, prior_lever_values

    p = argparse.ArgumentParser()
    add_prior_lever_arguments(p)
    section = PriorConfigDefaults(permute_classes=True, feature_transforms=("rank",),
                                  max_query_rows=7)
    assert prior_lever_values(p.parse_args([]), section) == {
        "permute_classes": True, "feature_transforms": ("rank",), "max_query_rows": 7,
    }
    args = p.parse_args(["--no-permute-classes", "--feature-transforms", "none, power",
                         "--max-query-rows", "3"])
    assert prior_lever_values(args, section) == {
        "permute_classes": False, "feature_transforms": ("none", "power"), "max_query_rows": 3,
    }
