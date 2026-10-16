"""Pretraining levers (audit AU5.4-AU5.6, review DR6.1) and the task dump (AU5.15 prerequisite).

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
from fintfm.modeling.model import FinancialTFM, ModelConfig
from fintfm.prior import mixture
from fintfm.prior.base import (
    Task,
    TaskBatch,
    collate,
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
        replace(_MIX, permute_classes=False, feature_transforms=(), max_query_rows=None,
                pad_to_batch_width=False),
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
def test_a_column_empty_in_the_context_is_conditioned_as_inference_conditions_it(kind):
    # FeatureConditioner("power") used to raise on such a column; a small context over a
    # sparse column produces one, so a pretraining step must survive it and see what
    # inference would feed the model.
    rng = np.random.default_rng(1)
    X = rng.normal(size=(20, 3)).astype(np.float32)
    X[:8, 1] = np.nan
    task = Task(X=X, y=np.zeros(20, np.int64), n_classes=2, is_categorical=np.zeros(3, bool))
    out = condition_task_features(task, 8, kind)
    expected = FeatureConditioner(kind).fit(X[:8]).transform(X)  # type: ignore[arg-type]
    np.testing.assert_array_equal(out.X, expected)


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


def _assert_dumps_equal(a_dir, b_dir) -> None:
    """Fail unless two dumps hold the same manifest and the same batches, array for array."""
    import json

    a, b = TaskDump.open(a_dir), TaskDump.open(b_dir)
    assert a.manifest == b.manifest
    for i in range(len(a)):
        _assert_batches_equal(a.batch(i), b.batch(i))
    for name in a.manifest["shards"]:
        with np.load(a.path / name) as fa, np.load(b.path / name) as fb:
            assert sorted(fa.files) == sorted(fb.files)
            for key in fa.files:
                np.testing.assert_array_equal(fa[key], fb[key])
    assert json.loads((a.path / "manifest.json").read_text())["shards"] == [
        f"shard_{s:06d}.npz" for s in range(len(a.manifest["shards"]))
    ]


@pytest.mark.parametrize("compress", [False, True])
def test_per_shard_dump_is_the_same_whatever_the_worker_count(tmp_path, compress):
    """AU5.15: each shard draws from ``default_rng([seed, shard])`` whichever process writes
    it, so two workers write exactly what one does (and compression is lossless)."""
    kw = {"seed": 4, "batch_size": 2, "n_batches": 5, "shard_size": 2, "stream": "per_shard",
          "compress": compress}
    write_task_dump(tmp_path / "k1", _DUMP_CFG, workers=1, **kw)
    write_task_dump(tmp_path / "k2", _DUMP_CFG, workers=2, **kw)
    _assert_dumps_equal(tmp_path / "k1", tmp_path / "k2")
    if compress:
        write_task_dump(tmp_path / "raw", _DUMP_CFG, workers=1, **{**kw, "compress": False})
        dump, raw = TaskDump.open(tmp_path / "k1"), TaskDump.open(tmp_path / "raw")
        for i in range(len(dump)):
            _assert_batches_equal(dump.batch(i), raw.batch(i))


def test_dump_script_workers_flag_writes_the_single_process_dump(tmp_path, capsys):
    """``scripts/dump_prior_tasks.py --stream per_shard --workers 2 --compress`` gives the same
    dump as ``--workers 1``; the script's own report names the workers and the load."""
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "dump_prior_tasks.py"
    spec = importlib.util.spec_from_file_location("dump_prior_tasks", path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    common = ["--n-batches", "5", "--shard-size", "2", "--seed", "7", "--batch-size", "2",
              "--n-rows", "24", "--max-features", "8", "--max-classes", "3",
              "--stream", "per_shard", "--compress"]
    script.main(["--out", str(tmp_path / "k1"), "--workers", "1", *common])
    script.main(["--out", str(tmp_path / "k2"), "--workers", "2", *common])
    out = capsys.readouterr().out
    assert "workers 2, compress True" in out and "load average" in out
    _assert_dumps_equal(tmp_path / "k1", tmp_path / "k2")
    with pytest.raises(SystemExit):  # the sequential stream is one generator
        script.main(["--out", str(tmp_path / "s"), "--workers", "2", "--n-batches", "2",
                     "--stream", "sequential"])


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


def test_dump_written_before_a_prior_edit_is_refused_after_it(tmp_path, monkeypatch, capsys):
    """DR1.2: the manifest records a fingerprint of the prior's code, and a dump whose
    fingerprint differs from the running code's is refused -- an edit to ``prior/*.py`` can
    change every batch under an unchanged configuration. The override accepts it with a
    warning, for byte-identity checks; a manifest with no fingerprint is refused too."""
    import json

    from fintfm.prior import dump as dump_mod

    write_task_dump(tmp_path / "d", _DUMP_CFG, seed=0, batch_size=2, n_batches=3)
    manifest = json.loads((tmp_path / "d" / "manifest.json").read_text())
    assert manifest["code_fingerprint"] == dump_mod.code_fingerprint()
    open_task_dump(tmp_path / "d", _DUMP_CFG, batch_size=2, steps=3)

    # the prior is edited after the dump was written
    monkeypatch.setattr(dump_mod, "code_fingerprint", lambda package_root=None: "f" * 64)
    with pytest.raises(ValueError, match="different prior code.*allow-code-mismatch"):
        open_task_dump(tmp_path / "d", _DUMP_CFG, batch_size=2, steps=3)
    dump = open_task_dump(tmp_path / "d", _DUMP_CFG, 2, 3, allow_code_mismatch=True)
    assert "used anyway" in capsys.readouterr().out
    assert dump.record()["code_fingerprint"] == manifest["code_fingerprint"]

    del manifest["code_fingerprint"]
    (tmp_path / "d" / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.undo()
    with pytest.raises(ValueError, match="no code fingerprint"):
        open_task_dump(tmp_path / "d", _DUMP_CFG, batch_size=2, steps=3)


def test_code_fingerprint_covers_the_prior_and_preprocessing_files(tmp_path):
    """The fingerprint changes when any ``prior/*.py`` file or ``inference/preprocess.py``
    changes, and not when another module does."""
    import shutil

    from fintfm.prior import dump as dump_mod

    root = tmp_path / "fintfm"
    shutil.copytree(dump_mod._PACKAGE_ROOT / "prior", root / "prior",
                    ignore=shutil.ignore_patterns("__pycache__"))
    (root / "inference").mkdir()
    shutil.copy(dump_mod._PACKAGE_ROOT / "inference" / "preprocess.py", root / "inference")
    (root / "inference" / "classifier.py").write_text("x = 1\n")
    base = dump_mod.code_fingerprint(root)
    assert base == dump_mod.code_fingerprint()  # the copy is the running code
    (root / "inference" / "classifier.py").write_text("x = 2\n")
    assert dump_mod.code_fingerprint(root) == base
    for edited in ("prior/tree.py", "inference/preprocess.py"):
        path = root / edited
        original = path.read_text()
        path.write_text(original + "# edited\n")
        assert dump_mod.code_fingerprint(root) != base, edited
        path.write_text(original)
    assert dump_mod.code_fingerprint(root) == base


def test_training_from_a_dump_records_its_manifest_and_honours_the_override(
    tmp_path, monkeypatch
):
    """A run from a dump records the dump's manifest hash; after a prior edit the run is
    refused unless ``task_dump_allow_code_mismatch`` is set."""
    import hashlib

    from fintfm.modeling.model import ModelConfig
    from fintfm.modeling.train import TrainConfig, train
    from fintfm.prior import dump as dump_mod
    from fintfm.provenance import read_checkpoint_provenance

    model_cfg = ModelConfig(max_features=8, max_classes=2, d_cell=8, d_model=16, n_layers=1,
                            n_col_layers=1, n_heads=2)
    prior_cfg = PriorConfig(max_features=8, max_classes=2, n_rows=32)
    common = {"steps": 2, "batch_size": 2, "eval_every": 1000, "log_every": 1000, "seed": 3,
              "task_dump": str(tmp_path / "d")}
    write_task_dump(tmp_path / "d", prior_cfg, seed=3, batch_size=2, n_batches=2)
    monkeypatch.setattr(dump_mod, "code_fingerprint", lambda package_root=None: "f" * 64)
    with pytest.raises(ValueError, match="different prior code"):
        train(model_cfg, prior_cfg, TrainConfig(**common), str(tmp_path / "x.pt"))
    train(model_cfg, prior_cfg, TrainConfig(**common, task_dump_allow_code_mismatch=True),
          str(tmp_path / "m.pt"))
    rec = read_checkpoint_provenance(tmp_path / "m.pt")
    want = hashlib.sha256((tmp_path / "d" / "manifest.json").read_bytes()).hexdigest()
    assert rec["task_dump"]["manifest_sha256"] == want
    assert rec["task_dump"]["path"] == str(tmp_path / "d")
    assert rec["train_config"]["task_dump_allow_code_mismatch"] is True


def test_lever_flags_resolve_over_the_config_section():
    from fintfm.config import PriorConfigDefaults
    from fintfm.prior.levers import add_prior_lever_arguments, prior_lever_values

    p = argparse.ArgumentParser()
    add_prior_lever_arguments(p)
    section = PriorConfigDefaults(permute_classes=True, feature_transforms=("rank",),
                                  max_query_rows=7, pad_to_batch_width=True)
    assert prior_lever_values(p.parse_args([]), section) == {
        "permute_classes": True, "feature_transforms": ("rank",), "max_query_rows": 7,
        "pad_to_batch_width": True,
    }
    args = p.parse_args(["--no-permute-classes", "--feature-transforms", "none, power",
                         "--max-query-rows", "3", "--no-pad-to-batch-width"])
    assert prior_lever_values(args, section) == {
        "permute_classes": False, "feature_transforms": ("none", "power"), "max_query_rows": 3,
        "pad_to_batch_width": False,
    }


# --- DR6.1: pad each batch to its widest task ----------------------------------------------


def _task(n_rows: int, n_features: int, seed: int) -> Task:
    """A small binary task with a few missing cells, for the padding tests."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_rows, n_features)).astype(np.float32)
    X[rng.random(X.shape) < 0.1] = np.nan
    return Task(X=X, y=rng.integers(0, 2, n_rows), n_classes=2,
                is_categorical=np.zeros(n_features, dtype=bool))


def test_collate_pads_to_the_widest_task_only_when_asked():
    tasks = [_task(10, 3, 0), _task(10, 5, 1)]
    full = collate(tasks, n_ctx=6, max_features=8)
    narrow = collate(tasks, n_ctx=6, max_features=8, pad_to_batch_width=True)
    assert full.X.shape == (2, 10, 8) and narrow.X.shape == (2, 10, 5)
    np.testing.assert_array_equal(full.X[:, :, :5].numpy(), narrow.X.numpy())
    assert torch.isnan(full.X[:, :, 5:]).all() and torch.isnan(narrow.X[0, :, 3:]).all()
    assert torch.equal(full.y, narrow.y) and torch.equal(full.n_classes, narrow.n_classes)


def test_pad_to_batch_width_removes_only_padding_and_draws_nothing():
    # Same tasks, same split, same random stream; only all-NaN padding columns are dropped.
    cfg = replace(_MIX, pad_to_batch_width=True)
    a_rng, b_rng = np.random.default_rng(11), np.random.default_rng(11)
    widths = set()
    for _ in range(6):
        full, narrow = sample_batch(a_rng, _MIX, 4), sample_batch(b_rng, cfg, 4)
        w = narrow.X.shape[2]
        widths.add(w)
        _assert_batches_equal(
            TaskBatch(full.X[:, :, :w], full.y, full.n_ctx, full.n_classes, full.period), narrow
        )
        assert torch.isnan(full.X[:, :, w:]).all()
    assert a_rng.random() == b_rng.random()
    assert min(widths) < _MIX.max_features  # the lever actually narrowed some batch


def test_sequential_dump_replays_a_batch_width_prior_exactly(tmp_path):
    cfg = replace(_DUMP_CFG, pad_to_batch_width=True)
    write_task_dump(tmp_path / "d", cfg, seed=5, batch_size=3, n_batches=4, shard_size=2)
    dump = TaskDump.open(tmp_path / "d")
    rng = np.random.default_rng(5)
    for i in range(4):
        _assert_batches_equal(dump.batch(i), sample_batch(rng, cfg, 3))


#: Every architecture branch that reads the padding mask: plain pooling, the cell-attention
#: block with cell labels, attention pooling, summary tokens, the basis/ECDF/mask-embedding
#: value channel, and the split row attention.
_WIDTH_ARCHS = {
    "pooled": {},
    "cell_blocks": {"n_cell_blocks": 1, "cell_labels": True},
    "attention_pool": {"pooling": "attention"},
    "summary_tokens": {"n_summary_tokens": 2},
    "basis_ecdf": {"value_embedding": "rbf", "value_ecdf": True, "mask_embedding": True},
    "split_attention": {"n_cell_blocks": 1, "attn_impl": "split"},
}


def _width_model(**arch: object) -> FinancialTFM:
    """A seeded tiny model at ``max_features=12`` in eval mode."""
    torch.manual_seed(0)
    cfg = ModelConfig(max_features=12, max_classes=5, d_cell=16, d_model=32, n_heads=2,
                      n_layers=1, n_col_layers=1, d_ff=64, column_id_dim=4, **arch)
    return FinancialTFM(cfg).eval()


@pytest.mark.parametrize("arch", list(_WIDTH_ARCHS))
def test_outputs_do_not_depend_on_the_padded_width(arch):
    """A task's logits are the same padded to its batch's width or to max_features, at a
    fixed column_id_seed, once column identities are drawn at full width; the old
    width-dependent draw is why the model flag is needed."""
    model = _width_model(**_WIDTH_ARCHS[arch])
    tasks = [_task(30, 4, 0), _task(30, 9, 1), _task(30, 6, 2)]
    full = collate(tasks, n_ctx=20, max_features=12)
    narrow = collate(tasks, n_ctx=20, max_features=12, pad_to_batch_width=True)
    assert narrow.X.shape[2] == 9
    with torch.no_grad():
        want = model(full.X, full.y, full.n_ctx, full.n_classes, column_id_seed=7)
        old = model(narrow.X, narrow.y, narrow.n_ctx, narrow.n_classes, column_id_seed=7)
        model.column_ids_full_width = True
        got = model(narrow.X, narrow.y, narrow.n_ctx, narrow.n_classes, column_id_seed=7)
        again = model(full.X, full.y, full.n_ctx, full.n_classes, column_id_seed=7)
    torch.testing.assert_close(got, want, rtol=0, atol=1e-5)
    assert torch.equal(again, want)  # at full width the flag changes nothing, byte for byte
    assert (old - want)[..., :2].abs().max() > 1e-3  # classes 2-4 are masked to -inf


def test_a_batch_whose_widest_task_is_max_features_is_byte_identical():
    # DR6.1's stated check: at full width the narrow batch *is* the full batch.
    model = _width_model(n_cell_blocks=1, cell_labels=True)
    tasks = [_task(30, 12, 0), _task(30, 5, 1)]
    full = collate(tasks, n_ctx=20, max_features=12)
    narrow = collate(tasks, n_ctx=20, max_features=12, pad_to_batch_width=True)
    _assert_batches_equal(full, narrow)
    with torch.no_grad():
        want = model(full.X, full.y, full.n_ctx, full.n_classes, column_id_seed=3)
        model.column_ids_full_width = True
        got = model(narrow.X, narrow.y, narrow.n_ctx, narrow.n_classes, column_id_seed=3)
    assert torch.equal(got, want)


@pytest.mark.parametrize("full_width_ids", [True, False])
def test_a_training_step_at_batch_width_matches_full_width(full_width_ids):
    """In training mode (fresh column ids from the global RNG, no seed) the loss and every
    gradient match full padding, and the global RNG ends where it would, so no later draw
    shifts. Without the full-width draw neither holds, which is why fintfm-train sets it."""
    model = _width_model(n_cell_blocks=1, cell_labels=True).train()
    tasks = [_task(30, 4, 0), _task(30, 9, 1), _task(30, 6, 2)]
    full = collate(tasks, n_ctx=20, max_features=12)
    narrow = collate(tasks, n_ctx=20, max_features=12, pad_to_batch_width=True)

    def step(batch: TaskBatch, full_width: bool) -> tuple[torch.Tensor, list, torch.Tensor]:
        """Loss, gradients and the RNG state after one forward/backward from seed 9."""
        model.column_ids_full_width = full_width
        model.zero_grad(set_to_none=True)
        torch.manual_seed(9)
        loss = model.loss(batch.X, batch.y, batch.n_ctx, batch.n_classes)
        loss.backward()
        grads = [p.grad.clone() for p in model.parameters() if p.grad is not None]
        return loss.detach(), grads, torch.get_rng_state()

    want_loss, want_grads, want_rng = step(full, False)
    loss, grads, rng = step(narrow, full_width_ids)
    if full_width_ids:
        torch.testing.assert_close(loss, want_loss, rtol=0, atol=1e-6)
        for g, w in zip(grads, want_grads, strict=True):
            torch.testing.assert_close(g, w, rtol=1e-4, atol=1e-6)
        assert torch.equal(rng, want_rng)
    else:
        assert not torch.equal(rng, want_rng)
        assert abs(loss.item() - want_loss.item()) > 1e-4


def test_fintfm_train_sets_the_full_width_draw_from_the_prior(tmp_path):
    from fintfm.modeling.train import TrainConfig, benchmark, train

    model_cfg = ModelConfig(max_features=12, max_classes=5, d_cell=8, d_model=16, n_layers=1,
                            n_col_layers=1, n_heads=2, d_ff=32, n_cell_blocks=1,
                            cell_labels=True)
    common = {"steps": 2, "batch_size": 3, "eval_every": 1, "log_every": 1000,
              "device": "cpu", "seed": 4}
    narrow_cfg = replace(_MIX, pad_to_batch_width=True)
    assert not train(model_cfg, _MIX, TrainConfig(**common),
                     str(tmp_path / "f.pt")).column_ids_full_width
    assert train(model_cfg, narrow_cfg, TrainConfig(**common),
                 str(tmp_path / "n.pt")).column_ids_full_width
    assert benchmark(model_cfg, narrow_cfg, TrainConfig(**common), n_steps=1, warmup=0)["steps"]
