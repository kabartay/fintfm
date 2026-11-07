"""48.15's masked-column task swap: default-off identity, validity of swapped tasks, a smoke run."""

import importlib

import numpy as np
import pytest
import torch

from fintfm.modeling import ModelConfig
from fintfm.modeling.masked_column import swap_batch, swap_key, swap_task
from fintfm.modeling.train import TrainConfig, build_configs, build_parser, train
from fintfm.prior import PriorConfig
from fintfm.prior.base import TaskBatch

train_mod = importlib.import_module("fintfm.modeling.train")

N, F, N_CTX = 64, 6, 40


def _task(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """A fully observed random table with a binary label correlated to column 0."""
    r = np.random.default_rng(seed)
    X = r.normal(size=(N, F)).astype(np.float32)
    y = (X[:, 0] + r.normal(size=N) > 0).astype(np.int64)
    return X, y


def _batch(B: int = 3) -> TaskBatch:
    Xs, ys = zip(*[_task(s) for s in range(B)])
    return TaskBatch(
        torch.from_numpy(np.stack(Xs)),
        torch.from_numpy(np.stack(ys)),
        N_CTX,
        torch.full((B,), 2, dtype=torch.int64),
    )


def test_probability_one_swaps_every_eligible_task_with_valid_shapes_and_dtypes():
    batch = _batch()
    out, n = swap_batch(batch, 1.0, np.random.default_rng(0))
    assert n == 3
    assert out.X.shape == batch.X.shape and out.X.dtype == torch.float32
    assert out.y.shape == batch.y.shape and out.y.dtype == torch.int64
    assert out.n_ctx == N_CTX and out.period is None
    assert out.n_classes.tolist() == [2, 2, 2]
    assert set(out.y.unique().tolist()) <= {0, 1}
    assert torch.equal(batch.y, _batch().y)  # the input is not modified


def test_target_comes_from_a_removed_column_and_the_label_becomes_a_feature():
    X, y = _task(1)
    X2, y2 = swap_task(X, y, N_CTX, np.random.default_rng(5))
    changed = [j for j in range(F) if not np.array_equal(X[:, j], X2[:, j])]
    assert len(changed) == 1
    j = changed[0]
    assert np.array_equal(X2[:, j], y.astype(np.float32))  # original label is now a feature
    # the new target is a monotone cut of the removed column, in one direction or the other
    col = X[:, j]
    up = col[y2 == 1].min() > col[y2 == 0].max()
    down = col[y2 == 1].max() < col[y2 == 0].min()
    assert up or down
    assert min(int(y2[:N_CTX].sum()), N_CTX - int(y2[:N_CTX].sum())) >= 2


def test_bin_edge_uses_context_rows_only():
    """Changing query rows never moves the cut: context-side targets are identical."""
    X, y = _task(2)
    X_alt = X.copy()
    X_alt[N_CTX:] = np.random.default_rng(9).normal(size=X[N_CTX:].shape) * 100
    a = swap_task(X, y, N_CTX, np.random.default_rng(7))
    b = swap_task(X_alt, y, N_CTX, np.random.default_rng(7))
    assert a is not None and b is not None
    assert np.array_equal(a[1][:N_CTX], b[1][:N_CTX])


def test_constant_missing_and_padded_columns_are_skipped():
    X, y = _task(3)
    X[:, 1] = 4.0  # constant in context
    X[3, 2] = np.nan  # a missing cell: no target for that row
    X[:, 5] = np.nan  # padding
    for seed in range(20):
        out = swap_task(X, y, N_CTX, np.random.default_rng(seed))
        assert out is not None
        changed = [j for j in range(F) if not np.array_equal(X[:, j], out[0][:, j], equal_nan=True)]
        assert changed[0] in (0, 3, 4)
    only_bad = np.full((N, 2), np.nan, dtype=np.float32)
    only_bad[:, 0] = 1.0
    assert swap_task(only_bad, y, N_CTX, np.random.default_rng(0)) is None
    batch = TaskBatch(
        torch.from_numpy(only_bad)[None],
        torch.from_numpy(y)[None],
        N_CTX,
        torch.tensor([2]),
    )
    out, n = swap_batch(batch, 1.0, np.random.default_rng(0))
    assert n == 0 and torch.allclose(out.X, batch.X, equal_nan=True)
    assert torch.equal(out.y, batch.y)


def test_probability_zero_is_the_identity_and_a_stream_of_its_own():
    batch = _batch()
    out, n = swap_batch(batch, 0.0, np.random.default_rng(0))
    assert n == 0 and torch.equal(out.y, batch.y)
    assert torch.equal(out.X, batch.X)
    assert swap_key(0, 0, 0)[-1] == 1 and swap_key(0, 0, 0)[1] not in (1, 2)


def test_periods_are_refused():
    batch = _batch()
    batch.period = torch.zeros_like(batch.y)
    with pytest.raises(ValueError, match="periods"):
        swap_batch(batch, 1.0, np.random.default_rng(0))


_MODEL = ModelConfig(
    max_features=8,
    max_classes=2,
    d_cell=8,
    d_model=16,
    n_layers=1,
    n_col_layers=1,
    n_heads=2,
)
_PRIOR = PriorConfig(max_features=8, max_classes=2, n_rows=32)


def _run(prob: float, monkeypatch, tmp_path, name: str):
    seen: list[torch.Tensor] = []
    real = train_mod._training_step

    def spy(model, batch, *a, **k):
        seen.append(batch.y.clone())
        return real(model, batch, *a, **k)

    monkeypatch.setattr(train_mod, "_training_step", spy)
    model = train(
        _MODEL,
        _PRIOR,
        TrainConfig(
            steps=4,
            batch_size=2,
            eval_every=1000,
            log_every=1,
            device="cpu",
            seed=3,
            masked_column_prob=prob,
        ),
        str(tmp_path / name),
    )
    monkeypatch.setattr(train_mod, "_training_step", real)
    return seen, model


def test_default_off_batch_sequence_and_weights_match_an_explicit_zero(monkeypatch, tmp_path):
    a, ma = _run(0.0, monkeypatch, tmp_path, "a.pt")
    b, mb = _run(0.0, monkeypatch, tmp_path, "b.pt")
    assert all(torch.equal(x, y) for x, y in zip(a, b))
    assert TrainConfig().masked_column_prob == 0.0
    for (_, wa), (_, wb) in zip(ma.state_dict().items(), mb.state_dict().items()):
        assert torch.equal(wa, wb)


def test_swap_does_not_perturb_the_underlying_task_sequence(monkeypatch, tmp_path):
    """With p=0.5 the sampled tasks are the p=0 ones, some relabelled: the swap draws from
    its own generator and never from the prior's."""
    drawn: list[list] = [[], []]

    def run(i: int, prob: float):
        real = train_mod.sample_batch

        def rec(rng, cfg, bs):
            b = real(rng, cfg, bs)
            drawn[i].append(b.X.clone())
            return b

        monkeypatch.setattr(train_mod, "sample_batch", rec)
        _run(prob, monkeypatch, tmp_path, f"s{i}.pt")
        monkeypatch.setattr(train_mod, "sample_batch", real)

    run(0, 0.0)
    run(1, 0.5)
    assert len(drawn[0]) == len(drawn[1]) == 4
    for x, y in zip(*drawn):
        assert torch.allclose(x, y, equal_nan=True)


def test_tiny_cpu_training_with_half_the_tasks_swapped_completes(monkeypatch, tmp_path):
    seen, model = _run(0.5, monkeypatch, tmp_path, "m.pt")
    assert len(seen) == 4
    assert all(torch.isfinite(p).all() for p in model.parameters())
    assert (tmp_path / "m.pt").exists()


def test_flag_and_config_reach_train_config_and_bad_values_are_refused():
    cfg_args = build_parser().parse_args(["--masked-column-prob", "0.25"])
    from fintfm.config import load_config

    assert build_configs(cfg_args, load_config())[2].masked_column_prob == 0.25
    assert build_configs(build_parser().parse_args([]), load_config())[2].masked_column_prob == 0.0
    with pytest.raises(ValueError, match="masked_column_prob"):
        train(_MODEL, _PRIOR, TrainConfig(steps=1, masked_column_prob=1.5), "x.pt")


def test_kernel_differs_from_its_control_by_the_lever_alone_and_cannot_launch():
    """The 48.15 kernel is au53's with one flag swapped, and its commit is the placeholder."""
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "scripts" / "kaggle"

    def flags(kernel: str) -> list[str]:
        for node in ast.parse((root / kernel / "kernel.py").read_text()).body:
            if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", "") == "ARM_FLAGS":
                return ast.literal_eval(node.value)
        raise AssertionError(kernel)

    arm, ref = flags("ts_4815_maskcol_s0"), flags("ts_au53_cell2_s0")
    assert arm[:-2] == ref[:-2] == arm[: len(arm) - 2]
    assert arm[-2:] == ["--masked-column-prob", "0.25"] and ref[-2:] == ["--n-cell-blocks", "2"]
    text = (root / "ts_4815_maskcol_s0" / "kernel.py").read_text()
    assert 'COMMIT = "PIN-AFTER-MERGE"' in text and "--checkpoint-every" in text
