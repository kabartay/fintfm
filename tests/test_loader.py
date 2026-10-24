"""Streaming loader and data-parallel training (design review DR5.2).

The properties a loss curve cannot show: the stream's batch ``i`` is a dumped batch ``i``,
the stream does not depend on how many workers generate it, ranks take disjoint indices, a
resume needs no generator state, and a two-process run averages its ranks' gradients.
"""

import os
import subprocess
import sys

import pytest
import torch

from fintfm.modeling.model import ModelConfig
from fintfm.prior import PriorConfig

_PRIOR = PriorConfig(max_features=8, max_classes=2, n_rows_choices=(24, 32))
_MODEL = ModelConfig(
    max_features=8, max_classes=2, d_cell=8, d_model=16, n_layers=1, n_col_layers=1, n_heads=2
)


def _assert_batches_equal(a, b) -> None:
    """Every field of two ``TaskBatch`` objects is equal, NaN padding included."""
    assert a.n_ctx == b.n_ctx
    assert torch.equal(torch.nan_to_num(a.X, nan=-9.0), torch.nan_to_num(b.X, nan=-9.0))
    assert torch.isnan(a.X).equal(torch.isnan(b.X))
    assert torch.equal(a.y, b.y) and torch.equal(a.n_classes, b.n_classes)
    assert (a.period is None) == (b.period is None)


@pytest.mark.parametrize("world_size", [1, 2, 3])
@pytest.mark.parametrize("start", [0, 4, 7])
def test_ranks_take_disjoint_indices_that_cover_the_interval(start, world_size):
    from fintfm.modeling.loader import rank_indices

    parts = [list(rank_indices(start, 20, r, world_size)) for r in range(world_size)]
    for r, part in enumerate(parts):
        assert all(i % world_size == r for i in part), (r, part)
    flat = [i for part in parts for i in part]
    assert sorted(flat) == list(range(start, 20)) and len(set(flat)) == len(flat)
    with pytest.raises(ValueError):
        rank_indices(0, 10, world_size, world_size)


def test_stream_batch_i_equals_the_dumps_batch_i(tmp_path):
    """A ``per_shard`` dump with one batch per shard seeds shard ``s`` as ``[seed, s]``: the
    stream's batch ``s``. Dumps stay as the byte-identity check of the stream."""
    from fintfm.modeling.loader import stream_batches
    from fintfm.prior.dump import TaskDump, write_task_dump

    write_task_dump(tmp_path / "d", _PRIOR, seed=5, batch_size=3, n_batches=6,
                    shard_size=1, stream="per_shard")
    dump = TaskDump.open(tmp_path / "d")
    got = list(stream_batches(_PRIOR, 3, seed=5, start=0, stop=6))
    assert [i for i, _ in got] == list(range(6))
    for i, batch in got:
        _assert_batches_equal(batch, dump.batch(i))


def test_two_workers_give_the_same_stream_as_one():
    """Worker processes split the indices and the DataLoader restores their order, so the
    stream is the same at any worker count, from any start index, and per rank."""
    from fintfm.modeling.loader import stream_batches

    one = list(stream_batches(_PRIOR, 2, seed=1, start=3, stop=9, workers=1))
    two = list(stream_batches(_PRIOR, 2, seed=1, start=3, stop=9, workers=2, prefetch=1))
    inline = list(stream_batches(_PRIOR, 2, seed=1, start=3, stop=9, workers=0))
    assert [i for i, _ in one] == [i for i, _ in two] == [i for i, _ in inline] == list(
        range(3, 9)
    )
    for (_, a), (_, b), (_, c) in zip(one, two, inline):
        _assert_batches_equal(a, b)
        _assert_batches_equal(a, c)
    # two ranks of the same interval interleave back into the single-process stream
    rank1 = list(stream_batches(_PRIOR, 2, seed=1, start=3, stop=9, rank=1, world_size=2))
    assert [i for i, _ in rank1] == [3, 5, 7]
    for (i, a), (j, b) in zip(rank1, [(i, b) for i, b in inline if i % 2 == 1]):
        assert i == j
        _assert_batches_equal(a, b)


def _train(tmp_path, name, **kw):
    from fintfm.modeling.train import TrainConfig, train

    common = {"steps": 6, "batch_size": 2, "eval_every": 1000, "log_every": 1000, "seed": 2}
    return train(_MODEL, _PRIOR, TrainConfig(**{**common, **kw}), str(tmp_path / name))


def test_stream_training_trains_on_the_dumps_batches(tmp_path):
    """Training under ``--loader stream`` is training on the equivalent dump, weight for
    weight: step ``i`` consumed batch ``i``."""
    from fintfm.prior.dump import write_task_dump

    write_task_dump(tmp_path / "d", _PRIOR, seed=2, batch_size=2, n_batches=6, shard_size=1,
                    stream="per_shard")
    streamed = _train(tmp_path, "s.pt", loader="stream")
    dumped = _train(tmp_path, "d.pt", task_dump=str(tmp_path / "d"))
    for (name, a), (_, b) in zip(streamed.state_dict().items(), dumped.state_dict().items()):
        assert torch.equal(a, b), name


def test_stream_resume_starts_at_the_step_index_without_generator_state(tmp_path):
    """A stream run resumed from its state equals the uninterrupted one, and does so even
    when the state's generator was never advanced: the stream reads only ``(seed, i)``."""
    whole = _train(tmp_path, "whole.pt", loader="stream")
    _train(tmp_path, "part.pt", loader="stream", run_steps=3)
    state_path = tmp_path / "part.pt.state"
    state = torch.load(state_path, weights_only=False)
    assert state["loader"] == "stream" and state["world_size"] == 1
    # scramble the sync generator's state: a stream resume must not read it
    import numpy as np

    state["numpy_rng"] = np.random.default_rng(12345).bit_generator.state
    torch.save(state, state_path)
    resumed = _train(tmp_path, "resumed.pt", loader="stream", resume=str(state_path))
    for (name, a), (_, b) in zip(whole.state_dict().items(), resumed.state_dict().items()):
        assert torch.equal(a, b), name


def test_stream_with_workers_trains_to_the_same_weights(tmp_path):
    inline = _train(tmp_path, "a.pt", loader="stream", steps=3)
    workers = _train(tmp_path, "b.pt", loader="stream", steps=3, loader_workers=2)
    for (name, a), (_, b) in zip(inline.state_dict().items(), workers.state_dict().items()):
        assert torch.equal(a, b), name


def test_stream_evaluation_has_a_stream_of_its_own(tmp_path, monkeypatch):
    """``[seed, 1]`` is both the AU1.18 evaluation key and the stream's batch 1, so under the
    stream evaluation must draw from another key, and evaluating must not move training."""
    import importlib

    import numpy as np

    train_mod = importlib.import_module("fintfm.modeling.train")
    seen = []
    real = train_mod._eval_quality

    def recording(model, prior_cfg, rng, **kw):
        seen.append(rng.bit_generator.state["state"]["state"])
        return real(model, prior_cfg, rng, **kw)

    monkeypatch.setattr(train_mod, "_eval_quality", recording)
    evaluated = _train(tmp_path, "e.pt", loader="stream", steps=2, eval_every=1)
    plain = _train(tmp_path, "p.pt", loader="stream", steps=2)
    batch_one = np.random.default_rng([2, 1]).bit_generator.state["state"]["state"]
    assert seen and seen[0] != batch_one
    for (name, a), (_, b) in zip(evaluated.state_dict().items(), plain.state_dict().items()):
        assert torch.equal(a, b), name


@pytest.mark.parametrize(
    "kw, match",
    [
        ({"loader_workers": 2}, "needs --loader stream"),
        ({"loader": "stream", "task_dump": "x"}, "two batch sources"),
        ({"ddp": True}, "--ddp needs --loader stream"),
        ({"loader": "bulk"}, "loader must be one of"),
        ({"loader": "stream", "ddp": True}, "torchrun"),
    ],
    ids=["workers-sync", "stream-dump", "ddp-sync", "unknown", "ddp-no-torchrun"],
)
def test_loader_settings_that_would_misbehave_are_refused(tmp_path, monkeypatch, kw, match):
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    with pytest.raises(ValueError, match=match):
        _train(tmp_path, "x.pt", **kw)


def test_resume_refuses_a_different_loader(tmp_path):
    _train(tmp_path, "m.pt", run_steps=2)
    with pytest.raises(ValueError, match="--loader sync"):
        _train(tmp_path, "r.pt", loader="stream", resume=str(tmp_path / "m.pt.state"))


def test_cli_passes_the_loader_flags(tmp_path, monkeypatch):
    import importlib

    train_mod = importlib.import_module("fintfm.modeling.train")
    seen = {}
    monkeypatch.setattr(train_mod, "train", lambda m, p, t, out: seen.update(t=t))
    monkeypatch.delenv("FINTFM_CONFIG", raising=False)
    train_mod.main(["--out", str(tmp_path / "x.pt")])
    assert (seen["t"].loader, seen["t"].loader_workers, seen["t"].ddp) == ("sync", 0, False)
    train_mod.main(["--out", str(tmp_path / "x.pt"), "--loader", "stream", "--loader-workers",
                    "3", "--loader-prefetch", "4", "--ddp"])
    t = seen["t"]
    assert (t.loader, t.loader_workers, t.loader_prefetch, t.ddp) == ("stream", 3, 4, True)


_TINY_CLI = [
    "--steps", "3", "--batch-size", "2", "--max-features", "8", "--max-classes", "2",
    "--d-model", "16", "--d-cell", "8", "--n-layers", "1", "--n-col-layers", "1",
    "--n-heads", "2", "--n-rows-choices", "24,32", "--seed", "4", "--eval-every", "1000",
    "--log-every", "1", "--loader", "stream",
]


def _torchrun(args: list[str], script: str | None = None) -> subprocess.CompletedProcess:
    """Launch ``fintfm-train`` on two CPU (gloo) ranks with ``torchrun``, loopback only.

    ``--standalone`` advertises the machine's hostname, which does not always resolve (it
    hung here on macOS), so the rendezvous is pinned to 127.0.0.1 on a free port. With
    ``script`` the two ranks run that file with ``args`` instead of ``fintfm-train``.
    """
    import socket

    env = {k: v for k, v in os.environ.items() if k not in ("FINTFM_CONFIG", "VIRTUAL_ENV")}
    env.update(OMP_NUM_THREADS="1", PYTHONUNBUFFERED="1")
    if sys.platform == "darwin":
        env["GLOO_SOCKET_IFNAME"] = "lo0"
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    target = (["-m", "fintfm.modeling.train", *_TINY_CLI, *args, "--ddp"] if script is None
              else [script, *args])
    proc = subprocess.run(
        [sys.executable, "-m", "torch.distributed.run", "--nnodes", "1", "--nproc-per-node",
         "2", "--master-addr", "127.0.0.1", "--master-port", str(port), "--local-addr",
         "127.0.0.1", *target],
        env=env, capture_output=True, text=True, timeout=300, check=False,
    )
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    return proc


_NEEDS_GLOO = pytest.mark.skipif(
    not torch.distributed.is_gloo_available(), reason="no gloo backend"
)


@_NEEDS_GLOO
def test_two_process_gloo_run_averages_the_ranks_gradients(tmp_path):
    """A CPU ``torchrun`` launch of two ranks: each step trains on stream batches ``2s`` and
    ``2s + 1``, one per rank, and the saved weights are those of a single process stepping
    on the mean of the two batches' gradients. Rank 0 alone writes files."""
    import numpy as np

    from fintfm.config import load_config
    from fintfm.modeling.loader import stream_batch
    from fintfm.modeling.model import FinancialTFM
    from fintfm.modeling.train import _lr_schedule, build_configs, build_parser
    from fintfm.provenance import read_checkpoint_provenance

    out = tmp_path / "ddp.pt"
    proc = _torchrun(["--out", str(out)])
    assert proc.stdout.count("saved checkpoint") == 1  # rank 0 only
    assert "ddp: 2 processes, 4 tasks per step" in proc.stdout
    assert sorted(p.name for p in tmp_path.iterdir()) == ["ddp.pt", "ddp.pt.state"]
    rec = read_checkpoint_provenance(out)
    assert (rec["world_size"], rec["tasks_seen"]) == (2, 3 * 2 * 2)

    args = build_parser().parse_args([*_TINY_CLI, "--out", str(out)])
    model_cfg, prior_cfg, train_cfg = build_configs(args, load_config(None))
    torch.manual_seed(train_cfg.seed)
    ref = FinancialTFM(model_cfg)
    opt = torch.optim.AdamW(ref.parameters(), lr=train_cfg.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: _lr_schedule(s, train_cfg))
    # each rank's forward draws its column identities from its own torch generator, seeded
    # alike, so the reference keeps one generator state per rank
    rank_rng = [torch.get_rng_state(), torch.get_rng_state()]
    for step in range(train_cfg.steps):
        grads = []
        for rank in range(2):
            b = stream_batch(prior_cfg, train_cfg.batch_size, train_cfg.seed, 2 * step + rank)
            torch.set_rng_state(rank_rng[rank])
            opt.zero_grad(set_to_none=True)
            ref.loss(b.X, b.y, b.n_ctx, b.n_classes).backward()
            rank_rng[rank] = torch.get_rng_state()
            grads.append([p.grad.clone() for p in ref.parameters()])
        for p, g0, g1 in zip(ref.parameters(), *grads):
            p.grad = (g0 + g1) / 2
        torch.nn.utils.clip_grad_norm_(ref.parameters(), 1.0)
        opt.step()
        sched.step()
    got = FinancialTFM.load(str(out))
    for (name, a), (_, b) in zip(ref.state_dict().items(), got.state_dict().items()):
        assert np.allclose(a.numpy(), b.numpy(), atol=1e-6), name


@_NEEDS_GLOO
def test_two_process_run_resumes_to_the_uninterrupted_weights(tmp_path):
    """A ``--ddp`` run chunked with ``--run-steps`` and resumed equals the uninterrupted one
    bit for bit: each rank restarts at its own stream indices and its own torch generator."""
    from fintfm.modeling.model import FinancialTFM

    _torchrun(["--steps", "4", "--out", str(tmp_path / "whole.pt")])
    _torchrun(["--steps", "4", "--run-steps", "2", "--out", str(tmp_path / "part.pt")])
    state = torch.load(tmp_path / "part.pt.state", weights_only=False)
    assert state["world_size"] == 2 and len(state["torch_rng_ranks"]) == 2
    _torchrun(["--steps", "4", "--resume", str(tmp_path / "part.pt.state"),
               "--out", str(tmp_path / "resumed.pt")])
    whole = FinancialTFM.load(str(tmp_path / "whole.pt"))
    resumed = FinancialTFM.load(str(tmp_path / "resumed.pt"))
    for (name, a), (_, b) in zip(whole.state_dict().items(), resumed.state_dict().items()):
        assert torch.equal(a, b), name


#: Two ranks whose clocks disagree: rank 0 alone has spent the budget, then rank 1 alone.
_BUDGET_SCRIPT = '''
import torch.distributed as dist

from fintfm.modeling.train import _time_budget_spent

dist.init_process_group("gloo")
rank = dist.get_rank()
first = _time_budget_spent(10.0 if rank == 0 else 0.0, 5.0, 2, "cpu")
second = _time_budget_spent(0.0 if rank == 0 else 10.0, 5.0, 2, "cpu")
print(f"rank {rank}: {first} {second}", flush=True)
dist.destroy_process_group()
'''


@_NEEDS_GLOO
def test_two_ranks_take_rank_zeros_time_budget_decision(tmp_path):
    """Design review DR8.2: under ``--ddp`` each rank reads its own clock, and a rank that
    stopped alone would leave the other waiting in its next reduction. Rank 0's decision is
    broadcast, so both ranks stop when rank 0's budget is spent and neither stops when only
    rank 1's is."""
    script = tmp_path / "budget.py"
    script.write_text(_BUDGET_SCRIPT)
    proc = _torchrun([], script=str(script))
    assert "rank 0: True False" in proc.stdout and "rank 1: True False" in proc.stdout


@_NEEDS_GLOO
def test_two_process_run_seconds_stop_resumes_to_the_uninterrupted_weights(tmp_path):
    """A ``--ddp`` run stopped by ``--run-seconds`` (0 here, so it stops at the first step
    boundary) exits cleanly on both ranks at one step, records the stop, and resumed to the
    full ``--steps`` equals the uninterrupted run bit for bit."""
    from fintfm.modeling.model import FinancialTFM
    from fintfm.provenance import read_checkpoint_provenance

    _torchrun(["--steps", "4", "--out", str(tmp_path / "whole.pt")])
    proc = _torchrun(["--steps", "4", "--run-seconds", "0", "--out", str(tmp_path / "part.pt")])
    assert proc.stdout.count("stopping at step 1") == 1  # rank 0 alone logs
    budget = read_checkpoint_provenance(tmp_path / "part.pt")["time_budget"]
    assert (budget["stop_step"], budget["stopped_early"]) == (1, True)
    state = torch.load(tmp_path / "part.pt.state", weights_only=False)
    assert state["step"] == 1 and len(state["torch_rng_ranks"]) == 2
    _torchrun(["--steps", "4", "--resume", str(tmp_path / "part.pt.state"),
               "--out", str(tmp_path / "resumed.pt")])
    whole = FinancialTFM.load(str(tmp_path / "whole.pt"))
    resumed = FinancialTFM.load(str(tmp_path / "resumed.pt"))
    for (name, a), (_, b) in zip(whole.state_dict().items(), resumed.state_dict().items()):
        assert torch.equal(a, b), name


def test_benchmark_measures_the_stream_loader(tmp_path, monkeypatch, capsys):
    """``--benchmark-steps`` with ``--loader stream`` reads the stream, not the sync sampler,
    and records the loader settings beside the rate."""
    import importlib
    import json

    train_mod = importlib.import_module("fintfm.modeling.train")
    monkeypatch.delenv("FINTFM_CONFIG", raising=False)

    def no_sync_sampling(*a, **k):
        raise AssertionError("sampled on the sync path despite --loader stream")

    monkeypatch.setattr(train_mod, "sample_batch", no_sync_sampling)
    train_mod.main([*_TINY_CLI, "--benchmark-steps", "2", "--loader-workers", "1",
                    "--out", str(tmp_path / "x.pt")])
    record = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert (record["loader"], record["loader_workers"], record["tasks"]) == ("stream", 1, 4)
    with pytest.raises(ValueError, match="one process"):
        train_mod.main([*_TINY_CLI, "--benchmark-steps", "2", "--ddp",
                        "--out", str(tmp_path / "x.pt")])
