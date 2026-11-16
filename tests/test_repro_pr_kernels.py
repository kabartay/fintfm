"""Preflight logic of the TabArena-Lite reproduction kernels (no network, no fits)."""

import importlib.util
import json
from pathlib import Path

import pytest

KAGGLE = Path(__file__).resolve().parents[1] / "scripts" / "kaggle"
KERNELS = ["task_lite_repro_pr_bin", "task_lite_repro_pr_mcreg"]

PUBLISHED = """
BINARY_CHECKPOINT = {"repo_id": "a/b", "filename": "f", "revision": "r", "sha256": "s"}
MULTICLASS_CHECKPOINT = {"repo_id": "a/b", "filename": "f", "revision": "r", "sha256": "s"}
REGRESSION_CHECKPOINT = {"repo_id": "a/b", "filename": "f", "revision": "r", "sha256": "s"}
"""
PLACEHOLDERS = """
BINARY_CHECKPOINT: dict = {"repo_id": "a/b", "filename": "f", "revision": "r", "sha256": "s"}
MULTICLASS_CHECKPOINT_UNPUBLISHED = {"unpublished": "x", "sha256": "s"}
REGRESSION_CHECKPOINT_UNPUBLISHED = {"unpublished": "y", "sha256": "s"}
"""


def load(kernel: str):
    """Import a generated kernel script by path."""
    path = KAGGLE / kernel / f"{kernel}.py"
    spec = importlib.util.spec_from_file_location(kernel, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("kernel", KERNELS)
def test_shipped_placeholder_commit_is_refused(kernel):
    mod = load(kernel)
    assert mod.commit_problem(mod.FORK_COMMIT) is not None
    with pytest.raises(SystemExit) as exc:
        mod.refuse([mod.commit_problem(mod.FORK_COMMIT)])
    assert "FORK_COMMIT" in str(exc.value)


def test_commit_and_version_checks():
    mod = load(KERNELS[0])
    assert mod.commit_problem("eb396e3") is None
    assert mod.commit_problem("EB396E3") is not None
    assert mod.commit_problem("abc") is not None
    assert mod.version_problem("0.11.0", lambda v: True) is None
    assert "not installable" in mod.version_problem("0.11.0", lambda v: False)
    assert "release number" in mod.version_problem("latest", lambda v: True)
    mod.refuse([])  # no problems: returns


def test_unpublished_checkpoints_by_problem_type():
    mod = load(KERNELS[0])
    assert mod.unpublished_checkpoints(PUBLISHED, ("binary", "multiclass", "regression")) == []
    assert mod.unpublished_checkpoints(PLACEHOLDERS, ("binary",)) == []
    found = mod.unpublished_checkpoints(PLACEHOLDERS, ("binary", "multiclass", "regression"))
    assert [f.split(":")[0] for f in found] == ["multiclass", "regression"]
    # A type with no definition at all cannot be verified, so it is refused too.
    assert mod.unpublished_checkpoints("BINARY_CHECKPOINT = {'repo_id': 'a'}", ("regression",))


def test_each_kernel_checks_only_its_own_problem_types():
    bin_mod, mcreg = load(KERNELS[0]), load(KERNELS[1])
    assert bin_mod.problem_types() == ("binary",)
    assert mcreg.problem_types() == ("multiclass", "regression")
    # Multiclass and regression are still placeholders: only the binary kernel passes.
    assert bin_mod.unpublished_checkpoints(PLACEHOLDERS, bin_mod.problem_types()) == []
    assert len(mcreg.unpublished_checkpoints(PLACEHOLDERS, mcreg.problem_types())) == 2


def test_runs_cover_the_51_lite_datasets_once():
    names, counts = [], {}
    for kernel in KERNELS:
        for run, (ptype, datasets) in load(kernel).RUNS.items():
            names += datasets
            counts[ptype] = counts.get(ptype, 0) + len(datasets)
            assert run.startswith("fintfm_pr_repro_")
    assert counts == {"binary": 30, "multiclass": 8, "regression": 13}
    assert len(names) == len(set(names)) == 51


def test_child_env_strips_fintfm_variables(monkeypatch):
    mod = load(KERNELS[0])
    monkeypatch.setenv("FINTFM_DEVICE", "cpu")
    env, removed = mod.child_env()
    assert removed == ["FINTFM_DEVICE"]
    assert not any(k.startswith("FINTFM_") for k in env)


@pytest.mark.parametrize("kernel", KERNELS)
def test_generated_files_are_current(kernel):
    spec = importlib.util.spec_from_file_location("gen", KAGGLE / "make_repro_pr_kernels.py")
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    assert (KAGGLE / kernel / f"{kernel}.py").read_text() == gen.render(kernel)
    meta = json.loads((KAGGLE / kernel / "kernel-metadata.json").read_text())
    assert meta == gen.metadata(kernel)
    assert meta["is_private"] == "true"
    assert meta["enable_gpu"] == "true"
    assert meta["enable_internet"] == "true"
