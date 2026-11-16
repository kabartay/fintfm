"""The Kaggle CPU kernels for openspec 37.3 and 37.4 match the template and the runner's defaults.

No network: this reads the generated files and the pinned constants only.
"""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

from fintfm.evaluation.tabular_benchmark import CLEARED
from fintfm.experiments import suite_scoring

KAGGLE = Path(__file__).resolve().parent.parent / "scripts" / "kaggle"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _gen():
    return _load(KAGGLE / "make_cpu_37_kernels.py", "make_cpu_37_kernels")


def test_generated_kernels_match_template() -> None:
    """Every kernel equals the rendered template and parses."""
    gen = _gen()
    for part, datasets in gen.PARTS.items():
        path = KAGGLE / f"task_cpu_37_suite_{part}" / f"task_cpu_37_suite_{part}.py"
        text = path.read_text()
        assert text == gen.render(part, datasets)
        ast.parse(text)


def test_parts_cover_the_ten_cleared_datasets_once() -> None:
    """The parts partition exactly the ten cleared datasets, and every CSV has a pinned hash."""
    gen = _gen()
    scored = [d for ds in gen.PARTS.values() for d in ds]
    assert sorted(scored) == sorted(CLEARED)
    mod = _load(KAGGLE / "cpu_37_suite_template.py", "cpu_37_template")
    assert sorted(mod.CSV_SHA256) == sorted(CLEARED)
    assert all(len(h) == 64 for h in mod.CSV_SHA256.values())
    assert len(mod.CKPT_SHA256) == 64 and len(mod.HF_DATASET_REVISION) == 40


def test_metadata_is_cpu_private_with_no_inputs_from_this_machine() -> None:
    """CPU only, private, internet on, and no kernel, dataset or model sources."""
    for part in _gen().PARTS:
        meta = json.loads(
            (KAGGLE / f"task_cpu_37_suite_{part}" / "kernel-metadata.json").read_text()
        )
        assert meta["id"] == f"muhakabartay/fintfm-task-cpu-37-suite-{part}"
        assert meta["code_file"] == f"task_cpu_37_suite_{part}.py"
        assert (meta["enable_gpu"], meta["enable_internet"], meta["is_private"]) == (
            "false", "true", "true",
        )  # fmt: skip
        assert meta["dataset_sources"] == meta["kernel_sources"] == meta["model_sources"] == []


def test_command_uses_the_openspec_defaults_and_a_relative_checkpoint() -> None:
    """The command carries no flag that changes a runner default, and the checkpoint path is relative.

    A relative ``runs/...`` path keeps the stored configuration equal to the local command's, so the
    per-dataset JSON files merge instead of being refused as a different configuration.
    """
    mod = _load(KAGGLE / "task_cpu_37_suite_a" / "task_cpu_37_suite_a.py", "k37a")
    cmd = mod.suite_command("clf_num/Higgs", Path("/o"))
    flags = {a for a in cmd if a.startswith("--")}
    assert flags == {"--checkpoint", "--datasets", "--out", "--threads"}
    assert cmd[cmd.index("--checkpoint") + 1] == "v4-cellattn-labels=runs/v4-cellattn-labels.pt"
    assert not Path(mod.CKPT_ARG.partition("=")[2]).is_absolute()
    parsed = {a.dest: a.default for a in _runner_defaults()}
    assert (parsed["folds"], parsed["max_rows"], parsed["max_context"]) == (3, 30_000, 1000)
    assert (parsed["n_ensemble"], parsed["feature_chunk"]) == (8, 16)
    assert suite_scoring.DEFAULT_MAX_ROWS == 30_000


def _runner_defaults():
    """The runner's argparse actions, read by parsing an empty command line."""
    import argparse
    from unittest import mock

    captured: list[argparse.ArgumentParser] = []
    real = argparse.ArgumentParser.parse_args

    def spy(self, args=None, namespace=None):
        captured.append(self)
        raise SystemExit

    with mock.patch.object(argparse.ArgumentParser, "parse_args", spy):
        try:
            suite_scoring.main([])
        except SystemExit:
            pass
    assert real
    return captured[0]._actions
