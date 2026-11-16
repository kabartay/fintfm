"""The Kaggle CPU kernel for openspec 4.3 and 12.5: pins, commands and plugin, with no network."""

from __future__ import annotations

import ast
import dataclasses
import importlib
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KDIR = ROOT / "scripts" / "kaggle" / "task_cpu_fitcal_attr"


def _load():
    spec = importlib.util.spec_from_file_location("k_fitcal_attr", KDIR / "task_cpu_fitcal_attr.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_metadata_is_private_cpu_with_internet_and_no_inputs() -> None:
    """Private script kernel on CPU; nothing local is attached."""
    meta = json.loads((KDIR / "kernel-metadata.json").read_text())
    assert meta["id"] == "muhakabartay/fintfm-task-cpu-fitcal-attr"
    assert meta["is_private"] == "true" and meta["enable_gpu"] == "false"
    assert meta["enable_internet"] == "true" and meta["kernel_type"] == "script"
    assert meta["code_file"] == "task_cpu_fitcal_attr.py"
    for key in ("dataset_sources", "kernel_sources", "model_sources", "competition_sources"):
        assert meta[key] == []


def test_pins_and_checkpoint_hash_match_the_repo() -> None:
    """The commit is a SHA, the revision is the README's and the hash is the model card's."""
    k = _load()
    assert re.fullmatch(r"[0-9a-f]{7,40}", k.FINTFM_COMMIT)
    assert k.CKPT_SHA256 in (ROOT / "docs" / "models" / "fintfm-binary.md").read_text()
    assert "f116bfd43a2b15c65ed3551ea8c38e3364629ddc" in k.CKPT_URL
    assert "f116bfd43a2b15c65ed3551ea8c38e3364629ddc" in (ROOT / "README.md").read_text()


def test_fitcal_command_matches_the_queued_one() -> None:
    """Same arguments as tasks.md 4.3 and the queue, through the registered entry point."""
    k = _load()
    assert k.FITCAL_CMD[-4:] == [
        "--model",
        "runs/v4-cellattn-labels.pt",
        "--out",
        "runs/fitted-calibration",
    ]
    entry = re.search(r'fintfm-fitcal\s*=\s*"([^"]+)"', (ROOT / "pyproject.toml").read_text())
    module, func = entry.group(1).split(":")
    assert k.FITCAL_CMD[3] == f"from {module} import {func}; {func}()"


def test_attribution_command_and_plugin_parse() -> None:
    """pytest selects the thresholds test with -rx; the plugin source is valid Python."""
    k = _load()
    assert k.ATTR_CMD[3:8] == [
        "pytest",
        "tests/test_attribution_stability.py",
        "-k",
        "thresholds",
        "-rx",
    ]
    assert (ROOT / k.ATTR_CMD[4]).exists()
    ast.parse(k.PLUGIN)


def test_env_drops_fintfm_vars_and_sets_threads() -> None:
    """Inherited FINTFM_* variables do not leak in; the slow gate is set explicitly."""
    k = _load()
    os.environ["FINTFM_X"] = "1"
    try:
        env = k.child_env({"FINTFM_SLOW_ATTRIBUTION": "1"})
    finally:
        del os.environ["FINTFM_X"]
    assert "FINTFM_X" not in env and env["FINTFM_SLOW_ATTRIBUTION"] == "1"
    assert env["OMP_NUM_THREADS"] == "4" and "VIRTUAL_ENV" not in env


def test_pytest_counts_and_timeouts(tmp_path: Path) -> None:
    """The summary-line parser reads outcome counts; timeouts fit the session limit."""
    k = _load()
    log = tmp_path / "p.log"
    log.write_text("XFAIL tests/x.py::t\n=== 1 xfailed, 2 deselected in 790.12s (0:13:10) ===\n")
    assert k.pytest_counts(log) == {"xfailed": 1, "deselected": 2}
    assert k.FITCAL_TIMEOUT_S + k.ATTR_TIMEOUT_S < 12 * 3600


def test_plugin_writes_statistics(tmp_path: Path, monkeypatch) -> None:
    """With a stubbed ``stability`` the plugin writes the thresholds and the result as JSON."""
    k = _load()
    from fintfm.experiments import attribution_stability as ast_

    @dataclasses.dataclass
    class Fake:
        resample_spearman_mean: float = 0.4

    monkeypatch.setattr(ast_, "stability", lambda *a, **kw: Fake())
    out = tmp_path / "a.json"
    monkeypatch.setenv("ATTR_JSON", str(out))
    (tmp_path / "attr_dump.py").write_text(k.PLUGIN)
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop("attr_dump", None)
    try:
        importlib.import_module("attr_dump")
        ast_.stability()
    finally:
        sys.modules.pop("attr_dump", None)
    rec = json.loads(out.read_text())
    assert rec["results"][0]["resample_spearman_mean"] == 0.4
    assert rec["thresholds"]["MIN_RESAMPLE_SPEARMAN"] == ast_.MIN_RESAMPLE_SPEARMAN
