"""The 11.8 Kaggle CPU kernel runs exactly the task's Verify command and needs no inputs."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KERNEL = ROOT / "scripts" / "kaggle" / "task_cpu_11_8"


def _mod():
    spec = importlib.util.spec_from_file_location("task_cpu_11_8", KERNEL / "task_cpu_11_8.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_metadata_is_cpu_private_and_input_free() -> None:
    """CPU session, internet on, no sources: nothing is uploaded or attached."""
    meta = json.loads((KERNEL / "kernel-metadata.json").read_text())
    assert meta["id"] == "muhakabartay/fintfm-task-cpu-11-8"
    assert meta["code_file"] == "task_cpu_11_8.py" and (KERNEL / meta["code_file"]).exists()
    assert (meta["enable_gpu"], meta["enable_internet"], meta["is_private"]) == (
        "false",
        "true",
        "true",
    )
    for key in ("dataset_sources", "competition_sources", "kernel_sources", "model_sources"):
        assert meta[key] == []


def test_command_matches_the_task_text() -> None:
    """The argument vector equals the 11.8 command and tasks.md states the same flags."""
    mod = _mod()
    cmd = mod.command(Path("/o"))
    assert cmd[1:3] == ["-m", "fintfm.experiments.term_structure"]
    assert cmd[3:] == [
        "--steps",
        "1200",
        "--seeds",
        "0",
        "1",
        "2",
        "--n-boot",
        "2000",
        "--out",
        "/o",
    ]
    tasks = (ROOT / "openspec/changes/pd-term-structure/tasks.md").read_text()
    assert "--steps 1200 --seeds 0 1 2 --n-boot 2000" in tasks
    assert mod.FINTFM_COMMIT == "e475151"
    assert mod.TIMEOUT_S < 12 * 3600
