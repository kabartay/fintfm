"""No-network checks of the prepared Kaggle kernel ``scripts/kaggle/task_cpu_13_se`` (13.7, 13.8)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parent.parent
KDIR = ROOT / "scripts" / "kaggle" / "task_cpu_13_se"


@pytest.fixture(scope="module")
def kernel() -> ModuleType:
    """Import the kernel script by path (it is not a package)."""
    spec = importlib.util.spec_from_file_location("task_cpu_13_se", KDIR / "task_cpu_13_se.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_metadata_is_cpu_private_and_uploads_nothing() -> None:
    """CPU, private, internet on, only the public V4FinBench dataset as an input."""
    meta = json.loads((KDIR / "kernel-metadata.json").read_text())
    assert meta["id"] == "muhakabartay/fintfm-task-cpu-13-se"
    assert meta["code_file"] == "task_cpu_13_se.py"
    assert meta["enable_gpu"] == "false" and meta["enable_internet"] == "true"
    assert meta["is_private"] == "true"
    assert meta["dataset_sources"] == ["sebastiantomczak10/v4-group-corporate-bankruptcy"]
    assert meta["kernel_sources"] == [] and meta["model_sources"] == []


def test_pins_match_the_published_checkpoint_and_a_real_commit(kernel: ModuleType) -> None:
    """Commit is a short sha and the checkpoint hash equals the one publish_weights records."""
    assert re.fullmatch(r"[0-9a-f]{7}", kernel.COMMIT)
    text = (ROOT / "scripts" / "publish_weights.py").read_text()
    assert kernel.CKPT_SHA256 in text
    assert "kabartay/fintfm-binary/resolve/" in kernel.CKPT_URL


def test_panel_commands_match_the_cli(kernel: ModuleType) -> None:
    """Each command carries the flags 13.7 and 13.8 name, and v4finbench needs its root."""
    ckpt = Path("/x/c.pt")
    tw = kernel.panel_command("taiwan", ckpt, None)
    assert tw[tw.index("--panel") + 1] == "taiwan"
    assert tw[tw.index("--calibration-frac") + 1] == "0.3"
    assert "--v4-pool-rows" not in tw
    v4 = kernel.panel_command("v4finbench", ckpt, Path("/in/v4"))
    assert v4[v4.index("--v4-pool-rows") + 1] == "200000"
    assert v4[v4.index("--v4-root") + 1] == "/in/v4"
    with pytest.raises(ValueError):
        kernel.panel_command("v4finbench", ckpt, None)
    with pytest.raises(ValueError):
        kernel.panel_command("polish3", ckpt, None)


def test_cli_declares_every_flag_the_kernel_passes() -> None:
    """The flags exist in prior_ablation (guards drift against the CLI)."""
    src = (ROOT / "src/fintfm/experiments/prior_ablation.py").read_text()
    for flag in (
        "--sample-efficiency",
        "--panel",
        "--calibration-frac",
        "--v4-pool-rows",
        "--v4-root",
        "--out",
    ):
        assert f'"{flag}"' in src


def test_check_record_flags_missing_arms(kernel: ModuleType, tmp_path: Path) -> None:
    """A complete record passes; one lacking the held-out arms is reported."""
    row = {a: {} for a in kernel.REQUIRED_ARMS}
    rec = {
        "panel": "taiwan",
        "split": "stratified_rows",
        "panel_rows": 1,
        "pool_rows": 1,
        "test_rows": 1,
        "calibration_frac": 0.3,
        "sizes": {"100": row},
    }
    p = tmp_path / "sample_efficiency.json"
    p.write_text(json.dumps(rec))
    assert kernel.check_record(p, "taiwan") == []
    del rec["sizes"]["100"]["gboost_platt_holdout"]
    p.write_text(json.dumps(rec))
    assert any("gboost_platt_holdout" in x for x in kernel.check_record(p, "taiwan"))
    assert kernel.check_record(tmp_path / "none.json", "taiwan")


def test_sha256_helper(kernel: ModuleType, tmp_path: Path) -> None:
    """The hash helper agrees with hashlib."""
    f = tmp_path / "a"
    f.write_bytes(b"abc")
    assert kernel.sha256(f) == hashlib.sha256(b"abc").hexdigest()
