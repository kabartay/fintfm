"""Kernel ``task_cpu_32_ctx``: pins, the command equals the queued ``h-binctx`` line, hashes.

Nothing here pushes, launches or touches the network.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "scripts" / "kaggle" / "task_cpu_32_ctx"
QUEUE = Path("/Users/morganoko/proj/fintfm/runs/local_queue/queue.txt")
MANIFEST = Path("/Users/morganoko/proj/fintfm/runs/MANIFEST.json")


def _kernel():
    spec = importlib.util.spec_from_file_location("task_cpu_32_ctx", DIR / "task_cpu_32_ctx.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_metadata_is_a_private_cpu_script_with_no_uploaded_inputs() -> None:
    meta = json.loads((DIR / "kernel-metadata.json").read_text())
    assert meta["id"] == "muhakabartay/fintfm-task-cpu-32-ctx"
    assert meta["code_file"] == "task_cpu_32_ctx.py"
    assert (meta["enable_gpu"], meta["enable_internet"], meta["is_private"]) == (
        "false",
        "true",
        "true",
    )
    assert meta["kernel_sources"] == [] and meta["dataset_sources"] == []


def test_pins_and_checkpoint_source_is_the_public_hub_file() -> None:
    k = _kernel()
    assert k.COMMIT == "e475151"
    assert k.CKPT_URL.startswith("https://huggingface.co/kabartay/fintfm-binary/resolve/")
    assert k.CKPT_SHA256 == "d231b062483c9261aae4c503ad5359db295fc7206fa0183f51763e4953c518d2"
    assert k.TIMEOUT_S < 12 * 3600


def test_command_passes_only_model_and_out_like_the_queue_line() -> None:
    k = _kernel()
    cmd = k.sweep_command(
        Path("/kaggle/working/c.pt"), Path("/kaggle/working/binary-context-sweep")
    )
    assert cmd[0] == sys.executable
    assert cmd[1:3] == ["-m", "fintfm.experiments.binary_context_sweep"]
    assert cmd[3:] == [
        "--model",
        "/kaggle/working/c.pt",
        "--out",
        "/kaggle/working/binary-context-sweep",
    ]


def test_defaults_give_the_180_cells_the_task_states() -> None:
    from fintfm.experiments import binary_context_sweep as b

    n = len(b.PANEL_NAMES) * len(b.DEFAULT_SIZES) * len(b.DEFAULT_SEEDS) * len(b.DEFAULT_STRATEGIES)
    assert n == 180
    assert "prototype" in b.DEFAULT_STRATEGIES


def test_queue_line_matches_when_the_local_queue_is_present() -> None:
    if not QUEUE.exists():
        pytest.skip("local queue not on this machine")
    lines = [ln for ln in QUEUE.read_text().splitlines() if ln.startswith("h-binctx|")]
    if not lines:
        pytest.skip("h-binctx already left the queue")
    assert (
        "fintfm-binctx --model runs/v4-cellattn-labels.pt --out runs/binary-context-sweep"
        in lines[0]
    )
    assert "--n-boot" not in lines[0] and "--seeds" not in lines[0]


def test_checkpoint_hash_matches_the_manifest_when_present() -> None:
    if not MANIFEST.exists():
        pytest.skip("manifest not on this machine")
    rows = {c["path"]: c["sha256"] for c in json.loads(MANIFEST.read_text())["checkpoints"]}
    assert rows["runs/v4-cellattn-labels.pt"] == _kernel().CKPT_SHA256


def test_sha256_file(tmp_path: Path) -> None:
    f = tmp_path / "x"
    f.write_bytes(b"abc")
    assert _kernel().sha256_file(f) == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )
