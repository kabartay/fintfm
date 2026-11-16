"""No-network checks of the 35.4 Kaggle CPU kernel's configuration (``scripts/kaggle/task_cpu_35_4``)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KDIR = ROOT / "scripts" / "kaggle" / "task_cpu_35_4"


def _load():
    spec = importlib.util.spec_from_file_location("task_cpu_35_4", KDIR / "task_cpu_35_4.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_metadata_is_a_private_cpu_script() -> None:
    meta = json.loads((KDIR / "kernel-metadata.json").read_text())
    assert meta["id"] == "muhakabartay/fintfm-task-cpu-35-4"
    assert meta["code_file"] == "task_cpu_35_4.py"
    assert (meta["is_private"], meta["enable_gpu"], meta["enable_internet"]) == (
        "true",
        "false",
        "true",
    )
    # Nothing is uploaded: no datasets, models or kernel outputs are attached.
    for key in ("dataset_sources", "kernel_sources", "model_sources", "competition_sources"):
        assert meta[key] == []


def test_embedded_ids_match_the_repo_file_and_hash() -> None:
    k = _load()
    repo = (ROOT / "configs/tabbench/eligible_ids.json").read_text()
    assert k.IDS_JSON == repo
    assert hashlib.sha256(repo.encode()).hexdigest() == k.IDS_SHA256
    assert k.pinned_ids() == json.loads(repo)
    assert len(k.pinned_ids()) == 90


def test_checkpoint_pin_matches_the_v4_inference_kernels() -> None:
    k = _load()
    gen = (ROOT / "scripts/kaggle/make_v4_inference_kernels.py").read_text()
    assert f'CKPT_URL = "{k.CKPT_URL}"' in gen
    assert f'CKPT_SHA256 = "{k.CKPT_SHA256}"' in gen
    assert k.CKPT_URL.startswith("https://huggingface.co/kabartay/fintfm-binary/resolve/")
    assert re.fullmatch(r"[0-9a-f]{64}", k.CKPT_SHA256)


def test_command_is_the_documented_harness_and_budget_fits_a_session() -> None:
    k = _load()
    cmd = k.harness_command(Path("/c.pt"), Path("/i.json"))
    assert cmd[1:3] == ["-m", "fintfm.experiments.tabbench_protocol"]
    assert cmd[cmd.index("--model") + 1] == "/c.pt"
    assert cmd[cmd.index("--suite-size") + 1] == "189"
    assert "--aggregate-only" not in cmd
    assert k.BUDGET_S < 12 * 3600
    assert k.DATASET_TIMEOUT_S < k.BUDGET_S
    assert k.FINTFM_COMMIT == "e475151"


def test_no_local_upload_and_no_suite_code() -> None:
    src = (KDIR / "task_cpu_35_4.py").read_text()
    assert "kaggle datasets" not in src
    assert "tabbench @" not in src
    assert "pip install tabbench" not in src
