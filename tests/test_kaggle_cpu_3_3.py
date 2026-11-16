"""No-network checks of the task 3.3 Kaggle CPU kernel: pins, command lines, deadline logic."""

import importlib.util
import json
from pathlib import Path

import pytest

DIR = Path(__file__).resolve().parents[1] / "scripts" / "kaggle" / "task_cpu_3_3"


@pytest.fixture(scope="module")
def kernel():
    """Import the kernel script as a module (it does nothing at import)."""
    spec = importlib.util.spec_from_file_location("task_cpu_3_3", DIR / "task_cpu_3_3.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_metadata_is_a_private_cpu_script_with_the_public_dataset_only() -> None:
    """CPU, internet on, no kernel or model sources, only the public V4FinBench dataset."""
    meta = json.loads((DIR / "kernel-metadata.json").read_text())
    assert meta["id"] == "muhakabartay/fintfm-task-cpu-3-3"
    assert meta["code_file"] == "task_cpu_3_3.py"
    assert meta["is_private"] == "true"
    assert meta["enable_gpu"] == "false" and meta["enable_internet"] == "true"
    assert meta["dataset_sources"] == ["sebastiantomczak10/v4-group-corporate-bankruptcy"]
    assert meta["kernel_sources"] == [] and meta["model_sources"] == []


def test_pins(kernel) -> None:
    """Pinned commit and the published checkpoint's digest."""
    assert kernel.COMMIT == "e475151"
    assert len(kernel.CKPT_SHA256) == 64
    assert "huggingface.co/kabartay/fintfm-binary/resolve/" in kernel.CKPT_URL


def test_verify_checkpoint_refuses_a_wrong_hash(kernel, tmp_path: Path) -> None:
    """A file with another digest stops the kernel; the right digest passes."""
    f = tmp_path / "x.pt"
    f.write_bytes(b"abc")
    with pytest.raises(SystemExit):
        kernel.verify_checkpoint(f)
    assert kernel.verify_checkpoint(f, kernel.sha256(f)) == f


def test_find_v4_root_needs_all_six_files(kernel, tmp_path: Path) -> None:
    """Missing horizon files are refused; a complete directory is returned."""
    d = tmp_path / "ds"
    d.mkdir()
    (d / "company_years_h1.parquet").write_bytes(b"")
    with pytest.raises(SystemExit):
        kernel.find_v4_root(tmp_path)
    for i in range(2, 7):
        (d / f"company_years_h{i}.parquet").write_bytes(b"")
    assert kernel.find_v4_root(tmp_path) == d


def test_commands_match_the_verify(kernel) -> None:
    """The credit unit is the Verify's command line; the others omit ``--credit``."""
    c = kernel.bench_command(Path("m.pt"), Path("d"), Path("o.json"), 100_000, True)
    assert c[1:3] == ["-m", "fintfm.evaluation.bench"]
    assert "--credit" in c and "--time-split" in c
    assert c[c.index("--max-rows") + 1] == "100000"
    assert "--credit" not in kernel.bench_command(
        Path("m.pt"), Path("d"), Path("o"), 200_000, False
    )


def test_deadline_logic_fits_twelve_hours(kernel) -> None:
    """Fallback and 200k units start only with time left; worst case stays under 12 h."""
    assert kernel.should_run_fallback(3600, False) and not kernel.should_run_fallback(3600, True)
    assert not kernel.should_run_fallback(kernel.BUDGET_S - 600, False)
    assert kernel.should_run_ts200k(kernel.CREDIT_TIMEOUT_S)
    assert not kernel.should_run_ts200k(kernel.BUDGET_S - 3600)
    assert kernel.BUDGET_S < 12 * 3600
