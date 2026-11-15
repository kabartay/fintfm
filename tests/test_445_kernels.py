"""Task 44.4 and 44.5 kernels: pinned, matched, and the mixing flag is the only difference.

``ts_445_factorized_s0`` and ``ts_445_ctrl_s0`` are ``make_train_score.py`` arms; the cost
kernel ``task_444_cost`` is hand-written. Nothing here pushes or runs a kernel.
"""

import ast
import importlib.util
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
KAGGLE = ROOT / "scripts" / "kaggle"
PIN = "d58141e"
ARMS = {"factorized": "ts_445_factorized_s0", "ctrl": "ts_445_ctrl_s0"}


def _consts(path: Path) -> dict:
    """Module-level literal constants of a kernel, read without importing it."""
    out = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target, value = node.target.id, node.value
        elif isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            target, value = node.targets[0].id, node.value
        else:
            continue
        try:
            out[target] = ast.literal_eval(value)
        except ValueError:
            pass
    return out


@pytest.mark.parametrize("arm", ARMS)
def test_training_kernels_are_pinned_checkpoint_and_score(arm: str) -> None:
    """Pinned at the pushed commit, checkpointing, V4FinBench-scored, DDP, 48,000 tasks."""
    path = KAGGLE / ARMS[arm] / "kernel.py"
    consts = _consts(path)
    assert consts["COMMIT"] == PIN
    assert consts["SCORE_V4"] is True and consts["DDP"] is True
    assert consts["SWITCH_STEP"] == 0 and consts["SESSION"] == 0
    assert '"--checkpoint-every", "1000"' in path.read_text()
    flags = consts["ARM_FLAGS"]
    batch, steps = (
        int(flags[flags.index("--batch-size") + 1]),
        int(flags[flags.index("--steps") + 1]),
    )
    assert batch * 2 * steps == 48_000  # two ranks
    meta = json.loads((KAGGLE / ARMS[arm] / "kernel-metadata.json").read_text())
    assert meta["id"] == f"muhakabartay/fintfm-ts-445-{arm}-s0".replace("-s0-s0", "-s0")
    assert meta["dataset_sources"] == ["sebastiantomczak10/v4-group-corporate-bankruptcy"]
    assert meta["enable_gpu"] == "true"


def test_the_mixing_flag_is_the_only_flag_difference() -> None:
    """The control is the published recipe untouched; the arm adds ``--cell-mixing factorized``."""
    arm = _consts(KAGGLE / ARMS["factorized"] / "kernel.py")["ARM_FLAGS"]
    ctrl = _consts(KAGGLE / ARMS["ctrl"] / "kernel.py")["ARM_FLAGS"]
    assert arm[: len(ctrl)] == ctrl
    assert arm[len(ctrl) :] == ["--cell-mixing", "factorized"]
    assert "--cell-mixing" not in ctrl


def test_the_two_kernels_differ_only_in_arm_name_and_flags() -> None:
    """Everything else in the two kernel files is byte-identical."""
    a = (KAGGLE / ARMS["factorized"] / "kernel.py").read_text().splitlines()
    b = (KAGGLE / ARMS["ctrl"] / "kernel.py").read_text().splitlines()
    assert len(a) == len(b)
    assert all(x.startswith(("ARM = ", "ARM_FLAGS")) for x, y in zip(a, b) if x != y)


def _cost_module():
    spec = importlib.util.spec_from_file_location(
        "kernel_444_cost", KAGGLE / "task_444_cost" / "kernel.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_cost_kernel_is_pinned_and_unscored() -> None:
    """Pinned, no dataset, GPU on, and the recipe URL follows the pin."""
    path = KAGGLE / "task_444_cost" / "kernel.py"
    consts = _consts(path)
    assert consts["COMMIT"] == PIN
    meta = json.loads((KAGGLE / "task_444_cost" / "kernel-metadata.json").read_text())
    assert meta["dataset_sources"] == [] and meta["enable_gpu"] == "true"
    assert re.search(r"refusing a CPU run", path.read_text())


def test_cost_cells_pair_the_encoders_and_differ_only_in_the_mixing_flag() -> None:
    """Each (shape, rows, batch) cell appears once per encoder with identical commands."""
    mod = _cost_module()
    cells = mod.cells()
    rows = {c["rows"] for c in cells if c["shape"] == "published"}
    assert rows == {512, 1024, 2048, 4096}
    assert {c["batch"] for c in cells if c["shape"] == "medium5m"} == {2, 8}
    recipe = Path("recipe.yaml")
    by_key: dict = {}
    for c in cells:
        cmd = mod.command(recipe, c)
        i = cmd.index("--cell-mixing")
        by_key.setdefault((c["shape"], c["rows"], c["batch"]), {})[c["mixing"]] = (
            cmd[:i] + cmd[i + 2 :],
            cmd[i + 1],
        )
    for key, pair in by_key.items():
        assert set(pair) == {"attention", "factorized"}, key
        assert pair["attention"][0] == pair["factorized"][0], key
    # every command measures a few hundred or fewer steps, with fp16 split as the arms train
    for c in cells:
        cmd = mod.command(recipe, c)
        assert "--benchmark-steps" in cmd and "--ddp" not in cmd
        assert cmd[cmd.index("--amp") + 1] == "fp16"
