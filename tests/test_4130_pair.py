"""41.3's matched pair: the two Kaggle kernels differ by ``--task-token`` and nothing else.

The kernels (``scripts/kaggle/ts_4130_{token,ctrl}_s0``) train the published recipe for 48,000
tasks at one seed; the token arm adds the task token (task 41.2), the control does not.
"""

import ast
from dataclasses import asdict
from pathlib import Path

import pytest

from fintfm.config import load_config
from fintfm.modeling.train import build_configs, build_parser

ROOT = Path(__file__).resolve().parents[1]
PUBLISHED = ROOT / "configs" / "published" / "v4-cellattn-labels.yaml"
KERNELS = {
    arm: ROOT / "scripts" / "kaggle" / f"ts_4130_{arm}_s0" / "kernel.py"
    for arm in ("token", "ctrl")
}


def _constant(arm: str, name: str) -> object:
    """A module-level constant of the arm's kernel, read without importing it."""
    for node in ast.parse(KERNELS[arm].read_text()).body:
        if isinstance(node, ast.AnnAssign | ast.Assign):
            target = node.target if isinstance(node, ast.AnnAssign) else node.targets[0]
            if getattr(target, "id", "") == name and node.value is not None:
                return ast.literal_eval(node.value)
    raise AssertionError(name)


def test_flags_differ_only_by_task_token() -> None:
    """Same seed, steps, batch and execution flags; ``--task-token`` is the one difference."""
    token, ctrl = _constant("token", "ARM_FLAGS"), _constant("ctrl", "ARM_FLAGS")
    assert token == [*ctrl, "--task-token"]
    assert "--task-token" not in ctrl


def test_flags_build_matched_configs_and_48000_tasks() -> None:
    """The model configs differ in ``task_token`` alone; prior and train configs are equal."""
    built = {
        arm: build_configs(
            build_parser().parse_args(_constant(arm, "ARM_FLAGS")), load_config(PUBLISHED)
        )
        for arm in KERNELS
    }
    m_token, m_ctrl = asdict(built["token"][0]), asdict(built["ctrl"][0])
    assert m_token.pop("task_token") is True
    assert m_ctrl.pop("task_token") is False
    assert m_token == m_ctrl
    assert asdict(built["token"][1]) == asdict(built["ctrl"][1])
    assert asdict(built["token"][2]) == asdict(built["ctrl"][2])
    flags = _constant("token", "ARM_FLAGS")
    assert isinstance(flags, list)
    batch = int(flags[flags.index("--batch-size") + 1])
    steps = int(flags[flags.index("--steps") + 1])
    assert batch * 2 * steps == 48_000  # two DDP ranks


@pytest.mark.parametrize("arm", KERNELS)
def test_kernel_is_pinned_scores_v4_and_checkpoints(arm: str) -> None:
    """Pinned at d58141e, scored on V4FinBench, saves checkpoints."""
    assert _constant(arm, "COMMIT") == "d58141e"
    assert _constant(arm, "SCORE_V4") is True
    assert _constant(arm, "DDP") is True
    assert '"--checkpoint-every"' in KERNELS[arm].read_text()
    assert _constant(arm, "ARM") == f"4130-{arm}-s0"
