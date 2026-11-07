"""48.25's head-only arm: the config file, the kernel's flags and the guards agree.

The kernel (``scripts/kaggle/ts_4825_headonly_s0``) trains the published recipe with the arm as
CLI flags; ``configs/arms/48.25-head-only.yaml`` states the same arm as a file.
"""

import ast
from dataclasses import asdict
from pathlib import Path

import yaml

from fintfm.config import load_config
from fintfm.modeling.train import build_configs, build_parser

ROOT = Path(__file__).resolve().parents[1]
PUBLISHED = ROOT / "configs" / "published" / "v4-cellattn-labels.yaml"
ARM_FILE = ROOT / "configs" / "arms" / "48.25-head-only.yaml"
KERNEL = ROOT / "scripts" / "kaggle" / "ts_4825_headonly_s0" / "kernel.py"


def _constant(name: str) -> object:
    """A module-level constant of the kernel, read without importing it."""
    for node in ast.parse(KERNEL.read_text()).body:
        if isinstance(node, ast.AnnAssign | ast.Assign):
            target = node.target if isinstance(node, ast.AnnAssign) else node.targets[0]
            if getattr(target, "id", "") == name and node.value is not None:
                return ast.literal_eval(node.value)
    raise AssertionError(name)


def test_arm_file_is_the_published_recipe_plus_the_two_train_keys() -> None:
    """Nothing but ``train_only`` and ``reinit_trained`` differs from the published recipe."""
    arm = yaml.safe_load(ARM_FILE.read_text())
    ref = yaml.safe_load(PUBLISHED.read_text())
    assert arm["model"] == ref["model"] and arm["prior"] == ref["prior"]
    extra = {k: v for k, v in arm["train"].items() if ref["train"].get(k, object()) != v}
    assert extra == {"train_only": "head", "reinit_trained": True}


def test_kernel_flags_build_the_arm_files_configs() -> None:
    """Flags on the published recipe equal the arm file, up to the kernel's execution flags."""
    flags = _constant("ARM_FLAGS")
    assert isinstance(flags, list)
    from_flags = build_configs(build_parser().parse_args(flags), load_config(PUBLISHED))
    from_file = build_configs(build_parser().parse_args([]), load_config(ARM_FILE))
    assert asdict(from_flags[0]) == asdict(from_file[0])
    assert asdict(from_flags[1]) == asdict(from_file[1])
    a, b = asdict(from_flags[2]), asdict(from_file[2])
    for execution in ("amp", "loader"):
        a.pop(execution), b.pop(execution)
    assert a == b


def test_kernel_is_pinned_and_checkpoints() -> None:
    """The commit is pinned (4f46460, with the arm's cell_mixing pins), the start is hash-pinned, saves are on."""
    assert _constant("COMMIT") == "4f46460"
    assert len(str(_constant("INIT_SHA256"))) == 64
    assert "--checkpoint-every" in KERNEL.read_text()
    assert 'if COMMIT == "UNPINNED"' in KERNEL.read_text()
