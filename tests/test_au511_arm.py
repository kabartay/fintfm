"""AU5.11's prior-diversity arms: config files, kernel flags and the prior they build agree.

The kernels (``scripts/kaggle/ts_au511_*``) train the published recipe with the arm's prior as
CLI flags, like every ``make_train_score.py`` arm; ``configs/arms/au511-*.yaml`` states the same
arm as a file. A drift guard checks that both build one ``PriorConfig``, that the arm differs
from the published recipe in prior keys only, and that the new flags reach the generated tasks.
"""

import ast
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import yaml

from fintfm.config import load_config
from fintfm.modeling.train import build_configs, build_parser
from fintfm.prior.mixture import sample_batch

ROOT = Path(__file__).resolve().parents[1]
PUBLISHED = ROOT / "configs" / "published" / "v4-cellattn-labels.yaml"
NEW_FLAGS = {"heavy_tail_column_prob", "tree_split_by", "missing_label"}
ARMS = {"diverse": True, "mixture": False}  # arm -> new flags on


def _kernel_flags(arm: str) -> list[str]:
    """``ARM_FLAGS`` of the arm's kernel, read without importing it."""
    src = (ROOT / "scripts" / "kaggle" / f"ts_au511_{arm}_s0" / "kernel.py").read_text()
    for node in ast.parse(src).body:
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", "") == "ARM_FLAGS":
            return ast.literal_eval(node.value)
    raise AssertionError("no ARM_FLAGS")


@pytest.mark.parametrize("arm", ARMS)
def test_arm_file_differs_from_published_in_prior_keys_only(arm: str) -> None:
    """Model and train sections are the published recipe's, key for key."""
    a = yaml.safe_load((ROOT / "configs" / "arms" / f"au511-{arm}.yaml").read_text())
    ref = yaml.safe_load(PUBLISHED.read_text())
    assert a["model"] == ref["model"] and a["train"] == ref["train"]
    changed = {k for k in ref["prior"] if a["prior"][k] != ref["prior"][k]}
    assert {"p_financial", "p_tree", "p_task_family", "categorical_mode", "missing_mode",
            "feature_ratio_bands"} <= changed
    assert (NEW_FLAGS <= changed) is ARMS[arm]
    assert (changed & NEW_FLAGS) == (NEW_FLAGS if ARMS[arm] else set())


@pytest.mark.parametrize("arm", ARMS)
def test_kernel_flags_build_the_same_prior_and_model_as_the_file(arm: str) -> None:
    """Flags on the published recipe equal the arm file (model and prior)."""
    flags = _kernel_flags(arm)
    from_flags = build_configs(build_parser().parse_args(flags), load_config(PUBLISHED))
    arm_file = ROOT / "configs" / "arms" / f"au511-{arm}.yaml"
    from_file = build_configs(build_parser().parse_args([]), load_config(arm_file))
    # attn_impl is an execution flag of the kernel (split attention), not part of the arm
    model_flags = {**asdict(from_flags[0]), "attn_impl": "masked"}
    assert model_flags == asdict(from_file[0])
    assert asdict(from_flags[1]) == asdict(from_file[1])
    assert from_flags[1].heavy_tail_column_prob == (0.15 if ARMS[arm] else 0.0)
    # 48,000 tasks under DDP on two GPUs, as every screening arm
    batch, steps = int(flags[flags.index("--batch-size") + 1]), int(flags[flags.index("--steps") + 1])
    assert batch * 2 * steps == 48_000


@pytest.mark.parametrize("arm", ARMS)
def test_kernel_is_pinned_where_the_flags_exist_and_checkpoints(arm: str) -> None:
    """Pinned at 62a73c3, with ``--checkpoint-every`` in the training command."""
    src = (ROOT / "scripts" / "kaggle" / f"ts_au511_{arm}_s0" / "kernel.py").read_text()
    assert 'COMMIT = "62a73c3"' in src
    assert '"--checkpoint-every"' in src


def test_new_flags_reach_the_generated_tasks() -> None:
    """With the flags on, tasks differ from the flags-off mixture; the mixture arm equals off."""
    base = _kernel_flags("mixture")
    on = _kernel_flags("diverse")
    assert on[: len(base)] == base

    def batch(flags: list[str]) -> np.ndarray:
        cfg = build_configs(build_parser().parse_args(flags), load_config(PUBLISHED))[1]
        return sample_batch(np.random.default_rng(3), cfg, 4).X.numpy()

    x_off, x_on = batch(base), batch(on)
    assert x_off.size > 0 and x_on.size > 0
    assert x_off.shape != x_on.shape or not np.array_equal(
        np.nan_to_num(x_off), np.nan_to_num(x_on)
    )
