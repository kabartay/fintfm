"""34.2's arm config and Kaggle kernel: the weight is the only difference from the AU5.9 control.

The kernel is generated; the equality test fails when the committed kernel drifts from the
generator, and it may carry the placeholder or a short or full hex sha, nothing else.
"""

import ast
import importlib.util
import json
import re
from dataclasses import asdict
from pathlib import Path

from fintfm.config import load_config

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "scripts" / "kaggle" / "make_joint_kernels.py"
ARMS = ROOT / "configs" / "arms"


def _gen():
    spec = importlib.util.spec_from_file_location("make_joint_kernels", GEN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_arm_differs_from_the_au59_control_in_the_weight_alone():
    arm = load_config(ARMS / "joint-w1.yaml")
    ctrl = load_config(ARMS / "au59-hazard-ctrl.yaml")
    assert asdict(arm.model) == asdict(ctrl.model)
    assert asdict(arm.prior) == asdict(ctrl.prior)
    a, b = asdict(arm.train), asdict(ctrl.train)
    assert {k for k in a if a[k] != b[k]} == {"joint_classification_weight"}
    assert arm.model.n_horizons == 6 and arm.model.hazard_period_context is False
    assert arm.train.joint_classification_weight == _gen().DEFAULT_WEIGHT
    assert arm.train.steps * arm.train.batch_size == 48_000


def test_committed_kernel_matches_the_generator_and_is_unpinned_or_pinned_to_a_sha():
    gen = _gen()
    d = gen.kernel_dir()
    committed = (d / "kernel.py").read_text()
    match = re.search(r'^COMMIT = "([^"]+)"$', committed, flags=re.MULTILINE)
    assert match, "no COMMIT line"
    commit = match.group(1)
    assert commit == gen.PLACEHOLDER or re.fullmatch(r"[0-9a-f]{7,40}", commit), commit
    src, meta = gen.render(commit)
    assert committed == src
    assert json.loads((d / "kernel-metadata.json").read_text()) == meta
    ast.parse(src)


def test_kernel_flags_carry_checkpoints_horizons_weight_and_the_recipe_url():
    gen = _gen()
    src, _ = gen.render("abc1234")
    assert '"--checkpoint-every", "1000"' in src and "'--n-horizons', '6'" in src
    assert "'--joint-classification-weight', '1.0'" in src and "--ddp" in src
    assert "configs/arms/joint-w1.yaml" in src
    assert gen.arm_name(0.3) == "joint-w0p3-s0" and gen.arm_name(1.0) == "joint-w1-s0"
    unpinned, _ = gen.render(gen.PLACEHOLDER)
    assert f'COMMIT = "{gen.PLACEHOLDER}"' in unpinned
