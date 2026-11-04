"""DR8.3's 512-column arm: one recipe, three changes, and a prior that emits wide tasks.

The arm file is a full copy of the published recipe (``--config`` does not layer over it), so
a drift guard compares the two key by key. The kernel must refuse to run unpinned.
"""

import ast
from pathlib import Path

import numpy as np
import yaml

from fintfm.config import load_config
from fintfm.modeling.train import build_parser, build_prior_config
from fintfm.prior.mixture import sample_batch

ROOT = Path(__file__).resolve().parents[1]
ARM = ROOT / "configs" / "arms" / "dr83-width512.yaml"
PUBLISHED = ROOT / "configs" / "published" / "v4-cellattn-labels.yaml"
KERNEL = ROOT / "scripts" / "kaggle" / "ts_dr83_width512_s0" / "kernel.py"

#: The only keys the arm may change, as (section, key).
CHANGED = {
    ("model", "max_features"),
    ("prior", "feature_ratio_bands"),
    ("prior", "pad_to_batch_width"),
}


def test_arm_differs_from_the_published_recipe_in_exactly_three_keys() -> None:
    """Every other key of every section is the published value."""
    arm, ref = yaml.safe_load(ARM.read_text()), yaml.safe_load(PUBLISHED.read_text())
    assert arm.keys() == ref.keys()
    differing = {
        (sec, k) for sec in ref for k in set(ref[sec]) | set(arm[sec])
        if arm[sec].get(k) != ref[sec].get(k)
    }
    assert differing == CHANGED


def test_arm_loads_and_sets_the_wide_model() -> None:
    """The arm builds a 512-column model config and keeps the reference architecture."""
    cfg = load_config(ARM)
    assert cfg.model.max_features == 512
    assert (cfg.model.d_cell, cfg.model.d_model, cfg.model.n_layers) == (48, 128, 4)
    assert cfg.prior.pad_to_batch_width is True
    assert len(cfg.prior.feature_ratio_bands) == 3


def test_arm_prior_emits_wide_and_narrow_tasks_within_the_cap() -> None:
    """Batches reach beyond 136 columns, stay within 512, and pad only to the widest task."""
    cfg = load_config(ARM)
    prior = build_prior_config(build_parser().parse_args([]), cfg)
    rng = np.random.default_rng(0)
    widths = []
    for _ in range(6):
        batch = sample_batch(rng, prior, 2)
        widths.append(batch.X.shape[2])
        assert batch.X.shape[2] <= 512
    assert min(widths) < 512  # not padded to the cap


def test_kernel_is_unpinned_trains_only_and_checkpoints() -> None:
    """The kernel refuses to run until a pushed sha is set, and trains the same 48,000 tasks."""
    src = KERNEL.read_text()
    tree = ast.parse(src)
    consts = {
        t.targets[0].id: ast.literal_eval(t.value)
        for t in tree.body
        if isinstance(t, (ast.Assign, ast.AnnAssign))
        for t in [t if isinstance(t, ast.Assign) else ast.Assign(targets=[t.target], value=t.value)]
        if isinstance(t.targets[0], ast.Name) and isinstance(t.value, (ast.Constant, ast.List))
    }
    assert consts["COMMIT"] == "UNPINNED"
    assert 'if COMMIT == "UNPINNED"' in src
    assert '"--checkpoint-every"' in src
    assert consts["SCORE_V4"] is False
    flags = consts["ARM_FLAGS"]
    batch, steps = int(flags[flags.index("--batch-size") + 1]), int(flags[flags.index("--steps") + 1])
    assert batch * 2 * steps == 48_000  # two GPUs under DDP
    assert "configs/arms/dr83-width512.yaml" in src
