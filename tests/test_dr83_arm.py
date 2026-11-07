"""DR8.3's 512-column arm: one recipe, three changes, and a prior that emits wide tasks.

The arm file is a full copy of the published recipe (``--config`` does not layer over it), so
a drift guard compares the two key by key. The kernel must refuse to run unpinned.
"""

import ast
import re
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


def test_kernel_is_unpinned_or_pinned_trains_only_and_checkpoints() -> None:
    """The kernel refuses to run until a pushed sha is set, and trains the same 48,000 tasks.

    ``COMMIT`` is the placeholder before pinning and a hex sha after; the refusal guard stays.
    """
    src = KERNEL.read_text()
    tree = ast.parse(src)
    consts = {
        t.targets[0].id: ast.literal_eval(t.value)
        for t in tree.body
        if isinstance(t, (ast.Assign, ast.AnnAssign))
        for t in [t if isinstance(t, ast.Assign) else ast.Assign(targets=[t.target], value=t.value)]
        if isinstance(t.targets[0], ast.Name) and isinstance(t.value, (ast.Constant, ast.List))
    }
    assert consts["COMMIT"] == "UNPINNED" or re.fullmatch(r"[0-9a-f]{7,40}", consts["COMMIT"])
    assert 'if COMMIT == "UNPINNED"' in src
    assert '"--checkpoint-every"' in src
    assert consts["SCORE_V4"] is False
    flags = consts["ARM_FLAGS"]
    batch, steps = int(flags[flags.index("--batch-size") + 1]), int(flags[flags.index("--steps") + 1])
    assert batch * 2 * steps == 48_000  # two GPUs under DDP
    assert "configs/arms/dr83-width512.yaml" in src


MC_ARM = ROOT / "configs" / "arms" / "dr75-width512-mc.yaml"
MC_KERNEL = ROOT / "scripts" / "kaggle" / "ts_dr75_width512_mc_s0" / "kernel.py"

#: DR7.4's multiclass settings (FINDINGS 185), the only keys the multiclass arm adds to DR8.3's.
MC_CHANGED = {
    ("model", "max_classes"),
    ("prior", "p_financial"),
    ("prior", "scm_legacy"),
    ("prior", "permute_classes"),
}


def test_multiclass_arm_is_dr83_plus_dr74_multiclass_settings() -> None:
    """DR7.5's arm equals DR8.3's except DR7.4's four multiclass keys, with their values."""
    mc, wide = yaml.safe_load(MC_ARM.read_text()), yaml.safe_load(ARM.read_text())
    assert mc.keys() == wide.keys()
    differing = {
        (sec, k) for sec in wide for k in set(wide[sec]) | set(mc[sec])
        if mc[sec].get(k) != wide[sec].get(k)
    }
    assert differing == MC_CHANGED
    assert mc["model"]["max_classes"] == 10 and mc["model"]["max_features"] == 512
    assert mc["prior"]["p_financial"] == 0.5
    assert mc["prior"]["scm_legacy"] is True and mc["prior"]["permute_classes"] is True


def test_multiclass_arm_prior_emits_multiclass_tasks_wider_than_136_columns() -> None:
    """The prior reaches past the reference's 136 columns, within 512, with more than 2 classes."""
    cfg = load_config(MC_ARM)
    prior = build_prior_config(build_parser().parse_args([]), cfg)
    rng = np.random.default_rng(0)
    widths, classes = [], []
    for _ in range(40):
        batch = sample_batch(rng, prior, 2)
        assert batch.X.shape[2] <= 512
        widths.append(batch.X.shape[2])
        classes.append(int(batch.y.max()) + 1)
    assert max(widths) > 136
    assert max(classes) > 2


def test_multiclass_kernel_is_unpinned_or_pinned_trains_only() -> None:
    """Same guard and volume as DR8.3's kernel, pointing at the multiclass arm."""
    src = MC_KERNEL.read_text()
    tree = ast.parse(src)
    consts = {
        t.targets[0].id: ast.literal_eval(t.value)
        for t in tree.body
        if isinstance(t, (ast.Assign, ast.AnnAssign))
        for t in [t if isinstance(t, ast.Assign) else ast.Assign(targets=[t.target], value=t.value)]
        if isinstance(t.targets[0], ast.Name) and isinstance(t.value, (ast.Constant, ast.List))
    }
    assert consts["COMMIT"] == "UNPINNED" or re.fullmatch(r"[0-9a-f]{7,40}", consts["COMMIT"])
    assert 'if COMMIT == "UNPINNED"' in src
    assert '"--checkpoint-every"' in src
    assert consts["SCORE_V4"] is False
    flags = consts["ARM_FLAGS"]
    batch, steps = int(flags[flags.index("--batch-size") + 1]), int(flags[flags.index("--steps") + 1])
    assert batch * 2 * steps == 48_000
    assert "configs/arms/dr75-width512-mc.yaml" in src
