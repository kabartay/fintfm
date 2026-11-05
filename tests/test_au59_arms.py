"""AU5.9's arm configs and Kaggle kernels: the flag is the only difference, and kernels are pinned.

The kernels are generated; the equality test fails when a committed kernel drifts from the
generator, and the placeholder test fails if one is committed already pinned to a commit that was
not pushed (the pin is set by hand after the push).
"""

import ast
import importlib.util
import json
import re
from dataclasses import asdict
from pathlib import Path

from fintfm.config import load_config

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "scripts" / "kaggle" / "make_au59_kernels.py"
ARMS = ROOT / "configs" / "arms"


def _gen():
    spec = importlib.util.spec_from_file_location("make_au59_kernels", GEN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_arm_and_control_differ_only_in_the_flag_and_from_the_recipe_in_two_keys():
    ctx = load_config(ARMS / "au59-hazard-period-context.yaml")
    ctrl = load_config(ARMS / "au59-hazard-ctrl.yaml")
    pub = load_config(ROOT / "configs" / "published" / "v4-cellattn-labels.yaml")
    a, b, p = asdict(ctx.model), asdict(ctrl.model), asdict(pub.model)
    assert {k for k in a if a[k] != b[k]} == {"hazard_period_context"}
    assert {k for k in b if b[k] != p[k]} == {"n_horizons"}
    assert a["n_horizons"] == 6 and a["hazard_period_context"] is True
    assert asdict(ctx.train) == asdict(pub.train) and asdict(ctx.prior) == asdict(pub.prior)
    assert ctx.train.steps * ctx.train.batch_size == 48_000


def test_committed_kernels_match_the_generator_and_are_unpinned_or_pinned_to_a_sha():
    """Each committed kernel is the generator's output at its own ``COMMIT``.

    That value is the placeholder until the kernel is pinned with ``--commit <pushed sha>``,
    and then a short or full hex sha; anything else would install fintfm from an unknown ref.
    """
    gen = _gen()
    for key in gen.ARMS:
        d = gen.kernel_dir(key)
        committed = (d / "kernel.py").read_text()
        match = re.search(r'^COMMIT = "([^"]+)"$', committed, flags=re.MULTILINE)
        assert match, f"{key}: no COMMIT line"
        commit = match.group(1)
        assert commit == gen.PLACEHOLDER or re.fullmatch(r"[0-9a-f]{7,40}", commit), commit
        src, meta = gen.render(key, commit)
        assert committed == src
        assert json.loads((d / "kernel-metadata.json").read_text()) == meta
        ast.parse(src)


def test_kernel_flags_carry_checkpoints_horizons_and_only_the_arm_has_the_flag():
    gen = _gen()
    ctx, _ = gen.render("ctx", "abc1234")
    ctrl, _ = gen.render("ctrl", "abc1234")
    for src in (ctx, ctrl):
        assert '"--checkpoint-every", "1000"' in src and '"--n-horizons", "6"' in src
        assert "--ddp" in src
    assert "--hazard-period-context" in ctx and "--hazard-period-context" not in ctrl
