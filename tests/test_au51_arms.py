"""AU5.1's arm configs and Kaggle kernels: three learning rates, one setting otherwise.

The kernels embed their arm's YAML, so a config edited without regenerating the kernels would
run something other than what was pre-registered; the equality test below catches that.
"""

import ast
import importlib.util
import json
from pathlib import Path

import pytest

from fintfm.config import load_config

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "scripts" / "kaggle" / "make_au51_kernels.py"


def _gen():
    spec = importlib.util.spec_from_file_location("make_au51_kernels", GEN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("arm,lr", [("lr1e5", 1e-5), ("lr3e5", 3e-5), ("lr1e4", 1e-4)])
def test_arm_config_differs_from_the_others_only_in_the_learning_rate(arm, lr):
    cfg = load_config(ROOT / "configs" / "arms" / f"au51-{arm}.yaml").inference
    assert cfg.finetune_lr == lr
    assert cfg.finetune_steps == 3000
    assert cfg.finetune_val_frac == 0.1  # inner-validation companies, never the test fold
    assert (cfg.finetune_batch, cfg.finetune_query, cfg.finetune_context) == (2, 512, None)
    assert (cfg.finetune_eval_every, cfg.finetune_patience) == (10, 16)  # Rubachev et al. 2025


def test_committed_kernels_match_the_generator_and_cover_five_folds_per_arm():
    gen = _gen()
    for arm in gen.ARMS:
        folds: list[int] = []
        for group, group_folds in gen.FOLD_GROUPS.items():
            d = gen.kernel_dir(arm, group)
            src, meta = gen.render(arm, group, gen.PLACEHOLDER)
            ast.parse(src)
            # the committed kernel is unpinned or pinned to a sha; either way only COMMIT differs
            committed = (d / meta["code_file"]).read_text()
            line = next(x for x in committed.splitlines() if x.startswith("COMMIT = "))
            pinned = gen.render(arm, group, line.split('"')[1])[0]
            assert committed == pinned, f"{d} drifted from the generator or its config"
            assert json.loads((d / "kernel-metadata.json").read_text()) == meta
            folds += list(group_folds)
        assert sorted(folds) == [0, 1, 2, 3, 4]


def test_unpinned_kernel_refuses_to_run():
    gen = _gen()
    src, _ = gen.render("lr1e5", "f4", gen.PLACEHOLDER)
    assert 'COMMIT = "UNPINNED"' in src and 'if COMMIT == "UNPINNED"' in src
