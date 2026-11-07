"""DR3.11's three partition arms and their Kaggle kernels.

Each kernel embeds its arm's YAML, so a config edited without regenerating the kernels would run
something other than what was pre-registered; the equality tests below catch that. The arms
differ from §169's partition (``method: partition``, depth 2, one tree, one member) in one key.
"""

import ast
import importlib.util
import json
from pathlib import Path

import pytest
import yaml

from fintfm.config import load_config
from fintfm.inference.large_n import PartitionedTFMClassifier, wrap_large_n

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "scripts" / "kaggle" / "make_dr311_kernels.py"
ARMS = ROOT / "configs" / "arms"


def _gen():
    spec = importlib.util.spec_from_file_location("make_dr311_kernels", GEN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("arm,bags,members", [("bag4", 4, 1), ("bag8", 8, 1), ("ens8", 1, 8)])
def test_arm_changes_one_setting_of_the_partition(arm, bags, members):
    cfg = load_config(ARMS / f"dr311-{arm}.yaml")
    assert cfg.large_n.method == "partition"
    assert (cfg.large_n.partition_n_bags, cfg.inference.n_ensemble) == (bags, members)
    assert cfg.large_n.partition_max_depth == 2  # §169's depth
    assert cfg.large_n.partition_min_samples_leaf == 0.05
    assert cfg.inference.finetune_steps == 0  # fine-tuning per leaf is §208, measured null
    assert cfg.large_n.partition_bag_frac == 0.6


@pytest.mark.parametrize("arm", ["bag4", "bag8", "ens8"])
def test_arm_config_wraps_the_classifier_with_its_own_settings(arm):
    from fintfm.inference.classifier import FinancialTFMClassifier

    cfg = load_config(ARMS / f"dr311-{arm}.yaml")
    base = FinancialTFMClassifier(
        None, n_ensemble=cfg.inference.n_ensemble, max_context=cfg.inference.max_context
    )
    part = wrap_large_n(base, cfg.large_n)
    assert isinstance(part, PartitionedTFMClassifier)
    assert part.n_bags == cfg.large_n.partition_n_bags
    assert part.max_depth == 2 and part.estimator.n_ensemble == cfg.inference.n_ensemble


def test_committed_kernels_match_the_generator_and_cover_five_folds_per_arm():
    gen = _gen()
    for arm, groups in gen.ARM_GROUPS.items():
        folds: list[int] = []
        for group, group_folds in groups.items():
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


def test_embedded_config_is_the_arm_file_plus_the_identity_preserving_chunks():
    gen = _gen()
    for arm in gen.ARM_GROUPS:
        arm_file = yaml.safe_load((ARMS / f"dr311-{arm}.yaml").read_text())
        embedded = yaml.safe_load(gen.config_text(arm))
        assert embedded["inference"].pop("query_chunk") == 512
        assert embedded["inference"].pop("feature_chunk") == 8
        arm_file.setdefault("inference", {})
        assert embedded == arm_file  # nothing else changed, and ens8's n_ensemble survived


def test_unpinned_kernel_refuses_to_run_and_refuses_a_cpu_run():
    gen = _gen()
    src, _ = gen.render("bag4", "f23", gen.PLACEHOLDER)
    assert 'COMMIT = "UNPINNED"' in src and 'if COMMIT == "UNPINNED"' in src
    assert "torch.cuda.is_available()" in src and "refusing a CPU run" in src
