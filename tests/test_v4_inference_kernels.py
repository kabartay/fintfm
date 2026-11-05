"""The V4FinBench inference kernels embed their arm's YAML; it must match the config file.

A config edited without regenerating the kernels would run something other than what the
pre-registration (`docs/results/drafts/v4_inference_kernels_2026-11.md`) names.
"""

import ast
import importlib.util
from pathlib import Path

import pytest

from fintfm.config import load_config

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "scripts" / "kaggle" / "make_v4_inference_kernels.py"


def _gen():
    spec = importlib.util.spec_from_file_location("make_v4_inference_kernels", GEN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_committed_kernels_match_the_generator_and_cover_five_folds_per_arm():
    gen = _gen()
    for arm, groups in gen.GROUPS.items():
        folds: list[int] = []
        for group in groups:
            src, meta = gen.render(arm, group, gen.DEFAULT_COMMIT)
            d = gen.kernel_dir(arm, group)
            assert (d / meta["code_file"]).read_text() == src, f"{d.name} is stale"
            ast.parse(src)
            folds += list(groups[group])
        assert sorted(folds) == [0, 1, 2, 3, 4], arm


@pytest.mark.parametrize("arm", ["tfsq8", "ctx900e3", "au42proto", "pc16", "boost", "maj4000", "uni4000"])
def test_embedded_config_loads_and_carries_the_registered_setting(arm, tmp_path):
    gen = _gen()
    p = tmp_path / "arm.yaml"
    p.write_text(gen.arm_config(arm))
    cfg = load_config(p)
    assert (cfg.inference.query_chunk, cfg.inference.feature_chunk) == (512, 8)
    if arm == "tfsq8":
        assert cfg.inference.n_ensemble == 8
        assert list(cfg.inference.ensemble_transforms) == ["rank", "squash"]
    elif arm == "ctx900e3":
        assert (cfg.inference.max_context, cfg.inference.n_ensemble) == (900, 3)
    elif arm == "au42proto":
        assert cfg.inference.context_strategy == "prototype"
    elif arm == "pc16":
        assert (cfg.inference.context_strategy, cfg.inference.retrieval_per_class) == (
            "retrieval", 16)
    elif arm == "maj4000":
        assert (cfg.inference.max_context, cfg.inference.context_positive_floor) == (4000, 1000000)
    elif arm == "uni4000":
        assert (cfg.inference.max_context, cfg.inference.context_positive_floor) == (4000, None)
    else:
        assert (cfg.large_n.method, cfg.large_n.boost_pool_rows) == ("boost", 100000)
