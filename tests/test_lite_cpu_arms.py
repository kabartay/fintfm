"""The generated Kaggle CPU arm kernels match the template and carry one filled arm block each."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

KAGGLE = Path(__file__).resolve().parent.parent / "scripts" / "kaggle"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_generated_kernels_match_template() -> None:
    """Every kernel equals the rendered template and parses."""
    gen = _load(KAGGLE / "make_lite_cpu_arms.py", "make_lite_cpu_arms")
    for name, (run, problem, lever) in gen.ARMS.items():
        path = KAGGLE / f"task_lite_{name}" / f"task_lite_{name}.py"
        text = path.read_text()
        assert text == gen.render(run, problem, lever)
        ast.parse(text)


def test_arm_blocks_and_env() -> None:
    """Each kernel names its run, problem type and lever; cross-fitting is off; device is CPU."""
    gen = _load(KAGGLE / "make_lite_cpu_arms.py", "make_lite_cpu_arms2")
    for name, (run, problem, lever) in gen.ARMS.items():
        mod = _load(KAGGLE / f"task_lite_{name}" / f"task_lite_{name}.py", f"k_{name}")
        assert (mod.RUN_NAME, mod.PROBLEM_TYPE, mod.LEVER) == (run, problem, lever)
        env = mod.run_env(mod.PROBLEM[problem]["datasets"][0], Path("/x.pt"))
        assert env["FINTFM_PROBE_CROSSFIT_FOLDS"] == "0"
        assert env["FINTFM_DEVICE"] == "cpu"
        assert env["FINTFM_RUN_NAME"] == run
        for k, v in lever.items():
            assert env[k] == v
    assert len(mod.MC_DATASETS) == 8 and len(mod.REG_DATASETS) == 13
