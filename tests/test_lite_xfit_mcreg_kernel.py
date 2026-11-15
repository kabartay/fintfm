"""Kernel ``task_lite_xfit_mcreg``: pins, settings equal to the local queue scripts, checkpoint hashes.

Nothing here pushes or runs a kernel.
"""

import importlib.util
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "scripts" / "kaggle" / "task_lite_xfit_mcreg"
LOCAL = Path("/Users/morganoko/proj/fintfm/runs/local_queue")


def _kernel():
    spec = importlib.util.spec_from_file_location(
        "task_lite_xfit_mcreg", DIR / "task_lite_xfit_mcreg.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_pins() -> None:
    k = _kernel()
    assert k.FINTFM_COMMIT == "03d0270"
    assert k.FORK_COMMIT == "551259a"
    assert k.FORK_BRANCH == "fintfm-reg-levers"


def test_lever_arms_differ_from_the_regression_arm_by_exactly_one_variable() -> None:
    k = _kernel()
    assert list(k.ARMS) == ["mc", "reg", "reg_log1p", "reg_quantile", "reg_blendcv", "reg_wide"]
    expected = {
        "reg_log1p": ("fintfm_lp_reg_xfit_log1p_kaggle", "FINTFM_REG_TARGET_TRANSFORM", "log1p"),
        "reg_quantile": (
            "fintfm_lp_reg_xfit_quantile_kaggle",
            "FINTFM_REG_TARGET_TRANSFORM",
            "quantile",
        ),
        "reg_blendcv": ("fintfm_lp_reg_xfit_blendcv_kaggle", "FINTFM_REG_BLEND", "cv"),
        "reg_wide": ("fintfm_lp_reg_xfit_wide_kaggle", "FINTFM_REG_ALPHAS", "wide"),
    }
    base = k.ARMS["reg"]
    names = {a["run_name"] for a in k.ARMS.values()}
    assert len(names) == len(k.ARMS)
    for key, (run, var, val) in expected.items():
        arm = k.ARMS[key]
        assert arm["run_name"] == run
        assert arm["env"] == {**base["env"], var: val}
        for field in ("datasets", "checkpoint_file", "checkpoint_sha256", "checkpoint_env"):
            assert arm[field] == base[field]
        assert key in k.ARM_ESTIMATE_S


def test_settings_equal_the_local_runs() -> None:
    k = _kernel()
    params = {"feature_transform": "squash", "max_features_strategy": "select"}
    mc, reg = k.ARMS["mc"], k.ARMS["reg"]
    assert mc["run_name"] == "fintfm_lp_mc_scr_xfit_kaggle"
    assert reg["run_name"] == "fintfm_lp_reg_xfit_kaggle"
    assert mc["env"] == {
        "FINTFM_MODEL": "FinTFM-LinProbe",
        "FINTFM_PROBLEM_TYPES": "multiclass",
        "FINTFM_ALLOW_MULTICLASS": "1",
        "FINTFM_PARTITION": '{"n_bags": 8}',
        "FINTFM_PARAMS": json.dumps(params),
        "FINTFM_PROBE": '{"scale": true, "cv": true, "raw": true}',
    }
    assert reg["env"] == {
        "FINTFM_MODEL": "FinTFM-LinProbe",
        "FINTFM_PROBLEM_TYPES": "regression",
        "FINTFM_ALLOW_REGRESSION": "1",
        "FINTFM_PARTITION": '{"n_bags": 4}',
        "FINTFM_PARAMS": json.dumps(params),
        "FINTFM_PROBE": '{"raw": true}',
    }
    assert len(mc["datasets"]) == 8 and len(reg["datasets"]) == 13
    assert len(set(mc["datasets"])) == 8 and len(set(reg["datasets"])) == 13


def test_checkpoint_hashes_and_sources() -> None:
    k = _kernel()
    assert (
        k.ARMS["mc"]["checkpoint_sha256"]
        == "14d5b4d561fbea6e35c7a30dc2644cb8f64f437c25d62da087ee960717cc8f4d"
    )
    assert (
        k.ARMS["reg"]["checkpoint_sha256"]
        == "31d5b3ef49e89bbe405fc85ab1d746a9fd6c3a1c5282e1770111902cc4af526c"
    )
    meta = json.loads((DIR / "kernel-metadata.json").read_text())
    assert sorted(meta["kernel_sources"]) == sorted({a["source_kernel"] for a in k.ARMS.values()})
    assert meta["code_file"] == "task_lite_xfit_mcreg.py" and meta["enable_gpu"] == "true"


def test_a_wrong_checkpoint_is_refused(tmp_path: Path) -> None:
    k = _kernel()
    bad = tmp_path / "dr74-mc-s0.pt"
    bad.write_bytes(b"not the checkpoint")
    with pytest.raises(SystemExit):
        k.verify_checkpoint("mc", bad)


def test_env_is_clean_and_device_is_passed(monkeypatch: pytest.MonkeyPatch) -> None:
    k = _kernel()
    monkeypatch.setenv("FINTFM_PROBE", "stale")
    monkeypatch.setenv("FINTFM_REG_BINS", "20")
    env = k.arm_env("reg", Path("/x/c.pt"), "cuda")
    assert env["FINTFM_PROBE"] == '{"raw": true}'
    assert "FINTFM_REG_BINS" not in env
    assert env["FINTFM_DEVICE"] == "cuda" and env["FINTFM_CHECKPOINT_REGRESSION"] == "/x/c.pt"
    assert env["FINTFM_SKIP_COMPARE"] == "1"


@pytest.mark.skipif(not (LOCAL / "lite_reg_probe.sh").exists(), reason="local queue scripts absent")
def test_regression_datasets_match_the_local_script() -> None:
    text = (LOCAL / "lite_reg_probe.sh").read_text()
    block = re.search(r'DATASETS="([^"]+)"', text).group(1).split()
    assert block == _kernel().ARMS["reg"]["datasets"]
