"""DR6.4's arm check, §175 rule and composition rule behave as written, on synthetic outputs.

No real scoring: the "runs" are tiny arrays in a temporary directory. The kernels read are the
real `scripts/kaggle/ts_*` ones, so the expected flags come from what is queued.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import dr64_arm_verdict as d

ARM = "au517-lenscale-s0"
CFG = {"horizon": 0, "folds": [0, 1, 2, 3, 4], "max_rows": None, "n_features": 130,
       "fold_seed": 42, "n_splits": 5, "boosting": [], "classical": ["logistic_regression"],
       "tuned": False, "finetune_steps": 0, "n_ensemble": 1, "drop_columns": [],
       "large_n": {"method": "none"}}


def _write_scores(out: Path, shift: float) -> None:
    """Five folds of 400 rows; `shift` raises the arm's score on positives."""
    out.mkdir(parents=True)
    for k in range(5):
        r = np.random.default_rng(k)
        y = (r.random(400) < 0.1).astype(int)
        pred = r.random(400) + y * 0.5 + shift * y * r.random(400)
        np.savez(out / f"predictions_fold{k}.npz", y_true=y, pred_fintfm=pred,
                 company_test=np.arange(400) // 4)
    (out / "v4_protocol.json").write_text(json.dumps({"dataset": "v4finbench", "config": CFG}))


@pytest.fixture
def runs(tmp_path: Path) -> Path:
    for s in (0, 1):
        c = tmp_path / f"kaggle-ts-e1-v48-s{s}" / f"e1-v48-s{s}"
        _write_scores(c / "score", 0.0)
        (c / "score.yaml").write_text("inference:\n  query_chunk: 512\n")
    return tmp_path


def _fetched(runs: Path, ok: bool = True) -> Path:
    out = runs / f"kaggle-fintfm-ts-{ARM}"
    arm = out / ARM
    arm.mkdir(parents=True)
    kernel = d.read_kernel(ARM)
    (arm / f"{ARM}.pt").write_bytes(b"x")
    (arm / f"{ARM}.pt.step1000").write_bytes(b"x")
    (arm / "train.log").write_text(
        "model parameters: 885,650\nddp: 2 processes, 8 tasks per step\nstep 6000/6000  loss 0.1\n")
    gpu = "GPUs: Tesla T4\n" if ok else "nothing\n"
    (out / "k.log").write_text(gpu + "$ " + " ".join(kernel["flags"]) + "\n")
    return out


def test_thresholds_are_section_175s():
    assert d.YARDSTICK == 0.0117 and d.MIN_FOLDS == 4


def test_judge_boundaries():
    assert d.judge([0.02] * 5) == "promising"
    assert d.judge([0.0117] * 5) == "null"  # the boundary is not past it
    assert d.judge([0.05, 0.05, 0.05, -0.01, -0.01]) == "null"  # only 3 positive
    assert d.judge([0.02, 0.02, 0.02, 0.02, -0.1]) == "null"  # 4 positive, mean too low
    assert d.judge([-0.02] * 5) == "harmful"
    assert d.judge([-0.05, -0.05, -0.05, 0.01, 0.01]) == "null"


def test_lever_flags_drop_execution_flags():
    k = d.read_kernel(ARM)
    assert d.lever_flags(k["flags"]) == ["--attn-length-scale"]
    assert d.lever_flags(d.read_kernel("au54-ctx2k-s0")["flags"]) == [
        "--n-rows-choices", "256,512,1024,2223", "--max-query-rows", "512"]
    assert d.lever_of("au517-lenscale-s1") == "au517-lenscale"


def test_prepare_passes_and_writes_the_control_command(runs):
    out = _fetched(runs)
    assert d.main(["prepare", str(out), "--runs", str(runs)]) == 0
    cmd = (out / "dr64_score_command.sh").read_text()
    for part in ("fintfm-v4protocol", "--horizon 0", "--folds 0,1,2,3,4", "--no-boosting",
                 "--classical logistic_regression", "--device cpu", "score.yaml"):
        assert part in cmd
    assert f"{ARM}.pt " in cmd and ".step1000" not in cmd


def test_prepare_fails_without_gpu_line_or_last_step(runs):
    out = _fetched(runs, ok=False)
    assert d.main(["prepare", str(out), "--runs", str(runs)]) == 1
    out2 = runs / "other"
    (out2 / ARM).mkdir(parents=True)
    (out2 / ARM / f"{ARM}.pt").write_bytes(b"x")
    (out2 / ARM / "train.log").write_text("model parameters: 1\nstep 10/6000\n")
    (out2 / "k.log").write_text("CUDA available: False\n")
    fails = d.check_logs(out2, out2 / ARM / f"{ARM}.pt", d.read_kernel(ARM))
    assert any("CUDA available: False" in f for f in fails)
    assert any("step 6000/6000" in f for f in fails)
    assert any("ddp:" in f for f in fails)


def test_two_checkpoints_refuse(runs):
    out = _fetched(runs)
    (out / "extra.pt").write_bytes(b"x")
    with pytest.raises(SystemExit):
        d.find_checkpoint(out)


def test_verdict_writes_file_and_applies_rule(runs):
    out = _fetched(runs)
    _write_scores(out / ARM / "score", 0.8)
    assert d.main(["verdict", str(out), "--runs", str(runs), "--n-boot", "50"]) == 0
    rec = json.loads((out / "dr64_verdict.json").read_text())
    assert rec["arm"] == ARM and rec["control"] == "e1-v48-s0" and rec["seed"] == 0
    assert rec["verdict"] == d.judge([f["diff"] for f in rec["folds"]])
    assert rec["folds_positive"] == 5


def test_verdict_refuses_without_scores(runs):
    out = _fetched(runs)
    with pytest.raises(SystemExit):
        d.main(["verdict", str(out), "--runs", str(runs)])


def _v(lever, seed, verdict):
    return {"lever": lever, "seed": seed, "verdict": verdict}


def test_compose_needs_both_seeds_and_harm_at_neither():
    res = d.compose([_v("a", 0, "promising"), _v("a", 1, "promising"),
                     _v("b", 0, "promising"), _v("b", 1, "null"),
                     _v("c", 0, "promising"), _v("c", 1, "harmful"),
                     _v("e", 0, "null")])
    assert res["outcome"] == "single-lever" and res["bundle"] == ["a"]
    assert res["levers"]["b"]["status"] == "excluded"
    assert res["levers"]["c"]["reason"].startswith("harmful")
    assert res["levers"]["e"]["status"] == "excluded"


def test_compose_bundle_none_and_wait():
    two = [_v(x, s, "promising") for x in "ab" for s in (0, 1)]
    assert d.compose(two)["outcome"] == "bundle"
    assert d.compose([_v("a", 0, "null")])["outcome"] == "none"
    assert d.compose([_v("a", 0, "promising")])["outcome"] == "wait"
    assert d.compose([_v("a", 0, "null")], expect=["z"])["outcome"] == "wait"
    assert d.compose(two, not_composable=("a",))["outcome"] == "single-lever"
    assert d.compose([_v("au59-hazard-ctrl", 0, "promising")])["outcome"] == "none"
