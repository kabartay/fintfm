"""``fintfm-bench --temporal-gbm`` runs the falsification script in a subprocess (openspec 2.2).

SMOKE-TEST: synthetic panels only; the end-to-end test needs LightGBM.
"""

from __future__ import annotations

import sys

import pytest

from fintfm.evaluation import bench


def test_command_is_built_from_the_package_path_and_current_interpreter():
    cmd = bench.temporal_gbm_command("/data/v4", 5000, 3, "/tmp/o.json")
    assert cmd[0] == sys.executable
    assert cmd[1] == str(bench.TEMPORAL_GBM_SCRIPT)
    assert bench.TEMPORAL_GBM_SCRIPT.name == "temporal_gbm_falsification.py"
    assert bench.TEMPORAL_GBM_SCRIPT.exists()
    assert cmd[2:] == [
        "--seed",
        "3",
        "--root",
        "/data/v4",
        "--max-rows",
        "5000",
        "--out",
        "/tmp/o.json",
    ]
    assert bench.temporal_gbm_command()[2:] == ["--seed", "0"]


def test_cli_flag_dispatches_with_its_options(monkeypatch):
    seen = {}

    def fake(v4_root, max_rows, seed):
        seen.update(v4_root=v4_root, max_rows=max_rows, seed=seed)
        return 0

    monkeypatch.setattr(bench, "run_temporal_gbm", fake)
    monkeypatch.setattr(
        sys, "argv", ["bench", "--temporal-gbm", "--v4-root", "d", "--max-rows", "9", "--seed", "2"]
    )
    with pytest.raises(SystemExit) as exc:
        bench.main()
    assert exc.value.code == 0
    assert seen == {"v4_root": "d", "max_rows": 9, "seed": 2}


def test_format_prints_both_arms_per_fold():
    arm = {"average_precision": 0.25, "roc_auc": 0.75}
    text = bench.format_temporal_gbm(
        {
            "n_rows": 10,
            "n_companies": 4,
            "default_rate": 0.1,
            "folds": [{"fold": 0, "n_test": 5, "levels": arm, "levels+traj": arm}],
        }
    )
    assert "AP levels" in text and "AUC +traj" in text and "0.2500" in text and "0.7500" in text


def test_end_to_end_on_a_synthetic_panel(monkeypatch, capsys):
    pytest.importorskip("lightgbm")
    real = bench.subprocess.Popen

    def tiny(cmd, **kw):
        # Same script module and same run(), on a tiny synthetic panel instead of V4FinBench.
        code = (
            "import importlib.util, sys, json, numpy as np\n"
            f"s = importlib.util.spec_from_file_location('tg', {str(bench.TEMPORAL_GBM_SCRIPT)!r})\n"
            "tg = importlib.util.module_from_spec(s); sys.modules['tg'] = tg\n"
            "s.loader.exec_module(tg)\n"
            "rng = np.random.default_rng(0)\n"
            "g, yr = np.meshgrid(np.arange(400), np.arange(2010, 2016), indexing='ij')\n"
            "g, yr = g.ravel(), yr.ravel()\n"
            "X = rng.normal(size=(len(g), 3)).astype(np.float32)\n"
            "y = (X[:, 0] + rng.normal(size=len(g)) > 1.2).astype(int)\n"
            "r = tg.run(X, y, g, yr, n_boot=10, params={'n_estimators': 20})\n"
            "open(sys.argv[sys.argv.index('--out') + 1], 'w').write(json.dumps(r))\n"
            "for f in r['folds']: print(f\"fold {f['fold']}: streamed\")\n"
            "print('{ignored json')\n"
        )
        return real([sys.executable, "-c", code, *cmd[2:]], **kw)

    monkeypatch.setattr(bench.subprocess, "Popen", tiny)
    assert bench.run_temporal_gbm(None, 100, 0) == 0
    out = capsys.readouterr().out
    assert "fold 0: streamed" in out and "ignored json" not in out
    assert "AP levels" in out and "AP +traj" in out and "AUC levels" in out
    assert "TEMPORAL GBM" in out and out.count("\n") >= 6
