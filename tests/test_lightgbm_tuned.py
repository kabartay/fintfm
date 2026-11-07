"""Tests for the tuned LightGBM control. LightGBM runs in a subprocess that never imports torch
(this test process may have it loaded, and the two segfault together on macOS)."""

import json
import subprocess
import sys
import textwrap
from pathlib import Path

MODULE = Path(__file__).resolve().parents[1] / "src/fintfm/evaluation/lightgbm_tuned.py"

LOAD = textwrap.dedent('''
    import importlib.util, json, sys
    import numpy as np
    spec = importlib.util.spec_from_file_location("lt", sys.argv[1])
    lt = importlib.util.module_from_spec(spec)
    sys.modules["lt"] = lt
    spec.loader.exec_module(lt)
    assert "torch" not in sys.modules
''')

SNIPPET = LOAD + textwrap.dedent('''
    rng = np.random.RandomState(0)
    X = rng.randn(1200, 5)
    y = (X[:, 0] + 0.5 * rng.randn(1200) > 1.0).astype(int)
    t = np.arange(1200, dtype=float)
    # dated: validation is the latest rows; the fit side stops an embargo before it
    fit, val = lt.inner_split(y, time_=t, embargo=20.0)
    assert t[fit].max() < t[val].min() - 19 and not set(fit) & set(val)
    # undated: disjoint and stratified
    fit2, val2 = lt.inner_split(y, seed=1)
    assert not set(fit2) & set(val2) and abs(y[val2].mean() - y.mean()) < 0.02
    # grouped: no group on both sides
    g = rng.randint(0, 100, 1200)
    f3, v3 = lt.inner_split(y, groups=g, seed=2)
    assert not set(g[f3]) & set(g[v3])
    tuned = lt.tune_lightgbm(X, y, n_configs=4, seed=3, max_trees=60)
    again = lt.tune_lightgbm(X, y, n_configs=4, seed=3, max_trees=60)
    assert len(tuned["trials"]) == 4
    assert [round(x["val_ap"], 9) for x in tuned["trials"]] == \\
        [round(x["val_ap"], 9) for x in again["trials"]]
    assert tuned["val_ap"] == max(x["val_ap"] for x in tuned["trials"])
    p = lt.fit_predict_tuned(X[:900], y[:900], X[900:], tuned)
    assert p.shape == (300,) and ((p >= 0) & (p <= 1)).all()
    # a zero budget still runs the first config
    one = lt.tune_lightgbm(X, y, n_configs=5, time_budget_s=0.0, max_trees=30)
    assert len(one["trials"]) == 1
    # one-class side is refused
    try:
        lt.inner_split(np.zeros(50, dtype=int))
        raise SystemExit("no error")
    except ValueError:
        pass
    # grouped, undated: whole groups on one side, near the requested share of rows
    f4, v4 = lt.inner_split(y, groups=g, seed=4)
    assert not set(g[f4]) & set(g[v4]) and abs(len(v4) / 1200 - 0.25) < 0.05
    assert len(f4) + len(v4) == 1200
    # a starting score: the search and the refit both take it
    init = 2.0 * X[:, 0]
    tuned_i = lt.tune_lightgbm(X, y, n_configs=3, seed=3, max_trees=40, init_score=init)
    assert len(tuned_i["trials"]) == 3
    p_i = lt.fit_predict_tuned(X[:900], y[:900], X[900:], tuned_i, init_train=init[:900],
                               init_test=init[900:])
    assert p_i.shape == (300,) and ((p_i >= 0) & (p_i <= 1)).all()
    try:
        lt.fit_predict_tuned(X[:900], y[:900], X[900:], tuned_i, init_train=init[:900])
        raise SystemExit("no error")
    except ValueError:
        pass
    # the command-line entry takes a starting score and saves the predictions
    import os
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        npz, out, pr = (os.path.join(tmp, n) for n in ("d.npz", "o.json", "p.npy"))
        np.savez(npz, Xtr=X[:900], ytr=y[:900], Xte=X[900:], yte=y[900:],
                 btr=init[:900], bte=init[900:])
        assert lt.main([npz, out, "2", "30", "0", pr]) == 0
        assert np.load(pr).shape == (300,) and json.load(open(out))["n_trials"] == 2
    print(json.dumps({"ok": True}))
''')


def test_tuning_in_a_torch_free_process():
    r = subprocess.run([sys.executable, "-c", SNIPPET, str(MODULE)], capture_output=True,
                       text=True, check=False, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    assert json.loads(r.stdout.strip().splitlines()[-1])["ok"]
