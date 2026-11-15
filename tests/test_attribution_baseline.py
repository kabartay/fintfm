"""SMOKE-TEST for the SHAP-over-GBM baseline (openspec 12.2): synthetic data only.

The module runs in a subprocess that never imports torch: LightGBM and torch each bundle an
OpenMP runtime and segfault together on macOS (same arrangement as ``test_lightgbm_tuned.py``).
LightGBM is in the ``bench`` extra, so a bare sync skips these tests rather than failing.
"""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytest.importorskip("lightgbm")

MODULE = Path(__file__).resolve().parents[1] / "src/fintfm/experiments/attribution_baseline.py"

LOAD = textwrap.dedent("""
    import importlib.util, sys
    import numpy as np
    spec = importlib.util.spec_from_file_location("ab", sys.argv[1])
    ab = importlib.util.module_from_spec(spec)
    sys.modules["ab"] = ab
    spec.loader.exec_module(ab)

    def task(n=1500, seed=0):
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(n, 6))
        logit = 2.0 * X[:, 0] + 1.0 * X[:, 1]  # columns 2-5 carry no signal
        y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
        return X, y
""")


def _run(body: str) -> None:
    code = LOAD + textwrap.dedent(body) + "\nassert 'torch' not in sys.modules\n"
    r = subprocess.run(
        [sys.executable, "-c", code, str(MODULE)],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert r.returncode == 0, r.stdout + r.stderr


def test_treeshap_rows_sum_to_margin() -> None:
    _run("""
        X, y = task()
        contrib, margin, _p, _f, _s = ab.fit_and_explain(X[:1000], y[:1000], X[1000:], seed=0)
        assert contrib.shape == (500, 7)
        # contributions plus the expected value (last column) reproduce the raw margin
        assert np.allclose(contrib.sum(axis=1), margin, atol=1e-6)
    """)


def test_noise_feature_has_near_zero_importance() -> None:
    _run("""
        X, y = task()
        contrib, *_ = ab.fit_and_explain(X[:1000], y[:1000], X[1000:], seed=0)
        imp = ab.mean_abs_shap(contrib)
        imp = imp / imp.sum()
        assert imp[0] > imp[1] > imp[2:].max()
        # GBM spends some splits on noise (about 6% share each here), so "near zero" is
        # relative: every noise column under a third of the weaker signal column
        assert imp[2:].max() < imp[1] / 3, imp
    """)


def test_run_dataset_stability_fields() -> None:
    _run("""
        X, y = task(2000)
        res = ab.run_dataset(X, y, "synthetic", seeds=(0, 1), n_folds=3, top_k=2)
        assert res.top_features[0]["index"] == 0
        assert res.fold_topk_jaccard_mean == 1.0  # signal columns {0, 1} found every time
        assert res.seed_spearman_mean > 0.5
        assert abs(sum(res.mean_importance) - 1.0) < 1e-9
    """)


def test_rank_helpers() -> None:
    _run("""
        a = np.array([3.0, 2.0, 1.0, 0.0])
        assert ab.spearman(a, a) == 1.0
        assert ab.spearman(a, -a) == -1.0
        assert np.isnan(ab.spearman(a, np.ones(4)))
        assert ab.topk_jaccard(a, a[::-1], 2) == 0.0
    """)
