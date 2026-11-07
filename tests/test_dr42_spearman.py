"""Tests for ``scripts/dr42_spearman.py``: the DR4.2 Spearman Verify must take n = 9 inputs."""

from __future__ import annotations

import importlib.util
import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

SPEC = importlib.util.spec_from_file_location(
    "dr42_spearman", Path(__file__).resolve().parents[1] / "scripts" / "dr42_spearman.py"
)
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def _slow_p(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """The original per-permutation scipy loop, as the reference."""
    r = spearmanr(x, y).statistic
    perm = np.array(
        [spearmanr(x, y[list(q)]).statistic for q in itertools.permutations(range(len(x)))]
    )
    return float(np.mean(np.abs(perm) >= abs(r) - 1e-12)), float(np.mean(perm >= r - 1e-12))


def test_exact_p_matches_scipy_loop_at_n6() -> None:
    """The vectorised enumeration reproduces the earlier n = 6 result (0.242 two-sided)."""
    x = np.array([574.2, 677.2, 663.6, 721.1, 738.2, 673.0])
    y = np.array([755.9, 841.8, 775.4, 840.8, 783.0, 789.7])
    two, one, exact = mod.exact_permutation_p(x, y)
    assert exact
    assert two == pytest.approx(_slow_p(x, y)[0])
    assert one == pytest.approx(_slow_p(x, y)[1])
    assert spearmanr(x, y).statistic == pytest.approx(0.6, abs=0.01)
    assert two == pytest.approx(0.242, abs=0.001)


def test_n9_perfect_ranking_is_exact_and_fast() -> None:
    """With n = 9 and identical ranks only the identity (and ties) reach rho = 1: p = 1/9!."""
    x = np.arange(9, dtype=float)
    t0 = time.time()
    two, one, exact = mod.exact_permutation_p(x, x * 3 + 1)
    assert exact
    assert one == pytest.approx(1 / 362880)
    assert two == pytest.approx(2 / 362880)  # identity and the exact reversal
    assert time.time() - t0 < 30


def test_n9_matches_scipy_on_random_data() -> None:
    """The n = 9 p-value agrees with scipy's own exact value, within enumeration rounding."""
    rng = np.random.default_rng(3)
    x, y = rng.normal(size=9), rng.normal(size=9)
    two, _, exact = mod.exact_permutation_p(x, y)
    from scipy.stats import permutation_test

    ref = permutation_test(
        (x,), lambda a: spearmanr(a, y).statistic, permutation_type="pairings",
        alternative="two-sided", n_resamples=np.inf,
    ).pvalue
    assert exact
    assert two == pytest.approx(ref, abs=1e-9)


def test_above_nine_falls_back_to_monte_carlo() -> None:
    """n = 10 is flagged as not exact and gives a p-value in [0, 1]."""
    x = np.arange(10, dtype=float)
    two, one, exact = mod.exact_permutation_p(x, x)
    assert not exact
    assert 0 <= one <= two <= 1


def test_correlate_takes_nine_rows() -> None:
    """The table-level entry point handles nine checkpoints and returns every pairing."""
    rng = np.random.default_rng(0)
    table = pd.DataFrame({c: rng.normal(size=9) for c in (*mod.SCORES, *mod.LITE_COLUMNS)})
    out = mod.correlate(table)
    assert len(out) == len(mod.SCORES) * len(mod.LITE_COLUMNS)
    assert set(out.n) == {9}
    assert out.p_exact.all()
    assert (out.boot_lo <= out.boot_hi).all()


def test_build_inputs_skips_incomplete_and_unscored(tmp_path: Path) -> None:
    """An arm with fewer than 48 scored rows, or no Lite row, never enters the table."""
    d42 = tmp_path / "agents1104" / "DR4.2"
    d42.mkdir(parents=True)
    (tmp_path / "dr73").mkdir()
    rng = np.random.default_rng(1)

    def frame(method: str, n: int, status: str = "ok") -> pd.DataFrame:
        auc = rng.uniform(0.6, 0.9, n)
        return pd.DataFrame({
            "task_id": np.repeat(np.arange(16), 3)[:n], "kind": "binary",
            "split": np.tile(np.arange(3), 16)[:n], "method": method, "status": status,
            "error": 1 - auc, "roc_auc": auc, "ap": rng.uniform(0.4, 0.8, n),
        })

    base = pd.concat([frame(m, 48) for m in ("RF", "KNN", "FinTFM-binary")])
    base.to_csv(tmp_path / "dr73" / "proxy_binary.csv", index=False)
    frame("e1-v480-s0", 48).to_csv(tmp_path / "dr73" / "proxy_e1_v480-s0.csv", index=False)
    frame("armfull", 48).to_csv(d42 / "proxy_armfull.csv", index=False)
    frame("armpart", 20).to_csv(d42 / "proxy_armpart.csv", index=False)
    frame("armnolite", 48).to_csv(d42 / "proxy_armnolite.csv", index=False)
    pd.DataFrame({
        "arm": ["published_nan", "e1-v480-s0", "armfull", "armpart"],
        "elo_stored_pool": [750, 840, 800, 790], "elo_common_pool": [750, 840, 800, 790],
        "mean_error": [0.2] * 4,
    }).to_csv(d42 / "lite_elo.csv", index=False)
    table = mod.build_inputs(tmp_path, ["armfull", "armpart", "armnolite"])
    assert set(table.index) == {"published", "e1-v480-s0", "armfull"}
