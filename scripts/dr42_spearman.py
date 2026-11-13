"""DR4.2 Verify: Spearman rank correlation of the proxy suite against TabArena-Lite Elo.

One row per checkpoint that has both a complete proxy run (48 scored split rows) and a Lite Elo.
Takes any number of checkpoints: the exact permutation p-value is computed for n <= 9 (9! pairings,
vectorised) and by a seeded Monte Carlo above that. Reads ``proxy_<arm>.csv`` files from the
runs directory and ``lite_elo.csv`` (from ``runs/agents1104/DR4.2/lite_elo.py``); it fits no model.

Usage::

    nice -n 19 uv run --frozen python scripts/dr42_spearman.py \
        [--runs <repo>/runs] [--out-dir <dir>] [--arms a,b,c]

``--out-dir`` defaults to ``<runs>/agents1104/DR4.2`` and receives ``spearman_inputs.csv``.
A checkpoint missing from ``lite_elo.csv`` or with fewer than 48 scored rows is skipped with a
message on stderr, so a half-finished queue never contaminates the correlation.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

#: Scored split rows a complete proxy run holds (16 scorable tasks x 3 splits).
FULL_ROWS = 48
#: Largest n whose pairings are enumerated; above it the p-value is a Monte Carlo estimate.
EXACT_MAX_N = 9
#: Monte Carlo pairings used above EXACT_MAX_N.
MC_PERMUTATIONS = 200_000
#: Arms tried when ``--arms`` is not given: name -> (proxy csv under runs, method in csv, Lite row).
FIXED_ARMS = {
    "published": ("dr73/proxy_binary.csv", "FinTFM-binary", "published_nan"),
    "e1-v480-s0": ("dr73/proxy_e1_v480-s0.csv", "e1-v480-s0", "e1-v480-s0"),
}
#: Arms read from ``agents1104/DR4.2/proxy_<arm>.csv`` (the Lite row has the same name).
DR42_ARMS = (
    "priorctl", "priorctl-s1", "treeclean", "treeclean-s1", "treeprior", "widerprior", "priorwide",
)
SCORES = ("proxy_elo", "proxy_ap", "proxy_auc")
LITE_COLUMNS = ("lite_elo_common", "lite_elo_stored")


def exact_permutation_p(x: np.ndarray, y: np.ndarray, seed: int = 0) -> tuple[float, float, bool]:
    """Permutation p-values for Spearman's rho between ``x`` and ``y``.

    Args:
        x: First score per checkpoint.
        y: Second score per checkpoint.
        seed: Seed of the Monte Carlo draw used when n exceeds :data:`EXACT_MAX_N`.

    Returns:
        ``(two_sided, one_sided_ge, exact)``: the share of pairings with ``|rho| >= |observed|``
        and with ``rho >= observed``, and whether every pairing was enumerated.
    """
    n = len(x)
    rx = rankdata(x)
    ry = rankdata(y)
    rxc = rx - rx.mean()
    sx = np.sqrt((rxc**2).sum())
    ryc = ry - ry.mean()
    sy = np.sqrt((ryc**2).sum())
    exact = n <= EXACT_MAX_N
    if exact:
        perms = np.array(list(itertools.permutations(range(n))), dtype=np.int8)
    else:
        rng = np.random.default_rng(seed)
        perms = np.argsort(rng.random((MC_PERMUTATIONS, n)), axis=1)
    rho = (ryc[perms] @ rxc) / (sx * sy)
    obs = float(rxc @ ryc / (sx * sy))
    return (
        float(np.mean(np.abs(rho) >= abs(obs) - 1e-12)),
        float(np.mean(rho >= obs - 1e-12)),
        exact,
    )


def bootstrap_interval(
    x: np.ndarray, y: np.ndarray, n_boot: int = 5000, seed: int = 0
) -> tuple[float, float]:
    """95% percentile interval of rho, bootstrapping over checkpoints.

    Draws with fewer than two distinct values in either column are dropped (rho undefined).
    """
    rng = np.random.default_rng(seed)
    n = len(x)
    draws = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if len(set(x[i])) < 2 or len(set(y[i])) < 2:
            continue
        draws.append(spearmanr(x[i], y[i]).statistic)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(lo), float(hi)


def correlate(table: pd.DataFrame) -> pd.DataFrame:
    """Spearman rho, exact p and bootstrap interval for every score against every Lite column.

    Args:
        table: One row per checkpoint with :data:`SCORES` and :data:`LITE_COLUMNS` columns.

    Returns:
        One row per (score, Lite column) pair.
    """
    rows = []
    for x in SCORES:
        for y in LITE_COLUMNS:
            xv, yv = table[x].to_numpy(float), table[y].to_numpy(float)
            two, one, exact = exact_permutation_p(xv, yv)
            lo, hi = bootstrap_interval(xv, yv)
            rows.append({
                "score": x, "lite": y, "n": len(table),
                "rho": float(spearmanr(xv, yv).statistic), "p_two_sided": two,
                "p_one_sided": one, "p_exact": exact, "boot_lo": lo, "boot_hi": hi,
            })
    return pd.DataFrame(rows)


def build_inputs(runs: Path, arms: list[str] | None = None) -> pd.DataFrame:
    """Join complete proxy runs with Lite Elo, one row per checkpoint.

    Args:
        runs: The ``runs`` directory of the main checkout.
        arms: Checkpoint names to try (defaults to the published pair plus :data:`DR42_ARMS`).

    Returns:
        Rows indexed by arm: proxy Elo with its interval, proxy mean AP and ROC-AUC, Lite Elo on
        the stored and the common pool, Lite mean error.
    """
    from fintfm.evaluation.proxy_suite import elo_table

    d42 = runs / "agents1104" / "DR4.2"
    lite = pd.read_csv(d42 / "lite_elo.csv").set_index("arm")
    srcs = dict(FIXED_ARMS)
    for a in DR42_ARMS if arms is None else arms:
        if a not in srcs:
            srcs[a] = (f"agents1104/DR4.2/proxy_{a}.csv", a, a)
    base = pd.read_csv(runs / "dr73/proxy_binary.csv")
    frames = [base[base.method != "FinTFM-binary"]]  # the five baselines; RF anchors the Elo
    use: dict[str, str] = {}
    for arm, (rel, method, lite_row) in srcs.items():
        path = runs / rel
        if not path.exists():
            print("SKIP no proxy csv", arm, file=sys.stderr)
            continue
        f = pd.read_csv(path)
        f = f[f.method == method].copy()
        n_ok = int((f.status == "ok").sum())
        if n_ok < FULL_ROWS:
            print("SKIP incomplete", arm, n_ok, file=sys.stderr)
            continue
        if lite_row not in lite.index:
            print("SKIP no Lite Elo", arm, file=sys.stderr)
            continue
        f["method"] = arm
        frames.append(f)
        use[arm] = lite_row
    allf = pd.concat(frames, ignore_index=True)
    tab = elo_table(allf, anchor="RF", n_boot=1000, seed=0).set_index("method")
    rows = []
    for arm, lr in use.items():
        f = allf[(allf.method == arm) & (allf.status == "ok")]
        per_task = f.groupby("task_id")[["ap", "roc_auc"]].mean()
        rows.append({
            "arm": arm, "n_splits": len(f), "proxy_elo": tab.loc[arm, "elo"],
            "proxy_lo": tab.loc[arm, "lo"], "proxy_hi": tab.loc[arm, "hi"],
            "proxy_ap": per_task.ap.mean(), "proxy_auc": per_task.roc_auc.mean(),
            "lite_elo_stored": lite.loc[lr, "elo_stored_pool"],
            "lite_elo_common": lite.loc[lr, "elo_common_pool"],
            "lite_mean_err": lite.loc[lr, "mean_error"],
        })
    return pd.DataFrame(rows).set_index("arm")


def main(argv: list[str] | None = None) -> int:
    """Build the inputs, write ``spearman_inputs.csv`` and print the correlations."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", type=Path, default=Path.home() / "proj/fintfm/runs")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--arms", default=None, help="comma-separated checkpoint names")
    args = ap.parse_args(argv)
    out = args.out_dir or args.runs / "agents1104" / "DR4.2"
    table = build_inputs(args.runs, args.arms.split(",") if args.arms else None)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "spearman_inputs.csv")
    print(table.round(4).to_string())
    for r in correlate(table).itertuples():
        kind = "exact" if r.p_exact else "Monte Carlo"
        print(f"{r.score} vs {r.lite}: rho={r.rho:.3f} n={r.n} two-sided p={r.p_two_sided:.4f} "
              f"({kind}), one-sided p={r.p_one_sided:.4f}, bootstrap 95% "
              f"[{r.boot_lo:.3f}, {r.boot_hi:.3f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
