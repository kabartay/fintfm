"""Pair TabArena-Lite arms of FinTFM against a baseline run, per dataset, from cached results.

Reads the tabarena fork's ``eval/<run>/results_per_split.csv`` for each run (split 0, the
binary datasets), takes FinTFM's ``metric_error`` (1 - ROC-AUC on binary), and reports per arm:
better / worse / tied datasets against the baseline, a two-sided sign test over the untied ones,
the mean ROC-AUC change, the largest moves, and the arm's Elo from ``tabarena_leaderboard.csv``.
No model is fitted. The pairing DR3.5 used (``docs/results/drafts/inference_tasks_2026-11.md``).

Usage::

    .venv/bin/python -I scripts/compare_lite_arms.py fintfm_lite_ens1_1104 [more runs ...] \
        [--baseline fintfm_lite_nan_1019] [--tie 1e-6] [--markdown out.md]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest

EVAL = Path.home() / "proj" / "tabarena" / "eval"


def fintfm_errors(run: str) -> pd.Series:
    """FinTFM's split-0 error per dataset for one run.

    Args:
        run: Directory name under the fork's ``eval/``.

    Returns:
        ``metric_error`` indexed by dataset; exactly one ``[New]`` FinTFM method must be present.

    Raises:
        ValueError: When the run holds no ``[New]`` method or more than one.
    """
    df = pd.read_csv(EVAL / run / "results_per_split.csv")
    new = df[df["method"].str.startswith("[New]") & (df["fold"] == 0)]
    methods = new["method"].unique()
    if len(methods) != 1:
        raise ValueError(f"{run}: expected one [New] method, found {list(methods)}")
    return new.set_index("dataset")["metric_error"]


def elo(run: str) -> str:
    """The run's ``[New]`` row from its leaderboard: Elo and its interval, as printed there."""
    lb = pd.read_csv(EVAL / run / "tabarena_leaderboard.csv")
    row = lb[lb["method"].astype(str).str.startswith("[New]")].iloc[0]
    cols = [c for c in lb.columns if c.lower().startswith("elo")]
    return ", ".join(f"{c}={row[c]:.0f}" if isinstance(row[c], float) else f"{c}={row[c]}" for c in cols)


def compare(arm: str, baseline: str, tie: float) -> tuple[dict, pd.DataFrame]:
    """Per-dataset pairing of one arm against the baseline.

    Args:
        arm: The arm's run name.
        baseline: The baseline's run name.
        tie: Absolute ROC-AUC change at or below which a dataset counts as tied.

    Returns:
        A summary dict and the per-dataset table (ROC-AUC of both, change), largest move first.
    """
    a, b = fintfm_errors(arm), fintfm_errors(baseline)
    shared = a.index.intersection(b.index)
    delta = (b[shared] - a[shared]).rename("delta_auc")  # error down = AUC up
    better, worse = int((delta > tie).sum()), int((delta < -tie).sum())
    n = better + worse
    p = binomtest(better, n, 0.5).pvalue if n else float("nan")
    table = pd.DataFrame({"auc_arm": 1 - a[shared], "auc_base": 1 - b[shared], "delta_auc": delta})
    table = table.reindex(delta.abs().sort_values(ascending=False).index)
    summary = {
        "arm": arm, "datasets": len(shared), "better": better, "worse": worse,
        "tied": len(shared) - n, "sign_p": p, "mean_delta_auc": float(delta.mean()), "elo": elo(arm),
    }
    return summary, table


def main() -> None:
    """Print each arm's pairing; optionally write the summary rows as markdown."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("arms", nargs="+")
    ap.add_argument("--baseline", default="fintfm_lite_nan_1019")
    ap.add_argument("--tie", type=float, default=1e-6)
    ap.add_argument("--top", type=int, default=4)
    ap.add_argument("--markdown", type=Path)
    args = ap.parse_args()
    print(f"baseline {args.baseline}: {elo(args.baseline)}")
    rows = []
    for arm in args.arms:
        s, t = compare(arm, args.baseline, args.tie)
        rows.append(s)
        print(f"\n{arm}: {s['better']} better / {s['worse']} worse / {s['tied']} tied of "
              f"{s['datasets']}, sign p={s['sign_p']:.3f}, mean dAUC={s['mean_delta_auc']:+.5f}; {s['elo']}")
        print(t.head(args.top).to_string(float_format=lambda v: f"{v:+.4f}" if abs(v) < 0.5 else f"{v:.4f}"))
    if args.markdown:
        lines = ["| arm | better / worse / tied | sign test p | mean ROC-AUC change | Elo |",
                 "| --- | --- | --- | --- | --- |"]
        lines += [f"| `{r['arm']}` | {r['better']} / {r['worse']} / {r['tied']} | {r['sign_p']:.2f} | "
                  f"{r['mean_delta_auc']:+.5f} | {r['elo']} |" for r in rows]
        args.markdown.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
