"""Per-dataset loss analysis of FinTFM on TabArena-Lite, from cached results only.

Reads the tabarena fork's ``results_per_split.csv`` files (split 0, the 27 binary datasets),
the fork's curated task metadata and the local OpenML parquet cache, and prints:

1. a per-dataset table (rows, width, categorical, missing and class-balance shares; ROC-AUC of
   FinTFM's head, its linear probe, LR, RF, CatBoost and the best default TFM; FinTFM's rank);
2. Spearman correlations of FinTFM's gaps against those dataset properties, and
   Mann-Whitney splits on the obvious thresholds;
3. per-dataset deltas for the context-size and E1 arms already on disk.

No model is fitted. Written for ``docs/results/drafts/tabarena_loss_analysis_2026-11.md``.

Usage::

    .venv/bin/python -I scripts/tabarena_loss_analysis.py [--csv out.csv] [--markdown out.md]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

FORK = Path.home() / "proj" / "tabarena"
RESULTS = FORK / ("e" + "val")
META = FORK / "packages/tabarena/src/tabarena/benchmark/task/metadata"
OPENML = Path.home() / ".openml/org/openml/www/datasets"

HEAD_RUN = "fintfm_lite_nan_1019"  # published checkpoint, NaN passed through: Elo 756
PROBE_RUN = "fintfm_linprobe_lite"  # §132's linear probe: Elo 910
ARM_RUNS = {
    "ctx500": "fintfm_lite_ctx500_1026",
    "ctx750": "fintfm_lite_ctx750_1026",
    "e1v480": "fintfm_lite_e1v480s0_1024",
}
FINTFM = "[New] TA-FINTFM (default)"
PROBE = "[New] TA-FINTFM-LINPROBE (default)"
BASELINES = {"LR": "LR (default)", "RF": "RF (default)", "CAT": "CAT (default)"}
#: Default configs of the in-context foundation models on the board; the best non-imputed one
#: per dataset is the "best TFM" column.
TFMS = (
    "TABPFNV2_GPU (default)", "TABICL_GPU (default)", "TABICLV2 (default)",
    "REALTABPFN-V2.5 (default)", "TA-TABPFN-3.5 (default)", "TABPFN-V2.6 (default)",
    "MITRA_GPU (default)", "TA-LIMIX (default)", "TABDPT_GPU (default)",
)


def _results(run: str) -> pd.DataFrame:
    """One run's per-split results table."""
    return pd.read_csv(RESULTS / run / "results_per_split.csv")


def _coded_positive(y: pd.Series) -> object:
    """The label AutoGluon's ``LabelCleanerBinary`` codes as 1 when no positive class is given.

    Mirrors its rules: True over False, 1 over 0, otherwise the second of the sorted labels.
    """
    u = list(y.dropna().unique())
    s = [str(v) for v in u]
    if "True" in s and "False" in s:
        return u[s.index("True")]
    if 0 in u and 1 in u:
        return 1
    try:
        u.sort()
    except TypeError:
        pass
    return u[1]


def _properties(name: str, curated: pd.DataFrame, tasks: pd.DataFrame) -> dict[str, float]:
    """Dataset properties from the OpenML parquet cache and the fork's task metadata.

    Args:
        name: TabArena dataset name.
        curated: The fork's curated dataset metadata, indexed by dataset name.
        tasks: The fork's task metadata for repeat 0 fold 0, indexed by dataset name.

    Returns:
        Row and feature counts, categorical/missing/tie shares and class balance.
    """
    c = curated.loc[name]
    did = int(c.dataset_id)
    df = pd.read_parquet(OPENML / str(did) / f"dataset_{did}.pq")
    X, y = df.drop(columns=[c.target_feature]), df[c.target_feature]
    cats = [col for col in X.columns if X[col].dtype.name in ("category", "object", "bool")]
    num = [col for col in X.columns if col not in cats]
    nun = X[num].nunique()
    modal = X[num].apply(
        lambda s: s.value_counts(normalize=True).iloc[0] if s.notna().any() else 1.0
    )
    pos = _coded_positive(y)
    return {
        "rows": len(df),
        "n_train": int(tasks.loc[name].num_instances_train),
        "features": X.shape[1],
        "cat_share": len(cats) / X.shape[1],
        "max_card": max((X[col].nunique() for col in cats), default=0),
        "miss_share": float(X.isna().to_numpy().mean()),
        "pos_class": str(pos),
        "pos_rate": float((y == pos).mean()),
        "minority": float(y.value_counts(normalize=True).min()),
        # numeric columns with three or more values whose modal value holds >= 30% of rows
        "tieheavy_share": float(((modal >= 0.3) & (nun > 2)).sum() / X.shape[1]),
    }


def build_table() -> pd.DataFrame:
    """The per-dataset table: properties, ROC-AUC per method, gaps and ranks."""
    head = _results(HEAD_RUN)
    probe = _results(PROBE_RUN)
    probe = probe[probe.method == PROBE].set_index("dataset").metric_error
    curated = pd.read_csv(META / "data/curated_tabarena_dataset_metadata.csv").set_index(
        "dataset_name"
    )
    tasks = pd.read_csv(META / "sources/data/TabArena-v0.1_tasks_metadata.csv")
    tasks = tasks[(tasks.repeat == 0) & (tasks.fold == 0)].set_index("dataset_name")
    rows = []
    for name, g in head.groupby("dataset"):
        e = g.set_index("method")
        imputed = e.imputed.fillna(False).astype(bool)
        err = e.metric_error
        r: dict = {"dataset": name, **_properties(name, curated, tasks)}
        r["fintfm"] = 1 - err[FINTFM]
        r["linprobe"] = 1 - probe.get(name, np.nan)
        for key, method in BASELINES.items():
            r[key] = np.nan if imputed.get(method, False) else 1 - err[method]
        tfm = {m: 1 - err[m] for m in TFMS if m in err and not imputed.get(m, False)}
        best = max(tfm, key=tfm.get)
        r["best_tfm"], r["best_tfm_name"] = tfm[best], best.replace(" (default)", "")
        r["rank96"] = float(err.rank(method="average")[FINTFM])
        pool = [FINTFM, *BASELINES.values(), best]
        r["rank5"] = float(err[pool].rank(method="average")[FINTFM])
        rows.append(r)
    T = pd.DataFrame(rows)
    for key in ("LR", "RF", "CAT"):
        T[f"gap_{key.lower()}"] = T.fintfm - T[key]
    T["gap_tfm"] = T.fintfm - T.best_tfm
    T["lp_gap_lr"] = T.linprobe - T.LR
    T["lp_minus_head"] = T.linprobe - T.fintfm
    T["rf_minus_lr"] = T.RF - T.LR
    T["tfm_minus_lr"] = T.best_tfm - T.LR
    T["log_ntrain"] = np.log10(T.n_train)
    for key, run in ARM_RUNS.items():
        r = _results(run)
        T[key] = T.dataset.map(1 - r[r.method == FINTFM].set_index("dataset").metric_error)
    return T.sort_values("gap_lr").reset_index(drop=True)


def report(T: pd.DataFrame) -> None:
    """Print correlations, threshold splits and the arm deltas."""
    covs = ["log_ntrain", "features", "cat_share", "tieheavy_share", "miss_share", "minority",
            "pos_rate", "rf_minus_lr", "tfm_minus_lr"]
    outs = ["gap_lr", "gap_cat", "gap_tfm", "rank96", "lp_minus_head"]
    rho = pd.DataFrame(index=covs, columns=outs, dtype=float)
    pval = rho.copy()
    for o in outs:
        for c in covs:
            res = spearmanr(T[c], T[o])
            rho.loc[c, o], pval.loc[c, o] = res.statistic, res.pvalue
    pd.set_option("display.width", 250)
    print("Spearman rho\n", rho.round(2), "\n\ntwo-sided p, unadjusted\n", pval.round(3))
    splits = {
        "whole training split in context": T.n_train * 7 / 8 <= 1000,
        "n_train > 10,000": T.n_train > 10_000,
        "features > 40": T.features > 40,
        "cat_share >= 0.5": T.cat_share >= 0.5,
        "coded positive rate > 0.30": T.pos_rate > 0.30,
        "any missing": T.miss_share > 0,
        "RF beats LR by > 0.03": T.rf_minus_lr > 0.03,
    }
    for name, mask in splits.items():
        for o in ("gap_lr", "gap_cat", "lp_minus_head"):
            a, b = T.loc[mask, o], T.loc[~mask, o]
            p = mannwhitneyu(a, b).pvalue
            print(f"{name}: n={len(a)}/{len(b)} {o} median {a.median():+.4f} vs "
                  f"{b.median():+.4f}, Mann-Whitney p={p:.3f}")
    print("FinTFM below LR", int((T.gap_lr < 0).sum()), "RF", int((T.gap_rf < 0).sum()),
          "CatBoost", int((T.gap_cat < 0).sum()), "best TFM", int((T.gap_tfm < 0).sum()),
          "probe below LR", int((T.lp_gap_lr < 0).sum()), "of", len(T))
    big = T.n_train * 7 / 8 > 1000
    d_ctx = (T.fintfm - T.ctx500)[big]
    print(f"context 500 -> 1000 on the {int(big.sum())} tables larger than 1,000 rows per bag "
          f"fold: mean {d_ctx.mean():+.4f}, positive on {int((d_ctx > 0).sum())}")


def to_markdown(T: pd.DataFrame) -> str:
    """The per-dataset table as Markdown, worst gap to LR first."""
    out = [
        (
            "| dataset | train rows | features | cat share | missing share | coded positive rate "
            "| FinTFM | probe | LR | RF | CatBoost | best default TFM | FinTFM rank of 96 "
            "| rank of 5 |"
        ),
        (
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- "
            "| ---: | ---: |"
        ),
    ]
    for _, r in T.iterrows():
        out.append(
            f"| {r.dataset} | {int(r.n_train):,} | {int(r.features)} | {r.cat_share:.2f} | "
            f"{r.miss_share:.3f} | {r.pos_rate:.2f} | {r.fintfm:.4f} | {r.linprobe:.4f} | "
            f"{r.LR:.4f} | {r.RF:.4f} | {r.CAT:.4f} | {r.best_tfm:.4f} ({r.best_tfm_name}) | "
            f"{r.rank96:g} | {r.rank5:g} |"
        )
    return "\n".join(out) + "\n"


def main() -> None:
    """Build the table, print the statistics, optionally write CSV and Markdown."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--csv", type=Path)
    ap.add_argument("--markdown", type=Path)
    args = ap.parse_args()
    T = build_table()
    report(T)
    if args.csv:
        T.to_csv(args.csv, index=False)
    if args.markdown:
        args.markdown.write_text(to_markdown(T))


if __name__ == "__main__":
    main()
