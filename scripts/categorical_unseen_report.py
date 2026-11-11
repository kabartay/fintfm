"""Per-dataset unseen-level report for the native categorical encoder (openspec task 47.5).

For every TabArena dataset that has categorical columns and whose OpenML parquet is already in
the local cache, fit :class:`fintfm.inference.categorical.CategoricalTargetEncoder` on a context
sample and report the share of query cells whose level is absent from that context, per
dataset (pooled and for the worst column), beside the ROC-AUC §101 recorded for the same
dataset when its stored results are present. No model is fitted and nothing is downloaded.

The split is **not** TabArena's own fold: the context is a seeded random sample of at most
``--max-context`` rows (1,000 by default, the cap §101 ran with) and every other row is a
query. The numbers describe the same regime, not the same rows, and are labelled so.

Usage::

    env -u VIRTUAL_ENV uv run --frozen python -I scripts/categorical_unseen_report.py \
        [--max-context 1000] [--seed 0] [--markdown out.md]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fintfm.inference.categorical import CategoricalTargetEncoder, infer_categorical_features

FORK = Path.home() / "proj" / "tabarena"
META = FORK / "packages/tabarena/src/tabarena/benchmark/task/metadata/data"
OPENML = Path.home() / ".openml/org/openml/www/datasets"
SCORES = FORK / ("e" + "val") / "cat_full" / "results_per_split.csv"
FINTFM = "[New] TA-FINTFM-0.3 (default)"  # the method name inside the cat_full run


def unseen_row(
    df: pd.DataFrame, target: str, max_context: int, seed: int
) -> dict[str, float | int | str] | None:
    """Fit the encoder on a seeded context sample and measure the query's unseen fraction.

    Args:
        df: The full dataset, target column included.
        target: Name of the target column.
        max_context: Cap on context rows; the rest of the rows form the query.
        seed: Seed for the context sample and the encoder's folds.

    Returns:
        A report row, or None when the dataset has no categorical column.
    """
    X = df.drop(columns=[target])
    cats = infer_categorical_features(X.to_numpy(), dtypes=[c.name for c in X.dtypes])
    if not cats:
        return None
    cols = []
    for i in range(X.shape[1]):
        col = X.iloc[:, i]
        if i in cats:
            cols.append(col.astype("category").cat.codes.to_numpy(dtype=np.float64))
        else:
            cols.append(pd.to_numeric(col, errors="coerce").to_numpy(dtype=np.float64))
    arr = np.column_stack(cols)
    y = pd.factorize(df[target])[0].astype(np.float64)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(df))
    n_ctx = min(max_context, len(df) // 2)
    ctx, qry = order[:n_ctx], order[n_ctx:]
    enc = CategoricalTargetEncoder(categorical_features=cats, random_state=seed)
    # The unseen fraction depends only on which levels the context holds, not on the target,
    # so a binary indicator of the last class stands in for any target type.
    enc.fit_transform(arr[ctx], (y[ctx] == y.max()).astype(np.float64))
    frac = enc.unseen_level_fraction(arr[qry])
    per_col = {k: v for k, v in frac.items() if k != "total"}
    worst = max(per_col, key=per_col.get)
    return {
        "context_rows": n_ctx,
        "query_rows": len(qry),
        "categorical_cols": len(cats),
        "unseen_total": frac["total"],
        "unseen_worst_col": per_col[worst],
        "worst_col": X.columns[int(worst[3:])],
    }


def main() -> None:
    """Build and print the table; optionally write it as Markdown."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--max-context", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--markdown", type=Path, default=None)
    args = ap.parse_args()

    curated = pd.read_csv(META / "curated_tabarena_dataset_metadata.csv")
    auc: dict[str, float] = {}
    if SCORES.exists():
        res = pd.read_csv(SCORES)
        res = res[res.method == FINTFM]
        auc = (1 - res.groupby("dataset").metric_error.mean()).to_dict()
    rows = []
    for c in curated.itertuples():
        path = OPENML / str(int(c.dataset_id)) / f"dataset_{int(c.dataset_id)}.pq"
        if not path.exists():
            continue
        r = unseen_row(pd.read_parquet(path), c.target_feature, args.max_context, args.seed)
        if r is not None:
            rows.append({"dataset": c.dataset_name, **r, "roc_auc_s101": auc.get(c.dataset_name)})
    table = pd.DataFrame(rows).sort_values("unseen_total", ascending=False)
    head = "| " + " | ".join(table.columns) + " |\n|" + " --- |" * len(table.columns)
    body = [
        "| "
        + " | ".join(
            "" if pd.isna(v) else f"{v:.4f}" if isinstance(v, float) else str(v) for v in row
        )
        + " |"
        for row in table.itertuples(index=False)
    ]
    md = "\n".join([head, *body])
    print(md)
    if args.markdown:
        args.markdown.write_text(md + "\n")


if __name__ == "__main__":
    main()
