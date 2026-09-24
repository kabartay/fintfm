#!/usr/bin/env python3
"""Score Home Credit Default Risk with the in-context model and write a submission.

**This is a deliberately unflattering benchmark and the result should be read that way.**
Home Credit is *consumer* credit — bureau features, payment histories, demographics — while
this project's prior generates *corporate* balance sheets and P&L, so the domain prior
contributes nothing here and what is being measured is the generic architecture. The dataset
is also large (307k train, 49k test), which `docs/results/FINDINGS.md` §9 identifies as the regime
where tabular foundation models lose to gradient boosting.

It is run anyway because an external, unfakeable placement is cheap and a real reality check.

Deliberate constraints, so the entry means something:

- **Only `application_train`/`application_test`.** The six auxiliary tables are ignored, and
  the winning solutions are built almost entirely on engineered cross-table aggregates. This
  is an honest "no feature engineering" entry, which is what a foundation model claims.
- **No training on competition data.** ``fit`` stores rows as context; no gradient step is
  taken on Home Credit.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fintfm.evaluation.metrics import evaluate_binary
from fintfm.inference.classifier import FinancialTFMClassifier

DATA = Path("data/cache/home-credit")


def prepare() -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.Series, list[str]]:
    """Load and encode the two application tables, sharing one categorical mapping.

    Categories are encoded from the *combined* frame so a level seen only in test does not
    receive a different code from the same level in train, which would silently scramble
    that feature at prediction time.
    """
    train = pd.read_csv(DATA / "application_train.csv")
    test = pd.read_csv(DATA / "application_test.csv")
    y = train.pop("TARGET").to_numpy()
    test_ids = test["SK_ID_CURR"].copy()

    combined = pd.concat([train, test], keys=["train", "test"])
    combined = combined.drop(columns=["SK_ID_CURR"])
    # Detect by "is it numeric?" rather than by dtype name. pandas 3 reports these as
    # `str`, not `object`, so a name-based check silently left them as text and the
    # conversion failed only at to_numpy - two hours into a training run, had this not been
    # tested first.
    for col in combined.columns:
        if not pd.api.types.is_numeric_dtype(combined[col]):
            codes = combined[col].astype("category").cat.codes.astype(float)
            combined[col] = codes.where(codes >= 0, np.nan)

    features = list(combined.columns)
    X_train = combined.loc["train"].to_numpy(dtype=np.float32)
    X_test = combined.loc["test"].to_numpy(dtype=np.float32)
    return X_train, y, X_test, test_ids, features


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--out", default="runs/homecredit_submission.csv")
    p.add_argument("--max-context", type=int, default=2000)
    p.add_argument("--strategy", default="balanced")
    p.add_argument("--holdout", type=int, default=40_000,
                   help="rows held out of context to estimate our own AUC before submitting")
    args = p.parse_args()

    X_train, y, X_test, test_ids, features = prepare()
    print(f"train {X_train.shape}, test {X_test.shape}, {len(features)} features, "
          f"default rate {y.mean():.3%}")

    rng = np.random.default_rng(0)
    idx = rng.permutation(len(y))
    hold, fit_idx = idx[: args.holdout], idx[args.holdout :]

    clf = FinancialTFMClassifier(
        args.model, max_context=args.max_context, context_strategy=args.strategy
    )
    clf.fit(X_train[fit_idx], y[fit_idx])
    print(f"context: {clf._ctx_X.shape[0]} rows, {int(clf._ctx_y.sum())} defaults")

    # local estimate first: a submission whose score we cannot anticipate teaches less
    local = evaluate_binary(y[hold], clf.predict_proba(X_train[hold])[:, 1])
    print(f"held-out (n={args.holdout:,}): {local.summary()}")

    proba = clf.predict_proba(X_test)[:, 1]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"SK_ID_CURR": test_ids, "TARGET": proba}).to_csv(out, index=False)
    print(f"wrote {out}  (mean predicted {proba.mean():.4%})")


if __name__ == "__main__":
    main()
