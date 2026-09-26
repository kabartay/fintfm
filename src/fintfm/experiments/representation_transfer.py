"""Task 48.22: is fintfm's context-conditioned representation useful without its own head?

Every finding to date scores this project's own classification head. Neuralk-AI's
``TabPfnVectorizer`` (``docs/paper/RELATED_WORK.md``, Seldon section) treats a pretrained TFM
as a fixed feature extractor instead: hand its embeddings to a plain downstream model. That is
a different question, answerable only by comparing three arms on the same held-out rows:

  (a) fintfm's own head                             -- the number every other finding reports
  (b) a plain classifier on fintfm's representation  -- does the embedding carry signal
  (c) the same plain classifier on raw features      -- the embedding's baseline to beat

(b) losing to (a) is expected; (b) losing to (c) is the result that would say the
representation carries nothing (c) did not already have.

**Why the downstream classifiers are fitted on a bounded subsample, not the full training
split.** A first version of this experiment called
:meth:`~fintfm.inference.classifier.FinancialTFMClassifier.transform_representation` on the
full ~600,000-row training split, on top of the ``predict_proba`` call every other finding
already pays for. §119 already measured this model's median predict time at 8.6 s/1,000 rows;
extending that cost to the training split as well as the test split, across five folds, prices
at roughly 14 hours and was killed partway through rather than left to finish proving a cost
that was knowable in advance. A downstream logistic regression does not need 600,000 training
rows to fit 128 coefficients, so :data:`TRAIN_SAMPLE_SIZE` bounds it, built to keep every
minority-class row rather than a plain random sample -- V4FinBench's positive rate is near
0.36%, and a random subsample at this budget would likely contain too few positives to fit a
meaningful classifier on.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Literal

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from fintfm.evaluation.datasets import CACHE_DIR
from fintfm.experiments.v4_protocol import (
    HORIZON_FILES,
    _load_horizon,
    build_fold_assignments,
    split_indices_for_fold,
)
from fintfm.inference.classifier import FinancialTFMClassifier

Arm = Literal["own_head", "representation", "raw_features"]

#: Rows given to the downstream classifiers' *training* step, per fold. Every minority-class
#: row from the fold's training split is kept, then padded with a random majority-class sample
#: up to this budget -- see the module docstring for why a plain random subsample is unsafe at
#: V4FinBench's positive rate. The held-out test split is never subsampled: every arm, including
#: this experiment's own_head arm, is scored on the same full test rows every other finding uses.
TRAIN_SAMPLE_SIZE = 20_000


def _stratified_subsample(rng: np.random.Generator, y: np.ndarray, budget: int) -> np.ndarray:
    """Indices keeping every positive row, padded with a random negative sample to ``budget``.

    Returns all rows when the fold's training split is already at or under budget.
    """
    if len(y) <= budget:
        return np.arange(len(y))
    positive = np.where(y == 1)[0]
    negative = np.where(y == 0)[0]
    n_negative = max(budget - len(positive), 0)
    if n_negative < len(negative):
        negative = rng.choice(negative, size=n_negative, replace=False)
    return np.concatenate([positive, negative])


def run(
    model_path: str,
    out_path: Path,
    horizon: int = 0,
    folds: tuple[int, ...] = (0, 1, 2, 3, 4),
    max_rows: int | None = None,
    root: Path | None = None,
    device: str = "cpu",
    train_sample_size: int = TRAIN_SAMPLE_SIZE,
    predictions_dir: Path | None = None,
) -> dict:
    """Score the three arms on V4FinBench's published protocol. See module docstring.

    Args:
        model_path: A classification checkpoint, matching :func:`v4_protocol.run`'s contract.
        out_path: Where the result dictionary is written, as JSON.
        horizon: Paper horizon, 0-5.
        folds: Which rotations to run.
        max_rows: Development cap; companies are kept whole.
        root: Directory holding the parquet files.
        device: Torch device for the fintfm arm.
        train_sample_size: Budget for the downstream classifiers' training rows; see
            :data:`TRAIN_SAMPLE_SIZE`.
        predictions_dir: When given, per-row test predictions for every arm are written here
            as ``predictions_fold{N}.npz``, matching :func:`v4_protocol.run`'s convention --
            two arms scored on the same rows have correlated errors, so a paired bootstrap
            needs the raw vectors rather than the point estimates alone
            (``docs/results/FINDINGS.md`` §60).

    Returns:
        The recorded result dictionary, also written to ``out_path``.
    """
    root = root or (CACHE_DIR / "v4finbench")
    if predictions_dir is not None:
        predictions_dir.mkdir(parents=True, exist_ok=True)
    X, y, country, company, _names = _load_horizon(root, horizon, max_rows)
    print(
        f"horizon {horizon} ({HORIZON_FILES[horizon]}): {len(y):,} rows, {X.shape[1]} "
        f"features, {int(y.sum()):,} positive ({y.mean():.3%})",
        flush=True,
    )
    assignments = build_fold_assignments(country, company)

    results: list[dict] = []
    for fold in folds:
        rng = np.random.default_rng(fold)
        tr, _va, te = split_indices_for_fold(assignments, fold)
        imputer = SimpleImputer(strategy="median").fit(X[tr])
        scaler = StandardScaler().fit(imputer.transform(X[tr]))
        Xtr_full, Xte = (scaler.transform(imputer.transform(X[i])) for i in (tr, te))
        Xtr_full, Xte = Xtr_full.astype(np.float32), Xte.astype(np.float32)
        ytr_full, yte = y[tr], y[te]

        # The full training split is fintfm's in-context evidence -- it decides nothing about
        # cost, since fit() only stores and subsamples it to max_context. The subsample below
        # is what bounds the downstream classifiers' training cost; the two are independent.
        clf = FinancialTFMClassifier(model_path, device=device, random_state=fold)
        clf.fit(Xtr_full, ytr_full)

        sub = _stratified_subsample(rng, ytr_full, train_sample_size)
        Xtr, ytr = Xtr_full[sub], ytr_full[sub]

        p_own_head = clf.predict_proba(Xte)[:, 1]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rep_tr = clf.transform_representation(Xtr)
            rep_te = clf.transform_representation(Xte)
            lr_rep = LogisticRegression(max_iter=1000).fit(rep_tr, ytr)
            p_representation = lr_rep.predict_proba(rep_te)[:, 1]

            lr_raw = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
            p_raw_features = lr_raw.predict_proba(Xte)[:, 1]

        arms: dict[Arm, np.ndarray] = {
            "own_head": p_own_head,
            "representation": p_representation,
            "raw_features": p_raw_features,
        }
        row = {
            "fold": int(fold),
            "n_train_full": len(tr),
            "n_train_sample": len(sub),
            "n_test": len(te),
        }
        for name, p in arms.items():
            row[f"{name}_auc"] = float(roc_auc_score(yte, p))
            row[f"{name}_ap"] = float(average_precision_score(yte, p))
        results.append(row)
        print(
            f"  fold {fold}  (train sample {len(sub):,} of {len(tr):,})  "
            f"own_head AP {row['own_head_ap']:.4f}  "
            f"representation AP {row['representation_ap']:.4f}  "
            f"raw_features AP {row['raw_features_ap']:.4f}",
            flush=True,
        )
        if predictions_dir is not None:
            np.savez_compressed(
                predictions_dir / f"predictions_fold{fold}.npz",
                y_true=yte,
                **{f"pred_{name}": p for name, p in arms.items()},
            )

    record = {
        "horizon": horizon,
        "n_rows": len(y),
        "train_sample_size": train_sample_size,
        "folds": results,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import json

    out_path.write_text(json.dumps(record, indent=2))
    return record


def main() -> None:
    """CLI entry point."""
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--out", type=Path, default=Path("representation_transfer.json"))
    p.add_argument("--horizon", type=int, default=0, choices=sorted(HORIZON_FILES))
    p.add_argument("--folds", type=str, default="0,1,2,3,4")
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--train-sample-size", type=int, default=TRAIN_SAMPLE_SIZE)
    p.add_argument("--predictions-dir", type=Path, default=None)
    args = p.parse_args()
    run(
        args.model,
        args.out,
        horizon=args.horizon,
        folds=tuple(int(f) for f in args.folds.split(",")),
        max_rows=args.max_rows,
        device=args.device,
        train_sample_size=args.train_sample_size,
        predictions_dir=args.predictions_dir,
    )


if __name__ == "__main__":
    main()
