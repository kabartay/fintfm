"""Task 48.22: is fintfm's context-conditioned representation useful without its own head?

Every finding to date scores this project's own classification head. Neuralk-AI's
``TabPfnVectorizer`` (``docs/paper/RELATED_WORK.md``, Seldon section) treats a pretrained TFM
as a fixed feature extractor instead: hand its embeddings to a plain downstream model. That is
a different question, answerable only by comparing three arms on the same rows:

  (a) fintfm's own head                         -- the number every other finding reports
  (b) a plain classifier on fintfm's representation -- does the embedding carry signal
  (c) the same plain classifier on raw features     -- the embedding's baseline to beat

(b) losing to (a) is expected; (b) losing to (c) is the result that would say the
representation carries nothing (c) did not already have.
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


def run(
    model_path: str,
    out_path: Path,
    horizon: int = 0,
    folds: tuple[int, ...] = (0, 1, 2, 3, 4),
    max_rows: int | None = None,
    root: Path | None = None,
    device: str = "cpu",
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

    Returns:
        The recorded result dictionary, also written to ``out_path``.
    """
    root = root or (CACHE_DIR / "v4finbench")
    X, y, country, company, _names = _load_horizon(root, horizon, max_rows)
    print(
        f"horizon {horizon} ({HORIZON_FILES[horizon]}): {len(y):,} rows, {X.shape[1]} "
        f"features, {int(y.sum()):,} positive ({y.mean():.3%})"
    )
    assignments = build_fold_assignments(country, company)

    results: list[dict] = []
    for fold in folds:
        tr, _va, te = split_indices_for_fold(assignments, fold)
        imputer = SimpleImputer(strategy="median").fit(X[tr])
        scaler = StandardScaler().fit(imputer.transform(X[tr]))
        Xtr, Xte = (scaler.transform(imputer.transform(X[i])) for i in (tr, te))
        Xtr, Xte = Xtr.astype(np.float32), Xte.astype(np.float32)
        ytr, yte = y[tr], y[te]

        clf = FinancialTFMClassifier(model_path, device=device, random_state=fold)
        clf.fit(Xtr, ytr)

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
        row = {"fold": int(fold), "n_train": len(tr), "n_test": len(te)}
        for name, p in arms.items():
            row[f"{name}_auc"] = float(roc_auc_score(yte, p))
            row[f"{name}_ap"] = float(average_precision_score(yte, p))
        results.append(row)
        print(
            f"  fold {fold}  "
            f"own_head AP {row['own_head_ap']:.4f}  "
            f"representation AP {row['representation_ap']:.4f}  "
            f"raw_features AP {row['raw_features_ap']:.4f}"
        )

    record = {"horizon": horizon, "n_rows": len(y), "folds": results}
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
    args = p.parse_args()
    run(
        args.model,
        args.out,
        horizon=args.horizon,
        folds=tuple(int(f) for f in args.folds.split(",")),
        max_rows=args.max_rows,
        device=args.device,
    )


if __name__ == "__main__":
    main()
