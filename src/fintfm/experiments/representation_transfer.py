"""Task 48.22: is fintfm's context-conditioned representation useful without its own head?

Every finding to date scores this project's own classification head. Neuralk-AI's
``TabPfnVectorizer`` (``docs/paper/RELATED_WORK.md``, Seldon section) treats a pretrained TFM
as a fixed feature extractor instead: hand its embeddings to a plain downstream model. That is
a different question, answerable only by comparing four arms on the same held-out rows:

  (a) fintfm's own head                             -- the number every other finding reports
  (b) a linear classifier on fintfm's representation -- does the embedding carry signal
  (c) the same linear classifier on raw features     -- the embedding's baseline to beat
  (d) a small MLP on fintfm's representation         -- Phase A' item 3: is a nonlinear final
                                                         layer worth having, on its own terms

(b) losing to (a) is expected; (b) losing to (c) is the result that would say the
representation carries nothing (c) did not already have. (d) is not framed as "the fix" --
``docs/results/FINDINGS.md`` §133 found the linear-probe advantage decays gradually through the
whole stack rather than at one identifiable layer, so there is no single broken component a
nonlinear head repairs. It is scored as its own hypothesis: whether a nonlinear read of the
frozen representation does better than a linear one, which is a question about the
representation's shape, not a diagnosis of the existing head.

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

**The equal-label-count control (design review DR3.4).** That sample holds every training
positive, while the head conditions on a 2,000-row uniform context holding only the positives
its draw happens to include. A probe that beats the head may therefore be reading more labels,
not a better representation. ``probe_rows="context"`` fits the three downstream arms on the
classifier's own context rows (:attr:`FinancialTFMClassifier.context_indices_`) instead, so
probe and head see the same labelled rows, and keeps the stratified-sample arms beside them
under a ``_stratified`` suffix for reference. ``"stratified"``, the default, is §130's
experiment, unchanged. One caveat the measurement has to carry: a context row scored as a
query can attend to its own labelled copy in the context, so every probe row in the context
arm is scored with its label in view, where the stratified arm has that only for the context
rows its sample happens to include. Nothing here has measured the control yet.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from fintfm.evaluation.datasets import CACHE_DIR
from fintfm.experiments.v4_protocol import (
    HORIZON_FILES,
    _load_horizon,
    build_fold_assignments,
    split_indices_for_fold,
)
from fintfm.inference.classifier import FinancialTFMClassifier

if TYPE_CHECKING:  # the model class is only needed for the annotation
    from fintfm.modeling.model import FinancialTFM

Arm = Literal["own_head", "representation", "raw_features", "mlp_head"]

#: Which training rows the downstream arms are fitted on: §130's stratified sample, or the
#: classifier's own context rows (DR3.4). See the module docstring.
ProbeRows = Literal["stratified", "context"]

Stage = Literal["encode_rows", "encoder_mid", "pre_head"]

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


def _downstream_arms(
    clf: FinancialTFMClassifier,
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    X_test: np.ndarray,
    stage: Stage,
    seed: int,
) -> dict[str, np.ndarray]:
    """Fit the three downstream arms on one set of training rows and score the test rows.

    Args:
        clf: The fitted classifier whose representation the probe and the MLP read.
        X_fit: Training rows for the downstream models, scaled as ``clf``'s were.
        y_fit: Their labels, 0/1.
        X_test: The held-out rows every arm is scored on.
        stage: The representation depth; see
            :meth:`FinancialTFMClassifier.transform_representation`.
        seed: The MLP's ``random_state`` (the fold index).

    Returns:
        Test-row positive-class probabilities for ``"representation"``, ``"raw_features"``
        and ``"mlp_head"``. All NaN when ``y_fit`` holds one class, which no classifier can
        be fitted on, so the fold records a gap rather than the run raising.
    """
    if np.unique(y_fit).size < 2:
        return {
            name: np.full(len(X_test), np.nan)
            for name in ("representation", "raw_features", "mlp_head")
        }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rep_tr = clf.transform_representation(X_fit, stage=stage)
        rep_te = clf.transform_representation(X_test, stage=stage)
        lr_rep = LogisticRegression(max_iter=1000).fit(rep_tr, y_fit)
        p_representation = lr_rep.predict_proba(rep_te)[:, 1]

        lr_raw = LogisticRegression(max_iter=1000).fit(X_fit, y_fit)
        p_raw_features = lr_raw.predict_proba(X_test)[:, 1]

        mlp_rep = MLPClassifier(
            hidden_layer_sizes=(64,),
            early_stopping=True,
            n_iter_no_change=15,
            max_iter=500,
            random_state=seed,
        ).fit(rep_tr, y_fit)
        p_mlp_head = mlp_rep.predict_proba(rep_te)[:, 1]
    return {
        "representation": p_representation,
        "raw_features": p_raw_features,
        "mlp_head": p_mlp_head,
    }


def _scores(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """ROC-AUC and average precision; NaN for an arm :func:`_downstream_arms` could not fit."""
    if np.isnan(p).any():
        return float("nan"), float("nan")
    return float(roc_auc_score(y, p)), float(average_precision_score(y, p))


def run(
    model_path: FinancialTFM | str,
    out_path: Path,
    horizon: int = 0,
    folds: tuple[int, ...] = (0, 1, 2, 3, 4),
    max_rows: int | None = None,
    root: Path | None = None,
    device: str = "cpu",
    train_sample_size: int = TRAIN_SAMPLE_SIZE,
    predictions_dir: Path | None = None,
    stage: Stage = "encode_rows",
    probe_rows: ProbeRows = "stratified",
    max_context: int | None = None,
) -> dict:
    """Score the three arms on V4FinBench's published protocol. See module docstring.

    Args:
        model_path: A classification checkpoint, matching :func:`v4_protocol.run`'s contract,
            or a loaded :class:`~fintfm.modeling.model.FinancialTFM`.
        out_path: Where the result dictionary is written, as JSON.
        horizon: Paper horizon, 0-5.
        folds: Which rotations to run.
        max_rows: Development cap; companies are kept whole.
        root: Directory holding the parquet files.
        device: Torch device for the fintfm arm.
        train_sample_size: Budget for the downstream classifiers' training rows; see
            :data:`TRAIN_SAMPLE_SIZE`.
        stage: Which representation to extract, passed through to
            :meth:`FinancialTFMClassifier.transform_representation`. ``"encode_rows"`` is
            §130's stage; ``"encoder_mid"`` is partway through the row-to-row encoder
            (§131's open question, Phase A' item 2); ``"pre_head"`` is what the
            classification head actually reads.
        predictions_dir: When given, per-row test predictions for every arm are written here
            as ``predictions_fold{N}.npz``, matching :func:`v4_protocol.run`'s convention --
            two arms scored on the same rows have correlated errors, so a paired bootstrap
            needs the raw vectors rather than the point estimates alone
            (``docs/results/FINDINGS.md`` §60).
        probe_rows: ``"stratified"`` (the default, §130 unchanged) fits the downstream arms
            on the stratified sample; ``"context"`` fits them on the classifier's context
            rows and records the stratified arms beside them with a ``_stratified`` suffix
            (design review DR3.4).
        max_context: The classifier's context size; ``None`` keeps the classifier's default
            of 2,000, the size §130 ran at. A development knob for a small table.

    Returns:
        The recorded result dictionary, also written to ``out_path``.

    Raises:
        ValueError: For an unknown ``probe_rows``.
    """
    if probe_rows not in ("stratified", "context"):
        raise ValueError(f"probe_rows must be 'stratified' or 'context', got {probe_rows!r}")
    clf_kwargs = {} if max_context is None else {"max_context": max_context}
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
        clf = FinancialTFMClassifier(model_path, device=device, random_state=fold, **clf_kwargs)
        clf.fit(Xtr_full, ytr_full)

        sub = _stratified_subsample(rng, ytr_full, train_sample_size)
        Xtr, ytr = Xtr_full[sub], ytr_full[sub]

        p_own_head = clf.predict_proba(Xte)[:, 1]

        stratified = _downstream_arms(clf, Xtr, ytr, Xte, stage, fold)
        arms: dict[str, np.ndarray] = {"own_head": p_own_head}
        row = {
            "fold": int(fold),
            "n_train_full": len(tr),
            "n_train_sample": len(sub),
            "n_test": len(te),
        }
        if probe_rows == "context":
            # the rows the head conditions on, so the probe reads exactly as many labels
            ctx = clf.context_indices_
            arms.update(_downstream_arms(clf, Xtr_full[ctx], ytr_full[ctx], Xte, stage, fold))
            arms.update({f"{name}_stratified": p for name, p in stratified.items()})
            row["n_probe_rows"] = len(ctx)
            row["n_probe_positive"] = int(ytr_full[ctx].sum())
        else:
            arms.update(stratified)
        for name, p in arms.items():
            row[f"{name}_auc"], row[f"{name}_ap"] = _scores(yte, p)
        results.append(row)
        probe_note = (
            f"probe on {row['n_probe_rows']:,} context rows, {row['n_probe_positive']} "
            "positive; "
            if probe_rows == "context"
            else ""
        )
        stratified_note = (
            f"  representation_stratified AP {row['representation_stratified_ap']:.4f}"
            if probe_rows == "context"
            else ""
        )
        print(
            f"  fold {fold}  ({probe_note}train sample {len(sub):,} of {len(tr):,})  "
            f"own_head AP {row['own_head_ap']:.4f}  "
            f"representation AP {row['representation_ap']:.4f}  "
            f"mlp_head AP {row['mlp_head_ap']:.4f}  "
            f"raw_features AP {row['raw_features_ap']:.4f}{stratified_note}",
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
    if probe_rows != "stratified":
        # recorded only off the default, so a default run writes the JSON §130's code wrote
        record["probe_rows"] = probe_rows
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
    p.add_argument(
        "--stage", choices=("encode_rows", "encoder_mid", "pre_head"), default="encode_rows"
    )
    p.add_argument(
        "--probe-rows", choices=("stratified", "context"), default="stratified",
        help="rows the downstream arms are fitted on: the stratified sample (default, as "
             "§130 ran), or the classifier's own context rows, with the stratified arms kept "
             "beside them under a _stratified suffix (design review DR3.4)",
    )
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
        stage=args.stage,
        probe_rows=args.probe_rows,
    )


if __name__ == "__main__":
    main()
