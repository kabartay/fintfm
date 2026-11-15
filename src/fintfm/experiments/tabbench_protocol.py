"""TabBench public academic set under the suite's protocol: checkpoint against untrained control.

Openspec ``tabbench-comparison`` task 35.4. FINDINGS §134 ran the published checkpoint on the
public academic set at fold 0 only, with no floor and no accuracy. This module runs the whole
protocol the suite states (``docs/research/tabbench_v2.md``): **5 stratified shuffle folds,
20% of rows held out per fold**, ROC-AUC and accuracy per fold, averaged over folds per
dataset, for two arms:

* ``checkpoint``: the model at ``--model``, unchanged, zero-shot (``fit`` only stores context);
* ``untrained_control``: the same architecture with seeded random weights (CLAUDE.md "Every
  evaluation needs a floor"), same context, same folds.

**Internal only.** The benchmark's code licence adds an attribution clause that exempts
private benchmarking only while no part of the work or its results is communicated to an
external third party (``docs/research/tabbench_v2.md``, task 35.2, still open). Nothing here
may be quoted outside the project, and the output records say so.

**No suite code is imported.** The folds are produced by the same scikit-learn call the suite
uses (``StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)`` over the OpenML
table in its row order, with a class of one row kept in the training part), so fold ``k``
holds the same rows; the feature encoding is this module's own (ordinal codes for categorical
columns fitted on the training part, dates as days, blank strings as missing). Unlike the §134
wrapper, it parses dates and blank strings instead of failing on them. A score from here is
therefore comparable to the suite's published fold-by-fold files only up to that encoding.

**Coverage is native.** A model that skips datasets is flattered by the mean, so every dataset
is a :class:`~fintfm.evaluation.coverage.Coverage` unit per arm: scored when all five folds
scored, skipped with a reason otherwise (multi-class, wider than the checkpoint, a fold with
one class in the test part, a load or fit error). The suite has 189 datasets; ``--suite-size``
records that, and datasets absent from the ids file count as not run, never as scored.

**Resumable.** One JSON per dataset under ``<out>/datasets/``. A rerun skips every dataset
that has one, ``--shard i/n`` takes every n-th id so several processes can share the list,
and ``--aggregate-only`` rebuilds ``tabbench_protocol.json`` from what exists. Delete a
dataset's file to redo it.

Reference-only datasets: the suite's datasets may overlap TabArena-v0.1, so nothing here may
choose an arm (DR4.1); it reports. Nothing trains: numbers are labelled MEASURED only when
produced by the command in the openspec Progress note on the real checkpoint; the unit tests
are SMOKE-TEST only.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from fintfm.evaluation.coverage import Coverage
from fintfm.experiments.sign_inference import UNTRAINED_SEED

if TYPE_CHECKING:
    import pandas as pd

    from fintfm.modeling.model import FinancialTFM

#: The protocol's fold count and held-out share (report section 3.1, p. 10).
N_FOLDS = 5
TEST_SIZE = 0.2
#: The suite's split seed.
SPLIT_SEED = 0
#: Datasets in the public academic set.
SUITE_SIZE = 189
#: The two arms, in report order.
ARMS: tuple[str, ...] = ("checkpoint", "untrained_control")
#: Printed in the aggregate record and the summary, so a copied table carries it.
INTERNAL_NOTE = (
    "INTERNAL ONLY: the benchmark's code licence exempts private benchmarking only while no "
    "result is communicated externally (docs/research/tabbench_v2.md, task 35.2 open)."
)
#: Default ids: the 90 of 189 binary datasets within the 136-feature cap (FINDINGS §134).
DEFAULT_IDS = Path("configs/tabbench/eligible_ids.json")


def split_folds(
    y: np.ndarray,
    n_folds: int = N_FOLDS,
    test_size: float = TEST_SIZE,
    seed: int = SPLIT_SEED,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Stratified shuffle folds as the suite builds them, one-row classes kept in training.

    Args:
        y: Class labels in the table's row order.
        n_folds: Number of shuffled splits.
        test_size: Share of rows held out per fold.
        seed: Split seed; the suite uses 0.

    Returns:
        ``(train_index, test_index)`` per fold. A class with one row cannot be stratified, so a
        copy is added for the split and the original row is forced into the training part.
    """
    from sklearn.model_selection import StratifiedShuffleSplit

    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    rare = classes[counts < 2]
    n = len(y)
    rare_rows = [int(np.flatnonzero(y == c)[0]) for c in rare]
    y_aug = np.concatenate([y, y[rare_rows]]) if rare_rows else y
    dummy = np.zeros((len(y_aug), 1))
    cv = StratifiedShuffleSplit(n_splits=n_folds, test_size=test_size, random_state=seed)
    folds = []
    for tr, te in cv.split(dummy, y_aug):
        tr, te = tr[tr < n], te[te < n]
        for r in rare_rows:
            te = te[te != r]
            if r not in tr:
                tr = np.append(tr, r)
        folds.append((tr, te))
    return folds


def encode_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, int]:
    """Turn a mixed-type table into float32, fitting every encoding on the training rows.

    Numeric and boolean columns pass through (NaN kept: the model reads missingness through its
    own mask). Datetime columns, and text columns that parse as dates, become days since the
    epoch. Text columns that parse as numbers become numbers. Whitespace-only strings become
    missing. Remaining text and category columns become ordinal codes in order of first
    appearance in the training rows; a value unseen in training becomes missing.

    Args:
        train: Training features.
        test: Test features, same columns.

    Returns:
        ``(X_train, X_test, n_categorical)`` with columns in the input order.
    """
    import pandas as pd

    cols_tr: list[np.ndarray] = []
    cols_te: list[np.ndarray] = []
    n_cat = 0
    for col in train.columns:
        a, b = train[col], test[col]
        if pd.api.types.is_bool_dtype(a):
            ca, cb = a.astype(float), b.astype(float)
        elif pd.api.types.is_datetime64_any_dtype(a):
            ca, cb = _days(a), _days(b)
        elif pd.api.types.is_numeric_dtype(a):
            ca, cb = a.astype(float), b.astype(float)
        else:
            a2, b2 = _blank_to_nan(a), _blank_to_nan(b)
            num_a = pd.to_numeric(a2, errors="coerce")
            if num_a.notna().sum() == a2.notna().sum():
                ca, cb = num_a.astype(float), pd.to_numeric(b2, errors="coerce").astype(float)
            else:
                dt_a = _parse_dates(a2)
                if dt_a is not None and dt_a.notna().sum() == a2.notna().sum():
                    ca, cb = _days(dt_a), _days(_parse_dates(b2))
                else:
                    codes = {v: i for i, v in enumerate(pd.unique(a2.dropna()))}
                    ca = a2.map(codes).astype(float)
                    cb = b2.map(codes).astype(float)
                    n_cat += 1
        cols_tr.append(np.asarray(ca, dtype=np.float32))
        cols_te.append(np.asarray(cb, dtype=np.float32))
    if not cols_tr:
        return np.empty((len(train), 0), np.float32), np.empty((len(test), 0), np.float32), 0
    return np.column_stack(cols_tr), np.column_stack(cols_te), n_cat


def _blank_to_nan(s: pd.Series) -> pd.Series:
    """Object column with whitespace-only strings replaced by missing; category to object."""
    s = s.astype(object)
    return s.where(~s.map(lambda v: isinstance(v, str) and not v.strip()), None)


def _parse_dates(s: pd.Series) -> pd.Series | None:
    """Parse a text column as dates, or ``None`` when it does not read as dates."""
    import pandas as pd

    if s.dropna().empty:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = pd.to_datetime(s, errors="coerce", format="mixed")
    return parsed


def _days(s: pd.Series) -> np.ndarray:
    """Datetime series as float days since the epoch, NaT as NaN."""
    import pandas as pd

    s = pd.to_datetime(s, errors="coerce")
    if getattr(s.dt, "tz", None) is not None:
        s = s.dt.tz_convert(None)
    delta = s - pd.Timestamp("1970-01-01")
    return (delta / pd.Timedelta(days=1)).to_numpy(dtype=float)


def load_openml(dataset_id: int) -> tuple[pd.DataFrame, np.ndarray, str]:
    """Fetch one OpenML dataset in its own row order.

    Args:
        dataset_id: OpenML dataset id.

    Returns:
        ``(features, labels, name)``; labels are the raw target values.
    """
    import openml

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds = openml.datasets.get_dataset(
            dataset_id,
            download_data=True,
            download_qualities=False,
            download_features_meta_data=False,
        )
        X, y, _, _ = ds.get_data(target=ds.default_target_attribute)
    return X, np.asarray(y), str(ds.name)


def build_arms(model_path: str, seed: int = UNTRAINED_SEED) -> dict[str, FinancialTFM]:
    """The checkpoint and a seeded untrained control of the same architecture.

    Args:
        model_path: Checkpoint path.
        seed: Seed of the control's weights; the experiments' shared floor seed by default
            (an unseeded floor moves by 0.2, FINDINGS §42).

    Returns:
        ``{"checkpoint": ..., "untrained_control": ...}``.
    """
    import torch

    from fintfm.modeling.model import FinancialTFM

    ckpt = FinancialTFM.load(model_path, map_location="cpu")
    torch.manual_seed(seed)
    return {"checkpoint": ckpt, "untrained_control": FinancialTFM(ckpt.cfg)}


def eligibility(n_classes: int, n_features: int, max_features: int) -> str | None:
    """Why a dataset cannot be given to the checkpoint, or ``None`` when it can.

    Args:
        n_classes: Distinct target values.
        n_features: Columns after encoding.
        max_features: The checkpoint's width.

    Returns:
        The reason, written so it reads in the coverage table; ``None`` when eligible.
    """
    if n_classes != 2:
        return f"{n_classes} classes (binary checkpoint)"
    if n_features > max_features:
        return f"{n_features} features exceed the checkpoint's {max_features}"
    return None


def score_fold(
    model: FinancialTFM,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    max_context: int,
    n_ensemble: int,
    seed: int,
) -> dict[str, float]:
    """ROC-AUC and accuracy of one arm on one fold, through the public classifier.

    Args:
        model: The arm's network.
        X_train: Training features (the context), float32.
        y_train: Training labels as 0/1.
        X_test: Held-out features.
        y_test: Held-out labels as 0/1.
        max_context: Context rows kept.
        n_ensemble: Ensemble members.
        seed: Classifier seed.

    Returns:
        ``roc_auc``, ``accuracy`` and ``mean_predicted`` (mean positive probability, printed
        beside ``observed_rate``, CLAUDE.md "print the level, not only the score").

    Raises:
        ValueError: If the held-out part has one class, where ROC-AUC is undefined.
    """
    from sklearn.metrics import accuracy_score, roc_auc_score

    from fintfm.inference.classifier import FinancialTFMClassifier

    if len(np.unique(y_test)) < 2:
        raise ValueError("one class in the held-out part")
    clf = FinancialTFMClassifier(
        model, device="cpu", max_context=max_context, n_ensemble=n_ensemble, random_state=seed
    ).fit(X_train, y_train)
    proba = clf.predict_proba(X_test)
    p1 = proba[:, 1]
    return {
        "roc_auc": float(roc_auc_score(y_test, p1)),
        "accuracy": float(accuracy_score(y_test, proba.argmax(axis=1))),
        "mean_predicted": float(p1.mean()),
        "observed_rate": float(np.mean(y_test)),
    }


def run_dataset(
    dataset_id: int,
    arms: dict[str, FinancialTFM],
    max_features: int,
    max_context: int,
    n_ensemble: int,
    seed: int,
    n_folds: int = N_FOLDS,
) -> dict[str, Any]:
    """Score every arm on every fold of one dataset and return its record.

    A failure is recorded as a ``skip`` reason on the affected arm, never raised, so one bad
    dataset costs one row; the reason is printed by the caller.

    Args:
        dataset_id: OpenML dataset id.
        arms: Arm name to network.
        max_features: The checkpoint's width.
        max_context: Context rows kept.
        n_ensemble: Ensemble members.
        seed: Classifier seed.
        n_folds: Folds to run (the protocol's 5; smaller only for tests).

    Returns:
        ``id``, ``name``, shape fields and, per arm, either ``folds`` (one dict per fold) or
        ``skip`` (the reason). ``seconds`` is the dataset's wall time.
    """
    t0 = time.time()
    rec: dict[str, Any] = {"id": dataset_id, "arms": {}}
    try:
        X, labels, name = load_openml(dataset_id)
    except Exception as exc:  # noqa: BLE001 - recorded as a skip reason
        reason = f"load failed: {type(exc).__name__}: {str(exc)[:160]}"
        rec["arms"] = {a: {"skip": reason} for a in arms}
        rec["seconds"] = time.time() - t0
        return rec
    rec["name"] = name
    classes, y = np.unique(labels, return_inverse=True)
    rec.update(n_rows=len(y), n_features=int(X.shape[1]), n_classes=len(classes))
    why = eligibility(len(classes), X.shape[1], max_features)
    if why is not None:
        rec["arms"] = {a: {"skip": why} for a in arms}
        rec["seconds"] = time.time() - t0
        return rec
    folds = split_folds(y, n_folds=n_folds)
    results: dict[str, list[dict[str, float]]] = {a: [] for a in arms}
    failed: dict[str, str] = {}
    for k, (tr, te) in enumerate(folds):
        try:
            Xtr, Xte, n_cat = encode_features(X.iloc[tr], X.iloc[te])
            rec["n_categorical"] = n_cat
        except Exception as exc:  # noqa: BLE001 - recorded as a skip reason
            reason = f"fold {k}: encoding failed: {type(exc).__name__}: {str(exc)[:160]}"
            failed.update({a: reason for a in arms})
            break
        for arm, model in arms.items():
            if arm in failed:
                continue
            try:
                results[arm].append(
                    score_fold(model, Xtr, y[tr], Xte, y[te], max_context, n_ensemble, seed)
                )
            except Exception as exc:  # noqa: BLE001 - recorded as a skip reason
                failed[arm] = f"fold {k}: {type(exc).__name__}: {str(exc)[:160]}"
    for arm in arms:
        rec["arms"][arm] = {"skip": failed[arm]} if arm in failed else {"folds": results[arm]}
    rec["seconds"] = time.time() - t0
    return rec


def _dataset_path(out_dir: Path, dataset_id: int) -> Path:
    """Where one dataset's record lives."""
    return out_dir / "datasets" / f"openml-{dataset_id}.json"


def aggregate(
    out_dir: Path,
    ids: Sequence[int],
    suite_size: int = SUITE_SIZE,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Combine the per-dataset records into per-arm means, coverage and paired differences.

    Per-dataset values are the mean over folds; per-arm values are the mean over the datasets
    that arm scored. The paired difference is taken on datasets both arms scored.

    Args:
        out_dir: Directory holding ``datasets/``.
        ids: The dataset ids this run covers, in order. An id with no record is counted as
            ``pending``, not skipped.
        suite_size: The suite's dataset count, recorded beside the coverage.
        config: Run settings to store in the record.

    Returns:
        The record written to ``tabbench_protocol.json``.
    """
    records = {}
    for did in ids:
        p = _dataset_path(out_dir, did)
        if p.exists():
            records[did] = json.loads(p.read_text())
    pending = [d for d in ids if d not in records]
    per_arm: dict[str, Any] = {}
    for arm in ARMS:
        cov = Coverage()
        auc: dict[int, float] = {}
        acc: dict[int, float] = {}
        for did, r in records.items():
            unit = f"openml-{did}"
            cov.attempt(unit)
            a = r["arms"][arm]
            if "skip" in a:
                cov.skip(unit, a["skip"])
                continue
            cov.scored(unit)
            auc[did] = float(np.mean([f["roc_auc"] for f in a["folds"]]))
            acc[did] = float(np.mean([f["accuracy"] for f in a["folds"]]))
        per_arm[arm] = {
            "mean_roc_auc": float(np.mean(list(auc.values()))) if auc else None,
            "mean_accuracy": float(np.mean(list(acc.values()))) if acc else None,
            "n_scored_datasets": len(auc),
            "coverage": cov.to_dict(),
            "per_dataset": {
                f"openml-{d}": {"roc_auc": auc[d], "accuracy": acc[d]} for d in sorted(auc)
            },
        }
    both = sorted(
        set(per_arm["checkpoint"]["per_dataset"]) & set(per_arm["untrained_control"]["per_dataset"])
    )
    paired = None
    if both:
        d_auc = [
            per_arm["checkpoint"]["per_dataset"][u]["roc_auc"]
            - per_arm["untrained_control"]["per_dataset"][u]["roc_auc"]
            for u in both
        ]
        paired = {
            "n_datasets": len(both),
            "mean_roc_auc_difference": float(np.mean(d_auc)),
            "datasets_checkpoint_above_control": int(sum(x > 0 for x in d_auc)),
        }
    record = {
        "internal_only": INTERNAL_NOTE,
        "protocol": {
            "n_folds": N_FOLDS,
            "test_size": TEST_SIZE,
            "split_seed": SPLIT_SEED,
            "control_seed": UNTRAINED_SEED,
        },
        "suite_size_declared": suite_size,
        "n_ids_listed": len(ids),
        "n_datasets_with_record": len(records),
        "pending_ids": pending,
        "not_listed_count": suite_size - len(ids),
        "config": config or {},
        "arms": per_arm,
        "paired": paired,
        "seconds_per_dataset": {
            f"openml-{d}": round(r.get("seconds", 0.0), 1) for d, r in records.items()
        },
        "datasets": {
            f"openml-{d}": {
                k: r.get(k) for k in ("name", "n_rows", "n_features", "n_classes", "n_categorical")
            }
            for d, r in records.items()
        },
    }
    (out_dir / "tabbench_protocol.json").write_text(json.dumps(record, indent=2))
    return record


def summarise(record: dict[str, Any]) -> str:
    """Render the arm table, the coverage beside it and how to read it.

    Args:
        record: The :func:`aggregate` record.

    Returns:
        Text for the terminal.
    """
    lines = [record["internal_only"], ""]
    lines.append(
        f"{record['n_datasets_with_record']} of {record['n_ids_listed']} listed datasets done "
        f"({len(record['pending_ids'])} pending); the suite has {record['suite_size_declared']} "
        f"and {record['not_listed_count']} are not listed, so not run."
    )
    lines.append(f"{'arm':<20}{'ROC-AUC':>9}{'accuracy':>10}{'scored':>9}{'attempted':>11}")
    for arm in ARMS:
        a = record["arms"][arm]
        c = a["coverage"]
        auc = "n/a" if a["mean_roc_auc"] is None else f"{a['mean_roc_auc']:.4f}"
        acc = "n/a" if a["mean_accuracy"] is None else f"{a['mean_accuracy']:.4f}"
        lines.append(f"{arm:<20}{auc:>9}{acc:>10}{c['n_scored']:>9}{c['n_attempted']:>11}")
    p = record["paired"]
    if p:
        lines.append(
            f"paired over {p['n_datasets']} datasets both arms scored: mean ROC-AUC difference "
            f"(checkpoint minus control) {p['mean_roc_auc_difference']:+.4f}; checkpoint above "
            f"control on {p['datasets_checkpoint_above_control']}"
        )
    lines.append(
        "Means are over each arm's scored datasets only; read them beside the scored count. "
        "A checkpoint that does not clear the control here is below its own floor (CLAUDE.md)."
    )
    return "\n".join(lines)


def run(
    model_path: str,
    out_dir: Path,
    ids: Sequence[int],
    shard: tuple[int, int] = (0, 1),
    max_features: int | None = None,
    max_context: int = 2000,
    n_ensemble: int = 1,
    seed: int = 0,
    suite_size: int = SUITE_SIZE,
    arms: dict[str, FinancialTFM] | None = None,
) -> dict[str, Any]:
    """Run this shard's datasets (skipping finished ones) and write the aggregate.

    Args:
        model_path: Checkpoint path; ignored when ``arms`` is given.
        out_dir: Output directory; ``datasets/`` and ``tabbench_protocol.json`` go here.
        ids: OpenML dataset ids, the full list (all shards see the same one).
        shard: ``(index, count)``; this process runs ``ids[index::count]``.
        max_features: Checkpoint width; read from the checkpoint when omitted.
        max_context: Context rows kept (the classifier default, as in §134).
        n_ensemble: Ensemble members (the classifier default, as in §134).
        seed: Classifier seed.
        suite_size: The suite's dataset count.
        arms: Already-built arms (used by tests).

    Returns:
        The aggregate record.
    """
    index, count = shard
    if not (count >= 1 and 0 <= index < count):
        raise ValueError(f"shard must satisfy 0 <= index < count, got {index}/{count}")
    arms = arms if arms is not None else build_arms(model_path)
    width = max_features if max_features is not None else arms["checkpoint"].cfg.max_features
    (out_dir / "datasets").mkdir(parents=True, exist_ok=True)
    mine = list(ids)[index::count]
    for i, did in enumerate(mine):
        path = _dataset_path(out_dir, did)
        if path.exists():
            print(f"[{i + 1}/{len(mine)}] openml-{did} done already, skipped", flush=True)
            continue
        rec = run_dataset(did, arms, width, max_context, n_ensemble, seed)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(rec, indent=2))
        tmp.replace(path)  # a crash mid-write cannot leave a half record that reads as done
        cells = []
        for arm, a in rec["arms"].items():
            if "skip" in a:
                cells.append(f"{arm}: SKIPPED ({a['skip']})")
            else:
                auc = np.mean([f["roc_auc"] for f in a["folds"]])
                acc = np.mean([f["accuracy"] for f in a["folds"]])
                cells.append(f"{arm}: auc={auc:.4f} acc={acc:.4f}")
        print(
            f"[{i + 1}/{len(mine)}] openml-{did} {rec.get('name', '?')} "
            f"n={rec.get('n_rows', '?')} f={rec.get('n_features', '?')} "
            f"{rec['seconds']:.1f}s  " + "  ".join(cells),
            flush=True,
        )
    config = {
        "model": model_path,
        "max_context": max_context,
        "n_ensemble": n_ensemble,
        "seed": seed,
        "max_features": width,
    }
    return aggregate(out_dir, [int(d) for d in ids], suite_size, config)


def main(argv: Sequence[str] | None = None) -> int:
    """Command line: ``python -m fintfm.experiments.tabbench_protocol``.

    Args:
        argv: Arguments; ``sys.argv[1:]`` when omitted.

    Returns:
        Process exit status.
    """
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model", default="runs/v4-cellattn-labels.pt", help="checkpoint path")
    p.add_argument("--out", default="runs/tabbench-35-4", help="output directory")
    p.add_argument("--ids", default=str(DEFAULT_IDS), help="JSON list of OpenML dataset ids")
    p.add_argument("--limit", type=int, default=None, help="first N ids only (smoke tests)")
    p.add_argument("--shard", default="0/1", help="i/n: run ids[i::n], to split a long run")
    p.add_argument("--max-context", type=int, default=2000)
    p.add_argument("--n-ensemble", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--suite-size", type=int, default=SUITE_SIZE)
    p.add_argument("--aggregate-only", action="store_true", help="rebuild the aggregate and stop")
    args = p.parse_args(argv)
    ids = [int(i) for i in json.loads(Path(args.ids).read_text())]
    if args.limit is not None:
        ids = ids[: args.limit]
    out = Path(args.out)
    i, n = (int(x) for x in args.shard.split("/"))
    print(INTERNAL_NOTE, flush=True)
    if args.aggregate_only:
        out.mkdir(parents=True, exist_ok=True)
        record = aggregate(out, ids, args.suite_size)
    else:
        record = run(
            args.model,
            out,
            ids,
            shard=(i, n),
            max_context=args.max_context,
            n_ensemble=args.n_ensemble,
            seed=args.seed,
            suite_size=args.suite_size,
        )
    print("\n" + summarise(record))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
