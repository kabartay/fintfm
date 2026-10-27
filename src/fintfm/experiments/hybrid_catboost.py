"""CatBoost with out-of-fold FinTFM logits, under V4FinBench's published protocol (audit AU4.4).

What this measures
------------------
The ceiling of a **system** that puts FinTFM's in-context score into a gradient-boosted model:
CatBoost alone against CatBoost given FinTFM's logit, either as its starting score (CatBoost's
``baseline``, so the trees fit what FinTFM leaves over) or as one more feature. It is not a
FinTFM result and is reported as a system, per the task.

How the logits are kept honest
------------------------------
CatBoost needs a FinTFM logit for every **training** row, and a FinTFM context built from the
training rows would contain the very row being scored. So inside each outer fold the training
rows are split again, by company within country (the same dealing as the outer folds), into
``n_inner`` inner folds; each inner fold's rows are scored by a FinTFM whose context is drawn
only from the other inner folds. Validation and test rows are scored by one FinTFM fitted on
all training rows, which is exactly the protocol's own ``fintfm`` arm.

Two checks raise :class:`LeakageError` rather than relying on the construction: no inner context
holds a validation or test row, or any row of a validation or test company; and no inner
context holds a row of a company it is scoring.

Cost, stated because it multiplies
----------------------------------
Every training row is scored once as an inner query, so a fold costs ``n_inner`` FinTFM fits
and ``n_train + n_val + n_test`` query rows, against ``n_val + n_test`` for the protocol's arm;
each inner fit conditions on ``(n_inner - 1) / n_inner`` of the training rows, so the pools
the inner fits sample their contexts from total ``n_train * (n_inner - 1)`` rows. A fine-tuning
configuration fine-tunes ``n_inner + 1`` times per fold. ``--max-train-rows`` caps the training
rows (whole companies) for development; it changes what is trained on and is recorded.

Output
------
``predictions_fold<k>.npz`` in the layout ``fintfm-v4protocol`` writes (``y_true``,
``pred_<arm>``, ``y_val``, ``val_<arm>``, ``company_test``, ``company_val``), plus one
directory per arm under ``arms/<arm>/`` whose npz carries that arm as ``pred_fintfm`` and
``val_fintfm``, so ``scripts/compare_v4_scores.py`` pairs any two arms directly::

    uv run python scripts/compare_v4_scores.py --control <out>/arms/catboost \\
        baseline=<out>/arms/catboost_fintfm_baseline feature=<out>/arms/catboost_fintfm_feature
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from fintfm.config import Config, load_config
from fintfm.experiments.v4_protocol import (
    FOLD_SEED,
    HORIZON_FILES,
    N_SPLITS,
    build_fold_assignments,
    company_codes,
    fintfm_classifier,
    parse_booster_params,
    split_indices_for_fold,
)

if TYPE_CHECKING:  # torch is imported lazily below; annotations stay free
    from fintfm.inference.classifier import FinancialTFMClassifier
    from fintfm.modeling.model import FinancialTFM

#: The arms, in the order they are reported. ``fintfm`` is the protocol's own arm.
ARMS: tuple[str, ...] = (
    "fintfm", "catboost", "catboost_fintfm_baseline", "catboost_fintfm_feature",
)

#: Probability clip before a logit, so a saturated FinTFM score cannot become infinite.
_P_CLIP = 1e-6

#: Seed offset for the inner company dealing, so it never coincides with the outer one.
INNER_SEED_OFFSET = 1000

_WORKER = '''
import json, sys
import numpy as np
from catboost import CatBoostClassifier, Pool
d = np.load(sys.argv[1])
params = json.loads(sys.argv[3])
est = CatBoostClassifier(verbose=0, allow_writing_files=False, **params)
btr = d["btr"] if "btr" in d.files else None
est.fit(Pool(d["Xtr"], d["ytr"], baseline=btr))
# raw score without the baseline, which is then added explicitly: the result is the same
# whether or not this CatBoost version adds a Pool's baseline at predict time
raw = est.predict(Pool(d["Xte"]), prediction_type="RawFormulaVal")
if "bte" in d.files:
    raw = raw + d["bte"]
np.save(sys.argv[2], 1.0 / (1.0 + np.exp(-raw)))
'''


class LeakageError(AssertionError):
    """An inner FinTFM context holds a row it must never see.

    An :class:`AssertionError` in kind, raised explicitly so ``python -O`` cannot strip it.
    """


def to_logit(p: np.ndarray) -> np.ndarray:
    """Log-odds of a probability, clipped to ``[1e-6, 1 - 1e-6]`` first.

    Args:
        p: Probabilities.

    Returns:
        ``float64`` log-odds of the same shape.
    """
    p = np.clip(np.asarray(p, dtype=np.float64), _P_CLIP, 1.0 - _P_CLIP)
    return np.log(p) - np.log1p(-p)


def fold_group_key(country: np.ndarray, company: np.ndarray) -> np.ndarray:
    """One key per row for the unit the protocol's folds deal: a company within its country.

    :func:`build_fold_assignments` groups by ``(country, company)``, so a company name that
    appears in two countries (7 of V4FinBench's 188,058, e.g. ``MOL Nyrt.``) can sit in
    different outer folds by the published protocol's own design. A leak check keyed on the
    name alone refused fold 0 for exactly that; keying on the pair checks what the folds promise.

    Args:
        country: ``(n,)`` country per row.
        company: ``(n,)`` company per row.

    Returns:
        ``(n,)`` string keys, ``"<country>|<company>"``.
    """
    country_str = np.asarray(country).astype(str)
    return np.char.add(np.char.add(country_str, "|"), np.asarray(company).astype(str))


def assert_no_leak(
    context_rows: np.ndarray, forbidden_rows: np.ndarray, company: np.ndarray | None,
    what: str,
) -> None:
    """Refuse a context that shares a row, or a company, with rows it must not see.

    Args:
        context_rows: Panel row indices a FinTFM context is drawn from.
        forbidden_rows: Panel row indices that must stay out of it.
        company: ``(n,)`` company (or borrower) per panel row, or ``None`` for a panel
            without one, such as the credit suite's ULB and SBA, where only rows are checked.
        what: Names the check in the error, e.g. ``"fold 2 inner 1 vs outer val/test"``.

    Raises:
        LeakageError: If a row index or a company appears on both sides.
    """
    shared_rows = np.intersect1d(context_rows, forbidden_rows)
    if shared_rows.size:
        raise LeakageError(f"{what}: {shared_rows.size} forbidden row(s) in the context")
    if company is None:
        return
    shared = np.intersect1d(company[context_rows], company[forbidden_rows])
    if shared.size:
        raise LeakageError(f"{what}: {shared.size} forbidden company(ies) in the context")


def inner_oof_logits(
    X: np.ndarray,
    y: np.ndarray,
    train_rows: np.ndarray,
    country: np.ndarray,
    company: np.ndarray,
    forbidden_rows: np.ndarray,
    make_classifier: Callable[[int], FinancialTFMClassifier],
    n_inner: int,
    seed: int,
) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    """Out-of-fold FinTFM logits for every training row, from company-grouped inner folds.

    Args:
        X: ``(n, F)`` preprocessed features for the whole panel.
        y: ``(n,)`` labels for the whole panel.
        train_rows: Panel indices of the outer fold's training rows.
        country: ``(n,)`` country per panel row.
        company: ``(n,)`` company per panel row.
        forbidden_rows: Panel indices no inner context may touch: the outer validation and
            test rows.
        make_classifier: Builds an unfitted FinTFM classifier from a seed.
        n_inner: Inner folds.
        seed: Seed for dealing companies to inner folds.

    Returns:
        ``(logits, plan)``: ``(len(train_rows),)`` logits aligned with ``train_rows``, and per
        inner fold the ``(context_rows, query_rows)`` panel indices it used, for checking.

    Raises:
        LeakageError: Via :func:`assert_no_leak`, before any inner fit runs.
        ValueError: If an inner context holds a single class, which the classifier could not
            score the other class from.
    """
    inner = build_fold_assignments(
        country[train_rows], company[train_rows], n_splits=n_inner, seed=seed
    )
    key = fold_group_key(country, company)
    logits = np.full(len(train_rows), np.nan)
    plan: list[tuple[np.ndarray, np.ndarray]] = []
    for k in range(n_inner):
        ctx, qry = train_rows[inner != k], train_rows[inner == k]
        assert_no_leak(ctx, forbidden_rows, key, f"inner fold {k} vs outer val/test")
        assert_no_leak(ctx, qry, key, f"inner fold {k} vs its own queries")
        plan.append((ctx, qry))
    for k, (ctx, qry) in enumerate(plan):
        if qry.size == 0:
            continue  # fewer companies than inner folds in every country: nothing to score
        if len(np.unique(y[ctx])) < 2:
            raise ValueError(f"inner fold {k}'s context holds one class; lower --inner-folds")
        clf = make_classifier(k)
        clf.fit(X[ctx], y[ctx], groups=company[ctx])
        logits[inner == k] = to_logit(clf.predict_proba(X[qry])[:, 1])
    if np.isnan(logits).any():
        raise RuntimeError("a training row received no out-of-fold logit")
    return logits, plan


def fit_predict_catboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    params: dict[str, Any] | None = None,
    baseline_train: np.ndarray | None = None,
    baseline_test: np.ndarray | None = None,
    timeout: int = 3600,
) -> np.ndarray | None:
    """Fit CatBoost out of process, optionally from a per-row starting score.

    Out of process for the reason :mod:`fintfm.evaluation.boosting` gives (two OpenMP
    runtimes beside torch). With a baseline, CatBoost's trees fit what the baseline leaves
    over, and the test score is the baseline plus the trees.

    Args:
        X_train: Training features.
        y_train: Training labels.
        X_test: Rows to score.
        params: CatBoost parameters; library defaults when ``None``.
        baseline_train: Optional ``(n_train,)`` starting log-odds.
        baseline_test: Starting log-odds for ``X_test``; required with ``baseline_train``.
        timeout: Seconds before the subprocess is abandoned.

    Returns:
        ``(n_test,)`` positive-class probabilities, or ``None`` if the fit failed, which is
        printed rather than silent.

    Raises:
        ValueError: If only one of the two baselines is given.
    """
    if (baseline_train is None) != (baseline_test is None):
        raise ValueError("baseline_train and baseline_test go together")
    with tempfile.TemporaryDirectory() as tmp:
        data, out = Path(tmp) / "d.npz", Path(tmp) / "p.npy"
        arrays = {"Xtr": X_train, "ytr": y_train, "Xte": X_test}
        if baseline_train is not None:
            arrays |= {"btr": baseline_train, "bte": baseline_test}
        np.savez(data, **arrays)
        proc = subprocess.run(
            [sys.executable, "-c", _WORKER, str(data), str(out), json.dumps(params or {})],
            capture_output=True, text=True, timeout=timeout, check=False,
        )
        if proc.returncode != 0 or not out.exists():
            reason = (proc.stderr or "").strip().splitlines()
            print(f"  catboost failed: {reason[-1][:160] if reason else proc.returncode}")
            return None
        return np.load(out)


def cap_train_rows(
    train_rows: np.ndarray, company: np.ndarray, max_rows: int, seed: int
) -> np.ndarray:
    """Keep whole training companies, in a seeded random order, up to ``max_rows`` rows.

    Args:
        train_rows: Panel indices of the training rows.
        company: ``(n,)`` company per panel row.
        max_rows: Row cap. At least one company is always kept.
        seed: Seed for the company order.

    Returns:
        Sorted panel indices, a subset of ``train_rows`` made of whole companies.
    """
    if len(train_rows) <= max_rows:
        return train_rows
    names, inverse, counts = np.unique(
        company[train_rows], return_inverse=True, return_counts=True
    )
    order = np.random.RandomState(seed).permutation(len(names))
    kept = order[: max(1, int(np.searchsorted(np.cumsum(counts[order]), max_rows, "right")))]
    return np.sort(train_rows[np.isin(inverse.ravel(), kept)])


def _metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """AP, ROC-AUC, and the mean prediction beside the observed rate (``CLAUDE.md``)."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    both = len(np.unique(y)) > 1
    return {
        "avg_precision": float(average_precision_score(y, p)) if both else float("nan"),
        "roc_auc": float(roc_auc_score(y, p)) if both else float("nan"),
        "mean_predicted": float(np.mean(p)),
        "observed_rate": float(np.mean(y)),
    }


def run_arrays(
    X: np.ndarray,
    y: np.ndarray,
    country: np.ndarray,
    company: np.ndarray,
    model: FinancialTFM,
    out_dir: Path,
    folds: tuple[int, ...] = (0, 1, 2, 3, 4),
    cfg: Config | None = None,
    device: str = "cpu",
    n_inner: int = 5,
    max_train_rows: int | None = None,
    catboost_params: dict[str, Any] | None = None,
    protocol: dict[str, Any] | None = None,
) -> dict:
    """Score the hybrid arms on an already-loaded panel, fold by fold.

    Args:
        X: ``(n, F)`` raw features.
        y: ``(n,)`` binary labels.
        country: ``(n,)`` country per row, for the protocol's fold dealing.
        company: ``(n,)`` company per row.
        model: The loaded FinTFM checkpoint.
        out_dir: Output directory.
        folds: Outer rotations to run.
        cfg: Configuration; its ``inference`` section builds every FinTFM.
        device: Torch device for FinTFM.
        n_inner: Inner company-grouped folds for the out-of-fold logits.
        max_train_rows: Development cap on each outer fold's training rows (whole companies).
        catboost_params: CatBoost parameters for all three CatBoost arms.
        protocol: Protocol settings recorded for ``compare_v4_scores.py``'s check (``horizon``,
            ``max_rows``); the run's own ``n_features``, ``fold_seed``, ``n_splits`` and
            ``dataset`` are added.

    Returns:
        The run record, also written as ``hybrid_catboost.json`` and ``v4_protocol.json``.
    """
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    cfg = cfg or load_config()
    out_dir.mkdir(parents=True, exist_ok=True)
    X = np.asarray(X, dtype=np.float32)
    assignments = build_fold_assignments(country, company)
    codes = company_codes(company)
    fold_records = []
    for fold in folds:
        tr, va, te = split_indices_for_fold(assignments, fold)
        n_full = len(tr)
        if max_train_rows is not None:
            tr = cap_train_rows(tr, company, max_train_rows, seed=fold)
        held = np.concatenate([va, te])
        held_key = fold_group_key(country, company)
        assert_no_leak(tr, held, held_key, f"fold {fold} outer train vs val/test")
        # preprocessing fitted on the (capped) training rows alone, as the protocol does
        imputer = SimpleImputer(strategy="median").fit(X[tr])
        scaler = StandardScaler().fit(imputer.transform(X[tr]))
        Z = scaler.transform(imputer.transform(X)).astype(np.float32)

        members = cfg.inference.n_ensemble
        print(
            f"fold {fold}: {len(tr):,} training rows"
            + (f" (capped from {n_full:,})" if len(tr) < n_full else "")
            + f", {len(va):,} val, {len(te):,} test\n"
            f"  FinTFM cost: {n_inner} inner fits + 1 outer fit; query rows "
            f"{len(tr):,} (train, out-of-fold) + {len(held):,} (val+test) = "
            f"{len(tr) + len(held):,}, x {members} ensemble member(s); inner context pools "
            f"total {len(tr) * (n_inner - 1):,} rows ({len(tr):,} rows x {n_inner - 1} of "
            f"{n_inner} folds), each fit sampling <= {cfg.inference.max_context:,}"
            + (
                f"; fine-tunes {n_inner + 1} x {cfg.inference.finetune_steps} steps"
                if cfg.inference.finetune_steps else ""
            ),
            flush=True,
        )
        oof, plan = inner_oof_logits(
            Z, y, tr, country, company, held,
            lambda k, fold=fold: fintfm_classifier(
                model, cfg, device=device, random_state=fold + INNER_SEED_OFFSET * (k + 1)
            ),
            n_inner, seed=FOLD_SEED + INNER_SEED_OFFSET + fold,
        )
        outer = fintfm_classifier(model, cfg, device=device, random_state=fold)
        outer.fit(Z[tr], y[tr], groups=company[tr])
        p_held = outer.predict_proba(Z[held])[:, 1]
        b_held = to_logit(p_held)

        arms: dict[str, np.ndarray] = {"fintfm": p_held}
        base = fit_predict_catboost(Z[tr], y[tr], Z[held], catboost_params)
        if base is not None:
            arms["catboost"] = base
        hybrid = fit_predict_catboost(
            Z[tr], y[tr], Z[held], catboost_params, baseline_train=oof, baseline_test=b_held
        )
        if hybrid is not None:
            arms["catboost_fintfm_baseline"] = hybrid
        feature = fit_predict_catboost(
            np.column_stack([Z[tr], oof]), y[tr], np.column_stack([Z[held], b_held]),
            catboost_params,
        )
        if feature is not None:
            arms["catboost_fintfm_feature"] = feature

        nv = len(va)
        common = {
            "y_true": y[te], "y_val": y[va], "company_test": codes[te], "company_val": codes[va],
        }
        np.savez_compressed(
            out_dir / f"predictions_fold{fold}.npz",
            **common,
            **{f"pred_{a}": p[nv:] for a, p in arms.items()},
            **{f"val_{a}": p[:nv] for a, p in arms.items()},
        )
        for a, p in arms.items():
            d = out_dir / "arms" / a
            d.mkdir(parents=True, exist_ok=True)
            # the arm under the name compare_v4_scores.py reads, so any two arms pair directly
            np.savez_compressed(
                d / f"predictions_fold{fold}.npz", **common,
                pred_fintfm=p[nv:], val_fintfm=p[:nv],
            )
        scores = {a: {"test": _metrics(y[te], p[nv:]), "val": _metrics(y[va], p[:nv])}
                  for a, p in arms.items()}
        for a in ARMS:
            if a in scores:
                t = scores[a]["test"]
                print(f"  fold {fold} {a:>26}: AP {t['avg_precision']:.4f}  ROC-AUC "
                      f"{t['roc_auc']:.4f}  mean predicted {t['mean_predicted']:.4%} vs "
                      f"observed {t['observed_rate']:.4%}", flush=True)
            else:
                print(f"  fold {fold} {a:>26}: FAILED, excluded from this fold")
        fold_records.append({
            "fold": fold, "n_train": len(tr), "n_train_uncapped": n_full, "n_val": nv,
            "n_test": len(te), "inner_folds": n_inner,
            "inner_contexts_checked": len(plan),
            "fintfm_query_rows": len(tr) + len(held),
            "inner_pool_rows_total": len(tr) * (n_inner - 1),
            "scores": scores,
        })

    config = {
        **(protocol or {}),
        "n_features": int(X.shape[1]), "fold_seed": FOLD_SEED, "n_splits": N_SPLITS,
        "folds": list(folds), "inner_folds": n_inner, "max_train_rows": max_train_rows,
        "catboost_params": catboost_params or {}, "inference": asdict(cfg.inference),
        "config_sources": list(cfg.sources),
    }
    record = {
        "dataset": "v4finbench",
        "protocol": "V4FinBench published protocol (5-fold, company-grouped within country)",
        "measures": "a system (CatBoost plus FinTFM logits), not FinTFM; audit AU4.4",
        "config": config,
        "folds": fold_records,
    }
    (out_dir / "hybrid_catboost.json").write_text(json.dumps(record, indent=2))
    # what compare_v4_scores.py checks before pairing two directories
    (out_dir / "v4_protocol.json").write_text(json.dumps(record, indent=2))
    for a in ARMS:
        d = out_dir / "arms" / a
        if d.exists():
            (d / "v4_protocol.json").write_text(json.dumps({**record, "arm": a}, indent=2))
    return record


def main() -> None:
    """Entry point: ``uv run python -m fintfm.experiments.hybrid_catboost --model ...``."""
    from fintfm.evaluation.datasets import CACHE_DIR
    from fintfm.experiments.v4_protocol import _load_horizon
    from fintfm.modeling.model import FinancialTFM

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--out", type=str, default="runs/hybrid-catboost")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--horizon", type=int, default=0, choices=sorted(HORIZON_FILES))
    p.add_argument("--folds", type=str, default="0,1,2,3,4", help="comma-separated")
    p.add_argument("--max-rows", type=int, default=None,
                   help="development cap on the panel, whole companies, as fintfm-v4protocol")
    p.add_argument("--max-train-rows", type=int, default=None,
                   help="development cap on each fold's training rows, whole companies; "
                        "val/test rows are untouched, so arms still pair with a full run")
    p.add_argument("--inner-folds", type=int, default=5,
                   help="company-grouped inner folds for the out-of-fold FinTFM logits; "
                        "FinTFM fits per outer fold = this + 1")
    p.add_argument("--catboost-params", type=str, default="",
                   help="k=v,... for every CatBoost arm, e.g. iterations=200,depth=6; "
                        "empty is library defaults (1,000 iterations)")
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()
    if args.inner_folds < 2:
        p.error("--inner-folds must be at least 2")
    try:
        ((_, params),) = parse_booster_params([f"catboost:{args.catboost_params}"])
    except ValueError as e:
        p.error(str(e))
    cfg = load_config(args.config)
    X, y, country, company, _ = _load_horizon(CACHE_DIR / "v4finbench", args.horizon,
                                              args.max_rows)
    record = run_arrays(
        X, y, country, company, FinancialTFM.load(args.model), Path(args.out),
        folds=tuple(int(v) for v in args.folds.split(",")), cfg=cfg, device=args.device,
        n_inner=args.inner_folds, max_train_rows=args.max_train_rows,
        catboost_params=params,
        protocol={"model": args.model, "horizon": args.horizon, "max_rows": args.max_rows},
    )
    print(f"\n{record['measures']}. config: {cfg.provenance()}")


if __name__ == "__main__":
    main()
