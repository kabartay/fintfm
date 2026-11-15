"""Section 27 repeated across seeds (openspec ``strong-baselines`` 18.6).

Why this exists
---------------
``docs/results/FINDINGS.md`` §27 compared the ``mixed`` checkpoint with CatBoost, LightGBM and
XGBoost on the UCI Polish 3-year panel at n = 100, 250, 1,000 and 4,000 training rows, on one
seed. Its own caveat: the n = 100 AUC win is 0.0005 over CatBoost, which is noise. This
module is the same experiment with a seed loop, per-arm mean and standard deviation across
seeds, and a paired bootstrap per n.

Design
------
- **Same protocol as §27.** Stratified 70/30 split of the panel, then a stratified training
  subsample whose positive count is ``round(n * default_rate)`` (5, 12, 47, 189 defaults at
  the four sizes), exactly as ``prior_ablation.sample_efficiency_probe`` does. The seed
  changes the split, the subsample, the in-context sampling and every booster's own seed.
- **Boosters out-of-process** through :func:`fintfm.evaluation.boosting.fit_predict_boosting`
  (§25: torch and Homebrew OpenMP cannot share a process).
- **Paired bootstrap per n.** fintfm minus the reference arm (CatBoost) is bootstrapped with
  :func:`fintfm.evaluation.metrics.paired_auc_difference`, once with ``average_precision_score``
  and once with ROC-AUC, on the predictions of all seeds pooled. Test sets differ by seed and
  overlap, so ``groups`` is the panel row index: a bootstrap resample draws whole *rows* with
  every seed's copy of them, which keeps the pairing and does not count an overlapping row
  twice as two independent draws. The pooled metric is computed on the concatenation, so it
  is close to, not identical to, the mean of the per-seed metrics; both are printed.
- **Holm across n**, separately for AP and for ROC-AUC (:func:`holm_adjusted_p`).
- **Average precision first**, then ROC-AUC (CLAUDE.md), then ECE and Brier skill.

Numbers are MEASURED only when produced by the command in the openspec Progress note on the
real checkpoint; the unit tests are SMOKE-TEST only and run on synthetic data.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

from fintfm.evaluation.metrics import evaluate_binary, holm_adjusted_p, paired_auc_difference

#: Training sizes of §27 (rows).
SIZES: tuple[int, ...] = (100, 250, 1000, 4000)

#: Seeds of the repeat; §27 was a single seed, 0.
SEEDS: tuple[int, ...] = (0, 1, 2)

#: Arm the model is compared against, and the model arm's name.
REFERENCE_ARM = "catboost"
MODEL_ARM = "fintfm"

#: Metrics reported per arm, AP first.
METRICS: tuple[str, ...] = ("average_precision", "roc_auc", "ece", "brier_skill")

#: A predictor fits on ``(X_train, y_train, seed)`` and scores ``X_test``; ``None`` = skipped.
Predictor = Callable[[np.ndarray, np.ndarray, np.ndarray, int], "np.ndarray | None"]


def stratified_subsample(
    X: np.ndarray, y: np.ndarray, size: int, rate: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Draw a training subsample holding ``round(size * rate)`` positives.

    Same rule as ``sample_efficiency_probe``, so tiny sets still contain defaults.

    Args:
        X: Pool features.
        y: Pool labels.
        size: Rows to draw.
        rate: Panel default rate that sets the positive count.
        rng: Generator for the draw.

    Returns:
        ``(X_sub, y_sub)``, at most ``size`` rows (fewer if a class runs out).
    """
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    n_pos = min(max(1, round(size * rate)), len(pos))
    n_neg = min(size - n_pos, len(neg))
    idx = np.concatenate(
        [rng.choice(pos, n_pos, replace=False), rng.choice(neg, n_neg, replace=False)]
    )
    return X[idx], y[idx]


def _mean_std(values: list[float]) -> dict[str, float | list[float]]:
    """Mean and sample standard deviation across seeds (NaN-safe, ddof=1).

    Args:
        values: One value per seed.

    Returns:
        ``{"mean", "std", "per_seed"}``; the std is NaN with fewer than two finite values.
    """
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    return {
        "mean": float(finite.mean()) if len(finite) else float("nan"),
        "std": float(finite.std(ddof=1)) if len(finite) > 1 else float("nan"),
        "per_seed": [float(v) for v in arr],
    }


def run_seed_study(
    X: np.ndarray,
    y: np.ndarray,
    predictors: Mapping[str, Predictor],
    sizes: tuple[int, ...] = SIZES,
    seeds: tuple[int, ...] = SEEDS,
    n_boot: int = 2000,
    test_size: float = 0.3,
    reference: str = REFERENCE_ARM,
    model: str = MODEL_ARM,
) -> dict:
    """Run the §27 comparison at every size and seed and bootstrap model minus reference.

    Args:
        X: Panel features ``(n, d)``.
        y: Binary labels ``(n,)``.
        predictors: ``{arm name: fit-and-score callable}``; must contain ``model`` and
            ``reference``.
        sizes: Training sizes, ascending.
        seeds: Seeds; each changes the split, the subsample and the arms' own seed.
        n_boot: Bootstrap resamples per paired comparison.
        test_size: Held-out fraction of the panel per seed.
        reference: Arm the model is compared against.
        model: The model arm.

    Returns:
        A record with ``sizes[str(n)]`` holding, per arm, mean/std/per-seed of every metric,
        and ``paired[metric]`` (delta, 95% interval, raw and Holm-adjusted p). Sizes larger
        than the pool for any seed are listed under ``skipped``.

    Raises:
        ValueError: If ``model`` or ``reference`` is not among ``predictors``.
    """
    for arm in (model, reference):
        if arm not in predictors:
            raise ValueError(f"predictors must contain {arm!r}; got {sorted(predictors)}")
    y = np.asarray(y).astype(int)
    rate = float(y.mean())
    row_id = np.arange(len(y))
    out: dict = {
        "panel_rows": len(y),
        "default_rate": rate,
        "seeds": list(seeds),
        "sizes_requested": list(sizes),
        "n_boot": n_boot,
        "reference": reference,
        "sizes": {},
        "skipped": {},
    }
    # per size: arm -> seed -> metrics; and pooled predictions for the paired bootstrap
    cells: dict[int, dict[str, list]] = {}
    pooled: dict[int, dict[str, list[np.ndarray]]] = {}
    for seed in seeds:
        tr, te = train_test_split(row_id, test_size=test_size, random_state=seed, stratify=y)
        rng = np.random.default_rng(seed)
        for size in sizes:
            if size > len(tr):
                out["skipped"][f"n{size}"] = f"size {size} exceeds the pool of {len(tr)} rows"
                continue
            Xs, ys = stratified_subsample(X[tr], y[tr], size, rate, rng)
            if len(np.unique(ys)) < 2:
                out["skipped"][f"n{size}"] = "single-class training subsample"
                continue
            cell = cells.setdefault(size, {})
            store = pooled.setdefault(size, {"y": [], "g": []})
            store["y"].append(y[te])
            store["g"].append(row_id[te])
            for arm, fn in predictors.items():
                t0 = time.time()
                p = fn(Xs, ys, X[te], seed)
                print(f"  seed {seed} n={size:>5} {arm:<10} {time.time() - t0:6.1f}s", flush=True)
                if p is None:
                    continue
                store.setdefault(arm, []).append(np.asarray(p, dtype=float))
                cell.setdefault(arm, []).append(
                    {"seed": seed, "n_positive_train": int(ys.sum()), **_metrics(y[te], p)}
                )

    raw_p: dict[str, list[float]] = {m: [] for m in ("average_precision", "roc_auc")}
    sized: list[int] = []
    for size in sorted(cells):
        arms = cells[size]
        out["sizes"][str(size)] = {
            arm: {m: _mean_std([r[m] for r in rows]) for m in METRICS} | {"n_seeds": len(rows)}
            for arm, rows in arms.items()
        }
        paired: dict = {}
        complete = (
            model in pooled[size]
            and reference in pooled[size]
            and len(pooled[size][model]) == len(pooled[size]["y"])
            and len(pooled[size][reference]) == len(pooled[size]["y"])
        )
        for name, metric in (("average_precision", average_precision_score), ("roc_auc", None)):
            if not complete:
                paired[name] = None
                raw_p[name].append(float("nan"))
                continue
            d, (lo, hi), p = paired_auc_difference(
                np.concatenate(pooled[size]["y"]),
                np.concatenate(pooled[size][model]),
                np.concatenate(pooled[size][reference]),
                n_boot=n_boot,
                seed=0,
                metric=metric,
                groups=np.concatenate(pooled[size]["g"]),
            )
            seed_deltas = [
                a[name] - b[name] for a, b in zip(cells[size][model], cells[size][reference])
            ]
            paired[name] = {
                "delta_pooled": d,
                "ci95": [lo, hi],
                "p": p,
                "delta_seed_mean": _mean_std(seed_deltas)["mean"],
                "delta_seed_std": _mean_std(seed_deltas)["std"],
                "seeds_model_ahead": int(sum(v > 0 for v in seed_deltas)),
            }
            raw_p[name].append(p)
        out["sizes"][str(size)]["paired"] = paired
        sized.append(size)
    for name, ps in raw_p.items():
        for size, adj in zip(sized, holm_adjusted_p(ps)):
            block = out["sizes"][str(size)]["paired"][name]
            if block is not None:
                block["p_holm_across_n"] = adj
    return out


def _metrics(y_true: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """Score one arm on one test set.

    Args:
        y_true: Test labels.
        p: Predicted probabilities.

    Returns:
        The four :data:`METRICS`.
    """
    m = evaluate_binary(y_true, p)
    return {k: float(getattr(m, k)) for k in METRICS}


def format_report(record: dict) -> str:
    """Render the record, average precision first, then ROC-AUC, then ECE and Brier skill.

    Args:
        record: Output of :func:`run_seed_study`.

    Returns:
        Text with, per metric, a table of seed mean +/- std per arm and size, and the paired
        model-minus-reference line (pooled delta, 95% interval, raw and Holm-adjusted p).
    """
    ref = record["reference"]
    lines = [
        (
            f"seeds {record['seeds']}; panel rows {record['panel_rows']}; "
            f"default rate {record['default_rate']:.3%}; mean +/- std over seeds (ddof=1)"
        )
    ]
    titles = {
        "average_precision": "AVERAGE PRECISION (read first)",
        "roc_auc": "ROC-AUC",
        "ece": "ECE",
        "brier_skill": "BRIER SKILL",
    }
    for metric in METRICS:
        lines.append(f"\n=== {titles[metric]} ===")
        for size, cell in record["sizes"].items():
            lines.append(f"n={size}")
            for arm, vals in cell.items():
                if arm == "paired":
                    continue
                v = vals[metric]
                lines.append(
                    f"  {arm:<10} {v['mean']:.4f} +/- {v['std']:.4f}   per seed "
                    + " ".join(f"{s:.4f}" for s in v["per_seed"])
                )
            pr = cell["paired"].get(metric) if metric in cell["paired"] else None
            if pr:
                lines.append(
                    f"  paired {MODEL_ARM} - {ref}: {pr['delta_pooled']:+.4f} "
                    f"[{pr['ci95'][0]:+.4f}, {pr['ci95'][1]:+.4f}] p={pr['p']:.4f} "
                    f"Holm across n={pr.get('p_holm_across_n', float('nan')):.4f}; "
                    f"seed mean {pr['delta_seed_mean']:+.4f} +/- {pr['delta_seed_std']:.4f}; "
                    f"ahead on {pr['seeds_model_ahead']}/{len(record['seeds'])} seeds"
                )
    if record["skipped"]:
        lines.append(f"\nSKIPPED: {record['skipped']}")
    return "\n".join(lines)


def build_predictors(checkpoint: str, boosters: tuple[str, ...]) -> dict[str, Predictor]:
    """Build the fintfm arm and the out-of-process boosting arms.

    Args:
        checkpoint: fintfm checkpoint path (§27 used ``runs/phase1-5k/mixed.pt``).
        boosters: Names from :data:`fintfm.evaluation.boosting.BOOSTING_MODELS`.

    Returns:
        ``{arm: predictor}`` with fintfm first.
    """
    from fintfm.evaluation.boosting import fit_predict_boosting
    from fintfm.inference.classifier import FinancialTFMClassifier

    def tfm(Xs: np.ndarray, ys: np.ndarray, Xt: np.ndarray, seed: int) -> np.ndarray:
        """Fit the in-context classifier (balanced context, as §27) and score ``Xt``."""
        clf = FinancialTFMClassifier(
            checkpoint, context_strategy="balanced", random_state=seed
        ).fit(Xs, ys)
        return clf.predict_proba(Xt)[:, 1]

    def booster(name: str) -> Predictor:
        """Predictor for one booster, library defaults plus the run seed."""
        key = "random_seed" if name == "catboost" else "random_state"

        def fn(Xs: np.ndarray, ys: np.ndarray, Xt: np.ndarray, seed: int) -> np.ndarray | None:
            """Fit and score in a subprocess; ``None`` if the booster is unavailable."""
            return fit_predict_boosting(name, Xs, ys, Xt, params={key: seed})

        return fn

    return {MODEL_ARM: tfm, **{b: booster(b) for b in boosters}}


def main(argv: list[str] | None = None) -> None:
    """Command line entry: load the Polish panel, run the study, print and write JSON.

    Args:
        argv: Arguments; ``None`` reads ``sys.argv``.
    """
    from fintfm.evaluation.boosting import BOOSTING_MODELS
    from fintfm.evaluation.datasets import load_polish_bankruptcy

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default="runs/phase1-5k/mixed.pt")
    p.add_argument("--horizon", type=int, default=3, help="Polish panel horizon (§27: 3)")
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    p.add_argument("--sizes", type=int, nargs="+", default=list(SIZES))
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--boosters", nargs="+", default=list(BOOSTING_MODELS))
    p.add_argument("--threads", type=int, default=None, help="cap torch CPU threads")
    p.add_argument("--out", default="runs/strong-baselines-186")
    args = p.parse_args(argv)
    if args.threads:
        import torch

        torch.set_num_threads(args.threads)
    if REFERENCE_ARM not in args.boosters:
        p.error(f"--boosters must include the reference arm {REFERENCE_ARM!r}")
    ds = load_polish_bankruptcy(args.horizon)
    print(f"=== §27 across seeds: {ds.name}, {len(ds.y)} rows, default rate {ds.default_rate:.3%}")
    t0 = time.time()
    record = run_seed_study(
        ds.X,
        ds.y,
        build_predictors(args.checkpoint, tuple(args.boosters)),
        sizes=tuple(args.sizes),
        seeds=tuple(args.seeds),
        n_boot=args.n_boot,
    )
    record |= {
        "dataset": ds.name,
        "checkpoint": args.checkpoint,
        "wall_seconds": time.time() - t0,
    }
    out = Path(args.out) / "boosting_seeds.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2))
    print("\n" + format_report(record))
    print(f"\nwrote {out}  ({record['wall_seconds']:.0f}s)")


if __name__ == "__main__":
    main()
