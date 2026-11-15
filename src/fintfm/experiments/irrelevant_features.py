"""Irrelevant-feature robustness curve (openspec tasks 42.1 and 39.13).

Why this exists
---------------
``docs/results/FINDINGS.md`` §49/§50 measured the model losing accuracy as feature count grows
on a linear probe, but there every added column was *also* part of the signal. This harness
separates the two: the task is fixed at ``y = f(x0, x1)`` and only **pure-noise columns,
independent of y**, are added, up to a total width of ``d = 100``. A model that reads column
identity and weights columns would stay flat; one that averages over columns dilutes the two
informative ones as ``d`` grows. The question 42.1 asks is whether the architectural split seen
between ``fin00`` (pure SCM prior) and ``fin10`` (pure financial prior) in §74 shows up here too.

Design
------
- One fixed task per seed. The two informative columns, their label and the row split are drawn
  once, and the noise columns at each width are the **first k columns of one noise matrix**, so
  the curve's points differ only in how many irrelevant columns are present, never in the task
  or in the columns already there.
- Arms: each supplied checkpoint, an **untrained control** of the same architecture with seeded
  weights (the floor; CLAUDE.md "Every evaluation needs a floor"), logistic regression, and
  histogram gradient boosting. Logistic regression is near-chance on the interaction part by
  construction; gradient boosting is the arm known to cope with irrelevant columns.
- Reports average precision beside ROC-AUC (CLAUDE.md: AP first at low base rates), and the
  mean predicted probability beside the observed rate for the in-context arms.
- The slope is an ordinary-least-squares fit of AUC on the added column count ``k`` (reported per
  10 columns) and on ``log10(d)`` (per decade of width), plus the total change from the
  narrowest to the widest point. It is descriptive: a fit through a handful of grid points.

Nothing here trains. Run numbers are labelled MEASURED only when produced by the command in
the openspec Progress note, on real checkpoints; the unit tests are SMOKE-TEST only.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from fintfm.evaluation.coverage import Coverage

if TYPE_CHECKING:
    from fintfm.modeling.model import FinancialTFM

#: Total widths ``d`` (relevant plus irrelevant columns) of the curve; 42.1 asks for up to 100.
WIDTHS: tuple[int, ...] = (2, 5, 10, 20, 40, 70, 100)

#: Informative columns of the fixed task.
N_RELEVANT = 2

#: Seed for the untrained control's weights; shared with the capability suite's floor.
UNTRAINED_SEED = 20260910


@dataclass
class CurvePoint:
    """One arm at one width, summarised over seeds."""

    arm: str
    d: int
    k_irrelevant: int
    auc_mean: float
    auc_std: float
    ap_mean: float
    mean_pred: float | None
    observed_rate: float
    n_seeds: int


def relevant_score(x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
    """The fixed rule ``f(x0, x1)``: an additive part plus an interaction.

    ``x0 + x1 + 0.5 * x0 * x1``. Logistic regression captures the additive part only, a tree
    ensemble or an interaction-capable model can capture both, and neither ceiling is 1.0
    exactly, so the curve is read against the arms rather than against a constant.

    Args:
        x0: First informative column.
        x1: Second informative column.

    Returns:
        Real-valued score; the label thresholds it.
    """
    return x0 + x1 + 0.5 * x0 * x1


def make_task(
    seed: int, n: int = 2000, max_irrelevant: int = max(WIDTHS) - N_RELEVANT, rate: float = 0.25
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw one fixed task: informative columns, label, and a pool of noise columns.

    Args:
        seed: Random seed. The same seed gives bit-identical output.
        n: Total rows; the first half is context, the second half the query set.
        max_irrelevant: Width of the noise pool; a point with ``k`` noise columns uses the first
            ``k`` of them.
        rate: Positive-class rate, set by thresholding the score at that upper quantile.

    Returns:
        ``(X_relevant, y, noise)`` with shapes ``(n, 2)``, ``(n,)`` and ``(n, max_irrelevant)``.
        The noise is drawn from a separate stream, so it is independent of ``y`` by construction
        and changing ``max_irrelevant`` never changes the relevant columns or the label.
    """
    rel_rng, noise_rng = (np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(2))
    X_rel = rel_rng.normal(size=(n, N_RELEVANT)).astype(np.float32)
    score = relevant_score(X_rel[:, 0], X_rel[:, 1])
    y = (score >= np.quantile(score, 1 - rate)).astype(np.int64)
    # Drawn column-major so the first k columns do not depend on the pool width.
    noise = noise_rng.normal(size=(max_irrelevant, n)).astype(np.float32).T
    return X_rel, y, noise


def build_table(X_rel: np.ndarray, noise: np.ndarray, k: int) -> np.ndarray:
    """Concatenate the informative columns with the first ``k`` noise columns.

    Args:
        X_rel: ``(n, 2)`` informative columns.
        noise: ``(n, >= k)`` noise pool from :func:`make_task`.
        k: Irrelevant columns to append.

    Returns:
        ``(n, 2 + k)`` float32 table, informative columns first.

    Raises:
        ValueError: If the pool holds fewer than ``k`` columns.
    """
    if k > noise.shape[1]:
        raise ValueError(f"asked for {k} noise columns, pool has {noise.shape[1]}")
    return np.concatenate([X_rel, noise[:, :k]], axis=1)


def curve_slope(d: list[int], auc: list[float]) -> dict[str, float]:
    """Least-squares slope of AUC against width, plus the total change.

    Args:
        d: Total widths, at least two distinct values.
        auc: Mean AUC at each width.

    Returns:
        ``slope_per_10_columns`` (AUC change per 10 added columns, OLS on ``k = d - 2``),
        ``slope_per_decade`` (OLS on ``log10(d)``) and ``total_change`` (last minus first).

    Raises:
        ValueError: If fewer than two distinct widths are given.
    """
    if len(set(d)) < 2:
        raise ValueError("a slope needs at least two distinct widths")
    k = np.asarray(d, dtype=float) - N_RELEVANT
    a = np.asarray(auc, dtype=float)
    per_col = np.polyfit(k, a, 1)[0]
    per_dec = np.polyfit(np.log10(np.asarray(d, dtype=float)), a, 1)[0]
    order = np.argsort(d)
    return {
        "slope_per_10_columns": float(per_col * 10),
        "slope_per_decade": float(per_dec),
        "total_change": float(a[order[-1]] - a[order[0]]),
    }


def _score_in_context(
    model: FinancialTFM,
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xte: np.ndarray,
    max_context: int,
    seed: int,
) -> np.ndarray:
    """Positive-class probability for the query rows through the public classifier."""
    from fintfm.inference.classifier import FinancialTFMClassifier

    clf = FinancialTFMClassifier(
        model,
        max_context=max_context,
        context_strategy="uniform",
        feature_transform="rank",
        random_state=seed,
    ).fit(Xtr, ytr)
    return clf.predict_proba(Xte)[:, 1]


def run(
    model_paths: dict[str, str],
    out_dir: Path | None = None,
    widths: tuple[int, ...] = WIDTHS,
    seeds: tuple[int, ...] = (0, 1, 2),
    n: int = 2000,
    max_context: int = 1000,
    rate: float = 0.25,
    models: dict[str, FinancialTFM] | None = None,
) -> dict:
    """Score every arm at every width over seeds and record the curve.

    Args:
        model_paths: ``{name: checkpoint path}``; each is loaded, and one untrained control is
            built from the first one's architecture. Ignored when ``models`` is given.
        out_dir: If given, ``irrelevant_features.json`` is written there.
        widths: Total widths ``d``; each at least 2.
        seeds: Task seeds; the whole curve is repeated per seed.
        n: Rows per task, half context and half query.
        max_context: Context rows handed to the in-context arms.
        rate: Positive-class rate.
        models: Already-built models by name (used by tests); the untrained control is added.

    Returns:
        The record: config, per-point results, and per-arm slopes.

    Raises:
        ValueError: If a width is below the informative width, or exceeds a model's
            ``max_features``.
    """
    import torch
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score

    from fintfm.modeling.model import FinancialTFM

    if min(widths) < N_RELEVANT:
        raise ValueError(f"width must be at least {N_RELEVANT}")
    loaded = (
        dict(models)
        if models is not None
        else {name: FinancialTFM.load(p) for name, p in model_paths.items()}
    )
    if loaded:
        torch.manual_seed(UNTRAINED_SEED)  # seeded: an unseeded floor moves by 0.2 (§42)
        loaded["untrained_control"] = FinancialTFM(next(iter(loaded.values())).cfg)
    for name, m in loaded.items():
        if max(widths) > m.cfg.max_features:
            raise ValueError(
                f"width {max(widths)} exceeds {name}'s max_features {m.cfg.max_features}"
            )

    cells: dict[tuple[str, int], dict[str, list[float]]] = {}
    nc = n // 2
    cov = Coverage()
    for seed in seeds:
        X_rel, y, noise = make_task(seed, n, max(widths) - N_RELEVANT, rate)
        ytr, yte = y[:nc], y[nc:]
        unit = f"seed{seed}"
        cov.attempt(unit)
        if len(np.unique(yte)) < 2 or len(np.unique(ytr)) < 2:
            cov.skip(unit, "one class in the context or query half")
            continue
        cov.scored(unit)
        for d in widths:
            X = build_table(X_rel, noise, d - N_RELEVANT)
            Xtr, Xte = X[:nc], X[nc:]
            preds: dict[str, np.ndarray] = {
                name: _score_in_context(m, Xtr, ytr, Xte, max_context, seed)
                for name, m in loaded.items()
            }
            preds["logistic_regression"] = (
                LogisticRegression(max_iter=1000).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
            )
            preds["gradient_boosting"] = (
                HistGradientBoostingClassifier(max_iter=100, random_state=seed)
                .fit(Xtr, ytr)
                .predict_proba(Xte)[:, 1]
            )
            for arm, p in preds.items():
                c = cells.setdefault((arm, d), {"auc": [], "ap": [], "mean_pred": [], "rate": []})
                c["auc"].append(float(roc_auc_score(yte, p)))
                c["ap"].append(float(average_precision_score(yte, p)))
                c["mean_pred"].append(float(np.mean(p)))
                c["rate"].append(float(np.mean(yte)))
        print(f"  seed {seed} done", flush=True)

    in_context = set(loaded)
    points = []
    for (arm, d), c in cells.items():
        auc = np.array(c["auc"])
        points.append(
            CurvePoint(
                arm=arm,
                d=d,
                k_irrelevant=d - N_RELEVANT,
                auc_mean=float(auc.mean()),
                auc_std=float(auc.std(ddof=1)) if len(auc) > 1 else float("nan"),
                ap_mean=float(np.mean(c["ap"])),
                mean_pred=float(np.mean(c["mean_pred"])) if arm in in_context else None,
                observed_rate=float(np.mean(c["rate"])),
                n_seeds=len(auc),
            )
        )
    slopes: dict[str, dict[str, float]] = {}
    for arm in sorted({p.arm for p in points}):
        arm_pts = sorted((p for p in points if p.arm == arm), key=lambda p: p.d)
        if len({p.d for p in arm_pts}) >= 2:
            slopes[arm] = curve_slope([p.d for p in arm_pts], [p.auc_mean for p in arm_pts])
    record = {
        "task": "y = thresholded(x0 + x1 + 0.5*x0*x1); k independent N(0,1) columns added",
        "config": {
            "models": model_paths,
            "widths": list(widths),
            "seeds": list(seeds),
            "n": n,
            "max_context": max_context,
            "rate": rate,
        },
        "results": [asdict(p) for p in points],
        "slopes": slopes,
        "coverage": cov.to_dict(),
    }
    print(cov.summary("seeds"), flush=True)
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "irrelevant_features.json").write_text(json.dumps(record, indent=2))
    return record


def summarise(record: dict) -> str:
    """Render the AUC-vs-width table, the slopes, and how to read them.

    Args:
        record: Output of :func:`run`.

    Returns:
        A printable multi-line string.
    """
    rows = record["results"]
    widths = sorted({r["d"] for r in rows})
    arms = sorted({r["arm"] for r in rows})
    lines = [f"{'arm (ROC-AUC)':>22} " + " ".join(f"{'d=' + str(d):>13}" for d in widths)]
    for arm in arms:
        cells = []
        for d in widths:
            hit = [r for r in rows if r["arm"] == arm and r["d"] == d]
            cells.append(f"{hit[0]['auc_mean']:.3f}±{hit[0]['auc_std']:.3f}" if hit else "-")
        lines.append(f"{arm:>22} " + " ".join(f"{c:>13}" for c in cells))
    lines += [
        "",
        f"{'arm (avg precision)':>22} " + " ".join(f"{'d=' + str(d):>13}" for d in widths),
    ]
    for arm in arms:
        cells = [
            f"{h[0]['ap_mean']:.3f}"
            if (h := [r for r in rows if r["arm"] == arm and r["d"] == d])
            else "-"
            for d in widths
        ]
        lines.append(f"{arm:>22} " + " ".join(f"{c:>13}" for c in cells))
    lines += ["", "Mean predicted probability beside the observed rate (in-context arms):"]
    for arm in arms:
        pts = [r for r in rows if r["arm"] == arm and r["mean_pred"] is not None]
        if pts:
            lines.append(
                f"{arm:>22} observed {pts[0]['observed_rate']:.3f}; predicted "
                + " ".join(f"{r['mean_pred']:.3f}" for r in sorted(pts, key=lambda r: r["d"]))
            )
    lines += ["", f"{'slope':>22} {'per 10 cols':>12} {'per decade':>11} {'d_min->d_max':>13}"]
    for arm, s in record["slopes"].items():
        lines.append(
            f"{arm:>22} {s['slope_per_10_columns']:>+12.4f} {s['slope_per_decade']:>+11.4f} "
            f"{s['total_change']:>+13.4f}"
        )
    lines += [
        "",
        "Reading it: the untrained control is the floor and gradient boosting the arm known to",
        "tolerate irrelevant columns. A negative slope on a checkpoint that the floor does not",
        "share means the added columns dilute the two informative ones. Compare fin00 with",
        "fin10 (FINDINGS §74): the same split here would point at the same cause.",
    ]
    return "\n".join(lines)


def plot_curves(record: dict, out_dir: Path) -> Path:
    """Write the AUC-vs-total-width plot (one line per arm) next to the JSON.

    Args:
        record: Output of :func:`run`.
        out_dir: Directory that receives ``irrelevant_features.svg``.

    Returns:
        The path of the written SVG.
    """
    from fintfm.evaluation.svgplot import line_plot

    series: dict[str, list[tuple[float, float]]] = {}
    for r in record["results"]:
        series.setdefault(r["arm"], []).append((float(r["d"]), r["auc_mean"]))
    svg = line_plot(
        series,
        title="ROC-AUC vs total width (2 relevant columns plus noise)",
        x_label="total columns d",
        y_label="ROC-AUC (mean over seeds)",
        reference_lines={"chance 0.5": 0.5},
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "irrelevant_features.svg"
    path.write_text(svg)
    return path


def main() -> None:
    """Command-line entry point: ``python -m fintfm.experiments.irrelevant_features``."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--models",
        type=str,
        required=True,
        help="comma-separated name=path pairs, e.g. fin00=runs/a.pt,fin10=runs/b.pt",
    )
    p.add_argument("--out", type=str, default="runs/irrelevant_features")
    p.add_argument("--seeds", type=str, default="0,1,2")
    p.add_argument("--widths", type=str, default=",".join(str(w) for w in WIDTHS))
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--max-context", type=int, default=1000)
    p.add_argument("--rate", type=float, default=0.25)
    p.add_argument("--plot", action="store_true", help="also write irrelevant_features.svg")
    args = p.parse_args()
    paths = dict(pair.split("=", 1) for pair in args.models.split(",") if pair)
    record = run(
        paths,
        Path(args.out),
        widths=tuple(int(w) for w in args.widths.split(",")),
        seeds=tuple(int(s) for s in args.seeds.split(",")),
        n=args.n,
        max_context=args.max_context,
        rate=args.rate,
    )
    print("\n" + summarise(record))
    if args.plot:
        print(f"wrote {plot_curves(record, Path(args.out))}")


if __name__ == "__main__":
    main()
