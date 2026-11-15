"""Sample-size x task-difficulty grid (openspec task 39.16).

Why this exists
---------------
A model doing genuine Bayesian in-context learning should show a different learning curve per
difficulty class: an easy task is nearly solved from a few rows, a hard one keeps improving as
context grows. Reading only the endpoint (AUC at the largest context) cannot tell a model that
learns at the Bayes rate from one that plateaus early. This harness crosses context size with
difficulty and reports the whole curve per tier, so the *shape* (``dAUC/dn`` conditional on
difficulty) can be compared.

Design
------
- Task: two equal-variance Gaussian classes in one informative column plus inert companion
  columns, the same family as the 39.3 ceiling probe
  (:func:`fintfm.experiments.capability._make_bayes_task`). Class 1 is shifted by ``mu`` in
  column 0, chosen by :func:`~fintfm.experiments.capability._bayes_optimal_mu` for a target
  Bayes AUC per tier. The true probability is closed form, ``logit P(y=1|x) = mu * x0 -
  mu^2 / 2`` at a 50% class rate, so Bayesian regret (``capability.bayesian_regret``) is
  reported per cell with nothing estimated.
- Difficulty tiers are defined by target Bayes AUC: easy 0.99, medium 0.85, hard 0.65. The
  tier ordering is a property of the generator and is pinned by a test on the empirical Bayes
  AUC, not assumed.
- Rows: context sizes 16, 64, 256, 1024, 4096. Labels are exactly balanced in every context
  and query set, so no cell is single-class and the class rate never confounds the curve.
- Seeds: each (seed, tier, context size) cell has its own context stream; the query set is
  shared across context sizes within a (seed, tier), so the curve differs only in how many
  context rows the arm saw.
- Arms: each supplied checkpoint, an untrained control of the same architecture with seeded
  weights (the floor; CLAUDE.md "Every evaluation needs a floor"), logistic regression and
  histogram gradient boosting. Logistic regression is the Bayes-optimal family here (the true
  log-odds is linear), so it is the sharpest floor.
- Curve summary per arm and tier: AUC and regret at every size, OLS slope of AUC on
  ``log10(n)`` (per decade of context), and the AUC gain between adjacent sizes.

Nothing here trains. Numbers are MEASURED only when produced by the command in the openspec
Progress note on real checkpoints; the unit tests are SMOKE-TEST only.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from fintfm.experiments.capability import _bayes_optimal_mu, bayesian_regret
from fintfm.experiments.irrelevant_features import UNTRAINED_SEED, _score_in_context

if TYPE_CHECKING:
    from fintfm.modeling.model import FinancialTFM

#: Context sizes (rows of the grid).
SIZES: tuple[int, ...] = (16, 64, 256, 1024, 4096)

#: Difficulty tier -> target Bayes-optimal AUC (columns of the grid), easiest first.
TIERS: dict[str, float] = {"easy": 0.99, "medium": 0.85, "hard": 0.65}

#: Feature width: one informative column plus inert companions, as in the 39.3 probe.
WIDTH = 6

#: Query rows scored per cell; large enough that the empirical Bayes AUC is stable.
N_QUERY = 1000


@dataclass
class Cell:
    """One arm at one (context size, difficulty) cell, summarised over seeds."""

    arm: str
    tier: str
    n: int
    auc_mean: float
    auc_std: float
    auc_bayes_mean: float
    regret_mean: float
    regret_norm_mean: float | None
    n_seeds: int


def bayes_probability(x0: np.ndarray, mu: float) -> np.ndarray:
    """True ``P(y=1 | x)`` for the two-Gaussian task at a 50% class rate.

    Args:
        x0: Informative column values.
        mu: Class mean separation.

    Returns:
        Probabilities from ``logit = mu * x0 - mu**2 / 2``.
    """
    return 1.0 / (1.0 + np.exp(-(mu * x0 - 0.5 * mu * mu)))


def make_split(
    seed: int, tier: str, n: int, n_query: int = N_QUERY, width: int = WIDTH
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Draw one context and the shared query set for a (seed, tier, size) cell.

    Args:
        seed: Task seed; the same arguments give bit-identical output.
        tier: Key of :data:`TIERS`.
        n: Context rows.
        n_query: Query rows; independent of ``n``, so the query set is the same at every size.
        width: Feature width, at least 1.

    Returns:
        ``(Xtr, ytr, Xte, yte, p_bayes_te)``; both label vectors are exactly balanced.
    """
    mu = _bayes_optimal_mu(TIERS[tier])
    t = list(TIERS).index(tier)
    ctx_rng = np.random.default_rng(np.random.SeedSequence([seed, t, 0, n]))
    qry_rng = np.random.default_rng(np.random.SeedSequence([seed, t, 1]))

    def draw(rng: np.random.Generator, m: int) -> tuple[np.ndarray, np.ndarray]:
        y = rng.permutation(np.arange(m) % 2).astype(np.int64)
        X = rng.normal(size=(m, width))
        X[:, 0] += mu * y
        return X.astype(np.float32), y

    Xtr, ytr = draw(ctx_rng, n)
    Xte, yte = draw(qry_rng, n_query)
    return Xtr, ytr, Xte, yte, bayes_probability(Xte[:, 0].astype(np.float64), mu)


def curve_shape(ns: list[int], auc: list[float]) -> dict[str, float | list[float]]:
    """Describe one learning curve's shape, not just its endpoint.

    Args:
        ns: Context sizes, at least two distinct values.
        auc: Mean AUC at each size.

    Returns:
        ``slope_per_decade`` (OLS of AUC on ``log10(n)``), ``total_gain`` (last minus first)
        and ``step_gains`` (AUC gain between adjacent sizes, in size order).

    Raises:
        ValueError: If fewer than two distinct sizes are given.
    """
    if len(set(ns)) < 2:
        raise ValueError("a curve shape needs at least two distinct sizes")
    order = np.argsort(ns)
    x = np.log10(np.asarray(ns, dtype=float)[order])
    a = np.asarray(auc, dtype=float)[order]
    return {
        "slope_per_decade": float(np.polyfit(x, a, 1)[0]),
        "total_gain": float(a[-1] - a[0]),
        "step_gains": [float(v) for v in np.diff(a)],
    }


def run(
    model_paths: dict[str, str],
    out_dir: Path | None = None,
    sizes: tuple[int, ...] = SIZES,
    tiers: tuple[str, ...] = tuple(TIERS),
    seeds: tuple[int, ...] = (0, 1, 2),
    n_query: int = N_QUERY,
    models: dict[str, FinancialTFM] | None = None,
) -> dict:
    """Score every arm in every (size, tier) cell over seeds.

    Args:
        model_paths: ``{name: checkpoint path}``; one untrained control is built from the
            first one's architecture. Ignored when ``models`` is given.
        out_dir: If given, ``sample_size_difficulty.json`` is written there.
        sizes: Context sizes to run (restrict for a quick run).
        tiers: Difficulty tiers to run, keys of :data:`TIERS`.
        seeds: Task seeds.
        n_query: Query rows per cell.
        models: Already-built models by name (used by tests); the control is added.

    Returns:
        The record: config, per-cell results, and per-arm per-tier curve shapes (only when at
        least two sizes were run).

    Raises:
        ValueError: On an unknown tier or a model too narrow for :data:`WIDTH`.
    """
    import torch
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression

    from fintfm.modeling.model import FinancialTFM

    for t in tiers:
        if t not in TIERS:
            raise ValueError(f"unknown tier {t!r}; choose from {list(TIERS)}")
    loaded = (
        dict(models)
        if models is not None
        else {name: FinancialTFM.load(p) for name, p in model_paths.items()}
    )
    if loaded:
        torch.manual_seed(UNTRAINED_SEED)  # seeded: an unseeded floor moves by 0.2 (§42)
        loaded["untrained_control"] = FinancialTFM(next(iter(loaded.values())).cfg)
    for name, m in loaded.items():
        if WIDTH > m.cfg.max_features:
            raise ValueError(f"width {WIDTH} exceeds {name}'s max_features {m.cfg.max_features}")

    acc: dict[tuple[str, str, int], dict[str, list]] = {}
    for seed in seeds:
        for tier in tiers:
            for n in sizes:
                Xtr, ytr, Xte, yte, pb = make_split(seed, tier, n, n_query)
                preds: dict[str, np.ndarray] = {
                    name: _score_in_context(m, Xtr, ytr, Xte, max(sizes), seed)
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
                    r = bayesian_regret(yte, pb, p)
                    c = acc.setdefault((arm, tier, n), {k: [] for k in ("auc", "b", "r", "rn")})
                    c["auc"].append(r["auc_model"])
                    c["b"].append(r["auc_bayes"])
                    c["r"].append(r["regret"])
                    c["rn"].append(r["regret_norm"])
        print(f"  seed {seed} done", flush=True)

    cells = []
    for (arm, tier, n), c in acc.items():
        auc = np.array(c["auc"])
        rn = [v for v in c["rn"] if v is not None]
        cells.append(
            Cell(
                arm=arm,
                tier=tier,
                n=n,
                auc_mean=float(auc.mean()),
                auc_std=float(auc.std(ddof=1)) if len(auc) > 1 else float("nan"),
                auc_bayes_mean=float(np.mean(c["b"])),
                regret_mean=float(np.mean(c["r"])),
                regret_norm_mean=float(np.mean(rn)) if rn else None,
                n_seeds=len(auc),
            )
        )
    shapes: dict[str, dict[str, dict]] = {}
    if len(set(sizes)) >= 2:
        for arm in sorted({c.arm for c in cells}):
            for tier in tiers:
                pts = sorted(
                    (c for c in cells if c.arm == arm and c.tier == tier), key=lambda c: c.n
                )
                shapes.setdefault(arm, {})[tier] = curve_shape(
                    [c.n for c in pts], [c.auc_mean for c in pts]
                )
    record = {
        "task": "two Gaussians, one informative column; difficulty = target Bayes AUC",
        "config": {
            "models": model_paths,
            "sizes": list(sizes),
            "tiers": {t: TIERS[t] for t in tiers},
            "seeds": list(seeds),
            "n_query": n_query,
            "width": WIDTH,
        },
        "results": [asdict(c) for c in cells],
        "shapes": shapes,
    }
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "sample_size_difficulty.json").write_text(json.dumps(record, indent=2))
    return record


def summarise(record: dict) -> str:
    """Render one AUC grid per arm (rows = size, columns = tier), regret and curve shapes.

    Args:
        record: Output of :func:`run`.

    Returns:
        A printable multi-line string.
    """
    rows = record["results"]
    sizes = sorted({r["n"] for r in rows})
    tiers = list(record["config"]["tiers"])
    lines: list[str] = []
    for arm in sorted({r["arm"] for r in rows}):
        lines.append(f"{arm}: AUC / regret R = AUC_Bayes - AUC (rows: context size)")
        lines.append(f"{'n':>7} " + " ".join(f"{t:>20}" for t in tiers))
        for n in sizes:
            cells = []
            for t in tiers:
                h = [r for r in rows if r["arm"] == arm and r["tier"] == t and r["n"] == n]
                cells.append(f"{h[0]['auc_mean']:.3f} / {h[0]['regret_mean']:+.3f}" if h else "-")
            lines.append(f"{n:>7} " + " ".join(f"{c:>20}" for c in cells))
        lines.append("")
    if record["shapes"]:
        lines.append("Curve shape per arm and tier (AUC per decade of context; step gains):")
        for arm, by_tier in record["shapes"].items():
            for t, s in by_tier.items():
                steps = " ".join(f"{g:+.3f}" for g in s["step_gains"])
                lines.append(f"{arm:>22} {t:>7} slope {s['slope_per_decade']:+.4f}  steps {steps}")
    lines += [
        "",
        "Reading it: compare shapes, not endpoints. Logistic regression is the Bayes-optimal",
        "family on this task and the untrained control is the floor. Genuine Bayesian ICL would",
        "show the easy tier saturating early and the hard tier still gaining at large n.",
    ]
    return "\n".join(lines)


def plot_curves(record: dict, out_dir: Path) -> list[Path]:
    """Write one AUC-vs-context-size plot per tier (log x) next to the JSON.

    Args:
        record: Output of :func:`run`.
        out_dir: Directory that receives ``sample_size_difficulty_<tier>.svg``.

    Returns:
        The paths of the written SVG files, in tier order.
    """
    from fintfm.evaluation.svgplot import line_plot

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for tier in record["config"]["tiers"]:
        rows = [r for r in record["results"] if r["tier"] == tier]
        series: dict[str, list[tuple[float, float]]] = {}
        for r in rows:
            series.setdefault(r["arm"], []).append((float(r["n"]), r["auc_mean"]))
        bayes = float(np.mean([r["auc_bayes_mean"] for r in rows]))
        svg = line_plot(
            series,
            title=f"ROC-AUC vs context size, {tier} tier",
            x_label="context rows n (log scale)",
            y_label="ROC-AUC (mean over seeds)",
            log_x=True,
            reference_lines={f"Bayes AUC {bayes:.3f}": bayes},
        )
        path = out_dir / f"sample_size_difficulty_{tier}.svg"
        path.write_text(svg)
        paths.append(path)
    return paths


def main() -> None:
    """Command-line entry point: ``python -m fintfm.experiments.sample_size_difficulty``."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", type=str, required=True, help="name=path pairs, comma separated")
    p.add_argument("--out", type=str, default="runs/sample_size_difficulty")
    p.add_argument("--seeds", type=str, default="0,1,2")
    p.add_argument("--sizes", type=str, default=",".join(str(s) for s in SIZES))
    p.add_argument("--tiers", type=str, default=",".join(TIERS))
    p.add_argument("--n-query", type=int, default=N_QUERY)
    p.add_argument("--plot", action="store_true", help="also write one SVG per tier")
    args = p.parse_args()
    paths = dict(pair.split("=", 1) for pair in args.models.split(",") if pair)
    record = run(
        paths,
        Path(args.out),
        sizes=tuple(int(s) for s in args.sizes.split(",")),
        tiers=tuple(args.tiers.split(",")),
        seeds=tuple(int(s) for s in args.seeds.split(",")),
        n_query=args.n_query,
    )
    print("\n" + summarise(record))
    if args.plot:
        for path in plot_curves(record, Path(args.out)):
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
