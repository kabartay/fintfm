"""Context sufficiency against information content, not row count (openspec task 39.8).

Why this exists
---------------
Plotting AUC against the number of context rows treats a 1,024-row context holding 10
positives like one holding 512. What an in-context learner can extract from a context is
limited by how much *label information* it carries, and by how much the task demands. A
curve against row count alone cannot separate genuine in-context learning (AUC tracks the
information supplied, and the information a task needs grows with its complexity) from a
strong global prior that scores well on average whatever the context holds (AUC flat in both
row count and information).

Design
------
- Tasks of controlled information requirement, noise-free so the Bayes classifier is the
  generating rule and the true probability is the label itself (``capability.bayesian_regret``
  is then ``1 - AUC`` on the query rows; nothing is estimated):
  ``linear1`` ``1[x0 > t]`` (few examples suffice), ``and2`` ``1[x0 + x1 > t]`` and
  ``interaction4`` ``1[x0*x1*x2*x3 > t]`` (a 4-way interaction, many examples). ``t`` is set
  so the task's positive rate equals the cell's base rate.
- Information measures of a context with ``n`` rows and ``k`` positives (:func:`information`):
  ``label_bits = n * H(k/n)`` (the label entropy the context carries, in bits; the primary
  axis), ``n_minority = min(k, n - k)`` and ``n_eff = 2 n p (1 - p)`` (the balanced-context
  row count with the same harmonic information about both classes).
- Row count and information are decoupled by crossing rows with base rate: equal rows at
  different base rates carry different bits, and unequal rows at different base rates can
  carry (nearly) equal bits (:func:`matched_cells`). The context holds exactly
  ``round(n * rate)`` positives (at least one of each class); the query set is balanced and
  shared across the row counts of a (seed, task, rate), so no cell is single-class.
- Arms: each supplied checkpoint, a seeded untrained control (the floor; CLAUDE.md "Every
  evaluation needs a floor"), logistic regression and histogram gradient boosting.
- Summary per arm and task: AUC and regret per cell sorted by ``label_bits``, and the R^2 of
  a straight-line fit of AUC on ``log10(label_bits)`` versus on ``log10(rows)``. If
  information is the better axis, the first R^2 exceeds the second.

Nothing here trains. Numbers are MEASURED only when produced by the command in the openspec
Progress note on real checkpoints; the unit tests are SMOKE-TEST only.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from fintfm.experiments.capability import bayesian_regret
from fintfm.experiments.irrelevant_features import UNTRAINED_SEED, _score_in_context

if TYPE_CHECKING:
    from fintfm.modeling.model import FinancialTFM

#: Context row counts.
ROWS: tuple[int, ...] = (64, 256, 1024, 4096)

#: Positive rates of the task (and of its context).
RATES: tuple[float, ...] = (0.5, 0.25, 0.1, 0.03)

#: Task name -> number of columns the label depends on, easiest first.
TASKS: dict[str, int] = {"linear1": 1, "and2": 2, "interaction4": 4}

#: Feature width: relevant columns plus inert N(0,1) companions.
WIDTH = 8

#: Query rows (balanced) scored per cell.
N_QUERY = 1000


@dataclass
class Cell:
    """One arm at one (task, rows, rate) cell, summarised over seeds."""

    arm: str
    task: str
    rows: int
    rate: float
    n_pos: int
    label_bits: float
    n_eff: float
    auc_mean: float
    auc_std: float
    regret_mean: float
    n_seeds: int


def information(y: np.ndarray) -> dict[str, float]:
    """Label-information measures of one context.

    Args:
        y: Binary context labels.

    Returns:
        ``n`` (rows), ``n_pos``, ``n_minority``, ``label_bits`` (``n * H(n_pos / n)``, bits;
        0 for a single-class context) and ``n_eff`` (``2 n p (1 - p)``).
    """
    y = np.asarray(y)
    n = len(y)
    k = int(y.sum())
    p = k / n if n else 0.0
    h = 0.0 if p in (0.0, 1.0) else float(-(p * np.log2(p) + (1 - p) * np.log2(1 - p)))
    return {
        "n": n,
        "n_pos": k,
        "n_minority": min(k, n - k),
        "label_bits": n * h,
        "n_eff": 2.0 * n * p * (1.0 - p),
    }


def _score_fn(task: str, X: np.ndarray) -> np.ndarray:
    if task == "linear1":
        return X[:, 0]
    if task == "and2":
        return X[:, 0] + X[:, 1]
    return X[:, 0] * X[:, 1] * X[:, 2] * X[:, 3]


@cache
def _threshold(task: str, rate: float) -> float:
    """Threshold giving positive rate ``rate`` (fixed 200k-draw Monte Carlo, deterministic)."""
    s = _score_fn(task, np.random.default_rng(12345).normal(size=(200_000, 4)))
    return float(np.quantile(s, 1.0 - rate))


def _draw_labelled(
    rng: np.random.Generator, task: str, rate: float, n_pos: int, n_neg: int, width: int
) -> tuple[np.ndarray, np.ndarray]:
    """Rejection-sample exactly ``n_pos`` positives and ``n_neg`` negatives, shuffled."""
    t = _threshold(task, rate)
    xs_p: list[np.ndarray] = []
    xs_n: list[np.ndarray] = []
    got_p = got_n = 0
    while got_p < n_pos or got_n < n_neg:
        X = rng.normal(size=(8192, width))
        pos = _score_fn(task, X) > t
        if got_p < n_pos:
            xs_p.append(X[pos])
            got_p += int(pos.sum())
        if got_n < n_neg:
            xs_n.append(X[~pos])
            got_n += int((~pos).sum())
    X = np.concatenate([np.concatenate(xs_p)[:n_pos], np.concatenate(xs_n)[:n_neg]])
    y = np.concatenate([np.ones(n_pos, np.int64), np.zeros(n_neg, np.int64)])
    order = rng.permutation(len(y))
    return X[order].astype(np.float32), y[order]


def make_split(
    seed: int, task: str, rows: int, rate: float, n_query: int = N_QUERY, width: int = WIDTH
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Draw one context and the shared balanced query set for a cell.

    Args:
        seed: Task seed; the same arguments give bit-identical output.
        task: Key of :data:`TASKS`.
        rows: Context rows.
        rate: Positive rate; the context holds ``round(rows * rate)`` positives, clipped to
            ``[1, rows - 1]``.
        n_query: Query rows, half positive; the same at every ``rows`` for a given
            (seed, task, rate) (a different rate is a different task, so its own query set).
        width: Feature width, at least 4.

    Returns:
        ``(Xtr, ytr, Xte, yte)``.

    Raises:
        ValueError: On an unknown task, a width below 4, or ``rows`` below 2.
    """
    if task not in TASKS:
        raise ValueError(f"unknown task {task!r}; choose from {list(TASKS)}")
    if width < 4:
        raise ValueError(f"width {width} too small; the threshold draw needs 4 columns")
    if rows < 2:
        raise ValueError("a context needs at least two rows")
    k = min(max(round(rows * rate), 1), rows - 1)
    ti = list(TASKS).index(task)
    ctx = np.random.default_rng(np.random.SeedSequence([seed, ti, 0, rows, round(rate * 1e6)]))
    qry = np.random.default_rng(np.random.SeedSequence([seed, ti, 1, round(rate * 1e6)]))
    Xtr, ytr = _draw_labelled(ctx, task, rate, k, rows - k, width)
    Xte, yte = _draw_labelled(qry, task, rate, n_query // 2, n_query - n_query // 2, width)
    return Xtr, ytr, Xte, yte


def _bits(n: int, rate: float) -> float:
    k = min(max(round(n * rate), 1), n - 1)
    return information(np.r_[np.ones(k), np.zeros(n - k)])["label_bits"]


def matched_cells(
    rows: tuple[int, ...] = ROWS, rates: tuple[float, ...] = RATES, rel_tol: float = 0.15
) -> dict[str, list[tuple[tuple[int, float], tuple[int, float]]]]:
    """Show the design decouples rows from information.

    Args:
        rows: Row counts of the grid.
        rates: Base rates of the grid.
        rel_tol: Relative tolerance on ``label_bits`` for the equal-information pairs.

    Returns:
        ``same_rows`` (pairs of ``(rows, rate)`` cells with equal rows and different bits)
        and ``same_bits`` (pairs with different rows and bits within ``rel_tol``).
    """
    cells = [(n, r) for n in rows for r in rates]
    same_rows, same_bits = [], []
    for i, a in enumerate(cells):
        for b in cells[i + 1 :]:
            ba, bb = _bits(*a), _bits(*b)
            if a[0] == b[0] and abs(ba - bb) > rel_tol * max(ba, bb):
                same_rows.append((a, b))
            if a[0] != b[0] and abs(ba - bb) <= rel_tol * max(ba, bb):
                same_bits.append((a, b))
    return {"same_rows": same_rows, "same_bits": same_bits}


def axis_fit(x: list[float], auc: list[float]) -> float | None:
    """R^2 of a straight-line fit of AUC on ``log10(x)``.

    Args:
        x: Positive axis values (row counts or bits).
        auc: AUC at each.

    Returns:
        The R^2, or ``None`` when ``x`` or ``auc`` is constant or fewer than three points.
    """
    if len(x) < 3:
        return None
    lx = np.log10(np.asarray(x, dtype=float))
    a = np.asarray(auc, dtype=float)
    if np.ptp(lx) == 0 or np.ptp(a) == 0:
        return None
    return float(np.corrcoef(lx, a)[0, 1] ** 2)


def run(
    model_paths: dict[str, str],
    out_dir: Path | None = None,
    rows: tuple[int, ...] = ROWS,
    rates: tuple[float, ...] = RATES,
    tasks: tuple[str, ...] = tuple(TASKS),
    seeds: tuple[int, ...] = (0, 1, 2),
    n_query: int = N_QUERY,
    models: dict[str, FinancialTFM] | None = None,
) -> dict:
    """Score every arm in every (task, rows, rate) cell over seeds.

    Args:
        model_paths: ``{name: checkpoint path}``; one untrained control is built from the
            first one's architecture. Ignored when ``models`` is given.
        out_dir: If given, ``context_information.json`` is written there.
        rows: Context row counts.
        rates: Positive rates.
        tasks: Keys of :data:`TASKS`.
        seeds: Task seeds.
        n_query: Query rows per cell.
        models: Already-built models by name (used by tests); the control is added.

    Returns:
        The record: config, the decoupling pairs, per-cell results and per-arm per-task
        axis fits.

    Raises:
        ValueError: On an unknown task or a model too narrow for :data:`WIDTH`.
    """
    import torch
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression

    from fintfm.modeling.model import FinancialTFM

    for t in tasks:
        if t not in TASKS:
            raise ValueError(f"unknown task {t!r}; choose from {list(TASKS)}")
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

    acc: dict[tuple[str, str, int, float], dict[str, list]] = {}
    info_of: dict[tuple[str, int, float], dict[str, list]] = {}
    for seed in seeds:
        for task in tasks:
            for n in rows:
                for rate in rates:
                    Xtr, ytr, Xte, yte = make_split(seed, task, n, rate, n_query)
                    inf = information(ytr)
                    d = info_of.setdefault((task, n, rate), {"pos": [], "bits": [], "eff": []})
                    d["pos"].append(inf["n_pos"])
                    d["bits"].append(inf["label_bits"])
                    d["eff"].append(inf["n_eff"])
                    preds: dict[str, np.ndarray] = {
                        name: _score_in_context(m, Xtr, ytr, Xte, max(rows), seed)
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
                        # noise-free task: the Bayes probability is the label itself
                        r = bayesian_regret(yte, yte.astype(float), p)
                        c = acc.setdefault((arm, task, n, rate), {"auc": [], "r": []})
                        c["auc"].append(r["auc_model"])
                        c["r"].append(r["regret"])
        print(f"  seed {seed} done", flush=True)

    cells = []
    for (arm, task, n, rate), c in acc.items():
        auc = np.array(c["auc"])
        d = info_of[(task, n, rate)]
        cells.append(
            Cell(
                arm=arm,
                task=task,
                rows=n,
                rate=rate,
                n_pos=round(np.mean(d["pos"])),
                label_bits=float(np.mean(d["bits"])),
                n_eff=float(np.mean(d["eff"])),
                auc_mean=float(auc.mean()),
                auc_std=float(auc.std(ddof=1)) if len(auc) > 1 else float("nan"),
                regret_mean=float(np.mean(c["r"])),
                n_seeds=len(auc),
            )
        )
    fits: dict[str, dict[str, dict]] = {}
    for arm in sorted({c.arm for c in cells}):
        for task in tasks:
            pts = [c for c in cells if c.arm == arm and c.task == task]
            fits.setdefault(arm, {})[task] = {
                "r2_log_bits": axis_fit([c.label_bits for c in pts], [c.auc_mean for c in pts]),
                "r2_log_rows": axis_fit([c.rows for c in pts], [c.auc_mean for c in pts]),
            }
    record = {
        "task": "noise-free threshold tasks of rising complexity; axis = label bits in context",
        "config": {
            "models": model_paths,
            "rows": list(rows),
            "rates": list(rates),
            "tasks": {t: TASKS[t] for t in tasks},
            "seeds": list(seeds),
            "n_query": n_query,
            "width": WIDTH,
        },
        "decoupling": matched_cells(rows, rates),
        "results": [asdict(c) for c in cells],
        "fits": fits,
    }
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "context_information.json").write_text(json.dumps(record, indent=2))
    return record


def summarise(record: dict) -> str:
    """Render, per arm and task, AUC and regret sorted by label bits, and the axis fits.

    Args:
        record: Output of :func:`run`.

    Returns:
        A printable multi-line string.
    """
    rows = record["results"]
    lines: list[str] = []

    def fmt(v: float | None) -> str:
        return "n/a" if v is None else f"{v:.2f}"

    for arm in sorted({r["arm"] for r in rows}):
        for task in record["config"]["tasks"]:
            pts = sorted(
                (r for r in rows if r["arm"] == arm and r["task"] == task),
                key=lambda r: r["label_bits"],
            )
            f = record["fits"][arm][task]
            lines.append(
                f"{arm} / {task}: R2 of AUC on log10(bits) {fmt(f['r2_log_bits'])}, "
                f"on log10(rows) {fmt(f['r2_log_rows'])}"
            )
            lines.append(
                f"{'bits':>9} {'rows':>6} {'rate':>6} {'n_pos':>6} {'n_eff':>8} "
                f"{'AUC':>7} {'regret':>7}"
            )
            for r in pts:
                lines.append(
                    f"{r['label_bits']:>9.1f} {r['rows']:>6} {r['rate']:>6.2f} {r['n_pos']:>6} "
                    f"{r['n_eff']:>8.1f} {r['auc_mean']:>7.3f} {r['regret_mean']:>+7.3f}"
                )
            lines.append("")
    d = record["decoupling"]
    lines += [
        (
            f"Design: {len(d['same_rows'])} cell pairs share rows with different bits, "
            f"{len(d['same_bits'])} pairs share bits (within 15%) with different rows."
        ),
        "Reading it: genuine ICL tracks bits and the bits needed grow with task complexity;",
        "a global prior is flat in both. Compare to logistic regression and the untrained floor.",
    ]
    return "\n".join(lines)


def main() -> None:
    """Command-line entry point: ``python -m fintfm.experiments.context_information``."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", type=str, required=True, help="name=path pairs, comma separated")
    p.add_argument("--out", type=str, default="runs/context_information")
    p.add_argument("--seeds", type=str, default="0,1,2")
    p.add_argument("--rows", type=str, default=",".join(str(s) for s in ROWS))
    p.add_argument("--rates", type=str, default=",".join(str(s) for s in RATES))
    p.add_argument("--tasks", type=str, default=",".join(TASKS))
    p.add_argument("--n-query", type=int, default=N_QUERY)
    args = p.parse_args()
    paths = dict(pair.split("=", 1) for pair in args.models.split(",") if pair)
    record = run(
        paths,
        Path(args.out),
        rows=tuple(int(s) for s in args.rows.split(",")),
        rates=tuple(float(s) for s in args.rates.split(",")),
        tasks=tuple(args.tasks.split(",")),
        seeds=tuple(int(s) for s in args.seeds.split(",")),
        n_query=args.n_query,
    )
    print("\n" + summarise(record))


if __name__ == "__main__":
    main()
