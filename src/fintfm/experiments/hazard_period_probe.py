"""AU5.9's Verify: a term structure that only the context's default periods can reveal.

Why this exists
---------------
``hazard_period_context`` lets a hazard model read each context row's default period beside its
binary label (audit AU5.9). Without it the context says only *whether* a row defaulted, so a
hazard model must infer *when* defaults happen from the features. The Verify the audit asks for
is a task where that inference is impossible and the periods are the only evidence.

The task
--------
Every task has two possible timing regimes over a grid of ``K`` periods:

- **early**: the default period of a defaulter is drawn with pmf proportional to ``decay**k``;
- **late**: the same pmf reversed, so defaults cluster at the end of the grid.

Features ``X`` and binary labels ``y`` are generated from the task seed **alone**; the regime
only decides how the defaulters' periods are drawn, from a separate random stream. So an early
task and a late task with the same seed have *identical* ``X`` and ``y`` (asserted in the tests),
hence the same features, the same binary default rate and the same label-feature relationship.
A model that reads only ``(X, y)`` cannot tell the regimes apart. The truth for a query row is

    cumulative PD at horizon k  =  p(x) * F_regime(k)

with ``p(x)`` the binary default probability and ``F_regime`` the cumulative period pmf.

What is scored
--------------
For a checkpoint the scorer fits the classifier on each task's context twice and compares:

- **with periods**: the context's true default periods;
- **blind**: the same defaulters with a period drawn uniformly at random, which carries no
  information about the regime. This is the same checkpoint with the periods taken away.

A flag-off checkpoint ignores periods, so its two arms coincide; it is the control that cannot
pass.

Two errors, both against the truth above:

- **shape error** (primary; independent of the level calibration of the curve): RMS over the
  grid of the difference between the normalised mean predicted curve ``m_k / m_{K-1}`` and
  ``F_regime(k)``;
- **cumulative-PD error**: RMS over query rows and horizons of the predicted minus the true
  cumulative PD.

Floors: a periods-blind model sees the regimes as one mixture with ``F_mix = (F_early +
F_late) / 2``. The best it can do is predict that mixture, so its shape error cannot fall below
``rms(F_early - F_mix)`` (equal for both regimes) and its cumulative-PD error below
``rms(p(x) * (F_regime - F_mix))``, computed from each task's own ``p(x)``. These are the error
of a blind model that knows ``p(x)`` exactly; a real model adds estimation error on top, so a
checkpoint scoring below them is reading the periods. The scorer reports the floors beside the
errors and a pass verdict (:func:`verdict`).

Not a claim about real data. The task is synthetic and out of the prior's distribution; it says
whether the checkpoint can use context periods at all, not how it does on V4FinBench.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from fintfm.modeling.hazard import CENSORED

REGIMES = ("early", "late")


@dataclass(frozen=True)
class ProbeConfig:
    """Shape of the synthetic Verify tasks.

    Attributes:
        n_horizons: Length ``K`` of the period grid; set it to the checkpoint's ``n_horizons``.
        n_features: Feature columns. Column 0 carries the signal, column 1 a weak one, the rest
            are noise.
        n_ctx: Context rows per task.
        n_query: Query rows per task.
        decay: Early-regime pmf is proportional to ``decay**k``; the late regime is its
            reverse. Below 1; smaller makes the regimes easier to tell apart.
        signal: Coefficient of feature 0 in the default logit.
        intercept: Logit intercept, fixing the binary default rate near 0.25.
    """

    n_horizons: int = 6
    n_features: int = 6
    n_ctx: int = 400
    n_query: int = 200
    decay: float = 0.5
    signal: float = 1.5
    intercept: float = -1.3


@dataclass(frozen=True)
class ProbeTask:
    """One synthetic task.

    Attributes:
        regime: ``"early"`` or ``"late"``.
        X: ``(n_ctx + n_query, n_features)`` features; the first ``n_ctx`` rows are context.
        y: ``(n,)`` binary default labels.
        period: ``(n,)`` zero-based default period, ``CENSORED`` where ``y == 0``.
        p: ``(n,)`` the binary default probability of each row.
        n_ctx: Context length.
    """

    regime: str
    X: np.ndarray
    y: np.ndarray
    period: np.ndarray
    p: np.ndarray
    n_ctx: int


def regime_cdf(regime: str, n_horizons: int, decay: float) -> np.ndarray:
    """Cumulative default-period distribution ``F_regime(k)``, ``k = 0..K-1``.

    Args:
        regime: ``"early"`` or ``"late"``.
        n_horizons: Grid length ``K``.
        decay: See :class:`ProbeConfig`.

    Returns:
        ``(K,)`` non-decreasing array ending at 1.

    Raises:
        ValueError: On an unknown regime or a decay outside ``(0, 1)``.
    """
    if regime not in REGIMES:
        raise ValueError(f"regime must be one of {REGIMES}, got {regime!r}")
    if not 0.0 < decay < 1.0:
        raise ValueError(f"decay must lie in (0, 1), got {decay}")
    pmf = decay ** np.arange(n_horizons, dtype=np.float64)
    pmf /= pmf.sum()
    if regime == "late":
        pmf = pmf[::-1]
    return np.cumsum(pmf)


def make_task(cfg: ProbeConfig, regime: str, seed: int) -> ProbeTask:
    """Generate one task. ``X`` and ``y`` depend on ``seed`` only, not on ``regime``.

    Args:
        cfg: Task shape.
        regime: ``"early"`` or ``"late"``.
        seed: Task seed.

    Returns:
        The task; its periods are drawn from a stream seeded by ``(seed, regime)``.
    """
    n = cfg.n_ctx + cfg.n_query
    rng = np.random.default_rng([seed, 0])
    X = rng.normal(size=(n, cfg.n_features)).astype(np.float32)
    logit = cfg.signal * X[:, 0] + 0.5 * X[:, 1] + cfg.intercept
    p = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.random(n) < p).astype(np.int64)
    pmf = np.diff(regime_cdf(regime, cfg.n_horizons, cfg.decay), prepend=0.0)
    prng = np.random.default_rng([seed, 1 + REGIMES.index(regime)])
    drawn = prng.choice(cfg.n_horizons, size=n, p=pmf)
    period = np.where(y == 1, drawn, CENSORED).astype(np.int64)
    return ProbeTask(regime, X, y, period, p.astype(np.float64), cfg.n_ctx)


def make_tasks(cfg: ProbeConfig, n_tasks: int, seed: int = 0) -> list[ProbeTask]:
    """``n_tasks`` tasks, alternating regimes, each from its own seed.

    Args:
        cfg: Task shape.
        n_tasks: How many tasks.
        seed: Offset of the task seeds, so batteries of different seeds do not overlap.

    Returns:
        Tasks with regimes early, late, early, ...
    """
    return [make_task(cfg, REGIMES[i % 2], seed * 1_000_003 + i // 2) for i in range(n_tasks)]


def blind_periods(task: ProbeTask, n_horizons: int, seed: int) -> np.ndarray:
    """The task's periods with every defaulter's period redrawn uniformly at random.

    Args:
        task: The task.
        n_horizons: Grid length.
        seed: Seed of the redraw.

    Returns:
        Periods with the same censoring as ``task.period`` and no information about the regime.
    """
    rng = np.random.default_rng([seed, 99])
    return np.where(task.y == 1, rng.integers(0, n_horizons, size=task.y.size), CENSORED).astype(
        np.int64
    )


def floors(cfg: ProbeConfig, tasks: list[ProbeTask]) -> dict[str, float]:
    """Error floors of a periods-blind model, and the unit of the two errors.

    Args:
        cfg: Task shape.
        tasks: The battery.

    Returns:
        ``shape_floor``: ``rms(F_early - F_mix)`` over the grid. ``cum_floor``: mean over tasks
        of ``rms(p(x) * (F_regime - F_mix))`` over the query rows and horizons.
    """
    f = {r: regime_cdf(r, cfg.n_horizons, cfg.decay) for r in REGIMES}
    mix = 0.5 * (f["early"] + f["late"])
    shape_floor = float(np.sqrt(np.mean((f["early"] - mix) ** 2)))
    cum = []
    for t in tasks:
        pq = t.p[t.n_ctx :]
        cum.append(np.sqrt(np.mean((pq[:, None] * (f[t.regime] - mix)[None, :]) ** 2)))
    return {"shape_floor": shape_floor, "cum_floor": float(np.mean(cum))}


def task_errors(cfg: ProbeConfig, task: ProbeTask, pred: np.ndarray) -> dict[str, float]:
    """Shape and cumulative-PD error of one task's predicted curves.

    Args:
        cfg: Task shape.
        task: The task.
        pred: ``(n_query, K)`` predicted cumulative PD.

    Returns:
        ``shape_rmse``, ``cum_rmse`` and ``shape0`` (the normalised mean curve at the first
        horizon, whose truth is ``F_regime(0)``).
    """
    f = regime_cdf(task.regime, cfg.n_horizons, cfg.decay)
    m = pred.mean(axis=0).astype(np.float64)
    shape = m / max(m[-1], 1e-12)
    truth = task.p[task.n_ctx :][:, None] * f[None, :]
    return {
        "shape_rmse": float(np.sqrt(np.mean((shape - f) ** 2))),
        "cum_rmse": float(np.sqrt(np.mean((pred - truth) ** 2))),
        "shape0": float(shape[0]),
    }


def score_checkpoint(
    checkpoint: str,
    cfg: ProbeConfig,
    tasks: list[ProbeTask],
    device: str = "cpu",
    seed: int = 0,
) -> dict:
    """Score a checkpoint's term structure on the Verify, with and without context periods.

    Args:
        checkpoint: Path of a hazard checkpoint (``--n-horizons``), flag-on or flag-off.
        cfg: Task shape; ``n_horizons`` must equal the checkpoint's.
        tasks: The battery from :func:`make_tasks`.
        device: Torch device; ``cpu``.
        seed: Seed of the classifier and of the blind-period redraws.

    Returns:
        A record: per-task errors for both arms, their means, the floors, the paired
        with-minus-blind differences and the verdict.

    Raises:
        ValueError: If the checkpoint's horizon grid is not ``cfg.n_horizons``.
    """
    from fintfm.inference.classifier import FinancialTFMClassifier

    rows: list[dict] = []
    reads = False
    for i, t in enumerate(tasks):
        clf = FinancialTFMClassifier(
            checkpoint, device=device, max_context=t.n_ctx, random_state=seed
        )
        hz = clf.model.hazard
        if hz is None or hz.n_horizons != cfg.n_horizons:
            raise ValueError(
                f"checkpoint has {None if hz is None else hz.n_horizons} horizons, "
                f"the probe has {cfg.n_horizons}"
            )
        reads = bool(clf.model.cfg.hazard_period_context)
        rec: dict = {"task": i, "regime": t.regime}
        for arm, per in (
            ("with_periods", t.period[: t.n_ctx]),
            ("blind", blind_periods(t, cfg.n_horizons, seed * 7919 + i)[: t.n_ctx]),
        ):
            clf.fit(t.X[: t.n_ctx], t.y[: t.n_ctx], periods=per)
            rec[arm] = task_errors(cfg, t, clf.predict_term_structure(t.X[t.n_ctx :]))
        rows.append(rec)
    return summarise(cfg, tasks, rows, checkpoint, reads)


def _boot_ci(d: np.ndarray, seed: int = 0, n_boot: int = 2000) -> tuple[float, float]:
    """95% percentile bootstrap interval of the mean of ``d`` over tasks."""
    g = np.random.default_rng(seed)
    means = d[g.integers(0, d.size, size=(n_boot, d.size))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarise(
    cfg: ProbeConfig, tasks: list[ProbeTask], rows: list[dict], checkpoint: str, reads: bool
) -> dict:
    """Aggregate per-task errors into the record :func:`verdict` reads.

    Args:
        cfg: Task shape.
        tasks: The battery.
        rows: Per-task records of :func:`score_checkpoint`.
        checkpoint: Checkpoint path, recorded.
        reads: Whether the checkpoint has ``hazard_period_context``.

    Returns:
        The full record.
    """
    out: dict = {
        "checkpoint": checkpoint,
        "reads_periods": reads,
        "config": asdict(cfg),
        "n_tasks": len(rows),
        "floors": floors(cfg, tasks),
        "rows": rows,
    }
    for arm in ("with_periods", "blind"):
        out[arm] = {
            k: float(np.mean([r[arm][k] for r in rows])) for k in ("shape_rmse", "cum_rmse")
        }
        for reg in REGIMES:
            sel = [r[arm]["shape0"] for r in rows if r["regime"] == reg]
            out[arm][f"shape0_{reg}"] = float(np.mean(sel)) if sel else float("nan")
    for k in ("shape_rmse", "cum_rmse"):
        d = np.array([r["with_periods"][k] - r["blind"][k] for r in rows])
        lo, hi = _boot_ci(d)
        out.setdefault("paired_with_minus_blind", {})[k] = {
            "mean": float(d.mean()), "ci95": [lo, hi]
        }
    out["truth_shape0"] = {
        r: float(regime_cdf(r, cfg.n_horizons, cfg.decay)[0]) for r in REGIMES
    }
    out["verdict"] = verdict(out)
    return out


def verdict(record: dict) -> dict:
    """The pre-registered pass criterion (docs/results/drafts/au5.9_2026-11.md).

    Passes when, on the shape error, (1) the checkpoint with periods lies below the blind floor
    and (2) the paired with-minus-blind difference is negative with its 95% bootstrap interval
    wholly below zero. A strong pass additionally needs the with-periods error at or below half
    the floor. A flag-off checkpoint never passes: it reads no periods.

    Args:
        record: Output of :func:`summarise`.

    Returns:
        ``{"pass": bool, "strong": bool, "below_floor": bool, "ci_below_zero": bool}``.
    """
    shape = record["with_periods"]["shape_rmse"]
    floor = record["floors"]["shape_floor"]
    below = bool(shape < floor)
    ci = record["paired_with_minus_blind"]["shape_rmse"]["ci95"]
    ci_neg = bool(ci[1] < 0.0)
    ok = bool(record["reads_periods"] and below and ci_neg)
    return {
        "pass": ok,
        "strong": bool(ok and shape <= 0.5 * floor),
        "below_floor": below,
        "ci_below_zero": ci_neg,
    }


def render(record: dict) -> str:
    """A short text table of a record.

    Args:
        record: Output of :func:`summarise`.

    Returns:
        Lines giving both errors, both arms, the floors and the verdict.
    """
    fl, v = record["floors"], record["verdict"]
    lines = [
        (
            f"{record['checkpoint']}  reads_periods={record['reads_periods']}  "
            f"tasks={record['n_tasks']}"
        ),
        (
            f"{'':14s}{'shape RMSE':>12s}{'cum-PD RMSE':>14s}"
            f"{'shape[0] early':>16s}{'shape[0] late':>15s}"
        ),
    ]
    for arm in ("with_periods", "blind"):
        a = record[arm]
        lines.append(
            f"{arm:14s}{a['shape_rmse']:12.4f}{a['cum_rmse']:14.4f}"
            f"{a['shape0_early']:16.3f}{a['shape0_late']:15.3f}"
        )
    t = record["truth_shape0"]
    lines.append(
        f"{'blind floor':14s}{fl['shape_floor']:12.4f}{fl['cum_floor']:14.4f}"
        f"{'truth ' + format(t['early'], '.3f'):>16s}{'truth ' + format(t['late'], '.3f'):>15s}"
    )
    p = record["paired_with_minus_blind"]["shape_rmse"]
    lines.append(
        f"paired shape RMSE, with minus blind: {p['mean']:+.4f}  95% CI "
        f"[{p['ci95'][0]:+.4f}, {p['ci95'][1]:+.4f}]"
    )
    lines.append(
        f"pass={v['pass']} strong={v['strong']} (below floor {v['below_floor']}, "
        f"CI below zero {v['ci_below_zero']})"
    )
    return "\n".join(lines)


def main() -> None:
    """Score one or more checkpoints on the Verify and write the records as JSON."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", nargs="+", required=True, help="hazard checkpoint path(s)")
    ap.add_argument("--n-horizons", type=int, default=6)
    ap.add_argument("--n-tasks", type=int, default=200)
    ap.add_argument("--n-ctx", type=int, default=400)
    ap.add_argument("--n-query", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, required=True, help="directory for <name>.json records")
    args = ap.parse_args()
    cfg = ProbeConfig(n_horizons=args.n_horizons, n_ctx=args.n_ctx, n_query=args.n_query)
    tasks = make_tasks(cfg, args.n_tasks, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    for path in args.model:
        rec = score_checkpoint(path, cfg, tasks, device=args.device, seed=args.seed)
        print(render(rec), flush=True)
        (args.out / f"{Path(path).stem}.json").write_text(json.dumps(rec, indent=2))


if __name__ == "__main__":
    main()
