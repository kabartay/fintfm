"""Surrogate method test for task 48.16: lever settings to V4FinBench mean AP, on recorded runs.

Why this exists
---------------
Task 48.16 asks whether a cheap regressor fitted on runs already paid for can predict an
untried point of a sweep, before anyone trusts it on an expensive one. §149 did a narrower
version (leave-one-out on §104's five `column_id_dim` points, one input, no mean baseline).
This is the heterogeneous version the task text asks for: every row is one already-scored
checkpoint, the inputs are its config levers, the single score is V4FinBench five-fold mean
average precision, and a plain mean predictor is always reported beside the surrogates,
because with about a dozen rows a surrogate may be no better than the mean.

Which runs count. Only runs scored under one inference regime (full 1,000,087-row horizon-0
panel, five folds, `n_ensemble=8`, `context_strategy=uniform`), so the AP values are
comparable (CLAUDE.md: AP depends on the row set and the inference settings). That excludes
the n_ensemble=1 runs, the prior-variant arms and the fine-tuned arms. Each row names its
source section and run directory; `verify_rows` re-reads the per-run `v4_protocol.json`
files so the embedded numbers cannot drift from the runs.

Usage::

    nice -n 19 uv run --frozen python scripts/colid_surrogate.py
    nice -n 19 uv run --frozen python scripts/colid_surrogate.py --runs-dir /path/to/runs
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge

#: Per-fold seed SD of V4FinBench AP, pooled over matched replicates (FINDINGS §173).
SEED_SD_PER_FOLD = 0.0087
#: SD of the five-fold mean of one run, from the per-fold figure (§173: sqrt(0.0087^2 / 5)).
SEED_SD_MEAN = SEED_SD_PER_FOLD / np.sqrt(5)
#: §173's yardstick: 95% bar for the difference of two single runs' five-fold means.
NOISE_YARDSTICK = 0.0117

FEATURES = ("column_id_dim", "cell_attn", "max_context_k", "tasks_k")


@dataclass(frozen=True)
class Row:
    """One scored checkpoint.

    Attributes:
        name: Short label.
        run_dir: Directory under `runs/` holding `v4_protocol.json`.
        column_id_dim: Identity width; 0 for the pre-cell-attention architecture, which
            carries `column_id_dim=None` (§97, line "carry column_id_dim=None").
        cell_attn: 1 with cell attention and labels, 0 for the column-id-only architecture.
        max_context_k: `max_context` in thousands of rows at scoring.
        tasks_k: Training tasks in thousands. 24 for the §84 "A" arm is ESTIMATED (6,000
            steps at batch 4; the step count is not stated in the source read).
        ap: Five-fold mean AP, MEASURED, as recorded in the run's own JSON.
        source: FINDINGS section.
    """

    name: str
    run_dir: str
    column_id_dim: int
    cell_attn: int
    max_context_k: float
    tasks_k: float
    ap: float
    source: str

    def x(self) -> list[float]:
        """Feature vector in `FEATURES` order."""
        return [self.column_id_dim, self.cell_attn, self.max_context_k, self.tasks_k]


ROWS: tuple[Row, ...] = (
    Row("colid-only ctx1000", "v4-395-colid", 0, 0, 1.0, 48, 0.1454, "§83/§84"),
    Row("colid-only ctx2000", "v4-3925-colid-2000", 0, 0, 2.0, 48, 0.1523, "§83/§84"),
    Row("colid-only ctx4000", "v4-3925-colid-4000", 0, 0, 4.0, 48, 0.1512, "§83/§84"),
    Row("A cd12 deviated ctx1000", "v4-395-cellattn", 12, 1, 1.0, 24, 0.1941, "§84/§97"),
    Row("A cd12 deviated ctx2000", "v4-3925-cellattn-2000", 12, 1, 2.0, 24, 0.1909, "§84"),
    Row("B cd12 ctx1000", "v4-colid12-full", 12, 1, 1.0, 48, 0.1946, "§97/§104"),
    Row("cd16 ctx1000", "v4-scale-48k", 16, 1, 1.0, 48, 0.2072, "§93/§104"),
    Row("cd16 ctx2000", "v4-lever-ens8", 16, 1, 2.0, 48, 0.2044, "§91 lever run"),
    Row("cd16 240k ctx1000", "v4-scale-240k", 16, 1, 1.0, 240, 0.2061, "§93"),
    Row("cd20 ctx1000", "colid-sweep-20", 20, 1, 1.0, 48, 0.2005, "§104"),
    Row("cd24 ctx1000", "colid-sweep-24", 24, 1, 1.0, 48, 0.1923, "§104"),
    Row("cd32 ctx1000", "colid-sweep-32", 32, 1, 1.0, 48, 0.1885, "§104"),
)

#: Untried `column_id_dim` values to predict (cell attention, 1,000 context, 48,000 tasks).
UNTRIED = (8, 10, 14, 18, 28, 40)


def verify_rows(runs_dir: Path) -> list[str]:
    """Check each embedded AP against the run's own `v4_protocol.json`.

    Args:
        runs_dir: The `runs/` directory holding the per-run folders.

    Returns:
        Human-readable mismatch messages; empty when every row agrees to 4 decimals and
        was scored on 1,000,087 test rows with `n_ensemble=8`.
    """
    problems: list[str] = []
    for r in ROWS:
        j = json.loads((runs_dir / r.run_dir / "v4_protocol.json").read_text())
        arm = next(a for a in j["arms"] if a["arm"] == "fintfm")
        n_test = sum(f["n_test"] for f in arm["folds"])
        if round(arm["avg_precision_mean"], 4) != r.ap:
            problems.append(f"{r.name}: AP {arm['avg_precision_mean']:.4f} != {r.ap}")
        if n_test != 1_000_087 or j["config"].get("n_ensemble") != 8:
            problems.append(f"{r.name}: not the full-panel n_ensemble=8 regime")
    return problems


def _fit(kind: str, x: np.ndarray, y: np.ndarray):
    """Fit one surrogate. `mean` is a constant; `ridge` standardises its inputs first."""
    if kind == "mean":
        return lambda q: np.full(len(q), y.mean())
    if kind == "ridge":
        mu, sd = x.mean(0), x.std(0)
        sd[sd == 0] = 1.0
        m = Ridge(alpha=1.0).fit((x - mu) / sd, y)
        return lambda q: m.predict((q - mu) / sd)
    if kind == "gbm":
        m = GradientBoostingRegressor(
            n_estimators=50, max_depth=2, learning_rate=0.1, random_state=0
        ).fit(x, y)
        return m.predict
    raise ValueError(kind)


KINDS = ("mean", "ridge", "gbm")


def leave_one_out(rows: tuple[Row, ...]) -> dict[str, np.ndarray]:
    """Leave-one-out signed errors (prediction minus truth) per surrogate.

    Args:
        rows: The scored runs.

    Returns:
        Map from surrogate name to an array of signed errors, one per row.
    """
    x = np.array([r.x() for r in rows], float)
    y = np.array([r.ap for r in rows], float)
    out = {k: np.zeros(len(rows)) for k in KINDS}
    for i in range(len(rows)):
        keep = np.arange(len(rows)) != i
        for k in KINDS:
            out[k][i] = _fit(k, x[keep], y[keep])(x[i : i + 1])[0] - y[i]
    return out


def predict_untried(rows: tuple[Row, ...]) -> dict[int, dict[str, float]]:
    """Predict mean AP at each untried `column_id_dim`, plus a bootstrap GBM spread.

    Args:
        rows: The scored runs (all of them are used for the fit).

    Returns:
        Map from `column_id_dim` to {ridge, gbm, gbm_boot_sd}.
    """
    x = np.array([r.x() for r in rows], float)
    y = np.array([r.ap for r in rows], float)
    rng = np.random.default_rng(0)
    ridge, gbm = _fit("ridge", x, y), _fit("gbm", x, y)
    boots = []
    for _ in range(100):
        idx = rng.integers(0, len(rows), len(rows))
        boots.append(_fit("gbm", x[idx], y[idx]))
    res: dict[int, dict[str, float]] = {}
    for cd in UNTRIED:
        q = np.array([[cd, 1, 1.0, 48]], float)
        res[cd] = {
            "ridge": float(ridge(q)[0]),
            "gbm": float(gbm(q)[0]),
            "gbm_boot_sd": float(np.std([b(q)[0] for b in boots])),
        }
    return res


def summarise(errs: dict[str, np.ndarray]) -> dict[str, float]:
    """Mean absolute leave-one-out error per surrogate."""
    return {k: float(np.abs(v).mean()) for k, v in errs.items()}


def main() -> None:
    """Print the leave-one-out table, the untried predictions and the check-run plan."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", type=Path, default=None, help="verify rows against runs/")
    args = ap.parse_args()
    if args.runs_dir:
        bad = verify_rows(args.runs_dir)
        print("row verification:", "all 12 rows match" if not bad else bad)

    print(f"\nLeave-one-out, all {len(ROWS)} rows (MEASURED APs; errors are pred - true)")
    errs = leave_one_out(ROWS)
    print(f"{'row':26s}{'true':>8s}" + "".join(f"{k:>9s}" for k in KINDS))
    for i, r in enumerate(ROWS):
        print(f"{r.name:26s}{r.ap:8.4f}" + "".join(f"{errs[k][i]:+9.4f}" for k in KINDS))
    print("MAE all rows:", {k: round(v, 4) for k, v in summarise(errs).items()})

    ca = tuple(r for r in ROWS if r.cell_attn)
    e1 = leave_one_out(ca)
    print(f"\nCell-attention rows only, n={len(ca)} (the architecture jump removed)")
    print("MAE:", {k: round(v, 4) for k, v in summarise(e1).items()})

    cell = tuple(r for r in ROWS if r.cell_attn and r.max_context_k == 1.0 and r.tasks_k == 48)
    e2 = leave_one_out(cell)
    print(f"\nColumn_id_dim curve only, n={len(cell)} (cell attention, ctx 1000, 48k tasks)")
    print("MAE:", {k: round(v, 4) for k, v in summarise(e2).items()})
    print("worst row per surrogate:", {k: cell[int(np.abs(v).argmax())].name for k, v in e2.items()})

    print("\nUntried column_id_dim predictions (ctx 1000, 48k tasks), ESTIMATED not measured")
    preds = predict_untried(ROWS)
    for cd, p in preds.items():
        print(f"  {cd:3d}: ridge {p['ridge']:.4f}  gbm {p['gbm']:.4f}  gbm bootstrap sd {p['gbm_boot_sd']:.4f}")
    best_known = max(r.ap for r in ROWS if r.max_context_k == 1.0 and r.tasks_k == 48)
    print(f"\nbest recorded 48k/ctx1000 AP: {best_known:.4f}; one run's seed SD of the mean "
          f"{SEED_SD_MEAN:.4f}; two-run yardstick {NOISE_YARDSTICK}")


if __name__ == "__main__":
    main()
