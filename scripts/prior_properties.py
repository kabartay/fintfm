"""Re-measure §18's four prior properties on the published recipe's prior (openspec 15.3/15.4).

``docs/results/FINDINGS.md`` §18 compared the financial prior with four real credit panels on
width, logistic-regression difficulty, mean absolute inter-feature correlation and
missingness (plus the default rate). This script draws ``--n-tasks`` tasks from the prior a
YAML recipe builds (default ``configs/published/v4-cellattn-labels.yaml``) and reports each
property as quantiles, once for the recipe as written and once with
``financial_max_lr_auc`` set (task 15.4's opt-in rejection), so 15.4 gets its before/after.

No model is built or run. Difficulty is the held-out ROC-AUC of
``fintfm.prior.financial._holdout_lr_auc`` (fit on the first half of the rows, score the
second; median-imputed, standardised), the same statistic the rejection option thresholds on.
Correlation is the mean absolute off-diagonal Pearson correlation over non-constant columns
after median imputation.

Usage::

    env -u VIRTUAL_ENV uv run --frozen python scripts/prior_properties.py --n-tasks 300 --seed 0
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace

import numpy as np

from fintfm.config import load_config
from fintfm.modeling.train import build_parser, build_prior_config
from fintfm.prior.financial import _holdout_lr_auc
from fintfm.prior.mixture import PriorConfig, sample_task

REAL_RANGE = (0.697, 0.899)  # real-panel held-out LR AUC range, FINDINGS §18
QUANTILES = (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)


def task_properties(X: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """The §18 properties of one task.

    Args:
        X: ``(n_rows, n_features)`` features, may contain NaN.
        y: ``(n_rows,)`` binary labels.

    Returns:
        ``width``, ``lr_auc`` (NaN when unscorable), ``default_rate``, ``mean_abs_corr`` (NaN
        with fewer than two usable columns) and ``missing_frac``.
    """
    Xf = np.asarray(X, dtype=np.float64)
    finite = np.isfinite(Xf)
    auc = _holdout_lr_auc(Xf, y)
    med = np.nan_to_num(np.nanmedian(np.where(finite, Xf, np.nan), axis=0))
    Xi = np.where(finite, Xf, med[None, :])
    keep = Xi.std(axis=0) > 1e-12
    corr = float("nan")
    if keep.sum() >= 2:
        c = np.abs(np.corrcoef(Xi[:, keep], rowvar=False))
        corr = float((c.sum() - keep.sum()) / (keep.sum() * (keep.sum() - 1)))
    return {
        "width": float(Xf.shape[1]),
        "lr_auc": float("nan") if auc is None else auc,
        "default_rate": float(np.mean(y)),
        "mean_abs_corr": corr,
        "missing_frac": float(1.0 - finite.mean()),
    }


def measure(cfg: PriorConfig, n_tasks: int, seed: int) -> dict[str, dict[str, float]]:
    """Draw ``n_tasks`` tasks and summarise each property.

    Args:
        cfg: The prior to sample.
        n_tasks: Tasks drawn; task ``i`` uses ``default_rng(seed + i)``.
        seed: First seed.

    Returns:
        ``{property: {"n", "mean", "min", "q05", "q25", "q50", "q75", "q95", "max"}}`` plus
        ``frac_gt_0.95``, ``n_unscorable`` and the ``frac_in/below/above_*`` shares of
        the real-panel range [0.697, 0.899] and of [0.5, 0.95] under ``lr_auc``.
    """
    sizes = cfg.n_rows_choices or (cfg.n_rows,)
    rows: list[dict[str, float]] = []
    for i in range(n_tasks):
        rng = np.random.default_rng(seed + i)
        n = int(sizes[rng.integers(len(sizes))])
        t = sample_task(rng, cfg, n_rows=n)
        rows.append(task_properties(t.X, t.y))
    out: dict[str, dict[str, float]] = {}
    for key in rows[0]:
        v = np.array([r[key] for r in rows])
        ok = v[np.isfinite(v)]
        q = np.quantile(ok, QUANTILES)
        out[key] = {
            "n": float(len(ok)),
            "mean": float(ok.mean()),
            "min": float(q[0]),
            "q05": float(q[1]),
            "q25": float(q[2]),
            "q50": float(q[3]),
            "q75": float(q[4]),
            "q95": float(q[5]),
            "max": float(q[6]),
        }
    a = np.array([r["lr_auc"] for r in rows])
    ok_a = a[np.isfinite(a)]
    out["lr_auc"]["frac_gt_0.95"] = float(np.mean(ok_a > 0.95))
    # coverage of the real-panel range (FINDINGS §18) and of [0.5, 0.95], over scorable tasks
    for name, (lo, hi) in {"real_range": REAL_RANGE, "chance_to_0.95": (0.5, 0.95)}.items():
        out["lr_auc"][f"frac_in_{name}"] = float(np.mean((ok_a >= lo) & (ok_a <= hi)))
        out["lr_auc"][f"frac_below_{name}"] = float(np.mean(ok_a < lo))
        out["lr_auc"][f"frac_above_{name}"] = float(np.mean(ok_a > hi))
    out["lr_auc"]["n_unscorable"] = float(np.sum(~np.isfinite(a)))
    return out


def main() -> None:
    """Measure the recipe's prior as written and with ``financial_max_lr_auc`` set."""
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--config", default="configs/published/v4-cellattn-labels.yaml")
    p.add_argument("--n-tasks", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-lr-auc", type=float, default=0.95)
    p.add_argument("--json", default=None, help="also write the numbers here")
    args = p.parse_args()

    # the training parser's own defaults (every flag None), so the recipe's YAML supplies the values
    cfg = build_prior_config(build_parser().parse_args([]), load_config(args.config))
    result: dict[str, object] = {
        "config": args.config,
        "n_tasks": args.n_tasks,
        "seed": args.seed,
        "n_rows_choices": list(cfg.n_rows_choices or ()),
        "max_features": cfg.max_features,
    }
    for label, c in (
        ("as_written", cfg),
        (
            f"financial_max_lr_auc={args.max_lr_auc}",
            replace(cfg, financial_max_lr_auc=args.max_lr_auc),
        ),
    ):
        t0 = time.time()
        result[label] = measure(c, args.n_tasks, args.seed)
        print(f"[{label}] {time.time() - t0:.0f}s", flush=True)
    print(json.dumps(result, indent=1))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
