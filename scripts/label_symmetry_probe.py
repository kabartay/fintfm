"""Label-symmetry probe: does a checkpoint's prediction depend on which class is called "1"?

Audit AU5.12's first step, the probe ``docs/results/FINDINGS.md`` §45 ran on an older
checkpoint. Fit the classifier on a binary context, predict ``p = P(y=1)``; relabel the
context 0<->1, predict again, and invert, ``1 - p_swapped``. A model that reads its context
as evidence returns nearly the same probabilities either way, correlation near +1; §45
measured **-0.62**, meaning the two labellings drove systematically opposed predictions.

Both fits go through :class:`FinancialTFMClassifier`, never the network's forward method, so
the probe measures the inference path the benchmarks use. The context is drawn here and
handed over whole (``max_context`` at least its size), so the classifier does not resample
it and the base-rate correction is zero in both orientations.

Two sources of tasks:

- **synthetic**: small binary tasks from the pretraining prior, half from the financial
  generator (what the binary checkpoints were trained on, ``p_financial`` 1.0) and half from
  the SCM generator, each split once into context and query rows.
- **real**: the Polish (3-year horizon) and Taiwanese bankruptcy panels, **only if already
  cached** under ``--cache-dir`` and **only with** ``--allow-tabarena-for-reporting``; nothing
  is downloaded. Several uniform context draws each. Both panels are TabArena-v0.1 datasets
  and this probe's results feed arm choices, so without the flag they are skipped with a
  printed reason (design review DR4.1, ``fintfm.evaluation.overlap.SELECTION_FORBIDDEN``);
  with it the report carries ``reporting_only``.

Reported per source: the pooled Pearson correlation of ``p`` with ``1 - p_swapped`` over every
query, the median per-task correlation, the share of tasks below zero, and the mean absolute
change, with the task and query counts beside them.

Usage:
    uv run python scripts/label_symmetry_probe.py \\
        --model runs/v4-cellattn-labels.pt --cache-dir data/cache --out probe.json
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from fintfm.evaluation import datasets
from fintfm.evaluation.overlap import REPORTING_FLAG, SelectionForbiddenError, guard_selection
from fintfm.inference.classifier import FinancialTFMClassifier
from fintfm.modeling.model import FinancialTFM
from fintfm.prior import PriorConfig, sample_task


@dataclass
class ArmResult:
    """Paired predictions and per-task summaries for one source of tasks.

    Attributes:
        name: Source label, e.g. ``"synthetic-financial"``.
        p: Every query's ``P(y=1)`` under the original labels, concatenated over tasks.
        p_swapped_inverted: The same queries' ``1 - P(y=1)`` under swapped labels.
        task_corr: Pearson correlation per task, finite values only.
        skipped: Reason -> count of tasks that could not be probed.
    """

    name: str
    p: list[np.ndarray] = field(default_factory=list)
    p_swapped_inverted: list[np.ndarray] = field(default_factory=list)
    task_corr: list[float] = field(default_factory=list)
    skipped: dict[str, int] = field(default_factory=dict)

    def skip(self, reason: str) -> None:
        """Count one task that was not probed, by reason."""
        self.skipped[reason] = self.skipped.get(reason, 0) + 1

    def summary(self) -> dict[str, object]:
        """The arm's reported numbers.

        Returns:
            Task and query counts, pooled and median per-task correlation, share of tasks with
            a negative correlation, mean and max absolute change, and the skip counts.
        """
        if not self.p:
            return {"source": self.name, "n_tasks": 0, "skipped": self.skipped}
        p = np.concatenate(self.p)
        q = np.concatenate(self.p_swapped_inverted)
        corr = np.asarray(self.task_corr)
        return {
            "source": self.name,
            "n_tasks": len(self.p),
            "n_tasks_with_corr": int(corr.size),
            "n_queries": int(p.size),
            "pooled_corr": _corr(p, q),
            "median_task_corr": float(np.median(corr)) if corr.size else float("nan"),
            "share_tasks_negative": float((corr < 0).mean()) if corr.size else float("nan"),
            "mean_abs_change": float(np.abs(p - q).mean()),
            "max_abs_change": float(np.abs(p - q).max()),
            "mean_p": float(p.mean()),
            "skipped": self.skipped,
        }


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation, NaN when either side is constant rather than a numpy warning."""
    if a.size < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _probe_task(
    model: FinancialTFM,
    arm: ArmResult,
    X_ctx: np.ndarray,
    y_ctx: np.ndarray,
    X_q: np.ndarray,
    seed: int,
) -> None:
    """Predict one task under both labellings and record the pair.

    Args:
        model: The loaded checkpoint.
        arm: Where the result goes.
        X_ctx: ``(n_ctx, F)`` context features.
        y_ctx: ``(n_ctx,)`` binary labels, both classes present.
        X_q: ``(n_q, F)`` query features.
        seed: The classifier's ``random_state``, the same in both orientations so only the
            labels differ between the two fits.
    """
    kw = {"max_context": len(y_ctx), "random_state": seed}
    p = FinancialTFMClassifier(model, **kw).fit(X_ctx, y_ctx).predict_proba(X_q)[:, 1]
    p_sw = FinancialTFMClassifier(model, **kw).fit(X_ctx, 1 - y_ctx).predict_proba(X_q)[:, 1]
    inv = 1.0 - p_sw
    arm.p.append(p)
    arm.p_swapped_inverted.append(inv)
    c = _corr(p, inv)
    if np.isfinite(c):
        arm.task_corr.append(c)
    else:
        arm.skip("constant prediction (correlation undefined; pair still pooled)")


def probe_synthetic(
    model: FinancialTFM, p_financial: float, n_tasks: int, n_rows: int, ctx_frac: float, seed: int
) -> ArmResult:
    """Probe ``n_tasks`` binary tasks drawn from the pretraining prior.

    Args:
        model: The loaded checkpoint.
        p_financial: 1.0 for the financial generator, 0.0 for the SCM generator.
        n_tasks: Tasks to draw.
        n_rows: Rows per task, split into context and queries.
        ctx_frac: Share of each task's rows used as context.
        seed: Seed for the task draws and splits.

    Returns:
        The arm's paired predictions.
    """
    name = "synthetic-financial" if p_financial == 1.0 else "synthetic-scm"
    arm = ArmResult(name)
    cfg = PriorConfig(
        max_features=model.cfg.max_features,
        max_classes=2,
        p_financial=p_financial,
        n_rows=n_rows,
    )
    rng = np.random.default_rng(seed)
    for t in range(n_tasks):
        task = sample_task(rng, cfg)
        y = np.asarray(task.y)
        if len(np.unique(y)) != 2:
            arm.skip("task not binary")
            continue
        order = rng.permutation(len(y))
        n_ctx = round(len(y) * ctx_frac)
        ci, qi = order[:n_ctx], order[n_ctx:]
        if len(np.unique(y[ci])) != 2:
            arm.skip("context holds one class")
            continue
        _probe_task(model, arm, task.X[ci], y[ci].astype(np.int64), task.X[qi], seed + t)
    return arm


def probe_panel(
    model: FinancialTFM,
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    n_draws: int,
    n_ctx: int,
    n_q: int,
    seed: int,
) -> ArmResult:
    """Probe a real panel over several uniform context draws with disjoint queries.

    Args:
        model: The loaded checkpoint.
        name: Panel label.
        X: ``(n, F)`` features.
        y: ``(n,)`` binary labels.
        n_draws: Context draws.
        n_ctx: Context rows per draw.
        n_q: Query rows per draw, never in that draw's context.
        seed: Seed for the draws.

    Returns:
        The arm's paired predictions.
    """
    arm = ArmResult(name)
    rng = np.random.default_rng(seed)
    for d in range(n_draws):
        order = rng.permutation(len(y))
        ci, qi = order[:n_ctx], order[n_ctx : n_ctx + n_q]
        if len(np.unique(y[ci])) != 2:
            arm.skip("context holds one class")
            continue
        _probe_task(model, arm, X[ci], y[ci], X[qi], seed + d)
    return arm


def _cached_panels(
    cache_dir: Path, allow_reporting: bool = False
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Load whichever bankruptcy panels are already cached, never downloading.

    Args:
        cache_dir: Directory holding ``polish_bankruptcy.zip`` and ``taiwan_bankruptcy.zip``.
        allow_reporting: ``--allow-tabarena-for-reporting`` was passed. Without it no panel
            is loaded: both are in ``SELECTION_FORBIDDEN``.

    Returns:
        ``(name, X, y)`` for each panel found; an absent or forbidden archive is reported and
        skipped.
    """
    # the loaders read the module-level CACHE_DIR at call time and download on a miss, so
    # point it at the requested directory and only call a loader whose archive exists
    datasets.CACHE_DIR = cache_dir
    out = []
    for archive, loader in (
        ("polish_bankruptcy.zip", lambda: datasets.load_polish_bankruptcy(3)),
        ("taiwan_bankruptcy.zip", datasets.load_taiwan_bankruptcy),
    ):
        try:
            guard_selection([archive], allow_reporting, "label_symmetry_probe")
        except SelectionForbiddenError as exc:
            print(f"skip {archive}: {exc}", flush=True)
            continue
        if not (cache_dir / archive).exists():
            print(f"skip {archive}: not cached in {cache_dir}", flush=True)
            continue
        ds = loader()
        out.append((ds.name, ds.X, ds.y))
    return out


def main(argv: list[str] | None = None) -> int:
    """Run the probe and print, and optionally write, one summary per source.

    Args:
        argv: Command-line arguments; ``sys.argv[1:]`` when omitted.

    Returns:
        Process exit status.
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, help="checkpoint path (read only)")
    ap.add_argument("--n-tasks", type=int, default=50, help="synthetic tasks, split over two generators")
    ap.add_argument("--n-rows", type=int, default=512, help="rows per synthetic task")
    ap.add_argument("--ctx-frac", type=float, default=0.5, help="context share of a synthetic task")
    ap.add_argument("--cache-dir", type=Path, default=datasets.CACHE_DIR)
    ap.add_argument("--real-draws", type=int, default=3, help="context draws per real panel")
    ap.add_argument("--real-context", type=int, default=512)
    ap.add_argument("--real-queries", type=int, default=512)
    ap.add_argument("--threads", type=int, default=4, help="torch CPU threads")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None, help="optional JSON output path")
    ap.add_argument(REPORTING_FLAG, action="store_true",
                    help="probe the Polish and Taiwanese panels too, for reporting only")
    args = ap.parse_args(argv)

    torch.set_num_threads(args.threads)
    t0 = time.time()
    model = FinancialTFM.load(args.model, map_location="cpu")
    if model.cfg.max_classes < 2:
        print("checkpoint has no binary head", file=sys.stderr)
        return 1
    arms = [
        probe_synthetic(model, 1.0, args.n_tasks - args.n_tasks // 2, args.n_rows,
                        args.ctx_frac, args.seed),
        probe_synthetic(model, 0.0, args.n_tasks // 2, args.n_rows, args.ctx_frac,
                        args.seed + 10_000),
    ]
    panels = _cached_panels(args.cache_dir, args.allow_tabarena_for_reporting)
    for name, X, y in panels:
        arms.append(probe_panel(model, name, X, y, args.real_draws, args.real_context,
                                args.real_queries, args.seed))
    # ru_maxrss is bytes on macOS and kilobytes on Linux
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_gb = rss / 1e9 if sys.platform == "darwin" else rss / 1e6
    summaries = [a.summary() for a in arms]
    report = {
        "label": "MEASURED",
        "model": args.model,
        "args": {k: str(v) for k, v in vars(args).items()},
        "seconds": round(time.time() - t0, 1),
        "peak_rss_gb": round(peak_gb, 2),
        "sources": summaries,
        # DR4.1: TabArena-v0.1 panels may be reported on, never used to choose an arm
        "reporting_only": [name for name, _, _ in panels],
    }
    print(f"label-symmetry probe, MEASURED, {args.model}")
    print(f"{'source':<26}{'tasks':>6}{'queries':>9}{'pooled r':>10}{'median r':>10}"
          f"{'neg share':>10}{'mean |d|':>10}")
    for s in summaries:
        if not s["n_tasks"]:
            print(f"{s['source']:<26}{0:>6}  (no task probed: {s['skipped']})")
            continue
        print(f"{s['source']:<26}{s['n_tasks']:>6}{s['n_queries']:>9}{s['pooled_corr']:>10.3f}"
              f"{s['median_task_corr']:>10.3f}{s['share_tasks_negative']:>10.2f}"
              f"{s['mean_abs_change']:>10.4f}")
        if s["skipped"]:
            print(f"{'':<26}skipped: {s['skipped']}")
    print(f"{report['seconds']} s, peak RSS {report['peak_rss_gb']} GB")
    if args.out is not None:
        args.out.write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
