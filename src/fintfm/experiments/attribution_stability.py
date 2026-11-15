"""Attribution stability across context resamples and seeds (openspec task 12.5).

Why this exists
---------------
"An explanation that moves when the context is resampled is not an explanation a regulator can
use" (``openspec/changes/zero-shot-attribution/proposal.md``). 12.2 measured that for the
incumbent, LightGBM with TreeSHAP (``attribution_baseline.py``). This module measures it for
FinTFM with the same statistics: Spearman rank correlation and top-``k`` Jaccard overlap of the
normalised importance vectors, between context resamples of one seed and between seeds.

Design
------
- The harness (:func:`stability`) takes **any attribution function** with the signature
  :data:`AttributionFn`; nothing in it knows about FinTFM. The 12.4 attribution head, when it
  exists, plugs in unchanged. 12.4 is not built, so the function measured here is the
  model-agnostic one below.
- :func:`permutation_attribution` is permutation importance through the public
  :class:`~fintfm.inference.classifier.FinancialTFMClassifier` API (``fit`` then
  ``predict_proba``; no private attribute or forward call). For each query feature the column is
  shuffled across held-out rows and the importance is the mean increase of held-out log-loss.
  It is **true to the fitted model on those rows**, not to the data-generating process, and it
  needs the model and labelled held-out rows, so it is not the zero-shot attribution of the
  proposal; it is the stand-in that makes the stability question measurable now.
- Per seed one stratified held-out explain set is drawn; each resample then draws a fresh
  stratified context from the remaining rows. Resample-to-resample stability therefore isolates
  the context draw; seed-to-seed stability (on the resample-averaged importance) adds a new explain
  set, new contexts and a new permutation stream.
- :func:`ground_truth_agreement` scores an attribution function against the prior's own labels
  (``sample_financial_task(return_attributions=True)``, task 12.3) on fresh synthetic tasks.
  That is agreement with the generator's global driver importance, not with another explainer.

Thresholds
----------
12.5's Verify asks for a test asserting a threshold "chosen and recorded before it is measured".
They are the constants below, recorded in ``openspec/changes/zero-shot-attribution/tasks.md``
in a commit that precedes every number from this module.

Run numbers are MEASURED only when produced by the command in the openspec Progress note on a
real checkpoint; the unit tests are SMOKE-TEST.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

#: Thresholds, fixed before any measurement (see the module docstring). They apply to the pinned
#: prior task of ``tests/test_attribution_stability.py``.
MIN_RESAMPLE_SPEARMAN = 0.60
MIN_SEED_SPEARMAN = 0.70
MIN_SEED_TOPK_JACCARD = 0.40

#: Top-k size for the overlap statistic on the real panels (12.2 uses 10).
TOP_K = 10

#: ``fn(X_context, y_context, X_explain, y_explain, seed) -> (d,)`` non-negative importance.
AttributionFn = Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int], np.ndarray]


@dataclass
class RunResult:
    """One (seed, resample) attribution."""

    seed: int
    resample: int
    importance: list[float]
    context_rate: float
    explain_rate: float
    seconds: float


@dataclass
class StabilityResult:
    """Stability of one attribution function on one panel."""

    dataset: str
    method: str
    n_rows: int
    n_features: int
    n_context: int
    n_explain: int
    n_seeds: int
    n_resamples: int
    top_k: int
    mean_importance: list[float]
    top_features: list[dict]
    resample_spearman_mean: float
    seed_spearman_mean: float
    resample_topk_jaccard_mean: float
    seed_topk_jaccard_mean: float
    seconds_per_attribution: float
    runs: list[RunResult] = field(default_factory=list)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation of two importance vectors; ``nan`` if either is constant."""
    from scipy.stats import rankdata

    ra, rb = rankdata(a), rankdata(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def topk_jaccard(a: np.ndarray, b: np.ndarray, k: int) -> float:
    """Jaccard overlap of the top-``k`` index sets of two importance vectors."""
    ta = set(np.argsort(-a, kind="stable")[:k].tolist())
    tb = set(np.argsort(-b, kind="stable")[:k].tolist())
    return len(ta & tb) / len(ta | tb)


def normalise(v: np.ndarray) -> np.ndarray:
    """Clip negatives to zero and scale to sum one (all-zero stays all-zero)."""
    v = np.clip(np.asarray(v, dtype=float), 0.0, None)
    s = v.sum()
    return v / s if s > 0 else v


def _log_loss(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Per-row binary log-loss of positive-class probabilities, clipped away from 0 and 1."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def permutation_attribution(
    model: object,
    n_repeats: int = 2,
    max_context: int = 1000,
    classifier_kwargs: dict | None = None,
) -> AttributionFn:
    """Build a permutation-importance attribution function around a FinTFM checkpoint.

    Args:
        model: A :class:`~fintfm.modeling.model.FinancialTFM` or a checkpoint path.
        n_repeats: Shuffles per feature; all repeats of a feature go in one ``predict_proba``.
        max_context: Passed to the classifier.
        classifier_kwargs: Further :class:`FinancialTFMClassifier` arguments.

    Returns:
        A function with the :data:`AttributionFn` signature. Its output is the mean increase of
        held-out log-loss when the feature is shuffled across the explain rows, normalised with
        :func:`normalise`. Only the public ``fit`` / ``predict_proba`` API is used.
    """
    from fintfm.inference.classifier import FinancialTFMClassifier

    def fn(
        X_ctx: np.ndarray, y_ctx: np.ndarray, X_exp: np.ndarray, y_exp: np.ndarray, seed: int
    ) -> np.ndarray:
        clf = FinancialTFMClassifier(
            model,
            max_context=max_context,
            random_state=seed,
            **(classifier_kwargs or {}),
        ).fit(X_ctx, y_ctx)
        yb = (np.asarray(y_exp) == clf.classes_[-1]).astype(float)
        base = _log_loss(yb, clf.predict_proba(X_exp)[:, -1]).mean()
        rng = np.random.default_rng(seed)
        n, d = X_exp.shape
        out = np.zeros(d)
        for j in range(d):
            stack = np.tile(X_exp, (n_repeats, 1))
            for r in range(n_repeats):
                stack[r * n : (r + 1) * n, j] = X_exp[rng.permutation(n), j]
            p = clf.predict_proba(stack)[:, -1]
            loss = _log_loss(np.tile(yb, n_repeats), p).mean()
            out[j] = loss - base
        return normalise(out)

    return fn


def _split(
    y: np.ndarray, n_context: int, n_explain: int, seed: int, resample: int, explain: np.ndarray
) -> np.ndarray:
    """Stratified context indices drawn from the rows outside ``explain``."""
    from sklearn.model_selection import StratifiedShuffleSplit

    rest = np.setdiff1d(np.arange(len(y)), explain)
    sss = StratifiedShuffleSplit(
        n_splits=1, train_size=n_context, test_size=None, random_state=seed * 10_007 + resample
    )
    ctx, _ = next(sss.split(rest, y[rest]))
    return rest[ctx]


def stability(
    X: np.ndarray,
    y: np.ndarray,
    fn: AttributionFn,
    name: str,
    method: str,
    seeds: tuple[int, ...] = (0, 1),
    n_resamples: int = 3,
    n_context: int = 500,
    n_explain: int = 500,
    top_k: int = TOP_K,
    feature_names: list[str] | None = None,
) -> StabilityResult:
    """Measure how stable ``fn``'s importance vector is across context resamples and seeds.

    Args:
        X: ``(n, d)`` features.
        y: ``(n,)`` binary labels.
        fn: Any attribution function (:data:`AttributionFn`).
        name: Panel name for the record.
        method: Attribution method name for the record.
        seeds: Seeds; each draws its own explain set and context draws.
        n_resamples: Context resamples per seed.
        n_context: Context rows per draw.
        n_explain: Held-out explain rows per seed.
        top_k: Size of the overlap set.
        feature_names: Optional column names.

    Returns:
        A :class:`StabilityResult` with the same statistics as ``attribution_baseline``:
        resample-to-resample (within a seed) and seed-to-seed (resample-averaged) Spearman and
        top-``k`` Jaccard.
    """
    from sklearn.model_selection import StratifiedShuffleSplit

    X, y = np.asarray(X), np.asarray(y)
    d = X.shape[1]
    runs: list[RunResult] = []
    for seed in seeds:
        sss = StratifiedShuffleSplit(
            n_splits=1, train_size=n_explain, test_size=None, random_state=seed
        )
        explain, _ = next(sss.split(X, y))
        for r in range(n_resamples):
            ctx = _split(y, n_context, n_explain, seed, r, explain)
            t0 = time.perf_counter()
            imp = normalise(fn(X[ctx], y[ctx], X[explain], y[explain], seed * 1000 + r))
            runs.append(
                RunResult(
                    seed,
                    r,
                    imp.tolist(),
                    float(np.mean(y[ctx])),
                    float(np.mean(y[explain])),
                    time.perf_counter() - t0,
                )
            )
    imps = {(q.seed, q.resample): np.asarray(q.importance) for q in runs}
    res_sp, res_j = [], []
    for s in seeds:
        for i, j in itertools.combinations(range(n_resamples), 2):
            res_sp.append(spearman(imps[(s, i)], imps[(s, j)]))
            res_j.append(topk_jaccard(imps[(s, i)], imps[(s, j)], top_k))
    seed_imp = {s: np.mean([imps[(s, r)] for r in range(n_resamples)], axis=0) for s in seeds}
    seed_sp, seed_j = [], []
    for a, b in itertools.combinations(seeds, 2):
        seed_sp.append(spearman(seed_imp[a], seed_imp[b]))
        seed_j.append(topk_jaccard(seed_imp[a], seed_imp[b], top_k))
    mean_imp = np.mean(list(seed_imp.values()), axis=0)
    names = feature_names or [f"x{i}" for i in range(d)]
    order = np.argsort(-mean_imp, kind="stable")[:top_k]
    top = [{"index": int(i), "name": names[i], "importance": float(mean_imp[i])} for i in order]
    nan = float("nan")
    return StabilityResult(
        dataset=name,
        method=method,
        n_rows=len(y),
        n_features=d,
        n_context=n_context,
        n_explain=n_explain,
        n_seeds=len(seeds),
        n_resamples=n_resamples,
        top_k=top_k,
        mean_importance=mean_imp.tolist(),
        top_features=top,
        resample_spearman_mean=float(np.nanmean(res_sp)) if res_sp else nan,
        seed_spearman_mean=float(np.nanmean(seed_sp)) if seed_sp else nan,
        resample_topk_jaccard_mean=float(np.mean(res_j)) if res_j else nan,
        seed_topk_jaccard_mean=float(np.mean(seed_j)) if seed_j else nan,
        seconds_per_attribution=float(np.mean([q.seconds for q in runs])),
        runs=runs,
    )


def ground_truth_agreement(
    fn: AttributionFn,
    n_tasks: int = 4,
    n_rows: int = 900,
    n_context: int = 400,
    max_features: int = 24,
    top_k: int = 5,
    seed: int = 0,
) -> dict:
    """Agreement of ``fn`` with the prior's ground-truth attributions (task 12.3).

    Each task is drawn with ``return_attributions=True``; ``fn`` sees a stratified context and
    the remaining rows as explain set. Reported per task, never only as a mean: Spearman with
    the true vector (ties, mostly zeros, are rank-averaged) and Jaccard of the top-``k`` sets,
    ``k`` capped at the number of columns with non-zero true attribution.

    Returns:
        ``{"per_task": [...], "spearman_mean": float, "topk_jaccard_mean": float}``; tasks whose
        classes are degenerate or whose true attribution is constant are skipped and listed.
    """
    from fintfm.prior.financial import sample_financial_task

    per_task, skipped = [], []
    for t in range(n_tasks):
        task = sample_financial_task(
            np.random.default_rng(seed * 997 + t),
            n_rows,
            max_features=max_features,
            return_attributions=True,
        )
        X, y = np.asarray(task.X, dtype=float), np.asarray(task.y).astype(int)
        truth = np.asarray(task.meta["attribution"], dtype=float)
        if len(set(y.tolist())) < 2 or y.sum() < 10 or truth.std() == 0:
            skipped.append(t)
            continue
        from sklearn.model_selection import StratifiedShuffleSplit

        sss = StratifiedShuffleSplit(
            n_splits=1, train_size=n_context, test_size=None, random_state=t
        )
        ctx, exp = next(sss.split(X, y))
        imp = normalise(fn(X[ctx], y[ctx], X[exp], y[exp], seed * 1000 + t))
        k = int(min(top_k, (truth > 0).sum()))
        per_task.append(
            {
                "task": t,
                "n_features": int(X.shape[1]),
                "n_true_nonzero": int((truth > 0).sum()),
                "spearman": spearman(imp, truth),
                "topk": k,
                "topk_jaccard": topk_jaccard(imp, truth, k),
            }
        )
    sp = [p["spearman"] for p in per_task]
    jc = [p["topk_jaccard"] for p in per_task]
    return {
        "per_task": per_task,
        "skipped_tasks": skipped,
        "spearman_mean": float(np.nanmean(sp)) if sp else float("nan"),
        "topk_jaccard_mean": float(np.mean(jc)) if jc else float("nan"),
    }


def render_table(results: list[StabilityResult], truth: dict | None = None) -> str:
    """Printable table of the stability statistics (and ground-truth agreement if given)."""
    lines = [
        f"{'panel':<28}{'method':<14}{'rs-rho':>8}{'seed-rho':>9}{'rs-J':>7}{'seed-J':>8}{'s/attr':>8}"
    ]
    for r in results:
        lines.append(
            f"{r.dataset:<28}{r.method:<14}{r.resample_spearman_mean:>8.3f}"
            f"{r.seed_spearman_mean:>9.3f}{r.resample_topk_jaccard_mean:>7.3f}"
            f"{r.seed_topk_jaccard_mean:>8.3f}{r.seconds_per_attribution:>8.1f}"
        )
    if truth is not None:
        lines.append("prior ground truth (12.3), per task: task  n_true_nonzero  rho  top-k J")
        for p in truth["per_task"]:
            lines.append(
                f"  {p['task']:<5}{p['n_true_nonzero']:<16}{p['spearman']:>6.3f}{p['topk_jaccard']:>8.3f}"
            )
        lines.append(
            f"  mean rho {truth['spearman_mean']:.3f}  mean J {truth['topk_jaccard_mean']:.3f}"
        )
    return "\n".join(lines)


def main() -> None:
    """CLI: ``python -m fintfm.experiments.attribution_stability --model runs/x.pt``."""
    from fintfm.evaluation.datasets import load_polish_bankruptcy, load_taiwan_bankruptcy

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", required=True, help="checkpoint path")
    p.add_argument("--panels", nargs="+", default=["polish"], choices=["polish", "taiwan"])
    p.add_argument("--polish-horizon", type=int, default=3)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    p.add_argument("--resamples", type=int, default=3)
    p.add_argument("--n-context", type=int, default=500)
    p.add_argument("--n-explain", type=int, default=500)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--truth-tasks", type=int, default=4, help="prior tasks for 12.3 agreement")
    p.add_argument("--out", type=Path, default=Path("runs/attribution_stability"))
    args = p.parse_args()

    fn = permutation_attribution(args.model, n_repeats=args.repeats, max_context=args.n_context)
    loaders = {
        "polish": lambda: load_polish_bankruptcy(args.polish_horizon),
        "taiwan": load_taiwan_bankruptcy,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    results = []
    for key in args.panels:
        ds = loaders[key]()
        res = stability(
            ds.X,
            ds.y,
            fn,
            ds.name,
            "permutation",
            tuple(args.seeds),
            args.resamples,
            args.n_context,
            args.n_explain,
        )
        results.append(res)
        (args.out / f"{ds.name}.json").write_text(json.dumps(asdict(res), indent=2))
    truth = ground_truth_agreement(fn, n_tasks=args.truth_tasks) if args.truth_tasks else None
    if truth is not None:
        (args.out / "prior_ground_truth.json").write_text(json.dumps(truth, indent=2))
    print(f"checkpoint: {args.model}")
    print(render_table(results, truth))


if __name__ == "__main__":
    main()
