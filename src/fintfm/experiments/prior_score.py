"""Score a prior on MITRA's three criteria, without pretraining anything (task 48.14).

Why this exists
---------------

MITRA (arXiv:2510.21204) proposes that a synthetic prior be selected on **performance**,
**diversity** and **distinctiveness**, and reports that this is what separates a
state-of-the-art synthetic-only model from a mediocre one. `docs/FINDINGS.md` §110 measures
that the top fourteen TabArena ranks are all synthetic-pretrained, and §104/§102/§103/§108
between them close or weaken every architectural and scale lever this project had. The prior
is what is left, and this project has never evaluated its own prior on any of the three.

**Every statistic here is computed from fitted baselines on the prior's own tasks, never from
this project's model.** That is the design constraint, not an implementation convenience: a
diagnostic routed through our own checkpoint cannot distinguish "the prior does not contain
this structure" from "our model cannot learn this structure", which is exactly the confound
§49 and §51 cost days to untangle on the capacity question. The cost is that nothing here
predicts downstream accuracy; §93's 5x-volume null stands as the reminder that a prior change
can be measurably distinctive and still inert.

The three criteria, made operational
------------------------------------

- **Performance** — can *anything* learn this prior's tasks? Reported as the best fitted
  baseline's mean ROC-AUC. A prior of pure noise scores 0.5 and teaches a model to abstain and
  nothing else; a prior every baseline solves at 1.0 teaches nothing either. §42 established
  that this project's financial prior once sat at the noise end and that fixing the span was
  worth more than any architectural change of that period.
- **Diversity** — does the prior cover a range of difficulty, or emit one kind of task? The
  **standard deviation** of per-task AUC across draws. A high mean with low spread is the §42
  failure in its other direction.
- **Distinctiveness** — does it generate structure a *different* prior does not? The mean
  **AUC gap between a tree baseline and a linear one** (§111). Positive means axis-aligned,
  piecewise-constant structure; negative means smooth structure a linear model reaches. Two
  priors with the same sign and magnitude are, on this axis, the same prior.

Distinctiveness is a property of a *pair* of priors, and the single-prior number above is a
proxy for it: it places each prior on one axis, so two priors can be compared without training
anything. The direct cross-evaluation 48.14 also asks for — train on prior A, score on prior
B's tasks — needs checkpoints and belongs with the training runs, not here.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np

from fintfm.prior.base import Task

#: Priors this harness can score, by name. Diagnostic-only priors (`trivial`, `crossed`) are
#: included deliberately: `trivial` is the sanity check that the statistics can recognise an
#: easy prior, and a harness whose numbers nobody has seen on a known-easy case is not
#: calibrated.
PRIORS: dict[str, str] = {
    "financial": "fintfm.prior.financial:sample_financial_task",
    "scm": "fintfm.prior.scm:sample_scm_task",
    "tree": "fintfm.prior.tree:sample_tree_task",
    "trivial": "fintfm.prior.trivial:sample_trivial_task",
}


def _load(spec: str) -> Callable[..., Task]:
    module, fn = spec.split(":")
    import importlib

    return getattr(importlib.import_module(module), fn)


def score_prior(
    sampler: Callable[..., Task],
    n_tasks: int = 30,
    n_rows: int = 800,
    seed0: int = 0,
) -> dict:
    """Compute the three criteria for one prior.

    Args:
        sampler: A prior's task-sampling function, taking ``(rng, n_rows, ...)``.
        n_tasks: Tasks drawn. Each contributes one point to every statistic.
        n_rows: Rows per task, split half context / half query.
        seed0: First seed; tasks use ``seed0 .. seed0 + n_tasks``.

    Returns:
        ``{"performance", "diversity", "distinctiveness", "tree_auc", "linear_auc",
        "n_scored"}``. Values are NaN when no task yielded a scorable split.
    """
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import QuantileTransformer

    tree_aucs: list[float] = []
    linear_aucs: list[float] = []
    for seed in range(seed0, seed0 + n_tasks):
        rng = np.random.default_rng(seed)
        try:
            t = sampler(rng, n_rows, max_classes=2)
        except TypeError:
            # Not every prior takes max_classes; the financial one is binary by construction.
            t = sampler(rng, n_rows)
        X, y = np.nan_to_num(np.asarray(t.X, dtype=np.float64)), np.asarray(t.y)
        k = len(y) // 2
        # A split with one class on either side is unscorable, not a zero -- recording it as
        # a low AUC would make an imbalanced prior look like a weak one, which is the
        # distinction §43's base-rate work exists to preserve.
        if len(np.unique(y[:k])) < 2 or len(np.unique(y[k:])) < 2:
            continue
        et = ExtraTreesClassifier(n_estimators=100, random_state=0).fit(X[:k], y[:k])
        # The linear arm is conditioned before fitting, and this is load-bearing rather than
        # hygiene. Financial ratios are pathologically heavy-tailed -- §35 measured 110 of 136
        # V4FinBench features with a standard deviation over ten times their IQR -- so raw
        # features make lbfgs fail to converge, understate the linear AUC, and **inflate the
        # distinctiveness statistic exactly where the tails are worst**. An unconditioned
        # linear baseline would have reported the financial prior as far more tree-favourable
        # than it is. Rank conditioning is also what this project's own inference path uses by
        # default (§35), so the baseline is being given the same treatment as our model.
        lr = make_pipeline(
            QuantileTransformer(
                output_distribution="normal",
                n_quantiles=min(1000, k),
                random_state=0,
            ),
            LogisticRegression(max_iter=2000),
        ).fit(X[:k], y[:k])
        tree_aucs.append(float(roc_auc_score(y[k:], et.predict_proba(X[k:])[:, 1])))
        linear_aucs.append(float(roc_auc_score(y[k:], lr.predict_proba(X[k:])[:, 1])))

    if not tree_aucs:
        nan = float("nan")
        return {
            "performance": nan, "diversity": nan, "distinctiveness": nan,
            "tree_auc": nan, "linear_auc": nan, "n_scored": 0,
        }
    best = np.maximum(tree_aucs, linear_aucs)
    return {
        "performance": float(np.mean(best)),
        "diversity": float(np.std(best)),
        "distinctiveness": float(np.mean(np.array(tree_aucs) - np.array(linear_aucs))),
        "tree_auc": float(np.mean(tree_aucs)),
        "linear_auc": float(np.mean(linear_aucs)),
        "n_scored": len(tree_aucs),
    }


def summarise(scores: dict[str, dict]) -> str:
    """Render the scores, with each criterion's reading stated beside it."""
    lines = [
        f"{'prior':>12}{'perf':>9}{'diversity':>11}{'distinct':>10}"
        f"{'tree':>9}{'linear':>9}{'n':>5}",
    ]
    for name, s in scores.items():
        lines.append(
            f"{name:>12}{s['performance']:>9.4f}{s['diversity']:>11.4f}"
            f"{s['distinctiveness']:>+10.4f}{s['tree_auc']:>9.4f}"
            f"{s['linear_auc']:>9.4f}{s['n_scored']:>5}"
        )
    lines += [
        "",
        "perf       best fitted baseline's mean AUC -- 0.5 is noise, 1.0 teaches nothing",
        "diversity  standard deviation of that AUC -- a prior of one difficulty is a prior",
        "           that cannot teach a model when to abstain (§42)",
        "distinct   mean (tree - linear) AUC. Positive = axis-aligned structure a linear",
        "           model cannot reach; negative = smooth structure it can (§111). Two priors",
        "           with the same sign and magnitude are the same prior on this axis.",
        "",
        "All statistics come from fitted baselines, never from this project's model, so they",
        "cannot confuse 'the prior lacks this structure' with 'our model cannot learn it'.",
        "None of them predicts downstream accuracy: §93 measured a 5x volume increase as",
        "inert, and a distinctive prior can be inert the same way.",
    ]
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--priors", type=str, default="financial,scm,tree",
        help=f"comma-separated prior names; available: {','.join(PRIORS)}",
    )
    p.add_argument("--n-tasks", type=int, default=30)
    p.add_argument("--n-rows", type=int, default=800)
    p.add_argument("--seed0", type=int, default=0)
    p.add_argument("--out", type=str, default="runs/prior-score")
    args = p.parse_args()

    names = [n.strip() for n in args.priors.split(",") if n.strip()]
    unknown = [n for n in names if n not in PRIORS]
    if unknown:
        raise SystemExit(f"unknown prior(s) {unknown}; available: {sorted(PRIORS)}")

    scores = {}
    for name in names:
        scores[name] = score_prior(
            _load(PRIORS[name]), n_tasks=args.n_tasks, n_rows=args.n_rows, seed0=args.seed0
        )
        print(f"  {name} done", flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    record = {"scores": scores, "config": vars(args)}
    (out / "prior_score.json").write_text(json.dumps(record, indent=2))
    print("\n" + summarise(scores))


if __name__ == "__main__":
    main()
