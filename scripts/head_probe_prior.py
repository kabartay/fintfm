"""Head against linear probe against logistic regression, on the prior's own tasks.

§130 found a linear probe on FinTFM's pre-head representation beats the model's own head on
V4FinBench by +0.039 AP, and the TabArena page shows the same probe +154 Elo over the head. This
asks where that gap comes from, on tasks the checkpoint was trained on:

- the probe beats the head **here too**: the head under-reads its representation everywhere, a
  training or head problem the prior cannot be blamed for;
- the head matches or beats the probe here, and loses only on real data: the head is fitted to
  the prior and the prior differs from real tables where it matters.

The raw-feature logistic regression is the simple baseline the head must clear: on the prior's
own tasks, a model that cannot beat it has learnt less than one linear fit per task.

Every arm sees the same labelled rows, so no arm wins on label count (the confound DR3.4 tests on
V4FinBench). Per task: ``C`` context rows, ``P`` further labelled rows, ``Q`` scored rows.

- ``head``: the classifier with ``C`` plus ``P`` as its context, scored on ``Q``.
- ``probe``: the classifier with ``C`` as context; a logistic regression fitted on the pre-head
  representation of ``P`` (under that context), scored on the representation of ``Q``.
- ``raw_lr``: a standardised logistic regression on ``C`` plus ``P``'s raw features.

Usage::

    uv run python scripts/head_probe_prior.py --model runs/v4-cellattn-labels.pt --tasks 200
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fintfm.inference.classifier import FinancialTFMClassifier
from fintfm.prior.financial import sample_financial_task


def score_task(
    clf_full: FinancialTFMClassifier,
    clf_ctx: FinancialTFMClassifier,
    X: np.ndarray,
    y: np.ndarray,
    n_ctx: int,
    n_probe: int,
) -> dict[str, float]:
    """Score the three arms on one task.

    Args:
        clf_full: Classifier fitted here on ``C`` plus ``P``, for the head arm.
        clf_ctx: Classifier fitted here on ``C`` alone, for the probe arm.
        X: ``(n, F)`` the task's features, rows already shuffled.
        y: ``(n,)`` its 0/1 labels.
        n_ctx: Rows in ``C``.
        n_probe: Rows in ``P``; the rest are ``Q``.

    Returns:
        AP and ROC-AUC per arm, plus the task's ``Q`` base rate.
    """
    c, p = slice(0, n_ctx), slice(n_ctx, n_ctx + n_probe)
    lab, q = slice(0, n_ctx + n_probe), slice(n_ctx + n_probe, None)
    clf_full.fit(X[lab], y[lab])
    s_head = clf_full.predict_proba(X[q])[:, 1]
    clf_ctx.fit(X[c], y[c])
    rep_p = clf_ctx.transform_representation(X[p], stage="pre_head")
    rep_q = clf_ctx.transform_representation(X[q], stage="pre_head")
    s_probe = LogisticRegression(max_iter=1000).fit(rep_p, y[p]).predict_proba(rep_q)[:, 1]
    raw = make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(max_iter=1000))
    s_raw = raw.fit(X[lab], y[lab]).predict_proba(X[q])[:, 1]
    out = {"rate": float(y[q].mean())}
    for name, s in (("head", s_head), ("probe", s_probe), ("raw_lr", s_raw)):
        out[f"{name}_ap"] = float(average_precision_score(y[q], s))
        out[f"{name}_auc"] = float(roc_auc_score(y[q], s))
    return out


def main() -> None:
    """Draw prior tasks, score the three arms, print paired means and win counts."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--tasks", type=int, default=200)
    ap.add_argument("--n-rows", type=int, default=2048)
    ap.add_argument("--n-ctx", type=int, default=512)
    ap.add_argument("--n-probe", type=int, default=512)
    ap.add_argument("--min-pos", type=int, default=5, help="positives required in C, P and Q")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, default=Path("runs/head_probe_prior.json"))
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    common = {"device": args.device, "max_context": args.n_ctx + args.n_probe, "random_state": 0}
    clf_full = FinancialTFMClassifier(args.model, **common)
    clf_ctx = FinancialTFMClassifier(args.model, **common)
    rows, drawn = [], 0
    while len(rows) < args.tasks and drawn < 20 * args.tasks:
        drawn += 1
        task = sample_financial_task(rng, args.n_rows)
        order = rng.permutation(len(task.y))
        X, y = task.X[order], (task.y[order] > 0).astype(int)
        cuts = [(0, args.n_ctx), (args.n_ctx, args.n_ctx + args.n_probe),
                (args.n_ctx + args.n_probe, len(y))]
        if min(y[a:b].sum() for a, b in cuts) < args.min_pos:
            continue
        rows.append(score_task(clf_full, clf_ctx, X, y, args.n_ctx, args.n_probe))
        if len(rows) % 20 == 0:
            print(f"{len(rows)} tasks scored ({drawn} drawn)", flush=True)

    summary: dict[str, object] = {"n_tasks": len(rows), "n_drawn": drawn, "args": vars(args) | {"out": str(args.out)}}
    print(f"\n{len(rows)} tasks ({drawn} drawn), mean Q base rate {np.mean([r['rate'] for r in rows]):.3f}")
    for metric in ("ap", "auc"):
        m = {a: np.array([r[f"{a}_{metric}"] for r in rows]) for a in ("head", "probe", "raw_lr")}
        print(f"{metric.upper():>4}: " + "  ".join(f"{a} {v.mean():.4f}" for a, v in m.items()))
        for a, b in (("probe", "head"), ("head", "raw_lr"), ("probe", "raw_lr")):
            d = m[a] - m[b]
            print(f"      {a} - {b}: {d.mean():+.4f}, wins {int((d > 0).sum())}, losses {int((d < 0).sum())}")
            summary[f"{metric}_{a}_minus_{b}"] = float(d.mean())
        summary |= {f"{metric}_{a}": float(v.mean()) for a, v in m.items()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "tasks": rows}, indent=1))


if __name__ == "__main__":
    main()
