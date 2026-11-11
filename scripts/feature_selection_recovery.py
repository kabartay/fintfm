"""Feature-selection recovery (openspec task 39.9): does a checkpoint find the relevant column?

Five extremely simple binary tasks over standard-Gaussian columns, each labelled by an exact
rule on one or two columns (``y = 1[x_3 > 0]``, ``y = 1[-x_3 > 0]``, ``y = 1[x_7 > 0]``,
``y = 1[x_2 > 0]``, ``y = 1[x_2 + x_5 > 0]``). The labels are the rule with no noise, so a
model that reads the right column scores AUC 1.0 and any shortfall is the model's, not the
task's. The binary reading of "y = x_3" is its sign; the classifier API is binary.

Per task, context size and seed, the script reports:

- the checkpoint's AUC, through :class:`FinancialTFMClassifier` (never the network's forward
  method), beside logistic regression, an oracle (the true score, ``x_3``, ``-x_3``, ...),
  and the untrained-control floor (``FinancialTFM(cfg)`` at the capability harness's fixed
  weight seed, ``fintfm.experiments.capability.UNTRAINED_SEED``);
- **ablation importance** (the Verify clause's measure): for each column, the drop in the
  checkpoint's query AUC when that column is permuted across the query rows, averaged over
  several permutations. The recovered column is the arg-max; a task counts as *recovered*
  when the top-``k`` columns (``k`` = number of true columns) are exactly the true set, and the
  report gives the true columns' share of the total positive importance.

Usage:
    uv run --frozen python scripts/feature_selection_recovery.py \\
        --model /path/to/checkpoint.pt --out result.json
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from fintfm.experiments.capability import UNTRAINED_SEED
from fintfm.inference.classifier import FinancialTFMClassifier
from fintfm.modeling.model import FinancialTFM

#: ``name -> (true column -> coefficient)``; the label is ``1[sum coef * x_col > 0]``.
TASK_RULES: dict[str, dict[int, float]] = {
    "x3": {3: 1.0},
    "neg_x3": {3: -1.0},
    "x7": {7: 1.0},
    "x2": {2: 1.0},
    "x2_plus_x5": {2: 1.0, 5: 1.0},
}


@dataclass
class SimpleTask:
    """One synthetic task split into context and query rows.

    Attributes:
        name: Key of :data:`TASK_RULES`.
        X_ctx: ``(n_ctx, n_features)`` context features.
        y_ctx: ``(n_ctx,)`` context labels in ``{0, 1}``.
        X_q: ``(n_q, n_features)`` query features.
        y_q: ``(n_q,)`` query labels.
        score_q: ``(n_q,)`` the true linear score on the query rows (the oracle's ranking).
        true_cols: Sorted indices of the columns the rule reads.
    """

    name: str
    X_ctx: np.ndarray
    y_ctx: np.ndarray
    X_q: np.ndarray
    y_q: np.ndarray
    score_q: np.ndarray
    true_cols: tuple[int, ...]


def rule_score(name: str, X: np.ndarray) -> np.ndarray:
    """The task's true score ``sum coef_j x_j``; the label is its sign.

    Args:
        name: Key of :data:`TASK_RULES`.
        X: ``(n, n_features)`` features.

    Returns:
        ``(n,)`` score.
    """
    return sum(c * X[:, j] for j, c in TASK_RULES[name].items())


def make_task(name: str, n_ctx: int, n_q: int, n_features: int, seed: int) -> SimpleTask:
    """Build one task. Labels are exactly ``1[rule_score > 0]``.

    Args:
        name: Key of :data:`TASK_RULES`.
        n_ctx: Context rows.
        n_q: Query rows.
        n_features: Gaussian columns (must exceed the highest true column).
        seed: Seed for the draw.

    Returns:
        The task, with the context drawn to hold both classes (a redraw, same stream, if not).
    """
    rule = TASK_RULES[name]
    if n_features <= max(rule):
        raise ValueError(f"n_features={n_features} does not contain column {max(rule)}")
    rng = np.random.default_rng([seed, n_ctx, n_features, sorted(TASK_RULES).index(name)])
    while True:
        X = rng.standard_normal((n_ctx + n_q, n_features))
        y = (rule_score(name, X) > 0).astype(np.int64)
        if 0 < y[:n_ctx].sum() < n_ctx and 0 < y[n_ctx:].sum() < n_q:
            break
    return SimpleTask(
        name, X[:n_ctx], y[:n_ctx], X[n_ctx:], y[n_ctx:], rule_score(name, X[n_ctx:]),
        tuple(sorted(rule)),
    )


def _auc(model: FinancialTFM, task: SimpleTask, X_q: np.ndarray, seed: int, fitted=None):
    """Fit (unless ``fitted``) and score the query rows; returns ``(auc, clf)``."""
    clf = fitted
    if clf is None:
        clf = FinancialTFMClassifier(
            model, max_context=len(task.y_ctx), context_strategy="uniform",
            feature_transform="rank", n_ensemble=1, random_state=seed,
        ).fit(task.X_ctx, task.y_ctx)
    return float(roc_auc_score(task.y_q, clf.predict_proba(X_q)[:, 1])), clf


def ablation_importance(
    model: FinancialTFM, task: SimpleTask, seed: int, n_perm: int
) -> tuple[float, np.ndarray]:
    """AUC drop per column when that column is permuted across the query rows.

    Args:
        model: The checkpoint.
        task: The task.
        seed: Classifier and permutation seed.
        n_perm: Permutations averaged per column.

    Returns:
        ``(base_auc, drop)`` with ``drop`` of shape ``(n_features,)``.
    """
    base, clf = _auc(model, task, task.X_q, seed)
    rng = np.random.default_rng(seed + 1)
    drop = np.zeros(task.X_q.shape[1])
    for j in range(task.X_q.shape[1]):
        aucs = []
        for _ in range(n_perm):
            Xp = task.X_q.copy()
            Xp[:, j] = Xp[rng.permutation(len(Xp)), j]
            aucs.append(_auc(model, task, Xp, seed, clf)[0])
        drop[j] = base - float(np.mean(aucs))
    return base, drop


def recovery(drop: np.ndarray, true_cols: tuple[int, ...]) -> dict[str, object]:
    """Recovery measures from an importance vector.

    Args:
        drop: Per-column importance (AUC drop).
        true_cols: The columns the rule reads.

    Returns:
        ``top_col`` (arg-max), ``top_hit`` (it is a true column), ``exact`` (the top-k columns
        equal the true set) and ``true_share`` (true columns' share of the positive importance,
        NaN when none is positive).
    """
    k = len(true_cols)
    top = tuple(sorted(int(i) for i in np.argsort(-drop)[:k]))
    pos = np.clip(drop, 0, None)
    share = float(pos[list(true_cols)].sum() / pos.sum()) if pos.sum() > 0 else float("nan")
    return {
        "top_col": int(np.argmax(drop)),
        "top_hit": bool(int(np.argmax(drop)) in true_cols),
        "exact": top == true_cols,
        "true_share": share,
    }


def run(
    model_path: str, ctx_sizes: list[int], seeds: list[int], n_features: int, n_q: int,
    n_perm: int,
) -> list[dict[str, object]]:
    """Score every task at every context size and seed.

    Args:
        model_path: Checkpoint path.
        ctx_sizes: Context row counts.
        seeds: Task seeds.
        n_features: Gaussian columns per task.
        n_q: Query rows.
        n_perm: Permutations per column.

    Returns:
        One record per (task, context size, seed).
    """
    model = FinancialTFM.load(model_path)
    torch.manual_seed(UNTRAINED_SEED)
    untrained = FinancialTFM(model.cfg)
    rows = []
    for name in TASK_RULES:
        for n_ctx in ctx_sizes:
            for seed in seeds:
                t0 = time.time()
                task = make_task(name, n_ctx, n_q, n_features, seed)
                base, drop = ablation_importance(model, task, seed, n_perm)
                lr = LogisticRegression(max_iter=1000).fit(task.X_ctx, task.y_ctx)
                rec = {
                    "task": name, "n_ctx": n_ctx, "seed": seed,
                    "auc_checkpoint": base,
                    "auc_logreg": float(roc_auc_score(task.y_q, lr.predict_proba(task.X_q)[:, 1])),
                    "auc_oracle": float(roc_auc_score(task.y_q, task.score_q)),
                    "auc_untrained": _auc(untrained, task, task.X_q, seed)[0],
                    "true_cols": list(task.true_cols),
                    "importance": [float(d) for d in drop],
                    **recovery(drop, task.true_cols),
                }
                rows.append(rec)
                print(f"{name} n_ctx={n_ctx} seed={seed} auc={base:.3f} "
                      f"top={rec['top_col']} ({time.time() - t0:.0f}s)", flush=True)
    return rows


def summarise(rows: list[dict[str, object]]) -> str:
    """Markdown table: means over seeds per (task, context size).

    Args:
        rows: Output of :func:`run`.

    Returns:
        The table as text.
    """
    out = [
        (
            "| task | n_ctx | ckpt AUC | logreg AUC | oracle AUC | untrained AUC | "
            "true cols | top col per seed | exact recovery | true-col importance share |"
        ),
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    keys = sorted({(r["task"], r["n_ctx"]) for r in rows}, key=lambda k: (list(TASK_RULES).index(k[0]), k[1]))
    for task, n in keys:
        g = [r for r in rows if r["task"] == task and r["n_ctx"] == n]
        share = float(np.nanmean([r["true_share"] for r in g]))
        out.append(
            f"| {task} | {n} | {float(np.mean([r['auc_checkpoint'] for r in g])):.3f} | {float(np.mean([r['auc_logreg'] for r in g])):.3f} | "
            f"{float(np.mean([r['auc_oracle'] for r in g])):.3f} | {float(np.mean([r['auc_untrained'] for r in g])):.3f} | {g[0]['true_cols']} | "
            f"{[r['top_col'] for r in g]} | {sum(bool(r['exact']) for r in g)}/{len(g)} | "
            f"{share:.2f} |"
        )
    return "\n".join(out)


def main() -> None:
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--ctx-sizes", default="32,128,512")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--n-features", type=int, default=15)
    ap.add_argument("--n-query", type=int, default=300)
    ap.add_argument("--n-perm", type=int, default=2)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    rows = run(
        a.model, [int(s) for s in a.ctx_sizes.split(",")], [int(s) for s in a.seeds.split(",")],
        a.n_features, a.n_query, a.n_perm,
    )
    print(summarise(rows))
    if a.out:
        a.out.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
