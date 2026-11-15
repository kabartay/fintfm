"""Sign/direction inference harness (openspec task 39.10).

Why this exists
---------------
Aggregate AUC cannot say *why* a model fails a task. If a model scores 0.5 on ``y = 1[x0 < 0]``
it may have no idea which column matters, or it may know the column and assume the wrong
direction (a prior that always read "higher means riskier" would do exactly that; FINDINGS §47,
CLAUDE.md "check the prior admits no global rule"). This harness isolates the direction.

Design
------
- A **pair** is two tasks with the **same** feature matrix ``X`` (bit-identical, not merely the
  same distribution) and mirror-image labels: ``y_pos = 1[beta^T x > 0]`` and
  ``y_neg = 1[-beta^T x > 0] = 1 - y_pos``. Features are symmetric about zero, so both labels sit
  near a 50% rate and neither task is the easier one.
- The first half of the rows is the in-context set, the second half the query set. Each arm is
  scored once with the positive task's context labels and once with the negative task's, on the
  *same* query rows.
- Two families: ``sign`` (``beta = e_0``, so ``y = 1[x0 > 0]`` against ``1[x0 < 0]``) and
  ``beta`` (a random ``beta`` on the first ``n_relevant`` columns, redrawn per task and seed,
  the remaining columns pure noise), the ``sigma(beta^T x)`` extension.

Reported per arm and family, over seeds
---------------------------------------
- ``auc_pos`` / ``auc_neg``: AUC of each side against its own query labels.
- ``auc_flipped``: AUC of the positive-context predictions against the **negative** labels. A
  model that follows the context's direction scores ``1 - auc_pos`` here; one that ignores the
  context and keeps a fixed rule scores about ``auc_pos`` on one side and ``1 - auc_pos`` on the
  other, so its ``auc_pos`` and ``auc_neg`` split around 0.5 rather than both being high.
- ``flip_corr``: Spearman correlation of the two prediction vectors. -1 means the model reversed
  its ranking exactly when the labels reversed; +1 means the context did not change it.
- ``sign_correct``: the Verify of 39.10. For every relevant column ``j`` the recovered sign is
  ``sign(Spearman(prediction, x_j))`` and is compared with ``sign(beta_j)`` for each task
  separately (so AUC is not used as a proxy). Reported as the fraction of (task, column) pairs
  recovered correctly; 0.5 is chance.
- Floors: an **untrained control** of the same architecture with seeded weights (CLAUDE.md
  "Every evaluation needs a floor") and logistic regression fitted on the same context. The
  control shows what the architecture plus the context does with no training; logistic
  regression is the ceiling for a linear rule.

Nothing here trains. Numbers are labelled MEASURED only when produced by the command in the
openspec Progress note on real checkpoints; the unit tests are SMOKE-TEST only.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from fintfm.modeling.model import FinancialTFM

#: Seed for the untrained control's weights; shared with the other experiment floors.
UNTRAINED_SEED = 20260910

#: Task families.
FAMILIES: tuple[str, ...] = ("sign", "beta")


@dataclass
class PairResult:
    """One arm on one family, summarised over seeds and tasks."""

    arm: str
    family: str
    auc_pos: float
    auc_neg: float
    auc_flipped: float
    flip_corr: float
    sign_correct: float
    n_tasks: int


def make_pair(
    beta: np.ndarray, n: int, d: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw one mirrored pair of tasks sharing one feature matrix.

    Args:
        beta: Direction vector of length ``k <= d``; columns beyond ``k`` carry no signal.
        n: Rows; the first half is context, the second half the query set.
        d: Total columns.
        seed: Random seed. The same arguments give bit-identical output.

    Returns:
        ``(X, y_pos, y_neg)`` with ``X`` of shape ``(n, d)`` float32, drawn standard normal
        (symmetric about zero), ``y_pos = 1[beta^T x > 0]`` and ``y_neg = 1 - y_pos``, the exact
        mirror image.

    Raises:
        ValueError: If ``beta`` is longer than ``d`` or all zero.
    """
    beta = np.asarray(beta, dtype=np.float64)
    if len(beta) > d:
        raise ValueError(f"beta has {len(beta)} entries but only {d} columns")
    if not np.any(beta):
        raise ValueError("beta must have a nonzero entry")
    X = np.random.default_rng(seed).normal(size=(n, d)).astype(np.float32)
    score = X[:, : len(beta)].astype(np.float64) @ beta
    y_pos = (score > 0).astype(np.int64)
    return X, y_pos, 1 - y_pos


def draw_beta(family: str, seed: int, n_relevant: int) -> np.ndarray:
    """The direction vector of one task.

    Args:
        family: ``"sign"`` (``beta = e_0``) or ``"beta"`` (random per seed).
        seed: Random seed.
        n_relevant: Length of the random vector for the ``beta`` family.

    Returns:
        ``beta``; for the random family each entry has magnitude at least 0.3 so that every
        column's true sign is unambiguous.

    Raises:
        ValueError: On an unknown family.
    """
    if family == "sign":
        return np.array([1.0])
    if family == "beta":
        rng = np.random.default_rng(np.random.SeedSequence([seed, 7]))
        mag = np.abs(rng.normal(size=n_relevant)) + 0.3
        sgn = rng.choice([-1.0, 1.0], size=n_relevant)
        return mag * sgn
    raise ValueError(f"unknown family {family!r}")


def recovered_signs(pred: np.ndarray, X: np.ndarray, k: int) -> np.ndarray:
    """Sign of the rank correlation between predictions and each of the first ``k`` columns.

    Args:
        pred: Positive-class scores on the query rows.
        X: Query features, ``(m, >= k)``.
        k: Number of relevant columns to check.

    Returns:
        Array of ``k`` values in ``{-1, 0, +1}``; 0 only if the prediction is constant.
    """
    from scipy.stats import spearmanr

    out = np.zeros(k)
    if np.ptp(pred) == 0:
        return out
    for j in range(k):
        out[j] = np.sign(spearmanr(pred, X[:, j]).statistic)
    return out


def _score_in_context(
    model: FinancialTFM,
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xte: np.ndarray,
    max_context: int,
    seed: int,
) -> np.ndarray:
    """Positive-class probability for the query rows through the public classifier."""
    from fintfm.inference.classifier import FinancialTFMClassifier

    clf = FinancialTFMClassifier(
        model,
        max_context=max_context,
        context_strategy="uniform",
        feature_transform="rank",
        random_state=seed,
    ).fit(Xtr, ytr)
    return clf.predict_proba(Xte)[:, 1]


def _score_logreg(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray) -> np.ndarray:
    """Logistic-regression floor/ceiling fitted on the same context."""
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(max_iter=1000).fit(Xtr, ytr).predict_proba(Xte)[:, 1]


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman correlation, NaN (not 0) when either vector is constant.

    A constant prediction has no ranking to compare; NaN keeps it out of the mean rather than
    reading as "no relationship".
    """
    from scipy.stats import spearmanr

    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    return float(spearmanr(a, b).statistic)


def run(
    model_paths: dict[str, str],
    out_dir: Path | None = None,
    seeds: tuple[int, ...] = (0, 1, 2),
    tasks_per_seed: int = 4,
    n: int = 2000,
    d: int = 8,
    n_relevant: int = 4,
    max_context: int = 1000,
    models: dict[str, FinancialTFM] | None = None,
) -> dict:
    """Score every arm on every mirrored pair and record direction-following.

    Args:
        model_paths: ``{name: checkpoint path}``; one untrained control is built from the first
            one's architecture. Ignored when ``models`` is given.
        out_dir: If given, ``sign_inference.json`` is written there.
        seeds: Task seeds.
        tasks_per_seed: Random-``beta`` pairs per seed (the ``sign`` family has one per seed).
        n: Rows per task, half context and half query.
        d: Total columns per table.
        n_relevant: Columns carrying signal in the ``beta`` family.
        max_context: Context rows handed to the in-context arms.
        models: Already-built models by name (used by tests); the untrained control is added.

    Returns:
        The record: config, per-arm per-family results.

    Raises:
        ValueError: If ``d`` exceeds a model's ``max_features`` or is below ``n_relevant``.
    """
    import torch
    from sklearn.metrics import roc_auc_score

    from fintfm.modeling.model import FinancialTFM

    if d < n_relevant:
        raise ValueError("d must be at least n_relevant")
    loaded = (
        dict(models)
        if models is not None
        else {name: FinancialTFM.load(p) for name, p in model_paths.items()}
    )
    if loaded:
        torch.manual_seed(UNTRAINED_SEED)  # seeded: an unseeded floor moves by 0.2 (§42)
        loaded["untrained_control"] = FinancialTFM(next(iter(loaded.values())).cfg)
    for name, m in loaded.items():
        if d > m.cfg.max_features:
            raise ValueError(f"d={d} exceeds {name}'s max_features {m.cfg.max_features}")

    nc = n // 2
    acc: dict[tuple[str, str], dict[str, list[float]]] = {}
    for family in FAMILIES:
        n_tasks = 1 if family == "sign" else tasks_per_seed
        for seed in seeds:
            for t in range(n_tasks):
                task_seed = seed * 1000 + t
                beta = draw_beta(family, task_seed, n_relevant)
                X, y_pos, y_neg = make_pair(beta, n, d, task_seed)
                Xtr, Xte = X[:nc], X[nc:]
                ys = {"pos": y_pos, "neg": y_neg}
                if any(
                    len(np.unique(y[:nc])) < 2 or len(np.unique(y[nc:])) < 2 for y in ys.values()
                ):
                    continue
                preds: dict[str, dict[str, np.ndarray]] = {}
                for side, y in ys.items():
                    for name, m in loaded.items():
                        preds.setdefault(name, {})[side] = _score_in_context(
                            m, Xtr, y[:nc], Xte, max_context, task_seed
                        )
                    preds.setdefault("logistic_regression", {})[side] = _score_logreg(
                        Xtr, y[:nc], Xte
                    )
                k = len(beta)
                for arm, p in preds.items():
                    a = acc.setdefault(
                        (arm, family),
                        {"pos": [], "neg": [], "flip": [], "corr": [], "sign": []},
                    )
                    a["pos"].append(float(roc_auc_score(y_pos[nc:], p["pos"])))
                    a["neg"].append(float(roc_auc_score(y_neg[nc:], p["neg"])))
                    a["flip"].append(float(roc_auc_score(y_neg[nc:], p["pos"])))
                    a["corr"].append(_safe_corr(p["pos"], p["neg"]))
                    for side, true_sign in (("pos", np.sign(beta)), ("neg", -np.sign(beta))):
                        got = recovered_signs(p[side], Xte, k)
                        a["sign"].extend((got == true_sign).astype(float).tolist())
        print(f"  family {family} done", flush=True)

    results = []
    for (arm, family), a in sorted(acc.items()):
        corr = np.array(a["corr"])
        results.append(
            PairResult(
                arm=arm,
                family=family,
                auc_pos=float(np.mean(a["pos"])),
                auc_neg=float(np.mean(a["neg"])),
                auc_flipped=float(np.mean(a["flip"])),
                flip_corr=float(np.nanmean(corr)) if not np.all(np.isnan(corr)) else float("nan"),
                sign_correct=float(np.mean(a["sign"])),
                n_tasks=len(a["pos"]),
            )
        )
    record = {
        "task": "paired tasks: identical X, y_neg = 1 - y_pos, y_pos = 1[beta^T x > 0]",
        "config": {
            "models": model_paths,
            "seeds": list(seeds),
            "tasks_per_seed": tasks_per_seed,
            "n": n,
            "d": d,
            "n_relevant": n_relevant,
            "max_context": max_context,
        },
        "results": [asdict(r) for r in results],
    }
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "sign_inference.json").write_text(json.dumps(record, indent=2))
    return record


def summarise(record: dict) -> str:
    """Render the per-arm direction table and how to read it.

    Args:
        record: Output of :func:`run`.

    Returns:
        A printable multi-line string.
    """
    head = (
        f"{'arm':>20} {'family':>6} {'auc_pos':>8} {'auc_neg':>8} {'flipped':>8} "
        f"{'flip_corr':>10} {'sign_ok':>8} {'tasks':>6}"
    )
    lines = [head]
    for r in record["results"]:
        lines.append(
            f"{r['arm']:>20} {r['family']:>6} {r['auc_pos']:>8.3f} {r['auc_neg']:>8.3f} "
            f"{r['auc_flipped']:>8.3f} {r['flip_corr']:>+10.3f} {r['sign_correct']:>8.3f} "
            f"{r['n_tasks']:>6}"
        )
    lines += [
        "",
        "Reading it: a model that infers direction from context has auc_pos and auc_neg both",
        "high, flipped (positive-context predictions against the negative labels) near",
        "1 - auc_pos, flip_corr near -1 and sign_ok near 1. A fixed rule keeps flip_corr near",
        "+1 and splits auc_pos/auc_neg around 0.5. The untrained control is the floor and",
        "logistic regression the linear ceiling; sign_ok of 0.5 is chance.",
    ]
    return "\n".join(lines)


def main() -> None:
    """Command-line entry point: ``python -m fintfm.experiments.sign_inference``."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--models",
        type=str,
        required=True,
        help="comma-separated name=path pairs, e.g. fin00=runs/a.pt,fin10=runs/b.pt",
    )
    p.add_argument("--out", type=str, default="runs/sign_inference")
    p.add_argument("--seeds", type=str, default="0,1,2")
    p.add_argument("--tasks-per-seed", type=int, default=4)
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--d", type=int, default=8)
    p.add_argument("--n-relevant", type=int, default=4)
    p.add_argument("--max-context", type=int, default=1000)
    args = p.parse_args()
    paths = dict(pair.split("=", 1) for pair in args.models.split(",") if pair)
    record = run(
        paths,
        Path(args.out),
        seeds=tuple(int(s) for s in args.seeds.split(",")),
        tasks_per_seed=args.tasks_per_seed,
        n=args.n,
        d=args.d,
        n_relevant=args.n_relevant,
        max_context=args.max_context,
    )
    print("\n" + summarise(record))


if __name__ == "__main__":
    main()
