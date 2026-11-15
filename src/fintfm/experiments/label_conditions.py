"""Full label-condition matrix: what does the model do with each kind of context label? (39.20)

``docs/results/FINDINGS.md`` §47 found the financial-only checkpoint scoring *higher* with
shuffled context labels than with true ones (0.6895 against 0.6841); §48 confirmed the fix.
Shuffling is one cell of a larger design. This module scores the whole matrix on the same
context features, so every condition differs from the others in the context labels alone:

- ``true``: the real labels. The reference.
- ``shuffled``: a permutation of the true labels. Keeps the label marginal exactly and
  destroys the feature-label link, the §47 test.
- ``inverted``: ``1 - y``. Keeps the link and flips its direction. A model that reads its
  context should score below 0.5 on a task it solves with true labels; one that ignores the
  labels scores the same as with ``true``.
- ``none``: every context label equal to class 0. The model has no label-absent mode, so a
  constant label is the nearest thing it can be given: the context carries features and no
  label information. Any AUC above chance then comes from the features alone.
- ``unrelated``: labels of an independent task of the same kind (same label marginal,
  independent features), so the labels look like a real task's and relate to nothing here.
- ``flip_<rate>`` (optional): each true label flipped independently with probability ``rate``,
  a dose-response between ``true`` and ``shuffled``.

Queries are scored against their **true** labels in every condition. Only the context labels
are altered.

Floors, printed beside every row: chance (0.5), an untrained control of the same architecture
with a fixed seed (as ``fintfm-capability``), and the ceiling of the task (the closed-form
Bayes AUC for Bayes-ceiling tasks, 1.0 for ``orientation``, a logistic regression fitted on
the true context labels for the other probes).

Tasks: the capability probes (:data:`~fintfm.experiments.capability.PROBES`, ``orientation``
and ``linear`` by default) and the Bayes-ceiling tasks of
:func:`~fintfm.experiments.capability._make_bayes_task`. Scoring goes through
:func:`~fintfm.experiments.capability._probe_positive_proba`, the path every probe number in
``FINDINGS.md`` was taken on.

Usage (MEASURED only when run on a real checkpoint; the tests use a tiny untrained model)::

    uv run python -m fintfm.experiments.label_conditions \\
        --models name=runs/ckpt.pt --out runs/label_conditions
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from fintfm.experiments.capability import (
    UNTRAINED_SEED,
    _bayes_optimal_mu,
    _make_bayes_task,
    _probe_positive_proba,
    make_probe,
)

if TYPE_CHECKING:
    from fintfm.modeling.model import FinancialTFM

#: The five conditions task 39.20 requires, in reporting order.
CONDITIONS: tuple[str, ...] = ("true", "shuffled", "inverted", "none", "unrelated")

#: Capability probes scored by default: the trivial single-feature orientation task and the
#: linear task. Bayes-ceiling tasks are added through ``bayes_targets``.
DEFAULT_PROBES: tuple[str, ...] = ("orientation", "linear")

#: ``build(seed) -> (X, y, n_ctx, ceiling)``; ``ceiling`` is ``None`` when fitted per draw.
TaskBuilder = Callable[[int], tuple[np.ndarray, np.ndarray, int, "float | None"]]


def make_context_labels(
    condition: str, y_ctx: np.ndarray, rng: np.random.Generator, y_other: np.ndarray | None = None
) -> np.ndarray:
    """Return the context labels for one condition.

    Args:
        condition: One of :data:`CONDITIONS`, or ``"flip_<rate>"`` with ``0 <= rate <= 1``.
        y_ctx: ``(n_ctx,)`` true binary context labels.
        rng: Generator for the conditions that draw (``shuffled``, ``flip_<rate>``).
        y_other: ``(n_ctx,)`` labels of an independent task; required for ``unrelated``.

    Returns:
        ``(n_ctx,)`` int64 labels. ``shuffled`` has exactly ``y_ctx``'s class counts.

    Raises:
        ValueError: On an unknown condition, a bad flip rate, or ``unrelated`` without
            ``y_other`` of the right length.
    """
    y = np.asarray(y_ctx, dtype=np.int64)
    if condition == "true":
        return y.copy()
    if condition == "shuffled":
        return rng.permutation(y)
    if condition == "inverted":
        return 1 - y
    if condition == "none":
        return np.zeros_like(y)
    if condition == "unrelated":
        if y_other is None or len(y_other) != len(y):
            raise ValueError("'unrelated' needs y_other with the same length as y_ctx")
        return np.asarray(y_other, dtype=np.int64).copy()
    if condition.startswith("flip_"):
        rate = float(condition[len("flip_") :])
        if not 0.0 <= rate <= 1.0:
            raise ValueError(f"flip rate must be in [0, 1], got {rate}")
        return np.where(rng.random(len(y)) < rate, 1 - y, y).astype(np.int64)
    raise ValueError(f"unknown label condition {condition!r}")


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    """ROC-AUC, NaN (not 0.0) for a single-class query set, which has no AUC."""
    from sklearn.metrics import roc_auc_score

    return float("nan") if len(np.unique(y)) < 2 else float(roc_auc_score(y, p))


def _tasks(
    probes: tuple[str, ...], bayes_targets: tuple[float, ...], n: int, n_features: int, rate: float
) -> Iterator[tuple[str, TaskBuilder]]:
    """Yield ``(task name, builder)`` for every capability probe and Bayes-ceiling target."""
    for probe in probes:

        def build(seed: int, probe: str = probe):
            Xtr, ytr, Xte, yte = make_probe(probe, n=n, n_features=n_features, rate=rate, seed=seed)
            ceiling = 1.0 if probe == "orientation" else None
            return np.vstack([Xtr, Xte]), np.concatenate([ytr, yte]), len(ytr), ceiling

        yield probe, build
    for target in bayes_targets:
        mu = _bayes_optimal_mu(target)

        def build_b(seed: int, mu: float = mu, target: float = target):
            X, y = _make_bayes_task(np.random.default_rng(seed), n, mu)
            return X, y, n // 2, target

        yield f"bayes_{target}", build_b


def _fitted_ceiling(X: np.ndarray, y: np.ndarray, n_ctx: int) -> float:
    """Logistic regression fitted on the true context labels, scored on the queries."""
    from sklearn.linear_model import LogisticRegression

    if len(np.unique(y[:n_ctx])) < 2:
        return float("nan")
    lr = LogisticRegression(max_iter=1000).fit(X[:n_ctx], y[:n_ctx])
    return _auc(y[n_ctx:], lr.predict_proba(X[n_ctx:])[:, 1])


def score_arm(
    model: FinancialTFM,
    probes: tuple[str, ...] = DEFAULT_PROBES,
    bayes_targets: tuple[float, ...] = (),
    flip_rates: tuple[float, ...] = (),
    seeds: tuple[int, ...] = (0, 1, 2),
    n: int = 1600,
    n_features: int = 20,
    rate: float = 0.2,
    n_ensemble: int = 8,
) -> dict[str, dict]:
    """Score one model under every label condition on every task, same features throughout.

    Args:
        model: A classification :class:`~fintfm.modeling.model.FinancialTFM`.
        probes: Capability probe names.
        bayes_targets: Nominal Bayes AUCs of extra Bayes-ceiling tasks.
        flip_rates: Extra ``flip_<rate>`` conditions.
        seeds: Task draws per task kind; also seed the label draws.
        n: Rows per task, half context and half query.
        n_features: Width of the capability probes (Bayes tasks use their own).
        rate: Positive rate of the capability probes (must leave both classes in the context).
        n_ensemble: Column-identity draws averaged per prediction.

    Returns:
        ``{task: {condition: {auc_mean, auc_std, n_seeds, mean_p, base_rate}, "_ceiling": x}}``
        where ``mean_p`` is the mean predicted probability beside the query ``base_rate``.
    """
    conditions = CONDITIONS + tuple(f"flip_{r}" for r in flip_rates)
    out: dict[str, dict] = {}
    for name, build in _tasks(probes, bayes_targets, n, n_features, rate):
        aucs: dict[str, list[float]] = {c: [] for c in conditions}
        mps: dict[str, list[float]] = {c: [] for c in conditions}
        base: list[float] = []
        ceilings: list[float] = []
        for s in seeds:
            X, y, n_ctx, ceiling = build(s)
            y_other = build(s + 10_000)[1]  # independent draw: independent features and labels
            rng = np.random.default_rng(s)
            base.append(float(y[n_ctx:].mean()))
            ceilings.append(_fitted_ceiling(X, y, n_ctx) if ceiling is None else ceiling)
            for c in conditions:
                y_c = make_context_labels(c, y[:n_ctx], rng, y_other[:n_ctx])
                y_full = np.concatenate([y_c, y[n_ctx:]])
                p = _probe_positive_proba(model, X, y_full, n_ctx, n_ensemble)
                aucs[c].append(_auc(y[n_ctx:], p))
                mps[c].append(float(p.mean()))
        cell: dict = {}
        for c in conditions:
            v = np.asarray(aucs[c], dtype=float)
            ok = v[np.isfinite(v)]
            cell[c] = {
                "auc_mean": float(ok.mean()) if ok.size else float("nan"),
                "auc_std": float(ok.std(ddof=1)) if ok.size > 1 else float("nan"),
                "n_seeds": int(ok.size),
                "mean_p": float(np.mean(mps[c])),
                "base_rate": float(np.mean(base)),
            }
        cf = np.asarray(ceilings, dtype=float)
        cell["_ceiling"] = float(np.nanmean(cf)) if np.isfinite(cf).any() else float("nan")
        out[name] = cell
    return out


def run(
    model_paths: dict[str, str],
    out_dir: Path,
    models: dict[str, FinancialTFM] | None = None,
    **kwargs,
) -> dict:
    """Score every checkpoint plus the untrained control and write ``label_conditions.json``.

    Args:
        model_paths: ``{arm name: checkpoint path}``; loaded unless ``models`` is given.
        out_dir: Directory for the JSON record.
        models: Already-built models by arm name, used instead of loading (tests).
        **kwargs: Passed to :func:`score_arm`.

    Returns:
        ``{"label", "conditions", "config", "floors", "arms": {arm: score_arm output}}``.
        The ``untrained_control`` arm is the floor and is always added.
    """
    import torch

    from fintfm.modeling.model import FinancialTFM

    arms = (
        dict(models)
        if models is not None
        else {n: FinancialTFM.load(p) for n, p in model_paths.items()}
    )
    if arms:
        torch.manual_seed(UNTRAINED_SEED)
        arms["untrained_control"] = FinancialTFM(next(iter(arms.values())).cfg)
    config = {k: list(v) if isinstance(v, tuple) else v for k, v in kwargs.items()}
    record = {
        "label": "SMOKE-TEST unless run on a real checkpoint (then MEASURED, command recorded)",
        "conditions": list(CONDITIONS),
        "config": {"models": model_paths, **config},
        "floors": {"chance": 0.5, "untrained_control": f"seeded {UNTRAINED_SEED}"},
        "arms": {name: score_arm(m.eval(), **kwargs) for name, m in arms.items()},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "label_conditions.json").write_text(json.dumps(record, indent=2))
    return record


def summarise(record: dict) -> str:
    """Render conditions side by side per arm and task, with the floors beside them."""
    cols = list(record["conditions"])
    cols += sorted(
        {c for a in record["arms"].values() for t in a.values() for c in t if c.startswith("flip_")}
    )
    lines = []
    for arm, tasks in record["arms"].items():
        header = f"{'task':<16}" + "".join(f"{c:>11}" for c in cols)
        lines += ["", f"arm: {arm}", header + f"{'ceiling':>10}{'chance':>8}"]
        for task, cell in tasks.items():
            row = "".join(f"{cell[c]['auc_mean']:>11.3f}" for c in cols)
            lines.append(f"{task:<16}{row}{cell['_ceiling']:>10.3f}{0.5:>8.2f}")
    lines += [
        "",
        "AUC of query rows against their true labels; only the context labels differ by column.",
        "Floors: chance 0.5 and the untrained_control arm. `none` is a constant class-0 label.",
        "Reading it: a context-reading model drops from `true` to ~0.5 on `shuffled`, `none` and",
        "`unrelated`, and below 0.5 on `inverted`. Equal columns mean the labels are ignored.",
    ]
    return "\n".join(lines)


def main() -> None:
    """Command-line entry: score checkpoints under every label condition."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--models", required=True, help="comma-separated name=path pairs")
    p.add_argument("--out", default="runs/label_conditions")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--probes", default=",".join(DEFAULT_PROBES))
    p.add_argument("--bayes-targets", default="0.7,0.9,0.999")
    p.add_argument("--flip-rates", default="", help="optional, e.g. 0.1,0.25,0.5")
    p.add_argument("--n", type=int, default=1600)
    p.add_argument("--n-ensemble", type=int, default=8)
    a = p.parse_args()

    def floats(s: str) -> tuple[float, ...]:
        return tuple(float(x) for x in s.split(",") if x)

    paths = dict(kv.split("=", 1) for kv in a.models.split(","))
    record = run(
        paths,
        Path(a.out),
        probes=tuple(x for x in a.probes.split(",") if x),
        bayes_targets=floats(a.bayes_targets),
        flip_rates=floats(a.flip_rates),
        seeds=tuple(int(x) for x in a.seeds.split(",")),
        n=a.n,
        n_ensemble=a.n_ensemble,
    )
    print(summarise(record))


if __name__ == "__main__":
    main()
