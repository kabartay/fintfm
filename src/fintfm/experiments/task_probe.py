"""Task 39.7: does the model's context representation separate by task family?

An in-context model that forms a task posterior should hold different internal state after
reading a linear task than after reading an XOR task. If the state does not separate by the
family that generated the context, the model is not forming a task posterior, whatever its
downstream AUC (a Bayesian-regret-adjacent diagnostic and the complement to the Bayes-ceiling
test, ``openspec/changes/cell-attention-and-task-inference`` task 39.7).

**The families.** :data:`FAMILIES` holds 24 small closed-form generators: linear (sparse,
dense, noisy), single threshold, band, AND, OR, XOR, three-way and four-way interactions,
additive-plus-interaction, radial, saddle, sine, checkerboard, majority vote, depth-2 tree,
ratio, Gaussian mixture, latent factor, and three structural causal models (chain, fork with a
hidden cause, and a label-caused feature). Every generator draws its own parameters (which
columns, which weights, which signs) from a seed, so two seeds of one family are two different
tasks of the same kind. Labels are the score thresholded at its median, so every task is
balanced and the label rate carries no information about the family.

**The task representation.** For each task the model reads the context ``(X, y)`` through
:class:`~fintfm.inference.classifier.FinancialTFMClassifier` and is then asked for the
``pre_head`` representation
(:meth:`~fintfm.inference.classifier.FinancialTFMClassifier.transform_representation`) of one
fixed probe set of rows, the same for every task. The probe rows are identical, so any
difference between two tasks' representations comes from the context. The representation is
the mean and the standard deviation over probe rows of that tensor, a vector of
``2 * d_model`` numbers. Nothing here reaches into the model's private context state.

**The probe and its floors.** A standardised, L2-regularised logistic regression predicts the
family from the vector, cross-validated over task instances, so every score is on tasks whose
parameters the probe never saw (held-out instances of the families, not held-out families:
a classifier cannot name a family it was never trained on). It is read against:

- chance ``1 / n_families``, and a label-permutation chance (the same probe on shuffled family
  labels), which is the number to beat because a ``2 * d_model``-wide vector on a few hundred
  tasks can overfit;
- an **untrained control** of the same architecture (seeded weights), the floor
  CLAUDE.md's "Every evaluation needs a floor" asks for: the untrained network still passes the
  context through random attention, so some separation is expected from it;
- a **raw-feature-statistics baseline** with no model: the label-conditional statistics of
  the context table (sorted correlations of the label with each column, its square and its
  absolute value, the strongest pairwise-product correlations, and the mean absolute
  between-column correlation). A representation worth reading must beat it.

Labelling: the run below is MEASURED only when executed with a real checkpoint and the command
recorded; the unit tests are SMOKE-TEST only.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from fintfm.modeling.model import FinancialTFM

#: Seed for the untrained control's weights; shared with the capability suite's floor.
UNTRAINED_SEED = 20260910

#: Seed of the fixed probe rows every task is represented on.
PROBE_SEED = 20261115

#: Smallest table width: the widest family (majority vote) uses five columns.
MIN_WIDTH = 6

Generator = Callable[[np.random.Generator, int, int], tuple[np.ndarray, np.ndarray]]


def _signs(rng: np.random.Generator, k: int) -> np.ndarray:
    """``k`` weights with magnitude in [0.5, 1.5] and random signs."""
    return rng.uniform(0.5, 1.5, k) * rng.choice([-1.0, 1.0], k)


def _cols(rng: np.random.Generator, d: int, k: int) -> np.ndarray:
    """``k`` distinct column indices out of ``d``."""
    return rng.choice(d, size=k, replace=False)


def _normal(rng: np.random.Generator, n: int, d: int) -> np.ndarray:
    """An independent standard-normal table."""
    return rng.standard_normal((n, d))


def _label(rng: np.random.Generator, score: np.ndarray) -> np.ndarray:
    """Balanced 0/1 labels: ``score`` above its median, ties broken by a negligible jitter."""
    s = score + 1e-9 * rng.standard_normal(len(score))
    return (s > np.median(s)).astype(int)


def _additive(k: int | None, noise: float = 0.0) -> Generator:
    """A linear score on ``k`` random columns (all when ``None``), plus ``noise`` signal sd."""

    def gen(rng: np.random.Generator, n: int, d: int) -> tuple[np.ndarray, np.ndarray]:
        X = _normal(rng, n, d)
        kk = d if k is None else k
        c, w = _cols(rng, d, kk), _signs(rng, kk)
        s = X[:, c] @ w
        s = s + noise * s.std() * rng.standard_normal(n)
        return X, _label(rng, s)

    return gen


def _pointwise(k: int, f: Callable[[list[np.ndarray]], np.ndarray]) -> Generator:
    """A score ``f`` of ``k`` random columns of an independent table, each sign-flipped."""

    def gen(rng: np.random.Generator, n: int, d: int) -> tuple[np.ndarray, np.ndarray]:
        X = _normal(rng, n, d)
        c = _cols(rng, d, k)
        flip = rng.choice([-1.0, 1.0], k)
        return X, _label(rng, f([flip[i] * X[:, c[i]] for i in range(k)]))

    return gen


def _interaction(rng: np.random.Generator, n: int, d: int) -> tuple[np.ndarray, np.ndarray]:
    """``x_a + x_b + 1.5 * x_a * x_b`` on two random columns (additive plus interaction)."""
    X = _normal(rng, n, d)
    a, b = _cols(rng, d, 2)
    return X, _label(rng, X[:, a] + X[:, b] + 1.5 * X[:, a] * X[:, b])


def _majority(rng: np.random.Generator, n: int, d: int) -> tuple[np.ndarray, np.ndarray]:
    """Sign-weighted vote of five random columns: the sum of their signs."""
    X = _normal(rng, n, d)
    c = _cols(rng, d, 5)
    return X, _label(rng, np.sign(X[:, c]) @ rng.choice([-1.0, 1.0], 5) + 0.01 * X[:, c[0]])


def _tree(rng: np.random.Generator, n: int, d: int) -> tuple[np.ndarray, np.ndarray]:
    """Depth-2 tree: the root column picks which of two leaf columns is compared."""
    X = _normal(rng, n, d)
    a, b, c = _cols(rng, d, 3)
    return X, _label(rng, np.where(X[:, a] > 0, X[:, b] - 0.5, X[:, c] + 0.5))


def _mixture(rng: np.random.Generator, n: int, d: int) -> tuple[np.ndarray, np.ndarray]:
    """Two Gaussian blobs, the class set the mean of three random columns."""
    y = rng.permutation(np.arange(n) % 2)
    X = _normal(rng, n, d)
    c = _cols(rng, d, 3)
    X[:, c] += np.outer(2 * y - 1, 0.9 * _signs(rng, 3))
    return X, y.astype(int)


def _latent(rng: np.random.Generator, n: int, d: int) -> tuple[np.ndarray, np.ndarray]:
    """Two latent factors drive every column; the label is the first factor's sign."""
    z = rng.standard_normal((n, 2))
    X = z @ rng.normal(size=(2, d)) + 0.5 * rng.standard_normal((n, d))
    return X, _label(rng, z[:, 0])


def _scm_chain(rng: np.random.Generator, n: int, d: int) -> tuple[np.ndarray, np.ndarray]:
    """A chain ``x0 -> x1 -> ... -> x_{d-1}``; the label reads a middle node plus noise."""
    X = np.empty((n, d))
    X[:, 0] = rng.standard_normal(n)
    for j in range(1, d):
        X[:, j] = rng.choice([-1.0, 1.0]) * 0.8 * X[:, j - 1] + 0.6 * rng.standard_normal(n)
    return X, _label(rng, X[:, d // 2] + 0.5 * rng.standard_normal(n))


def _scm_fork(rng: np.random.Generator, n: int, d: int) -> tuple[np.ndarray, np.ndarray]:
    """A hidden common cause drives every column and the label; it is not itself a column."""
    cause = rng.standard_normal(n)
    X = np.outer(cause, _signs(rng, d)) + 0.7 * rng.standard_normal((n, d))
    return X, _label(rng, cause + 0.5 * rng.standard_normal(n))


def _scm_anticausal(rng: np.random.Generator, n: int, d: int) -> tuple[np.ndarray, np.ndarray]:
    """Two causes set the label; one column is an effect of the label, not a cause of it."""
    X = _normal(rng, n, d)
    a, b, e = _cols(rng, d, 3)
    y = _label(rng, X[:, a] + X[:, b])
    X[:, e] = (2 * y - 1) * rng.choice([-1.0, 1.0]) + 0.8 * rng.standard_normal(n)
    return X, y


#: The task families, in a fixed order (the label of a family is its index here).
FAMILIES: dict[str, Generator] = {
    "linear": _additive(3),
    "sparse_linear": _additive(2),
    "dense_linear": _additive(None),
    "noisy_linear": _additive(3, noise=1.0),
    "threshold": _pointwise(1, lambda v: v[0]),
    "band": _pointwise(1, lambda v: -np.abs(v[0])),
    "and": _pointwise(2, lambda v: np.minimum(v[0], v[1])),
    "or": _pointwise(2, lambda v: np.maximum(v[0], v[1])),
    "xor": _pointwise(2, lambda v: v[0] * v[1]),
    "parity3": _pointwise(3, lambda v: v[0] * v[1] * v[2]),
    "four_way": _pointwise(4, lambda v: v[0] * v[1] * v[2] * v[3]),
    "interaction": _interaction,
    "radial": _pointwise(2, lambda v: -(v[0] ** 2 + v[1] ** 2)),
    "saddle": _pointwise(2, lambda v: v[0] ** 2 - v[1] ** 2),
    "sine": _pointwise(2, lambda v: np.sin(3 * v[0]) + 0.3 * v[1]),
    "checkerboard": _pointwise(2, lambda v: np.sin(2 * v[0]) * np.sin(2 * v[1])),
    "majority": _majority,
    "tree": _tree,
    "ratio": _pointwise(2, lambda v: v[0] / (np.abs(v[1]) + 0.5)),
    "mixture": _mixture,
    "latent_factor": _latent,
    "scm_chain": _scm_chain,
    "scm_fork": _scm_fork,
    "scm_anticausal": _scm_anticausal,
}


def make_task(family: str, seed: int, n: int, d: int) -> tuple[np.ndarray, np.ndarray]:
    """Draw one task of ``family``: a table and its balanced binary labels.

    Args:
        family: A key of :data:`FAMILIES`.
        seed: Task seed; fixes the family's parameters and the rows. The same
            ``(family, seed)`` always returns the same task, and two families at one seed
            do not share a stream.
        n: Rows, at least 8.
        d: Columns, at least :data:`MIN_WIDTH`.

    Returns:
        ``(X, y)`` with ``X`` of shape ``(n, d)`` and ``y`` of 0/1 integers.

    Raises:
        ValueError: On an unknown family, ``d`` below :data:`MIN_WIDTH`, or ``n`` below 8.
    """
    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}; choose from {sorted(FAMILIES)}")
    if d < MIN_WIDTH:
        raise ValueError(f"d must be at least {MIN_WIDTH}")
    if n < 8:
        raise ValueError("n must be at least 8")
    rng = np.random.default_rng([seed, list(FAMILIES).index(family), 397])
    return FAMILIES[family](rng, n, d)


def probe_rows(n_probe: int, d: int) -> np.ndarray:
    """The fixed rows every task is represented on: independent standard normal.

    Args:
        n_probe: Number of rows.
        d: Columns.

    Returns:
        ``(n_probe, d)`` array, identical on every call.
    """
    return np.random.default_rng(PROBE_SEED).standard_normal((n_probe, d))


def task_representation(
    model: FinancialTFM,
    X: np.ndarray,
    y: np.ndarray,
    probe: np.ndarray,
    stage: str = "pre_head",
    seed: int = 0,
) -> np.ndarray:
    """The model's representation of the task defined by a context.

    The context ``(X, y)`` is handed to a classifier built on ``model``; the representation of
    the fixed ``probe`` rows under that context is mean- and std-pooled over rows.

    Args:
        model: Any :class:`~fintfm.modeling.model.FinancialTFM`, trained or not.
        X: ``(n, d)`` context features.
        y: ``(n,)`` binary context labels.
        probe: ``(m, d)`` rows to read the representation on, the same for every task.
        stage: ``transform_representation`` stage.
        seed: Classifier random state (column identities and context draw).

    Returns:
        A vector of ``2 * d_model`` floats, whatever ``n`` and ``m`` are.
    """
    from fintfm.inference.classifier import FinancialTFMClassifier

    clf = FinancialTFMClassifier(model, max_context=len(X), random_state=seed)
    clf.fit(X, y)
    h = clf.transform_representation(probe, stage=stage)  # type: ignore[arg-type]
    return np.concatenate([h.mean(axis=0), h.std(axis=0)]).astype(np.float64)


def raw_statistics(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Label-conditional statistics of a context table; the no-model baseline's features.

    Sorted (so column order carries nothing) absolute correlations of the label with each
    column, its square and its absolute value; the six strongest absolute correlations of the
    label with a pairwise column product; the mean absolute between-column correlation; and the
    label rate. The length depends only on the width ``d``.

    Args:
        X: ``(n, d)`` features.
        y: ``(n,)`` labels.

    Returns:
        A 1-D feature vector.
    """

    def corr(M: np.ndarray) -> np.ndarray:
        Mc = M - M.mean(axis=0)
        yc = y - y.mean()
        den = np.sqrt((Mc**2).sum(axis=0) * (yc**2).sum()) + 1e-12
        return np.abs(Mc.T @ yc / den)

    d = X.shape[1]
    iu, ju = np.triu_indices(d, 1)
    pair = corr(X[:, iu] * X[:, ju])
    cc = np.abs(np.corrcoef(X, rowvar=False)[iu, ju])
    return np.concatenate(
        [
            np.sort(corr(X))[::-1],
            np.sort(corr(X**2))[::-1],
            np.sort(corr(np.abs(X)))[::-1],
            np.sort(pair)[::-1][:6],
            [np.nan_to_num(cc).mean(), float(y.mean())],
        ]
    )


def cross_validated_probe(
    F: np.ndarray,
    labels: np.ndarray,
    folds: int = 5,
    seed: int = 0,
    n_permutations: int = 20,
    C: float = 0.1,
) -> dict:
    """Cross-validated logistic-regression probe from features to family, with a permutation floor.

    Args:
        F: ``(n_tasks, n_features)`` task representations.
        labels: ``(n_tasks,)`` integer family labels.
        folds: Stratified folds; every fold's tasks are unseen by that fold's probe.
        seed: Fold shuffling and permutation seed.
        n_permutations: Shuffled-label repeats for the permutation chance.
        C: Inverse L2 strength; small, because the vector is wide against few tasks.

    Returns:
        Accuracy, macro F1, per-class recall, the confusion matrix (rows true, columns
        predicted), the permutation chance (mean and 95th percentile) and the per-fold accuracy.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix, f1_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    k = int(labels.max()) + 1
    F = np.nan_to_num(F)

    def predict(lab: np.ndarray) -> tuple[np.ndarray, list[float]]:
        pred = np.empty_like(lab)
        accs = []
        skf = StratifiedKFold(folds, shuffle=True, random_state=seed)
        for tr, te in skf.split(F, lab):
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))
            clf.fit(F[tr], lab[tr])
            pred[te] = clf.predict(F[te])
            accs.append(float(np.mean(pred[te] == lab[te])))
        return pred, accs

    pred, fold_acc = predict(labels)
    cm = confusion_matrix(labels, pred, labels=range(k))
    rng = np.random.default_rng(seed)
    perm = [
        float(np.mean(predict(shuffled)[0] == shuffled))
        for shuffled in (rng.permutation(labels) for _ in range(n_permutations))
    ]
    return {
        "accuracy": float(np.mean(pred == labels)),
        "macro_f1": float(f1_score(labels, pred, average="macro")),
        "recall": (np.diag(cm) / np.maximum(cm.sum(axis=1), 1)).tolist(),
        "confusion": cm.tolist(),
        "permutation_accuracy_mean": float(np.mean(perm)) if perm else None,
        "permutation_accuracy_p95": float(np.percentile(perm, 95)) if perm else None,
        "fold_accuracy": fold_acc,
    }


def run(
    model_paths: dict[str, str],
    out_dir: Path | None = None,
    families: tuple[str, ...] | None = None,
    n_tasks: int = 20,
    n_rows: int = 256,
    d: int = 8,
    n_probe: int = 64,
    stage: str = "pre_head",
    folds: int = 5,
    seed: int = 0,
    n_permutations: int = 20,
    models: dict[str, FinancialTFM] | None = None,
) -> dict:
    """Generate the tasks, represent them under every arm, and fit the family probe.

    Args:
        model_paths: ``{name: checkpoint path}``; one untrained control is built from the
            first one's architecture. Ignored when ``models`` is given.
        out_dir: If given, ``task_probe.json`` is written there.
        families: Subset of :data:`FAMILIES` (default all).
        n_tasks: Task instances per family; must be at least ``folds``.
        n_rows: Context rows per task.
        d: Table width; at most every model's ``max_features``.
        n_probe: Fixed probe rows the representation is read on.
        stage: ``transform_representation`` stage.
        folds: Cross-validation folds.
        seed: Base seed for task instances and the probe.
        n_permutations: Shuffled-label repeats per arm.
        models: Already-built models by name (used by tests); the untrained control is added.

    Returns:
        The record: config, and per arm the probe result.

    Raises:
        ValueError: If ``n_tasks < folds`` or ``d`` exceeds a model's ``max_features``.
    """
    import torch

    from fintfm.modeling.model import FinancialTFM

    names = tuple(FAMILIES) if families is None else families
    if n_tasks < folds:
        raise ValueError("n_tasks must be at least folds")
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
            raise ValueError(f"d {d} exceeds {name}'s max_features {m.cfg.max_features}")

    tasks = [
        (fi, make_task(f, seed * 1000 + t, n_rows, d))
        for fi, f in enumerate(names)
        for t in range(n_tasks)
    ]
    labels = np.array([fi for fi, _ in tasks])
    probe = probe_rows(n_probe, d)
    feats: dict[str, np.ndarray] = {
        "raw_statistics": np.stack([raw_statistics(X, y) for _, (X, y) in tasks])
    }
    for name, m in loaded.items():
        feats[name] = np.stack(
            [task_representation(m, X, y, probe, stage, seed) for _, (X, y) in tasks]
        )
        print(f"  represented {len(tasks)} tasks under {name}", flush=True)
    results = {
        arm: cross_validated_probe(F, labels, folds, seed, n_permutations)
        for arm, F in feats.items()
    }
    record = {
        "task": "predict the generating task family from the model's context representation",
        "label": "SMOKE-TEST unless run on a real checkpoint with this command recorded",
        "config": {
            "models": model_paths,
            "families": list(names),
            "n_tasks": n_tasks,
            "n_rows": n_rows,
            "d": d,
            "n_probe": n_probe,
            "stage": stage,
            "folds": folds,
            "seed": seed,
            "n_permutations": n_permutations,
        },
        "chance": 1 / len(names),
        "feature_dim": {arm: int(F.shape[1]) for arm, F in feats.items()},
        "results": results,
    }
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "task_probe.json").write_text(json.dumps(record, indent=2))
    return record


def summarise(record: dict) -> str:
    """Printable accuracy table, weakest families and top confusions per arm.

    Args:
        record: Output of :func:`run`.

    Returns:
        A multi-line string.
    """
    names = record["config"]["families"]
    lines = [
        f"{'arm':>22} {'accuracy':>9} {'macro F1':>9} {'perm mean':>10} {'perm p95':>9}",
        f"{'chance (1/K)':>22} {record['chance']:>9.3f}",
    ]
    for arm, r in record["results"].items():
        pm, p95 = r["permutation_accuracy_mean"], r["permutation_accuracy_p95"]
        perm = f"{pm:>10.3f} {p95:>9.3f}" if pm is not None else f"{'-':>10} {'-':>9}"
        lines.append(f"{arm:>22} {r['accuracy']:>9.3f} {r['macro_f1']:>9.3f} {perm}")
    for arm, r in record["results"].items():
        cm = np.array(r["confusion"])
        rec = np.array(r["recall"])
        worst = np.argsort(rec)[:5]
        lines += [
            "",
            f"{arm}: lowest recall: " + ", ".join(f"{names[i]} {rec[i]:.2f}" for i in worst),
        ]
        off = [(cm[i, j], i, j) for i in range(len(cm)) for j in range(len(cm)) if i != j]
        top = sorted(off, reverse=True)[:5]
        lines.append(
            f"{arm}: top confusions (true -> predicted): "
            + ", ".join(f"{names[i]}->{names[j]} {c}" for c, i, j in top if c)
        )
    lines += [
        "",
        "Reading it: accuracy is on held-out task instances, cross-validated. Beat chance and the",
        "permutation p95 to claim any separation; beat raw_statistics to claim the model's state",
        "carries more than the context's label-feature statistics; beat untrained_control to claim",
        "training put the separation there.",
    ]
    return "\n".join(lines)


def main() -> None:
    """Command-line entry point: ``python -m fintfm.experiments.task_probe``."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", type=str, required=True, help="comma-separated name=path pairs")
    p.add_argument("--out", type=str, default="runs/task_probe")
    p.add_argument("--n-tasks", type=int, default=20)
    p.add_argument("--n-rows", type=int, default=256)
    p.add_argument("--d", type=int, default=8)
    p.add_argument("--n-probe", type=int, default=64)
    p.add_argument("--stage", type=str, default="pre_head")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--permutations", type=int, default=20)
    args = p.parse_args()
    paths = dict(pair.split("=", 1) for pair in args.models.split(",") if pair)
    record = run(
        paths,
        Path(args.out),
        n_tasks=args.n_tasks,
        n_rows=args.n_rows,
        d=args.d,
        n_probe=args.n_probe,
        stage=args.stage,
        folds=args.folds,
        seed=args.seed,
        n_permutations=args.permutations,
    )
    print("\n" + summarise(record))


if __name__ == "__main__":
    main()
