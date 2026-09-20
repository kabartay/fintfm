"""Can the model learn in context at all? The control suite §42 should have had from the start.

Why this exists
---------------
Every accuracy claim in this project was measured on real credit panels, where a strong
baseline already scores well and the interesting quantity is a small delta. That hid a
fundamental defect for two days: ``docs/FINDINGS.md`` §42 found the model scoring **0.68 on a
clean linear task that logistic regression solves at 0.9997**, not improving with context size,
and beating its own randomly-initialised weights by **0.015**.

Two measurement failures let that happen, and both are addressed here rather than in a
docstring:

**The benchmark was too easy to be diagnostic.** V4FinBench horizon 0 is nearly solved by one
column — ``Working_capital/total_assets`` alone reaches 0.9799 AUC — so 0.9811 looked like a
result and was 0.0012 above reading a single feature. Synthetic probes with a *known* ceiling
cannot flatter a model that way.

**There was no floor.** Without an untrained control, "0.66 on prior tasks" is unreadable: it
could be competence on a hard task or the architecture plus the context doing the work. The
untrained arm is not decoration; it is the only thing that makes the trained number mean
anything.

What the probes are
-------------------
Each is a synthetic in-context task with a known achievable ceiling, so the gap to a fitted
baseline is interpretable rather than relative:

- ``linear`` — a weighted sum. Logistic regression reaches ~1.0, so anything below that is the
  model's own limitation, not the task's.
- ``conjunction`` — an AND of three axis-aligned thresholds, which trees represent natively.
- ``xor`` — a parity interaction no additive model can express at all, where logistic
  regression is pinned at chance by construction and only an interaction-capable model scores.
- ``noise`` — labels independent of the features. **Every arm must score ~0.5.** An arm that
  beats chance here is leaking, and this probe exists to catch that rather than to rank models.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

#: Probe names, in the order they are reported.
PROBES: tuple[str, ...] = (
    "orientation", "linear", "conjunction", "xor", "noise",
    "symmetric_sum", "symmetric_count", "antisymmetric",
)

#: Probes whose rule is a **symmetric function of the row's feature values**, and so is
#: learnable even by an encoder that cannot tell its own columns apart. Paired with
#: ``antisymmetric``, they are the sharpest diagnostic in this repository: the gap between
#: them is exactly the §54 failure, and it is invisible in every aggregate score.
SYMMETRIC_PROBES: tuple[str, ...] = ("symmetric_sum", "symmetric_count")

#: Fixed seed for the untrained control's weights, so the floor is the same number every run.
UNTRAINED_SEED = 20260910


@dataclass
class ProbeResult:
    """One arm on one probe, averaged over seeds."""

    probe: str
    arm: str
    auc_mean: float
    auc_std: float
    n_seeds: int


def make_probe(
    kind: str, n: int = 8000, n_features: int = 20, rate: float = 0.05, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build one synthetic in-context task with a known ceiling.

    Args:
        kind: One of :data:`PROBES`.
        n: Total rows, split evenly into context and query.
        n_features: Feature width.
        rate: Positive-class rate.
        seed: Random seed.

    Returns:
        ``(X_train, y_train, X_test, y_test)``.

    Raises:
        ValueError: If ``kind`` is not a known probe.
    """
    if kind not in PROBES:
        raise ValueError(f"unknown probe {kind!r}; expected one of {PROBES}")
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_features)).astype(np.float32)
    if kind == "orientation":
        # The minimal context-necessary task: one informative feature, and a task-level sign
        # drawn per task. Marginally P(y=1 | x) = 0.5, so **no global predictor can beat
        # chance** — while a single labelled example reveals the orientation and makes the
        # task perfectly solvable. AUC 0.5 means the context was ignored; AUC near 1.0 means
        # it was used. Nothing else in the suite separates those two as cleanly.
        #
        # One dimension on purpose: §49 measured performance falling with feature count, so a
        # multi-feature control would confound "cannot use context" with "cannot aggregate".
        sign = rng.choice([-1.0, 1.0])
        score = sign * X[:, 0]
    elif kind == "linear":
        score = X @ rng.normal(size=n_features)
    elif kind == "conjunction":
        # three thresholds ANDed; each cut at rate^(1/3) so the conjunction lands near `rate`
        cut = np.quantile(X[:, :3], rate ** (1 / 3), axis=0)
        score = np.all(X[:, :3] < cut, axis=1) * 3.0 + rng.normal(0, 0.3, size=n)
    elif kind == "xor":
        # Parity of two features: no additive function of the inputs separates it.
        #
        # The magnitude term matters. A bare `sign(x0)*sign(x1) + noise` looks like parity but
        # is not learnable: the signal is ±1, so taking the top `rate` selects the *noise*
        # tail inside the positive quadrant and even a tree is left near chance. Scaling by
        # the distance from both axes makes the label "in a parity quadrant, and far from
        # both boundaries", which an interaction-capable model can actually find.
        parity = np.sign(X[:, 0]) * np.sign(X[:, 1])
        score = parity * np.minimum(np.abs(X[:, 0]), np.abs(X[:, 1]))
    elif kind == "symmetric_sum":
        # A symmetric function of the row: permuting a row's values leaves the label alone.
        # Learnable *without* any notion of column identity, so this is the control that says
        # whether a disappointing `antisymmetric` score is about columns or about something
        # else entirely.
        score = X.sum(axis=1)
    elif kind == "symmetric_count":
        # Symmetric but **nonlinear** in the feature values. Measured at 0.9232 where the
        # linear symmetric rule reached 0.9996 (§56), and barely moved when column identities
        # were added -- so it is the leading candidate for a second, independent bottleneck.
        score = (X > 0).sum(axis=1).astype(np.float64)
    elif kind == "antisymmetric":
        # x_0 - x_1. The label is **independent of the multiset** of the row's values, so the
        # best AUC achievable by any symmetric function of the row is exactly 0.5. That makes
        # this the one probe with a provable ceiling for the broken architecture, and the
        # measurement that settled §54: 0.5007 before column identities, 0.9995 after.
        score = X[:, 0] - X[:, 1]
    else:  # noise
        score = rng.normal(size=n)
    y = (score >= np.quantile(score, 1 - rate)).astype(np.int64)
    k = n // 2
    return X[:k], y[:k], X[k:], y[k:]


#: Bayes-optimal AUC targets for :func:`bayes_ceiling_probe`. Spans chance to near-certainty.
BAYES_AUC_TARGETS: tuple[float, ...] = (0.500, 0.550, 0.600, 0.700, 0.800, 0.900, 0.950, 0.990, 0.999)

#: Feature width for the ceiling task: one informative dimension, five inert companions,
#: matching this project's other symmetry probes.
_BAYES_TASK_DIM = 6


def _bayes_optimal_mu(target_auc: float) -> float:
    """The mean shift giving a 1-D two-Gaussian task exactly ``target_auc`` Bayes-optimal AUC.

    For class 0 ~ N(0,1) and class 1 ~ N(mu,1), the Bayes-optimal classifier thresholds the
    one informative dimension and its AUC has the closed form ``Phi(mu / sqrt(2))`` --
    standard signal-detection theory (d'/sqrt(2)). Solving for ``mu`` lets a task's true
    difficulty be dialled exactly, with nothing left to estimate. Verified against this
    formula by :func:`fintfm.experiments.capability` tests, and originally against an
    empirical Bayes-optimal-statistic AUC before this was trusted for a real measurement.
    """
    from scipy.stats import norm

    return float(np.sqrt(2.0) * norm.ppf(target_auc))


def _make_bayes_task(
    rng: np.random.Generator, n: int, mu: float, d: int = _BAYES_TASK_DIM
) -> tuple[np.ndarray, np.ndarray]:
    y = (rng.random(n) < 0.5).astype(np.int64)
    X = rng.normal(size=(n, d))
    X[:, 0] += mu * y  # mean shift only in the informative dimension, only for class 1
    return X.astype(np.float32), y


def bayes_ceiling_probe(
    model,
    targets: tuple[float, ...] = BAYES_AUC_TARGETS,
    seeds: int = 10,
    n: int = 1600,
    n_ensemble: int = 8,
) -> dict[float, tuple[float, float]]:
    """Achieved AUC against an *exactly known* Bayes-optimal AUC, not an estimated one.

    ``docs/FINDINGS.md`` §74: the decisive test of whether a capped predictor (§51, §53) is
    an architecture/capacity bottleneck or a prior-content effect. Found that training
    predominantly on the financial prior caps achieved AUC at ~0.73 regardless of how strong
    the true signal is (0.728 achieved at Bayes AUC 0.999), while the identical architecture
    trained on the generic SCM prior tracks the true curve almost exactly (0.997 at 0.999).
    §76 bisected four prior-content candidates and none closed the gap --
    `openspec/changes/cell-attention-and-task-inference` is the architecture-side experiment
    this probe exists to score.

    Uses the model directly (not :class:`~fintfm.inference.classifier.FinancialTFMClassifier`)
    with the context split via ``n_ctx`` and column-identity draws averaged over
    ``n_ensemble``, matching exactly how §74's original measurement was taken -- so numbers
    from this function are comparable to every value already recorded in ``docs/FINDINGS.md``.

    Args:
        model: A loaded :class:`~fintfm.modeling.model.FinancialTFM`, trained for
            classification with ``max_classes >= 2``.
        targets: Bayes-optimal AUCs to test. Default spans chance to near-certainty.
        seeds: Independent task draws averaged per target.
        n: Rows per task; half context, half query.
        n_ensemble: Column-identity draws averaged per prediction (D12).

    Returns:
        ``{target: (achieved_auc_mean, regret)}`` where ``regret = target - achieved_mean``
        (task 39.18) -- the number that makes "0.70 achieved" legible as excellent at a 0.71
        ceiling and terrible at a 0.995 one, which the raw AUC alone does not.
    """
    import torch
    from sklearn.metrics import roc_auc_score

    nc = n // 2
    out: dict[float, tuple[float, float]] = {}
    for target in targets:
        mu = _bayes_optimal_mu(target)
        achieved = []
        for s in range(seeds):
            rng = np.random.default_rng(s)
            X, y = _make_bayes_task(rng, n, mu)
            Xt = torch.tensor(X)[None]
            yt = torch.tensor(y)[None]
            with torch.no_grad():
                ps = [
                    torch.softmax(
                        model(Xt, yt, nc, torch.tensor([2]), column_id_seed=k).float(), -1
                    )[0, nc:, 1].numpy()
                    for k in range(n_ensemble)
                ]
            achieved.append(roc_auc_score(y[nc:], np.mean(ps, 0)))
        mean_achieved = float(np.mean(achieved))
        out[target] = (mean_achieved, target - mean_achieved)
    return out


def feature_sweep(
    model_paths: dict[str, str],
    widths: tuple[int, ...] = (5, 10, 20, 40, 80, 130),
    seeds: tuple[int, ...] = (0, 1, 2),
    max_context: int = 1000,
) -> dict:
    """How does each model hold up as the number of features grows?

    ``docs/FINDINGS.md`` §49: the model reaches 0.846 on a five-feature linear task and
    collapses to 0.551 at eighty, while logistic regression is flat at ~0.999 across the whole
    range. Both widths are **inside** the prior's training distribution — it produces a median
    of 77 columns with half of all tasks at 80 or more — so this is an aggregation failure
    rather than extrapolation.

    That points at the row representation: the column stage emits one token per feature and
    ``encode_rows`` reduces them by masked mean and max, which is exactly the operation that
    should struggle to preserve 130 weighted contributions.

    The sweep exists to settle capacity against design. **If larger models degrade less
    steeply, the constraint is capacity; if they degrade identically, it is the pooling.**

    Args:
        model_paths: ``{arm name: checkpoint path}``.
        widths: Feature counts to sweep.
        seeds: Seeds per cell.
        max_context: Context rows for the in-context arms.

    Returns:
        ``{"widths": [...], "arms": {name: [auc per width]}}``.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    from fintfm.inference.classifier import FinancialTFMClassifier
    from fintfm.modeling.model import FinancialTFM

    models = {n: FinancialTFM.load(p) for n, p in model_paths.items()}
    arms: dict[str, list[float]] = {n: [] for n in models}
    arms["logistic_regression"] = []
    for width in widths:
        per_arm: dict[str, list[float]] = {n: [] for n in arms}
        for seed in seeds:
            Xtr, ytr, Xte, yte = make_probe("linear", n_features=width, seed=seed)
            if len(np.unique(yte)) < 2:
                continue
            for name, m in models.items():
                clf = FinancialTFMClassifier(
                    m, max_context=max_context, context_strategy="uniform",
                    feature_transform="rank", random_state=seed,
                ).fit(Xtr, ytr)
                per_arm[name].append(roc_auc_score(yte, clf.predict_proba(Xte)[:, 1]))
            lr = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
            per_arm["logistic_regression"].append(
                roc_auc_score(yte, lr.predict_proba(Xte)[:, 1])
            )
        for name, vals in per_arm.items():
            arms[name].append(float(np.mean(vals)) if vals else float("nan"))
        print(f"  width {width} done", flush=True)
    return {"widths": list(widths), "arms": arms}


def summarise_sweep(sweep: dict) -> str:
    """Render the width sweep, with each arm's degradation from narrowest to widest."""
    widths = sweep["widths"]
    lines = [f"{'arm':>22} " + " ".join(f"{f'F={w}':>8}" for w in widths) + f"{'drop':>9}"]
    for name, vals in sweep["arms"].items():
        drop = vals[-1] - vals[0] if len(vals) > 1 else float("nan")
        lines.append(f"{name:>22} " + " ".join(f"{v:>8.4f}" for v in vals) + f"{drop:>+9.4f}")
    lines += [
        "",
        "`drop` is widest minus narrowest. A flat arm aggregates features; a steeply negative",
        "one does not. If larger models drop less, the constraint is capacity; if they drop",
        "the same, it is the pooling design (docs/FINDINGS.md §49).",
    ]
    return "\n".join(lines)


def base_rate_sweep(
    model_paths: dict[str, str],
    rates: tuple[float, ...] = (0.05, 0.15, 0.30, 0.50),
    n_features: int = 5,
    seeds: tuple[int, ...] = (0, 1, 2),
    max_context: int = 2000,
) -> dict:
    """How does each model hold up as the task becomes balanced?

    ``docs/FINDINGS.md`` §51: performance falls monotonically as the base rate rises — 0.850
    at 5% down to 0.658 at 50% — while a linear baseline holds 1.0000 throughout. That is
    backwards on its face, since more positives means more information about the positive
    class.

    The reading is that the model **detects extremes rather than ordering**. At a low base
    rate, ranking well largely means finding the tail, and max pooling is an extremeness
    detector. At 50% it means ordering the whole distribution, which a mean-and-max reduction
    cannot do.

    That predicts attention pooling should help **most at high base rates**. This sweep is how
    that prediction is checked — and if the gain is flat across rates, the reading is wrong
    and should be discarded rather than adjusted.

    Feature count is held low (5 by default) so the aggregation failure of §50 does not
    confound the base-rate effect.

    Args:
        model_paths: ``{arm name: checkpoint path}``.
        rates: Positive-class rates to sweep.
        n_features: Feature width, kept small deliberately.
        seeds: Seeds per cell.
        max_context: Context rows for the in-context arms.

    Returns:
        ``{"rates": [...], "arms": {name: [auc per rate]}}``.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    from fintfm.inference.classifier import FinancialTFMClassifier
    from fintfm.modeling.model import FinancialTFM

    models = {n: FinancialTFM.load(p) for n, p in model_paths.items()}
    arms: dict[str, list[float]] = {n: [] for n in models}
    arms["logistic_regression"] = []
    for rate in rates:
        per_arm: dict[str, list[float]] = {n: [] for n in arms}
        for seed in seeds:
            Xtr, ytr, Xte, yte = make_probe(
                "linear", n=16000, n_features=n_features, rate=rate, seed=seed
            )
            if len(np.unique(yte)) < 2:
                continue
            for name, m in models.items():
                clf = FinancialTFMClassifier(
                    m, max_context=max_context, context_strategy="uniform",
                    feature_transform="rank", random_state=seed,
                ).fit(Xtr, ytr)
                per_arm[name].append(roc_auc_score(yte, clf.predict_proba(Xte)[:, 1]))
            lr = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
            per_arm["logistic_regression"].append(
                roc_auc_score(yte, lr.predict_proba(Xte)[:, 1])
            )
        for name, vals in per_arm.items():
            arms[name].append(float(np.mean(vals)) if vals else float("nan"))
        print(f"  rate {rate:.0%} done", flush=True)
    return {"rates": list(rates), "arms": arms}


def summarise_rate_sweep(sweep: dict) -> str:
    """Render the base-rate sweep, with each arm's decline from rare to balanced."""
    rates = sweep["rates"]
    lines = [f"{'arm':>22} " + " ".join(f"{r:>8.0%}" for r in rates) + f"{'drop':>9}"]
    for name, vals in sweep["arms"].items():
        drop = vals[-1] - vals[0] if len(vals) > 1 else float("nan")
        lines.append(f"{name:>22} " + " ".join(f"{v:>8.4f}" for v in vals) + f"{drop:>+9.4f}")
    lines += [
        "",
        "`drop` is balanced minus rare. A model that orders the distribution is flat; one that",
        "only detects extremes falls as the task balances (docs/FINDINGS.md §51).",
    ]
    return "\n".join(lines)


def make_multiclass_probe(
    n: int = 8000, n_features: int = 20, n_classes: int = 3, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a ``n_classes``-way task whose ceiling a multinomial logit can reach.

    ``y = argmax_k (X @ W)_k`` for a per-task random ``W``. Two properties make this the
    right control for :func:`multiclass_sweep` rather than a harder, more interesting task:

    - **The ceiling is known and reachable.** Multinomial logistic regression is the correct
      model for this generative story, so it scores near the ceiling and any shortfall is the
      in-context model's own, not the task's. That is the same discipline ``linear`` enforces
      for the binary suite.
    - **Classes are near-balanced**, so accuracy stays legible as ``K`` grows. They are not
      *exactly* balanced — a random ``W`` gives unequal argmax regions, 4.4% to 15.3% at
      ``K=10`` — so :func:`multiclass_sweep` reports the measured majority-class rate as the
      floor rather than ``1/K``, which would understate it. A rare-class variant would
      confound "cannot do multiclass" with "cannot find a rare class", which the binary suite
      already measures separately (:func:`base_rate_sweep`).

    Args:
        n: Total rows, split evenly into context and query.
        n_features: Feature width.
        n_classes: Number of classes.
        seed: Random seed.

    Returns:
        ``(X_train, y_train, X_test, y_test)``.

    Raises:
        ValueError: If ``n_classes`` is below two.
    """
    if n_classes < 2:
        raise ValueError(f"n_classes must be >= 2, got {n_classes}")
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_features)).astype(np.float32)
    W = rng.normal(size=(n_features, n_classes))
    y = (X @ W).argmax(axis=1).astype(np.int64)
    k = n // 2
    return X[:k], y[:k], X[k:], y[k:]


def multiclass_sweep(
    model_paths: dict[str, str],
    class_counts: tuple[int, ...] = (3, 5, 10),
    n_features: int = 20,
    seeds: tuple[int, ...] = (0, 1, 2),
    max_context: int = 1000,
) -> dict:
    """Does the model classify into more than two classes at all?

    Task 46.1's verification instrument. Every accuracy number this project has published is
    binary, so "the checkpoint was trained at ``--max-classes 10``" is a statement about a
    command line, not about a capability. This measures the capability.

    Reported per class count:

    - **accuracy**, against the measured majority-class rate, which shrinks as ``K`` grows;
    - **macro one-vs-rest AUC**, whose chance floor is 0.5 at every ``K``, so a decline across
      the sweep cannot be confused with the floor moving.

    Both are reported because either alone is misleading here. Accuracy falling from ``K=3``
    to ``K=10`` is expected even for a perfect model's *relative* margin over chance; macro
    AUC holding flat while accuracy falls is the signature of a model that ranks classes
    correctly but is miscalibrated across them, which is a different defect with a different
    fix.

    Args:
        model_paths: ``{arm name: checkpoint path}``. An untrained control of the first
            architecture is added automatically — without it these numbers are unreadable.
        class_counts: Class counts to sweep.
        n_features: Probe feature width.
        seeds: Seeds per cell.
        max_context: Context rows for the in-context arms.

    Returns:
        ``{"class_counts": [...], "majority_rate": [...], "arms": {name: {"accuracy":
        [...], "macro_auc": [...]}}}``.
    """
    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score

    from fintfm.inference.classifier import FinancialTFMClassifier
    from fintfm.modeling.model import FinancialTFM

    models = {n: FinancialTFM.load(p) for n, p in model_paths.items()}
    if models:
        first = next(iter(models.values()))
        torch.manual_seed(UNTRAINED_SEED)
        models["untrained_control"] = FinancialTFM(first.cfg)

    arm_names = [*models, "logistic_regression"]
    arms: dict[str, dict[str, list[float]]] = {
        n: {"accuracy": [], "macro_auc": []} for n in arm_names
    }
    skipped: list[str] = []
    majority: list[float] = []
    for n_classes in class_counts:
        per_seed_majority: list[float] = []
        cell: dict[str, dict[str, list[float]]] = {
            n: {"accuracy": [], "macro_auc": []} for n in arm_names
        }
        for seed in seeds:
            Xtr, ytr, Xte, yte = make_multiclass_probe(
                n_features=n_features, n_classes=n_classes, seed=seed
            )
            if len(np.unique(ytr)) < n_classes or len(np.unique(yte)) < n_classes:
                continue
            per_seed_majority.append(float(np.bincount(yte).max() / len(yte)))
            for name, m in models.items():
                if m.cfg.max_classes < n_classes:
                    # A binary checkpoint cannot represent this task. Recording a number for
                    # it anyway would invite a comparison the architecture forbids, so the
                    # cell is left empty and the reason is carried into the record.
                    skipped.append(f"{name}@K={n_classes}: max_classes={m.cfg.max_classes}")
                    continue
                clf = FinancialTFMClassifier(
                    m, max_context=max_context, context_strategy="uniform",
                    feature_transform="rank", random_state=seed,
                ).fit(Xtr, ytr)
                proba = clf.predict_proba(Xte)
                cell[name]["accuracy"].append(
                    accuracy_score(yte, clf.classes_[proba.argmax(axis=1)])
                )
                cell[name]["macro_auc"].append(
                    roc_auc_score(yte, proba, multi_class="ovr", average="macro")
                )
            lr = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
            lp = lr.predict_proba(Xte)
            cell["logistic_regression"]["accuracy"].append(accuracy_score(yte, lr.predict(Xte)))
            cell["logistic_regression"]["macro_auc"].append(
                roc_auc_score(yte, lp, multi_class="ovr", average="macro")
            )
        for name, metrics in cell.items():
            for metric, vals in metrics.items():
                arms[name][metric].append(float(np.mean(vals)) if vals else float("nan"))
        majority.append(
            float(np.mean(per_seed_majority)) if per_seed_majority else float("nan")
        )
        print(f"  K={n_classes} done", flush=True)
    return {
        "class_counts": list(class_counts),
        "majority_rate": majority,
        "arms": arms,
        "skipped": skipped,
        "config": {
            "models": model_paths, "seeds": list(seeds),
            "max_context": max_context, "n_features": n_features,
        },
    }


def summarise_multiclass_sweep(sweep: dict) -> str:
    """Render the multiclass sweep, with the floor on its own row."""
    ks = sweep["class_counts"]
    lines = []
    floors = (
        ("accuracy", "majority class", sweep["majority_rate"]),
        ("macro_auc", "chance", [0.5] * len(ks)),
    )
    for metric, floor_name, floor in floors:
        lines.append(f"{metric}")
        lines.append(f"{'arm':>22} " + " ".join(f"{'K=' + str(k):>9}" for k in ks))
        lines.append(f"{floor_name:>22} " + " ".join(f"{f:>9.4f}" for f in floor))
        for name, metrics in sweep["arms"].items():
            vals = metrics[metric]
            lines.append(f"{name:>22} " + " ".join(f"{v:>9.4f}" for v in vals))
        lines.append("")
    for note in sweep.get("skipped", []):
        lines.append(f"skipped {note}")
    lines += [
        "",
        "Accuracy's floor is the majority class and moves across the sweep; macro one-vs-rest",
        "AUC's floor is 0.5 at every K. Read them together: the claim 'the model does",
        "multiclass' requires clearing the untrained control of the same architecture, not",
        "clearing the floor.",
    ]
    return "\n".join(lines)


def run(
    model_paths: dict[str, str],
    out_dir: Path,
    seeds: tuple[int, ...] = (0, 1, 2),
    max_context: int = 1000,
    n_features: int = 20,
) -> dict:
    """Score every checkpoint, an untrained control and fitted baselines on every probe.

    Args:
        model_paths: ``{arm name: checkpoint path}``.
        out_dir: Directory for ``capability.json``.
        seeds: Seeds per cell.
        max_context: Context rows handed to the in-context arms.
        n_features: Probe feature width.

    Returns:
        The recorded result dictionary.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    from fintfm.inference.classifier import FinancialTFMClassifier
    from fintfm.modeling.model import FinancialTFM

    models = {name: FinancialTFM.load(p) for name, p in model_paths.items()}
    if models:
        # The floor: same architecture, random weights. Without it a trained number is
        # unreadable, which is how §42 went unnoticed for two days.
        #
        # **Seeded.** Unseeded, this control scored 0.344, 0.357, 0.367 and 0.569 on `linear`
        # across four invocations — a floor that moves by 0.22 is not a floor, and it made
        # every "clears the control" statement depend on which draw it was compared against.
        import torch

        first = next(iter(models.values()))
        torch.manual_seed(UNTRAINED_SEED)
        models["untrained_control"] = FinancialTFM(first.cfg)

    results: list[ProbeResult] = []
    for probe in PROBES:
        arms: dict[str, list[float]] = {}
        for seed in seeds:
            Xtr, ytr, Xte, yte = make_probe(probe, n_features=n_features, seed=seed)
            if len(np.unique(yte)) < 2:
                continue
            for name, m in models.items():
                clf = FinancialTFMClassifier(
                    m, max_context=max_context, context_strategy="uniform",
                    feature_transform="rank", random_state=seed,
                ).fit(Xtr, ytr)
                arms.setdefault(name, []).append(
                    roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])
                )
            lr = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
            arms.setdefault("logistic_regression", []).append(
                roc_auc_score(yte, lr.predict_proba(Xte)[:, 1])
            )
            gb = HistGradientBoostingClassifier(max_iter=100, random_state=seed).fit(Xtr, ytr)
            arms.setdefault("gradient_boosting", []).append(
                roc_auc_score(yte, gb.predict_proba(Xte)[:, 1])
            )
        for arm, vals in arms.items():
            v = np.array(vals)
            results.append(
                ProbeResult(probe, arm, float(v.mean()),
                            float(v.std(ddof=1)) if len(v) > 1 else float("nan"), len(v))
            )
        print(f"  {probe} done", flush=True)

    record = {
        "probes": list(PROBES),
        "config": {
            "models": model_paths, "seeds": list(seeds),
            "max_context": max_context, "n_features": n_features,
        },
        "results": [asdict(r) for r in results],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "capability.json").write_text(json.dumps(record, indent=2))
    return record


def summarise(record: dict) -> str:
    """Render arms against probes, with the ceiling visible."""
    rows = record["results"]
    arms = sorted({r["arm"] for r in rows})
    lines = [f"{'arm':>22} " + " ".join(f"{p:>14}" for p in record["probes"])]
    for arm in arms:
        cells = []
        for probe in record["probes"]:
            hit = [r for r in rows if r["arm"] == arm and r["probe"] == probe]
            cells.append(f"{hit[0]['auc_mean']:.3f}±{hit[0]['auc_std']:.3f}" if hit else "—")
        lines.append(f"{arm:>22} " + " ".join(f"{c:>14}" for c in cells))
    lines += [
        "",
        "Reading it: `linear` has a ceiling near 1.0, so anything below is the model's own",
        "limit. `xor` pins logistic regression at chance by construction. On `noise` every arm",
        "must sit at ~0.5 — an arm above it is leaking, not learning.",
        "",
        "The untrained control is the floor. A trained arm that does not clear it by a wide",
        "margin has not learned to do in-context prediction (docs/FINDINGS.md §42).",
    ]
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--models", type=str, required=True,
        help="comma-separated name=path pairs, e.g. small=runs/a.pt,fixed=runs/b.pt",
    )
    p.add_argument("--out", type=str, default="runs/capability")
    p.add_argument("--seeds", type=str, default="0,1,2")
    p.add_argument("--max-context", type=int, default=1000)
    p.add_argument("--n-features", type=int, default=20)
    p.add_argument(
        "--feature-sweep", action="store_true",
        help="sweep feature count instead of running the probe suite (§49): the test that "
             "separates a capacity limit from a pooling-design limit",
    )
    p.add_argument("--widths", type=str, default="5,10,20,40,80,130")
    p.add_argument(
        "--rate-sweep", action="store_true",
        help="sweep the base rate instead (§51): does the model order the distribution, or "
             "only detect extremes?",
    )
    p.add_argument("--rates", type=str, default="0.05,0.15,0.30,0.50")
    p.add_argument(
        "--class-sweep", action="store_true",
        help="sweep the number of classes instead (task 46.1): the only instrument that "
             "makes 'the model does multiclass' a measurement rather than a command line",
    )
    p.add_argument("--class-counts", type=str, default="3,5,10")
    args = p.parse_args()
    paths = dict(pair.split("=", 1) for pair in args.models.split(",") if pair)
    if args.class_sweep:
        sweep = multiclass_sweep(
            paths, class_counts=tuple(int(k) for k in args.class_counts.split(",")),
            seeds=tuple(int(s) for s in args.seeds.split(",")),
            max_context=args.max_context, n_features=args.n_features,
        )
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "class_sweep.json").write_text(json.dumps(sweep, indent=2))
        print("\n" + summarise_multiclass_sweep(sweep))
        return
    if args.rate_sweep:
        sweep = base_rate_sweep(
            paths, rates=tuple(float(r) for r in args.rates.split(",")),
            seeds=tuple(int(s) for s in args.seeds.split(",")),
            max_context=args.max_context, n_features=args.n_features or 5,
        )
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "rate_sweep.json").write_text(json.dumps(sweep, indent=2))
        print("\n" + summarise_rate_sweep(sweep))
        return
    if args.feature_sweep:
        sweep = feature_sweep(
            paths, widths=tuple(int(w) for w in args.widths.split(",")),
            seeds=tuple(int(s) for s in args.seeds.split(",")),
            max_context=args.max_context,
        )
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "feature_sweep.json").write_text(json.dumps(sweep, indent=2))
        print("\n" + summarise_sweep(sweep))
        return
    record = run(
        paths, Path(args.out), seeds=tuple(int(s) for s in args.seeds.split(",")),
        max_context=args.max_context, n_features=args.n_features,
    )
    print("\n" + summarise(record))


if __name__ == "__main__":
    main()
