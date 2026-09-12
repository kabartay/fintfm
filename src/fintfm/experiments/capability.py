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
    args = p.parse_args()
    paths = dict(pair.split("=", 1) for pair in args.models.split(",") if pair)
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
