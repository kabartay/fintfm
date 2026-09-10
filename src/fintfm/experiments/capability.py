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
PROBES: tuple[str, ...] = ("linear", "conjunction", "xor", "noise")

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
    if kind == "linear":
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
    else:  # noise
        score = rng.normal(size=n)
    y = (score >= np.quantile(score, 1 - rate)).astype(np.int64)
    k = n // 2
    return X[:k], y[:k], X[k:], y[k:]


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
    args = p.parse_args()
    paths = dict(pair.split("=", 1) for pair in args.models.split(",") if pair)
    record = run(
        paths, Path(args.out), seeds=tuple(int(s) for s in args.seeds.split(",")),
        max_context=args.max_context, n_features=args.n_features,
    )
    print("\n" + summarise(record))


if __name__ == "__main__":
    main()
