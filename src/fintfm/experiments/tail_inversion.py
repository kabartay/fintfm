"""Does the symmetric-probe inversion track feature tail weight? (openspec task 38.14).

Why this exists
---------------
``docs/results/FINDINGS.md`` §72: the identity-shuffle checkpoint scored 0.2988 on
``symmetric_sum`` and 0.2597 on ``symmetric_count`` (AUC, 16 seeds), below the 0.5 chance floor,
where every other checkpoint scores >= 0.90. §72's candidate mechanism, untested: independently
permuted numerator and denominator accounts give far more extreme derived ratios than the true
accounts, so the model may have learned "extreme values across many columns" as a spurious cue,
which misfires on a probe whose positive class is a large sum or count.

This harness is the constructed-probe half of the 38.14 verification: the symmetric probes of
:mod:`fintfm.experiments.capability` with the **feature distribution varied from Gaussian to
heavy-tailed and everything else held fixed**. If the mechanism is right, a checkpoint's
symmetric AUC should fall as the tails get heavier, and the identity-shuffle checkpoint should
fall furthest. The other half (are identity-shuffled prior tasks heavier-tailed than
unshuffled ones?) is a property of the prior and is not measured here.

Design
------
- Families: ``gaussian`` and Student-t with ``df`` in :data:`T_DFS`, so tail weight rises as ``df``
  falls. Every non-Gaussian family is rescaled so its **interquartile range equals the standard
  normal's**, which matches the bulk and leaves the tails as the only thing that differs.
- **Signal is matched exactly**: the label is the top ``rate`` of the noise-free symmetric score
  (``sum`` or ``count(x > 0)``), as in ``make_probe``, so the oracle AUC is 1.0 in every family
  and the label rule is the same (the count probe has ties at the cut, so its rate exceeds ``rate``). ``gaussian`` reproduces ``make_probe`` bit for bit (tested).
  ``count`` is invariant to monotone distortions of the marginals, so it is the control for
  "heavy tails change the label"; ``sum`` is not (one extreme column can decide it).
- The in-context classifier is run with ``feature_transform="rank"``, exactly as in
  ``capability.run``, so any tail effect has to survive the per-column rank transform.
- Arms: each checkpoint, a seeded **untrained control** of the same architecture (the floor;
  CLAUDE.md "Every evaluation needs a floor"), logistic regression and gradient boosting.
- Reported per family: sample excess kurtosis and a robust tail ratio (99.9th percentile over the
  75th), mean AUC and AP per arm and probe. Reported per (arm, probe): the Spearman correlation
  between tail weight and AUC, the AUC slope per step of tail weight, and whether any family
  scores below 0.5.

Nothing here trains. Numbers are MEASURED only when produced by the command in the openspec
Progress note on real checkpoints; the unit tests are SMOKE-TEST only.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from fintfm.evaluation.coverage import Coverage
from fintfm.experiments.capability import SYMMETRIC_PROBES, UNTRAINED_SEED
from fintfm.experiments.irrelevant_features import _score_in_context

if TYPE_CHECKING:
    from fintfm.modeling.model import FinancialTFM

#: Student-t degrees of freedom, heaviest tail last. ``df`` <= 4 has infinite kurtosis.
T_DFS: tuple[float, ...] = (10.0, 5.0, 3.0, 2.0)

#: Family names in order of increasing tail weight.
FAMILIES: tuple[str, ...] = ("gaussian",) + tuple(f"t{int(d)}" for d in T_DFS)

#: Interquartile range of the standard normal; every family is rescaled to it.
_NORMAL_IQR = 1.3489795003921634


@dataclass
class TailPoint:
    """One arm on one symmetric probe in one tail family, averaged over seeds."""

    arm: str
    probe: str
    family: str
    auc_mean: float
    auc_std: float
    ap_mean: float
    n_seeds: int


def _family_df(family: str) -> float | None:
    """Degrees of freedom of a family name; ``None`` for the Gaussian."""
    if family == "gaussian":
        return None
    if family not in FAMILIES:
        raise ValueError(f"unknown tail family {family!r}; expected one of {FAMILIES}")
    return float(family[1:])


def sample_features(family: str, n: int, n_features: int, rng: np.random.Generator) -> np.ndarray:
    """Draw an ``(n, n_features)`` float32 matrix from a tail family.

    Args:
        family: One of :data:`FAMILIES`.
        n: Rows.
        n_features: Columns.
        rng: Generator; ``gaussian`` consumes it exactly as ``make_probe`` does.

    Returns:
        Independent columns with the family's tails and a Gaussian-matched interquartile range.

    Raises:
        ValueError: If ``family`` is unknown.
    """
    df = _family_df(family)
    if df is None:
        return rng.normal(size=(n, n_features)).astype(np.float32)
    from scipy.stats import t as student_t

    iqr = float(student_t.ppf(0.75, df) - student_t.ppf(0.25, df))
    raw = rng.standard_t(df, size=(n, n_features)) * (_NORMAL_IQR / iqr)
    return raw.astype(np.float32)


def tail_stats(X: np.ndarray) -> dict[str, float]:
    """Tail weight of a feature matrix, pooled over its columns.

    Args:
        X: Feature matrix.

    Returns:
        ``excess_kurtosis`` (sample, pooled) and ``tail_ratio`` (99.9th percentile of ``|x|``
        over its 75th, robust to the infinite-kurtosis families).
    """
    x = np.asarray(X, dtype=np.float64).ravel()
    x = x - x.mean()
    m2 = float(np.mean(x**2))
    kurt = float(np.mean(x**4) / m2**2 - 3.0) if m2 > 0 else float("nan")
    a = np.abs(x)
    return {
        "excess_kurtosis": kurt,
        "tail_ratio": float(np.quantile(a, 0.999) / np.quantile(a, 0.75)),
    }


def make_tail_probe(
    kind: str,
    family: str,
    n: int = 8000,
    n_features: int = 20,
    rate: float = 0.05,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A symmetric probe whose features follow a tail family at a matched label rule.

    Args:
        kind: One of :data:`~fintfm.experiments.capability.SYMMETRIC_PROBES`.
        family: One of :data:`FAMILIES`.
        n: Total rows, split evenly into context and query.
        n_features: Feature width.
        rate: Positive-class rate.
        seed: Random seed.

    Returns:
        ``(X_train, y_train, X_test, y_test)``. For ``family="gaussian"`` this equals
        :func:`fintfm.experiments.capability.make_probe`.

    Raises:
        ValueError: If ``kind`` is not a symmetric probe or ``family`` is unknown.
    """
    if kind not in SYMMETRIC_PROBES:
        raise ValueError(f"unknown symmetric probe {kind!r}; expected one of {SYMMETRIC_PROBES}")
    rng = np.random.default_rng(seed)
    X = sample_features(family, n, n_features, rng)
    if kind == "symmetric_sum":
        score = X.sum(axis=1)
    else:
        score = (X > 0).sum(axis=1).astype(np.float64)
    y = (score >= np.quantile(score, 1 - rate)).astype(np.int64)
    k = n // 2
    return X[:k], y[:k], X[k:], y[k:]


def tail_trend(families: list[str], auc: list[float]) -> dict[str, float | bool]:
    """Summarise how a score moves with tail weight.

    Args:
        families: Family names, any order, drawn from :data:`FAMILIES`.
        auc: Mean AUC per family, aligned with ``families``.

    Returns:
        ``spearman`` (rank correlation of AUC with tail weight; negative means AUC falls as tails
        thicken; NaN when AUC is constant), ``slope_per_step`` (OLS AUC change per step along
        :data:`FAMILIES`), ``total_change`` (heaviest minus lightest family present) and
        ``inverted`` (whether any family is below 0.5).

    Raises:
        ValueError: If fewer than two families are given or the lists differ in length.
    """
    if len(families) != len(auc) or len(families) < 2:
        raise ValueError("need at least two families with one AUC each")
    from scipy.stats import spearmanr

    order = np.array([FAMILIES.index(f) for f in families], dtype=float)
    a = np.asarray(auc, dtype=float)
    if np.ptp(a) == 0.0:
        rho = float("nan")
    else:
        rho = float(spearmanr(order, a).statistic)
    slope = float(np.polyfit(order, a, 1)[0])
    idx = np.argsort(order)
    return {
        "spearman": rho,
        "slope_per_step": slope,
        "total_change": float(a[idx[-1]] - a[idx[0]]),
        "inverted": bool(np.any(a < 0.5)),
    }


def run(
    model_paths: dict[str, str],
    out_dir: Path | None = None,
    families: tuple[str, ...] = FAMILIES,
    probes: tuple[str, ...] = SYMMETRIC_PROBES,
    seeds: tuple[int, ...] = (0, 1, 2),
    n: int = 8000,
    n_features: int = 20,
    max_context: int = 1000,
    rate: float = 0.05,
    models: dict[str, FinancialTFM] | None = None,
) -> dict:
    """Score every arm on every symmetric probe in every tail family.

    Args:
        model_paths: ``{arm name: checkpoint path}``. Ignored when ``models`` is given.
        out_dir: If given, ``tail_inversion.json`` is written there.
        families: Tail families to run, from :data:`FAMILIES`.
        probes: Symmetric probes to run.
        seeds: Seeds per cell.
        n: Rows per task, half context and half query.
        n_features: Probe width; must not exceed a model's ``max_features``.
        max_context: Context rows handed to the in-context arms.
        rate: Positive-class rate.
        models: Already-built models by name (used by tests); the untrained control is added.

    Returns:
        The record: config, per-family tail statistics, per-cell results, per-(arm, probe) trends
        and coverage.

    Raises:
        ValueError: If a model's ``max_features`` is below ``n_features``.
    """
    import torch
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score

    from fintfm.modeling.model import FinancialTFM

    loaded = (
        dict(models)
        if models is not None
        else {name: FinancialTFM.load(p) for name, p in model_paths.items()}
    )
    if loaded:
        torch.manual_seed(UNTRAINED_SEED)  # seeded: an unseeded floor moves by 0.2 (§42)
        loaded["untrained_control"] = FinancialTFM(next(iter(loaded.values())).cfg)
    for name, m in loaded.items():
        if n_features > m.cfg.max_features:
            raise ValueError(f"n_features {n_features} exceeds {name}'s max_features")

    cells: dict[tuple[str, str, str], dict[str, list[float]]] = {}
    stats: dict[str, dict[str, float]] = {}
    cov = Coverage()
    for family in families:
        for probe in probes:
            for seed in seeds:
                unit = f"{family}/{probe}/seed{seed}"
                cov.attempt(unit)
                Xtr, ytr, Xte, yte = make_tail_probe(probe, family, n, n_features, rate, seed)
                if len(np.unique(yte)) < 2 or len(np.unique(ytr)) < 2:
                    cov.skip(unit, "single-class split: AUC undefined")
                    continue
                cov.scored(unit)
                if family not in stats:
                    stats[family] = tail_stats(Xtr)
                preds: dict[str, np.ndarray] = {
                    name: _score_in_context(m, Xtr, ytr, Xte, max_context, seed)
                    for name, m in loaded.items()
                }
                preds["logistic_regression"] = (
                    LogisticRegression(max_iter=1000).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
                )
                preds["gradient_boosting"] = (
                    HistGradientBoostingClassifier(max_iter=100, random_state=seed)
                    .fit(Xtr, ytr)
                    .predict_proba(Xte)[:, 1]
                )
                for arm, p in preds.items():
                    c = cells.setdefault((arm, probe, family), {"auc": [], "ap": []})
                    c["auc"].append(float(roc_auc_score(yte, p)))
                    c["ap"].append(float(average_precision_score(yte, p)))
            print(f"  {family}/{probe} done", flush=True)

    points = []
    for (arm, probe, family), c in cells.items():
        auc = np.array(c["auc"])
        points.append(
            TailPoint(
                arm=arm,
                probe=probe,
                family=family,
                auc_mean=float(auc.mean()),
                auc_std=float(auc.std(ddof=1)) if len(auc) > 1 else float("nan"),
                ap_mean=float(np.mean(c["ap"])),
                n_seeds=len(auc),
            )
        )
    trends: dict[str, dict[str, dict]] = {}
    for arm in sorted({p.arm for p in points}):
        for probe in probes:
            pts = [p for p in points if p.arm == arm and p.probe == probe]
            if len(pts) >= 2:
                trends.setdefault(arm, {})[probe] = tail_trend(
                    [p.family for p in pts], [p.auc_mean for p in pts]
                )
    record = {
        "task": "symmetric probes with Gaussian-to-Student-t features, bulk-matched, "
        "noise-free label (oracle AUC 1.0 in every family)",
        "config": {
            "models": model_paths,
            "families": list(families),
            "probes": list(probes),
            "seeds": list(seeds),
            "n": n,
            "n_features": n_features,
            "max_context": max_context,
            "rate": rate,
        },
        "tail_stats": stats,
        "results": [asdict(p) for p in points],
        "trends": trends,
        "coverage": cov.to_dict(),
    }
    print(cov.summary("family-probe-seed cells"), flush=True)
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "tail_inversion.json").write_text(json.dumps(record, indent=2))
    return record


def summarise(record: dict) -> str:
    """Render the record as a table: AUC per arm and probe across families, then the trends.

    Args:
        record: The dictionary returned by :func:`run`.

    Returns:
        A printable multi-line string.
    """
    fams = record["config"]["families"]
    lines = ["tail weight (excess kurtosis / tail ratio):"]
    for f in fams:
        s = record["tail_stats"].get(f)
        if s:
            lines.append(f"  {f:<9} {s['excess_kurtosis']:>9.2f}  {s['tail_ratio']:>7.2f}")
    for probe in record["config"]["probes"]:
        lines.append(f"\n{probe}: mean AUC (below 0.5 = inverted)")
        lines.append(f"  {'arm':<22}" + "".join(f"{f:>10}" for f in fams) + "   spearman")
        arms = sorted({r["arm"] for r in record["results"] if r["probe"] == probe})
        for arm in arms:
            by = {
                r["family"]: r["auc_mean"]
                for r in record["results"]
                if r["arm"] == arm and r["probe"] == probe
            }
            rho = record["trends"].get(arm, {}).get(probe, {}).get("spearman", float("nan"))
            row = "".join(f"{by[f]:>10.4f}" if f in by else f"{'-':>10}" for f in fams)
            lines.append(f"  {arm:<22}{row}   {rho:>8.2f}")
    return "\n".join(lines)


def main() -> None:
    """Command-line entry point: ``python -m fintfm.experiments.tail_inversion``."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--models",
        type=str,
        required=True,
        help="comma-separated name=path pairs, e.g. idshuffle=runs/dl/v4-colid-idshuffle.pt",
    )
    p.add_argument("--out", type=str, default="runs/tail_inversion")
    p.add_argument("--seeds", type=str, default="0,1,2")
    p.add_argument("--families", type=str, default=",".join(FAMILIES))
    p.add_argument("--n", type=int, default=8000)
    p.add_argument("--n-features", type=int, default=20)
    p.add_argument("--max-context", type=int, default=1000)
    p.add_argument("--rate", type=float, default=0.05)
    args = p.parse_args()
    paths = dict(pair.split("=", 1) for pair in args.models.split(",") if pair)
    record = run(
        paths,
        Path(args.out),
        families=tuple(f for f in args.families.split(",") if f),
        seeds=tuple(int(s) for s in args.seeds.split(",")),
        n=args.n,
        n_features=args.n_features,
        max_context=args.max_context,
        rate=args.rate,
    )
    print("\n" + summarise(record))


if __name__ == "__main__":
    main()
