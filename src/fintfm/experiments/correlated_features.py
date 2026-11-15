"""Correlated-feature and confounding tests (openspec task 39.14).

Why this exists
---------------
39.13 (``irrelevant_features``) adds columns that carry no signal. This harness adds columns
that carry *redundant or misleading* signal, and asks the question 39.14's Verify poses: how much
does the model rely on a confound versus the true cause ``z``, measured **directly** by an
intervention that breaks the confound only.

Designs (all share the same latent draws across correlation levels, so only the strength moves)
------------------------------------------------------------------------------------------------
- ``redundant``: ``x1 = sqrt(r) z + sqrt(1-r) e1``, ``x2 = sqrt(r) z + sqrt(1-r) e2``,
  ``y = f(z)``. ``corr(x1, x2) = r``. Both columns are equally good proxies of ``z``; roles:
  primary = x1, secondary = x2. The question is how the model splits its reliance.
- ``proxy``: primary ``x0 = z`` (the cause, observed exactly), secondary
  ``c = r z + sqrt(1-r^2) e``; ``y = f(z)``. ``c`` is a spurious proxy: ``y`` is independent
  of ``c`` given ``z``. ``corr(x0, c) = r``.
- ``confounder``: hidden ``u``; primary ``x0 = sqrt(r) u + sqrt(1-r) e0`` is the cause,
  ``y = f(x0)``; secondary ``c = sqrt(r) u + sqrt(1-r) e1`` is a child of ``u`` only.
  ``corr(x0, c) = r``; ``c`` predicts ``y`` only through ``u``.
- ``collider``: primary ``x0`` causes ``y``; ``y_latent = x0 + 0.5 e_y``, ``y = 1[y_latent >= q]``;
  secondary ``c = sqrt(r) std(y_latent) + sqrt(1-r) e``, a child of the outcome, so it is
  predictive of ``y`` in the context rows yet is not a cause. Here ``r`` is the squared
  correlation ``corr(c, y_latent)^2``, not ``corr(x0, c)``. ``y`` depends on ``x0`` and ``e_y``.

Because the rule ``f`` is fixed, ``y`` is identical across correlation levels for ``redundant``
and ``proxy`` (only the columns change), which the tests check. At ``r = 0`` the secondary
column (and for ``redundant`` the primary too) is pure noise, so signal in the columns grows with
``r`` in ``redundant``; the ``z_oracle`` row gives each design's ceiling.

Intervention
------------
The model, the floors and the baselines are fitted on the observational context. On the query
rows each column is then replaced by a **permutation of itself across the query rows**
(``do(column ~ its own marginal)``), which breaks that column's link to ``z``, to the other
column and to ``y`` while keeping its marginal. Per arm and design/level this records:

- ``auc_obs``: AUC on the untouched query rows;
- ``auc_do_primary`` / ``auc_do_secondary``: AUC after intervening on that one column;
- ``drop_primary`` / ``drop_secondary``: ``auc_obs`` minus the intervened AUC;
- ``reliance_secondary``: ``max(drop_secondary, 0) / (max(drop_primary, 0) + max(drop_secondary,
  0))``, the share of the total drop caused by the secondary column. ``None`` if both drops are
  non-positive. Under ``redundant`` a value near 0.5 is an even split (attribution between the
  two proxies); under the other designs a value near 0 means the model uses the cause and
  ignores the confound/collider.

Logistic regression and histogram gradient boosting are the classical floors; the untrained
control of the same architecture (seeded) is the model floor (CLAUDE.md "Every evaluation
needs a floor").

This is verification of reliance on synthetic structure, not validation against any real data.
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
from fintfm.experiments.irrelevant_features import UNTRAINED_SEED, _score_in_context

if TYPE_CHECKING:
    from fintfm.modeling.model import FinancialTFM

#: Design names, in report order.
DESIGNS: tuple[str, ...] = ("redundant", "proxy", "confounder", "collider")

#: Correlation strengths ``r`` (see each design for what is correlated).
LEVELS: tuple[float, ...] = (0.0, 0.3, 0.6, 0.8, 0.9, 0.95, 0.99)


@dataclass
class Cell:
    """One arm on one design at one correlation level, summarised over seeds."""

    arm: str
    design: str
    r: float
    auc_obs: float
    auc_do_primary: float
    auc_do_secondary: float
    drop_primary: float
    drop_secondary: float
    reliance_secondary: float | None
    n_seeds: int


def _mix(shared: np.ndarray, own: np.ndarray, r: float) -> np.ndarray:
    """Unit-variance column with correlation ``sqrt(r)`` to ``shared`` (both unit variance)."""
    return np.sqrt(r) * shared + np.sqrt(1.0 - r) * own


def make_design(
    design: str, r: float, seed: int, n: int = 2000, rate: float = 0.25
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate one design at one correlation level.

    The latent draws depend on ``seed`` only, never on ``r``, so across levels the same rows get
    the same ``z`` (and the same ``y`` for ``redundant`` and ``proxy``).

    Args:
        design: One of :data:`DESIGNS`.
        r: Correlation strength in ``[0, 1]`` (what is correlated depends on the design).
        seed: Random seed; the same arguments give bit-identical output.
        n: Rows.
        rate: Positive-class rate, set by thresholding the score at that upper quantile.

    Returns:
        ``(X, y, z)``: ``X`` is ``(n, 2)`` float32 with the primary column first and the
        secondary second; ``y`` is ``(n,)`` int64; ``z`` is the ``(n,)`` latent that ``y`` is a
        threshold of (``x0`` for ``confounder``, ``y_latent`` for ``collider``).

    Raises:
        ValueError: On an unknown design or ``r`` outside ``[0, 1]``.
    """
    if design not in DESIGNS:
        raise ValueError(f"unknown design {design!r}; choose from {DESIGNS}")
    if not 0.0 <= r <= 1.0:
        raise ValueError(f"r must be in [0, 1], got {r}")
    rng = np.random.default_rng(np.random.SeedSequence([seed, DESIGNS.index(design)]))
    z, e0, e1 = rng.normal(size=(3, n))

    def label(score: np.ndarray) -> np.ndarray:
        return (score >= np.quantile(score, 1 - rate)).astype(np.int64)

    if design == "redundant":
        X = np.stack([_mix(z, e0, r), _mix(z, e1, r)], axis=1)
        y, latent = label(z), z
    elif design == "proxy":
        X = np.stack([z, _mix(z, e0, r**2)], axis=1)  # loading r, so corr(x0, c) = r
        y, latent = label(z), z
    elif design == "confounder":
        x0 = _mix(z, e0, r)  # z plays the hidden u
        X = np.stack([x0, _mix(z, e1, r)], axis=1)
        y, latent = label(x0), x0
    else:  # collider
        latent = z + 0.5 * e0  # y_latent = x0 + noise
        X = np.stack([z, _mix(latent / latent.std(), e1, r)], axis=1)
        y = label(latent)
    return X.astype(np.float32), y, np.asarray(latent, dtype=np.float32)


def realised_correlation(design: str, X: np.ndarray, z: np.ndarray) -> float:
    """The quantity that ``r`` is defined as, measured on a generated sample.

    Args:
        design: One of :data:`DESIGNS`.
        X: ``(n, 2)`` columns from :func:`make_design`.
        z: Latent from :func:`make_design`.

    Returns:
        ``corr(x1, x2)`` for ``redundant``, ``proxy`` and ``confounder``;
        ``corr(c, y_latent)`` squared for ``collider``.
    """
    if design == "collider":
        return float(np.corrcoef(X[:, 1], z)[0, 1] ** 2)
    return float(np.corrcoef(X[:, 0], X[:, 1])[0, 1])


def intervene(X: np.ndarray, col: int, seed: int) -> np.ndarray:
    """Replace one column by a permutation of itself (``do(col ~ own marginal)``).

    Args:
        X: ``(n, d)`` query rows; not modified.
        col: Column to intervene on.
        seed: Seed of the permutation.

    Returns:
        Copy of ``X`` with column ``col`` permuted across rows, so its link to every other
        column and to ``y`` is broken while its marginal is unchanged.
    """
    out = X.copy()
    out[:, col] = X[np.random.default_rng(seed).permutation(len(X)), col]
    return out


def reliance_share(drop_primary: float, drop_secondary: float) -> float | None:
    """Share of the total AUC drop caused by the secondary column.

    Args:
        drop_primary: AUC lost when the primary column is intervened on.
        drop_secondary: AUC lost when the secondary column is intervened on.

    Returns:
        ``max(ds, 0) / (max(dp, 0) + max(ds, 0))``; ``None`` if both are non-positive.
    """
    dp, ds = max(drop_primary, 0.0), max(drop_secondary, 0.0)
    return None if dp + ds <= 0 else ds / (dp + ds)


def run(
    model_paths: dict[str, str],
    out_dir: Path | None = None,
    designs: tuple[str, ...] = DESIGNS,
    levels: tuple[float, ...] = LEVELS,
    seeds: tuple[int, ...] = (0, 1, 2),
    n: int = 2000,
    max_context: int = 1000,
    rate: float = 0.25,
    models: dict[str, FinancialTFM] | None = None,
) -> dict:
    """Score every arm on every design and level, observationally and under intervention.

    Args:
        model_paths: ``{name: checkpoint path}``; one untrained control is built from the first
            one's architecture. Ignored when ``models`` is given.
        out_dir: If given, ``correlated_features.json`` is written there.
        designs: Designs to run, from :data:`DESIGNS`.
        levels: Correlation strengths.
        seeds: Task seeds; everything is repeated per seed and averaged.
        n: Rows per task, half context and half query.
        max_context: Context rows handed to the in-context arms.
        rate: Positive-class rate.
        models: Already-built models by name (used by tests); the control is added.

    Returns:
        The record: config, per-cell results, and the largest deviation of the realised
        correlation from the requested one per design.
    """
    import torch
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    from fintfm.modeling.model import FinancialTFM

    loaded = (
        dict(models)
        if models is not None
        else {name: FinancialTFM.load(p) for name, p in model_paths.items()}
    )
    if loaded:
        torch.manual_seed(UNTRAINED_SEED)  # seeded: an unseeded floor moves by 0.2 (§42)
        loaded["untrained_control"] = FinancialTFM(next(iter(loaded.values())).cfg)

    keys = ("obs", "primary", "secondary")
    acc: dict[tuple[str, str, float], dict[str, list[float]]] = {}
    corr_dev: dict[str, float] = {}
    cov = Coverage()
    nc = n // 2
    for design in designs:
        for r in levels:
            for seed in seeds:
                X, y, z = make_design(design, r, seed, n, rate)
                dev = abs(realised_correlation(design, X, z) - r)
                corr_dev[design] = max(corr_dev.get(design, 0.0), dev)
                ytr, yte = y[:nc], y[nc:]
                unit = f"{design}/r{r}/seed{seed}"
                cov.attempt(unit)
                if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
                    cov.skip(unit, "one class in the train or test half")
                    continue
                cov.scored(unit)
                Xtr, Xte = X[:nc], X[nc:]
                views = {
                    "obs": Xte,
                    "primary": intervene(Xte, 0, seed + 1000),
                    "secondary": intervene(Xte, 1, seed + 2000),
                }
                lr = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
                gb = HistGradientBoostingClassifier(max_iter=100, random_state=seed).fit(Xtr, ytr)
                preds: dict[str, dict[str, np.ndarray]] = {
                    name: {
                        k: _score_in_context(m, Xtr, ytr, Q, max_context, seed)
                        for k, Q in views.items()
                    }
                    for name, m in loaded.items()
                }
                preds["logistic_regression"] = {
                    k: lr.predict_proba(Q)[:, 1] for k, Q in views.items()
                }
                preds["gradient_boosting"] = {
                    k: gb.predict_proba(Q)[:, 1] for k, Q in views.items()
                }
                for arm, by_view in preds.items():
                    c = acc.setdefault((arm, design, r), {k: [] for k in keys})
                    for k in keys:
                        c[k].append(float(roc_auc_score(yte, by_view[k])))
                oracle = acc.setdefault(("z_oracle", design, r), {k: [] for k in keys})
                for k in keys:
                    oracle[k].append(float(roc_auc_score(yte, z[nc:])))
        print(f"  design {design} done", flush=True)

    cells = []
    for (arm, design, r), c in acc.items():
        # Drops are averaged per seed first, so the share is not a ratio of means of differences.
        drop_p = float(np.mean(np.array(c["obs"]) - np.array(c["primary"])))
        drop_s = float(np.mean(np.array(c["obs"]) - np.array(c["secondary"])))
        cells.append(
            Cell(
                arm=arm,
                design=design,
                r=r,
                auc_obs=float(np.mean(c["obs"])),
                auc_do_primary=float(np.mean(c["primary"])),
                auc_do_secondary=float(np.mean(c["secondary"])),
                drop_primary=drop_p,
                drop_secondary=drop_s,
                reliance_secondary=None if arm == "z_oracle" else reliance_share(drop_p, drop_s),
                n_seeds=len(c["obs"]),
            )
        )
    record = {
        "task": "correlated and confounded designs; query-time permutation interventions",
        "config": {
            "models": model_paths,
            "designs": list(designs),
            "levels": list(levels),
            "seeds": list(seeds),
            "n": n,
            "max_context": max_context,
            "rate": rate,
        },
        "max_correlation_deviation": corr_dev,
        "results": [asdict(c) for c in cells],
        "coverage": cov.to_dict(),
    }
    print(cov.summary("design-level-seeds"), flush=True)
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "correlated_features.json").write_text(json.dumps(record, indent=2))
    return record


def summarise(record: dict) -> str:
    """Render, per design, AUC and the reliance share by arm and correlation level.

    Args:
        record: Output of :func:`run`.

    Returns:
        A printable multi-line string.
    """
    rows = record["results"]
    lines: list[str] = []
    for design in record["config"]["designs"]:
        sub = [r for r in rows if r["design"] == design]
        if not sub:
            continue
        levels = sorted({r["r"] for r in sub})
        arms = sorted({r["arm"] for r in sub})
        head = " ".join(f"{'r=' + format(v, 'g'):>8}" for v in levels)
        for title, key, fmt in (
            ("ROC-AUC, observational", "auc_obs", "{:.3f}"),
            ("secondary share of AUC drop", "reliance_secondary", "{:.2f}"),
        ):
            lines += ["", f"[{design}] {title}", f"{'arm':>22} {head}"]
            for arm in arms:
                cells = []
                for v in levels:
                    hit = [r for r in sub if r["arm"] == arm and r["r"] == v]
                    val = hit[0][key] if hit else None
                    cells.append(fmt.format(val) if val is not None else "-")
                lines.append(f"{arm:>22} " + " ".join(f"{c:>8}" for c in cells))
    lines += [
        "",
        "Reading it: the secondary share is the fraction of the AUC lost under intervention that",
        "comes from permuting the secondary column (proxy, confounder child, collider, second",
        "copy) rather than the primary one. 0 = only the cause is used; 0.5 under 'redundant' is",
        "an even split. The untrained control is the floor; z_oracle is each design's ceiling.",
        "This is verification on synthetic structure, not validation against observed data.",
    ]
    return "\n".join(lines)


def main() -> None:
    """Command-line entry point: ``python -m fintfm.experiments.correlated_features``."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--models",
        type=str,
        required=True,
        help="comma-separated name=path pairs, e.g. fin00=runs/a.pt,fin10=runs/b.pt",
    )
    p.add_argument("--out", type=str, default="runs/correlated_features")
    p.add_argument("--seeds", type=str, default="0,1,2")
    p.add_argument("--designs", type=str, default=",".join(DESIGNS))
    p.add_argument("--levels", type=str, default=",".join(str(v) for v in LEVELS))
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--max-context", type=int, default=1000)
    p.add_argument("--rate", type=float, default=0.25)
    args = p.parse_args()
    paths = dict(pair.split("=", 1) for pair in args.models.split(",") if pair)
    record = run(
        paths,
        Path(args.out),
        designs=tuple(args.designs.split(",")),
        levels=tuple(float(v) for v in args.levels.split(",")),
        seeds=tuple(int(s) for s in args.seeds.split(",")),
        n=args.n,
        max_context=args.max_context,
        rate=args.rate,
    )
    print("\n" + summarise(record))


if __name__ == "__main__":
    main()
