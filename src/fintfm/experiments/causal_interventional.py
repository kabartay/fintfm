"""Observational versus interventional comparison on a simple SCM (openspec task 39.15).

Why this exists
---------------
39.14 (``correlated_features``) measures reliance by permuting columns at query time. This
harness asks the sharper question of 39.15: does the in-context model answer ``P(Y | X1)`` (an
association) or ``P(Y | do(X1))`` (an effect), measured directly on held-out *interventional*
data whose labels come from the structural equation, with closed-form truths to compare against.

The SCM (linear-Gaussian, unit-variance nodes, ``Z`` hidden from every learner)
-------------------------------------------------------------------------------
``Z ~ N(0, 1)``; ``X1 = a Z + sqrt(1-a^2) e1``; ``X2 = b Z + sqrt(1-b^2) e2``;
``Y ~ Bernoulli(sigmoid(w X1 + g Z + c))``. Edges ``Z->X1``, ``Z->X2``, ``X1->Y``; ``g`` is the
strength of an optional hidden ``Z->Y`` edge. ``g = 0`` is exactly the task's SCM, where
``X1`` is unconfounded and the observational and interventional answers for ``X1`` coincide
(the null case a correct learner must reproduce). ``g > 0`` confounds ``X1->Y`` and makes
``P(Y|x1,x2) != P(Y|do(x1),x2)``.

Closed forms (Gauss-Hermite over the Gaussian posterior of ``Z``)
-----------------------------------------------------------------
- observational, ``P(Y=1 | x1, x2)``: ``Z | x1, x2`` is Gaussian with precision
  ``1 + a^2/s1 + b^2/s2`` and mean ``(a x1/s1 + b x2/s2) / precision`` (``s1 = 1 - a^2``);
- interventional, ``P(Y=1 | do(X1=x1), x2)``: ``do`` cuts ``Z->X1``, so ``x1`` carries no
  information about ``Z`` and ``Z | x2`` alone is used.

Data and arms
-------------
Context: observational rows ``(x1, x2, y)``. Interventional query rows: query rows where the
``X1`` column is replaced by a permutation of itself (``do(X1 ~ own marginal)``, reusing
``correlated_features.intervene``); ``Z`` and ``X2`` are untouched and ``y`` is redrawn from
the structural equation at the new ``x1``. Every arm is fitted on the observational context and
scored on both query sets: the model, a seeded untrained control, logistic regression and
histogram gradient boosting (the floors), and the closed-form ``oracle_do`` / ``oracle_obs``
rows. Reported per arm: AUC on observational and on interventional queries, the mean absolute
error of the predicted probability against the closed-form interventional and observational
probabilities (``mae_do``, ``mae_obs``), ``leans_to`` naming the closer truth, and the mean
prediction shift from observational to interventional queries beside the closed-form shifts.

This is verification against a synthetic SCM, not validation against observed data. Nothing
here trains. Numbers are MEASURED only when produced by the command in the openspec Progress
note on real checkpoints; the unit tests are SMOKE-TEST only.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from fintfm.evaluation.coverage import Coverage
from fintfm.experiments.correlated_features import intervene
from fintfm.experiments.irrelevant_features import UNTRAINED_SEED, _score_in_context

if TYPE_CHECKING:
    from fintfm.modeling.model import FinancialTFM

#: Strengths of the hidden ``Z->Y`` edge to sweep (0 is the task's SCM exactly).
CONFOUNDING: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0)

_GH_NODES, _GH_WEIGHTS = np.polynomial.hermite_e.hermegauss(40)
_GH_WEIGHTS = _GH_WEIGHTS / _GH_WEIGHTS.sum()


@dataclass(frozen=True)
class SCM:
    """Parameters of the SCM (see the module docstring).

    Attributes:
        a: Loading of ``X1`` on ``Z``, in ``[0, 1)``.
        b: Loading of ``X2`` on ``Z``, in ``[0, 1)``.
        w: Coefficient of ``X1`` in the outcome logit.
        g: Coefficient of the hidden ``Z`` in the outcome logit (0 = unconfounded).
        c: Outcome-logit intercept.
    """

    a: float = 0.8
    b: float = 0.8
    w: float = 1.5
    g: float = 0.0
    c: float = -0.5

    def __post_init__(self) -> None:
        """Reject loadings that make a node deterministic in ``Z``."""
        for name in ("a", "b"):
            if not 0.0 <= getattr(self, name) < 1.0:
                raise ValueError(f"{name} must be in [0, 1), got {getattr(self, name)}")


def _sigmoid(t: np.ndarray) -> np.ndarray:
    """Logistic function."""
    return 1.0 / (1.0 + np.exp(-t))


def sample_observational(scm: SCM, n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw ``n`` observational rows.

    Args:
        scm: The SCM.
        n: Rows.
        seed: Seed; identical arguments give identical output.

    Returns:
        ``(X, y, z)``: ``X`` is ``(n, 2)`` float32 ``[x1, x2]``, ``y`` is ``(n,)`` int64 drawn
        from the structural equation, ``z`` the hidden ``(n,)`` float64 latent.
    """
    rng = np.random.default_rng(np.random.SeedSequence([seed, 15]))
    z, e1, e2 = rng.normal(size=(3, n))
    u = rng.uniform(size=n)
    x1 = scm.a * z + np.sqrt(1 - scm.a**2) * e1
    x2 = scm.b * z + np.sqrt(1 - scm.b**2) * e2
    y = (u < _sigmoid(scm.w * x1 + scm.g * z + scm.c)).astype(np.int64)
    return np.stack([x1, x2], axis=1).astype(np.float32), y, z


def sample_interventional(
    scm: SCM, X: np.ndarray, z: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Apply ``do(X1 ~ its own marginal)`` to rows and redraw ``y`` from the structural equation.

    Only the mechanism of ``X1`` changes: ``Z`` and ``X2`` are the original rows', and ``y`` is
    redrawn with the unchanged outcome mechanism at the new ``x1``.

    Args:
        scm: The SCM.
        X: ``(n, 2)`` observational rows from :func:`sample_observational`; not modified.
        z: Their hidden latent.
        seed: Seed of the permutation and of the label draw.

    Returns:
        ``(X_do, y_do)``.
    """
    Xd = intervene(X, 0, seed)
    u = np.random.default_rng(np.random.SeedSequence([seed, 16])).uniform(size=len(X))
    y = (u < _sigmoid(scm.w * Xd[:, 0] + scm.g * z + scm.c)).astype(np.int64)
    return Xd, y


def _expect_sigmoid(scm: SCM, x1: np.ndarray, mean: np.ndarray, var: np.ndarray) -> np.ndarray:
    """``E[sigmoid(w x1 + g Z + c)]`` for ``Z ~ N(mean, var)`` by Gauss-Hermite quadrature."""
    zs = mean[:, None] + np.sqrt(var)[:, None] * _GH_NODES[None, :]
    return _sigmoid(scm.w * x1[:, None] + scm.g * zs + scm.c) @ _GH_WEIGHTS


def p_y_observational(scm: SCM, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """Closed-form ``P(Y=1 | X1=x1, X2=x2)``.

    Args:
        scm: The SCM.
        x1: ``(n,)`` values of ``X1``.
        x2: ``(n,)`` values of ``X2``.

    Returns:
        ``(n,)`` probabilities, with ``Z`` integrated over its posterior given both columns.
    """
    s1, s2 = 1 - scm.a**2, 1 - scm.b**2
    prec = 1 + scm.a**2 / s1 + scm.b**2 / s2
    mean = (scm.a * x1 / s1 + scm.b * x2 / s2) / prec
    return _expect_sigmoid(scm, x1, mean, np.full(len(x1), 1 / prec))


def p_y_interventional(scm: SCM, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """Closed-form ``P(Y=1 | do(X1=x1), X2=x2)``.

    Args:
        scm: The SCM.
        x1: ``(n,)`` values ``X1`` is set to.
        x2: ``(n,)`` observed values of ``X2`` (a non-descendant of ``X1``, so it still
            informs about ``Z``).

    Returns:
        ``(n,)`` probabilities, with ``Z`` integrated over its posterior given ``x2`` only.
    """
    s2 = 1 - scm.b**2
    prec = 1 + scm.b**2 / s2
    mean = (scm.b * x2 / s2) / prec
    return _expect_sigmoid(scm, x1, mean, np.full(len(x1), 1 / prec))


def p_y_do_marginal(scm: SCM, x1: np.ndarray) -> np.ndarray:
    """Closed-form ``P(Y=1 | do(X1=x1))`` with ``X2`` unobserved (``Z ~ N(0, 1)``).

    Args:
        scm: The SCM.
        x1: ``(n,)`` values ``X1`` is set to.

    Returns:
        ``(n,)`` probabilities.
    """
    return _expect_sigmoid(scm, x1, np.zeros(len(x1)), np.ones(len(x1)))


@dataclass
class Cell:
    """One arm at one confounding strength, averaged over seeds."""

    arm: str
    g: float
    auc_obs: float
    auc_do: float
    mae_do: float
    mae_obs: float
    leans_to: str
    shift_pred: float
    shift_do_true: float
    shift_obs_true: float
    n_seeds: int


def run(
    model_paths: dict[str, str],
    out_dir: Path | None = None,
    gs: tuple[float, ...] = CONFOUNDING,
    seeds: tuple[int, ...] = (0, 1, 2),
    n: int = 2000,
    max_context: int = 1000,
    a: float = 0.8,
    b: float = 0.8,
    w: float = 1.5,
    c: float = -0.5,
    models: dict[str, FinancialTFM] | None = None,
) -> dict:
    """Compare every arm against closed-form observational and interventional truth.

    Args:
        model_paths: ``{name: checkpoint path}``; one untrained control is built from the first
            one's architecture. Ignored when ``models`` is given.
        out_dir: If given, ``causal_interventional.json`` is written there.
        gs: Hidden ``Z->Y`` strengths to sweep.
        seeds: Seeds; everything is repeated per seed and averaged.
        n: Rows per task, half context and half query.
        max_context: Context rows handed to the in-context arms.
        a: Loading of ``X1`` on ``Z``.
        b: Loading of ``X2`` on ``Z``.
        w: Coefficient of ``X1`` in the outcome logit.
        c: Outcome-logit intercept.
        models: Already-built models by name (used by tests); the control is added.

    Returns:
        The record: config, per-cell results, and coverage.
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

    names = ("auc_obs", "auc_do", "mae_do", "mae_obs", "shift_pred", "shift_do", "shift_obs")
    acc: dict[tuple[str, float], dict[str, list[float]]] = {}
    cov = Coverage()
    nc = n // 2
    for g in gs:
        scm = SCM(a=a, b=b, w=w, g=g, c=c)
        for seed in seeds:
            X, y, z = sample_observational(scm, n, seed)
            Xtr, ytr, Xq, zq = X[:nc], y[:nc], X[nc:], z[nc:]
            Xd, yd = sample_interventional(scm, Xq, zq, seed + 1000)
            yq = y[nc:]
            unit = f"g{g}/seed{seed}"
            cov.attempt(unit)
            if min(len(np.unique(v)) for v in (ytr, yq, yd)) < 2:
                cov.skip(unit, "one class in the context, observational or interventional query")
                continue
            cov.scored(unit)
            x1d, x2 = Xd[:, 0].astype(np.float64), Xd[:, 1].astype(np.float64)
            truth_do = p_y_interventional(scm, x1d, x2)
            truth_obs = p_y_observational(scm, x1d, x2)
            truth_obs_q = p_y_observational(scm, Xq[:, 0].astype(np.float64), x2)
            lr = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
            gb = HistGradientBoostingClassifier(max_iter=100, random_state=seed).fit(Xtr, ytr)
            preds: dict[str, tuple[np.ndarray, np.ndarray]] = {
                name: (
                    _score_in_context(m, Xtr, ytr, Xq, max_context, seed),
                    _score_in_context(m, Xtr, ytr, Xd, max_context, seed),
                )
                for name, m in loaded.items()
            }
            preds["logistic_regression"] = (lr.predict_proba(Xq)[:, 1], lr.predict_proba(Xd)[:, 1])
            preds["gradient_boosting"] = (gb.predict_proba(Xq)[:, 1], gb.predict_proba(Xd)[:, 1])
            preds["oracle_do"] = (truth_obs_q, truth_do)
            preds["oracle_obs"] = (truth_obs_q, truth_obs)
            for arm, (p_obs, p_do) in preds.items():
                cell = acc.setdefault((arm, g), {k: [] for k in names})
                cell["auc_obs"].append(float(roc_auc_score(yq, p_obs)))
                cell["auc_do"].append(float(roc_auc_score(yd, p_do)))
                cell["mae_do"].append(float(np.mean(np.abs(p_do - truth_do))))
                cell["mae_obs"].append(float(np.mean(np.abs(p_do - truth_obs))))
                cell["shift_pred"].append(float(np.mean(p_do) - np.mean(p_obs)))
                cell["shift_do"].append(float(np.mean(truth_do) - np.mean(truth_obs_q)))
                cell["shift_obs"].append(float(np.mean(truth_obs) - np.mean(truth_obs_q)))
        print(f"  g={g} done", flush=True)

    cells = []
    for (arm, g), v in acc.items():
        m = {k: float(np.mean(x)) for k, x in v.items()}
        cells.append(
            Cell(
                arm=arm,
                g=g,
                auc_obs=m["auc_obs"],
                auc_do=m["auc_do"],
                mae_do=m["mae_do"],
                mae_obs=m["mae_obs"],
                leans_to="interventional" if m["mae_do"] < m["mae_obs"] else "observational",
                shift_pred=m["shift_pred"],
                shift_do_true=m["shift_do"],
                shift_obs_true=m["shift_obs"],
                n_seeds=len(v["auc_obs"]),
            )
        )
    record = {
        "task": "observational vs interventional queries on Z->X1, Z->X2, X1->Y",
        "config": {
            "models": model_paths,
            "gs": list(gs),
            "seeds": list(seeds),
            "n": n,
            "max_context": max_context,
            "a": a,
            "b": b,
            "w": w,
            "c": c,
        },
        "results": [asdict(x) for x in cells],
        "coverage": cov.to_dict(),
    }
    print(cov.summary("confounding-level-seeds"), flush=True)
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "causal_interventional.json").write_text(json.dumps(record, indent=2))
    return record


def summarise(record: dict) -> str:
    """Render, per confounding strength, one row per arm.

    Args:
        record: Output of :func:`run`.

    Returns:
        A printable multi-line string.
    """
    lines: list[str] = []
    for g in record["config"]["gs"]:
        sub = [r for r in record["results"] if r["g"] == g]
        if not sub:
            continue
        lines += [
            "",
            f"[g={g:g}] interventional queries; MAE is against the closed-form probability",
            (
                f"{'arm':>22} {'AUCobs':>7} {'AUCdo':>7} {'MAEdo':>7} {'MAEobs':>7} "
                f"{'shift':>7} {'trueDo':>7} {'trueObs':>7}  leans to"
            ),
        ]
        for r in sorted(sub, key=lambda r: r["arm"]):
            lines.append(
                f"{r['arm']:>22} {r['auc_obs']:7.3f} {r['auc_do']:7.3f} {r['mae_do']:7.3f} "
                f"{r['mae_obs']:7.3f} {r['shift_pred']:7.3f} {r['shift_do_true']:7.3f} "
                f"{r['shift_obs_true']:7.3f}  {r['leans_to']}"
            )
    lines += [
        "",
        "Reading it: 'shift' is the mean change in predicted P(Y=1) from observational to",
        "interventional queries; trueDo / trueObs are the closed-form shifts of the",
        "interventional and the associational truth. At g=0 the two truths coincide for X1, so",
        "a correct learner leans to neither. At g>0 an arm that leans 'observational' answers",
        "P(Y|x1,x2) where P(Y|do(x1),x2) was asked. The untrained control is the floor.",
        "This is verification against a synthetic SCM, not validation against observed data.",
    ]
    return "\n".join(lines)


def main() -> None:
    """Command-line entry point: ``python -m fintfm.experiments.causal_interventional``."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", type=str, required=True, help="name=path pairs, comma-separated")
    p.add_argument("--out", type=str, default="runs/causal_interventional")
    p.add_argument("--seeds", type=str, default="0,1,2")
    p.add_argument("--gs", type=str, default=",".join(str(v) for v in CONFOUNDING))
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--max-context", type=int, default=1000)
    args = p.parse_args()
    paths = dict(pair.split("=", 1) for pair in args.models.split(",") if pair)
    record = run(
        paths,
        Path(args.out),
        gs=tuple(float(v) for v in args.gs.split(",")),
        seeds=tuple(int(s) for s in args.seeds.split(",")),
        n=args.n,
        max_context=args.max_context,
    )
    print("\n" + summarise(record))


if __name__ == "__main__":
    main()
