"""Out-of-time evaluation of the PD term structure on real corporate panels.

**The first experiment in this project that tests the actual thesis on real data.**
Everything before it was either synthetic, or on panels that structurally could not carry
the claim: the UCI sets have no firm identifiers and no dates (``docs/FINDINGS.md`` §7), so
no per-firm hazard path could be scored and no split could cross a date.

V4FinBench changes that — 1,000,087 company-years over 188,338 companies, 2006-2020
(§22) — and this harness uses it the way a model-risk function would:

**Split by time, not at random.** Train on the early window, test on the later one, with no
year appearing in both. A random split lets a model see 2009 firms while predicting 2008
ones, which is the optimism a supervisory reviewer looks for first, and every number this
project has produced so far has that flaw.

**Score the whole curve.** AUC at each horizon, coherence violations, and calibration
against the observed rate at that horizon, because a monotone curve can still state the
wrong levels (§17).

**Compare against what the field does**: independent per-horizon models, which is the
construction §11 measured as incoherent 39% of the time.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from fintfm.evaluation.datasets import SurvivalDataset, load_v4finbench
from fintfm.evaluation.metrics import evaluate_binary
from fintfm.inference.classifier import FinancialTFMClassifier
from fintfm.modeling.model import FinancialTFM


@dataclass
class HorizonScore:
    """Scores at one horizon of the term structure."""

    horizon: int
    n_eval: int
    n_positive: int
    auc: float
    ece: float
    brier_skill: float
    observed_rate: float
    mean_predicted: float


@dataclass
class ArmScore:
    """One way of producing the term structure, scored out of time."""

    name: str
    horizons: list[HorizonScore] = field(default_factory=list)
    violation_rate: float = float("nan")
    fully_monotone: float = float("nan")

    @property
    def mean_auc(self) -> float:
        vals = [h.auc for h in self.horizons if np.isfinite(h.auc)]
        return float(np.mean(vals)) if vals else float("nan")


def time_split(
    ds: SurvivalDataset, train_until: int, test_from: int
) -> tuple[np.ndarray, np.ndarray]:
    """Indices for an out-of-time split, refusing any overlap.

    Args:
        ds: A dataset carrying ``year``.
        train_until: Last year (inclusive) allowed in training.
        test_from: First year (inclusive) allowed in test.

    Returns:
        ``(train_idx, test_idx)``.

    Raises:
        ValueError: If the dataset has no years, or the windows overlap. An overlapping
            "time split" is a random split wearing a costume, and would be reported as
            out-of-time validation while not being it.
    """
    if ds.year is None:
        raise ValueError(
            f"{ds.name} carries no year column, so it cannot support a time-based split; "
            "see docs/FINDINGS.md §7"
        )
    if test_from <= train_until:
        raise ValueError(
            f"windows overlap (train<={train_until}, test>={test_from}); an overlapping "
            "split is not out-of-time validation"
        )
    return np.flatnonzero(ds.year <= train_until), np.flatnonzero(ds.year >= test_from)


def _curve_truth(ds: SurvivalDataset, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative default indicator and observation mask per horizon."""
    period, observed = ds.period[idx], ds.n_observed[idx]
    K = ds.n_horizons
    truth = np.stack(
        [((period >= 0) & (period <= k)).astype(int) for k in range(K)], axis=1
    )
    # a horizon a row was never observed for is not evidence either way
    seen = np.stack([observed > k for k in range(K)], axis=1)
    return truth, seen


def score_curve(name: str, pd_curve: np.ndarray, truth: np.ndarray, seen: np.ndarray) -> ArmScore:
    """Score a cumulative-PD curve horizon by horizon, honouring the observation mask."""
    arm = ArmScore(name=name)
    for k in range(pd_curve.shape[1]):
        m = seen[:, k]
        if m.sum() < 100 or len(np.unique(truth[m, k])) < 2:
            arm.horizons.append(
                HorizonScore(k, int(m.sum()), int(truth[m, k].sum()), float("nan"),
                             float("nan"), float("nan"), float("nan"), float("nan"))
            )
            continue
        met = evaluate_binary(truth[m, k], pd_curve[m, k])
        arm.horizons.append(
            HorizonScore(
                horizon=k, n_eval=met.n, n_positive=met.n_positive, auc=met.roc_auc,
                ece=met.ece, brier_skill=met.brier_skill, observed_rate=met.base_rate,
                mean_predicted=met.mean_predicted,
            )
        )
    d = np.diff(pd_curve, axis=1)
    arm.violation_rate = float((d < 0).mean())
    arm.fully_monotone = float((d >= 0).all(axis=1).mean())
    return arm


def run(
    model_path: str,
    out_dir: Path,
    max_rows: int = 250_000,
    train_until: int = 2016,
    test_from: int = 2017,
    max_context: int = 2000,
    seed: int = 0,
) -> dict:
    """Score the hazard head and per-horizon baselines out of time on V4FinBench."""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    ds = load_v4finbench(max_rows=max_rows)
    tr, te = time_split(ds, train_until, test_from)
    print(f"{ds.name}: {ds.X.shape[0]:,} rows, {ds.X.shape[1]} features")
    print(f"  train {len(tr):,} rows (<= {train_until})   test {len(te):,} rows (>= {test_from})")
    print(f"  train default rate {ds.y[tr].mean():.3%}   test {ds.y[te].mean():.3%}")

    truth, seen = _curve_truth(ds, te)
    arms: list[ArmScore] = []

    model = FinancialTFM.load(model_path)
    if model.cfg.max_features < ds.X.shape[1]:
        raise ValueError(
            f"model takes {model.cfg.max_features} features, data has {ds.X.shape[1]}; "
            f"pretrain with --max-features {ds.X.shape[1]}"
        )

    # --- the hazard head, if this checkpoint has one -------------------------------
    if model.hazard is not None:
        clf = FinancialTFMClassifier(model, max_context=max_context, random_state=seed)
        clf.fit(ds.X[tr], ds.y[tr])
        ctx_X, ctx_y = clf._ctx_X, clf._ctx_y
        curves = []
        for start in range(0, len(te), clf.query_chunk):
            q = ds.X[te][start : start + clf.query_chunk]
            Xp = np.full((1, len(ctx_X) + len(q), model.cfg.max_features), np.nan, np.float32)
            Xp[0, : len(ctx_X), : ctx_X.shape[1]] = ctx_X
            Xp[0, len(ctx_X) :, : q.shape[1]] = q
            yp = np.zeros((1, len(ctx_X) + len(q)), dtype=np.int64)
            yp[0, : len(ctx_X)] = ctx_y
            with torch.no_grad():
                curves.append(
                    model.term_structure(torch.from_numpy(Xp), torch.from_numpy(yp), len(ctx_X))[0]
                    .cpu()
                    .numpy()
                )
        arms.append(score_curve("fintfm_hazard", np.concatenate(curves), truth, seen))

    # --- per-horizon logistic regression, the field's construction ------------------
    cols = []
    for k in range(ds.n_horizons):
        m_tr = ds.n_observed[tr] > k
        y_k = ((ds.period[tr] >= 0) & (ds.period[tr] <= k)).astype(int)[m_tr]
        if len(np.unique(y_k)) < 2:
            cols.append(np.full(len(te), np.nan))
            continue
        pipe = make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            LogisticRegression(max_iter=1000),
        ).fit(ds.X[tr][m_tr], y_k)
        cols.append(pipe.predict_proba(ds.X[te])[:, 1])
    arms.append(score_curve("per_horizon_logreg", np.stack(cols, axis=1), truth, seen))

    record = {
        "dataset": ds.name,
        "attribution": ds.attribution,
        "config": {
            "model": model_path, "max_rows": max_rows, "train_until": train_until,
            "test_from": test_from, "max_context": max_context, "seed": seed,
            "n_features": int(ds.X.shape[1]), "n_horizons": ds.n_horizons,
        },
        "split": {
            "n_train": len(tr), "n_test": len(te),
            "train_rate": float(ds.y[tr].mean()), "test_rate": float(ds.y[te].mean()),
        },
        "arms": [asdict(a) | {"mean_auc": a.mean_auc} for a in arms],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "v4_out_of_time.json").write_text(json.dumps(record, indent=2))
    return record


def summarise(record: dict) -> str:
    """Render the comparison, with coherence beside discrimination."""
    lines = [
        (
            f"{record['dataset']}  train<={record['config']['train_until']}  "
            f"test>={record['config']['test_from']}  "
            f"({record['split']['n_train']:,} / {record['split']['n_test']:,} rows)"
        ),
        "",
        f"{'arm':>22} {'mean AUC':>9} {'violations':>11} {'monotone':>9}",
    ]
    for a in record["arms"]:
        lines.append(
            f"{a['name']:>22} {a['mean_auc']:>9.4f} {a['violation_rate']:>11.2%} "
            f"{a['fully_monotone']:>9.1%}"
        )
    lines += ["", f"{'arm':>22} " + " ".join(f"{'h'+str(k):>8}" for k in range(6))]
    for a in record["arms"]:
        lines.append(f"{a['name']:>22} " + " ".join(f"{h['auc']:>8.4f}" for h in a["horizons"]))
    lines += ["", "calibration (ECE) by horizon:"]
    for a in record["arms"]:
        lines.append(f"{a['name']:>22} " + " ".join(f"{h['ece']:>8.4f}" for h in a["horizons"]))
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--out", type=str, default="runs/v4-out-of-time")
    p.add_argument("--max-rows", type=int, default=250_000)
    p.add_argument("--train-until", type=int, default=2016)
    p.add_argument("--test-from", type=int, default=2017)
    p.add_argument("--max-context", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    record = run(
        args.model, Path(args.out), max_rows=args.max_rows, train_until=args.train_until,
        test_from=args.test_from, max_context=args.max_context, seed=args.seed,
    )
    print("\n" + summarise(record))
    print(f"\n{record['attribution']}")


if __name__ == "__main__":
    main()
