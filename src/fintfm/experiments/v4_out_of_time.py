"""Out-of-time evaluation of the PD term structure on real corporate panels.

**The first experiment in this project that tests the actual thesis on real data.**
Everything before it was either synthetic, or on panels that structurally could not carry
the claim: the UCI sets have no firm identifiers and no dates (``docs/results/FINDINGS.md`` §7), so
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
construction §229 measured as incoherent on 40% of horizon steps out of time.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from fintfm.config import Config, load_config
from fintfm.evaluation.datasets import SurvivalDataset, load_v4finbench
from fintfm.evaluation.metrics import evaluate_binary
from fintfm.evaluation.splits import time_based_split
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
    pred_sd: float = float("nan")
    pred_p99_over_p50: float = float("nan")


@dataclass
class ArmScore:
    """One way of producing the term structure, scored out of time."""

    name: str
    horizons: list[HorizonScore] = field(default_factory=list)
    violation_rate: float = float("nan")
    fully_monotone: float = float("nan")
    n_comparable_pairs: int = 0
    n_nan_pairs: int = 0
    n_rows_compared: int = 0

    @property
    def mean_auc(self) -> float:
        """Mean ROC-AUC across horizons, ignoring any that could not be scored.

        Returns:
            The mean, or NaN when no horizon was scorable. A degenerate single-class split is
            skipped rather than counted as zero, which would drag the mean toward a number
            describing the split rather than the model.
        """
        vals = [h.auc for h in self.horizons if np.isfinite(h.auc)]
        return float(np.mean(vals)) if vals else float("nan")


SplitMode = Literal["embargoed", "legacy"]


@dataclass(frozen=True)
class OutOfTimeSplit:
    """An out-of-time split of a survival dataset and what it took to make it leak-free.

    Attributes:
        train: Training row indices.
        test: Test row indices.
        mode: ``"embargoed"`` (default) or ``"legacy"``.
        train_until: The last training year the caller asked for.
        effective_train_until: The last training year actually used, after the embargo.
        embargo_years: Years removed before ``test_from`` so no training label window reaches
            into the test period (0 in legacy mode).
        n_shared_ids: Companies present in both windows before exclusion (``-1`` in legacy
            mode, which does not look).
        n_train_dropped: Training rows removed because their company is in the test window.
        n_shared_ids_after: Companies on both sides of the returned split, measured after
            the fact in either mode, so a legacy record states its own leakage (``-1`` when
            the dataset carries no company identifiers).
    """

    train: np.ndarray
    test: np.ndarray
    mode: str
    train_until: int
    effective_train_until: int
    embargo_years: int
    n_shared_ids: int
    n_train_dropped: int
    n_shared_ids_after: int

    def record(self) -> dict:
        """Return the split's bookkeeping as a JSON-ready dictionary.

        Returns:
            Everything except the index arrays, with the row counts added.
        """
        return {
            "mode": self.mode,
            "train_until": self.train_until,
            "effective_train_until": self.effective_train_until,
            "embargo_years": self.embargo_years,
            "n_shared_ids_before": self.n_shared_ids,
            "n_train_dropped_for_shared_ids": self.n_train_dropped,
            "n_shared_ids_after": self.n_shared_ids_after,
        }


def split_dataset(
    ds: SurvivalDataset,
    train_until: int,
    test_from: int,
    split: SplitMode = "embargoed",
    embargo_years: int | None = None,
) -> OutOfTimeSplit:
    """Split a survival dataset by year with no company on both sides and a label embargo.

    **Embargo.** V4FinBench's horizon files ``h1..h6`` give, for a company-year ``y``, the
    default indicator within 1..6 years after ``y`` (``n_horizons`` = 6, see
    :func:`fintfm.evaluation.datasets.load_v4finbench`). A training row therefore carries
    label information up to year ``y + n_horizons``. For none of that to fall inside the test
    period we need ``y + n_horizons < test_from``, i.e. the last usable training year is
    ``test_from - 1 - n_horizons``. The embargo defaults to ``ds.n_horizons`` years, the
    longest horizon any label spans. Rows with fewer observed horizons (the panel's last
    years) reach no further and are excluded by the same bound.

    **Companies.** :func:`fintfm.evaluation.splits.time_based_split` is called with
    ``on_overlap="drop_from_train"``: a company seen in the test window loses its training
    rows, the test window is never altered, and the count is recorded.

    Args:
        ds: A dataset carrying ``year`` (and ``groups`` unless ``split="legacy"``).
        train_until: Last year (inclusive) the caller would allow in training.
        test_from: First year (inclusive) of the test window.
        split: ``"embargoed"`` (default, corrected) or ``"legacy"``. ``"legacy"`` reproduces
            the split used by FINDINGS §28-§35: windows only, no company check, no embargo.
            It leaks, and exists only to reproduce those entries.
        embargo_years: Override the embargo (years). ``None`` uses ``ds.n_horizons``.

    Returns:
        An :class:`OutOfTimeSplit`.

    Raises:
        ValueError: If the dataset has no years, ``split`` is unknown, the windows overlap,
            the embargo leaves no training years, or (embargoed mode) the dataset carries no
            company identifiers.
    """
    if split not in ("embargoed", "legacy"):
        raise ValueError(f"unknown split {split!r}; use 'embargoed' or 'legacy'")
    if ds.year is None:
        raise ValueError(
            f"{ds.name} carries no year column, so it cannot support a time-based split; "
            "see docs/results/FINDINGS.md §7"
        )
    if test_from <= train_until:
        raise ValueError(
            f"windows overlap (train<={train_until}, test>={test_from}); an overlapping "
            "split is not out-of-time validation"
        )
    if split == "legacy":
        tr = np.flatnonzero(ds.year <= train_until)
        te = np.flatnonzero(ds.year >= test_from)
        leaked = (
            len(np.intersect1d(ds.groups[tr], ds.groups[te])) if ds.groups is not None else -1
        )
        return OutOfTimeSplit(tr, te, "legacy", train_until, train_until, 0, -1, 0, leaked)
    if ds.groups is None:
        raise ValueError(
            f"{ds.name} carries no company identifiers, so the company check cannot run; "
            "pass split='legacy' to reproduce the unchecked split"
        )
    embargo = ds.n_horizons if embargo_years is None else int(embargo_years)
    if embargo < 0:
        raise ValueError(f"embargo_years must be >= 0, got {embargo}")
    effective = min(train_until, test_from - 1 - embargo)
    ts = time_based_split(
        ds.year, ds.groups, effective, test_from, on_overlap="drop_from_train"
    )
    leaked = len(np.intersect1d(ds.groups[ts.train], ds.groups[ts.test]))
    return OutOfTimeSplit(
        ts.train, ts.test, "embargoed", train_until, effective, embargo,
        ts.n_shared_ids, ts.n_train_dropped, leaked,
    )


def time_split(
    ds: SurvivalDataset,
    train_until: int,
    test_from: int,
    split: SplitMode = "embargoed",
    embargo_years: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Indices for an out-of-time split; see :func:`split_dataset` for the rules.

    Args:
        ds: A dataset carrying ``year``.
        train_until: Last year (inclusive) allowed in training.
        test_from: First year (inclusive) allowed in test.
        split: ``"embargoed"`` (default) or ``"legacy"`` (the old, leaking split).
        embargo_years: Embargo override; ``None`` uses ``ds.n_horizons``.

    Returns:
        ``(train_idx, test_idx)``.

    Raises:
        ValueError: As :func:`split_dataset`.
    """
    sp = split_dataset(ds, train_until, test_from, split, embargo_years)
    return sp.train, sp.test


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


def score_curve(
    name: str,
    pd_curve: np.ndarray,
    truth: np.ndarray,
    seen: np.ndarray,
    min_rows: int | None = None,
) -> ArmScore:
    """Score a cumulative-PD curve horizon by horizon, honouring the observation mask.

    Args:
        name: Arm name.
        pd_curve: ``(n, K)`` cumulative PD.
        truth: ``(n, K)`` cumulative default indicator.
        seen: ``(n, K)`` observation mask.
        min_rows: Minimum observed rows for a horizon to be scored at all; below it the
            horizon is reported as NaN rather than scored, because a handful of rows produces
            a number that looks like a result. A horizon whose predictions contain NaN (no
            model could be fitted because it was single-class in training) is also NaN.
            Defaults to
            ``evaluation.min_rows_per_horizon`` from the configuration.

    Returns:
        The scored arm. ``violation_rate`` and ``fully_monotone`` use only comparable pairs
        of adjacent horizons (both finite); ``n_comparable_pairs``, ``n_nan_pairs`` and
        ``n_rows_compared`` record how many that was, and both rates are NaN when none.
    """
    if min_rows is None:
        min_rows = load_config().evaluation.min_rows_per_horizon
    arm = ArmScore(name=name)
    for k in range(pd_curve.shape[1]):
        m = seen[:, k]
        if (
            m.sum() < min_rows
            or len(np.unique(truth[m, k])) < 2
            or not np.isfinite(pd_curve[m, k]).all()
        ):
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
                mean_predicted=met.mean_predicted, pred_sd=met.pred_sd,
                pred_p99_over_p50=met.pred_p99_over_p50,
            )
        )
    # A NaN comparison is neither a violation nor a pass. Count only pairs of adjacent
    # horizons where both predictions exist, and record how many that was.
    d = np.diff(pd_curve, axis=1)
    comparable = np.isfinite(d)
    arm.n_comparable_pairs = int(comparable.sum())
    arm.n_nan_pairs = int(d.size - comparable.sum())
    if arm.n_comparable_pairs:
        bad = (d < 0) & comparable
        arm.violation_rate = float(bad.sum() / arm.n_comparable_pairs)
        rows = comparable.any(axis=1)
        arm.n_rows_compared = int(rows.sum())
        arm.fully_monotone = float((~bad.any(axis=1))[rows].mean())
    return arm


def run(
    model_path: str,
    out_dir: Path,
    max_rows: int | None = None,
    train_until: int | None = None,
    test_from: int | None = None,
    max_context: int | None = None,
    seed: int = 0,
    cfg: Config | None = None,
    split: SplitMode = "embargoed",
) -> dict:
    """Score the hazard head and per-horizon baselines out of time on V4FinBench.

    Args:
        model_path: Checkpoint carrying a hazard head.
        out_dir: Directory for ``v4_out_of_time.json``.
        max_rows: Row cap for the loader. Defaults to ``v4finbench.max_rows``.
        train_until: Last training year, inclusive. Defaults to ``v4finbench.train_until``.
        test_from: First test year, inclusive. Defaults to ``v4finbench.test_from``.
        max_context: Context cap. Defaults to ``v4finbench.max_context``.
        seed: Seed for context selection.
        cfg: Configuration; loaded from the packaged default when omitted. Every value it
            supplies is written into the run record, so a result can be traced to the
            settings that produced it.
        split: ``"embargoed"`` (default) or ``"legacy"``; see :func:`split_dataset`. Legacy
            reproduces FINDINGS §28-§35 and leaks companies and label windows.

    Returns:
        The recorded result dictionary.
    """
    cfg = cfg or load_config()
    max_rows = cfg.v4finbench.max_rows if max_rows is None else max_rows
    train_until = cfg.v4finbench.train_until if train_until is None else train_until
    test_from = cfg.v4finbench.test_from if test_from is None else test_from
    max_context = cfg.v4finbench.max_context if max_context is None else max_context
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    ds = load_v4finbench(max_rows=max_rows)
    sp = split_dataset(ds, train_until, test_from, split)
    tr, te = sp.train, sp.test
    print(f"{ds.name}: {ds.X.shape[0]:,} rows, {ds.X.shape[1]} features")
    print(f"  split={sp.mode}: embargo {sp.embargo_years}y, companies on both sides "
          f"{sp.n_shared_ids_after}, train rows dropped for shared companies "
          f"{sp.n_train_dropped}")
    print(f"  train {len(tr):,} rows (<= {sp.effective_train_until})   test {len(te):,} rows (>= {test_from})")
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
    # Three arms, not one. The first out-of-time run scored only a balanced, *uncorrected*
    # context and reported a 12.8% mean PD against a 0.47% truth; that was misread as the
    # prior failing to reach low default rates, and a 6,000-step retrain was spent on it
    # before the cause turned out to be the missing correction (docs/results/FINDINGS.md §28). The
    # uncorrected arm stays in the harness permanently so the distortion is measured beside
    # the fix rather than argued about.
    if model.hazard is not None:
        for arm in cfg.v4finbench.hazard_arms:
            # every other setting from cfg.inference (DR1.3); the arm sets its own two
            clf = FinancialTFMClassifier.from_config(
                model,
                cfg.inference,
                max_context=max_context,
                context_strategy=arm.strategy,
                correct_prior=arm.correct_prior,
                random_state=seed,
            ).fit(ds.X[tr], ds.y[tr], periods=ds.period[tr])
            curve = clf.predict_term_structure(ds.X[te])
            summary = clf.context_summary_
            rate = (
                summary.pooled_context_rate
                if arm.strategy == "retrieval"
                else summary.context_rate
            )
            print(
                f"  {arm.name}: context at {rate:.3%} vs population "
                f"{summary.population_rate:.3%}"
            )
            arms.append(
                score_curve(
                    arm.name, curve, truth, seen, cfg.evaluation.min_rows_per_horizon
                )
            )

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
    arms.append(
        score_curve(
            "per_horizon_logreg",
            np.stack(cols, axis=1),
            truth,
            seen,
            cfg.evaluation.min_rows_per_horizon,
        )
    )

    record = {
        "dataset": ds.name,
        "attribution": ds.attribution,
        "config": {
            "model": model_path, "max_rows": max_rows, "train_until": train_until,
            "test_from": test_from, "max_context": max_context, "seed": seed,
            "n_features": int(ds.X.shape[1]), "n_horizons": ds.n_horizons,
            "split": split,
            "feature_transform": cfg.inference.feature_transform,
            "retrieval_groups": cfg.inference.retrieval_groups,
            "min_rows_per_horizon": cfg.evaluation.min_rows_per_horizon,
            "config_sources": list(cfg.sources),
        },
        "split": {
            "n_train": len(tr), "n_test": len(te),
            "train_rate": float(ds.y[tr].mean()), "test_rate": float(ds.y[te].mean()),
            **sp.record(),
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
        (
            f"{'arm':>26} {'mean AUC':>9} {'violations':>11} {'monotone':>9} "
            f"{'pairs':>9} {'NaN':>7}"
        ),
    ]
    for a in record["arms"]:
        lines.append(
            f"{a['name']:>26} {a['mean_auc']:>9.4f} {a['violation_rate']:>11.2%} "
            f"{a['fully_monotone']:>9.1%} {a.get('n_comparable_pairs', -1):>9} "
            f"{a.get('n_nan_pairs', -1):>7}"
        )
    n_h = len(record["arms"][0]["horizons"])
    lines += ["", f"{'arm':>26} " + " ".join(f"{'h'+str(k):>8}" for k in range(n_h))]
    for a in record["arms"]:
        lines.append(f"{a['name']:>26} " + " ".join(f"{h['auc']:>8.4f}" for h in a["horizons"]))
    lines += ["", "calibration (ECE) by horizon:"]
    for a in record["arms"]:
        lines.append(f"{a['name']:>26} " + " ".join(f"{h['ece']:>8.4f}" for h in a["horizons"]))
    # spread beside every calibration number: a low ECE with p99/p50 near 1 is conservatism
    lines += ["", "prediction spread (p99/p50) by horizon:"]
    for a in record["arms"]:
        lines.append(
            f"{a['name']:>26} "
            + " ".join(f"{h.get('pred_p99_over_p50', float('nan')):>8.2f}" for h in a["horizons"])
        )
    # The level, printed beside the truth. A term structure can be perfectly monotone and
    # perfectly ranked while stating a default rate 27x too high, and AUC and the violation
    # rate both report that curve as healthy (docs/results/FINDINGS.md §28).
    obs = record["arms"][0]["horizons"]
    lines += ["", "mean predicted PD by horizon (observed in the last row):"]
    for a in record["arms"]:
        lines.append(
            f"{a['name']:>26} " + " ".join(f"{h['mean_predicted']:>8.4f}" for h in a["horizons"])
        )
    lines.append(f"{'OBSERVED':>26} " + " ".join(f"{h['observed_rate']:>8.4f}" for h in obs))
    return "\n".join(lines)


def main() -> None:
    """Run the out-of-time split, which the published protocol is not.

    Entry point for the ``v4-out-of-time`` console script; see
    ``--help`` for the flags. Writes its record as JSON under ``--out``.
    """
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--out", type=str, default="runs/v4-out-of-time")
    p.add_argument("--config", type=str, default=None, help="YAML overriding the defaults")
    # every value below defaults to None so the configuration supplies it; a flag given
    # explicitly wins, which keeps the layering (default file, override file, flag) honest
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--train-until", type=int, default=None)
    p.add_argument("--test-from", type=int, default=None)
    p.add_argument("--max-context", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--split", choices=("embargoed", "legacy"), default="embargoed",
        help="'legacy' reproduces the leaking split of FINDINGS 28-35 (no company check, "
        "no embargo)",
    )
    args = p.parse_args()
    cfg = load_config(args.config)
    record = run(
        args.model, Path(args.out), max_rows=args.max_rows, train_until=args.train_until,
        test_from=args.test_from, max_context=args.max_context, seed=args.seed, cfg=cfg,
        split=args.split,
    )
    print(f"\nconfig: {cfg.provenance()}")
    print("\n" + summarise(record))
    print(f"\n{record['attribution']}")


if __name__ == "__main__":
    main()
