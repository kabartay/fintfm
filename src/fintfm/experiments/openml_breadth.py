"""Is fintfm broadly weak, or specifically weak on low-prevalence credit panels?

Every real-data number this project has is from corporate-default data: V4FinBench, Polish
and Taiwan bankruptcy. All three are credit, and two of the three are severely imbalanced. So
the ~0.22 AP deficit to tuned gradient boosting (``docs/results/FINDINGS.md`` §80, §93) has never been
separated from the *kind of data* it was measured on.

This module scores the same checkpoint across a spread of public OpenML binary tasks chosen to
vary **class prevalence** from ~2% to ~50% at roughly comparable width, so the deficit can be
plotted against prevalence rather than quoted as one number. Two readings are possible and they
imply different work:

* the deficit is flat in prevalence -> fintfm is broadly behind, and the credit framing is
  incidental;
* the deficit shrinks as prevalence rises -> fintfm's problem is imbalance, which is a prior
  and inference-time question rather than a general capability gap.

**Evaluation only.** No OpenML data trains anything here; `CLAUDE.md`'s licensing boundary
bans third-party *training* data and explicitly permits evaluation. Datasets are fetched from
OpenML (predominantly CC-BY) and cached outside version control.
"""

from __future__ import annotations

import argparse
import os
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

#: Binary-classification OpenML tasks spanning prevalence, all within the 136-feature cap the
#: current checkpoints carry. Chosen for prevalence spread rather than for difficulty, so the
#: suite deliberately includes tasks a booster solves trivially.
#:
#: No TabArena-v0.1 or BeyondArena dataset may appear here (design review task DR4.1): arms are
#: chosen on this suite, so an overlap biases the eventual TabArena number. bank-marketing,
#: churn, blood-transfusion-service-center, credit-g, qsar-biodeg and diabetes were removed on
#: 2026-10-16 for that reason; ``tests/test_openml_breadth_overlap.py`` keeps them out.
SUITE: tuple[tuple[str, int], ...] = (
    ("mammography", 1),
    ("wilt", 1),
    ("ozone-level-8hr", 1),
    ("pc4", 1),
    ("kc1", 1),
    ("MagicTelescope", 1),
    ("phoneme", 1),
    ("spambase", 1),
    ("banknote-authentication", 1),
)


@dataclass
class DatasetResult:
    """Scores for one dataset, with the properties needed to interpret them."""

    name: str
    n_rows: int
    n_features: int
    prevalence: float
    scores: dict[str, float] = field(default_factory=dict)

    @property
    def deficit(self) -> float:
        """fintfm's average precision minus the best baseline's, or NaN if unscorable.

        Negative means fintfm is behind. Reported against the *best* baseline rather than a
        chosen one, so the number cannot be flattered by picking a weak comparator -- the
        error §69 made and §60 had to correct.
        """
        if "fintfm" not in self.scores:
            return float("nan")
        others = [v for k, v in self.scores.items() if k != "fintfm" and not k.endswith("@auc")]
        return self.scores["fintfm"] - max(others) if others else float("nan")


def load_task(name: str, version: int, max_rows: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Fetch one OpenML task as numeric features and a binary 0/1 target.

    Args:
        name: OpenML dataset name.
        version: OpenML dataset version, pinned so a silent upstream revision cannot change
            what this suite measures.
        max_rows: Stratified subsample cap, applied because several tasks run to 100k rows and
            this is a diagnostic rather than a leaderboard entry.

    Returns:
        ``(X, y)`` with ``X`` float32 and ``y`` int64 in ``{0, 1}``, the minority class mapped
        to 1 so average precision is computed on the rare class. ``None`` if the task cannot be
        loaded or is not binary.
    """
    from sklearn.datasets import fetch_openml

    os.environ.setdefault("SSL_CERT_FILE", _certifi_path())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            data = fetch_openml(name, version=version, as_frame=True, parser="auto")
        except (OSError, ValueError) as exc:
            # A missing dataset, a renamed version or a network failure should cost this
            # suite one row, not the whole run -- but the reason is printed rather than
            # swallowed, since "SKIPPED" with no cause is how a benchmark quietly measures
            # fewer datasets than it claims (CLAUDE.md, on distrusting a skip count).
            print(f"    {name}: fetch failed ({type(exc).__name__}: {exc})", flush=True)
            return None
    frame = data.data.select_dtypes(include=[np.number])
    if frame.shape[1] == 0:
        return None
    labels = np.asarray(data.target)
    classes, counts = np.unique(labels, return_counts=True)
    if len(classes) != 2:
        return None
    minority = classes[int(np.argmin(counts))]
    y = (labels == minority).astype(np.int64)
    X = frame.to_numpy(dtype=np.float32)
    if len(y) > max_rows:
        rng = np.random.default_rng(0)
        pos = np.flatnonzero(y == 1)
        neg = np.flatnonzero(y == 0)
        keep_pos = rng.choice(pos, min(len(pos), max(1, int(max_rows * y.mean()))), replace=False)
        keep_neg = rng.choice(neg, max_rows - len(keep_pos), replace=False)
        idx = np.sort(np.concatenate([keep_pos, keep_neg]))
        X, y = X[idx], y[idx]
    return X, y


def _certifi_path() -> str:
    """Certifi's CA bundle, because the python.org macOS build ships without root certs.

    Same failure and the same reasoning as ``evaluation/datasets._download``; that path solves
    it with a curl fallback, which is unavailable here since ``fetch_openml`` owns its own
    transport.
    """
    import certifi

    return certifi.where()


def score_task(
    X: np.ndarray, y: np.ndarray, model_path: str, seed: int = 0
) -> dict[str, float]:
    """Average precision for fintfm and baselines on one stratified split.

    A single 70/30 split rather than five folds: this suite exists to compare a *pattern*
    across fifteen datasets, and per-dataset precision matters less than coverage. Any headline
    number that comes out of it must be re-measured at the five-fold paired-bootstrap standard
    the credit panels use (§84).

    Args:
        X: Numeric features.
        y: Binary target with the minority class as 1.
        model_path: fintfm checkpoint. Skipped, not failed, when the task is wider than the
            checkpoint's ``max_features``.
        seed: Split seed.

    Returns:
        ``"<arm>"`` to average precision and ``"<arm>@auc"`` to ROC-AUC. Both are recorded
        because this project reads AP first at low prevalence (``docs/design/DECISIONS.md`` D13)
        while TabArena and TabBench report ROC-AUC, and a deficit in one does not convert to
        the other. fintfm is absent when the checkpoint cannot take the task's width.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    from fintfm.inference.classifier import FinancialTFMClassifier
    from fintfm.modeling.model import FinancialTFM

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )
    imp = SimpleImputer(strategy="median").fit(Xtr)
    Xtr_i, Xte_i = imp.transform(Xtr), imp.transform(Xte)
    out: dict[str, float] = {}

    scaler = StandardScaler().fit(Xtr_i)
    lr = LogisticRegression(max_iter=2000).fit(scaler.transform(Xtr_i), ytr)
    p_lr = lr.predict_proba(scaler.transform(Xte_i))[:, 1]
    out["logreg"] = float(average_precision_score(yte, p_lr))
    out["logreg@auc"] = float(roc_auc_score(yte, p_lr))
    hgb = HistGradientBoostingClassifier(random_state=seed).fit(Xtr_i, ytr)
    p_hgb = hgb.predict_proba(Xte_i)[:, 1]
    out["hgb"] = float(average_precision_score(yte, p_hgb))
    out["hgb@auc"] = float(roc_auc_score(yte, p_hgb))

    model = FinancialTFM.load(model_path)
    if model.cfg.max_features >= X.shape[1]:
        clf = FinancialTFMClassifier(
            model, max_context=1000, n_ensemble=8, feature_chunk=16, random_state=seed
        ).fit(Xtr.astype(np.float32), ytr)
        p_ft = clf.predict_proba(Xte.astype(np.float32))[:, 1]
        out["fintfm"] = float(average_precision_score(yte, p_ft))
        out["fintfm@auc"] = float(roc_auc_score(yte, p_ft))
    return out


def run(
    model_path: str,
    max_rows: int = 20_000,
    seed: int = 0,
    suite: Sequence[tuple[str, int]] | None = None,
    allow_tabarena_for_reporting: bool = False,
) -> list[DatasetResult]:
    """Score the suite and print a prevalence-ordered table.

    Args:
        model_path: fintfm checkpoint to evaluate.
        max_rows: Stratified subsample cap per dataset.
        seed: Split seed.
        suite: ``(name, version)`` pairs; :data:`SUITE` when None.
        allow_tabarena_for_reporting: Permit a dataset in
            :data:`fintfm.evaluation.overlap.SELECTION_FORBIDDEN`, for reporting only; the
            suite chooses arms, so without this such a dataset is refused (DR4.1).

    Returns:
        One :class:`DatasetResult` per dataset that loaded, ordered by prevalence.

    Raises:
        SelectionForbiddenError: If the suite holds a forbidden dataset and the override is
            not given.
    """
    from fintfm.evaluation.overlap import guard_selection

    suite = SUITE if suite is None else suite
    reporting = guard_selection([n for n, _ in suite], allow_tabarena_for_reporting,
                                "openml_breadth")
    if reporting:
        print(f"REPORTING ONLY ({', '.join(reporting)}): not for choosing an arm (DR4.1)")
    results: list[DatasetResult] = []
    for name, version in suite:
        task = load_task(name, version, max_rows)
        if task is None:
            print(f"  {name:<34} SKIPPED (unavailable or not binary)", flush=True)
            continue
        X, y = task
        scores = score_task(X, y, model_path, seed=seed)
        r = DatasetResult(name, X.shape[0], X.shape[1], float(y.mean()), scores)
        results.append(r)
        cells = "  ".join(f"{k}={v:.4f}" for k, v in sorted(scores.items()))
        print(
            f"  {name:<34} n={X.shape[0]:>6} f={X.shape[1]:>3} prev={y.mean():6.2%}  "
            f"{cells}  deficit={r.deficit:+.4f}",
            flush=True,
        )
    return sorted(results, key=lambda r: r.prevalence)


def suite_features(name: str, version: int) -> Any:
    """One SUITE dataset's features with the target dropped, for the content checks (DR4.1).

    Through scikit-learn's default OpenML cache, where :func:`load_task` puts the same
    datasets, so a dataset scored here is the one checked. The ``--row-hash`` check and the
    overlap report (:mod:`fintfm.evaluation.overlap_report`) both read SUITE through this.

    Args:
        name: OpenML dataset name, as in SUITE.
        version: Its pinned OpenML version.

    Returns:
        The features as a DataFrame, or None when the fetch fails (the reason is printed).
    """
    from sklearn.datasets import fetch_openml

    from fintfm.evaluation.proxy_suite import ensure_ca_bundle

    ensure_ca_bundle()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return fetch_openml(name, version=version, as_frame=True, parser="auto").data
        except (OSError, ValueError) as exc:
            print(f"  {name}: fetch failed ({type(exc).__name__}: {exc})")
            return None


def _row_hash(reference_dir: Path) -> int:
    """Compare each SUITE dataset's contents with the cached TabArena-v0.1 datasets (DR4.1).

    The same check, thresholds and reference cache as the proxy suite's ``--row-hash``
    (:func:`fintfm.evaluation.overlap.content_fingerprint`); datasets come from
    scikit-learn's default OpenML cache, where :func:`load_task` puts them.

    Returns:
        0 when every dataset is clean or the check was skipped (printed), 1 on an overlap.
    """
    from fintfm.evaluation.overlap import load_reference_fingerprints, row_hash_report

    references, notes = load_reference_fingerprints(reference_dir)
    for note in notes:
        print(note)
    if not references:
        print("SKIPPED: no TabArena-v0.1 dataset available locally; nothing was compared")
        return 0

    def loader(name: str, version: int) -> Callable[[], Any]:
        """Defer :func:`suite_features` for one SUITE dataset until the report reaches it."""
        return lambda: suite_features(name, version)

    hits = row_hash_report([(n, loader(n, v)) for n, v in SUITE], references)
    print(f"\n{len(hits)} of {len(SUITE)} SUITE datasets overlap a cached TabArena-v0.1 dataset"
          + (f": {', '.join(hits)}" if hits else ""))
    return 1 if hits else 0


def main(argv: Sequence[str] | None = None) -> int:
    """Command line: score the suite, or check it against TabArena by contents.

    Args:
        argv: Command-line arguments; ``sys.argv[1:]`` when omitted.

    Returns:
        Process exit status.
    """
    from fintfm.evaluation.overlap import REPORTING_FLAG, TABARENA_OPENML_CACHE

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default=None, help="fintfm checkpoint to score")
    p.add_argument("--max-rows", type=int, default=20_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(REPORTING_FLAG, action="store_true",
                   help="permit a selection-forbidden dataset, for reporting only")
    p.add_argument("--row-hash", action="store_true",
                   help="compare SUITE's contents with the cached TabArena-v0.1 datasets")
    p.add_argument("--tabarena-cache", type=Path, default=TABARENA_OPENML_CACHE)
    args = p.parse_args(argv)
    if args.row_hash:
        return _row_hash(args.tabarena_cache)
    if args.model is None:
        p.error("--model is required unless --row-hash is given")
    run(args.model, args.max_rows, args.seed,
        allow_tabarena_for_reporting=args.allow_tabarena_for_reporting)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
