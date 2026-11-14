"""Task 48.22: is fintfm's context-conditioned representation useful without its own head?

Every finding to date scores this project's own classification head. A published vectorizer
(``TabPfnVectorizer``; see ``docs/paper/RELATED_WORK.md``) treats a pretrained TFM
as a fixed feature extractor instead: hand its embeddings to a plain downstream model. That is
a different question, answerable only by comparing four arms on the same held-out rows:

  (a) fintfm's own head                             -- the number every other finding reports
  (b) a linear classifier on fintfm's representation -- does the embedding carry signal
  (c) the same linear classifier on raw features     -- the embedding's baseline to beat
  (d) a small MLP on fintfm's representation         -- Phase A' item 3: is a nonlinear final
                                                         layer worth having, on its own terms

(b) losing to (a) is expected; (b) losing to (c) is the result that would say the
representation carries nothing (c) did not already have. (d) is not framed as "the fix" --
``docs/results/FINDINGS.md`` §133 found the linear-probe advantage decays gradually through the
whole stack rather than at one identifiable layer, so there is no single broken component a
nonlinear head repairs. It is scored as its own hypothesis: whether a nonlinear read of the
frozen representation does better than a linear one, which is a question about the
representation's shape, not a diagnosis of the existing head.

**Why the downstream classifiers are fitted on a bounded subsample, not the full training
split.** A first version of this experiment called
:meth:`~fintfm.inference.classifier.FinancialTFMClassifier.transform_representation` on the
full ~600,000-row training split, on top of the ``predict_proba`` call every other finding
already pays for. §119 already measured this model's median predict time at 8.6 s/1,000 rows;
extending that cost to the training split as well as the test split, across five folds, prices
at roughly 14 hours and was killed partway through rather than left to finish proving a cost
that was knowable in advance. A downstream logistic regression does not need 600,000 training
rows to fit 128 coefficients, so :data:`TRAIN_SAMPLE_SIZE` bounds it, built to keep every
minority-class row rather than a plain random sample -- V4FinBench's positive rate is near
0.36%, and a random subsample at this budget would likely contain too few positives to fit a
meaningful classifier on.

**The equal-label-count control (design review DR3.4).** That sample holds every training
positive, while the head conditions on a 2,000-row uniform context holding only the positives
its draw happens to include. A probe that beats the head may therefore be reading more labels,
not a better representation. ``probe_rows="context"`` fits the three downstream arms on the
classifier's own context rows (:attr:`FinancialTFMClassifier.context_indices_`) instead, so
probe and head see the same labelled rows, and keeps the stratified-sample arms beside them
under a ``_stratified`` suffix for reference. ``"stratified"``, the default, is §130's
experiment, unchanged. Nothing here has measured the control yet.

**Why the context arm is cross-fitted.** A context row scored as a query attends to its own
labelled copy in the context -- through the row-to-row encoder at every stage past
``encode_rows``, and through cell attention at ``encode_rows`` on a ``cell_labels``
checkpoint -- so its representation can carry its label, and a probe fitted on such rows
would be inflated. ``"context"`` therefore splits the context rows into
:data:`CROSSFIT_FOLDS` folds and takes each fold's representations from the classifier
restricted to the other folds (:meth:`FinancialTFMClassifier.restrict_context`), so no row
is ever represented with its own label in view. The test rows are represented under each of
the same restricted contexts and the probe's probabilities averaged over them, so training
and test representations come from contexts of one size; the head is scored the same way
(``own_head_crossfit``: the restricted heads' probabilities averaged), so probe and head read
the same labels with no self-label in either. ``"context-leaky"`` fits on the context rows'
representations under the full context, label in view: a diagnostic of the leak's size, not
a result. The stratified sample has the same leak for the context rows it includes, which
are few (all the context's positives, since the sample keeps every positive).

**Probe plus partitioned contexts (``partition=True``, 2026-11-09).** The fintfm arm becomes a
:class:`~fintfm.inference.large_n.PartitionedTFMClassifier` at its default settings (``docs/results/FINDINGS.md``
§169), and the probe is fitted per leaf: in each bag, a leaf with its own estimator gets a
logistic regression on that estimator's representation of a stratified sample of the leaf's
training rows (every positive, padded with negatives to the leaf's share of
``train_sample_size``, at least :data:`LEAF_MIN_SAMPLE`), and scores the test rows routed to
that leaf; test rows in a fallback leaf take the global probe's probability. Bags are averaged,
as the wrapper averages its heads. ``own_head`` is then the partitioned head (§169's arm), and
``own_head_global`` and ``representation_global`` are the global estimator's head and probe,
which equal the default run's ``own_head`` and ``representation`` (§131's pair). The
TabArena fork's ``FinTFM-LinProbe`` adapter does the same on Lite (§215). The stratified leak
above applies per leaf: a leaf sample holds every positive of the leaf's context.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from fintfm.evaluation.datasets import CACHE_DIR
from fintfm.experiments.v4_protocol import (
    HORIZON_FILES,
    _load_horizon,
    build_fold_assignments,
    company_codes,
    split_indices_for_fold,
)
from fintfm.inference.classifier import FinancialTFMClassifier
from fintfm.inference.large_n import PartitionedTFMClassifier
from fintfm.inference.preprocess import FeatureConditioner

if TYPE_CHECKING:  # the model class is only needed for the annotation
    from fintfm.modeling.model import FinancialTFM

Arm = Literal["own_head", "representation", "raw_features", "mlp_head"]

#: Which training rows the downstream arms are fitted on: §130's stratified sample, the
#: classifier's own context rows cross-fitted (DR3.4), or those rows with their labels in view,
#: for diagnosis only. See the module docstring.
ProbeRows = Literal["stratified", "context", "context-leaky"]

#: Folds the context arm cross-fits over: each fold is represented against the other 4/5.
CROSSFIT_FOLDS = 5

Stage = Literal["encode_rows", "encoder_mid", "pre_head"]

#: Rows given to the downstream classifiers' *training* step, per fold. Every minority-class
#: row from the fold's training split is kept, then padded with a random majority-class sample
#: up to this budget -- see the module docstring for why a plain random subsample is unsafe at
#: V4FinBench's positive rate. The held-out test split is never subsampled: every arm, including
#: this experiment's own_head arm, is scored on the same full test rows every other finding uses.
TRAIN_SAMPLE_SIZE = 20_000

#: Floor on a leaf probe's sample under ``partition=True``: a leaf's proportional share of
#: :data:`TRAIN_SAMPLE_SIZE` can be too small to fit a probe on, so it never drops below this
#: (or the leaf's whole training set, where that is smaller). Matches the TabArena fork's
#: ``FinTFM-LinProbe`` adapter.
LEAF_MIN_SAMPLE = 2_000


def _stratified_subsample(rng: np.random.Generator, y: np.ndarray, budget: int) -> np.ndarray:
    """Indices keeping every positive row, padded with a random negative sample to ``budget``.

    Returns all rows when the fold's training split is already at or under budget. When the
    positives alone fill the budget, which happens in a default-rich partition leaf, both
    classes are capped instead: negatives take up to half the budget and positives fill the
    rest. Keeping every positive there left no negatives, so the leaf's probe could not be
    fitted and the leaf fell back to its own head on another scale (FINDINGS §219).
    """
    if len(y) <= budget:
        return np.arange(len(y))
    positive = np.where(y == 1)[0]
    negative = np.where(y == 0)[0]
    if len(positive) >= budget:
        n_negative = min(len(negative), budget // 2)
        negative = rng.choice(negative, size=n_negative, replace=False)
        positive = rng.choice(positive, size=budget - n_negative, replace=False)
        return np.concatenate([positive, negative])
    n_negative = max(budget - len(positive), 0)
    if n_negative < len(negative):
        negative = rng.choice(negative, size=n_negative, replace=False)
    return np.concatenate([positive, negative])


def _downstream_arms(
    clf: FinancialTFMClassifier,
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    X_test: np.ndarray,
    stage: Stage,
    seed: int,
) -> dict[str, np.ndarray]:
    """Fit the three downstream arms on one set of training rows and score the test rows.

    Args:
        clf: The fitted classifier whose representation the probe and the MLP read.
        X_fit: Training rows for the downstream models, scaled as ``clf``'s were.
        y_fit: Their labels, 0/1.
        X_test: The held-out rows every arm is scored on.
        stage: The representation depth; see
            :meth:`FinancialTFMClassifier.transform_representation`.
        seed: The MLP's ``random_state`` (the fold index).

    Returns:
        Test-row positive-class probabilities for ``"representation"``, ``"raw_features"``
        and ``"mlp_head"``. All NaN when ``y_fit`` holds one class, which no classifier can
        be fitted on, so the fold records a gap rather than the run raising.
    """
    if np.unique(y_fit).size < 2:
        return _gap(len(X_test))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rep_tr = clf.transform_representation(X_fit, stage=stage)
        rep_te = clf.transform_representation(X_test, stage=stage)
        return _fit_arms(rep_tr, X_fit, y_fit, [rep_te], X_test, seed)


def _leaf_probe(
    part: PartitionedTFMClassifier,
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    X_test: np.ndarray,
    p_global: np.ndarray,
    stage: Stage,
    rng: np.random.Generator,
    train_sample_size: int,
    probe_options: dict | None = None,
    raw_conditioner: FeatureConditioner | None = None,
) -> tuple[np.ndarray, dict[str, int], np.ndarray | None]:
    """A linear probe per partition leaf, scored on the test rows each leaf receives.

    Args:
        part: The fitted partitioned classifier.
        X_fit: ``(n, F)`` every training row ``part`` was fitted on, scaled.
        y_fit: ``(n,)`` their labels, 0/1.
        X_test: ``(m, F)`` the held-out rows.
        p_global: ``(m,)`` the global probe's test probabilities, used for rows routed to a
            leaf with no estimator of its own.
        stage: The representation depth, as in :func:`_downstream_arms`.
        rng: Draws each leaf's stratified sample.
        train_sample_size: The whole-split probe budget each leaf takes its share of.
        probe_options: When given, a second probe per leaf is fitted on the same rows and
            representations with these readout options (:func:`_options_probe`), so the
            plain probe and its draws are unchanged.
        raw_conditioner: The squash for the ``raw`` option, fitted on the training rows.

    Returns:
        Test-row positive-class probabilities averaged over bags; counts: leaf probes
        fitted, leaves falling back to the global probe, and leaves whose sample held one
        class and so were scored by the leaf's own head instead; and the options probe's
        probabilities (``None`` without ``probe_options``). Where the plain probe falls back
        to the global probe or the head, the options probe takes the same value.
    """
    routes_fit, routes_test = part.route(X_fit), part.route(X_test)
    out = np.zeros(len(X_test), dtype=np.float64)
    out_opts = np.zeros(len(X_test), dtype=np.float64) if probe_options else None
    counts = {"n_leaf_probes": 0, "n_fallback_leaves": 0, "n_single_class_leaves": 0}
    for b, leaves in enumerate(part.leaf_estimators_):
        bag = part.bag_rows_[b]
        for leaf in np.unique(routes_test[b]):
            rows = np.flatnonzero(routes_test[b] == leaf)
            est = leaves.get(int(leaf))
            if est is None:
                out[rows] += p_global[rows]
                if out_opts is not None:
                    out_opts[rows] += p_global[rows]
                counts["n_fallback_leaves"] += 1
                continue
            members = bag[routes_fit[b][bag] == leaf]
            budget = max(LEAF_MIN_SAMPLE, round(train_sample_size * len(members) / len(bag)))
            sub = members[_stratified_subsample(rng, y_fit[members], budget)]
            if np.unique(y_fit[sub]).size < 2:
                p_head = est.predict_proba(X_test[rows])[:, 1]
                out[rows] += p_head
                if out_opts is not None:
                    out_opts[rows] += p_head
                counts["n_single_class_leaves"] += 1
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rep_fit = est.transform_representation(X_fit[sub], stage=stage)
                rep_test = est.transform_representation(X_test[rows], stage=stage)
                probe = LogisticRegression(max_iter=1000).fit(rep_fit, y_fit[sub])
                if out_opts is not None:
                    out_opts[rows] += _options_probe(
                        est, rep_fit, X_fit[sub], y_fit[sub], rep_test, X_test[rows],
                        probe_options, raw_conditioner,
                    )
            out[rows] += probe.predict_proba(rep_test)[:, 1]
            counts["n_leaf_probes"] += 1
    n_bags = len(part.leaf_estimators_)
    return out / n_bags, counts, (None if out_opts is None else out_opts / n_bags)


#: Keys ``--probe-options`` accepts; see :func:`_options_probe`.
PROBE_OPTIONS = frozenset({"scale", "cv", "raw", "head", "no_rep"})


def _options_probe(
    est: FinancialTFMClassifier,
    rep_fit: np.ndarray,
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    rep_test: np.ndarray,
    X_test: np.ndarray,
    options: dict,
    raw_conditioner: FeatureConditioner | None,
) -> np.ndarray:
    """Fit a probe with readout options on given representations and score the test rows.

    The V4FinBench counterpart of the TabArena fork's ``FINTFM_PROBE`` (fintfm DR7.6,
    FINDINGS §222/§223), so a binary readout change is scored here before TabArena.

    Args:
        est: The leaf's fitted classifier, read for the ``head`` option.
        rep_fit: ``(m, d)`` the probe rows' representations.
        X_fit: ``(m, F)`` the same rows' features, as the classifier received them.
        y_fit: ``(m,)`` their labels, both classes present.
        rep_test: ``(n, d)`` the test rows' representations.
        X_test: ``(n, F)`` the test rows' features.
        options: ``scale`` standardises the inputs; ``cv`` picks C from 1e-3 to 1e2 by 3-fold
            log loss; ``raw`` appends the squashed features (``raw_conditioner``), missing at
            the median; ``head`` appends the log of the classifier's own probabilities;
            ``no_rep`` drops the representation, so with ``raw`` the probe reads the squashed
            features alone on the same leaves and rows (the matched raw-only control, §230).
        raw_conditioner: Fitted squash for ``raw``.

    Returns:
        ``(n,)`` positive-class probabilities.
    """
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.pipeline import make_pipeline

    def inputs(rep: np.ndarray, X: np.ndarray) -> np.ndarray:
        """The probe's input: the representation plus the blocks the options add."""
        blocks = [] if options.get("no_rep") else [np.asarray(rep, dtype=np.float64)]
        if options.get("raw"):
            z = raw_conditioner.transform(X)
            blocks.append(np.nan_to_num(z, nan=0.0, posinf=3.0, neginf=-3.0).astype(np.float64))
        if options.get("head"):
            proba = np.asarray(est.predict_proba(X), dtype=np.float64)
            blocks.append(np.log(np.clip(proba, 1e-6, 1.0)))
        return blocks[0] if len(blocks) == 1 else np.hstack(blocks)

    if options.get("cv"):
        lr = LogisticRegressionCV(
            Cs=np.logspace(-3, 2, 6), cv=3, scoring="neg_log_loss", max_iter=2000
        )
    else:
        lr = LogisticRegression(max_iter=1000)
    probe = make_pipeline(StandardScaler(), lr) if options.get("scale") else lr
    probe.fit(inputs(rep_fit, X_fit), y_fit)
    return probe.predict_proba(inputs(rep_test, X_test))[:, 1]


def _gap(n: int) -> dict[str, np.ndarray]:
    """NaN for every downstream arm: the record of a probe set no classifier can fit."""
    return {name: np.full(n, np.nan) for name in ("representation", "raw_features", "mlp_head")}


def _fit_arms(
    rep_fit: np.ndarray,
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    rep_tests: list[np.ndarray],
    X_test: np.ndarray,
    seed: int,
) -> dict[str, np.ndarray]:
    """Fit the probe, the raw-feature model and the MLP, and score the test rows.

    Args:
        rep_fit: ``(m, d)`` representations of the training rows.
        X_fit: ``(m, F)`` the same rows' scaled features, for the raw-feature arm.
        y_fit: ``(m,)`` their labels, 0/1, both classes present.
        rep_tests: The test rows' representations, once per context they were taken under;
            a probe's probabilities are averaged over them (one entry is used as it is).
        X_test: ``(n, F)`` the test rows' scaled features.
        seed: The MLP's ``random_state``.

    Returns:
        Test-row positive-class probabilities for the three arms.
    """

    def averaged(model: LogisticRegression | MLPClassifier) -> np.ndarray:
        """The model's test probabilities, averaged over ``rep_tests``."""
        if len(rep_tests) == 1:
            return model.predict_proba(rep_tests[0])[:, 1]
        return np.mean([model.predict_proba(r)[:, 1] for r in rep_tests], axis=0)

    lr_rep = LogisticRegression(max_iter=1000).fit(rep_fit, y_fit)
    p_representation = averaged(lr_rep)

    lr_raw = LogisticRegression(max_iter=1000).fit(X_fit, y_fit)
    p_raw_features = lr_raw.predict_proba(X_test)[:, 1]

    mlp_rep = MLPClassifier(
        hidden_layer_sizes=(64,),
        early_stopping=True,
        n_iter_no_change=15,
        max_iter=500,
        random_state=seed,
    ).fit(rep_fit, y_fit)
    p_mlp_head = averaged(mlp_rep)
    return {
        "representation": p_representation,
        "raw_features": p_raw_features,
        "mlp_head": p_mlp_head,
    }


def crossfit_representations(
    clf: FinancialTFMClassifier,
    X_train: np.ndarray,
    stage: Stage,
    seed: int,
    k: int = CROSSFIT_FOLDS,
) -> tuple[np.ndarray, list[FinancialTFMClassifier]]:
    """Represent each context row against a context that leaves it out.

    The context rows are shuffled at ``seed`` and split into ``k`` folds; fold ``j``'s rows
    are represented by ``clf`` restricted to the other folds, so none is scored with its own
    label in view.

    Args:
        clf: The fitted classifier; its :attr:`~FinancialTFMClassifier.context_indices_` are
            the rows cross-fitted.
        X_train: The rows ``clf`` was fitted on, as passed to its ``fit``.
        stage: The representation depth.
        seed: Seed for the fold split.
        k: Folds, at least 2.

    Returns:
        ``(m, d)`` representations of the context rows, in ``context_indices_`` order, and the
        ``k`` restricted classifiers, each holding ``(k - 1)/k`` of the context.

    Raises:
        ValueError: If ``k < 2`` or the context has fewer than ``k`` rows.
    """
    ctx = clf.context_indices_
    if k < 2 or ctx.size < k:
        raise ValueError(f"cross-fitting needs k >= 2 and at least k context rows, got k={k}")
    parts = np.array_split(np.random.default_rng(seed).permutation(ctx.size), k)
    restricted = [clf.restrict_context(np.delete(ctx, part)) for part in parts]
    rep: np.ndarray | None = None
    for part, sub_clf in zip(parts, restricted, strict=True):
        r = sub_clf.transform_representation(X_train[ctx[part]], stage=stage)
        if rep is None:
            rep = np.empty((ctx.size, r.shape[1]), dtype=r.dtype)
        rep[part] = r
    assert rep is not None
    return rep, restricted


def _crossfit_arms(
    clf: FinancialTFMClassifier,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    stage: Stage,
    seed: int,
) -> dict[str, np.ndarray]:
    """The context arm: cross-fitted probe, raw-feature model and MLP, and the head beside them.

    Args:
        clf: The fitted classifier.
        X_train: The rows ``clf`` was fitted on.
        y_train: Their labels.
        X_test: The held-out rows.
        stage: The representation depth.
        seed: Seed for the fold split and the MLP.

    Returns:
        The three downstream arms (NaN when the context holds one class) and
        ``"own_head_crossfit"``, the head's test probabilities averaged over the same
        restricted contexts the probe's test representations were taken under.
    """
    ctx = clf.context_indices_
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rep_fit, restricted = crossfit_representations(clf, X_train, stage, seed)
        head = np.mean([c.predict_proba(X_test)[:, 1] for c in restricted], axis=0)
        if np.unique(y_train[ctx]).size < 2:
            arms = _gap(len(X_test))
        else:
            rep_tests = [c.transform_representation(X_test, stage=stage) for c in restricted]
            arms = _fit_arms(rep_fit, X_train[ctx], y_train[ctx], rep_tests, X_test, seed)
    return {**arms, "own_head_crossfit": head}


def _scores(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """ROC-AUC and average precision; NaN for an arm :func:`_downstream_arms` could not fit."""
    if np.isnan(p).any():
        return float("nan"), float("nan")
    return float(roc_auc_score(y, p)), float(average_precision_score(y, p))


def run(
    model_path: FinancialTFM | str,
    out_path: Path,
    horizon: int = 0,
    folds: tuple[int, ...] = (0, 1, 2, 3, 4),
    max_rows: int | None = None,
    root: Path | None = None,
    device: str = "cpu",
    train_sample_size: int = TRAIN_SAMPLE_SIZE,
    predictions_dir: Path | None = None,
    stage: Stage = "encode_rows",
    probe_rows: ProbeRows = "stratified",
    max_context: int | None = None,
    partition: bool = False,
    probe_options: dict | None = None,
) -> dict:
    """Score the three arms on V4FinBench's published protocol. See module docstring.

    Args:
        model_path: A classification checkpoint, matching :func:`v4_protocol.run`'s contract,
            or a loaded :class:`~fintfm.modeling.model.FinancialTFM`.
        out_path: Where the result dictionary is written, as JSON.
        horizon: Paper horizon, 0-5.
        folds: Which rotations to run.
        max_rows: Development cap; companies are kept whole.
        root: Directory holding the parquet files.
        device: Torch device for the fintfm arm.
        train_sample_size: Budget for the downstream classifiers' training rows; see
            :data:`TRAIN_SAMPLE_SIZE`.
        stage: Which representation to extract, passed through to
            :meth:`FinancialTFMClassifier.transform_representation`. ``"encode_rows"`` is
            §130's stage; ``"encoder_mid"`` is partway through the row-to-row encoder
            (§131's open question, Phase A' item 2); ``"pre_head"`` is what the
            classification head actually reads.
        predictions_dir: When given, per-row test predictions for every arm are written here
            as ``predictions_fold{N}.npz``, matching :func:`v4_protocol.run`'s convention --
            two arms scored on the same rows have correlated errors, so a paired bootstrap
            needs the raw vectors rather than the point estimates alone
            (``docs/results/FINDINGS.md`` §60).
        probe_rows: ``"stratified"`` (the default, §130 unchanged) fits the downstream arms
            on the stratified sample; ``"context"`` fits them on the classifier's context
            rows, cross-fitted (see the module docstring), adds ``own_head_crossfit``, and
            records the stratified arms beside them with a ``_stratified`` suffix (design
            review DR3.4); ``"context-leaky"`` is the same without cross-fitting, every probe
            row's label in view, for measuring the leak only.
        max_context: The classifier's context size; ``None`` keeps the classifier's default
            of 2,000, the size §130 ran at. A development knob for a small table.
        partition: Wrap the classifier in partitioned contexts and fit the probe per leaf
            (see the module docstring); only with ``probe_rows="stratified"``.

    Returns:
        The recorded result dictionary, also written to ``out_path``.

    Raises:
        ValueError: For an unknown ``probe_rows``, or ``partition`` with any other than
            ``"stratified"``.
    """
    if probe_rows not in ("stratified", "context", "context-leaky"):
        raise ValueError(
            "probe_rows must be 'stratified', 'context' or 'context-leaky', got "
            f"{probe_rows!r}"
        )
    if partition and probe_rows != "stratified":
        raise ValueError("partition=True supports probe_rows='stratified' only")
    clf_kwargs = {} if max_context is None else {"max_context": max_context}
    root = root or (CACHE_DIR / "v4finbench")
    if predictions_dir is not None:
        predictions_dir.mkdir(parents=True, exist_ok=True)
    X, y, country, company, _names = _load_horizon(root, horizon, max_rows)
    print(
        f"horizon {horizon} ({HORIZON_FILES[horizon]}): {len(y):,} rows, {X.shape[1]} "
        f"features, {int(y.sum()):,} positive ({y.mean():.3%})",
        flush=True,
    )
    assignments = build_fold_assignments(country, company)

    results: list[dict] = []
    for fold in folds:
        rng = np.random.default_rng(fold)
        tr, _va, te = split_indices_for_fold(assignments, fold)
        imputer = SimpleImputer(strategy="median").fit(X[tr])
        scaler = StandardScaler().fit(imputer.transform(X[tr]))
        Xtr_full, Xte = (scaler.transform(imputer.transform(X[i])) for i in (tr, te))
        Xtr_full, Xte = Xtr_full.astype(np.float32), Xte.astype(np.float32)
        ytr_full, yte = y[tr], y[te]

        # The full training split is fintfm's in-context evidence -- it decides nothing about
        # cost, since fit() only stores and subsamples it to max_context. The subsample below
        # is what bounds the downstream classifiers' training cost; the two are independent.
        clf = FinancialTFMClassifier(model_path, device=device, random_state=fold, **clf_kwargs)
        part = None
        if partition:
            # the global estimator is fitted exactly as `clf` would be (same settings and
            # seed), so the global arms below reproduce the default run's own_head and
            # representation; §169's wrapper settings are the class defaults
            part = PartitionedTFMClassifier(clf, random_state=fold).fit(Xtr_full, ytr_full)
            clf = part.global_estimator_
        else:
            clf.fit(Xtr_full, ytr_full)

        sub = _stratified_subsample(rng, ytr_full, train_sample_size)
        Xtr, ytr = Xtr_full[sub], ytr_full[sub]

        p_own_head = clf.predict_proba(Xte)[:, 1]

        stratified = _downstream_arms(clf, Xtr, ytr, Xte, stage, fold)
        arms: dict[str, np.ndarray] = {"own_head": p_own_head}
        row = {
            "fold": int(fold),
            "n_train_full": len(tr),
            "n_train_sample": len(sub),
            "n_test": len(te),
        }
        if part is not None:
            cond = None
            if probe_options and probe_options.get("raw"):
                cond = FeatureConditioner("squash", random_state=fold).fit(Xtr_full)
            p_leaf, leaf_counts, p_leaf_opts = _leaf_probe(
                part, Xtr_full, ytr_full, Xte, stratified["representation"], stage,
                np.random.default_rng(1000 + fold), train_sample_size,
                probe_options, cond,
            )
            arms = {
                "own_head": part.predict_proba(Xte)[:, 1],
                "representation": p_leaf,
                "own_head_global": p_own_head,
                "representation_global": stratified["representation"],
                "raw_features": stratified["raw_features"],
                "mlp_head": stratified["mlp_head"],
            }
            if p_leaf_opts is not None:
                arms["representation_opts"] = p_leaf_opts
            row.update(leaf_counts)
        elif probe_rows != "stratified":
            # the rows the head conditions on, so the probe reads exactly as many labels
            ctx = clf.context_indices_
            if probe_rows == "context":
                arms.update(_crossfit_arms(clf, Xtr_full, ytr_full, Xte, stage, fold))
            else:
                arms.update(
                    _downstream_arms(clf, Xtr_full[ctx], ytr_full[ctx], Xte, stage, fold)
                )
            arms.update({f"{name}_stratified": p for name, p in stratified.items()})
            row["n_probe_rows"] = len(ctx)
            row["n_probe_positive"] = int(ytr_full[ctx].sum())
            # the stratified arm's own leak: sample rows that are also context rows
            row["n_sample_context_rows"] = int(np.isin(sub, ctx).sum())
        else:
            arms.update(stratified)
        for name, p in arms.items():
            row[f"{name}_auc"], row[f"{name}_ap"] = _scores(yte, p)
        results.append(row)
        probe_note = (
            f"probe on {row['n_probe_rows']:,} context rows, {row['n_probe_positive']} "
            "positive; "
            if probe_rows != "stratified"
            else ""
        )
        partition_note = (
            f"  (partitioned: {row['n_leaf_probes']} leaf probes, "
            f"{row['n_fallback_leaves']} fallback, {row['n_single_class_leaves']} single-class; "
            f"global head AP {row['own_head_global_ap']:.4f}, "
            f"global probe AP {row['representation_global_ap']:.4f}"
            + (
                f", options probe AP {row['representation_opts_ap']:.4f}"
                if "representation_opts_ap" in row
                else ""
            )
            + ")"
            if part is not None
            else ""
        )
        stratified_note = (
            f"  representation_stratified AP {row['representation_stratified_ap']:.4f}"
            if probe_rows != "stratified"
            else ""
        ) + (
            f"  own_head_crossfit AP {row['own_head_crossfit_ap']:.4f}"
            if probe_rows == "context"
            else ""
        )
        print(
            f"  fold {fold}  ({probe_note}train sample {len(sub):,} of {len(tr):,})  "
            f"own_head AP {row['own_head_ap']:.4f}  "
            f"representation AP {row['representation_ap']:.4f}  "
            f"mlp_head AP {row['mlp_head_ap']:.4f}  "
            f"raw_features AP {row['raw_features_ap']:.4f}{stratified_note}{partition_note}",
            flush=True,
        )
        if predictions_dir is not None:
            # company per test row, as v4_protocol saves it, so scripts/compare_v4_scores.py
            # can resample whole companies; added with the partition arm only, so a default
            # run still writes the file §130's code wrote
            extra = {"company_test": company_codes(company)[te]} if part is not None else {}
            np.savez_compressed(
                predictions_dir / f"predictions_fold{fold}.npz",
                y_true=yte,
                **{f"pred_{name}": p for name, p in arms.items()},
                **extra,
            )

    record = {
        "horizon": horizon,
        "n_rows": len(y),
        "train_sample_size": train_sample_size,
        "folds": results,
    }
    if probe_rows != "stratified":
        # recorded only off the default, so a default run writes the JSON §130's code wrote
        record["probe_rows"] = probe_rows
    if partition:
        record["partition"] = True
    if probe_options:
        record["probe_options"] = probe_options
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import json

    out_path.write_text(json.dumps(record, indent=2))
    return record


def main() -> None:
    """CLI entry point."""
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--out", type=Path, default=Path("representation_transfer.json"))
    p.add_argument("--horizon", type=int, default=0, choices=sorted(HORIZON_FILES))
    p.add_argument("--folds", type=str, default="0,1,2,3,4")
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--train-sample-size", type=int, default=TRAIN_SAMPLE_SIZE)
    p.add_argument("--predictions-dir", type=Path, default=None)
    p.add_argument(
        "--stage", choices=("encode_rows", "encoder_mid", "pre_head"), default="encode_rows"
    )
    p.add_argument(
        "--probe-rows", choices=("stratified", "context", "context-leaky"),
        default="stratified",
        help="rows the downstream arms are fitted on: the stratified sample (default, as "
             "§130 ran); the classifier's own context rows, cross-fitted over 5 folds so no "
             "row sees its own label, with the head scored the same way (own_head_crossfit) "
             "and the stratified arms kept under a _stratified suffix (design review DR3.4); "
             "or context-leaky, the context rows with their labels in view, a diagnostic of "
             "the leak only",
    )
    p.add_argument(
        "--partition", action="store_true",
        help="wrap the classifier in partitioned contexts (FINDINGS §169, default settings) "
             "and fit the probe per leaf; own_head becomes the partitioned head, and the "
             "global head and probe are kept as own_head_global and representation_global",
    )
    p.add_argument(
        "--probe-options", type=str, default=None,
        help="with --partition: a JSON object of readout options (scale, cv, raw, head, no_rep) for a "
             "second leaf probe, written as the representation_opts arm beside the plain one",
    )
    args = p.parse_args()
    probe_options = None
    if args.probe_options:
        import json

        probe_options = json.loads(args.probe_options)
        unknown = set(probe_options) - PROBE_OPTIONS
        if not isinstance(probe_options, dict) or unknown:
            p.error(f"--probe-options: unknown key(s) {sorted(unknown)}; known {sorted(PROBE_OPTIONS)}")
        if not args.partition:
            p.error("--probe-options needs --partition")
        if probe_options.get("no_rep") and not (probe_options.get("raw") or probe_options.get("head")):
            p.error("--probe-options: no_rep needs raw or head, or the probe has no input")
    run(
        args.model,
        args.out,
        horizon=args.horizon,
        folds=tuple(int(f) for f in args.folds.split(",")),
        max_rows=args.max_rows,
        device=args.device,
        train_sample_size=args.train_sample_size,
        predictions_dir=args.predictions_dir,
        stage=args.stage,
        probe_rows=args.probe_rows,
        partition=args.partition,
        probe_options=probe_options,
    )


if __name__ == "__main__":
    main()
