"""The linear readout: a fintfm network's pre-head representation, decided by a linear probe.

:class:`LinearReadoutModel` is an sklearn-style estimator. It owns everything after the feature
matrix: categorical target encoding, loading the network, the base ``FinancialTFMClassifier`` or
``FinancialTFMRegressor`` with its partition, the minority flip, and the readout itself (probes,
splines, one-hot indicators, cross-fitting, probability floor, ridge or random Fourier features,
target transform, clip). The measured entry is :func:`fintfm.readout.entry.entry_defaults`.
"""

from __future__ import annotations

import copy
import logging
import math
from collections.abc import Sequence
from typing import Any, ClassVar

import numpy as np

from fintfm.readout.checkpoints import resolve_checkpoint, validate_checkpoint_ref
from fintfm.readout.entry import (
    LEAF_MIN_SAMPLE,
    PARTITION_DEPTH_CAP,
    PARTITION_LIBRARY_MAX_DEPTH,
    PARTITION_SCALE_K,
    PROB_FLOOR_GRID,
    PROB_FLOOR_MAX_ROWS,
    PROBE_CROSSFIT_FOLDS,
    PROBLEM_TYPES,
    TRAIN_SAMPLE_SIZE,
    WIDE_RIDGE_ALPHAS,
    check_bags_max_rows,
    check_onehot,
    check_prob_floor,
    check_probe_budget,
    check_reg_alphas,
    check_reg_blend,
    check_reg_clip_margin,
    check_reg_readout,
    check_reg_rff_max_rows,
    check_reg_transform,
    check_scale_k,
    check_spline,
    check_spline_max_rows,
    validated_partition,
    validated_probe,
)
from fintfm.readout.target_transform import TargetTransform, smearing_factor

logger = logging.getLogger(__name__)


class LinearReadoutModel:
    """fintfm, decided by a linear probe on its representation rather than its own head.

    ``fit`` takes the float32 feature matrix (NaN kept for missing values; categorical columns as
    integer codes, with their positions in ``categorical_features``) and the target, and stores the
    table as the network's in-context evidence. ``predict_proba`` (classification) and ``predict``
    (regression, or the arg-max class) score new rows of the same layout. The estimator encodes the
    categorical columns as out-of-fold smoothed target statistics, loads the checkpoint, and fits the
    readout. ``time_limit`` and validation splits do not exist here: nothing is trained against a time
    budget or selected on a validation split, and the readout chooses its regularisation, cross-fit
    folds and multiclass floor by internal cross-validation on the training rows only.

    Settings are keyword arguments and are all optional; the measured entry is
    :func:`fintfm.readout.entry.entry_defaults`. A setting that is not named below is passed on to
    ``FinancialTFMClassifier`` (``n_ensemble``, ``max_context``, ``feature_chunk``,
    ``feature_transform``, ``max_features_strategy``, ...).

    - ``n_bins``: the regression bin count, clamped to the checkpoint's ``max_classes``.
    - ``partition``: ``None`` (no partition) or a dict of ``PartitionedTFMClassifier`` arguments
      (``PartitionedTFMRegressor`` for regression); see :func:`fintfm.readout.entry.validated_partition`.
    - ``probe``: the readout options dict (``scale``, ``cv``, ``raw``, ``head``, ``no_rep``); see
      :func:`fintfm.readout.entry.validated_probe`.
    - ``minority_positive`` (bool, default ``False``): binary only. Code the training rows so the
      rarer class is 1 when class 1 holds more than half the rows, and swap the head's probability
      columns back (the financial prior always made class 1 the rarer one, at most 30% of rows).
    - ``reg_thresholds`` (int or ``None``): regression only. K binary classifiers on ``y > t_k``
      instead of one classifier over the head's bins (``ThresholdTFMRegressor``); not combined with a
      partition.

    **Cross-fitting (``probe_crossfit_folds``, entry default 5 for binary, 0 for multiclass and
    regression).** The rows the readout trains on are also rows of the stored in-context set, so a plain
    ``transform_representation`` call lets a context row attend to its own labelled copy, which a test
    row never can. With ``probe_crossfit_folds`` of K >= 2, the context rows among the readout's training
    rows are split into K folds and each fold is represented by the classifier restricted
    (``FinancialTFMClassifier.restrict_context``) to the other K - 1 folds; probe rows that are not in
    the context are represented under the full context, as test rows are. The probe rows' categorical
    encoding is then the out-of-fold encoding the classifier was fitted on, not the full-statistics
    encoding applied in-sample. ``0`` or ``1`` disables both and reproduces the earlier readout exactly.
    The ``head`` probe option is not cross-fitted and is refused while cross-fitting is on; threshold
    regression is not cross-fitted and keeps the earlier readout with a warning.

    **Gated levers (the entry sets each for one problem type; off for the others).**

    - ``probe_onehot`` (``None``, ``True`` or ``{"min_count", "max_columns", "max_levels"}``; entry
      default for multiclass ``{"min_count": 3, "max_columns": 256, "max_levels": 64}``, off for
      binary and regression): adds one-hot indicator columns of the categorical features to the
      multiclass readout, next to the level-frequency encoding. A level needs ``min_count`` training
      rows; at most ``max_columns`` indicators are kept (most frequent first); a categorical column with
      more than ``max_levels`` distinct levels gets none. Ignored with a warning outside a multiclass
      fit whose probe has the ``raw`` option.
    - ``reg_readout`` (``"ridge"`` or ``"rff"``; entry default ``"rff"`` for regression) and
      ``reg_rff_max_rows`` (``None`` or an int >= 1; entry default 5,000 for regression): ``"rff"``
      replaces the linear ridge by ridge on random Fourier features
      (:class:`~fintfm.readout.rff.RFFRidgeCV`, bandwidth and alpha chosen by inner K-fold on the
      training rows). A fit with more than ``reg_rff_max_rows`` training rows uses the plain ridge,
      decided once per fit. Classification ignores both.
    - ``reg_clip_margin`` (``None`` or a float >= 0; entry default 0.1 for regression): every
      regression prediction is clipped to the training target's range widened by this fraction of it on
      each side, ``[min - m * (max - min), max + m * (max - min)]``. The ridge on spline features of
      near-collinear columns can predict far off the training manifold for a few rows (diamonds: -19
      million against prices of 326 to 18,806; fintfm FINDINGS §295). Classification ignores it.

    **Large-table levers (off in the class defaults; off reproduces the earlier fit exactly).**

    - ``partition_scale_with_n`` (bool): choose the partition depth from the training-row count
      instead of fixing it. With ``n_eff`` the rows one tree is fitted on (all rows for one bag,
      ``bag_frac * n`` above one bag), the depth is
      ``clip(ceil(log2(n_eff / (partition_scale_k * max_context))), base, min(partition_depth_cap, 3))``
      where ``base`` is the partition's own ``max_depth`` (2 when unset). ``partition_scale_k``
      defaults to 4 and ``partition_depth_cap`` to 5, but the fintfm partition classes refuse a
      depth above 3 (8 leaves), so the effective cap is 3; the log line and ``partition_depth_chosen_``
      record the requested and the effective depth. Only the depth changes. Has no effect without a
      partition. Cost: each extra depth level doubles the leaf contexts.
    - ``partition_bags_max_rows`` (int >= 1 or ``None``, default ``None``): a fit with more training
      rows than this builds its partition with one bag, exactly as if ``n_bags`` were absent; a fit
      at or below it keeps the requested ``n_bags``. Decided once per fit from the fit's row count.
    - ``probe_spline`` (``None``/``False``, ``True`` or ``{"n_knots", "degree", "max_columns"}``):
      spline-expands the numeric columns of the ``raw`` readout block (``n_knots`` quantile knots from
      the training rows, B-spline ``degree``, at most ``max_columns`` columns). Needs the ``raw`` probe
      option; a request without it is ignored with a warning.
    - ``spline_max_rows`` (int >= 1 or ``None``): a fit with more training rows than this treats
      ``probe_spline`` as off; a fit at or below it, or ``None``, is unchanged.
    - ``probe_budget`` (int or ``"all"``, class default 20,000 = ``TRAIN_SAMPLE_SIZE``; the binary
      entry sets 50,000): the readout's training-row budget, in total and split across leaves in
      proportion to their rows (at least ``LEAF_MIN_SAMPLE`` per leaf). ``"all"`` uses every training
      row. Cost: the representation pass over the readout rows is linear in the budget, one pass per leaf
      classifier (and one more per cross-fit fold), and the representation matrix is held in memory.

    Args:
        problem_type: ``binary``, ``multiclass`` or ``regression``.
        checkpoint: A local checkpoint path, or a published reference dict (see
            :mod:`fintfm.readout.checkpoints`), resolved and checked when ``fit`` runs.
        device: The torch device the network is loaded on.
        random_state: The seed of every random draw of the fit.
        **settings: The settings named above, plus ``FinancialTFMClassifier`` arguments.

    Raises:
        ValueError: ``problem_type`` is not binary, multiclass or regression.
    """

    #: The probe options of the current fit, set by ``_fit``.
    _probe_opts: ClassVar[dict] = {}
    #: Cross-fit folds of the current fit (0 or 1: off), set with the probe options.
    _crossfit_folds: int = 0
    #: Whether the readout of the current fit is cross-fitted.
    _crossfit_on: bool = False
    #: The matrix the classifier was fitted on, held only while ``_fit`` builds the readout.
    _train_matrix = None
    #: Counters of the cross-fit work of the current fit, for the log line.
    _crossfit_stats: ClassVar[dict] = {}
    #: Regression levers of the current fit (see :meth:`_parse_reg_levers`).
    _reg_transform_name: str = "none"
    _reg_blend = None
    _reg_alphas = None
    _reg_readout: str = "ridge"
    #: ``reg_rff_max_rows`` of the current fit: ``None`` (no cap) or the most training rows at which
    #: ``reg_readout="rff"`` is still used.
    _reg_rff_max_rows: int | None = None
    #: The readout the current fit uses: ``_reg_readout``, except ``"ridge"`` when it is ``"rff"``
    #: and the fit has more than ``_reg_rff_max_rows`` rows. Decided once per fit.
    _reg_readout_eff: str = "ridge"
    #: Training rows of the current fit, read by :meth:`_resolve_entry` for the depth scaling.
    _n_fit_rows: int | None = None
    #: ``partition_bags_max_rows`` of the current fit: ``None`` (no gate) or the most training rows at
    #: which the partition keeps its requested ``n_bags``.
    _bags_max_rows: int | None = None
    #: ``spline_max_rows`` of the current fit: ``None`` (no gate) or the most training rows at which
    #: ``probe_spline`` still applies.
    _spline_max_rows: int | None = None
    #: Large-table levers of the current fit, set by :meth:`_resolve_entry`.
    _scale_with_n: bool = False
    _scale_k: float = PARTITION_SCALE_K
    _depth_cap: int = PARTITION_DEPTH_CAP
    _probe_budget_setting: int | str = TRAIN_SAMPLE_SIZE
    #: ``probe_prob_floor`` of the current fit: ``None`` (off), ``"cv"`` or a float in [0, 1).
    _prob_floor_setting: float | str | None = None
    #: Held-out ``(label index, probabilities)`` blocks gathered while the probes are fitted, for ``"cv"``.
    _floor_oof: ClassVar[list] = []
    #: The floor ``e`` applied at predict time; ``0.0`` is no floor.
    prob_floor_: float = 0.0
    #: Pooled held-out log loss per candidate floor when ``"cv"`` chose it, else ``None``.
    prob_floor_losses_: dict | None = None
    #: What the depth scaling chose for the current fit (``None`` when off or unpartitioned).
    partition_depth_chosen_: dict | None = None
    #: ``probe_onehot`` of the current fit as ``{"min_count", "max_columns", "max_levels"}``, or ``None`` (off).
    _onehot_cfg: dict | None = None
    #: The fitted indicator columns ``(column index among the categoricals, level code)`` as two
    #: arrays, or ``None`` when none were made (off, or no categorical columns).
    _oh_spec: tuple | None = None
    #: ``probe_spline`` of the current fit as ``{"n_knots", "degree", "max_columns"}``, or ``None`` (off or moot).
    _spline_cfg: dict | None = None
    #: Positions of the categorical columns in the preprocessed matrix, recorded on the training
    #: rows while ``probe_spline`` is on (``None`` otherwise); these columns are never splined.
    _cat_cols: np.ndarray | None = None
    #: The fitted spline stage of the raw block, or ``None`` (off). ``{"keep", "sel", "tf"}``: the
    #: columns passed through as they are, the columns expanded, and the fitted
    #: ``SplineTransformer`` of the latter. Holds knot positions and coefficients only.
    _raw_spline: dict | None = None
    #: The fitted target transform, or ``None`` when none is applied.
    _target_tf: TargetTransform | None = None
    #: What the transform did: requested name, effective name, and the reason for a fallback.
    _reg_transform_info: ClassVar[dict] = {}
    #: Weight on the ridge prediction in the blend with the binned head, ``None`` for no blend.
    _reg_blend_weight: float | None = None
    #: Duan smearing factor of the log1p back-transform.
    _reg_smear: float = 1.0
    #: ``(low, high)`` every regression prediction is clipped to, or ``None`` (no clip).
    _reg_clip: tuple[float, float] | None = None
    #: Held-out readout-row targets and predictions the blend weight and smearing are chosen on.
    _reg_oof: ClassVar[dict] = {}

    def __init__(
        self,
        problem_type: str,
        checkpoint: Any,
        device: str = "cpu",
        random_state: int = 0,
        **settings: Any,
    ) -> None:
        if problem_type not in PROBLEM_TYPES:
            raise ValueError(f"fintfm supports {PROBLEM_TYPES}, not problem type {problem_type!r}")
        self.problem_type = problem_type
        self.checkpoint = checkpoint
        self.device = device
        self.random_state = random_state
        self.settings = dict(settings)
        #: The fitted base classifier or regressor (possibly partitioned), set by :meth:`fit`.
        self.model = None
        self.classes_: np.ndarray | None = None
        self._categorical_features: tuple[int, ...] = ()
        self._feature_names: list[str] | None = None
        self._categorical_encoder = None
        self._minority_flipped = False
        self._minority_positive = False
        self._reg_thresholds: int | None = None
        self._linear_probe = None
        self._leaf_probes = None
        self._ridge_probe = None
        self._ridge_leaf_probes = None
        self._raw_conditioner = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        categorical_features: Sequence[int] = (),
        feature_names: Sequence[str] | None = None,
    ) -> LinearReadoutModel:
        """Store the table as in-context evidence and fit the readout on it.

        Args:
            X: The ``(n, d)`` float32 feature matrix; NaN marks a missing value and is kept.
                Categorical columns hold integer codes (``-1`` or NaN for missing).
            y: The target: class codes ``0 .. K - 1`` for classification, floats for regression.
            categorical_features: Positions of the categorical columns of ``X``, in the order their
                encoded statistics are written.
            feature_names: Column names of ``X``, used only to name columns in log lines; positions
                are logged when absent.

        Returns:
            ``self``.

        Raises:
            AssertionError: the table has more classes or features than the checkpoint represents
                (and no column-selection strategy is set), or cross-fitting lost the training matrix.
            ValueError: a setting is malformed.
        """
        X = np.asarray(X)
        y = np.asarray(y)
        self._categorical_features = tuple(int(i) for i in categorical_features)
        self._feature_names = None if feature_names is None else [str(n) for n in feature_names]
        self._categorical_encoder = None
        self._parse_reg_levers(copy.deepcopy(self.settings), pop=False)
        self._n_fit_rows = len(X)
        self._resolve_reg_readout()
        y = self._apply_target_transform(y)
        self._fit_network(X, y)
        crossfit = self._crossfit_folds >= 2
        if crossfit and hasattr(self.model, "thresholds_"):
            logger.warning(
                "FinTFM-LinProbe: probe_crossfit_folds=%d ignored, threshold regression is not cross-fitted.",
                self._crossfit_folds,
            )
            crossfit = False
        train_matrix, self._train_matrix = self._train_matrix, None
        self._crossfit_stats = {"restricted": 0, "context_rows": 0, "other_rows": 0, "skipped": 0}
        if crossfit:
            # The matrix the classifier was fitted on: out-of-fold categorical encoding, and
            # the routing the partition used at fit time.
            Xp = train_matrix
            if Xp is None:
                raise AssertionError(
                    "cross-fitting needs the training matrix, which the encoding did not keep"
                )
        else:
            # The same encoding the network fit just used, replayed with is_train=False so the already-
            # fitted categorical encoder is reused rather than refit on the same rows twice.
            Xp = self._encode(X, is_train=False)
        self._crossfit_on = crossfit
        y_arr = np.asarray(y)
        self._ridge_probe = None
        self._floor_oof = []
        self.prob_floor_losses_ = None
        self.prob_floor_ = 0.0
        if self.problem_type == "regression":
            self._leaf_probes = None
            self._linear_probe = None
            if self._reg_blend == "cv" and hasattr(self.model, "thresholds_"):
                raise ValueError(
                    "reg_readout_blend='cv' is not supported with threshold regression"
                )
            self._reg_oof = {"y": [], "ridge": [], "head": []}
            if self._reg_blend is None:
                self._reg_blend_weight = None
            else:
                self._reg_blend_weight = 1.0 if self._reg_blend == "cv" else self._reg_blend
            self._reg_smear = 1.0
            self._fit_ridge_probe(np.asarray(Xp, dtype=np.float32), y_arr.astype(np.float64))
            self._finish_reg_levers()
            self._reg_clip = self._clip_bounds(y_arr.astype(np.float64))
            self._log_crossfit()
            return self
        self.classes_ = np.unique(y_arr)
        if hasattr(self.model, "trees_"):
            # The partition setting wrapped the classifier: one probe per leaf context.
            self._fit_leaf_probes(_as_f32(Xp), y_arr)
            self._finish_prob_floor()
            self._log_crossfit()
            return self
        self._leaf_probes = None

        sub = _probe_rows(y_arr, self._probe_budget(len(y_arr)), int(self.random_state))
        y_sub = y_arr[sub]
        if len(np.unique(y_sub)) < 2:
            # A linear probe cannot be fit on one class, so fall back to the network's own head
            # for this fit rather than let one degenerate call fail the whole run. The sampling
            # in _probe_rows no longer produces a single-class subsample from a two-class y
            # (fintfm's TabArena loss analysis, D1, traced the fallbacks seen on Lite to it);
            # this guard remains for a y that arrives with one class.
            logger.warning(
                "FinTFM-LinProbe: training subsample had %d class(es), not 2; falling back "
                "to the network's own head for this fit.",
                len(np.unique(y_sub)),
            )
            self._linear_probe = None
            return self
        Xp = _as_f32(Xp)
        self._fit_raw_conditioner(Xp)
        # The classifier was fitted on every row, so a row's position in its fit rows is its index.
        fit_idx = sub if crossfit else None
        rep = self._probe_input(self.model, Xp[sub], fit_idx)
        self._linear_probe = self._new_classifier_probe(y_sub).fit(rep, y_sub)
        self._collect_floor_oof(self._linear_probe, rep, y_sub)
        self._finish_prob_floor()
        self._log_crossfit()
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Class probabilities of the readout for rows ``X`` (classification only).

        Args:
            X: The ``(n, d)`` float32 feature matrix, laid out as for :meth:`fit`.

        Returns:
            ``(n, K)`` probabilities in the class order of the fit, floored by ``prob_floor_``.

        Raises:
            ValueError: the problem type is regression.
        """
        if self.problem_type == "regression":
            raise ValueError("predict_proba is for classification; use predict for regression")
        if getattr(self, "_leaf_probes", None) is not None:
            Xp = _as_f32(self._encode(X))
            part = self.model
            routes = part.route(Xp)
            out = np.zeros((Xp.shape[0], len(part.classes_)), dtype=np.float64)
            for b, leaves in enumerate(part.leaf_estimators_):
                for leaf in np.unique(routes[b]):
                    rows = np.flatnonzero(routes[b] == leaf)
                    est, probe = leaves.get(int(leaf)), self._leaf_probes[b].get(int(leaf))
                    if est is None:
                        est, probe = part.global_estimator_, self._linear_probe
                    proba = self._score(est, probe, Xp[rows])
                    if proba.shape[1] < len(part.classes_):
                        # A leaf lacking some classes (missing_classes="global"): the global
                        # probe supplies their probability, the leaf's probe shares the rest.
                        classes = probe.classes_ if probe is not None else est.classes_
                        glob = self._score(part.global_estimator_, self._linear_probe, Xp[rows])
                        proba = part.combine_with_global(proba, classes, glob)
                    out[rows] += proba
            return self._apply_prob_floor(out / len(part.leaf_estimators_))
        if self._linear_probe is None:
            # Fell back to the network's own head at fit time.
            proba = self.model.predict_proba(self._encode(X))
            return proba[:, ::-1] if self._minority_flipped else proba
        Xp = _as_f32(self._encode(X))
        proba = self._linear_probe.predict_proba(self._probe_input(self.model, Xp))
        return self._apply_prob_floor(proba)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """The prediction for rows ``X``: the regression value, or the most probable class.

        Args:
            X: The ``(n, d)`` float32 feature matrix, laid out as for :meth:`fit`.

        Returns:
            ``(n,)`` floats for regression (inverted from the target transform and clipped when those
            are on); the arg-max class label for classification.
        """
        if self.problem_type != "regression":
            return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
        if getattr(self, "_ridge_probe", None) is None:
            return np.asarray(self.model.predict(self._encode(X)))
        # Regression: the ridge readout, blended and back-transformed as configured.
        Xp = np.asarray(self._encode(X), dtype=np.float32)
        pred = self._predict_ridge(Xp)
        if self._reg_blend_weight is not None and self._reg_blend_weight < 1.0:
            head = np.asarray(self.model.predict(Xp), dtype=np.float64)
            pred = self._reg_blend_weight * pred + (1.0 - self._reg_blend_weight) * head
        if self._target_tf is not None:
            pred = self._target_tf.inverse(pred, self._reg_smear)
        if self._reg_clip is not None:
            pred = np.clip(pred, *self._reg_clip)
        return pred

    def inner_classifiers(self) -> list:
        """Every fitted ``FinancialTFMClassifier`` behind ``self.model``.

        One for the plain classifier; for a partitioned one, the template, the global
        fallback and each leaf's. They share one network, but each records its own device.

        Returns:
            The classifiers, in no particular order.
        """
        if not hasattr(self.model, "leaf_estimators_"):
            return [self.model]
        inner = [self.model.estimator, self.model.global_estimator_]
        for leaves in self.model.leaf_estimators_:
            inner.extend(est for est in leaves.values() if est is not None)
        return inner

    def current_device(self) -> str:
        """The device the fitted network's parameters live on.

        Returns:
            A torch device string such as ``"cpu"`` or ``"cuda:0"``.
        """
        return str(next(self.inner_classifiers()[0].model.parameters()).device)

    def to(self, device: str) -> LinearReadoutModel:
        """Move the fitted network, and every classifier's recorded device, to ``device``.

        Args:
            device: A torch device string.

        Returns:
            ``self``.
        """
        for clf in self.inner_classifiers():
            clf.model.to(device)
            clf.device = device
        self.device = device
        return self

    def _encode(
        self, X: np.ndarray, y: np.ndarray | None = None, is_train: bool = False
    ) -> np.ndarray:
        """Encode categoricals as out-of-fold target statistics; add the one-hot indicators when on.

        fintfm's cell embedding takes numeric values only: it z-scores each column against the
        context and embeds the scalar, so a string category has no representation, and label codes
        impose an arbitrary total order on an unordered variable (fintfm ``docs/results/FINDINGS.md``
        §100: a 0.0894 ROC-AUC deficit on mostly-categorical datasets against 0.0320 on numeric
        ones). Each level gets the smoothed target statistic among training rows carrying it, encoded
        out of fold on training rows so no row informs its own statistic; query rows use the full
        training statistics. See ``fintfm.inference.categorical``. Numeric NaN is passed through for
        the model's missingness mask.

        With ``probe_onehot`` on, the categorical columns' label codes are also turned into an
        indicator matrix and attached to the returned array (see :class:`_AugmentedMatrix`); the
        classifier sees the unchanged matrix. On training rows the indicator columns are chosen: per
        categorical column the levels with at least ``min_count`` rows (a two-level column gets only
        its rarer level, the other is its complement), then the ``max_columns`` most frequent of them
        across all columns (ties by column order, then level code). Query rows get the same columns;
        an unseen level is zero everywhere.

        Args:
            X: The float32 feature matrix, categorical columns as label codes.
            y: Training targets (training rows only).
            is_train: Whether these are the training rows.

        Returns:
            The float32 array the network reads, carrying the indicator block when one was made.
        """
        from fintfm.inference.categorical import CategoricalTargetEncoder

        arr = np.array(X, dtype=np.float32, copy=True)
        cat = list(self._categorical_features)
        if not cat:
            out = arr
        elif is_train:
            self._categorical_encoder = CategoricalTargetEncoder(
                categorical_features=cat, random_state=self.random_state
            )
            out = self._categorical_encoder.fit_transform(arr, np.asarray(y))
        else:
            if self._categorical_encoder is None:
                raise AssertionError("predict called before fit: no categorical encoder")
            out = self._categorical_encoder.transform(arr)
        if is_train and self._spline_cfg is not None:
            # Positions of the categorical columns, kept so the splines skip them.
            self._cat_cols = np.sort(np.array(cat, dtype=np.intp))
        cfg = self._onehot_cfg
        if cfg is not None:
            if not cat:
                self._oh_spec = None
            else:
                codes = np.asarray(X, dtype=np.float64)[:, cat]
                codes = np.where(np.isnan(codes), -1.0, codes)
                if is_train:
                    self._oh_spec = self._choose_indicators(
                        codes,
                        cfg,
                        names=[
                            str(i) if self._feature_names is None else self._feature_names[i]
                            for i in cat
                        ],
                    )
                if self._oh_spec is not None:
                    cols, levels = self._oh_spec
                    out = _AugmentedMatrix.attach(
                        out, (codes[:, cols] == levels[None, :]).astype(np.float32)
                    )
        if is_train and self._crossfit_folds >= 2:
            self._train_matrix = out
        return out

    def _fit_network(self, X: np.ndarray, y: np.ndarray) -> None:
        """Load the checkpoint, encode the table and fit the base classifier or regressor on it.

        Args:
            X: The float32 feature matrix.
            y: The target, in the readout's space (transformed when a target transform is on).

        Raises:
            AssertionError: the table has more classes or features than the checkpoint represents.
        """
        from fintfm.inference.classifier import FinancialTFMClassifier
        from fintfm.modeling.model import FinancialTFM

        params = copy.deepcopy(self.settings)
        entry = self._resolve_entry(params)
        n_bins = entry["n_bins"]
        # Binary and multiclass take **different checkpoints on purpose.** A multiclass
        # checkpoint is trained at max_classes=10 with the generic SCM prior mixed in, which
        # is a different model from the financial-prior binary one; using it for binary tasks
        # would silently change every binary number this project has published. Regression
        # takes its own checkpoint again: the binned head only means anything if the prior
        # emitted continuous targets during pretraining, which the others never did.
        checkpoint = resolve_checkpoint(self.checkpoint, problem_type=self.problem_type)

        device = self.device
        network = FinancialTFM.load(checkpoint, map_location=device)

        X = self._encode(X, y=y, is_train=True)
        if self.problem_type == "regression":
            self._fit_regression(
                X, np.asarray(y), network, device, n_bins, params, entry["partition"]
            )
            return
        n_classes = int(np.unique(np.asarray(y)).size)
        if n_classes > network.cfg.max_classes:
            raise AssertionError(
                f"fintfm checkpoint represents at most {network.cfg.max_classes} classes, "
                f"task has {n_classes}. Pass a checkpoint trained at --max-classes {n_classes} or "
                f"higher; scoring anyway would measure a task the model cannot represent."
            )
        n_features = X.shape[1]
        if n_features > network.cfg.max_features and params.get(
            "max_features_strategy", "raise"
        ) not in (
            "subsample",
            "select",
        ):
            raise AssertionError(
                f"fintfm checkpoint accepts at most {network.cfg.max_features} features, "
                f"task has {n_features}. Truncating would score a different task, so this "
                f"raises instead. Pass max_features_strategy='subsample' (an ensemble over "
                f"column subsets) or 'select' (the top columns ranked on the training rows, "
                f"fintfm DR8.3) to score it."
            )

        params.setdefault("random_state", self.random_state)
        y_fit = np.asarray(y)
        self._minority_flipped = self._flip_to_minority(y_fit)
        if self._minority_flipped:
            y_fit = 1 - y_fit
        classifier = FinancialTFMClassifier(network, device=device, **params)
        partition = entry["partition"]
        if partition is not None:
            from fintfm.inference.large_n import PartitionedTFMClassifier

            classifier = PartitionedTFMClassifier(
                classifier, random_state=params["random_state"], **partition
            )
        self.model = classifier.fit(X, y_fit)

    def _flip_to_minority(self, y: np.ndarray) -> bool:
        """Whether this fit codes the minority class as 1 (fintfm loss analysis, fix 2).

        AutoGluon codes a binary target's second sorted label as 1, so on some tasks the class
        the model treats as the positive outcome is the majority, while the financial prior
        always made class 1 the rarer one (at most 30% of rows). With ``minority_positive`` set the
        codes are swapped when class 1 holds more than half of this fit's training rows, and the head's
        probability columns are swapped back when it scores, so the caller reads the class it coded.
        Off (the default), nothing changes.

        Args:
            y: The fit's 0/1 target.

        Returns:
            ``True`` when the setting is on, the task is binary and class 1 is the majority.
        """
        if not self._minority_positive or self.problem_type != "binary":
            return False
        flip = 2 * int(np.count_nonzero(y == 1)) > y.shape[0]
        logger.warning("fintfm: minority_positive set, class 1 recoded as 0: %s", flip)
        return flip

    def _fit_regression(
        self, X, y, network, device: str, n_bins: int, params: dict, partition: dict | None
    ):
        """Store the table as context for the binned-regression wrapper.

        ``n_bins`` is clamped to the checkpoint's ``max_classes`` rather than raising,
        because unlike a class count it is *our* free parameter, not a property of the task:
        a 10-bin request against an 8-logit head is an over-ask by us, and the honest
        response is to use the 8 bins available and say so in the log. A class count that
        exceeds the head is a different thing entirely and still raises.
        """
        from fintfm.inference.regressor import FinancialTFMRegressor

        n_features = X.shape[1]
        if (
            n_features > network.cfg.max_features
            and params.get("max_features_strategy", "raise") != "select"
        ):
            # "select" ranks the columns against the bin indices on the training rows and is
            # an ordinary narrower table from then on; "subsample" averages member
            # probabilities over column subsets and is left to classification.
            raise AssertionError(
                f"fintfm checkpoint accepts at most {network.cfg.max_features} features, "
                f"task has {n_features}. Truncating would score a different task, so this "
                f"raises instead. Pass max_features_strategy='select' to score it."
            )
        bins = min(n_bins, network.cfg.max_classes)
        if bins < n_bins:
            logger.warning(
                "fintfm: n_bins reduced %d -> %d to fit the checkpoint's head; the "
                "prediction grid is coarser than requested.",
                n_bins,
                bins,
            )
        params.setdefault("random_state", self.random_state)
        regressor = FinancialTFMRegressor(
            network, n_bins=bins, classifier_params={"device": device, **params}
        )
        thresholds = self._reg_thresholds
        if thresholds is not None:
            # K binary classifiers on y > t_k instead of one classifier over max_classes bins
            # (fintfm ThresholdTFMRegressor, FINDINGS §235): the grid is K, not the head's cap.
            from fintfm.inference.threshold_regressor import ThresholdTFMRegressor

            if partition is not None:
                raise ValueError("reg_thresholds and a regression partition are not combined yet")
            regressor = ThresholdTFMRegressor(
                network,
                n_thresholds=int(thresholds),
                classifier_params={"device": device, **params},
            )
            logger.warning("fintfm: threshold regression, %s thresholds", thresholds)
        if partition is not None:
            # A regression tree on the target; each leaf bins its own targets (fintfm 49.1).
            import inspect

            from fintfm.inference.large_n import PartitionedTFMRegressor

            # missing_classes is a classification setting; the regressor has no classes.
            accepted = set(inspect.signature(PartitionedTFMRegressor.__init__).parameters)
            partition = {k: v for k, v in partition.items() if k in accepted}
            regressor = PartitionedTFMRegressor(
                regressor, random_state=params["random_state"], **partition
            )
        self.model = regressor.fit(X, y)

    def _resolve_entry(self, params: dict) -> dict:
        """Take this fit's readout settings out of ``params``, validate them and keep them on the model.

        Removes every key that is not a ``FinancialTFMClassifier`` argument so the rest can be passed
        on unchanged, and returns the base settings of the fit. Also takes ``probe_crossfit_folds``
        out of ``params`` so the classifier never sees it.

        Args:
            params: A copy of the model's settings; modified in place.

        Returns:
            A dict with ``partition`` (``None`` or a validated dict, after the bag gate and depth
            scaling), ``probe`` (a validated dict) and ``n_bins`` (an int of at least 2).

        Raises:
            ValueError: ``probe_crossfit_folds`` is not a non-negative integer, or is 2 or more
                with the ``head`` probe option, which is not cross-fitted; a setting is malformed;
                or no checkpoint is set.
        """
        self._parse_reg_levers(params, pop=True)
        self._parse_scale_levers(params)
        self._bags_max_rows = check_bags_max_rows(
            params.pop("partition_bags_max_rows", None), source="partition_bags_max_rows"
        )
        self._spline_max_rows = check_spline_max_rows(
            params.pop("spline_max_rows", None), source="spline_max_rows"
        )
        self._prob_floor_setting = check_prob_floor(
            params.pop("probe_prob_floor", None), source="probe_prob_floor"
        )
        folds = params.pop("probe_crossfit_folds", PROBE_CROSSFIT_FOLDS)
        if isinstance(folds, bool) or not isinstance(folds, int) or folds < 0:
            raise ValueError(f"probe_crossfit_folds must be a non-negative integer, got {folds!r}")
        onehot = params.pop("probe_onehot", None)
        minority = params.pop("minority_positive", False)
        if not isinstance(minority, bool):
            raise ValueError(f"minority_positive must be a bool, got {minority!r}")  # noqa: TRY004
        self._minority_positive = minority
        thresholds = params.pop("reg_thresholds", None)
        if thresholds is not None and (
            isinstance(thresholds, bool) or not isinstance(thresholds, int) or thresholds < 1
        ):
            raise ValueError(f"reg_thresholds must be None or an integer >= 1, got {thresholds!r}")
        self._reg_thresholds = thresholds
        partition = params.pop("partition", None)
        probe = params.pop("probe", None)
        n_bins = int(params.pop("n_bins", 10))
        if n_bins < 2:
            raise ValueError(f"n_bins must be at least 2, got {n_bins}")
        if self.checkpoint is None:
            raise ValueError(f"no checkpoint is set for problem type {self.problem_type!r}")
        validate_checkpoint_ref(self.checkpoint, source="checkpoint")
        entry = {
            "partition": validated_partition(partition, source="partition"),
            "probe": probe,
            "n_bins": n_bins,
        }
        entry["partition"] = self._scaled_partition(self._gate_partition_bags(entry["partition"]))
        self._probe_opts = entry["probe"] = validated_probe(entry["probe"], source="probe")
        if folds >= 2 and self._probe_opts.get("head"):
            raise ValueError(
                "probe option 'head' reads the network's own prediction on the readout rows, which "
                "is not cross-fitted; set probe_crossfit_folds to 0 to use it"
            )
        self._crossfit_folds = folds
        self._onehot_cfg = self._resolve_onehot(onehot)
        self._spline_cfg = self._resolve_spline(params.pop("probe_spline", None))
        return entry

    def _parse_scale_levers(self, params: dict) -> None:
        """Validate ``partition_scale_*``, ``partition_depth_cap`` and ``probe_budget``, popping them.

        Raises:
            ValueError: a lever is malformed.
        """
        on = params.pop("partition_scale_with_n", False)
        if not isinstance(on, bool):
            raise ValueError(f"partition_scale_with_n must be a bool, got {on!r}")  # noqa: TRY004
        self._scale_with_n = on
        self._scale_k = check_scale_k(
            params.pop("partition_scale_k", PARTITION_SCALE_K), source="partition_scale_k"
        )
        cap = params.pop("partition_depth_cap", PARTITION_DEPTH_CAP)
        if isinstance(cap, bool) or not isinstance(cap, int) or cap < 1:
            raise ValueError(f"partition_depth_cap must be a positive integer, got {cap!r}")
        self._depth_cap = cap
        self._probe_budget_setting = check_probe_budget(
            params.pop("probe_budget", TRAIN_SAMPLE_SIZE), source="probe_budget"
        )

    def _probe_budget(self, n: int) -> int:
        """Readout training-row budget for ``n`` candidate rows: ``probe_budget``, or ``n`` for ``"all"``."""
        return max(1, n) if self._probe_budget_setting == "all" else int(self._probe_budget_setting)

    def _scaled_partition(self, partition: dict | None) -> dict | None:
        """``partition`` with ``max_depth`` chosen from the training-row count, when scaling is on.

        Args:
            partition: The validated partition dict, or ``None``.

        Returns:
            A copy with ``max_depth`` set, or ``partition`` itself when scaling is off, there is
            no partition, or the row count is not known.
        """
        self.partition_depth_chosen_ = None
        n = self._n_fit_rows
        if not self._scale_with_n or partition is None or not n:
            return partition
        base = int(partition.get("max_depth", 2))
        n_bags = int(partition.get("n_bags", 1))
        n_eff = n if n_bags <= 1 else max(1, round(float(partition.get("bag_frac", 0.6)) * n))
        max_context = int(self.settings.get("max_context", 1000))
        target = n_eff / (self._scale_k * max_context)
        wanted = math.ceil(math.log2(target)) if target > 1 else 0
        cap = min(self._depth_cap, PARTITION_LIBRARY_MAX_DEPTH)
        depth = max(base, min(wanted, cap))
        self.partition_depth_chosen_ = {
            "n_rows": n,
            "n_effective": n_eff,
            "base": base,
            "wanted": max(base, min(wanted, self._depth_cap)),
            "chosen": depth,
            "cap_requested": self._depth_cap,
            "cap_effective": cap,
        }
        logger.warning(
            "FinTFM-LinProbe: partition depth %d for n=%d (n_eff=%d, k=%g, max_context=%d, base %d, "
            "cap %d requested, %d effective).",
            depth,
            n,
            n_eff,
            self._scale_k,
            max_context,
            base,
            self._depth_cap,
            cap,
        )
        return {**partition, "max_depth": depth}

    def _clip_bounds(self, y: np.ndarray) -> tuple[float, float] | None:
        """The clip range of this fit's regression predictions, from its training targets.

        Args:
            y: Training targets in the readout's space (transformed when a target transform is on).

        Returns:
            ``(low, high)`` on the original target scale, or ``None`` when ``reg_clip_margin`` is off.
        """
        import numpy as np

        if self._reg_clip_margin is None:
            return None
        lo, hi = float(np.min(y)), float(np.max(y))
        if self._target_tf is not None:
            # Monotone transforms: the ends map to the ends. Smearing (a factor on the
            # back-transform) is left out so the range is the observed one.
            lo, hi = (float(v) for v in self._target_tf.inverse(np.array([lo, hi]), 1.0))
        pad = self._reg_clip_margin * (hi - lo)
        return lo - pad, hi + pad

    def _gate_partition_bags(self, partition: dict | None) -> dict | None:
        """``partition`` without ``n_bags`` when this fit has more rows than ``partition_bags_max_rows``.

        Decided once per fit, before the classifier is built, and logged once. Predict uses the
        fitted model, so it follows the same choice.

        Args:
            partition: The validated partition dict, or ``None``.

        Returns:
            A copy with ``n_bags`` removed (one bag, the class default) when the gate fires,
            otherwise ``partition`` itself. Unchanged when there is no gate, no partition, no
            ``n_bags`` above 1, or the row count is unknown.
        """
        cap, n = self._bags_max_rows, self._n_fit_rows
        if cap is None or partition is None or n is None or int(partition.get("n_bags", 1)) <= 1:
            return partition
        if n > cap:
            logger.warning(
                "fintfm: partition n_bags=%d -> 1: %d training rows exceed partition_bags_max_rows=%d",
                int(partition["n_bags"]),
                n,
                cap,
            )
            return {k: v for k, v in partition.items() if k != "n_bags"}
        logger.warning(
            "fintfm: partition n_bags=%d kept: %d training rows within partition_bags_max_rows=%d",
            int(partition["n_bags"]),
            n,
            cap,
        )
        return partition

    def _parse_reg_levers(self, params: dict, *, pop: bool) -> None:
        """Validate the regression levers in ``params`` and keep them on the model.

        Args:
            params: The model's hyperparameters.
            pop: Remove the keys, so the classifier never sees them.

        Raises:
            ValueError: a lever is malformed, or ``reg_readout_blend='cv'`` is set without
                cross-fitting, which its out-of-fold head predictions need.
        """
        get = params.pop if pop else params.get
        self._reg_transform_name = check_reg_transform(
            get("reg_target_transform", "none"), source="reg_target_transform"
        )
        self._reg_blend = check_reg_blend(
            get("reg_readout_blend", None), source="reg_readout_blend"
        )
        self._reg_alphas = check_reg_alphas(
            get("reg_ridge_alphas", None), source="reg_ridge_alphas"
        )
        self._reg_readout = check_reg_readout(get("reg_readout", "ridge"), source="reg_readout")
        self._reg_rff_max_rows = check_reg_rff_max_rows(
            get("reg_rff_max_rows", None), source="reg_rff_max_rows"
        )
        self._reg_clip_margin = check_reg_clip_margin(
            get("reg_clip_margin", None), source="reg_clip_margin"
        )
        self._reg_readout_eff = self._effective_reg_readout()
        folds = params.get("probe_crossfit_folds", PROBE_CROSSFIT_FOLDS)
        if self._reg_blend == "cv" and not (isinstance(folds, int) and folds >= 2):
            raise ValueError(
                "reg_readout_blend='cv' needs probe_crossfit_folds >= 2 for out-of-fold head predictions"
            )

    def _effective_reg_readout(self) -> str:
        """The readout this fit uses: ``rff`` falls back to ``ridge`` above ``reg_rff_max_rows`` rows.

        Returns:
            ``"rff"`` or ``"ridge"``. The row count is the fit's (``_n_fit_rows``), not the probe
            subsample's; with the count unknown or no cap set the requested readout stands.
        """
        cap, n = self._reg_rff_max_rows, self._n_fit_rows
        if self._reg_readout == "rff" and cap is not None and n is not None and n > cap:
            return "ridge"
        return self._reg_readout

    def _resolve_reg_readout(self) -> None:
        """Decide the regression readout of this fit from its row count and log it once.

        Stores the result in ``_reg_readout_eff``, which the probe fit reads; predict uses the
        fitted pipeline, so it follows the same choice. Called from ``_fit`` after the row count is
        known; ``_parse_reg_levers`` recomputes the same value silently.
        """
        self._reg_readout_eff = self._effective_reg_readout()
        cap = self._reg_rff_max_rows
        if self._reg_readout == "rff" and cap is not None:
            if self._reg_readout_eff == "ridge":
                logger.warning(
                    "fintfm: reg_readout=rff -> ridge: %d training rows exceed reg_rff_max_rows=%d",
                    self._n_fit_rows,
                    cap,
                )
            else:
                logger.warning(
                    "fintfm: reg_readout=rff kept: %d training rows within reg_rff_max_rows=%d",
                    self._n_fit_rows,
                    cap,
                )

    def _resolve_spline(self, value) -> dict | None:
        """The ``probe_spline`` settings that apply to this fit, ``None`` when it is off or moot.

        The spline features replace numeric columns of the ``raw`` block, so without that probe
        option there is nothing to expand and a request is ignored with a warning. It applies to
        binary, multiclass and regression readouts alike. A fit with more training rows than
        ``spline_max_rows`` (``_n_fit_rows``; this fit's own rows, also for a bagged child) is
        gated off and logged once; ``None`` never gates.

        Args:
            value: The ``probe_spline`` hyperparameter.

        Returns:
            The settings from :func:`~fintfm.readout.entry.check_spline`, or ``None``.
        """
        cfg = check_spline(value, source="probe_spline")
        if cfg is None:
            return None
        if not self._probe_opts.get("raw"):
            logger.warning(
                "FinTFM-LinProbe: probe_spline ignored, it needs the 'raw' probe option."
            )
            return None
        cap, n = self._spline_max_rows, self._n_fit_rows
        if cap is not None and n is not None:
            if n > cap:
                logger.warning(
                    "fintfm: probe_spline gated off: %d training rows exceed spline_max_rows=%d",
                    n,
                    cap,
                )
                return None
            logger.warning(
                "fintfm: probe_spline kept: %d training rows within spline_max_rows=%d", n, cap
            )
        return cfg

    def _resolve_onehot(self, value) -> dict | None:
        """The ``probe_onehot`` settings that apply to this fit, ``None`` when it is off or moot.

        Only a multiclass fit whose readout has the ``raw`` block uses it; elsewhere a request is
        ignored with a warning, so the binary and regression fits are untouched.

        Args:
            value: The ``probe_onehot`` hyperparameter.

        Returns:
            The settings from :func:`~fintfm.readout.entry.check_onehot`, or ``None``.
        """
        cfg = check_onehot(value, source="probe_onehot")
        if cfg is None:
            return None
        if self.problem_type != "multiclass" or not self._probe_opts.get("raw"):
            logger.warning(
                "FinTFM-LinProbe: probe_onehot ignored, it needs a multiclass fit with the 'raw' probe option."
            )
            return None
        return cfg

    @staticmethod
    def _choose_indicators(
        codes: np.ndarray, cfg: dict, names: list[str] | None = None
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Pick the indicator columns from the training rows' label codes.

        Args:
            codes: ``(n, c)`` label codes of the categorical columns, ``-1`` for missing.
            cfg: ``{"min_count", "max_columns"}`` and optionally ``"max_levels"`` (``None`` or
                absent: no cap). A column with more distinct non-missing levels than
                ``max_levels`` gets no indicators; the skipped columns are logged once.
            names: Column names for the log line, else positions are logged.

        Returns:
            ``(column indices, level codes)``, sorted by column then level, or ``None`` when no
            level qualifies.
        """
        cand: list[tuple[int, int, float]] = []
        max_levels = cfg.get("max_levels")
        skipped: list[str] = []
        for j in range(codes.shape[1]):
            values, counts = np.unique(codes[:, j], return_counts=True)
            if max_levels is not None:
                n_levels = int((values >= 0).sum())
                if n_levels > max_levels:
                    skipped.append(f"{names[j] if names is not None else j}={n_levels}")
                    continue
            keep = [int(np.argmin(counts))] if len(values) == 2 else list(range(len(values)))
            cand += [
                (-int(counts[i]), j, float(values[i]))
                for i in keep
                if counts[i] >= cfg["min_count"]
            ]
        if skipped:
            logger.warning(
                "fintfm: probe_onehot max_levels=%d skipped %d column(s) (name=levels): %s",
                max_levels,
                len(skipped),
                ", ".join(skipped),
            )
        if not cand:
            return None
        chosen = sorted(sorted(cand)[: cfg["max_columns"]], key=lambda t: (t[1], t[2]))
        return np.array([c[1] for c in chosen], dtype=np.int64), np.array(
            [c[2] for c in chosen], dtype=np.float64
        )

    def _apply_target_transform(self, y):
        """The training target in the readout's space, and the transform kept for predict.

        Only regression with ``reg_target_transform`` other than ``none`` changes ``y``. The
        transformed target is what everything trained on ``y`` sees: the binned regressor (its
        quantile bin edges are rank-based, so bin membership is the same for a monotone
        transform, and only the bin representatives move into the transformed space), the target
        encoding of categoricals, and the ridge readouts. The binned head's prediction is
        therefore in the transformed space too, and is inverted with the ridge's.

        Args:
            y: Training target.

        Returns:
            ``y``, or the transformed array.
        """
        self._target_tf = None
        self._reg_transform_info = {
            "requested": self._reg_transform_name,
            "effective": "none",
            "note": None,
        }
        if self.problem_type != "regression" or self._reg_transform_name == "none":
            return y
        tf, reason = TargetTransform.fit(
            self._reg_transform_name, np.asarray(y), seed=int(self.random_state)
        )
        if tf is None:
            logger.warning(
                "FinTFM-LinProbe: reg_target_transform=%s not applied: %s",
                self._reg_transform_name,
                reason,
            )
            self._reg_transform_info["note"] = reason
            return y
        self._target_tf = tf
        self._reg_transform_info["effective"] = tf.name
        return tf.forward(np.asarray(y))

    def _finish_reg_levers(self) -> None:
        """Choose the cv blend weight and the smearing factor, on training rows only.

        Both come from held-out predictions of the readout rows collected while the ridge probes
        were fitted (:meth:`_collect_oof`): inner K-fold ridge predictions and, for the cv blend,
        cross-fitted head predictions. No test row is involved.
        """
        import numpy as np

        oof = self._reg_oof
        if not oof["y"]:
            if self._reg_blend == "cv":
                logger.warning(
                    "FinTFM-LinProbe: no held-out rows for the cv blend; using the ridge alone."
                )
            if self._target_tf is not None and self._target_tf.name == "log1p":
                logger.warning(
                    "FinTFM-LinProbe: no held-out rows for smearing; using a factor of 1."
                )
            return
        y = np.concatenate(oof["y"])
        ridge = np.concatenate(oof["ridge"])
        pred = ridge
        if self._reg_blend == "cv":
            head = np.concatenate(oof["head"])
            self._reg_blend_weight = self._blend_weight(y, ridge, head)
            pred = self._reg_blend_weight * ridge + (1.0 - self._reg_blend_weight) * head
            logger.warning(
                "FinTFM-LinProbe: cv blend weight on the ridge %.3f (%d rows).",
                self._reg_blend_weight,
                len(y),
            )
        if self._target_tf is not None and self._target_tf.name == "log1p":
            self._reg_smear = smearing_factor(y - pred)
            logger.warning(
                "FinTFM-LinProbe: log1p smearing factor %.4f (%d held-out rows).",
                self._reg_smear,
                len(y),
            )

    @staticmethod
    def _blend_weight(y: np.ndarray, ridge: np.ndarray, head: np.ndarray) -> float:
        """The weight ``w`` in [0, 1] minimising ``|w * ridge + (1 - w) * head - y|^2``.

        Args:
            y: Training targets of the held-out rows.
            ridge: Their held-out ridge predictions.
            head: Their held-out head predictions.

        Returns:
            The weight on the ridge; 1.0 when the two predictions coincide.
        """
        import numpy as np

        d = ridge - head
        den = float(d @ d)
        if den <= 0.0:
            return 1.0
        return float(np.clip(float((y - head) @ d) / den, 0.0, 1.0))

    def _alpha_grid(self):
        """The ridge alpha grid: ``reg_ridge_alphas`` when set, else the one chosen by the ``raw`` option."""
        import numpy as np

        if self._reg_alphas == "wide":
            return np.logspace(*WIDE_RIDGE_ALPHAS)
        if self._reg_alphas is not None:
            return np.asarray(self._reg_alphas, dtype=np.float64)
        return np.logspace(-3, 5, 17) if self._probe_opts.get("raw") else np.logspace(-3, 3, 13)

    def _collect_oof(
        self, pipe, rep: np.ndarray, y_sub: np.ndarray, clf, binner, X_sub: np.ndarray, fit_idx
    ) -> None:
        """Record held-out ridge (and head) predictions for the readout rows of one probe.

        The ridge predictions come from an inner K-fold over the readout rows, at the alpha the
        fitted ``RidgeCV`` chose, on the same cross-fitted representation. The head predictions
        come from the cross-fit machinery, so no row is scored against a context holding it.

        Args:
            pipe: The fitted ridge pipeline.
            rep: The readout's input for the rows.
            y_sub: Their targets (transformed space).
            clf: The classifier whose context conditions the rows.
            binner: Its regressor's target binner.
            X_sub: The rows' preprocessed features.
            fit_idx: Per row, its index among ``clf``'s fit rows, or ``None``.
        """
        import numpy as np
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import KFold
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        k = self._crossfit_folds if self._crossfit_folds >= 2 else 5
        if len(y_sub) < 2 * k:
            return
        alpha = float(pipe[-1].alpha_)
        ridge = np.empty(len(y_sub), dtype=np.float64)
        for tr, te in KFold(k, shuffle=True, random_state=int(self.random_state)).split(rep):
            ridge[te] = (
                make_pipeline(StandardScaler(), Ridge(alpha=alpha))
                .fit(rep[tr], y_sub[tr])
                .predict(rep[te])
            )
        if self._reg_blend == "cv":
            if fit_idx is None or binner is None:
                return
            head = self._crossfit_apply(
                clf, X_sub, fit_idx, lambda c, Xr: _binned_head(binner, c, Xr)
            )
            self._reg_oof["head"].append(np.asarray(head, dtype=np.float64))
        self._reg_oof["y"].append(np.asarray(y_sub, dtype=np.float64))
        self._reg_oof["ridge"].append(ridge)

    def _log_crossfit(self) -> None:
        """Log what cross-fitting cost in this fit."""
        st = self._crossfit_stats
        if not self._crossfit_on:
            return
        logger.warning(
            "FinTFM-LinProbe: cross-fit K=%d: %d restricted context(s) built, %d context-row and %d "
            "other readout row(s) represented, %d estimator(s) left uncross-fitted.",
            self._crossfit_folds,
            st["restricted"],
            st["context_rows"],
            st["other_rows"],
            st["skipped"],
        )

    def _fit_ridge_probe(self, Xp: np.ndarray, y_arr: np.ndarray) -> None:
        """Fit a ridge readout on the binned regressor's pre-head representation.

        The regression counterpart of the classification probe. ``FinancialTFMRegressor``
        conditions a classifier on quantile-bin indices of the target; its pre-head
        representation of a row is read here and a standardised ``RidgeCV`` maps it to the
        continuous target directly, so the prediction is no longer limited to a weighted
        average of the bin representatives. Up to ``probe_budget`` training rows, drawn
        uniformly at random.

        With a partition the model is a ``PartitionedTFMRegressor``: the global
        fallback regressor gets this probe, and every leaf with its own regressor gets a ridge
        probe of its own on the training rows routed to it, read through that leaf's
        classifier, with the row budget split as in :meth:`_fit_leaf_probes`.

        Args:
            Xp: Preprocessed training features, the rows the regressor was fitted on.
            y_arr: Continuous training targets.
        """
        import numpy as np

        n = len(y_arr)
        seed = int(self.random_state)
        self._fit_raw_conditioner(Xp)
        self._ridge_leaf_probes = None
        if hasattr(self.model, "thresholds_"):
            # Threshold regression (``reg_thresholds``): the readout reads the K threshold logits, not a
            # representation, beside the raw features when the raw option is set.
            self._ridge_probe = self._fit_threshold_ridge(Xp, y_arr, seed)
            return
        part = self.model if hasattr(self.model, "trees_") else None
        glob = part.global_estimator_ if part is not None else self.model
        self._ridge_probe = self._fit_ridge(
            glob.classifier_,
            Xp,
            y_arr,
            np.arange(n),
            self._probe_budget(n),
            seed,
            fit_rows=self._all_rows(n),
            binner=getattr(glob, "binner_", None),
        )
        self._ridge_leaf_probes = None
        if part is None:
            return
        routes = part.route(Xp)
        self._ridge_leaf_probes = []
        for b, leaves in enumerate(part.leaf_estimators_):
            probes = {}
            for leaf, est in leaves.items():
                if est is None:
                    probes[leaf] = None
                    continue
                members = np.flatnonzero(routes[b] == leaf)
                if members.size == 0:
                    # No training row routes here once the replayed preprocessing re-encodes
                    # the categoricals; the leaf then predicts through the global readout.
                    probes[leaf] = None
                    continue
                budget = max(LEAF_MIN_SAMPLE, round(self._probe_budget(n) * len(members) / n))
                probes[leaf] = self._fit_ridge(
                    est.classifier_,
                    Xp,
                    y_arr,
                    members,
                    budget,
                    seed + 1 + b * 1000 + leaf,
                    fit_rows=self._leaf_fit_rows(part, b, members, est.classifier_),
                    binner=getattr(est, "binner_", None),
                )
            self._ridge_leaf_probes.append(probes)
        n_leaf = sum(p is not None for probes in self._ridge_leaf_probes for p in probes.values())
        logger.warning(
            "FinTFM-LinProbe: partitioned regression, %d leaf ridge probe(s) in %d bag(s).",
            n_leaf,
            len(part.leaf_estimators_),
        )

    def _threshold_input(self, X: np.ndarray) -> np.ndarray:
        """The threshold regressor's readout input: its K survival logits, plus raw features.

        Args:
            X: Preprocessed rows.

        Returns:
            ``(n, K)`` logits, widened by the squashed features when the ``raw`` option is set.
        """
        import numpy as np

        blocks = [self.model.readout_features(X)]
        cond = getattr(self, "_raw_conditioner", None)
        if cond is not None:
            blocks.append(self._raw_block(X))
        return blocks[0] if len(blocks) == 1 else np.hstack(blocks)

    def _fit_threshold_ridge(self, Xp: np.ndarray, y_arr: np.ndarray, seed: int):
        """A standardised ``RidgeCV`` on :meth:`_threshold_input` of up to ``probe_budget`` rows.

        Args:
            Xp: Preprocessed training features.
            y_arr: Continuous training targets.
            seed: Seed of the row draw, the same draw :meth:`_fit_ridge` makes.

        Returns:
            The fitted pipeline.
        """
        import numpy as np
        from sklearn.linear_model import RidgeCV
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        rng = np.random.default_rng(seed)
        n = len(y_arr)
        budget = self._probe_budget(n)
        sub = np.sort(rng.choice(n, size=budget, replace=False)) if n > budget else np.arange(n)
        alphas = self._alpha_grid()
        feats = self._threshold_input(Xp[sub])
        return make_pipeline(StandardScaler(), RidgeCV(alphas=alphas)).fit(feats, y_arr[sub])

    def _fit_ridge(
        self,
        clf,
        Xp: np.ndarray,
        y_arr: np.ndarray,
        members: np.ndarray,
        budget: int,
        seed: int,
        fit_rows: np.ndarray | None = None,
        binner=None,
    ):
        """A standardised ``RidgeCV`` on ``clf``'s representation of up to ``budget`` of ``members``.

        Args:
            clf: The fitted classifier inside a binned regressor (a leaf's, or the global one).
            Xp: Preprocessed training features.
            y_arr: Continuous training targets.
            members: Indices of the training rows this probe may learn from.
            budget: Most rows given to the probe, drawn uniformly at random.
            seed: Seed of that draw.
            fit_rows: Sorted indices into ``Xp`` of the rows ``clf`` was fitted on, to cross-fit
                the representation of those among the drawn rows; ``None`` for no cross-fitting.
            binner: ``clf``'s regressor's target binner, for the head predictions of the cv blend.

        Returns:
            The fitted pipeline.
        """
        import numpy as np
        from sklearn.linear_model import RidgeCV
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        rng = np.random.default_rng(seed)
        m = len(members)
        sub = members[np.sort(rng.choice(m, size=budget, replace=False))] if m > budget else members
        # A wider grid when raw columns join the representation: a 1,024-column table can need
        # far more shrinkage than the representation alone.
        alphas = self._alpha_grid()
        fit_idx = None if fit_rows is None else _positions_in(fit_rows, sub)
        rep = self._probe_input(clf, Xp[sub], fit_idx)
        if self._reg_readout_eff == "rff":
            from fintfm.readout.rff import RFFRidgeCV

            head = RFFRidgeCV(alphas=alphas, random_state=seed)
        else:
            head = RidgeCV(alphas=alphas)
        pipe = make_pipeline(StandardScaler(), head).fit(rep, y_arr[sub])
        if self._reg_blend == "cv" or (
            self._target_tf is not None and self._target_tf.name == "log1p"
        ):
            self._collect_oof(pipe, rep, y_arr[sub], clf, binner, Xp[sub], fit_idx)
        return pipe

    def _fit_leaf_probes(self, Xp: np.ndarray, y_arr: np.ndarray) -> None:
        """Fit one probe per leaf context of a partitioned classifier, plus one global probe.

        A leaf's representation is conditioned on that leaf's own context, so representations
        from two leaves are not in one space and a single probe across leaves would mix them.
        Each leaf with its own context therefore gets a probe trained on the training rows the
        tree routes to it, read through that leaf's classifier. The global fallback context
        gets the plain probe. The probe-row budget is split across leaves in proportion to
        their share of the training rows (at least ``LEAF_MIN_SAMPLE`` each), so the total
        representation cost stays near twice the plain probe's rather than growing with the
        number of leaves.

        Args:
            Xp: Preprocessed training features, the rows the partition was fitted on.
            y_arr: Training labels in the original class order.
        """
        import numpy as np

        part = self.model
        n = len(y_arr)
        seed = int(self.random_state)
        self._fit_raw_conditioner(Xp)
        self._linear_probe = self._fit_probe(
            part.global_estimator_,
            Xp,
            y_arr,
            np.arange(n),
            self._probe_budget(n),
            seed,
            fit_rows=self._all_rows(n),
        )
        routes = part.route(Xp)
        self._leaf_probes = []
        for b, leaves in enumerate(part.leaf_estimators_):
            probes = {}
            for leaf, est in leaves.items():
                if est is None:
                    probes[leaf] = None
                    continue
                members = np.flatnonzero(routes[b] == leaf)
                budget = max(LEAF_MIN_SAMPLE, round(self._probe_budget(n) * len(members) / n))
                probes[leaf] = self._fit_probe(
                    est,
                    Xp,
                    y_arr,
                    members,
                    budget,
                    seed + 1 + b * 1000 + leaf,
                    fit_rows=self._leaf_fit_rows(part, b, members, est),
                )
            self._leaf_probes.append(probes)
        n_leaf = sum(p is not None for probes in self._leaf_probes for p in probes.values())
        n_ctx = sum(e is not None for leaves in part.leaf_estimators_ for e in leaves.values())
        logger.warning(
            "FinTFM-LinProbe: partitioned, %d leaf probe(s) over %d leaf context(s) in %d bag(s); global probe %s.",
            n_leaf,
            n_ctx,
            len(part.leaf_estimators_),
            "fitted" if self._linear_probe is not None else "absent (head fallback)",
        )

    def _fit_probe(
        self,
        est,
        Xp: np.ndarray,
        y_arr: np.ndarray,
        members: np.ndarray,
        budget: int,
        seed: int,
        fit_rows: np.ndarray | None = None,
    ):
        """A logistic regression on ``est``'s pre-head representation of ``members``, or ``None``.

        Args:
            est: A fitted ``FinancialTFMClassifier`` (a leaf's, or the global fallback).
            Xp: Preprocessed training features.
            y_arr: Training labels in the original class order.
            members: Indices of the training rows this probe may learn from.
            budget: Most rows given to the probe, drawn by :func:`_probe_rows`.
            seed: Seed of that draw.
            fit_rows: Sorted indices into ``Xp`` of the rows ``est`` was fitted on, to cross-fit
                the representation of those among the drawn rows; ``None`` for no cross-fitting.

        Returns:
            The fitted probe, or ``None`` when no training row routes to the leaf or the draw
            misses a class present among ``members``; the caller then scores with ``est``'s own
            head. A leaf fitted under
            ``missing_classes="global"`` can hold fewer classes than the table; its probe
            covers the leaf's classes and the caller widens it with the global probe.
        """
        import numpy as np

        if members.size == 0:
            # A leaf context to which none of the replayed training rows routes: the replay
            # (preprocess with is_train=False) can encode categoricals differently from the
            # fit, so the tree sends them elsewhere (HR_Analytics, 2026-11-14). The caller
            # scores such a leaf with its own head, as when the draw misses a class.
            return None
        sub = members[_probe_rows(y_arr[members], budget, seed)]
        y_sub = y_arr[sub]
        if len(np.unique(y_sub)) < len(np.unique(y_arr[members])):
            return None
        fit_idx = None if fit_rows is None else _positions_in(fit_rows, sub)
        rep = self._probe_input(est, Xp[sub], fit_idx)
        probe = self._new_classifier_probe(y_sub).fit(rep, y_sub)
        self._collect_floor_oof(probe, rep, y_sub)
        return probe

    def _all_rows(self, n: int) -> np.ndarray | None:
        """Fit rows of an estimator fitted on all ``n`` rows when cross-fitting, else ``None``."""
        import numpy as np

        return np.arange(n) if self._crossfit_on else None

    def _leaf_fit_rows(self, part, bag: int, members: np.ndarray, est) -> np.ndarray | None:
        """Indices into the training matrix of the rows a leaf's estimator was fitted on.

        The partition fits a leaf on the bag's rows that its tree routes to the leaf. ``members``
        holds every training row the same tree routes there, bag or not, so the fit rows are
        their intersection with the bag.

        Args:
            part: The fitted partitioned classifier or regressor.
            bag: Bag number.
            members: Training rows routed to the leaf, sorted.
            est: The leaf's fitted classifier.

        Returns:
            Sorted indices, or ``None`` when not cross-fitting.

        Raises:
            RuntimeError: the count disagrees with the rows ``est`` stored, which means the
                matrix routed here is not the one the partition was fitted on.
        """
        import numpy as np

        if not self._crossfit_on:
            return None
        rows = np.intersect1d(members, part.bag_rows_[bag], assume_unique=True)
        stored = getattr(est, "_raw_y", None)
        if stored is not None and len(stored) != rows.size:
            raise RuntimeError(
                f"cross-fit: leaf holds {len(stored)} fitted rows but routing the training matrix "
                f"gives {rows.size}; the readout rows would be matched to the wrong context rows"
            )
        return rows

    def _restricted(self, clf, rows: np.ndarray):
        """``clf`` restricted to ``rows`` of its fit rows, whatever its ensemble setting.

        ``restrict_context`` refuses an estimator whose ``predict_proba`` averages ensemble
        members, as the default ``n_ensemble`` of 8 does. The readout reads
        ``transform_representation``, which uses the one stored context and no members, so the
        restriction is taken on a shallow copy with ``n_ensemble`` 1 and is used for
        representations only.

        Args:
            clf: A fitted ``FinancialTFMClassifier``.
            rows: Indices into the rows its ``fit`` received.

        Returns:
            The restricted copy.
        """
        shallow = copy.copy(clf)
        shallow.n_ensemble = 1
        return shallow.restrict_context(rows)

    def _crossfit_representation(self, clf, X: np.ndarray, fit_idx: np.ndarray) -> np.ndarray:
        """Pre-head representation of ``X`` with no row seeing its own labelled copy.

        Args:
            clf: A fitted ``FinancialTFMClassifier``.
            X: Preprocessed rows.
            fit_idx: Per row, its index among the rows ``clf``'s ``fit`` received, or -1.

        Returns:
            ``(n, d)`` representations.
        """
        import numpy as np

        return self._crossfit_apply(
            clf,
            X,
            fit_idx,
            lambda c, Xr: np.asarray(
                c.transform_representation(Xr, stage="pre_head"), dtype=np.float64
            ),
        )

    def _crossfit_apply(self, clf, X: np.ndarray, fit_idx: np.ndarray, fn) -> np.ndarray:
        """``fn(estimator, rows)`` for ``X`` with no row seen by a context holding it.

        Rows whose ``fit_idx`` is in ``clf``'s stored context are split into
        ``probe_crossfit_folds`` folds by a seeded shuffle of the context, and each fold is
        evaluated under ``clf`` restricted to the rest of the context. Other rows are evaluated
        under the full context, as test rows are.

        Args:
            clf: A fitted ``FinancialTFMClassifier``.
            X: Preprocessed rows.
            fit_idx: Per row, its index among the rows ``clf``'s ``fit`` received, or -1.
            fn: Maps an estimator and rows to one output row per input row.

        Returns:
            The outputs, stacked in the order of ``X``.
        """
        import numpy as np

        ctx = np.asarray(clf.context_indices_)
        fit_idx = np.asarray(fit_idx)
        stats = self._crossfit_stats
        pos = np.minimum(np.searchsorted(ctx, fit_idx), ctx.size - 1)
        in_ctx = (fit_idx >= 0) & (ctx[pos] == fit_idx)
        k = min(self._crossfit_folds, ctx.size)
        if k < 2 and in_ctx.any():
            stats["skipped"] += 1
            logger.warning(
                "FinTFM-LinProbe: a context of %d row(s) cannot be cross-fitted.", ctx.size
            )
            in_ctx[:] = False
        blocks: list[tuple[np.ndarray, np.ndarray]] = []
        out = np.flatnonzero(~in_ctx)
        if out.size:
            blocks.append((out, fn(clf, X[out])))
            stats["other_rows"] += int(out.size)
        if in_ctx.any():
            parts = np.array_split(
                np.random.default_rng(int(self.random_state) + 7919).permutation(ctx.size), k
            )
            fold_of = np.empty(ctx.size, dtype=np.int64)
            for j, part in enumerate(parts):
                fold_of[part] = j
            row_fold = fold_of[pos]
            for j, part in enumerate(parts):
                rows = np.flatnonzero(in_ctx & (row_fold == j))
                if rows.size == 0:
                    continue
                twin = self._restricted(clf, np.delete(ctx, part))
                stats["restricted"] += 1
                stats["context_rows"] += int(rows.size)
                blocks.append((rows, fn(twin, X[rows])))
        res = np.empty((len(X), *blocks[0][1].shape[1:]), dtype=np.float64)
        for rows, r in blocks:
            res[rows] = r
        return res

    def _score(self, est, probe, X: np.ndarray) -> np.ndarray:
        """Class probabilities for ``X`` from ``probe`` on ``est``'s representation, or ``est``'s head.

        Args:
            est: The fitted classifier whose context conditions the representation.
            probe: Its probe, or ``None`` to use the network's own head.
            X: Preprocessed query rows.

        Returns:
            ``(n, n_classes)`` probabilities in the original class order.
        """
        if probe is None:
            proba = est.predict_proba(X)
            # The head scores the labels it was fitted on, flipped when minority_positive
            # flipped them; the probes always see the original labels.
            return proba[:, ::-1] if self._minority_flipped else proba
        return probe.predict_proba(self._probe_input(est, X))

    def _new_classifier_probe(self, y: np.ndarray):
        """An unfitted classification readout, as :func:`~fintfm.readout.entry.validated_probe` configures it.

        With the ``cv`` option the folds are 3, or fewer when the rarest class in ``y`` has
        fewer than 3 rows (a stratified split needs a row of every class in each fold). With
        fewer than 2 rows of some class there is nothing to select ``C`` on, so the readout is
        the fixed ``C=1`` one. On real data, where every class has at least 3 rows, the readout
        is unchanged.

        Args:
            y: The labels the readout will be fitted on.

        Returns:
            A scikit-learn classifier with ``fit`` and ``predict_proba``.
        """
        import numpy as np
        from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        opts = self._probe_opts
        folds = min(3, int(np.unique(y, return_counts=True)[1].min())) if len(y) else 0
        if opts.get("cv") and folds >= 2:
            probe = LogisticRegressionCV(
                Cs=np.logspace(-3, 2, 6), cv=folds, scoring="neg_log_loss", max_iter=2000
            )
        else:
            probe = LogisticRegression(max_iter=1000)
        return make_pipeline(StandardScaler(), probe) if opts.get("scale") else probe

    def _fit_raw_conditioner(self, Xp: np.ndarray) -> None:
        """Fit the squash applied to the raw features the ``raw`` option appends, if set.

        Args:
            Xp: Preprocessed training features, all training rows.
        """
        from fintfm.inference.preprocess import FeatureConditioner

        self._raw_conditioner = None
        self._raw_spline = None
        if self._probe_opts.get("raw"):
            self._raw_conditioner = FeatureConditioner(
                "squash", random_state=int(self.random_state)
            ).fit(Xp)
            if self._spline_cfg is not None:
                self._raw_spline = self._fit_raw_spline(
                    self._conditioned(self._raw_conditioner, Xp)
                )

    @staticmethod
    def _conditioned(cond, X: np.ndarray) -> np.ndarray:
        """``X`` squashed by the fitted raw conditioner, missing and infinite values made finite."""
        return np.nan_to_num(cond.transform(X), nan=0.0, posinf=3.0, neginf=-3.0).astype(np.float64)

    def _fit_raw_spline(self, Z: np.ndarray) -> dict | None:
        """Fit the spline stage of the raw block on the squashed training rows ``Z``.

        Called only from :meth:`_fit_raw_conditioner`, so it sees exactly the rows the raw
        conditioner is fitted on (training rows; never a query row). A numeric column is expanded
        when its ``n_knots`` quantile knots are all distinct; a column with fewer distinct knot
        positions (a constant, a binary indicator, a column that is mostly one value) is left as it
        is, because tied quantile knots give degenerate basis functions. When more than
        ``max_columns`` columns qualify, only the ``max_columns`` with the most distinct training
        values are expanded (ties: lower column index first) and the rest pass through unchanged;
        this bounds the readout width on wide tables. The choice reads X only, never the labels.
        Categorical columns (their encoded values, positions recorded by :meth:`_encode`) are never
        expanded. The squash bounds each column, so the linear extrapolation of a query value
        outside the training range stays bounded too.

        The transformer reads X only, never the labels, so sharing one across the cross-fit
        folds and partition leaves of a readout leaks no label information.

        Args:
            Z: The squashed, finite training matrix, ``(n, d)``.

        Returns:
            ``{"keep": columns passed through, "sel": columns expanded, "tf": SplineTransformer}``
            or ``None`` when no column qualifies.
        """
        from sklearn.preprocessing import SplineTransformer

        cfg = self._spline_cfg or {}
        n_knots, degree = cfg["n_knots"], cfg["degree"]
        cat = set(self._cat_cols.tolist()) if self._cat_cols is not None else set()
        levels = np.linspace(0.0, 100.0, n_knots)
        sel = [
            j
            for j in range(Z.shape[1])
            if j not in cat and np.unique(np.percentile(Z[:, j], levels)).size == n_knots
        ]
        if not sel:
            return None
        n_qualify = len(sel)
        cap = cfg["max_columns"]
        if n_qualify > cap:
            distinct = {j: np.unique(Z[:, j]).size for j in sel}
            sel = sorted(sorted(sel, key=lambda j: (-distinct[j], j))[:cap])
        sel_arr = np.array(sel, dtype=np.intp)
        keep = np.setdiff1d(np.arange(Z.shape[1]), sel_arr).astype(np.intp)
        tf = SplineTransformer(
            n_knots=n_knots,
            degree=degree,
            knots="quantile",
            extrapolation="linear",
            include_bias=False,
        ).fit(Z[:, sel_arr])
        logger.warning(
            "FinTFM-LinProbe: probe_spline expands %d of %d raw columns%s.",
            len(sel),
            Z.shape[1],
            f" ({n_qualify} qualified; max_columns={cap} bound)" if n_qualify > cap else "",
        )
        return {"keep": keep, "sel": sel_arr, "tf": tf}

    def _raw_block(self, X: np.ndarray) -> np.ndarray:
        """The raw block of the readout input: the squashed features, spline-expanded when fitted.

        Args:
            X: Preprocessed rows.

        Returns:
            The squashed features as float64; with a fitted spline stage, the columns it left
            alone followed by the spline features of the others. Applying it refits nothing.
        """
        Z = self._conditioned(self._raw_conditioner, X)
        sp = getattr(self, "_raw_spline", None)
        if sp is None:
            return Z
        return np.hstack([Z[:, sp["keep"]], sp["tf"].transform(Z[:, sp["sel"]])])

    def _probe_input(self, est, X: np.ndarray, fit_idx: np.ndarray | None = None) -> np.ndarray:
        """The readout's input for rows ``X``: ``est``'s pre-head representation, plus options.

        Args:
            est: The fitted network whose context conditions the representation.
            X: Preprocessed rows.
            fit_idx: Only when fitting a cross-fitted readout: per row, its index among the rows
                ``est`` was fitted on, or -1. ``None`` represents every row under the full
                context, as at predict time.

        Returns:
            ``(n, d)`` representation, widened by the squashed features when the ``raw``
            option is set and by the network's own output when ``head`` is set.
        """
        import numpy as np

        opts = self._probe_opts
        if opts.get("no_rep"):
            blocks = []
        elif fit_idx is not None:
            blocks = [self._crossfit_representation(est, X, fit_idx)]
        else:
            blocks = [
                np.asarray(est.transform_representation(X, stage="pre_head"), dtype=np.float64)
            ]
        cond = getattr(self, "_raw_conditioner", None)
        if cond is not None:
            blocks.append(self._raw_block(X))
            if self._onehot_cfg is not None and self._oh_spec is not None:
                oh = getattr(X, "oh", None)
                if oh is None or len(oh) != len(X):
                    raise AssertionError(
                        "probe_onehot is on but the rows carry no indicator matrix"
                    )
                blocks.append(np.asarray(oh, dtype=np.float64))
        if self._probe_opts.get("head"):
            if self.problem_type == "regression":
                blocks.append(np.asarray(self.model.predict(X), dtype=np.float64).reshape(-1, 1))
            else:
                blocks.append(
                    np.log(np.clip(np.asarray(est.predict_proba(X), dtype=np.float64), 1e-6, 1.0))
                )
        return blocks[0] if len(blocks) == 1 else np.hstack(blocks)

    def _collect_floor_oof(self, probe, rep: np.ndarray, y_sub: np.ndarray) -> None:
        """Keep held-out probabilities of a fitted probe for the ``"cv"`` floor selection.

        Only when ``probe_prob_floor="cv"``. The probe's pipeline is refitted on stratified
        folds of its own training rows (``rep``, ``y_sub``), so only rows of the fold the model
        sees are used, never validation or test rows. When the probe is a
        ``LogisticRegressionCV``, the refits use the ``C`` it selected, so ``C`` is not
        re-selected inside each fold (a mild optimism, the same for every candidate floor).
        At most ``PROB_FLOOR_MAX_ROWS`` rows, drawn with the model seed, are used.

        Args:
            probe: The fitted readout (a pipeline or a classifier).
            rep: The readout's input for its training rows.
            y_sub: Its training labels.
        """
        if self._prob_floor_setting != "cv":
            return
        import numpy as np
        from sklearn.base import clone
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
        from sklearn.pipeline import Pipeline

        if len(np.unique(y_sub)) < 2:
            return
        if len(y_sub) > PROB_FLOOR_MAX_ROWS:
            rows = np.sort(
                np.random.default_rng(int(self.random_state) + 104729).choice(
                    len(y_sub), PROB_FLOOR_MAX_ROWS, replace=False
                )
            )
            rep, y_sub = rep[rows], y_sub[rows]
        classes, counts = np.unique(y_sub, return_counts=True)
        folds = min(3, int(counts.min()))
        if folds < 2 or len(classes) < 2:
            return
        est = clone(probe)
        last_name, last = probe.steps[-1] if isinstance(probe, Pipeline) else (None, probe)
        if hasattr(last, "C_"):
            # A fitted LogisticRegressionCV: refit with the C it chose, fixed.
            plain = LogisticRegression(C=float(np.mean(last.C_)), max_iter=1000)
            est = est.set_params(**{last_name: plain}) if last_name else plain
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=int(self.random_state))
        held = cross_val_predict(est, rep, y_sub, cv=cv, method="predict_proba")
        self._floor_oof.append(
            (np.searchsorted(classes, y_sub), np.asarray(held, dtype=np.float64))
        )

    def _finish_prob_floor(self) -> None:
        """Set :attr:`prob_floor_` from the setting, choosing it on the held-out blocks for ``"cv"``.

        The choice minimises the pooled held-out log loss of ``(1 - e) p + e / K`` over
        :data:`PROB_FLOOR_GRID`; ties go to the smaller ``e``. With no held-out block the floor is 0.
        """
        import numpy as np

        setting = self._prob_floor_setting
        self.prob_floor_losses_ = None
        if setting is None:
            self.prob_floor_ = 0.0
        elif setting != "cv":
            self.prob_floor_ = float(setting)
        elif not self._floor_oof:
            self.prob_floor_ = 0.0
            logger.warning(
                "FinTFM-LinProbe: probe_prob_floor='cv' found no held-out rows, no floor applied."
            )
        else:
            losses = {}
            for e in PROB_FLOOR_GRID:
                total, count = 0.0, 0
                for idx, proba in self._floor_oof:
                    p = (1.0 - e) * proba[np.arange(len(idx)), idx] + e / proba.shape[1]
                    total += float(-np.log(np.clip(p, 1e-15, 1.0)).sum())
                    count += len(idx)
                losses[e] = total / count
            self.prob_floor_ = min(PROB_FLOOR_GRID, key=lambda e: (losses[e], e))
            self.prob_floor_losses_ = losses
            logger.warning(
                "FinTFM-LinProbe: probability floor %g chosen on held-out probe rows.",
                self.prob_floor_,
            )
        self._floor_oof = []

    def _apply_prob_floor(self, proba: np.ndarray) -> np.ndarray:
        """``(1 - e) p + e / K`` with ``e = prob_floor_``; ``proba`` itself when ``e`` is 0.

        Args:
            proba: Class probabilities, ``(n, K)``, or the positive-class column ``(n,)`` of a
                binary problem.

        Returns:
            The floored probabilities, same shape; rows still sum to 1.
        """
        e = getattr(self, "prob_floor_", 0.0)
        if not e:
            return proba
        if proba.ndim == 1:
            return (1.0 - e) * proba + e / 2.0
        return (1.0 - e) * proba + e / proba.shape[1]

    def _predict_ridge(self, Xp: np.ndarray) -> np.ndarray:
        """The ridge readout's prediction for preprocessed rows, in the transformed target space."""
        import numpy as np

        if hasattr(self.model, "thresholds_"):
            return self._ridge_probe.predict(self._threshold_input(Xp))
        if getattr(self, "_ridge_leaf_probes", None) is None:
            return self._ridge_probe.predict(self._probe_input(self.model.classifier_, Xp))
        part = self.model
        routes = part.route(Xp)
        out = np.zeros(Xp.shape[0], dtype=np.float64)
        for b, leaves in enumerate(part.leaf_estimators_):
            for leaf in np.unique(routes[b]):
                rows = np.flatnonzero(routes[b] == leaf)
                est, probe = leaves.get(int(leaf)), self._ridge_leaf_probes[b].get(int(leaf))
                if est is None or probe is None:
                    est, probe = part.global_estimator_, self._ridge_probe
                out[rows] += probe.predict(self._probe_input(est.classifier_, Xp[rows]))
        return out / len(part.leaf_estimators_)


class _AugmentedMatrix(np.ndarray):
    """A float32 feature matrix that carries an extra per-row block (the ``probe_onehot`` indicators).

    Behaves as the plain matrix to the classifier and the partition router (same shape and values).
    Row selection with an index array, boolean mask or slice selects the same rows of the extra
    block, so every ``Xp[rows]`` the readout code takes keeps the two aligned; any other indexing
    (column selection, scalars) returns a plain array. Pickling drops the extra block.
    """

    oh: np.ndarray | None = None

    @classmethod
    def attach(cls, arr: np.ndarray, extra: np.ndarray) -> _AugmentedMatrix:
        """``arr`` as a matrix carrying ``extra``.

        Args:
            arr: The ``(n, d)`` float32 feature matrix.
            extra: ``(n, k)`` per-row block.

        Returns:
            A view of ``arr`` carrying ``extra`` as ``oh``.
        """
        out = arr.view(cls)
        out.oh = extra
        return out

    def __getitem__(self, key):
        """Select rows from both blocks; any other indexing returns a plain array."""
        out = super().__getitem__(key)
        if (
            isinstance(key, tuple)
            or not isinstance(out, _AugmentedMatrix)
            or out.ndim != 2
            or self.oh is None
        ):
            return out.view(np.ndarray) if isinstance(out, np.ndarray) else out
        out.oh = self.oh[key]
        return out

    def __reduce__(self):
        """Pickle as a plain array."""
        return self.view(np.ndarray).__reduce__()


def _as_f32(a) -> np.ndarray:
    """``a`` as a float32 array, keeping an :class:`_AugmentedMatrix` (already float32) as it is."""
    return a if isinstance(a, _AugmentedMatrix) else np.asarray(a, dtype=np.float32)


def _binned_head(binner, est, X: np.ndarray) -> np.ndarray:
    """The binned regressor's mean prediction from classifier ``est`` for rows ``X``.

    Args:
        binner: The regressor's fitted target binner.
        est: A fitted classifier over the bin indices, possibly with a restricted context.
        X: Preprocessed rows.

    Returns:
        ``(n,)`` expected values, in the target space the regressor was fitted in.
    """
    import numpy as np

    proba = est.predict_proba(X)
    full = np.zeros((proba.shape[0], binner.n_bins_), dtype=np.float64)
    full[:, np.asarray(est.classes_).astype(int)] = proba
    return binner.expected_value(full)


def _logspace(lo: float, hi: float, n: int) -> list[float]:
    """``n`` values from ``10**lo`` to ``10**hi``, log-spaced."""
    import numpy as np

    return list(np.logspace(lo, hi, n))


def _positions_in(fit_rows: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Position of each of ``rows`` within sorted ``fit_rows``, or -1 where it is absent.

    Args:
        fit_rows: Sorted, distinct indices of the rows an estimator was fitted on.
        rows: Indices to look up.

    Returns:
        Integer array aligned with ``rows``.
    """
    import numpy as np

    if fit_rows.size == 0:
        return np.full(len(rows), -1, dtype=np.int64)
    pos = np.minimum(np.searchsorted(fit_rows, rows), fit_rows.size - 1)
    return np.where(fit_rows[pos] == rows, pos, -1).astype(np.int64)


def _probe_rows(y_arr: np.ndarray, budget: int, seed: int) -> np.ndarray:
    """Positions in ``y_arr`` given to a probe: every minority row, padded with majority rows.

    Args:
        y_arr: Labels of the candidate rows.
        budget: Most rows returned.
        seed: Seed of the random draw.

    Returns:
        Sorted or concatenated positions into ``y_arr``, at most ``budget`` of them.
    """
    import numpy as np

    classes, counts = np.unique(y_arr, return_counts=True)
    if len(classes) > 2 or (len(classes) == 2 and set(classes.tolist()) != {0, 1}):
        return _stratified_rows(y_arr, classes, counts, budget, seed)
    positive = np.where(y_arr == 1)[0]
    negative = np.where(y_arr == 0)[0]
    if len(y_arr) > budget and len(positive) >= budget:
        # Class 1 alone fills the budget (class 1 is the majority on some tasks, e.g.
        # Amazon_employee_access at 0.94). Keeping every class-1 row would leave no room
        # for class 0 and force the head fallback, so cap both classes instead:
        # each gets up to half the budget and the larger one fills what the smaller leaves.
        rng = np.random.default_rng(seed)
        n_negative = min(len(negative), budget // 2)
        n_positive = budget - n_negative
        negative = rng.choice(negative, size=n_negative, replace=False)
        positive = rng.choice(positive, size=n_positive, replace=False)
        return np.sort(np.concatenate([positive, negative]))
    if len(y_arr) > budget:
        rng = np.random.default_rng(seed)
        n_negative = budget - len(positive)
        if n_negative < len(negative):
            negative = rng.choice(negative, size=n_negative, replace=False)
        return np.concatenate([positive, negative])
    return np.arange(len(y_arr))


def _stratified_rows(
    y_arr: np.ndarray, classes: np.ndarray, counts: np.ndarray, budget: int, seed: int
) -> np.ndarray:
    """Positions for a probe on labels other than {0, 1}: every class kept, smallest first.

    The binary rule in :func:`_probe_rows` reads class 1 as the minority and class 0 as the
    rest, so on a multiclass ``y`` it would drop every class above 1. Here the budget is
    water-filled across classes instead: classes are visited from smallest to largest, each
    taking all its rows or an equal share of what is left, whichever is smaller, so a rare
    class is kept whole and the largest classes absorb the cut.

    Args:
        y_arr: Labels of the candidate rows.
        classes: The distinct labels, from ``np.unique``.
        counts: Rows per label, aligned with ``classes``.
        budget: Most rows returned.
        seed: Seed of the random draw.

    Returns:
        Sorted positions into ``y_arr``, at most ``budget`` of them, every class present.
    """
    import numpy as np

    if len(y_arr) <= budget:
        return np.arange(len(y_arr))
    rng = np.random.default_rng(seed)
    order = np.argsort(counts, kind="stable")
    remaining = budget
    picked = []
    for i, c in enumerate(order):
        take = min(int(counts[c]), remaining // (len(order) - i))
        rows = np.flatnonzero(y_arr == classes[c])
        picked.append(rows if take >= len(rows) else rng.choice(rows, size=take, replace=False))
        remaining -= take
    return np.sort(np.concatenate(picked))
