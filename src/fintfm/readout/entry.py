"""The measured entry settings of the linear readout, and the checks every setting passes.

``entry_defaults`` is the single source of truth for the configuration measured in fintfm
``docs/results/FINDINGS.md`` §232, §235, §237, §283, §290, §293 and §295. The TabArena wrapper builds its
default hyperparameters from it, so a change here changes the entry everywhere. The ``check_*``
functions validate one setting each and are shared by :class:`~fintfm.readout.estimator.LinearReadoutModel`
and any caller that wants to reject a value early.
"""

from __future__ import annotations

import copy
import math
from typing import Any

from fintfm.readout.checkpoints import (
    BINARY_CHECKPOINT,
    MULTICLASS_CHECKPOINT,
    REGRESSION_CHECKPOINT,
)
from fintfm.readout.target_transform import TARGET_TRANSFORMS

#: Problem types a checkpoint, partition and probe can be set for.
PROBLEM_TYPES = ("binary", "multiclass", "regression")

#: Rows given to the downstream classifier's training step. Mirrors
#: ``fintfm.experiments.representation_transfer.TRAIN_SAMPLE_SIZE``: every minority-class row
#: is kept, then padded with a random majority-class sample up to this budget, because a plain
#: random subsample at a low positive rate can miss the minority class outright.
TRAIN_SAMPLE_SIZE = 20_000
#: Fewest probe rows a partition leaf is given, however small its share of the training rows.
LEAF_MIN_SAMPLE = 2_000
#: Keys a probe option dict accepts; see :func:`validated_probe`.
PROBE_OPTIONS = frozenset({"scale", "cv", "raw", "head", "no_rep"})
#: Cross-fit folds of the binary entry and the fallback when the setting is absent. The
#: multiclass and regression entries use 0.
PROBE_CROSSFIT_FOLDS = 5
#: Deepest tree ``fintfm.inference.large_n.PartitionedTFMClassifier`` and ``PartitionedTFMRegressor``
#: accept (``max_depth`` is checked to lie in [1, 3]); ``partition_depth_cap`` is clipped to it.
PARTITION_LIBRARY_MAX_DEPTH = 3
#: Defaults of ``partition_scale_k`` and ``partition_depth_cap``.
PARTITION_SCALE_K = 4.0
PARTITION_DEPTH_CAP = 5
#: Defaults of ``probe_onehot``: a level needs at least ``min_count`` training rows to get an
#: indicator column, and at most ``max_columns`` indicator columns are added in total. There is
#: no default level cap: ``max_levels`` is ``None`` (off) unless set.
ONEHOT_MIN_COUNT = 3
ONEHOT_MAX_COLUMNS = 256
#: Defaults of ``probe_spline``: knots per numeric column (placed at training-row quantiles), the
#: B-spline degree, and the most numeric columns that are expanded.
SPLINE_N_KNOTS = 5
SPLINE_DEGREE = 3
SPLINE_MAX_COLUMNS = 64
#: The alpha grid ``reg_ridge_alphas="wide"`` names.
WIDE_RIDGE_ALPHAS = (-3.0, 7.0, 21)
#: Candidate floors ``e`` of the ``probe_prob_floor="cv"`` selection; 0 (no floor) is the first.
PROB_FLOOR_GRID = (0.0, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2)
#: Most rows of one probe's held-out predictions kept for the floor selection.
PROB_FLOOR_MAX_ROWS = 5_000
#: Values of ``reg_readout``: the linear ridge, or ridge on random Fourier features.
REG_READOUTS = ("ridge", "rff")

#: The published checkpoint of each problem type, pinned by commit and SHA-256.
_ENTRY_CHECKPOINT = {
    "binary": BINARY_CHECKPOINT,
    "multiclass": MULTICLASS_CHECKPOINT,
    "regression": REGRESSION_CHECKPOINT,
}
#: Partition of each problem type: ``n_bags`` of the tree partition.
_ENTRY_PARTITION = {
    "binary": {"n_bags": 4},
    "multiclass": {"n_bags": 8},
    "regression": {"n_bags": 4},
}
#: ``partition_bags_max_rows`` per problem type: the binary entry's four bags apply only to fits of at most
#: 5,000 training rows (FINDINGS §286); larger binary fits use the single partition. Multiclass and
#: regression are not gated.
_ENTRY_BAGS_MAX_ROWS = {"binary": 5_000, "multiclass": None, "regression": None}
#: ``probe_spline`` per problem type (``True`` is ``SPLINE_N_KNOTS`` knots, degree ``SPLINE_DEGREE``, at most
#: ``SPLINE_MAX_COLUMNS`` columns) and ``spline_max_rows``, the fit-row count above which the splines are off
#: (FINDINGS §290, ``fintfm_g287_bag4_spline``, binary; §293, ``fintfm_g292_emr``, multiclass and
#: regression). Multiclass splines were measured only on fits of at most 10,000 rows, so they are gated
#: there like binary's; regression was measured at every size and is not gated.
#: Splines apply only with the ``raw`` probe option, which all three entries have.
_ENTRY_PROBE_SPLINE = {"binary": True, "multiclass": True, "regression": True}
_ENTRY_SPLINE_MAX_ROWS = {"binary": 10_000, "multiclass": 10_000, "regression": None}
_ENTRY_PROBE = {
    "binary": {"scale": True, "cv": True, "raw": True},
    "multiclass": {"scale": True, "cv": True, "raw": True},
    "regression": {"raw": True},
}
#: Multiclass probability floor ``probe_prob_floor`` of the entry. ``"cv"`` picks ``e`` in
#: ``(1 - e) p + e / K`` by log loss on held-out readout rows (never validation or test rows).
#: Adopted for multiclass only (fintfm FINDINGS §265, §269): 4 of 8 tables better, 3 worse, 1 tied, a
#: configuration choice on one fold, not a measured improvement.
_ENTRY_PROB_FLOOR = {"binary": None, "multiclass": "cv", "regression": None}
#: Binary depth scaling and probe budget, adopted together for binary only (§257).
_ENTRY_SCALE_WITH_N = {"binary": True, "multiclass": False, "regression": False}
_ENTRY_PROBE_BUDGET = {
    "binary": 50_000,
    "multiclass": TRAIN_SAMPLE_SIZE,
    "regression": TRAIN_SAMPLE_SIZE,
}
#: One-hot categorical indicators of the multiclass readout (``probe_onehot``), as measured in
#: fintfm FINDINGS §283. ``None`` is off.
_ENTRY_PROBE_ONEHOT = {
    "binary": None,
    "multiclass": {
        "min_count": ONEHOT_MIN_COUNT,
        "max_columns": ONEHOT_MAX_COLUMNS,
        "max_levels": 64,
    },
    "regression": None,
}
#: Regression readout and the training-row cap above which it falls back to the ridge (§283).
_ENTRY_REG_READOUT = {"binary": "ridge", "multiclass": "ridge", "regression": "rff"}
_ENTRY_REG_RFF_MAX_ROWS = {"binary": None, "multiclass": None, "regression": 5_000}
#: Clip of regression predictions to the widened training target range (§295); off elsewhere.
_ENTRY_REG_CLIP_MARGIN = {"binary": None, "multiclass": None, "regression": 0.1}
#: Cross-fit folds of the entry. Binary cross-fits (§254). Multiclass and regression do not (§255).
_ENTRY_CROSSFIT_FOLDS = {"binary": PROBE_CROSSFIT_FOLDS, "multiclass": 0, "regression": 0}


def entry_defaults(problem_type: str) -> dict[str, Any]:
    """The measured entry's settings for one problem type, as keyword arguments for the estimator.

    The keys are ``checkpoint`` (a published reference dict, see :mod:`fintfm.readout.checkpoints`)
    and every setting :class:`~fintfm.readout.estimator.LinearReadoutModel` reads. The defaults are
    the configuration measured in fintfm ``docs/results/FINDINGS.md`` §283: all three checkpoints use
    ``feature_transform`` ``squash`` and ``max_features_strategy`` ``select``; binary cross-fits and
    scales the partition depth with the row count; multiclass adds one-hot indicators and a
    probability floor; regression reads a random-Fourier-feature ridge and clips to the widened target
    range. Every call returns fresh copies, so a caller may modify the result.

    Args:
        problem_type: ``binary``, ``multiclass`` or ``regression``.

    Returns:
        The entry for that problem type. Pass ``checkpoint`` as the estimator's second argument and
        the rest as keyword settings.

    Raises:
        ValueError: ``problem_type`` is not one of :data:`PROBLEM_TYPES`.
    """
    if problem_type not in PROBLEM_TYPES:
        raise ValueError(f"fintfm supports {PROBLEM_TYPES}, not problem type {problem_type!r}")
    pt = problem_type
    onehot = _ENTRY_PROBE_ONEHOT[pt]
    return {
        "checkpoint": copy.deepcopy(_ENTRY_CHECKPOINT[pt]),
        # D12: a single column-identity draw is the configuration that decision's own text calls
        # incomplete, so ensemble over draws by default.
        "n_ensemble": 8,
        # docs/results/FINDINGS.md §83/§84 measured context as nearly flat on a 1M-row panel;
        # 1000 is the largest value that is cheap on every machine this runs on.
        "max_context": 1000,
        # §81: identity-preserving, and materially cheaper wherever attention scores are
        # materialised (CPU and MPS). No effect on CUDA (§95).
        "feature_chunk": 16,
        "feature_transform": "squash",
        "max_features_strategy": "select",
        "n_bins": 10,
        "partition": copy.deepcopy(_ENTRY_PARTITION[pt]),
        "probe": copy.deepcopy(_ENTRY_PROBE[pt]),
        "probe_crossfit_folds": _ENTRY_CROSSFIT_FOLDS[pt],
        "reg_target_transform": "none",
        "reg_readout_blend": None,
        "reg_ridge_alphas": None,
        "reg_readout": _ENTRY_REG_READOUT[pt],
        "reg_rff_max_rows": _ENTRY_REG_RFF_MAX_ROWS[pt],
        "reg_clip_margin": _ENTRY_REG_CLIP_MARGIN[pt],
        "partition_bags_max_rows": _ENTRY_BAGS_MAX_ROWS[pt],
        "probe_spline": _ENTRY_PROBE_SPLINE[pt],
        "spline_max_rows": _ENTRY_SPLINE_MAX_ROWS[pt],
        "partition_scale_with_n": _ENTRY_SCALE_WITH_N[pt],
        "partition_scale_k": PARTITION_SCALE_K,
        "partition_depth_cap": PARTITION_DEPTH_CAP,
        "probe_budget": _ENTRY_PROBE_BUDGET[pt],
        "probe_prob_floor": _ENTRY_PROB_FLOOR[pt],
        "probe_onehot": None if onehot is None else dict(onehot),
    }


def check_spline(value, *, source: str) -> dict | None:
    """Validate a ``probe_spline`` value and resolve it to its settings.

    Args:
        value: ``None``/``False`` (off), ``True`` (defaults) or a dict with the optional keys
            ``n_knots`` (an int >= 3: knots per numeric column, at training-row quantiles),
            ``degree`` (an int in 1..3: the B-spline degree) and ``max_columns`` (an int >= 1:
            the most numeric columns expanded; see :meth:`LinearReadoutModel._fit_raw_spline`).
        source: Where the value came from, for the error message.

    Returns:
        ``None`` for off, else ``{"n_knots": int, "degree": int, "max_columns": int}``.

    Raises:
        ValueError: the value is of another type, names an unknown key, ``n_knots`` is below
            3, ``degree`` is outside 1..3 or ``max_columns`` is below 1.
    """
    if value is None or value is False:
        return None
    cfg = {"n_knots": SPLINE_N_KNOTS, "degree": SPLINE_DEGREE, "max_columns": SPLINE_MAX_COLUMNS}
    if value is True:
        return cfg
    if not isinstance(value, dict):
        raise ValueError(f"{source} must be true, false or a JSON object, got {value!r}")  # noqa: TRY004
    unknown = set(value) - set(cfg)
    if unknown:
        raise ValueError(f"{source} has unknown key(s) {sorted(unknown)}; known: {sorted(cfg)}")
    bounds = {"n_knots": (3, None), "degree": (1, 3), "max_columns": (1, None)}
    for key, v in value.items():
        lo, hi = bounds[key]
        if isinstance(v, bool) or not isinstance(v, int) or v < lo or (hi is not None and v > hi):
            want = f">= {lo}" if hi is None else f"in {lo}..{hi}"
            raise ValueError(f"{source}: {key} must be an integer {want}, got {v!r}")
        cfg[key] = v
    return cfg


def check_onehot(value, *, source: str) -> dict | None:
    """Validate a ``probe_onehot`` value and resolve it to its settings.

    Args:
        value: ``None``/``False`` (off), ``True`` (defaults) or a dict with the optional keys
            ``min_count`` (an int >= 1: a level needs this many training rows to get an
            indicator) and ``max_columns`` (an int >= 1: the most indicator columns in total;
            when more levels qualify, the most frequent ones are kept) and ``max_levels`` (an
            int >= 2, default ``None`` = no cap: a categorical column with more distinct
            levels than this in the training rows, missing not counted, gets no indicator
            columns at all; label-free).
        source: Where the value came from, for the error message.

    Returns:
        ``None`` for off, else ``{"min_count": int, "max_columns": int, "max_levels": int | None}``.

    Raises:
        ValueError: the value is of another type, names an unknown key, or a bound is not
            an integer >= 1 (``max_levels``: >= 2).
    """
    if value is None or value is False:
        return None
    cfg = {"min_count": ONEHOT_MIN_COUNT, "max_columns": ONEHOT_MAX_COLUMNS, "max_levels": None}
    if value is True:
        return cfg
    if not isinstance(value, dict):
        raise ValueError(f"{source} must be true, false or a JSON object, got {value!r}")  # noqa: TRY004
    unknown = set(value) - set(cfg)
    if unknown:
        raise ValueError(f"{source} has unknown key(s) {sorted(unknown)}; known: {sorted(cfg)}")
    for key, v in value.items():
        if key == "max_levels" and v is None:
            cfg[key] = None
            continue
        lo = 2 if key == "max_levels" else 1
        if isinstance(v, bool) or not isinstance(v, int) or v < lo:
            raise ValueError(f"{source}: {key} must be an integer >= {lo}, got {v!r}")
        cfg[key] = v
    return cfg


def check_prob_floor(value, *, source: str):
    """Validate a ``probe_prob_floor`` value.

    Args:
        value: ``None``, ``"cv"`` or a number in [0, 1).
        source: Where the value came from, for the error message.

    Returns:
        ``None``, ``"cv"`` or the number as a float.

    Raises:
        ValueError: the value is none of those.
    """
    if value is None or (isinstance(value, str) and value == "cv"):
        return value
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= value < 1.0:
        raise ValueError(f"{source} must be None, 'cv' or a number in [0, 1), got {value!r}")
    return float(value)


def check_scale_k(value, *, source: str) -> float:
    """Validate a ``partition_scale_k`` value.

    Args:
        value: A finite number > 0.
        source: Where the value came from, for the error message.

    Returns:
        The value as a float.

    Raises:
        ValueError: the value is not a finite number > 0.
    """
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{source} must be a positive number, got {value!r}")
    return float(value)


def check_probe_budget(value, *, source: str):
    """Validate a ``probe_budget`` value.

    Args:
        value: An int >= 1 or ``"all"``.
        source: Where the value came from, for the error message.

    Returns:
        The value.

    Raises:
        ValueError: the value is neither ``"all"`` nor an integer >= 1.
    """
    if value == "all":
        return "all"
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{source} must be a positive integer or 'all', got {value!r}")
    return value


def check_reg_transform(value, *, source: str) -> str:
    """Validate a ``reg_target_transform`` value.

    Args:
        value: ``None`` or one of :data:`~fintfm.readout.target_transform.TARGET_TRANSFORMS`.
        source: Where the value came from, for the error message.

    Returns:
        The transform name; ``"none"`` for ``None``.

    Raises:
        ValueError: the value is not a known transform.
    """
    if value is None:
        return "none"
    if value not in TARGET_TRANSFORMS:
        raise ValueError(f"{source} must be one of {TARGET_TRANSFORMS}, got {value!r}")
    return value


def check_reg_blend(value, *, source: str):
    """Validate a ``reg_readout_blend`` value.

    Args:
        value: ``None`` (no blend), ``"cv"`` or a number in [0, 1], the weight on the ridge.
        source: Where the value came from, for the error message.

    Returns:
        ``None``, ``"cv"`` or the weight as a float.

    Raises:
        ValueError: the value is none of those.
    """
    if value is None or value == "cv":
        return value
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise ValueError(f"{source} must be None, 'cv' or a number in [0, 1], got {value!r}")
    return float(value)


def check_reg_readout(value, *, source: str) -> str:
    """Validate a ``reg_readout`` value.

    Args:
        value: ``None`` (the ridge) or one of :data:`REG_READOUTS`.
        source: Where the value came from, for the error message.

    Returns:
        ``"ridge"`` or ``"rff"``.

    Raises:
        ValueError: the value is not a known readout.
    """
    if value is None:
        return "ridge"
    if value not in REG_READOUTS:
        raise ValueError(f"{source} must be one of {REG_READOUTS}, got {value!r}")
    return value


def check_reg_clip_margin(value, *, source: str) -> float | None:
    """Validate a ``reg_clip_margin`` value.

    Args:
        value: ``None`` (no clip) or a finite number >= 0.
        source: Where the value came from, for the error message.

    Returns:
        ``None`` or the value as a float.

    Raises:
        ValueError: the value is not ``None`` and not a finite number >= 0.
    """
    import math

    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{source} must be None or a finite number >= 0, got {value!r}")
    return float(value)


def check_reg_rff_max_rows(value, *, source: str) -> int | None:
    """Validate a ``reg_rff_max_rows`` value.

    Args:
        value: ``None`` (no cap) or an int >= 1.
        source: Where the value came from, for the error message.

    Returns:
        ``None`` or the int.

    Raises:
        ValueError: the value is not ``None`` and not an integer >= 1.
    """
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{source} must be None or an integer >= 1, got {value!r}")
    return value


def check_bags_max_rows(value, *, source: str) -> int | None:
    """Validate a ``partition_bags_max_rows`` value.

    Args:
        value: ``None`` (no gate) or an int >= 1.
        source: Where the value came from, for the error message.

    Returns:
        ``None`` or the int.

    Raises:
        ValueError: the value is not ``None`` and not an integer >= 1.
    """
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{source} must be None or an integer >= 1, got {value!r}")
    return value


def check_spline_max_rows(value, *, source: str) -> int | None:
    """Validate a ``spline_max_rows`` value.

    Args:
        value: ``None`` (no gate) or an int >= 1.
        source: Where the value came from, for the error message.

    Returns:
        ``None`` or the int.

    Raises:
        ValueError: the value is not ``None`` and not an integer >= 1.
    """
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{source} must be None or an integer >= 1, got {value!r}")
    return value


def check_reg_alphas(value, *, source: str):
    """Validate a ``reg_ridge_alphas`` value.

    Args:
        value: ``None`` (the readout's own grid), ``"wide"`` or a non-empty list of positive finite numbers.
        source: Where the value came from, for the error message.

    Returns:
        ``None``, ``"wide"`` or the grid as a list of floats.

    Raises:
        ValueError: the value is none of those.
    """
    if value is None or value == "wide":
        return value
    try:
        grid = [float(a) for a in value]
    except (TypeError, ValueError):
        raise ValueError(
            f"{source} must be None, 'wide' or a list of positive numbers, got {value!r}"
        ) from None
    if not grid or any(not math.isfinite(a) or a <= 0 for a in grid):
        raise ValueError(
            f"{source} must be a non-empty list of positive finite numbers, got {value!r}"
        )
    return grid


def validated_probe(opts, *, source: str) -> dict:
    """Check readout options; ``None`` or ``{}`` is the plain readout.

    The plain readout is a ``LogisticRegression`` at ``C=1`` on the raw representation, and
    a standardised ``RidgeCV`` for regression. The keys, each a boolean:

    - ``scale``: standardise the probe's inputs before the logistic regression.
    - ``cv``: choose the logistic regression's ``C`` by 3-fold cross-validated log loss on
      the probe rows, instead of the fixed ``C=1``.
    - ``raw``: append the row's own features to the representation, squashed (median, IQR
      and soft clip, fitted on the training rows) with missing values at the median. The
      readout can then reach a linear model of the features as well as the network's
      representation of them. Applies to the ridge readout too.
    - ``head``: append the network's own output for the row: the log of its head's class
      probabilities, or for regression the binned regressor's own prediction. The readout
      can then keep what the head gets right and correct the rest.
    - ``no_rep``: with ``raw``, drop the representation and read the squashed features
      alone, on the same leaves and rows: the matched raw-only control (fintfm FINDINGS
      §238). The network then only supplies the head for leaves whose probe is ``None``.

    Args:
        opts: The ``probe_<type>`` hyperparameter: ``None`` or a dict of the keys above.
        source: Where the value came from, for the message.

    Returns:
        The options as a dict, empty for the plain readout.

    Raises:
        ValueError: the value is not a dict, names an unknown key, or sets ``no_rep``
            without ``raw``.
    """
    if opts is None:
        return {}
    if not isinstance(opts, dict):
        raise ValueError(f"{source} must be a JSON object, got {opts!r}")  # noqa: TRY004
    unknown = set(opts) - PROBE_OPTIONS
    if unknown:
        raise ValueError(
            f"{source} has unknown key(s) {sorted(unknown)}; known: {sorted(PROBE_OPTIONS)}"
        )
    if opts.get("no_rep") and not opts.get("raw"):
        raise ValueError(f"{source}: no_rep needs raw, the readout would have no input")
    return dict(opts)


def validated_partition(value, *, source: str) -> dict | None:
    """Check a partition setting; ``None`` means no partition, ``{}`` the class defaults.

    Fix 4 of fintfm's TabArena loss analysis: wrap the classifier in
    ``fintfm.inference.large_n.PartitionedTFMClassifier``, a shallow decision tree whose
    leaves each carry their own context, with the global context as the fallback for a
    leaf too small or missing a class (fintfm ``docs/results/FINDINGS.md`` §169, +0.0375
    mean AP on V4FinBench at depth 2 with a 5% leaf floor). The value is a dict of that
    class's arguments; ``{}`` takes its defaults, which are §169's. ``estimator`` and
    ``random_state`` are set by the adapter and refused here, and a misspelled key raises
    rather than being ignored, because an ignored setting completes the run and reports
    the default's numbers under the arm's name.

    Args:
        value: ``None`` or a dict of partition arguments.
        source: Where the value came from, for the message.

    Returns:
        ``value``, validated.

    Raises:
        ValueError: ``value`` is neither ``None`` nor a dict, or names an unknown or
            reserved argument.
    """
    import inspect

    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"{source} must be a JSON object or null, got {type(value).__name__}")  # noqa: TRY004

    from fintfm.inference.large_n import PartitionedTFMClassifier

    reserved = {"self", "estimator", "random_state"}
    allowed = set(inspect.signature(PartitionedTFMClassifier.__init__).parameters) - reserved
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(
            f"{source} names {unknown}, which are not PartitionedTFMClassifier "
            f"arguments the adapter lets you set; allowed: {sorted(allowed)}"
        )
    return value
