"""Native categorical handling: out-of-fold smoothed target statistics.

Why this module exists
----------------------

``docs/results/FINDINGS.md`` §100 measured where this project's TabArena deficit actually lives. On
the eight datasets with no categorical columns, fintfm is 0.0320 ROC-AUC behind tuned logistic
regression. On the nine that are more than half categorical it is 0.0894 behind, and on the
five with a level count above 25 it is 0.1109 behind. The correlation between the per-dataset
gap and log maximum cardinality is **-0.668**. Categorical content, not the model, accounts
for roughly 82% of the distance to the arm ranked #89.

The cause is not subtle. ``modeling/model.py`` embeds each cell as a **numeric scalar**, so a
categorical column has to arrive as a number. The integration used in §98 label-encoded it,
which asserts that ``Amazon_employee_access``'s resource code 4127 lies between 4126 and 4128
on some meaningful axis. It does not, and the measured consequence was 0.5455 ROC-AUC against
a baseline's 0.8442 -- close enough to chance to be reported as a floor rather than a score.

What this replaces the arbitrary order with
-------------------------------------------

For each level of a categorical column, a **smoothed estimate of the target rate among
context rows carrying that level**:

``encode(v) = (sum of y over rows with level v + m * prior) / (count of rows with level v + m)``

That is an ordered quantity by construction -- higher means likelier positive -- so the
scalar the model receives means something on the axis it is read on. ``m`` shrinks rare levels
toward the global base rate, which is what stops a level seen twice from being encoded as
1.0.

The out-of-fold requirement, which is the whole difficulty
----------------------------------------------------------

Naive target encoding is notorious, and the reason is worth stating precisely rather than
gesturing at. If row ``i``'s own label is included in the statistic that encodes row ``i``,
then on a high-cardinality column where most levels appear once, ``encode(v_i)`` is very
nearly ``y_i``. The encoded column becomes an almost perfect predictor **inside the context**
and carries no signal at all at query time, where the level is unseen. A model conditioned on
such a context learns to trust a feature that will not be there. This is a leak that makes
the model worse, not a leak that flatters the score -- which is why it survives a careless
validation and shows up only as inexplicable weakness.

:class:`CategoricalTargetEncoder` avoids it by computing each context row's encoding from a
**K-fold partition that excludes the fold the row is in**, so no row contributes to its own
statistic. Query rows are encoded from the full context, since their labels are not available
to leak. This is the K-fold analogue of CatBoost's ordered target statistics; it is chosen
over the ordered variant because it needs no row ordering and is exactly reproducible from a
seed.

**Continuous targets take the same treatment**, with the level's smoothed *mean* target in
place of its smoothed rate. The arithmetic is identical -- a rate is a mean of an indicator --
and so is the leak, so the out-of-fold machinery applies unchanged. This matters because 12
of TabArena's 46 eligible datasets are regression, and frequency-encoding all of them would
have thrown away the label information on a quarter of the suite.

**Multiclass targets take frequency encoding** -- the level's relative count in the context.
It is weaker, but it is ordered, leak-free without folding, and it does not multiply the
feature count by the number of classes, which would collide with the architecture's
``max_features`` cap. The alternative that suggests itself -- encode against the integer class
code -- is the §100 mistake in a new place: class 3 is not "more" than class 1, so a mean
taken against those codes is a number with no meaning. Extending proper target statistics to
multiclass is deliberately left until a measurement asks for it.
"""

from __future__ import annotations

import numpy as np

#: Default smoothing weight, in units of "prior observations". A level must be seen about this
#: many times before its own rate outweighs the global base rate. 10 is the conventional
#: starting point and is **not** tuned here -- tuning it against the benchmark this module
#: exists to improve would be fitting the encoder to the test set.
DEFAULT_SMOOTHING = 10.0

#: Default number of out-of-fold partitions. Five keeps 80% of the context in each statistic
#: while ensuring no row informs its own encoding.
DEFAULT_N_FOLDS = 5


def _supports_target_statistics(y: np.ndarray) -> bool:
    """Whether target statistics are meaningful for this target, or frequency is the fallback.

    A mean is meaningful when the target's *values* carry magnitude: a 0/1 indicator (whose
    mean is a rate) or a continuous quantity (whose mean is a mean). It is meaningless for an
    integer class code, where "class 3" is a name rather than a quantity -- averaging those is
    the same error as label-encoding a categorical feature, moved to the other side of the
    problem.

    Args:
        y: ``(n,)`` context targets.

    Returns:
        True for binary ``{0, 1}`` and for continuous targets; False otherwise.
    """
    y = np.asarray(y)
    if not np.issubdtype(y.dtype, np.number):
        return False
    uniq = np.unique(y[np.isfinite(y)] if np.issubdtype(y.dtype, np.floating) else y)
    if len(uniq) == 2 and set(np.asarray(uniq, dtype=np.float64).tolist()) <= {0.0, 1.0}:
        return True
    # Continuous: a float target that is not a disguised integer code. Ten distinct values is
    # a deliberately loose floor -- a genuinely continuous column clears it trivially, and a
    # class code that does not is safer in the frequency branch either way.
    if np.issubdtype(y.dtype, np.floating) and not np.all(uniq == np.round(uniq)):
        return True
    return np.issubdtype(y.dtype, np.floating) and len(uniq) > 10


class CategoricalTargetEncoder:
    """Encode categorical columns as out-of-fold smoothed target statistics.

    Fitted on context rows only. :meth:`fit_transform` returns the **out-of-fold** encoding
    that the context rows must carry; :meth:`transform` returns the **full-context** encoding
    that query rows must carry. Those are deliberately different functions, and using
    :meth:`transform` on the context rows would reintroduce exactly the leak this class
    exists to prevent.

    Args:
        categorical_features: Column indices to encode. Columns not listed pass through
            untouched.
        smoothing: Weight of the global prior, in pseudo-observations.
        n_folds: Out-of-fold partitions used for the context rows' own encoding.
        random_state: Seed for the fold assignment.

    Attributes:
        prior_: Global target mean over the context rows.
        maps_: ``{column index: {level: encoded value}}`` from the full context.
    """

    def __init__(
        self,
        categorical_features: list[int] | tuple[int, ...] = (),
        smoothing: float = DEFAULT_SMOOTHING,
        n_folds: int = DEFAULT_N_FOLDS,
        random_state: int = 0,
    ) -> None:
        """Configure the encoder. See the class docstring for what each argument means."""
        if smoothing < 0:
            raise ValueError(f"smoothing must be >= 0, got {smoothing}")
        if n_folds < 2:
            raise ValueError(f"n_folds must be >= 2, got {n_folds}")
        self.categorical_features = tuple(categorical_features)
        self.smoothing = float(smoothing)
        self.n_folds = int(n_folds)
        self.random_state = int(random_state)
        self.prior_: float = 0.0
        self.maps_: dict[int, dict[float, float]] = {}
        self._frequency: bool = False

    def _stats(self, levels: np.ndarray, target: np.ndarray) -> dict[float, float]:
        """Smoothed per-level statistic over the rows given."""
        out: dict[float, float] = {}
        for level in np.unique(levels):
            sel = levels == level
            count = float(sel.sum())
            if self._frequency:
                out[float(level)] = count / len(levels)
            else:
                total = float(target[sel].sum())
                out[float(level)] = (total + self.smoothing * self.prior_) / (
                    count + self.smoothing
                )
        return out

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit on the context and return the context's **out-of-fold** encoding.

        Args:
            X: ``(n, d)`` context features. Categorical columns hold integer-coded levels.
            y: ``(n,)`` context targets.

        Returns:
            ``(n, d)`` with the listed columns replaced. Non-categorical columns are copied
            through unchanged.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        # Binary and continuous targets get target statistics; integer-coded multiclass falls
        # back to frequency, which is ordered and leak-free but carries no label information.
        self._frequency = not _supports_target_statistics(y)
        target = y.astype(np.float64) if not self._frequency else np.zeros(len(y))
        self.prior_ = float(target.mean()) if len(target) else 0.0
        self.maps_ = {
            col: self._stats(X[:, col], target) for col in self.categorical_features
        }

        out = X.copy()
        if not self.categorical_features or len(X) == 0:
            return out.astype(np.float32)

        rng = np.random.default_rng(self.random_state)
        folds = rng.permutation(len(X)) % min(self.n_folds, max(len(X), 1))
        for col in self.categorical_features:
            column = X[:, col]
            encoded = np.full(len(X), self.prior_ if not self._frequency else 0.0)
            for fold in np.unique(folds):
                held = folds == fold
                rest = ~held
                if not rest.any():
                    continue
                # The statistic for these rows is computed without them. A level appearing
                # only inside the held-out fold has no out-of-fold evidence and correctly
                # falls back to the prior rather than to its own label.
                stats = self._stats(column[rest], target[rest])
                fallback = self.prior_ if not self._frequency else 0.0
                encoded[held] = [stats.get(float(v), fallback) for v in column[held]]
            out[:, col] = encoded
        return out.astype(np.float32)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Encode query rows from the **full** context statistics.

        Query labels are not available, so there is nothing to leak and the full context is
        the best estimate. A level absent from the context falls back to the prior.

        Args:
            X: ``(n, d)`` query features.

        Returns:
            ``(n, d)`` with the listed columns replaced.

        Raises:
            RuntimeError: If called before :meth:`fit_transform`.
        """
        if not self.maps_ and self.categorical_features:
            raise RuntimeError("fit_transform must be called before transform")
        X = np.asarray(X, dtype=np.float64)
        out = X.copy()
        fallback = self.prior_ if not self._frequency else 0.0
        for col in self.categorical_features:
            mapping = self.maps_[col]
            out[:, col] = [mapping.get(float(v), fallback) for v in X[:, col]]
        return out.astype(np.float32)


def infer_categorical_features(
    X: np.ndarray, max_levels: int = 2, dtypes: list[str] | None = None
) -> list[int]:
    """Columns that should be treated as categorical.

    Deliberately conservative: a column is categorical only if its declared dtype says so.
    Guessing from level counts would silently reclassify a genuinely ordinal integer column
    -- a credit grade, a year -- and destroy the order that makes it useful.

    Args:
        X: ``(n, d)`` feature matrix, used only for its width.
        max_levels: Unused placeholder kept so callers can pass a cap once a measurement
            justifies one.
        dtypes: Per-column dtype names. ``None`` means nothing is categorical.

    Returns:
        Sorted column indices.
    """
    if dtypes is None:
        return []
    if len(dtypes) != X.shape[1]:
        raise ValueError(f"dtypes has {len(dtypes)} entries for {X.shape[1]} columns")
    return [i for i, d in enumerate(dtypes) if d in ("category", "object", "bool", "string")]
