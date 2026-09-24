"""Continuous targets as a distribution over quantile bins (tasks 46.3, 46.5, 46.6).

Why this is an inference-boundary concern and not an architecture change
-----------------------------------------------------------------------

`openspec/changes/regression-and-multiclass` task 46.3 proposes "the binned regression head",
which reads as model surgery. It is not, and saying so saves a retrain of the wrong thing.

`FinancialTFM.head` is already `nn.Linear(d_model, max_classes)` producing logits over
`max_classes` outcomes, trained by cross-entropy on an integer index. A continuous target
binned into `K` quantile bins **is** an integer index over `K` outcomes. So the existing head,
the existing loss and the existing per-cell label injection all work unchanged; what regression
needs is a transform at the boundary (here) and a prior that emits continuous targets.

That also means the output is natively **distributional** rather than a point estimate. A
Gaussian head would have to assume unimodality; this one does not, which is the property LGD
actually requires — see the bimodality note below.

Why quantile bins specifically
------------------------------

Bin edges come from the **context** targets' quantiles, per task, for three reasons:

- **Scale invariance.** The model sees bin indices, never the target's units, so a task
  denominated in euros and the same task in thousands of euros are the same task. Fixed-width
  bins would make the model relearn every scale.
- **Balanced occupancy.** Quantile bins are equally populated by construction, so no bin is
  starved of training signal and cross-entropy is not dominated by one bin.
- **No query leakage.** Edges are fitted on context rows only, exactly as
  `modeling/model.py`'s feature normalisation is.

The cost is that bin widths vary, so the point estimate is a weighted sum of *bin
representatives* rather than of evenly spaced centres. :meth:`QuantileBinner.expected_value`
uses within-bin means from the context, which is the estimator that stays correct when bins
are wide in the tails — the usual case for financial targets.

The bounded, bimodal case this exists to serve
----------------------------------------------

Loss given default lives in [0, 1] and piles up at both ends: most defaults recover almost
everything or almost nothing, and the mean is a value that rarely occurs. A point-estimate or
Gaussian head reports that rarely-occurring middle and is confidently wrong. A distribution
over bins can place mass at both ends simultaneously, which is why task 46.6 asks for the
bimodal test specifically rather than trusting RMSE.
"""

from __future__ import annotations

import numpy as np

#: Default bin count. Enough resolution to express a bimodal shape, small enough that each bin
#: keeps a usable share of a few-hundred-row context. **Not tuned** — tuning it against a
#: benchmark would fit the discretisation to the test set.
DEFAULT_N_BINS = 10


class QuantileBinner:
    """Map a continuous target to bin indices and a predicted distribution back to values.

    Args:
        n_bins: Number of bins. Collapses to the number of distinct edges when the target has
            fewer distinct values than bins, which is the normal case for a target that piles
            up at a few points.

    Attributes:
        edges_: ``(n_edges,)`` interior bin boundaries.
        representatives_: ``(n_bins,)`` within-bin means of the fitted targets.
        n_bins_: Number of bins actually used.
    """

    def __init__(self, n_bins: int = DEFAULT_N_BINS) -> None:
        """Configure the bin count. Edges are fitted later, from context targets only.

        Args:
            n_bins: Requested bins. The fitted count may be lower when the target has fewer
                distinct values, which is the normal case for a target piling up at a few
                points.

        Raises:
            ValueError: If ``n_bins`` is below two.
        """
        if n_bins < 2:
            raise ValueError(f"n_bins must be >= 2, got {n_bins}")
        self.n_bins = int(n_bins)
        self.edges_: np.ndarray = np.empty(0)
        self.representatives_: np.ndarray = np.empty(0)
        self.n_bins_: int = 0

    def fit(self, y: np.ndarray) -> QuantileBinner:
        """Fit bin edges from the targets' quantiles.

        Args:
            y: ``(n,)`` continuous context targets.

        Returns:
            self.

        Raises:
            ValueError: If ``y`` is empty or has no finite values.
        """
        y = np.asarray(y, dtype=np.float64).ravel()
        finite = y[np.isfinite(y)]
        if finite.size == 0:
            raise ValueError("cannot fit bins: no finite target values")
        qs = np.linspace(0, 1, self.n_bins + 1)[1:-1]
        # Ties collapse edges. A target with three distinct values gets three bins, not ten
        # empty ones -- and `unique` is what makes the degenerate case work rather than
        # producing zero-width bins that no row can ever fall into.
        edges = np.unique(np.quantile(finite, qs))
        # Drop any edge at or below the minimum. `digitize` sends a value equal to an edge to
        # the bin *above* it, so an edge at the minimum leaves bin 0 permanently unreachable
        # -- a bin no row can enter still consumes a logit and dilutes the softmax.
        self.edges_ = edges[edges > finite.min()]
        idx = np.digitize(finite, self.edges_)
        self.n_bins_ = len(self.edges_) + 1
        # Within-bin means, not midpoints: the outermost bins are unbounded, so a midpoint is
        # undefined there, and interior bins are wide exactly where the data is sparse.
        reps = np.empty(self.n_bins_)
        for b in range(self.n_bins_):
            sel = idx == b
            reps[b] = finite[sel].mean() if sel.any() else np.nan
        # An empty bin can only arise from ties at an edge; fall back to the global mean so
        # the representative is defined rather than propagating NaN into every prediction.
        reps[~np.isfinite(reps)] = finite.mean()
        self.representatives_ = reps
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        """Targets to bin indices.

        Args:
            y: ``(n,)`` continuous targets.

        Returns:
            ``(n,)`` integer bin indices in ``[0, n_bins_)``.

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        self._check_fitted()
        return np.digitize(np.asarray(y, dtype=np.float64).ravel(), self.edges_).astype(np.int64)

    def expected_value(self, proba: np.ndarray) -> np.ndarray:
        """Point estimate: the distribution's mean under the bin representatives.

        Args:
            proba: ``(n, n_bins_)`` predicted probabilities.

        Returns:
            ``(n,)`` point estimates.
        """
        self._check_fitted()
        proba = self._validate(proba)
        return proba @ self.representatives_

    def quantile(self, proba: np.ndarray, q: float) -> np.ndarray:
        """The predicted distribution's ``q``-quantile, by linear interpolation in bin space.

        Args:
            proba: ``(n, n_bins_)`` predicted probabilities.
            q: Quantile in ``(0, 1)``.

        Returns:
            ``(n,)`` quantile estimates.

        Raises:
            ValueError: If ``q`` is not in ``(0, 1)``.
        """
        self._check_fitted()
        if not 0.0 < q < 1.0:
            raise ValueError(f"q must be in (0, 1), got {q}")
        proba = self._validate(proba)
        cdf = np.cumsum(proba, axis=1)
        # The first bin whose cumulative mass reaches q. Interpolating *within* that bin would
        # need its width, which is undefined for the unbounded outer bins, so the
        # representative is returned instead -- coarser, but never outside the data's range.
        idx = np.argmax(cdf >= q, axis=1)
        return self.representatives_[idx]

    def interval(self, proba: np.ndarray, level: float = 0.8) -> tuple[np.ndarray, np.ndarray]:
        """Central prediction interval at ``level`` coverage.

        Args:
            proba: ``(n, n_bins_)`` predicted probabilities.
            level: Nominal coverage, e.g. 0.8 for an 80% interval.

        Returns:
            ``(lower, upper)``, each ``(n,)``.

        Raises:
            ValueError: If ``level`` is not in ``(0, 1)``.
        """
        if not 0.0 < level < 1.0:
            raise ValueError(f"level must be in (0, 1), got {level}")
        tail = (1.0 - level) / 2.0
        return self.quantile(proba, tail), self.quantile(proba, 1.0 - tail)

    def _check_fitted(self) -> None:
        """Raise if the binner has no edges yet.

        Raises:
            RuntimeError: If :meth:`fit` has not been called. Without this the binner would
                silently use an empty edge array, sending every value to bin 0.
        """
        if self.n_bins_ == 0:
            raise RuntimeError("fit must be called before this method")

    def _validate(self, proba: np.ndarray) -> np.ndarray:
        """Coerce a predicted distribution to float64 and check its width.

        Args:
            proba: ``(n, n_bins_)`` probabilities.

        Returns:
            The same array as float64.

        Raises:
            ValueError: If the shape does not match the fitted bin count. A silent mismatch
                would pair probabilities with the wrong representatives and produce a
                plausible number from the wrong bins.
        """
        proba = np.asarray(proba, dtype=np.float64)
        if proba.ndim != 2 or proba.shape[1] != self.n_bins_:
            raise ValueError(
                f"proba must be (n, {self.n_bins_}), got {proba.shape}"
            )
        return proba


def interval_coverage(
    y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> float:
    """Fraction of targets inside their predicted interval (task 46.5).

    An 80% interval should contain the truth 80% of the time. Reporting RMSE alone would pass
    a model that is accurate on average and useless for the risk quantities this exists to
    serve, which is why coverage is reported beside it rather than instead of it.

    Args:
        y_true: ``(n,)`` realised targets.
        lower: ``(n,)`` interval lower bounds.
        upper: ``(n,)`` interval upper bounds.

    Returns:
        Empirical coverage in ``[0, 1]``.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    return float(np.mean((y_true >= np.asarray(lower)) & (y_true <= np.asarray(upper))))
