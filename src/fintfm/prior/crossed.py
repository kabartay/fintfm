"""The crossed-design prior: swap each generator's label function onto the other's features.

``docs/FINDINGS.md`` §66 closed seven candidate explanations for why the financial prior
fails to teach column-specific in-context inference while the generic SCM prior succeeds
(0.535 against 0.993 on the antisymmetric probe) -- width, dilution, base rate, identity
demand, learnability, label-dependence concentration, in-context value. All seven are
continuous task statistics, and none crosses the gap.

What remains is the generator's **functional form**: the financial generator produces every
task from one family (a monotone function of a signed linear combination of accounting
drivers), while the generic SCM generator samples a fresh nonlinear computational graph per
task. This module isolates that variable with a 2x2 crossed design:

    +----------------+------------------+------------------+
    |                | financial label  | SCM label        |
    +----------------+------------------+------------------+
    | financial feat | (existing prior) | financial_features_scm_label   |
    | SCM features    | scm_features_financial_label | (existing prior) |
    +----------------+------------------+------------------+

If the off-diagonal cells track the **label function**, functional diversity is the fix and
belongs on the financial generator directly. If they track the **features**, the accounting
identities themselves are implicated -- a much sharper trade, since removing them stops the
financial generator being a financial generator at all.

Diagnostic module, like ``prior/trivial.py``: not part of the default mixture, wired in only
through ``PriorConfig.p_crossed`` for this experiment.
"""

from __future__ import annotations

import numpy as np

from fintfm.prior.base import Task
from fintfm.prior.financial import (
    _ABSOLUTE_RATE_FLOOR,
    _RATE_CEILING,
    _SHARPNESS_MAX,
    _SHARPNESS_MIN,
    MIN_EXPECTED_POSITIVES,
    _sigmoid,
    _solve_intercept,
    sample_financial_task,
)
from fintfm.prior.scm import _ACTS, sample_scm_task


def _standardize(X: np.ndarray) -> np.ndarray:
    """Z-score each column, NaN treated as 0 after scoring (mean-imputed by construction)."""
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd = np.where(sd > 1e-9, sd, 1.0)
    Z = (X - mu) / sd
    return np.nan_to_num(Z, nan=0.0)


def _label_from_features_scm_style(
    rng: np.random.Generator, X: np.ndarray
) -> np.ndarray:
    """An SCM-style computational graph applied to *existing* columns, not fresh noise.

    Mirrors ``scm.sample_scm_task``'s node-building loop -- a random sparse linear map,
    a randomly chosen nonlinearity, additive noise, repeated for 1-4 layers -- except the
    input is the financial task's own standardized feature matrix rather than an
    independently sampled noise seed. This is the label function that would apply if the
    financial prior instead used the SCM generator's rule family on the same table.
    """
    Z = _standardize(X)
    h = Z
    n_layers = int(rng.integers(1, 5))
    for _ in range(n_layers):
        width = int(rng.integers(max(4, h.shape[1] + 1), 3 * h.shape[1] + 8))
        w = rng.normal(0, 1, size=(h.shape[1], width)) / np.sqrt(h.shape[1])
        w *= rng.random((h.shape[1], width)) < rng.uniform(0.3, 1.0)
        act = _ACTS[int(rng.integers(len(_ACTS)))]
        h = act(h @ w + rng.normal(0, 0.3, size=width))
        h = h + rng.normal(0, rng.uniform(0, 0.2), size=h.shape)
    idx = int(rng.integers(h.shape[1]))
    return h[:, idx]


def _label_from_features_financial_style(
    rng: np.random.Generator,
    X: np.ndarray,
    sharpness_min: float,
    sharpness_max: float,
) -> np.ndarray:
    """The financial generator's label mechanism applied to *arbitrary* columns.

    Mirrors ``sample_financial_task``'s driver-to-distress step exactly: pick a subset of the
    standardized columns as "drivers", give each a random magnitude and a random sign per
    task (the §47 fix -- a fixed sign lets one global rule solve every task), optionally mix
    in a two-way nonlinear interaction, then scale by a log-uniform sharpness. The input here
    is the SCM generator's own feature pool rather than accounting ratios.
    """
    Z = _standardize(X)
    n_drivers = min(Z.shape[1], int(rng.integers(3, 10)))
    idx = rng.choice(Z.shape[1], size=n_drivers, replace=False)
    drivers = Z[:, idx]
    w = np.abs(rng.normal(1.0, 0.5, size=n_drivers)) * rng.uniform(0.3, 1.0, n_drivers)
    w = w * rng.choice([-1.0, 1.0], size=n_drivers)
    distress = drivers @ w
    if rng.random() < 0.7:
        k = int(rng.integers(2, min(5, n_drivers + 1)))
        sub = rng.choice(n_drivers, size=k, replace=False)
        hidden = np.tanh(drivers[:, sub] @ rng.normal(0, 1, size=(k, 4)) + rng.normal(0, 0.5, 4))
        distress += hidden @ rng.normal(0, 1.0, size=4)
    sharpness = float(np.exp(rng.uniform(np.log(sharpness_min), np.log(sharpness_max))))
    s = distress.std()
    distress = (distress - distress.mean()) / (s if s > 1e-9 else 1.0) * sharpness
    return distress


def _finish_binary(
    rng: np.random.Generator,
    distress: np.ndarray,
    n_rows: int,
    min_expected_positives: float,
    absolute_rate_floor: float,
    rate_ceiling: float,
) -> np.ndarray:
    """Calibrate an intercept to a sampled base rate and draw Bernoulli labels.

    Identical policy to ``sample_financial_task``'s base-rate step, so base rate is not a
    confound between the on-diagonal and off-diagonal cells (``docs/FINDINGS.md`` §62 already
    showed base rate does not drive the teaching gap; holding it fixed here removes any doubt).
    """
    rate_floor = max(absolute_rate_floor, min_expected_positives / max(n_rows, 1))
    ceiling = max(rate_floor * 2.0, rate_ceiling)
    base_rate = float(np.exp(rng.uniform(np.log(rate_floor), np.log(ceiling))))
    b = _solve_intercept(distress, base_rate)
    y = (rng.random(n_rows) < _sigmoid(distress + b)).astype(np.int64)
    if y.min() == y.max():
        flip = rng.choice(n_rows, size=max(1, n_rows // 50), replace=False)
        y[flip] = 1 - y[flip]
    return y


def sample_financial_features_scm_label(
    rng: np.random.Generator,
    n_rows: int,
    max_features: int = 136,
    min_features: int = 4,
    min_expected_positives: float = MIN_EXPECTED_POSITIVES,
    absolute_rate_floor: float = _ABSOLUTE_RATE_FLOOR,
    rate_ceiling: float = _RATE_CEILING,
) -> Task:
    """Financial features, SCM-style label. One cell of the §66 crossed design."""
    t = sample_financial_task(
        rng, n_rows, max_features=max_features, min_features=min_features,
        min_expected_positives=min_expected_positives,
        absolute_rate_floor=absolute_rate_floor, rate_ceiling=rate_ceiling,
    )
    distress = _label_from_features_scm_style(rng, t.X.astype(np.float64))
    y = _finish_binary(
        rng, distress, n_rows, min_expected_positives, absolute_rate_floor, rate_ceiling
    )
    return Task(X=t.X, y=y, n_classes=2, is_categorical=t.is_categorical, source="fin_x_scm_y")


def sample_scm_features_financial_label(
    rng: np.random.Generator,
    n_rows: int,
    max_features: int = 136,
    min_features: int = 2,
    min_expected_positives: float = MIN_EXPECTED_POSITIVES,
    absolute_rate_floor: float = _ABSOLUTE_RATE_FLOOR,
    rate_ceiling: float = _RATE_CEILING,
    sharpness_min: float = _SHARPNESS_MIN,
    sharpness_max: float = _SHARPNESS_MAX,
) -> Task:
    """SCM features, financial-style label. The other cell of the §66 crossed design."""
    t = sample_scm_task(rng, n_rows, max_features=max_features, min_features=min_features, max_classes=2)
    distress = _label_from_features_financial_style(
        rng, t.X.astype(np.float64), sharpness_min, sharpness_max
    )
    y = _finish_binary(
        rng, distress, n_rows, min_expected_positives, absolute_rate_floor, rate_ceiling
    )
    return Task(X=t.X, y=y, n_classes=2, is_categorical=t.is_categorical, source="scm_x_fin_y")
