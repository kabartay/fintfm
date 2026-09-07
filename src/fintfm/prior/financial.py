"""Financial prior: synthetic company tables with default labels.

The generative story is deliberately *structural* rather than a random MLP:
we sample a macro regime, a sector mix, company scale, then accounting
quantities that obey plausible identities (assets = debt + equity, interest
expense = debt * rate, coverage = EBITDA / interest ...). A latent distress
score combines leverage, profitability, liquidity, growth and macro pressure
with random weights and a random nonlinearity; the default label is a
Bernoulli draw whose base rate is itself sampled (1 %-30 %) so the model sees
heavy class imbalance during pretraining.

Observation noise is then applied: a random subset of the raw and ratio
features is exposed, columns are randomly log/rank-transformed, some values go
missing (MCAR and MNAR-on-distress), a few redundant/noisy columns are added
and the column order is shuffled. Nothing here is fitted to real data.
"""

from __future__ import annotations

import numpy as np

from fintfm.prior.base import Task

_N_SECTORS = 12


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _solve_intercept(z: np.ndarray, target_rate: float) -> float:
    """Bisect for the intercept ``b`` such that ``mean(sigmoid(z + b)) == target_rate``."""
    lo, hi = -30.0, 30.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if _sigmoid(z + mid).mean() < target_rate:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def sample_financial_task(
    rng: np.random.Generator,
    n_rows: int,
    max_features: int = 24,
    min_features: int = 4,
) -> Task:
    """Sample one synthetic corporate-default classification task.

    Args:
        rng: NumPy random generator.
        n_rows: Number of companies (rows) to generate.
        max_features: Upper bound on exposed columns (after redundant/noise columns).
        min_features: Lower bound on exposed columns.

    Returns:
        A binary :class:`Task` whose columns are a random subset of financial
        quantities with random transforms and missingness.
    """
    # --- macro regime --------------------------------------------------------
    rate = rng.uniform(0.005, 0.12)  # policy/base interest rate
    cycle = rng.normal(0.0, 1.0)  # >0 boom, <0 recession
    # --- sector structure ----------------------------------------------------
    n_sectors = int(rng.integers(2, _N_SECTORS + 1))
    sector = rng.integers(0, n_sectors, size=n_rows)
    sector_margin = rng.normal(0.10, 0.06, size=n_sectors)
    sector_leverage = rng.uniform(0.2, 0.7, size=n_sectors)
    sector_hazard = rng.normal(0.0, 0.6, size=n_sectors)
    sector_cyclicality = rng.uniform(0.0, 1.5, size=n_sectors)
    # --- company scale & balance sheet --------------------------------------
    log_assets = rng.normal(rng.uniform(13, 18), rng.uniform(0.8, 2.0), size=n_rows)
    assets = np.exp(log_assets)
    age = np.exp(rng.normal(2.3, 0.8, size=n_rows))  # years
    leverage = np.clip(
        rng.beta(2, 3, size=n_rows) * 0.6 + sector_leverage[sector] * 0.6 - 0.15, 0.0, 0.98
    )
    debt = leverage * assets
    equity = assets - debt
    cash_ratio = np.clip(rng.beta(1.5, 8, size=n_rows) + rng.normal(0, 0.02, n_rows), 0.0, 0.8)
    cash = cash_ratio * assets
    current_ratio = np.exp(rng.normal(0.3, 0.5, size=n_rows)) + cash_ratio
    # --- P&L -----------------------------------------------------------------
    turnover = np.exp(rng.normal(0.0, 0.6, size=n_rows))  # revenue / assets
    revenue = turnover * assets
    margin = (
        sector_margin[sector]
        + rng.normal(0, 0.08, size=n_rows)
        + 0.03 * cycle * sector_cyclicality[sector]
        - 0.04 * np.log1p(1.0 / np.maximum(age, 0.5))  # young firms less profitable
    )
    ebitda = margin * revenue
    interest = debt * (rate + rng.uniform(0.01, 0.08, size=n_rows) * (0.5 + leverage))
    coverage = ebitda / np.maximum(interest, 1e-6 * assets)
    growth = rng.normal(0.05 + 0.05 * cycle, 0.25, size=n_rows)
    payment_delay = np.maximum(
        0.0, rng.normal(15, 20, size=n_rows) + 30 * leverage - 20 * cash_ratio
    )
    employees = np.maximum(1.0, revenue / np.exp(rng.normal(11.5, 0.5, size=n_rows)))
    # --- latent distress -----------------------------------------------------
    def z(v: np.ndarray) -> np.ndarray:
        s = v.std()
        return (v - v.mean()) / (s if s > 1e-9 else 1.0)

    drivers = np.stack(
        [
            z(leverage),
            -z(np.log1p(np.maximum(coverage, -0.99))),
            -z(margin),
            -z(cash_ratio),
            -z(np.log(current_ratio)),
            -z(growth),
            z(payment_delay),
            -z(np.log(age)),
            -z(log_assets),
        ],
        axis=1,
    )
    w = np.abs(rng.normal(1.0, 0.5, size=drivers.shape[1])) * rng.uniform(0.3, 1.0, drivers.shape[1])
    distress = drivers @ w
    distress += sector_hazard[sector] - cycle * sector_cyclicality[sector] * rng.uniform(0.2, 1.0)
    distress += rate * rng.uniform(5, 25) * leverage
    # random nonlinearity so the boundary is not purely additive
    if rng.random() < 0.7:
        k = int(rng.integers(2, 5))
        idx = rng.choice(drivers.shape[1], size=k, replace=False)
        hidden = np.tanh(drivers[:, idx] @ rng.normal(0, 1, size=(k, 4)) + rng.normal(0, 0.5, 4))
        distress += hidden @ rng.normal(0, 1.0, size=4)
    distress = z(distress) * rng.uniform(0.8, 3.0)  # sharpness = label noise level
    base_rate = float(np.exp(rng.uniform(np.log(0.01), np.log(0.30))))
    b = _solve_intercept(distress, base_rate)
    p_default = _sigmoid(distress + b)
    y = (rng.random(n_rows) < p_default).astype(np.int64)
    if y.min() == y.max():  # guarantee both classes are present
        flip = rng.choice(n_rows, size=max(1, n_rows // 50), replace=False)
        y[flip] = 1 - y[flip]
    # --- observation model ---------------------------------------------------
    candidates: dict[str, tuple[np.ndarray, bool]] = {
        "sector": (sector.astype(np.float32), True),
        "log_assets": (log_assets, False),
        "assets": (assets, False),
        "revenue": (revenue, False),
        "ebitda": (ebitda, False),
        "margin": (margin, False),
        "debt": (debt, False),
        "equity": (equity, False),
        "leverage": (leverage, False),
        "cash": (cash, False),
        "cash_ratio": (cash_ratio, False),
        "current_ratio": (current_ratio, False),
        "interest": (interest, False),
        "coverage": (coverage, False),
        "growth": (growth, False),
        "age": (age, False),
        "employees": (employees, False),
        "payment_delay": (payment_delay, False),
        "turnover": (turnover, False),
        "debt_to_ebitda": (debt / np.where(np.abs(ebitda) < 1e-6, 1e-6, ebitda), False),
    }
    names = list(candidates)
    n_expose = int(rng.integers(min_features, min(max_features, len(names)) + 1))
    chosen = rng.choice(len(names), size=n_expose, replace=False)
    cols: list[np.ndarray] = []
    cats: list[bool] = []
    for j in chosen:
        v, is_cat = candidates[names[j]]
        v = v.astype(np.float64).copy()
        if not is_cat:
            t = rng.random()
            if t < 0.25 and v.min() > 0:
                v = np.log(v)
            elif t < 0.35:
                v = np.argsort(np.argsort(v)).astype(np.float64) / n_rows  # rank transform
            v = v + rng.normal(0, rng.uniform(0.0, 0.1) * (v.std() + 1e-9), size=n_rows)
        cols.append(v)
        cats.append(is_cat)
    # redundant / pure-noise columns
    room = max_features - len(cols)
    n_extra = int(rng.integers(0, min(room, 4) + 1)) if room > 0 else 0
    for _ in range(n_extra):
        if rng.random() < 0.5 and cols:
            src = cols[int(rng.integers(len(cols)))]
            cols.append(src * rng.normal(1, 0.3) + rng.normal(0, 0.5 * (src.std() + 1e-9), n_rows))
        else:
            cols.append(rng.normal(0, 1, size=n_rows))
        cats.append(False)
    X = np.stack(cols, axis=1)
    # missingness: MCAR everywhere plus MNAR concentrated on distressed firms
    mcar = rng.uniform(0.0, 0.15)
    mask = rng.random(X.shape) < mcar
    if rng.random() < 0.5:
        mnar_cols = rng.random(X.shape[1]) < 0.3
        mask |= (rng.random(X.shape) < 0.3 * p_default[:, None]) & mnar_cols[None, :]
    X[mask] = np.nan
    perm = rng.permutation(X.shape[1])
    return Task(
        X=X[:, perm].astype(np.float32),
        y=y,
        n_classes=2,
        is_categorical=np.array(cats)[perm],
        source="financial",
    )
