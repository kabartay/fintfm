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

**Width comes from a derived ratio family, which is how real panels are wide.**
Measured 2026-09-08 (``docs/FINDINGS.md`` §18): the earlier version was capped
near 24 columns by a fixed dictionary of named quantities, while the real panels
it must transfer to carry 64-95 features. Those real features are overwhelmingly
*ratios over one balance sheet and P&L* — the UCI Polish panel's 64 columns are
of exactly that kind. So :func:`_ratio_family` samples numerator/denominator
pairs from a set of accounts that obey accounting identities, which fixes the
width mismatch and makes the prior more faithful rather than merely padded.

Difficulty was already well matched before this change (logistic-regression AUC
0.761 synthetic against 0.769 real) and **that match is a property to preserve**:
re-measure it after any change here, and treat a loss as a regression.
"""

from __future__ import annotations

import numpy as np

from fintfm.prior.base import Task

_N_SECTORS = 12

#: Expected defaults per synthetic task. This is the knob that decides how imbalanced a task
#: of a given size can be, since the base-rate floor is ``MIN_EXPECTED_POSITIVES / n_rows``.
#:
#: Set to 2.0, not 3.0, for a measured reason. Step cost is **worse than quadratic** in task
#: size on Metal — 0.62 s at 256 rows, 1.61 s at 1,024, 19.76 s at 2,048 and 310.71 s at
#: 4,096 (``docs/COMPUTE.md``) — so reaching a low base rate by growing the task is
#: prohibitively expensive past about 1,024 rows. Lowering the expected count instead puts a
#: 0.195% floor within reach of a 1,024-row task, which covers V4FinBench's lowest
#: cumulative rate of 0.19% (``docs/FINDINGS.md`` §26) at roughly a twelfth of the cost.
#:
#: Two defaults per task is thin, and it is also what a real low-default portfolio looks
#: like: a Basel LDP may carry two defaults in a thousand obligors. Below two the
#: both-classes guard starts fabricating positives and the task teaches nothing.
MIN_EXPECTED_POSITIVES = 2.0

#: Hard floor on the sampled base rate, reached only when the task is large enough to carry
#: it. Chosen to cover Basel low-default portfolios, whose rates run well below 1%.
_ABSOLUTE_RATE_FLOOR = 0.001

#: Upper end of the base-rate range, covering consumer-credit-like books.
_RATE_CEILING = 0.30


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


def _accounts(rng: np.random.Generator, n: int, macro: dict, sector_of: np.ndarray,
              sector_par: dict) -> dict[str, np.ndarray]:
    """Generate a balance sheet and P&L per firm, respecting accounting identities.

    Real credit datasets are wide because many ratios are computed over a few underlying
    accounts, so the accounts are the right thing to generate and the ratios are derived.
    Identities enforced here, because a ratio family over inconsistent accounts would teach
    relationships that cannot occur in real filings:

    ``total_assets = total_liabilities + equity``,
    ``current_assets = cash + receivables + inventory``,
    ``total_liabilities = short_term_liabilities + long_term_debt``,
    ``working_capital = current_assets - short_term_liabilities``,
    ``ebit = ebitda - depreciation``, ``net_profit = ebit - interest - tax``.

    Args:
        rng: Random generator.
        n: Number of firms.
        macro: Sampled macro regime (``rate``, ``cycle``).
        sector_of: Sector index per firm.
        sector_par: Sampled per-sector parameters.

    Returns:
        Mapping of account name to a length-``n`` array. All strictly level quantities;
        ratios are derived from these by :func:`_ratio_family`.
    """
    log_assets = rng.normal(rng.uniform(13, 18), rng.uniform(0.8, 2.0), size=n)
    total_assets = np.exp(log_assets)
    age = np.exp(rng.normal(2.3, 0.8, size=n))

    leverage = np.clip(
        rng.beta(2, 3, size=n) * 0.6 + sector_par["leverage"][sector_of] * 0.6 - 0.15, 0.0, 0.98
    )
    total_liabilities = leverage * total_assets
    equity = total_assets - total_liabilities
    short_frac = np.clip(rng.beta(2, 2, size=n), 0.05, 0.95)
    short_term_liabilities = short_frac * total_liabilities
    long_term_debt = total_liabilities - short_term_liabilities

    cash = np.clip(rng.beta(1.5, 8, size=n), 0.0, 0.8) * total_assets
    receivables = np.clip(rng.beta(2, 6, size=n), 0.0, 0.6) * total_assets
    inventory = np.clip(rng.beta(2, 7, size=n), 0.0, 0.6) * total_assets
    current_assets = cash + receivables + inventory
    working_capital = current_assets - short_term_liabilities
    retained_earnings = equity * rng.uniform(-0.5, 0.9, size=n)

    revenue = np.exp(rng.normal(0.0, 0.6, size=n)) * total_assets
    margin = (
        sector_par["margin"][sector_of]
        + rng.normal(0, 0.08, size=n)
        + 0.03 * macro["cycle"] * sector_par["cyclicality"][sector_of]
        - 0.04 * np.log1p(1.0 / np.maximum(age, 0.5))
    )
    ebitda = margin * revenue
    depreciation = rng.uniform(0.01, 0.09, size=n) * total_assets
    ebit = ebitda - depreciation
    interest = total_liabilities * (
        macro["rate"] + rng.uniform(0.01, 0.08, size=n) * (0.5 + leverage)
    )
    tax = np.maximum(0.0, ebit - interest) * rng.uniform(0.0, 0.35)
    net_profit = ebit - interest - tax
    operating_expenses = revenue - ebitda
    employees = np.maximum(1.0, revenue / np.exp(rng.normal(11.5, 0.5, size=n)))

    return {
        "total_assets": total_assets,
        "current_assets": current_assets,
        "cash": cash,
        "receivables": receivables,
        "inventory": inventory,
        "working_capital": working_capital,
        "total_liabilities": total_liabilities,
        "short_term_liabilities": short_term_liabilities,
        "long_term_debt": long_term_debt,
        "equity": equity,
        "retained_earnings": retained_earnings,
        "revenue": revenue,
        "ebitda": ebitda,
        "ebit": ebit,
        "net_profit": net_profit,
        "operating_expenses": operating_expenses,
        "depreciation": depreciation,
        "interest": interest,
        "employees": employees,
        "age": age,
    }


#: Ratio classes a real credit panel computes. Numerator/denominator names index accounts.
_RATIO_CLASSES: tuple[tuple[str, str], ...] = (
    ("net_profit", "total_assets"), ("ebit", "total_assets"), ("ebitda", "total_assets"),
    ("net_profit", "revenue"), ("ebit", "revenue"), ("ebitda", "revenue"),
    ("net_profit", "equity"), ("retained_earnings", "total_assets"),
    ("total_liabilities", "total_assets"), ("equity", "total_assets"),
    ("equity", "total_liabilities"), ("long_term_debt", "equity"),
    ("total_liabilities", "ebitda"), ("short_term_liabilities", "total_assets"),
    ("current_assets", "short_term_liabilities"), ("cash", "short_term_liabilities"),
    ("working_capital", "total_assets"), ("cash", "total_assets"),
    ("receivables", "revenue"), ("inventory", "revenue"),
    ("revenue", "total_assets"), ("revenue", "receivables"), ("revenue", "inventory"),
    ("ebit", "interest"), ("ebitda", "interest"), ("operating_expenses", "revenue"),
    ("depreciation", "total_assets"), ("net_profit", "total_liabilities"),
    ("revenue", "employees"), ("total_assets", "employees"),
)


def _ratio_family(
    rng: np.random.Generator, acc: dict[str, np.ndarray], n_wanted: int
) -> tuple[list[np.ndarray], list[str]]:
    """Derive up to ``n_wanted`` ratio columns from the accounts.

    Canonical credit ratios come first, then arbitrary account pairs, which is a fair model
    of how a real panel accumulates features: a core set everyone computes plus a long tail
    of variations. Denominators are floored away from zero rather than dropped, because a
    real filing with near-zero equity produces an extreme ratio and the model should see that.

    Args:
        rng: Random generator.
        acc: Accounts from :func:`_accounts`.
        n_wanted: Number of ratio columns to produce.

    Returns:
        A tuple of (columns, names), of length at most ``n_wanted``.
    """
    names = list(acc)
    pairs = list(_RATIO_CLASSES)
    rng.shuffle(pairs)
    # a long tail of arbitrary pairs, as real panels have
    extra = [
        (names[i], names[j])
        for i, j in rng.integers(0, len(names), size=(max(0, n_wanted) * 2, 2))
        if i != j
    ]
    cols: list[np.ndarray] = []
    used: list[str] = []
    for num, den in pairs + extra:
        if len(cols) >= n_wanted:
            break
        key = f"{num}/{den}"
        if key in used:
            continue
        d = acc[den]
        floor = np.maximum(np.abs(d), 1e-6 * (np.abs(d).mean() + 1e-12))
        cols.append(acc[num] / np.where(d < 0, -floor, floor))
        used.append(key)
    return cols, used


def _sample_survival(
    rng: np.random.Generator, distress: np.ndarray, target_rate: float, n_horizons: int
) -> tuple[np.ndarray, np.ndarray]:
    """Turn a latent distress score into a default *period* on a horizon grid.

    A binary label discards the question IFRS 9 actually asks — *when* — so the prior emits
    the period and lets the binary label fall out of it. Per-period hazards are

        h_{i,k} = sigmoid(b_k + shape_k + scale * distress_i)

    with the shape sampled per task so the model sees rising, falling, flat and hump-shaped
    hazard profiles. Real credit hazards are not flat: seasoning, refinancing walls and
    cyclical exposure all bend the curve, and which way depends on the book.

    The intercept ``b`` is solved so that the **cumulative** default rate over the whole grid
    hits ``target_rate``, which keeps the sampled 1-30% base-rate range meaning what it did
    before this change.

    Args:
        rng: Random generator.
        distress: ``(n,)`` standardised latent distress, higher meaning riskier.
        target_rate: Desired cumulative default rate over the full grid.
        n_horizons: Number of periods ``K``.

    Returns:
        ``(period, y)`` where ``period`` is the zero-based default period or ``-1`` for a
        firm surviving the grid, and ``y`` is the binary "defaulted within the grid" label.
    """
    shape_kind = rng.integers(0, 4)
    k = np.arange(n_horizons, dtype=np.float64)
    if shape_kind == 0:  # rising: leverage and refinancing pressure accumulate
        shape = rng.uniform(0.1, 0.6) * k
    elif shape_kind == 1:  # falling: early seasoning, survivors are sturdier
        shape = -rng.uniform(0.1, 0.5) * k
    elif shape_kind == 2:  # hump: a refinancing wall mid-grid
        shape = -rng.uniform(0.2, 0.8) * (k - rng.uniform(0.5, n_horizons - 0.5)) ** 2 / 2
    else:
        shape = np.zeros(n_horizons)
    shape = shape - shape.mean()

    scale = rng.uniform(0.4, 1.6)
    z = scale * distress

    def cumulative(b: float) -> np.ndarray:
        h = _sigmoid(z[:, None] + shape[None, :] + b).clip(1e-7, 1 - 1e-7)
        return 1.0 - np.cumprod(1.0 - h, axis=1)[:, -1]

    lo, hi = -30.0, 30.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if cumulative(mid).mean() < target_rate:
            lo = mid
        else:
            hi = mid
    b = 0.5 * (lo + hi)

    h = _sigmoid(z[:, None] + shape[None, :] + b).clip(1e-7, 1 - 1e-7)
    # walk the grid: default in the first period whose Bernoulli draw fires
    fired = rng.random(h.shape) < h
    any_default = fired.any(axis=1)
    period = np.where(any_default, fired.argmax(axis=1), -1).astype(np.int64)
    return period, any_default.astype(np.int64)


def sample_financial_task(
    rng: np.random.Generator,
    n_rows: int,
    max_features: int = 24,
    min_features: int = 4,
    n_horizons: int | None = None,
    min_expected_positives: float = MIN_EXPECTED_POSITIVES,
    absolute_rate_floor: float = _ABSOLUTE_RATE_FLOOR,
    rate_ceiling: float = _RATE_CEILING,
    n_sectors_max: int = _N_SECTORS,
) -> Task:
    """Sample one synthetic corporate-default classification task.

    Args:
        rng: NumPy random generator.
        n_rows: Number of companies (rows) to generate.
        max_features: Upper bound on exposed columns (after redundant/noise columns).
        min_features: Lower bound on exposed columns.
        n_horizons: When set, sample a **default period** on a grid of this many periods via
            :func:`_sample_survival`, so a hazard head can be trained (``docs/FINDINGS.md``
            §20). The binary label still falls out of it, so this is backwards compatible.
        min_expected_positives: Expected defaults per task; see
            :data:`MIN_EXPECTED_POSITIVES` for why this is 2.0 and what it costs.
        absolute_rate_floor: Hard floor on the sampled base rate.
        rate_ceiling: Upper end of the sampled base-rate range.
        n_sectors_max: Upper bound on the number of sectors drawn.

        The four rate/sector arguments default to this module's constants, which carry the
        measured reasoning for their values. They are arguments rather than constants so a
        run can vary the envelope from configuration (``prior`` in
        ``fintfm/configs/default.yaml``) without editing code, and so a test can drive an
        extreme envelope without monkey-patching a module global.

    Returns:
        A binary :class:`Task` whose columns are a random subset of financial quantities with
        random transforms and missingness, carrying ``period`` when ``n_horizons`` is set.
    """
    # --- macro regime --------------------------------------------------------
    macro = {"rate": rng.uniform(0.005, 0.12), "cycle": rng.normal(0.0, 1.0)}
    rate, cycle = macro["rate"], macro["cycle"]
    # --- sector structure ----------------------------------------------------
    n_sectors = int(rng.integers(2, n_sectors_max + 1))
    sector = rng.integers(0, n_sectors, size=n_rows)
    sector_par = {
        "margin": rng.normal(0.10, 0.06, size=n_sectors),
        "leverage": rng.uniform(0.2, 0.7, size=n_sectors),
        "cyclicality": rng.uniform(0.0, 1.5, size=n_sectors),
    }
    sector_hazard = rng.normal(0.0, 0.6, size=n_sectors)
    sector_cyclicality = sector_par["cyclicality"]
    # --- accounts, obeying accounting identities -----------------------------
    acc = _accounts(rng, n_rows, macro, sector, sector_par)
    assets = acc["total_assets"]
    log_assets = np.log(assets)
    age = acc["age"]
    debt = acc["total_liabilities"]
    equity = acc["equity"]
    cash = acc["cash"]
    revenue = acc["revenue"]
    ebitda = acc["ebitda"]
    interest = acc["interest"]
    employees = acc["employees"]
    leverage = debt / np.maximum(assets, 1e-12)
    cash_ratio = cash / np.maximum(assets, 1e-12)
    current_ratio = acc["current_assets"] / np.maximum(acc["short_term_liabilities"], 1e-12)
    turnover = revenue / np.maximum(assets, 1e-12)
    margin = ebitda / np.where(np.abs(revenue) < 1e-12, 1e-12, revenue)
    coverage = ebitda / np.maximum(np.abs(interest), 1e-6 * assets)
    growth = rng.normal(0.05 + 0.05 * cycle, 0.25, size=n_rows)
    payment_delay = np.maximum(
        0.0, rng.normal(15, 20, size=n_rows) + 30 * leverage - 20 * cash_ratio
    )
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
    # Sharpness = how deterministic default is given fundamentals. Lower means a larger
    # idiosyncratic component, which is economically right: management quality, fraud,
    # litigation and customer concentration drive real defaults and appear in no ratio.
    # Widening the ratio family (docs/FINDINGS.md §18) gave a linear model more views of the
    # same distress signal and made tasks too easy (AUC 0.815 against 0.769 real), so this
    # range was reduced from (0.8, 3.0) to restore the difficulty match. Published credit
    # scorecard performance sits around Gini 0.4-0.6, i.e. AUC 0.70-0.80, which is the
    # target this range is set against — see the provenance note in §19.
    distress = z(distress) * rng.uniform(0.7, 2.7)  # sharpness = label noise level
    # Base rate range, and why the floor is not a constant.
    #
    # This floor was 1% and V4FinBench's cumulative default rates are 0.36% down to 0.19%
    # (docs/FINDINGS.md §26), so the model had never seen a task as imbalanced as the
    # low-default portfolios docs/STRATEGY.md targets — a contradiction between the prior and
    # the strategy that stood until a real panel was scored.
    #
    # But a low rate is only *learnable* if the task actually contains defaults. At 0.2% with
    # 256 rows the expected count is 0.5, so most tasks would have none, and the
    # both-classes-present guard below would then flip rows up to ~2% and quietly undo the
    # change. The floor is therefore derived from the task size so that a task carries at
    # least `MIN_EXPECTED_POSITIVES` defaults in expectation.
    #
    # The consequence is a real cost, stated rather than hidden: **reaching a 0.2% base rate
    # requires ~1,500+ rows per task**, and attention is quadratic in that. Low-default
    # pretraining is expensive, and no choice of floor avoids it.
    rate_floor = max(absolute_rate_floor, min_expected_positives / max(n_rows, 1))
    rate_ceiling = max(rate_floor * 2.0, rate_ceiling)
    base_rate = float(np.exp(rng.uniform(np.log(rate_floor), np.log(rate_ceiling))))
    b = _solve_intercept(distress, base_rate)
    p_default = _sigmoid(distress + b)
    if n_horizons is not None:
        period, y = _sample_survival(rng, distress, base_rate, n_horizons)
    else:
        period = None
        y = (rng.random(n_rows) < p_default).astype(np.int64)
    if y.min() == y.max():  # guarantee both classes are present
        flip = rng.choice(n_rows, size=max(1, n_rows // 50), replace=False)
        y[flip] = 1 - y[flip]
        if period is not None:
            # keep period consistent with the flipped label, or the survival likelihood
            # would be trained against a contradiction
            period[flip] = np.where(
                y[flip] == 1, rng.integers(0, n_horizons, size=len(flip)), -1
            )
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
    # Width: real panels carry 64-95 features because they compute many ratios over one
    # balance sheet (docs/FINDINGS.md §18). Expose a sampled mix of named quantities and
    # derived ratios up to max_features, rather than capping at the named set.
    n_expose = int(rng.integers(min_features, max_features + 1))
    n_named = int(min(len(names), max(1, round(n_expose * rng.uniform(0.2, 0.6)))))
    chosen = rng.choice(len(names), size=n_named, replace=False)
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
    # derived ratio family — the bulk of the width, as in a real panel
    n_ratios = max(0, n_expose - len(cols))
    ratio_cols, _ratio_names = _ratio_family(rng, acc, n_ratios)
    for v in ratio_cols:
        v = v.astype(np.float64).copy()
        t = rng.random()
        if t < 0.20 and np.all(v > 0):
            v = np.log(v)
        elif t < 0.28:
            v = np.argsort(np.argsort(v)).astype(np.float64) / n_rows
        v = v + rng.normal(0, rng.uniform(0.0, 0.05) * (np.nanstd(v) + 1e-9), size=n_rows)
        cols.append(v)
        cats.append(False)
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
    # A ratio with a near-zero denominator is legitimate in a real filing but must not
    # arrive as inf: NaN is reserved to mean "missing", and an inf would silently poison
    # normalisation. Replace non-finite with a large finite sentinel of the right sign.
    X = np.nan_to_num(X, nan=np.nan, posinf=1e12, neginf=-1e12)
    X = np.clip(X, -1e12, 1e12)
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
        period=period,
        n_horizons=n_horizons,
    )
