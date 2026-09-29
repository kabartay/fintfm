"""Labelled task-family generators, each independently difficulty-controlled.

Phase C item 18 / openspec task 40.2 (`mechanism-diverse-prior`). §112 measured the production
prior mixture's distinctiveness as weak: the model does not obviously need to read its context
differently across the financial and SCM priors it already sees. This module is the first step
toward finding out whether a richer, explicitly-labelled set of causal structures teaches
something those two do not -- nine families, each tagged with its own family identifier
(`Task.source`) for `cell-attention-and-task-inference` task 39.7's planned DGP-classification
probe.

§74 measured an *exact* closed-form Bayes-AUC target for a 1-D Gaussian mean shift:
`Phi(mu / sqrt(2))`, verified there to 3 decimals against the true formula. Five families here
reduce to that same construction and reuse it directly:

- a linear projection is exactly Gaussian regardless of how many features carry its weight
  (rotation invariance of an isotropic Gaussian), which is what separates ``linear`` (a few
  active features), ``sparse`` (one), and ``dense`` (all) -- same formula, different weight
  support;
- AUC is invariant to any strictly monotone transform of the decision statistic, which is what
  lets ``threshold`` reuse the identical formula on a reshaped, non-Gaussian marginal;
- ``latent_factor`` admits its own closed form by averaging noisy proxies of a hidden Gaussian
  driver: the sufficient statistic is again a mean-shifted Gaussian, just with an inflated
  variance from proxy noise, so the same formula applies to an *effective* mean shift.

The other four families (``xor``, ``interaction``, ``max_min``, ``piecewise``) have no known
closed form, so their generative sharpness is calibrated by bisection against a large
Monte-Carlo sample scored with the family's own true decision statistic -- the same "cheap
bisection before committing GPU spend" discipline task 40.3 names for the interaction-order
curriculum.

Only binary tasks are generated (`n_classes=2`); the existing SCM and tree priors already cover
multiclass targets, and task 40.2 does not ask for a multiclass generalisation of this
construction.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.special import ndtri
from sklearn.metrics import roc_auc_score

from fintfm.prior.base import Task

#: The nine families task 40.2 names, in the order it names them.
FAMILIES: tuple[str, ...] = (
    "linear",
    "threshold",
    "xor",
    "interaction",
    "max_min",
    "piecewise",
    "sparse",
    "dense",
    "latent_factor",
)

_AUC_CLIP = 1e-6


def _mu_for_target_auc(target_auc: float) -> float:
    """Exact mean-shift magnitude for a 1-D equal-variance Gaussian split (§74)."""
    p = float(np.clip(target_auc, _AUC_CLIP, 1.0 - _AUC_CLIP))
    return float(np.sqrt(2.0) * ndtri(p))


def _gaussian_mean_shift(
    rng: np.random.Generator,
    n_rows: int,
    n_features: int,
    target_auc: float,
    k_active: int,
    monotone_reshape: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw ``(X, y)`` from the §74 construction, generalised to ``k_active`` weighted features.

    Class 0 is drawn `N(0, I)`; class 1 is shifted by `mu` along a random unit vector supported
    on `k_active` of the `n_features` columns, the rest pure noise. The Bayes-optimal statistic
    is the projection onto that vector, and its AUC is exactly `Phi(mu / sqrt(2))` regardless of
    `k_active` or which columns it touches -- an isotropic Gaussian's projection onto any unit
    vector is itself standard normal.

    Args:
        rng: NumPy random generator.
        n_rows: Rows to generate.
        n_features: Total feature columns; only `k_active` of them carry signal.
        target_auc: Requested Bayes-optimal AUC.
        k_active: Number of features the informative direction is supported on.
        monotone_reshape: Apply a strictly monotone, order-preserving reshape to the active
            columns after the mean shift. AUC is invariant to this (it depends only on rank),
            so it changes the feature's marginal shape without changing its difficulty --
            used by `threshold` to differentiate from `sparse` in shape, not in Bayes AUC.

    Returns:
        `(X, y)`: `X` is `(n_rows, n_features)` float64, `y` is `(n_rows,)` int64 in `{0, 1}`.
    """
    k_active = max(1, min(k_active, n_features))
    active = rng.choice(n_features, size=k_active, replace=False)
    w = rng.normal(size=k_active)
    w /= np.linalg.norm(w) + 1e-12
    mu = _mu_for_target_auc(target_auc)

    y = rng.integers(0, 2, size=n_rows)
    X = rng.normal(size=(n_rows, n_features))
    rows1 = np.flatnonzero(y == 1)
    X[np.ix_(rows1, active)] += mu * w

    if monotone_reshape:
        col = X[:, active]
        X[:, active] = np.sign(col) * np.abs(col) ** 1.5

    return X, y


def _calibrate_sharpness(
    raw_score_fn: Callable[[np.ndarray], np.ndarray],
    rng: np.random.Generator,
    n_features: int,
    target_auc: float,
    n_calib: int = 20_000,
    tol: float = 0.01,
    max_iter: int = 40,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Bisect a logistic sharpness so a family's true Bayes AUC hits `target_auc`.

    Draws one calibration sample and a fixed vector of uniforms, then searches `sharpness` in
    `p(y=1|x) = sigmoid(sharpness * raw_score_fn(x))`. Holding the calibration sample and the
    uniforms fixed across the search makes `AUC(sharpness)` monotone (raising `sharpness` only
    pushes each row's `p` further toward 0 or 1 in the direction its own score already points),
    so plain bisection converges instead of chasing resampling noise.

    Args:
        raw_score_fn: Maps `(n, n_features)` features to a `(n,)` continuous score; the
            family's true (by construction) Bayes-optimal decision statistic.
        rng: NumPy random generator.
        n_features: Feature width the calibration sample is drawn at.
        target_auc: Requested Bayes-optimal AUC.
        n_calib: Calibration sample size.
        tol: Stop once the achieved AUC is within this of `target_auc`.
        max_iter: Bisection step cap.

    Returns:
        `(X_calib, r_calib, sharpness)`: the calibration draw and standardised score (returned
        so callers needing a second calibration sample do not have to know the standardisation
        constants) and the calibrated sharpness.
    """
    X = rng.normal(size=(n_calib, n_features))
    r = raw_score_fn(X)
    r = (r - r.mean()) / (r.std() + 1e-12)
    u = rng.random(n_calib)

    lo, hi = 0.0, 50.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        p = 1.0 / (1.0 + np.exp(-mid * r))
        y = (u < p).astype(np.int64)
        auc = 0.5 if len(np.unique(y)) < 2 else roc_auc_score(y, r)
        if abs(auc - target_auc) < tol:
            lo = hi = mid
            break
        if auc < target_auc:
            lo = mid
        else:
            hi = mid
    return X, r, 0.5 * (lo + hi)


def _sample_by_sharpness(
    rng: np.random.Generator,
    n_rows: int,
    n_features: int,
    target_auc: float,
    raw_score_fn: Callable[[np.ndarray], np.ndarray],
    _score_out: list[np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Calibrate `raw_score_fn`'s sharpness, then draw a fresh `(X, y)` at that sharpness.

    Args:
        _score_out: Internal, mirrors `scm.sample_scm_task`'s `_latent_out` convention. When a
            list is passed, the standardised score used to generate `y` is appended to it --
            this is the family's true, by-construction Bayes-optimal statistic, and lets a test
            measure realised AUC directly against it rather than through a learner that may not
            recover the structure (the structure being hard to recover from raw features is the
            entire point of families like `xor`).
    """
    _, _, sharpness = _calibrate_sharpness(raw_score_fn, rng, n_features, target_auc)
    X = rng.normal(size=(n_rows, n_features))
    r = raw_score_fn(X)
    r = (r - r.mean()) / (r.std() + 1e-12)
    if _score_out is not None:
        _score_out.append(r.copy())
    p = 1.0 / (1.0 + np.exp(-sharpness * r))
    y = rng.binomial(1, p).astype(np.int64)
    return X, y


def _finish(X: np.ndarray, y: np.ndarray, family: str) -> Task:
    """Wrap raw `(X, y)` as a `Task`, tagged with its family identifier.

    Every family here generates purely continuous features -- `is_categorical` is all-False,
    unlike the SCM and financial priors, which is the point: this module isolates the causal
    structure axis from the categorical-coding axis the others already cover.
    """
    return Task(
        X=X.astype(np.float32),
        y=y.astype(np.int64),
        n_classes=2,
        is_categorical=np.zeros(X.shape[1], dtype=bool),
        source=f"family:{family}",
    )


def sample_linear_task(
    rng: np.random.Generator, n_rows: int, n_features: int = 12, target_auc: float = 0.85
) -> Task:
    """A linear decision boundary over a handful of active features (closed-form difficulty)."""
    k = int(rng.integers(2, min(5, n_features) + 1))
    X, y = _gaussian_mean_shift(rng, n_rows, n_features, target_auc, k_active=k)
    return _finish(X, y, "linear")


def sample_sparse_task(
    rng: np.random.Generator, n_rows: int, n_features: int = 12, target_auc: float = 0.85
) -> Task:
    """A linear decision boundary supported on exactly one feature (closed-form difficulty)."""
    X, y = _gaussian_mean_shift(rng, n_rows, n_features, target_auc, k_active=1)
    return _finish(X, y, "sparse")


def sample_dense_task(
    rng: np.random.Generator, n_rows: int, n_features: int = 12, target_auc: float = 0.85
) -> Task:
    """A linear decision boundary spread over every feature (closed-form difficulty)."""
    X, y = _gaussian_mean_shift(rng, n_rows, n_features, target_auc, k_active=n_features)
    return _finish(X, y, "dense")


def sample_threshold_task(
    rng: np.random.Generator, n_rows: int, n_features: int = 12, target_auc: float = 0.85
) -> Task:
    """A single-feature hard-cutoff rule on a reshaped, non-Gaussian marginal.

    Same closed-form Bayes AUC as `sparse` -- AUC depends only on rank, and the reshape is
    strictly monotone -- but the observed feature is heavy-tailed rather than Gaussian, so a
    model relying on a fixed linear scale (rather than rank) sees a differently-shaped problem.
    """
    X, y = _gaussian_mean_shift(
        rng, n_rows, n_features, target_auc, k_active=1, monotone_reshape=True
    )
    return _finish(X, y, "threshold")


def sample_latent_factor_task(
    rng: np.random.Generator, n_rows: int, n_features: int = 12, target_auc: float = 0.85
) -> Task:
    """A hidden Gaussian factor observed only through several independently-noisy proxies.

    `z` drives the label via the §74 mean shift; each of `k` proxy columns is `z` plus its own
    noise. The sufficient statistic from `k` iid-noise proxies of `z` is their mean, which is
    again Gaussian with the same class-conditional mean shift and an inflated variance
    `1 + tau^2/k` from averaging out the proxy noise -- so the same closed form applies to an
    *effective* mean shift solved backwards from the inflated variance, rather than to `mu`
    directly. The rest of the features are pure noise, so the model must find and combine the
    correlated proxies rather than read one clean column.
    """
    k = int(rng.integers(2, min(4, n_features) + 1))
    active = rng.choice(n_features, size=k, replace=False)
    tau = float(rng.uniform(0.5, 2.0))
    var_inflation = 1.0 + tau**2 / k
    mu = _mu_for_target_auc(target_auc) * np.sqrt(var_inflation)

    y = rng.integers(0, 2, size=n_rows)
    z = mu * y + rng.normal(size=n_rows)
    X = rng.normal(size=(n_rows, n_features))
    X[:, active] = z[:, None] + rng.normal(0.0, tau, size=(n_rows, k))
    return _finish(X, y, "latent_factor")


def _sample_interaction_order(
    rng: np.random.Generator,
    n_rows: int,
    n_features: int,
    k: int,
    target_auc: float,
    _score_out: list[np.ndarray] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Shared generator behind `xor` (`k=2`) and `sample_interaction_order_task` (`k` general).

    Raises:
        ValueError: If `k` exceeds `n_features` or is below 1.
    """
    if not 1 <= k <= n_features:
        raise ValueError(f"k must be in [1, n_features={n_features}], got {k}")

    def raw(X: np.ndarray) -> np.ndarray:
        signs = np.sign(X[:, :k])
        signs[signs == 0] = 1.0
        return np.prod(signs, axis=1)

    return _sample_by_sharpness(rng, n_rows, n_features, target_auc, raw, _score_out)


def sample_xor_task(
    rng: np.random.Generator,
    n_rows: int,
    n_features: int = 12,
    target_auc: float = 0.85,
    _score_out: list[np.ndarray] | None = None,
) -> Task:
    """Noisy parity of two features' signs -- unsolvable by any single linear feature."""
    X, y = _sample_interaction_order(rng, n_rows, n_features, 2, target_auc, _score_out)
    return _finish(X, y, "xor")


def sample_interaction_order_task(
    rng: np.random.Generator,
    n_rows: int,
    n_features: int = 12,
    k: int = 2,
    target_auc: float = 0.85,
    _score_out: list[np.ndarray] | None = None,
) -> Task:
    """Noisy parity of `k` features' signs -- `xor` generalised to an explicit interaction order.

    Task 40.3 (`docs/roadmap/ROADMAP.md` Phase C item 19): the achieved-AUC-vs-`k` curve is the
    instrument, not any single task. `k=1` is a plain sign threshold on one feature (recoverable
    by a linear probe, same as `sparse`/`threshold`); `k=2` is exactly `xor`; each additional `k`
    requires jointly reading one more feature before the label carries any information at all --
    no `k-1`-way marginal or lower-order combination of the active features is correlated with
    the label by construction, since the parity of `k` independent fair-coin-like signs is
    uniform unless all `k` are read together. Calibrated the same way as every other
    no-closed-form family: bisected sharpness against a fixed Monte-Carlo sample.

    Args:
        rng: NumPy random generator.
        n_rows: Rows to generate.
        n_features: Total feature columns; only `k` of them carry signal.
        k: Interaction order -- number of features whose joint sign-parity the label depends on.
        target_auc: Requested Bayes-optimal AUC.
        _score_out: See `_sample_by_sharpness`.

    Returns:
        A `Task` tagged `source="family:interaction_order_k{k}"`.
    """
    X, y = _sample_interaction_order(rng, n_rows, n_features, k, target_auc, _score_out)
    return _finish(X, y, f"interaction_order_k{k}")


def sample_interaction_task(
    rng: np.random.Generator,
    n_rows: int,
    n_features: int = 12,
    target_auc: float = 0.85,
    _score_out: list[np.ndarray] | None = None,
) -> Task:
    """A smooth multiplicative interaction between two features, no additive main effect."""

    def raw(X: np.ndarray) -> np.ndarray:
        return X[:, 0] * X[:, 1]

    X, y = _sample_by_sharpness(rng, n_rows, n_features, target_auc, raw, _score_out)
    return _finish(X, y, "interaction")


def sample_max_min_task(
    rng: np.random.Generator,
    n_rows: int,
    n_features: int = 12,
    target_auc: float = 0.85,
    _score_out: list[np.ndarray] | None = None,
) -> Task:
    """The label tracks an order statistic (the max) of several features, not their sum."""
    k = min(4, n_features)

    def raw(X: np.ndarray) -> np.ndarray:
        return X[:, :k].max(axis=1)

    X, y = _sample_by_sharpness(rng, n_rows, n_features, target_auc, raw, _score_out)
    return _finish(X, y, "max_min")


def sample_piecewise_task(
    rng: np.random.Generator,
    n_rows: int,
    n_features: int = 12,
    target_auc: float = 0.85,
    _score_out: list[np.ndarray] | None = None,
) -> Task:
    """A single feature drives the label through a non-monotonic piecewise-linear rule.

    Breakpoints and per-segment slopes (alternating sign) are fixed once per task, so a linear
    probe on this one feature cannot separate the classes even though the label depends on
    nothing else.
    """
    breakpoints = np.sort(rng.uniform(-1.5, 1.5, size=3))
    slopes = rng.choice([-1.0, 1.0], size=4) * rng.uniform(0.5, 2.0, size=4)
    edges = np.concatenate([[-np.inf], breakpoints, [np.inf]])

    def raw(X: np.ndarray) -> np.ndarray:
        x0 = X[:, 0]
        r = np.zeros_like(x0)
        for i in range(4):
            seg = (x0 >= edges[i]) & (x0 < edges[i + 1])
            r = np.where(seg, slopes[i] * x0, r)
        return r

    X, y = _sample_by_sharpness(rng, n_rows, n_features, target_auc, raw, _score_out)
    return _finish(X, y, "piecewise")


_SAMPLERS: dict[str, Callable[..., Task]] = {
    "linear": sample_linear_task,
    "threshold": sample_threshold_task,
    "xor": sample_xor_task,
    "interaction": sample_interaction_task,
    "max_min": sample_max_min_task,
    "piecewise": sample_piecewise_task,
    "sparse": sample_sparse_task,
    "dense": sample_dense_task,
    "latent_factor": sample_latent_factor_task,
}


def sample_family_task(
    rng: np.random.Generator,
    family: str,
    n_rows: int,
    n_features: int = 12,
    target_auc: float = 0.85,
) -> Task:
    """Dispatch to the named family's generator.

    Args:
        rng: NumPy random generator.
        family: One of `FAMILIES`.
        n_rows: Rows to generate.
        n_features: Feature columns (only some may carry signal; see each family).
        target_auc: Requested Bayes-optimal AUC.

    Raises:
        ValueError: If `family` is not one of `FAMILIES`.
    """
    if family not in _SAMPLERS:
        raise ValueError(f"unknown task family {family!r}; expected one of {FAMILIES}")
    return _SAMPLERS[family](rng, n_rows, n_features=n_features, target_auc=target_auc)


def sample_composition_task(
    rng: np.random.Generator,
    n_rows: int,
    n_features: int = 12,
    target_component_auc: float = 0.9,
    mode: str = "composed",
) -> Task:
    """Two independent single-feature rules, and their held-out AND-composition.

    Task 40.4 (`docs/roadmap/ROADMAP.md` Phase C item 20): tests compositional generalisation,
    not interaction order (task 40.3/§137's `sample_interaction_order_task`). Two disjoint
    active features each carry an independent §74-style mean-shift rule (`y_a`, `y_b`), sharing
    one `X` draw regardless of `mode` -- only which label is exposed differs, so `component_a`,
    `component_b` and `composed` are the same feature distribution under three different label
    functions, letting the same checkpoint be scored on each without any distribution shift
    confounding the comparison.

    `component_a` and `component_b` each admit §74's exact closed form (the other component's
    shift lands on a disjoint column and is independent of this one's label, so it does not
    change this component's own conditional distribution). `composed` -- `y_a AND y_b` -- has no
    simple closed form and is not calibrated to a target; it is scored only relative to the two
    components' achieved AUC, which is what task 40.4 asks for.

    Args:
        rng: NumPy random generator.
        n_rows: Rows to generate.
        n_features: Total feature columns; must be at least 2.
        target_component_auc: Bayes-optimal AUC each individual component is calibrated to.
        mode: `"component_a"`, `"component_b"`, or `"composed"`.

    Returns:
        A `Task` tagged `source="family:composition_{mode}"`.

    Raises:
        ValueError: If `n_features < 2` or `mode` is not recognised.
    """
    if n_features < 2:
        raise ValueError("n_features must be >= 2 for two disjoint components")
    if mode not in ("component_a", "component_b", "composed"):
        raise ValueError(f"unknown mode {mode!r}; expected component_a, component_b, or composed")

    active = rng.choice(n_features, size=2, replace=False)
    active_a, active_b = int(active[0]), int(active[1])
    mu = _mu_for_target_auc(target_component_auc)

    y_a = rng.integers(0, 2, size=n_rows)
    y_b = rng.integers(0, 2, size=n_rows)
    X = rng.normal(size=(n_rows, n_features))
    X[y_a == 1, active_a] += mu
    X[y_b == 1, active_b] += mu

    if mode == "component_a":
        y = y_a
    elif mode == "component_b":
        y = y_b
    else:
        y = (y_a & y_b).astype(np.int64)
    return _finish(X, y, f"composition_{mode}")


def sample_confound_collider_pair(
    rng: np.random.Generator,
    n_ctx: int,
    n_query: int,
    n_features: int = 12,
    target_auc: float = 0.9,
    cause_weight: float = 0.70710678,
    proxy_noise: float = 0.3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A confound and a true cause both point into the label; only the confound's proxy is cut.

    Task 40.5 (`docs/roadmap/ROADMAP.md` Phase C item 21). The label `y` is a **collider**: two
    independent upstream causes point into it. `Z` is a *hidden* confound, observed only through
    a noisy proxy column `P = Z + noise` -- exactly as legitimate a covariate as any other in the
    observational distribution this pair's context rows are drawn from, since `Z` really does
    drive `y`. `C` is a *directly observed* true cause, independent of `Z`, also driving `y`.

    Returns two parallel arrays sharing everything -- the same context rows, the same query
    `y`, the same query `C` -- except the query rows' proxy column: `X_obs`'s query proxy still
    reflects the row's own `Z` (the confound path intact, matching the context distribution);
    `X_intervened`'s query proxy is regenerated from an **independent** fresh `Z'` (`do(P)`,
    severing only the `Z -> P` edge). `y` is unchanged in both, because `y` was generated from
    the original `Z` and `C` before the intervention and an intervention on `P` cannot act
    backwards on its own cause. A model that predicts query rows mainly from `P` will lose
    accuracy from `X_obs` to `X_intervened`; a model that has learned to weight the directly
    observed `C` will not, because `C`'s relationship to `y` is untouched.

    `cause_weight` (`alpha`, with `beta = sqrt(1 - alpha^2)`) splits the class-conditional mean
    shift between `Z` and `C`; the oracle statistic `alpha*Z + beta*C` is itself standard normal
    (both are independent standard normals), so `target_auc` is exact **only for a predictor
    with direct access to `Z` and `C`** -- the same §74 closed form as every other family here.
    The achievable AUC using only the *observed* `(P, C)` is strictly lower, since `P` is a noisy
    proxy for `Z` rather than `Z` itself (the same proxy-noise gap `latent_factor` measures);
    this function does not calibrate to that lower, observed-space number, since the point of
    the construction is the *drop* between `X_obs` and `X_intervened`, not hitting a fixed
    observed-space target.

    Args:
        rng: NumPy random generator.
        n_ctx: Context rows, always drawn from the intact (observational) confound path.
        n_query: Query rows, returned once under each condition.
        n_features: Total feature columns; must be at least 2.
        target_auc: Oracle-space (direct `Z`, `C` access) Bayes-optimal AUC.
        cause_weight: Share of the mean shift assigned to `Z` (`alpha`); the rest (`beta`) goes
            to `C`. `0.70710678` (`1/sqrt(2)`) splits it evenly by default.
        proxy_noise: Standard deviation of the noise added to `Z` to form the proxy `P`.

    Returns:
        `(X_obs, X_intervened, y, active_idx)`: the first two are `(n_ctx + n_query, n_features)`
        float32, identical in their first `n_ctx` rows; `y` is `(n_ctx + n_query,)` int64,
        identical between conditions; `active_idx` is `(2,)`, `[proxy_column, cause_column]`.

    Raises:
        ValueError: If `n_features < 2` or `cause_weight` is outside `[0, 1]`.
    """
    if n_features < 2:
        raise ValueError("n_features must be >= 2 (one proxy column, one cause column)")
    if not 0.0 <= cause_weight <= 1.0:
        raise ValueError(f"cause_weight must be in [0, 1], got {cause_weight}")

    n = n_ctx + n_query
    alpha = cause_weight
    beta = float(np.sqrt(max(0.0, 1.0 - alpha**2)))
    mu = _mu_for_target_auc(target_auc)

    y = rng.integers(0, 2, size=n)
    z = rng.normal(size=n)
    c = rng.normal(size=n)
    rows1 = y == 1
    z[rows1] += mu * alpha
    c[rows1] += mu * beta

    active = rng.choice(n_features, size=2, replace=False)
    active_p, active_c = int(active[0]), int(active[1])

    X_obs = rng.normal(size=(n, n_features))
    X_obs[:, active_c] = c
    X_obs[:, active_p] = z + rng.normal(0.0, proxy_noise, size=n)

    X_intervened = X_obs.copy()
    z_fresh = rng.normal(size=n_query)
    X_intervened[n_ctx:, active_p] = z_fresh + rng.normal(0.0, proxy_noise, size=n_query)

    return (
        X_obs.astype(np.float32),
        X_intervened.astype(np.float32),
        y.astype(np.int64),
        np.array([active_p, active_c]),
    )


def apply_nuisance_axes(
    task: Task,
    rng: np.random.Generator,
    missing_frac: float = 0.0,
    shift: float = 0.0,
    extrapolate: float = 0.0,
) -> Task:
    """Apply missingness, distribution shift and support extrapolation, independently of family.

    Task 40.6 (`docs/roadmap/ROADMAP.md` Phase C item 22): three nuisance axes that must be
    freely combinable with any of task 40.2's nine families (or 40.4/40.5's constructions)
    **without changing which family a task belongs to or the difficulty it was calibrated to**
    -- the factorisation task 40.6 asks for, rather than a tenth family that bundles them in.

    `shift` (additive) and `extrapolate` (multiplicative, applied first) are both **exact**
    no-ops on Bayes AUC: adding a constant to every row's feature values, or scaling every row's
    feature values by a positive constant, preserves every pairwise ordering of the informative
    statistic, and AUC depends only on that ordering -- the same invariance the `threshold`
    family's rank-preserving reshape already relies on (task 40.2). What they change is purely
    the feature *distribution* a downstream model sees: `shift` moves the whole support away
    from wherever training data sat, `extrapolate` widens it past whatever range the family's
    own calibration sample was drawn from. Both are covariate-shift probes with a known,
    exactly-zero effect on the label-generating process's own difficulty.

    `missing_frac` is not AUC-invariant -- it destroys information and cannot be, by
    construction -- but it also does not touch `y`, `task.n_classes` or `task.source`, so a
    task's family identity and its requested difficulty target remain exactly what they were;
    only the achieved AUC downstream of the missingness is free to move, and by how much is an
    empirical question for whoever measures it, not a silent change to the task's own labels.

    Args:
        task: A `Task` from any family in this module (or elsewhere).
        rng: NumPy random generator, for the missingness mask.
        missing_frac: Independent per-cell probability of being set to NaN.
        shift: Additive constant applied to every feature value.
        extrapolate: Multiplicative widening applied to every feature value before `shift`
            (`X *= 1 + extrapolate`); must leave `1 + extrapolate > 0` or the scaling would
            flip sign and reverse every ranking, which is not what "extrapolation" means here.

    Returns:
        A new `Task` with the same `y`, `n_classes`, `is_categorical` and `source` as `task`,
        and `X` transformed by the three axes in order (extrapolate, then shift, then
        missingness).

    Raises:
        ValueError: If `missing_frac` is outside `[0, 1]` or `1 + extrapolate <= 0`.
    """
    if not 0.0 <= missing_frac <= 1.0:
        raise ValueError(f"missing_frac must be in [0, 1], got {missing_frac}")
    if 1.0 + extrapolate <= 0.0:
        raise ValueError(f"1 + extrapolate must be > 0, got {1.0 + extrapolate}")

    X = task.X.astype(np.float64)
    if extrapolate:
        X = X * (1.0 + extrapolate)
    if shift:
        X = X + shift
    if missing_frac:
        X[rng.random(X.shape) < missing_frac] = np.nan

    return Task(
        X=X.astype(np.float32),
        y=task.y,
        n_classes=task.n_classes,
        is_categorical=task.is_categorical,
        source=task.source,
        period=task.period,
        n_horizons=task.n_horizons,
        y_continuous=task.y_continuous,
    )
