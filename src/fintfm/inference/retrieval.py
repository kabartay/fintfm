"""Build a query's context from similar firms instead of a blind sample.

Why this exists
---------------
``docs/FINDINGS.md`` §31 decomposed the out-of-time gap against per-horizon logistic
regression and found **84% of it is horizon-independent**: a constant 0.132 AUC deficit
present already at the first horizon, against a baseline fitted on 72,622 rows while the
model sees a 2,000-row context. That is §16's crossover appearing on real data.

Two escape routes are closed. Supplying more rows does not work — uniform context scores
0.6921 / 0.7192 / 0.6986 mean AUC at 1,000 / 2,000 / 4,000 rows, peaking in the middle and
falling beyond it, because pretraining used 256-1,024-row tasks. Pretraining on larger tasks
is priced out at 32x per step for 2,048 rows and 500x for 4,096 (``docs/COMPUTE.md``).

So the only remaining lever on the dominant term is **which** rows go in the context. A
credit analyst comparing a firm to its sector and size peers is doing retrieval; a uniform
sample of the book is not.

The invariant this breaks, stated plainly
-----------------------------------------
Every other context strategy here selects rows **without looking at the query**, which is
what makes chunked inference exact: the row mask forbids query-to-query attention, so a
prediction depends only on the context and the query itself, and batching cannot change it.

**Retrieval necessarily makes the context query-dependent**, and per-query retrieval costs
one forward pass per query — roughly 500x the work of chunked scoring at a 2,000-row context,
which is not usable on 47,378 firms. So :func:`group_queries` retrieves one context per
*group* of similar queries, and a prediction then depends on that query's group-mates.

That is a real weakening of a documented design property, and it is why both modes exist:
``exact`` for validating on a subsample, ``grouped`` for scale, with the gap between them
measured rather than assumed small.
"""

from __future__ import annotations

import numpy as np

#: Feature-wise scale floor, so a constant or near-constant column cannot dominate distance.
_SCALE_FLOOR = 1e-6


def normalise_for_distance(
    X: np.ndarray, centre: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    """Put features on a common scale and make missing values distance-neutral.

    Financial ratios differ by orders of magnitude, so an unnormalised Euclidean distance is
    effectively a distance on whichever column has the largest units. Robust statistics are
    used because the tails here are extreme by nature — a leverage ratio at a firm about to
    default is not a mild outlier.

    Missing values are set to the centre **after** normalisation, which makes an absent
    feature contribute zero to the distance rather than excluding the row. That is the right
    default for a panel where missingness is structural, but note it makes a row with many
    missing features look artificially average and therefore artificially close to
    everything.

    Args:
        X: ``(n, F)`` raw features, NaN for missing.
        centre: ``(F,)`` per-feature centre, from the training rows only.
        scale: ``(F,)`` per-feature scale, from the training rows only.

    Returns:
        ``(n, F)`` normalised features with no NaN.
    """
    Z = (np.asarray(X, dtype=np.float32) - centre) / np.maximum(scale, _SCALE_FLOOR)
    return np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)


def distance_stats(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Robust per-feature centre and scale for distance, computed on training rows only.

    Args:
        X: ``(n, F)`` training features, NaN for missing.

    Returns:
        ``(centre, scale)``, the median and the interquartile range.
    """
    with np.errstate(all="ignore"):
        centre = np.nanmedian(X, axis=0)
        q1, q3 = np.nanpercentile(X, 25, axis=0), np.nanpercentile(X, 75, axis=0)
    centre = np.nan_to_num(centre, nan=0.0)
    return centre.astype(np.float32), np.nan_to_num(q3 - q1, nan=1.0).astype(np.float32)


def group_queries(
    Zq: np.ndarray, n_groups: int, max_group: int, rng: np.random.Generator
) -> list[np.ndarray]:
    """Partition queries into similar groups, each sharing one retrieved context.

    Uses k-means on the normalised features, then splits any group larger than ``max_group``
    so a single forward pass stays affordable. Splitting an over-large group is arbitrary
    within that group, which is acceptable precisely because its members are already similar
    — that is the whole basis of the approximation.

    Args:
        Zq: ``(n, F)`` normalised query features.
        n_groups: Target number of groups. More groups means sharper contexts and more
            forward passes; the cost is linear in this.
        max_group: Hard cap on group size, to bound the attention cost per pass.
        rng: Random generator, for reproducible initialisation and splitting.

    Returns:
        A list of index arrays into ``Zq``, together covering every row exactly once.
    """
    n = Zq.shape[0]
    if n_groups <= 1 or n <= max_group and n_groups == 1:
        return [np.arange(n)]
    from sklearn.cluster import MiniBatchKMeans

    k = max(1, min(n_groups, n))
    labels = MiniBatchKMeans(
        n_clusters=k, random_state=int(rng.integers(1 << 31)), n_init=3, batch_size=4096
    ).fit_predict(Zq)
    groups: list[np.ndarray] = []
    for c in range(k):
        idx = np.flatnonzero(labels == c)
        if idx.size == 0:
            continue
        for start in range(0, idx.size, max_group):
            groups.append(idx[start : start + max_group])
    return groups


def retrieve(
    Zt: np.ndarray,
    yt: np.ndarray,
    target: np.ndarray,
    max_context: int,
    min_positive: int = 0,
) -> np.ndarray:
    """Indices of the training rows nearest a target point, with a floor on positives.

    Nearest-neighbour retrieval preserves the population base rate only in expectation, and
    in a low-default portfolio a single group can easily draw **zero** defaults — a context
    with no positive examples tells an in-context model nothing about default at all.
    ``min_positive`` guarantees a floor by admitting the nearest positives even when they
    fall outside the plain neighbourhood.

    Note this is deliberately *not* class balancing. ``docs/FINDINGS.md`` §29 measured
    balancing as costing 10-12 AUC points here, so the floor is set low enough to insure
    against an empty class rather than to equalise the classes.

    Args:
        Zt: ``(n_train, F)`` normalised training features.
        yt: ``(n_train,)`` coded training labels.
        target: ``(F,)`` normalised point to retrieve around.
        max_context: Number of rows to return.
        min_positive: Minimum rows of the positive class to include.

    Returns:
        Sorted indices into ``Zt``, of length ``min(max_context, n_train)``.
    """
    k = min(max_context, Zt.shape[0])
    d = np.linalg.norm(Zt - target, axis=1)
    picked = np.argpartition(d, k - 1)[:k] if k < Zt.shape[0] else np.arange(Zt.shape[0])
    if min_positive > 0:
        pos_class = int(yt.max())
        have = int((yt[picked] == pos_class).sum())
        short = min_positive - have
        if short > 0:
            pos = np.flatnonzero(yt == pos_class)
            if pos.size:
                extra = pos[np.argsort(d[pos])[: min(short, pos.size)]]
                # drop the farthest current picks to make room, keeping the size fixed
                keep = picked[np.argsort(d[picked])][: max(k - extra.size, 0)]
                picked = np.union1d(keep, extra)
    return np.sort(picked)
