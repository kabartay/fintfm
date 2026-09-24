"""Discrete-time hazard head: a PD term structure that cannot contradict itself.

Why this exists
---------------
Independent per-horizon predictions produce curves that disagree with themselves. Measured
on real data (``docs/results/FINDINGS.md`` §11): **11% of horizon steps and 39% of firms** received a
cumulative default probability that *fell* as the horizon grew, while the portfolio aggregate
stayed monotone and hid it. A firm that has defaulted by year three has defaulted by year
five, so such a curve is not merely inaccurate, it is incoherent — and IFRS 9 lifetime
expected credit loss consumes exactly this curve.

The fix is structural rather than statistical. Predict a **hazard** per period,

    h_k = P(default in period k | survived to period k)  in (0, 1),

and derive the cumulative default probability as

    F_k = 1 - prod_{j<=k} (1 - h_j).

Because every factor ``(1 - h_j)`` lies in ``(0, 1]``, the survival function is
non-increasing and therefore ``F_k`` is **non-decreasing in k by construction**. No amount of
training can produce a violation, and no monotonicity penalty is needed. That is the
difference between a design advantage and a scale advantage: this one is proved, not bought.

The training objective is the standard discrete-time survival likelihood, which is a proper
scoring rule over the whole curve rather than over one horizon at a time.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

#: Sentinel period for a firm that never defaults within the observed horizon grid.
CENSORED = -1


class HazardHead(nn.Module):
    """Maps a row representation to a monotone cumulative-PD term structure.

    Args:
        d_model: Width of the incoming row representation.
        n_horizons: Number of discrete periods on the horizon grid, ``K``.
        max_hazard: Upper clamp on any single-period hazard. Keeps the log-likelihood
            finite and stops one saturated period from forcing ``F_k`` to 1 for every
            later horizon, which would destroy the curve's shape.
    """

    def __init__(self, d_model: int, n_horizons: int, max_hazard: float = 0.999) -> None:
        """Build the hazard head.

        Args:
            d_model: Width of the row representation this head reads.
            n_horizons: Length of the horizon grid. Cumulative PD is produced for each.
            max_hazard: Ceiling on a single period's hazard, keeping the cumulative product
                strictly below one so a log-likelihood stays finite.

        Raises:
            ValueError: If ``n_horizons`` is below one.
        """
        super().__init__()
        if n_horizons < 1:
            raise ValueError(f"n_horizons must be >= 1, got {n_horizons}")
        self.n_horizons = n_horizons
        self.max_hazard = max_hazard
        self.proj = nn.Linear(d_model, n_horizons)

    def hazards(self, h: torch.Tensor) -> torch.Tensor:
        """Per-period hazards in ``(0, max_hazard]``.

        Args:
            h: ``(..., d_model)`` row representations.

        Returns:
            ``(..., K)`` hazards.
        """
        return torch.sigmoid(self.proj(h)).clamp(1e-7, self.max_hazard)

    def cumulative_pd(self, h: torch.Tensor) -> torch.Tensor:
        """Cumulative default probability at each horizon, non-decreasing by construction.

        Args:
            h: ``(..., d_model)`` row representations.

        Returns:
            ``(..., K)`` cumulative PD, where index ``k`` is the probability of default by
            the end of period ``k + 1``. Guaranteed ``F_k <= F_{k+1}`` for every row.
        """
        survival = torch.cumprod(1.0 - self.hazards(h), dim=-1)
        return 1.0 - survival

    def loss(
        self,
        h: torch.Tensor,
        period: torch.Tensor,
        n_observed: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Discrete-time survival negative log-likelihood.

        For a firm defaulting in period ``t`` the likelihood is
        ``h_t * prod_{j<t} (1 - h_j)``; for a censored firm (no default within the grid) it
        is ``prod_{j<=K} (1 - h_j)``. Optimising this fits the **whole curve** rather than
        one horizon at a time, which is what makes the horizons mutually consistent instead
        of merely non-contradictory.

        **Censoring is per row, not global.** A firm observed for only three horizons because
        the panel ends has not "survived five"; it contributed evidence about three. Real
        panels are ragged this way — V4FinBench drops from 1,000,087 rows at h=0 to 598,832
        at h=5 for exactly this reason — and scoring a short-observed firm as a long-run
        survivor would bias every hazard downward.

        Args:
            h: ``(N, d_model)`` row representations.
            period: ``(N,)`` zero-based default period, or :data:`CENSORED` for no default
                observed within that row's observation window.
            n_observed: ``(N,)`` number of horizons actually observed for each row. Defaults
                to the full grid for every row, which is correct only for synthetic tasks
                where every firm is followed for the whole grid.

        Returns:
            Scalar mean negative log-likelihood.

        Raises:
            ValueError: If a period index falls outside the horizon grid.
        """
        hz = self.hazards(h)
        n, k = hz.shape[-2], hz.shape[-1]
        if int(period.max()) >= k:
            raise ValueError(f"period index {int(period.max())} outside grid of {k} horizons")

        log_surv = torch.log1p(-hz)  # log(1 - h_j), numerically safer than log(1-h)
        idx = torch.arange(k, device=hz.device).expand(n, k)
        censored = period.view(-1, 1) == CENSORED
        if n_observed is None:
            observed = torch.full_like(period.view(-1, 1), k)
        else:
            observed = n_observed.view(-1, 1).clamp(0, k)
        # survived strictly before the default period; for a censored row, survived only
        # through the horizons it was actually observed for
        before = idx < torch.where(censored, observed, period.view(-1, 1))
        ll = (log_surv * before).sum(dim=-1)
        # plus the hazard of defaulting in that period, for uncensored rows only
        at = F.one_hot(period.clamp(min=0), num_classes=k).to(hz.dtype)
        ll = ll + torch.where(
            censored.squeeze(-1), torch.zeros_like(ll), (torch.log(hz) * at).sum(dim=-1)
        )
        return -ll.mean()


def coherence_violations(cumulative_pd: torch.Tensor) -> torch.Tensor:
    """Count places where a cumulative-PD curve decreases.

    A diagnostic kept permanently rather than a one-off check: §11's defect was invisible at
    portfolio level, so it must be measured per row wherever a term structure is produced.

    Args:
        cumulative_pd: ``(..., K)`` cumulative PD.

    Returns:
        Scalar count of ``(row, step)`` pairs where the curve falls. Zero is required for any
        curve produced by :meth:`HazardHead.cumulative_pd`.
    """
    return (cumulative_pd.diff(dim=-1) < 0).sum()


def base_rate_shift(context_rate: float, true_rate: float) -> float:
    """Log-odds shift undoing the base-rate distortion of a resampled context.

    An in-context model reads its default rate out of its context, so a context resampled
    to be class-balanced makes it state a default rate near the context's, not the
    portfolio's (``docs/results/FINDINGS.md`` §5, decision D5). On the term-structure path this is
    severe: a balanced 2,000-row context against a 1.5% portfolio inflates the stated PD by
    more than an order of magnitude while leaving AUC untouched, because every prediction
    inflates alike.

    Args:
        context_rate: Fraction of the context labelled as eventually defaulting.
        true_rate: Fraction of the full training population that eventually defaults.

    Returns:
        ``logit(true_rate) - logit(context_rate)``, to be added to a log-odds prediction.
        Zero when the two agree, so applying it to an unresampled context is a no-op.

    Raises:
        ValueError: If either rate is not strictly inside ``(0, 1)``. A degenerate rate has
            no finite log-odds, and silently clamping it would fabricate a correction.
    """
    for name, p in (("context_rate", context_rate), ("true_rate", true_rate)):
        if not 0.0 < p < 1.0:
            raise ValueError(f"{name} must lie strictly in (0, 1), got {p}")
    from math import log

    return log(true_rate / (1.0 - true_rate)) - log(context_rate / (1.0 - context_rate))


def shift_cumulative_pd(cumulative_pd: torch.Tensor, delta: float) -> torch.Tensor:
    """Apply a log-odds shift to a cumulative-PD curve, preserving its coherence.

    The shift is applied in logit space and inverted, which is a strictly increasing map on
    ``(0, 1)``. A strictly increasing map applied elementwise cannot reorder a
    non-decreasing sequence, so **a monotone curve stays monotone** and the structural
    guarantee of :meth:`HazardHead.cumulative_pd` survives the correction. It is also
    rank-preserving across rows at a fixed horizon, so AUC is unchanged by construction.

    The same ``delta`` is applied at every horizon. That is exact only if the context's
    distortion is horizon-independent, which it is not: a binary context says *who*
    defaulted and never *when*, so no horizon-specific rate can be recovered from it. Treat
    the levels as corrected to first order, not calibrated. Supplying per-horizon context
    labels is the real fix (``openspec/changes/survival-context-labels``).

    Args:
        cumulative_pd: ``(..., K)`` cumulative PD in ``(0, 1)``.
        delta: Log-odds shift, e.g. from :func:`base_rate_shift`.

    Returns:
        ``(..., K)`` shifted cumulative PD, non-decreasing wherever the input was.
    """
    if delta == 0.0:
        return cumulative_pd
    p = cumulative_pd.clamp(1e-7, 1.0 - 1e-7)
    return torch.sigmoid(torch.logit(p) + delta)
