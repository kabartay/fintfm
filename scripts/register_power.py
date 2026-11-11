"""Power calculation for the forward prediction register (openspec task 6.2).

Answers, before any cohort is fixed: for a cohort of ``n`` companies with annual default
rate ``p``, how many defaults arrive per year, how wide is the CI on one model's AUC, and what
is the smallest AUC difference between two models detectable at alpha 0.05 and 80% power.

Method (all numbers it prints are ESTIMATED, none is measured on any model or cohort):

* Variance of one AUC: Hanley and McNeil (1982), Radiology 143:29-36, equation 2.
* Variance of a difference between two AUCs on the same cases: ``V1 + V2 - 2 r sqrt(V1 V2)``,
  the form Hanley and McNeil (1983) use with the correlation ``r`` between the two AUC
  estimates. DeLong et al. (1988) estimate that covariance non-parametrically; here ``r`` is
  an assumed input and the output is shown for several values of it.
* Minimum detectable difference: ``(z_{1-alpha/2} + z_{power}) * sd(diff)``, solved by
  fixed-point iteration because the variance of the better model depends on the difference.

Usage::

    nice -n 10 uv run --frozen python scripts/register_power.py [--auc 0.80]
"""

from __future__ import annotations

import argparse
import math
from statistics import NormalDist

#: External figure: England and Wales company insolvencies per 10,000 companies on the effective
#: register, 12 months to 30 September 2026. Insolvency Service, "Company Insolvency Statistics
#: September 2026", https://www.gov.uk/government/statistics/company-insolvencies-september-2026,
#: published 2026-10-20. The denominator is registered companies, not active ones.
UK_INSOLVENCY_PER_10K = 49.9

#: Measured in this repo: V4FinBench positive rate at h=0 (next-year label), FINDINGS section 22.
V4_RATE = 0.0036

#: Sensitivity only, DERIVED here (not published as a rate): 23,938 company insolvencies in 2025
#: over about 2.1 million active businesses with legal status company. Mixes two Insolvency
#: Service releases; treat as an upper-side illustration.
UK_ACTIVE_DENOMINATOR_RATE = 23938 / 2.1e6

ALPHA = 0.05
POWER = 0.80


def hanley_mcneil_var(auc: float, n_pos: float, n_neg: float) -> float:
    """Variance of an AUC estimate, Hanley and McNeil (1982) equation 2.

    Args:
        auc: True or assumed area under the ROC curve, strictly between 0.5 and 1.
        n_pos: Number of positive cases (defaults).
        n_neg: Number of negative cases (survivors).

    Returns:
        The variance of the AUC estimate.
    """
    q1 = auc / (2.0 - auc)
    q2 = 2.0 * auc * auc / (1.0 + auc)
    num = auc * (1 - auc) + (n_pos - 1) * (q1 - auc**2) + (n_neg - 1) * (q2 - auc**2)
    return num / (n_pos * n_neg)


def diff_var(auc1: float, auc2: float, n_pos: float, n_neg: float, r: float) -> float:
    """Variance of the difference of two correlated AUC estimates.

    Args:
        auc1: AUC of the first model.
        auc2: AUC of the second model.
        n_pos: Number of defaults.
        n_neg: Number of survivors.
        r: Assumed correlation between the two AUC estimates.

    Returns:
        ``V1 + V2 - 2 r sqrt(V1 V2)``.
    """
    v1 = hanley_mcneil_var(auc1, n_pos, n_neg)
    v2 = hanley_mcneil_var(auc2, n_pos, n_neg)
    return v1 + v2 - 2.0 * r * math.sqrt(v1 * v2)


def min_detectable_diff(auc: float, n_pos: float, n_neg: float, r: float) -> float:
    """Smallest AUC gain over a baseline detectable at ``ALPHA`` (two-sided) and ``POWER``.

    Args:
        auc: AUC of the weaker model.
        n_pos: Number of defaults.
        n_neg: Number of survivors.
        r: Assumed correlation between the two AUC estimates.

    Returns:
        The minimum detectable difference, or ``nan`` if it exceeds ``1 - auc``.
    """
    z = NormalDist().inv_cdf(1 - ALPHA / 2) + NormalDist().inv_cdf(POWER)
    delta = 0.01
    for _ in range(100):
        if auc + delta >= 0.9999:
            return float("nan")
        new = z * math.sqrt(diff_var(auc, auc + delta, n_pos, n_neg, r))
        if abs(new - delta) < 1e-9:
            return new
        delta = new
    return delta


def ci_half_width(auc: float, n_pos: float, n_neg: float) -> float:
    """Half-width of the 95% normal CI on a single AUC.

    Args:
        auc: Assumed AUC.
        n_pos: Number of defaults.
        n_neg: Number of survivors.

    Returns:
        ``1.96 * sd`` of the AUC estimate.
    """
    return NormalDist().inv_cdf(0.975) * math.sqrt(hanley_mcneil_var(auc, n_pos, n_neg))


def main() -> None:
    """Print the power tables for the cohort-size and default-rate grid."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--auc", type=float, default=0.80, help="assumed AUC of the weaker model")
    args = ap.parse_args()
    rates = [
        ("UK insolvency, external, 49.9/10k", UK_INSOLVENCY_PER_10K / 1e4),
        ("V4FinBench h=0, measured here, 0.36%", V4_RATE),
        ("UK active-company denominator, derived, 1.14%", UK_ACTIVE_DENOMINATOR_RATE),
    ]
    sizes = [5_000, 20_000, 100_000, 500_000, 2_000_000]
    rs = [0.3, 0.5, 0.7, 0.9]
    print(f"ESTIMATED. Weaker-model AUC {args.auc}, alpha {ALPHA}, power {POWER}, one year.")
    for label, p in rates:
        print(f"\nRate: {label}")
        head = "cohort      defaults/yr  CI half-width |" + "".join(f" MDD r={r:<4}" for r in rs)
        print(head)
        for n in sizes:
            d = n * p
            if d < 2:
                print(f"{n:>9,}  {d:>10.1f}  (fewer than 2 defaults)")
                continue
            cells = "".join(
                f" {min_detectable_diff(args.auc, d, n - d, r):>9.4f}" for r in rs
            )
            print(f"{n:>9,}  {d:>10.1f}  {ci_half_width(args.auc, d, n - d):>13.4f} |{cells}")


if __name__ == "__main__":
    main()
