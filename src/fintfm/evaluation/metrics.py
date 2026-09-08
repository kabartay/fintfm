"""Evaluation metrics for credit-risk prediction.

**AUC is not enough, and for this project it is not even the important one.** AUC measures
only ranking: whether defaulters score above non-defaulters. A model can rank perfectly and
still state that a borrower has a 40% chance of default when the true rate is 2%. A bank
cannot price, provision, or hold capital against a ranking — it needs a probability that
means what it says, and a supervisor validating the model will test exactly that.

So every reported result carries three kinds of number:

- **Discrimination** (``roc_auc``): can the model tell the classes apart at all.
- **Calibration** (``brier``, ``ece``, ``calibration_curve``): do the stated probabilities
  match observed frequencies.
- **Minority recall** at a chosen operating point: does it actually find defaulters, which
  a model can fail at while posting a fine AUC on a 4% base rate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score


@dataclass
class CreditMetrics:
    """A full scorecard for one model on one dataset.

    Attributes:
        roc_auc: Area under the ROC curve. Ranking only; blind to calibration.
        brier: Brier score (mean squared error of the predicted probability). Lower is
            better. A proper scoring rule, so it rewards honesty rather than confidence.
        ece: Expected calibration error, the average gap between stated probability and
            observed frequency, weighted by bin population.
        base_rate: Observed positive rate in the evaluation set.
        mean_predicted: Mean predicted probability. Compare against ``base_rate``: a large
            gap means the model is systematically over- or under-stating risk.
        recall_at_base_rate: Fraction of true defaults caught when flagging the
            highest-risk ``base_rate`` share of the population. A budget-realistic
            operating point, since a credit team can only review so many files.
        n: Number of evaluation rows.
        n_positive: Number of positives, which sets how much any of this can be trusted.
        brier_skill: Improvement over a **feature-free constant predictor** at the base rate,
            ``1 - brier / brier_reference``. On an imbalanced problem raw Brier is dominated
            by the negatives, so a trivial baseline scores well and the achievable range is
            narrow — measured at 1-2% for this project's best model (``docs/FINDINGS.md``
            §17). Skill is the number that means something; raw Brier is not.
        is_degenerate: True when the model has essentially no discriminative content
            (AUC at or below 0.55) *despite* possibly excellent calibration. A constant
            base-rate predictor has ECE near zero and AUC exactly 0.5, so calibration alone
            must never read as success.
    """

    roc_auc: float
    brier: float
    ece: float
    base_rate: float
    mean_predicted: float
    recall_at_base_rate: float
    n: int
    n_positive: int
    brier_skill: float = float("nan")
    is_degenerate: bool = False
    bins: list[tuple[float, float, int]] = field(default_factory=list)

    def summary(self) -> str:
        """One-line rendering for benchmark output."""
        flag = "  [DEGENERATE: no discriminative content]" if self.is_degenerate else ""
        return (
            f"AUC={self.roc_auc:.4f} Brier={self.brier:.4f} (skill={self.brier_skill:+.2%}) "
            f"ECE={self.ece:.4f} recall@{self.base_rate:.1%}={self.recall_at_base_rate:.3f} "
            f"(pred mean {self.mean_predicted:.3%} vs actual {self.base_rate:.3%}, "
            f"{self.n_positive}/{self.n} positive){flag}"
        )


def expected_calibration_error(
    y_true: np.ndarray, p: np.ndarray, n_bins: int = 10
) -> tuple[float, list[tuple[float, float, int]]]:
    """Compute expected calibration error with equal-width bins.

    Args:
        y_true: Binary outcomes.
        p: Predicted probability of the positive class.
        n_bins: Number of probability bins.

    Returns:
        A tuple of the ECE and a list of ``(mean_predicted, observed_rate, count)`` per
        non-empty bin, so a caller can render a reliability table rather than one number.
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    total, bins = 0.0, []
    for b in range(n_bins):
        sel = idx == b
        count = int(sel.sum())
        if count == 0:
            continue
        predicted, observed = float(p[sel].mean()), float(y_true[sel].mean())
        bins.append((predicted, observed, count))
        total += count * abs(predicted - observed)
    return total / max(len(p), 1), bins


def recall_at_top_k(y_true: np.ndarray, p: np.ndarray, k: int) -> float:
    """Fraction of positives captured in the ``k`` highest-risk cases.

    Args:
        y_true: Binary outcomes.
        p: Predicted probability of the positive class.
        k: Size of the reviewed pool.

    Returns:
        Recall in ``[0, 1]``; ``0.0`` when there are no positives to find.
    """
    positives = int(y_true.sum())
    if positives == 0 or k <= 0:
        return 0.0
    order = np.argsort(-p)[: min(k, len(p))]
    return float(y_true[order].sum() / positives)


def evaluate_binary(y_true: np.ndarray, p: np.ndarray, n_bins: int = 10) -> CreditMetrics:
    """Score binary probabilistic predictions on discrimination *and* calibration.

    Args:
        y_true: Binary outcomes, shape ``(n,)``.
        p: Predicted probability of the positive class, shape ``(n,)``.
        n_bins: Bins for the calibration error.

    Returns:
        A :class:`CreditMetrics`.

    Raises:
        ValueError: If the inputs disagree in length.
    """
    y_true = np.asarray(y_true).astype(int)
    p = np.asarray(p, dtype=float)
    if y_true.shape != p.shape:
        raise ValueError(f"shape mismatch: y_true {y_true.shape} vs p {p.shape}")
    base_rate = float(y_true.mean()) if len(y_true) else 0.0
    # AUC is undefined with a single class present; report NaN rather than inventing a value.
    auc = roc_auc_score(y_true, p) if len(np.unique(y_true)) > 1 else float("nan")
    ece, bins = expected_calibration_error(y_true, p, n_bins=n_bins)
    brier = float(brier_score_loss(y_true, p)) if len(y_true) else float("nan")
    # reference: predict the base rate for everyone, using no features at all
    brier_ref = base_rate * (1.0 - base_rate) if len(y_true) else float("nan")
    skill = 1.0 - brier / brier_ref if brier_ref and np.isfinite(brier) else float("nan")
    degenerate = bool(np.isfinite(auc) and auc <= 0.55)
    return CreditMetrics(
        roc_auc=float(auc),
        brier=brier,
        brier_skill=float(skill),
        is_degenerate=degenerate,
        ece=float(ece),
        base_rate=base_rate,
        mean_predicted=float(p.mean()) if len(p) else float("nan"),
        recall_at_base_rate=recall_at_top_k(y_true, p, round(base_rate * len(y_true))),
        n=len(y_true),
        n_positive=int(y_true.sum()),
        bins=bins,
    )


def paired_auc_difference(
    y_true: np.ndarray,
    p_a: np.ndarray,
    p_b: np.ndarray,
    n_boot: int = 2000,
    seed: int = 0,
) -> tuple[float, tuple[float, float], float]:
    """Bootstrap the AUC difference between two models scored on the *same* rows.

    Comparing two models on one test set gives correlated AUCs, so an unpaired comparison
    overstates uncertainty and a bare difference understates it. Resampling rows and
    recomputing both AUCs on each resample keeps the pairing.

    This exists because a benchmark that compares several variants across several panels
    manufactures winners by chance. The credit-risk literature is explicit about it:
    Baesens et al. (arXiv:2605.18147) found statistical significance in only 22 of 406
    pairwise comparisons, and the finance forecasting literature routinely applies
    data-snooping controls (White's Reality Check, Hansen's SPA) for the same reason. A
    win count is not a result.

    Args:
        y_true: Binary outcomes, shape ``(n,)``.
        p_a: Model A's predicted probability of the positive class.
        p_b: Model B's predicted probability, on the same rows in the same order.
        n_boot: Bootstrap resamples.
        seed: Random seed.

    Returns:
        ``(delta, (lo, hi), p_two_sided)`` where ``delta`` is ``AUC(a) - AUC(b)`` on the
        full sample, the interval is the 95% percentile bootstrap interval, and the p-value
        is the two-sided bootstrap proportion of resamples whose sign disagrees with
        ``delta``. Returns NaNs if either class is absent.

    Raises:
        ValueError: If the three arrays disagree in length.
    """
    y_true = np.asarray(y_true).astype(int)
    p_a = np.asarray(p_a, dtype=float)
    p_b = np.asarray(p_b, dtype=float)
    if not (y_true.shape == p_a.shape == p_b.shape):
        raise ValueError(
            f"shape mismatch: y_true {y_true.shape}, p_a {p_a.shape}, p_b {p_b.shape}"
        )
    if len(np.unique(y_true)) < 2:
        return float("nan"), (float("nan"), float("nan")), float("nan")

    delta = float(roc_auc_score(y_true, p_a) - roc_auc_score(y_true, p_b))
    rng = np.random.default_rng(seed)
    n = len(y_true)
    deltas = np.empty(n_boot)
    drawn = 0
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        # a resample missing a class has no defined AUC; redraw rather than silently skew
        if len(np.unique(y_true[idx])) < 2:
            continue
        deltas[drawn] = roc_auc_score(y_true[idx], p_a[idx]) - roc_auc_score(
            y_true[idx], p_b[idx]
        )
        drawn += 1
    if drawn < 100:  # too few usable resamples to say anything
        return delta, (float("nan"), float("nan")), float("nan")
    deltas = deltas[:drawn]
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    # two-sided: how often the resampled difference contradicts the observed sign
    p = 2.0 * min((deltas <= 0).mean(), (deltas >= 0).mean())
    return delta, (float(lo), float(hi)), float(min(p, 1.0))


def holm_bonferroni(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """Holm-Bonferroni step-down correction for a family of comparisons.

    Controls the family-wise error rate, which matters here because a benchmark sweeping
    variants across panels and context strategies runs dozens of tests. Without a
    correction, roughly one in twenty reads as significant by construction.

    Args:
        p_values: Raw two-sided p-values. NaNs are treated as non-significant.
        alpha: Family-wise error rate.

    Returns:
        A list of booleans, aligned with ``p_values``, marking which survive.
    """
    n = len(p_values)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: (np.isnan(p_values[i]), p_values[i]))
    verdict = [False] * n
    for rank, i in enumerate(order):
        p = p_values[i]
        if np.isnan(p) or p > alpha / (n - rank):
            break  # step-down: once one fails, all larger p-values fail too
        verdict[i] = True
    return verdict
