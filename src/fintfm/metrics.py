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
    """

    roc_auc: float
    brier: float
    ece: float
    base_rate: float
    mean_predicted: float
    recall_at_base_rate: float
    n: int
    n_positive: int
    bins: list[tuple[float, float, int]] = field(default_factory=list)

    def summary(self) -> str:
        """One-line rendering for benchmark output."""
        return (
            f"AUC={self.roc_auc:.4f} Brier={self.brier:.4f} ECE={self.ece:.4f} "
            f"recall@{self.base_rate:.1%}={self.recall_at_base_rate:.3f} "
            f"(pred mean {self.mean_predicted:.3%} vs actual {self.base_rate:.3%}, "
            f"{self.n_positive}/{self.n} positive)"
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
    return CreditMetrics(
        roc_auc=float(auc),
        brier=float(brier_score_loss(y_true, p)) if len(y_true) else float("nan"),
        ece=float(ece),
        base_rate=base_rate,
        mean_predicted=float(p.mean()) if len(p) else float("nan"),
        recall_at_base_rate=recall_at_top_k(y_true, p, round(base_rate * len(y_true))),
        n=len(y_true),
        n_positive=int(y_true.sum()),
        bins=bins,
    )
