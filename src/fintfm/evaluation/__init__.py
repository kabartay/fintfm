"""Evaluation: real datasets, metrics that see calibration, and benchmark harnesses.

Datasets loaded here are for **evaluation only**. Nothing real may reach pretraining — that
invariant is what makes a benchmark number auditable (``docs/FINDINGS.md`` §1).
"""

from fintfm.evaluation.datasets import CreditDataset, load_polish_bankruptcy
from fintfm.evaluation.metrics import CreditMetrics, evaluate_binary

__all__ = [
    "CreditDataset",
    "CreditMetrics",
    "evaluate_binary",
    "load_polish_bankruptcy",
]
