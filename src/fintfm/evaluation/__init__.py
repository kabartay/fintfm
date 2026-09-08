"""Evaluation: real datasets, metrics that see calibration, and benchmark harnesses.

Datasets loaded here are for **evaluation only**. Nothing real may reach pretraining — that
invariant is what makes a benchmark number auditable (``docs/FINDINGS.md`` §1).
"""

from fintfm.evaluation.datasets import (
    CreditDataset,
    SurvivalDataset,
    load_polish_bankruptcy,
    load_taiwan_bankruptcy,
    load_v4finbench,
)
from fintfm.evaluation.metrics import CreditMetrics, evaluate_binary

__all__ = [
    "CreditDataset",
    "CreditMetrics",
    "SurvivalDataset",
    "evaluate_binary",
    "load_polish_bankruptcy",
    "load_taiwan_bankruptcy",
    "load_v4finbench",
]
