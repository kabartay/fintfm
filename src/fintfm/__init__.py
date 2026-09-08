"""FinancialTFM: prior-fitted tabular foundation model for financial risk.

The model is pretrained once on synthetic tasks drawn from a *financial prior* (a generative
story for company tables and their default labels) mixed with a generic structural-causal
prior. At inference the labelled table is passed as context and unlabelled rows as queries;
no gradient steps happen on customer data.

Package layout follows the pipeline:

    prior/        synthetic task generation — the only source of pretraining data
    modeling/     architecture and the pretraining loop
    inference/    in-context prediction, context construction, calibration correction
    evaluation/   real datasets, calibration-aware metrics, benchmark harnesses
    experiments/  designed experiments with pre-stated exit conditions

See ``docs/ARCHITECTURE.md`` for how the model works and ``docs/STRATEGY.md`` for why.
"""

from fintfm.evaluation.metrics import CreditMetrics, evaluate_binary
from fintfm.inference.classifier import ContextStrategy, FinancialTFMClassifier
from fintfm.modeling.model import FinancialTFM, ModelConfig

__all__ = [
    "ContextStrategy",
    "CreditMetrics",
    "FinancialTFM",
    "FinancialTFMClassifier",
    "ModelConfig",
    "evaluate_binary",
]
