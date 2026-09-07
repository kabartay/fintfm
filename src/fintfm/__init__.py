"""FinancialTFM: prior-fitted tabular foundation model for financial risk.

The model is pretrained once on synthetic tasks drawn from a *financial prior*
(a generative story for company/customer tables and their default labels) mixed
with a generic structural-causal prior. At inference the labelled table is
passed as context and unlabelled rows as queries; no gradient steps happen on
customer data.
"""

from fintfm.classifier import FinancialTFMClassifier
from fintfm.model import FinancialTFM, ModelConfig

__all__ = ["FinancialTFM", "FinancialTFMClassifier", "ModelConfig"]
