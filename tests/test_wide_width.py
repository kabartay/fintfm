"""The prior can emit, and the model can take, a table 512 columns wide (design review DR8.3).

Before this the financial prior stopped near 370 columns however wide it was asked to go (its
20 accounts give 380 ordered ratios), so a ``max_features`` 512 checkpoint would never have
seen a table wider than that.
"""

import numpy as np
import pytest
import torch

from fintfm.modeling.model import FinancialTFM, ModelConfig
from fintfm.prior import financial
from fintfm.prior.financial import sample_financial_task


@pytest.mark.parametrize("width", [300, 450, 500])
def test_financial_prior_reaches_the_requested_width(width: int) -> None:
    """A task asked for ``width`` columns carries at least that many, within the cap."""
    for seed in range(3):
        t = sample_financial_task(
            np.random.default_rng(seed), 96, max_features=512, n_features=width
        )
        assert width <= t.n_features <= 512
        assert t.X.shape[1] == len(t.is_categorical)


def test_extension_is_a_no_op_for_a_task_within_the_old_reach(monkeypatch) -> None:
    """A 100-column task never calls the extension, so its stream is what it was."""
    a = sample_financial_task(np.random.default_rng(7), 64, max_features=136, n_features=100)

    def refuse(*args, **kwargs):
        raise AssertionError("the extension ran for a task the pair loop filled")

    monkeypatch.setattr(financial, "_extend_ratio_family", refuse)
    b = sample_financial_task(np.random.default_rng(7), 64, max_features=136, n_features=100)
    np.testing.assert_array_equal(a.X, b.X)


def test_extension_keys_are_distinct() -> None:
    """The ratio family returns no duplicate column key at 480 columns."""
    rng = np.random.default_rng(0)
    names = ["total_assets", "current_assets", "cash", "receivables", "inventory",
             "working_capital", "total_liabilities", "short_term_liabilities", "long_term_debt",
             "equity", "retained_earnings", "revenue", "ebitda", "ebit", "net_profit",
             "operating_expenses", "depreciation", "interest", "employees", "age"]
    acc = {n: np.exp(rng.normal(size=32)) for n in names}
    cols, used = financial._ratio_family(rng, acc, 480)
    assert len(cols) == 480
    assert len(used) == len(set(used))


def test_model_takes_512_columns_and_refuses_513() -> None:
    """A tiny model at ``max_features`` 512 encodes a 512-wide batch, back-propagates, and
    refuses 513."""
    cfg = ModelConfig(max_features=512, max_classes=2, d_cell=8, d_model=16, n_heads=2,
                      n_col_layers=1, n_layers=1, d_ff=32, n_cell_blocks=1, cell_labels=True,
                      column_id_dim=4)
    model = FinancialTFM(cfg)
    model.feature_chunk = 8
    y = torch.randint(0, 2, (1, 12))
    out = model.encode_rows(torch.randn(1, 12, 512), 8, y=y)
    assert out.shape == (1, 12, 16)
    out.sum().backward()
    with pytest.raises(ValueError, match="512"):
        model.encode_rows(torch.randn(1, 12, 513), 8, y=y)
