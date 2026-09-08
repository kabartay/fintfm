import numpy as np
import torch

from fintfm.modeling.model import FinancialTFM, ModelConfig, normalize_features
from fintfm.prior import PriorConfig
from fintfm.prior.mixture import sample_batch


def test_normalize_features_masks_missing():
    X = torch.tensor([[[1.0, float("nan")], [3.0, 4.0]]])
    Z, missing, pad = normalize_features(X, n_ctx=2)
    assert Z.shape == missing.shape == (1, 2, 2)
    assert pad.shape == (1, 2)
    assert torch.isfinite(Z).all()
    assert missing[0, 0, 1] == 1.0  # the NaN cell is flagged
    assert Z[0, 0, 1] == 0.0  # and its value is neutralised
    assert not pad.any()  # neither column is entirely absent


def test_normalize_features_detects_padded_columns():
    X = torch.full((1, 4, 3), float("nan"))
    X[:, :, 0] = torch.tensor([1.0, 2.0, 3.0, 4.0])
    _Z, _missing, pad = normalize_features(X, n_ctx=3)
    assert pad.tolist() == [[False, True, True]]


def test_forward_shapes_and_class_masking():
    cfg = ModelConfig(max_features=10, max_classes=5, d_model=32, n_heads=2, n_layers=2, d_ff=64)
    model = FinancialTFM(cfg)
    B, N, F = 3, 12, 10
    X = torch.randn(B, N, F)
    y = torch.randint(0, 3, (B, N))
    n_classes = torch.tensor([2, 3, 5])
    logits = model(X, y, n_ctx=8, n_classes=n_classes)
    assert logits.shape == (B, N, 5)
    assert torch.isinf(logits[0, :, 2:]).all()  # task 0 has only 2 valid classes
    assert torch.isfinite(logits[2]).all()  # task 2 uses all 5 classes


def test_loss_decreases_after_a_few_steps():
    torch.manual_seed(0)
    cfg = ModelConfig(max_features=8, max_classes=4, d_model=32, n_heads=2, n_layers=2, d_ff=64)
    model = FinancialTFM(cfg)
    prior_cfg = PriorConfig(max_features=8, max_classes=4, n_rows=32)
    rng = np.random.default_rng(0)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    for _ in range(30):
        batch = sample_batch(rng, prior_cfg, batch_size=8)
        loss = model.loss(batch.X, batch.y, batch.n_ctx, batch.n_classes)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert np.mean(losses[-5:]) < np.mean(losses[:5])


def test_save_and_load_roundtrip(tmp_path):
    cfg = ModelConfig(max_features=6, max_classes=3, d_model=16, n_heads=2, n_layers=1, d_ff=32)
    model = FinancialTFM(cfg)
    path = str(tmp_path / "ckpt.pt")
    model.save(path)
    loaded = FinancialTFM.load(path)
    X = torch.randn(1, 5, 6)
    y = torch.zeros(1, 5, dtype=torch.long)
    out1 = model.eval()(X, y, n_ctx=3, n_classes=torch.tensor([3]))
    out2 = loaded(X, y, n_ctx=3, n_classes=torch.tensor([3]))
    torch.testing.assert_close(out1, out2)


def test_predictions_are_invariant_to_column_permutation():
    """The property the architecture was rewritten for.

    A table has no canonical column order, so permuting features must not change any
    prediction. The previous flat-vector design failed this by construction.
    """
    torch.manual_seed(0)
    cfg = ModelConfig(max_features=8, max_classes=3, d_cell=16, d_model=32, n_heads=2,
                      n_col_layers=1, n_layers=2, d_ff=64)
    model = FinancialTFM(cfg).eval()
    X = torch.randn(2, 10, 8)
    X[0, 3, 5] = float("nan")  # keep a missing cell in the picture
    y = torch.randint(0, 3, (2, 10))
    perm = torch.randperm(8)

    with torch.no_grad():
        a = model(X, y, n_ctx=6, n_classes=torch.tensor([3, 3]))
        b = model(X[:, :, perm], y, n_ctx=6, n_classes=torch.tensor([3, 3]))
    torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5)


def test_padding_width_does_not_change_predictions():
    """A 5-feature table must score the same whether padded to 8 columns or 16."""
    torch.manual_seed(0)
    narrow = ModelConfig(max_features=8, max_classes=2, d_cell=16, d_model=32, n_heads=2,
                         n_col_layers=1, n_layers=2, d_ff=64)
    model = FinancialTFM(narrow).eval()
    real = torch.randn(1, 8, 5)
    X8 = torch.full((1, 8, 8), float("nan"))
    X8[:, :, :5] = real
    y = torch.randint(0, 2, (1, 8))
    with torch.no_grad():
        out8 = model(X8, y, n_ctx=5, n_classes=torch.tensor([2]))
    # same content, padding moved to the front instead of the back
    X8_shifted = torch.full((1, 8, 8), float("nan"))
    X8_shifted[:, :, 3:] = real
    with torch.no_grad():
        shifted = model(X8_shifted, y, n_ctx=5, n_classes=torch.tensor([2]))
    torch.testing.assert_close(out8, shifted, rtol=1e-4, atol=1e-5)


def test_query_rows_never_see_each_other():
    """Changing one query row must not move another query row's prediction."""
    torch.manual_seed(0)
    cfg = ModelConfig(max_features=6, max_classes=2, d_cell=16, d_model=32, n_heads=2,
                      n_col_layers=1, n_layers=2, d_ff=64)
    model = FinancialTFM(cfg).eval()
    X = torch.randn(1, 10, 6)
    y = torch.randint(0, 2, (1, 10))
    with torch.no_grad():
        before = model(X, y, n_ctx=6, n_classes=torch.tensor([2]))
        X2 = X.clone()
        X2[0, 9] = torch.randn(6)  # perturb the last query row only
        after = model(X2, y, n_ctx=6, n_classes=torch.tensor([2]))
    torch.testing.assert_close(before[:, 6:9], after[:, 6:9], rtol=1e-4, atol=1e-5)
