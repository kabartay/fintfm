import numpy as np
import torch

from fintfm.model import FinancialTFM, ModelConfig, normalize_features
from fintfm.prior import PriorConfig
from fintfm.prior.mixture import sample_batch


def test_normalize_features_masks_missing():
    X = torch.tensor([[[1.0, float("nan")], [3.0, 4.0]]])
    out = normalize_features(X, n_ctx=2)
    assert out.shape == (1, 2, 4)
    assert torch.isfinite(out).all()
    assert out[0, 0, 3] == 1.0  # missingness indicator for the NaN cell


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
