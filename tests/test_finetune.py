"""Dataset-specific fine-tuning in `FinancialTFMClassifier.fit` (§163).

Three properties carry the experiment's validity, so each is pinned: off by default changes
nothing; on, it never touches the model the caller passed in (the V4FinBench protocol reuses one
model across folds, so a mutation would leak fold k's training into fold k+1); and a task's
query companies never appear in its own context.
"""

import numpy as np
import torch

from fintfm.inference.classifier import FinancialTFMClassifier, _finetune_rows
from fintfm.modeling.model import FinancialTFM, ModelConfig


def _model() -> FinancialTFM:
    torch.manual_seed(0)
    model = FinancialTFM(
        ModelConfig(max_features=8, max_classes=2, d_cell=16, d_model=32, n_heads=2,
                    n_col_layers=1, n_layers=2, d_ff=64)
    ).eval()
    model.trained_objectives = ("classification",)
    return model


def _data(n: int = 300, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6)).astype(np.float32)
    y = (X[:, 0] - X[:, 1] > 0).astype(np.int64)
    groups = np.repeat(np.arange(n // 3), 3)  # three rows per "company"
    return X, y, groups


def _state(model: FinancialTFM) -> dict[str, torch.Tensor]:
    return {k: v.clone() for k, v in model.state_dict().items()}


def test_off_by_default_is_byte_identical():
    X, y, g = _data()
    Xq = np.random.default_rng(1).normal(size=(30, 6)).astype(np.float32)
    a = FinancialTFMClassifier(_model(), max_context=100).fit(X, y).predict_proba(Xq)
    b = FinancialTFMClassifier(_model(), max_context=100, finetune_steps=0).fit(
        X, y, groups=g
    ).predict_proba(Xq)
    assert np.array_equal(a, b)


def test_finetuning_copies_the_model_and_changes_the_copy():
    X, y, g = _data()
    model = _model()
    before = _state(model)
    clf = FinancialTFMClassifier(
        model, max_context=100, finetune_steps=3, finetune_lr=1e-3, finetune_batch=2,
        finetune_query=32,
    ).fit(X, y, groups=g)
    for k, v in model.state_dict().items():
        assert torch.equal(v, before[k]), f"caller's model was modified at {k}"
    assert clf.model is not model
    assert any(not torch.equal(v, before[k]) for k, v in clf.model.state_dict().items())
    assert len(clf.finetune_losses_) == 3
    assert not clf.model.training  # returned to eval mode for prediction


def test_query_groups_never_appear_in_the_context():
    _, _, g = _data(n=600)
    uniq = np.unique(g)
    rng = np.random.default_rng(0)
    for _ in range(20):
        ci, qi = _finetune_rows(rng, g, uniq, n_q_groups=60, n_ctx=200, n_q=64)
        assert len(ci) == 200 and len(qi) == 64
        assert not set(g[ci]) & set(g[qi])


def test_finetuning_reduces_the_loss_on_a_learnable_task():
    X, y, g = _data(n=600)
    clf = FinancialTFMClassifier(
        _model(), max_context=128, finetune_steps=60, finetune_lr=3e-3, finetune_batch=4,
        finetune_query=64, random_state=0,
    ).fit(X, y, groups=g)
    losses = clf.finetune_losses_
    assert np.mean(losses[-10:]) < np.mean(losses[:10])
