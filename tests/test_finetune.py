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
        ci, qi, w = _finetune_rows(rng, g, uniq, n_q_groups=60, n_ctx=200, n_q=64)
        assert w is None
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


def test_string_company_ids_work_and_stay_linear():
    """V4FinBench's company ids are strings; np.isin over them was O(n x k) and stalled §163."""
    import time

    n, n_groups = 600_000, 150_000
    raw = np.array([f"company-{i:06d}" for i in range(n_groups)])[
        np.random.default_rng(0).integers(0, n_groups, size=n)
    ]
    uniq, codes = np.unique(raw, return_inverse=True)
    rng = np.random.default_rng(1)
    t = time.perf_counter()
    ci, qi, _ = _finetune_rows(rng, codes, uniq, n_q_groups=45_000, n_ctx=2000, n_q=512)
    assert time.perf_counter() - t < 2.0
    assert not set(raw[ci]) & set(raw[qi])

    X, y, _ = _data()
    names = np.array([f"firm-{i // 3}" for i in range(len(y))])
    clf = FinancialTFMClassifier(
        _model(), max_context=100, finetune_steps=2, finetune_batch=2, finetune_query=32,
    ).fit(X, y, groups=names)
    assert len(clf.finetune_losses_) == 2


def test_uniform_weights_reproduce_the_plain_loss():
    model = _model().train()
    torch.manual_seed(0)
    X = torch.randn(2, 40, 8)
    y = torch.randint(0, 2, (2, 40))
    nc = torch.full((2,), 2)
    torch.manual_seed(1)
    plain = model.loss(X, y, 30, nc)
    torch.manual_seed(1)
    weighted = model.loss(X, y, 30, nc, weights=torch.ones(2, 10))
    assert torch.allclose(plain, weighted)


def test_enriched_queries_hit_their_share_and_weight_back_to_the_true_rate():
    rng = np.random.default_rng(0)
    n = 20_000
    y = (rng.random(n) < 0.004).astype(np.int64)
    groups = np.arange(n) // 4
    uniq = np.unique(groups)
    ci, qi, w = _finetune_rows(
        np.random.default_rng(1), groups, uniq, n_q_groups=1500, n_ctx=500, n_q=500,
        y=y, pos_frac=0.1,
    )
    assert y[qi].mean() == 0.1
    # weights are true/sampled per class, so the weighted positive share comes back to the
    # pool's true rate exactly: w_pos * 0.1 is that rate, and it is the population's order
    true_rate = w[y[qi] == 1][0] * 0.1
    weighted_rate = (w * y[qi]).sum() / w.sum()
    assert np.isclose(weighted_rate, true_rate, rtol=1e-5)
    assert 0.0 < true_rate < 0.02
    assert set(groups[ci]).isdisjoint(groups[qi])


def test_same_seed_reproduces_and_a_different_seed_differs():
    X, y, g = _data()
    kw = {"max_context": 100, "finetune_steps": 3, "finetune_lr": 1e-3, "finetune_query": 32}

    def weights(seed):
        clf = FinancialTFMClassifier(_model(), finetune_seed=seed, **kw).fit(X, y, groups=g)
        return torch.cat([p.detach().flatten() for p in clf.model.parameters()])

    assert torch.equal(weights(0), weights(0))
    assert not torch.equal(weights(0), weights(1))


def test_finetuning_leaves_the_callers_torch_rng_alone():
    X, y, g = _data()
    clf = FinancialTFMClassifier(_model(), max_context=100, finetune_steps=2, finetune_query=32)
    torch.manual_seed(123)  # after _model(), which seeds torch itself
    expected = torch.rand(3)
    torch.manual_seed(123)
    clf.fit(X, y, groups=g)
    assert torch.equal(torch.rand(3), expected)
