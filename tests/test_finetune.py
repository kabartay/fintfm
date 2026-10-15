"""Dataset-specific fine-tuning in `FinancialTFMClassifier.fit` (§163).

Three properties carry the experiment's validity, so each is pinned: off by default changes
nothing; on, it never touches the model the caller passed in (the V4FinBench protocol reuses one
model across folds, so a mutation would leak fold k's training into fold k+1); and a task's
query companies never appear in its own context.
"""

import numpy as np
import pytest
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
    assert clf.model is model and clf.model_ is not model
    assert any(not torch.equal(v, before[k]) for k, v in clf.model_.state_dict().items())
    assert len(clf.finetune_losses_) == 3
    assert not clf.model_.training  # returned to eval mode for prediction


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
    from unittest import mock

    n, n_groups = 600_000, 150_000
    raw = np.array([f"company-{i:06d}" for i in range(n_groups)])[
        np.random.default_rng(0).integers(0, n_groups, size=n)
    ]
    uniq, codes = np.unique(raw, return_inverse=True)
    rng = np.random.default_rng(1)
    t = time.perf_counter()
    # The regression was np.isin over the ids, so the helper must not call it at all; that is
    # a check a loaded machine cannot make flaky. The wall-clock bound stays only as a backstop
    # against a gross regression (the original took hours), loose enough for a busy machine
    # (audit AU1.12; it was 2 s).
    with mock.patch.object(np, "isin", side_effect=AssertionError("np.isin is O(n x k) here")):
        ci, qi, _ = _finetune_rows(rng, codes, uniq, n_q_groups=45_000, n_ctx=2000, n_q=512)
    assert time.perf_counter() - t < 60.0
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
        return torch.cat([p.detach().flatten() for p in clf.model_.parameters()])

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


@pytest.mark.parametrize(
    "device",
    [
        pytest.param(
            "mps",
            marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no MPS"),
        ),
        pytest.param(
            "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
        ),
    ],
)
def test_finetuning_leaves_the_callers_device_rng_alone(device):
    """AU1.8: fork_rng(devices=[]) restored the CPU generator only, while manual_seed reseeded
    the accelerator's too. Fine-tuning itself runs on CPU here: the reseed reached the
    accelerator generator whatever device the model was on."""
    mod = torch.mps if device == "mps" else torch.cuda
    X, y, g = _data()
    clf = FinancialTFMClassifier(_model(), max_context=100, finetune_steps=1, finetune_query=32)
    mod.manual_seed(123)
    before = mod.get_rng_state().clone()
    clf.fit(X, y, groups=g)
    assert torch.equal(mod.get_rng_state(), before)


def _weights(model: FinancialTFM) -> torch.Tensor:
    return torch.cat([p.detach().flatten() for p in model.parameters()])


def test_refitting_fine_tunes_from_the_model_passed_in_not_the_tuned_copy():
    """AU1.6: fit replaced self.model with the tuned copy, so a second fit (or a clone of a
    fitted estimator) tuned the tuned weights again and could carry one fold into the next."""
    from sklearn.base import clone

    X, y, g = _data()
    model = _model()
    kw = {"max_context": 100, "finetune_steps": 2, "finetune_lr": 1e-3, "finetune_query": 32}
    clf = FinancialTFMClassifier(model, **kw).fit(X, y, groups=g)
    once = _weights(clf.model_)
    assert clf.model is model
    clf.fit(X, y, groups=g)
    assert torch.equal(_weights(clf.model_), once)
    cloned = clone(clf)
    assert torch.equal(_weights(cloned.model), _weights(model))
    assert torch.equal(_weights(cloned.fit(X, y, groups=g).model_), once)


def test_finetune_context_never_exceeds_the_context_pool(monkeypatch):
    """AU1.7(a): a context larger than the rows outside the held-out groups was filled with
    duplicates, at a context size inference never uses on so small a table."""
    import fintfm.inference.classifier as mod

    drawn = []
    real = mod._finetune_rows

    def spy(*args, **kwargs):
        ci, qi, w = real(*args, **kwargs)
        drawn.append(ci)
        return ci, qi, w

    monkeypatch.setattr(mod, "_finetune_rows", spy)
    X, y, g = _data(n=60)  # 20 companies of three rows
    FinancialTFMClassifier(
        _model(), max_context=2000, finetune_steps=2, finetune_query=8,
    ).fit(X, y, groups=g)
    assert drawn
    for ci in drawn:
        assert len(np.unique(ci)) == len(ci), "context rows were duplicated"


def test_finetuning_needs_two_groups_and_always_leaves_a_context():
    """AU1.7(b): one group, or two with a large query fraction, emptied the context pool."""
    import pytest

    X, y, _ = _data(n=60)
    kw = {"max_context": 30, "finetune_steps": 1, "finetune_query": 8}
    with pytest.raises(ValueError, match="at least two groups"):
        FinancialTFMClassifier(_model(), **kw).fit(X, y, groups=np.zeros(60, dtype=int))
    two = np.repeat([0, 1], 30)
    clf = FinancialTFMClassifier(_model(), finetune_query_group_frac=0.8, **kw)
    assert len(clf.fit(X, y, groups=two).finetune_losses_) == 1


def test_query_positive_fraction_must_lie_strictly_between_zero_and_one():
    """AU1.7(c): at 1 the negative share divided by zero, above 1 its size went negative."""
    import pytest

    X, y, g = _data(n=60)
    for frac in (0.0, 1.0, 1.5):
        clf = FinancialTFMClassifier(
            _model(), max_context=30, finetune_steps=1, finetune_query=8,
            finetune_query_pos_frac=frac,
        )
        with pytest.raises(ValueError, match="finetune_query_pos_frac"):
            clf.fit(X, y, groups=g)
    # near 1, rounding gave every query to the positives and the same division by zero
    _, y_big, g_big = _data(n=600)
    _, qi, w = _finetune_rows(
        np.random.default_rng(0), g_big, np.unique(g_big), n_q_groups=60, n_ctx=50, n_q=32,
        y=y_big, pos_frac=0.99,
    )
    assert np.isfinite(w).all() and set(y_big[qi]) == {0, 1}


def test_groups_must_match_the_rows():
    """AU1.7(d): a groups array of the wrong length was used without a check."""
    import pytest

    X, y, g = _data(n=60)
    clf = FinancialTFMClassifier(_model(), max_context=30, finetune_steps=1, finetune_query=8)
    with pytest.raises(ValueError, match="groups"):
        clf.fit(X, y, groups=g[:-3])


def test_a_zero_finetune_context_is_an_error_not_max_context():
    """AU1.7(e): finetune_context=0 silently meant max_context."""
    import pytest

    X, y, g = _data(n=60)
    clf = FinancialTFMClassifier(
        _model(), max_context=30, finetune_steps=1, finetune_query=8, finetune_context=0,
    )
    with pytest.raises(ValueError, match="finetune_context"):
        clf.fit(X, y, groups=g)


def test_finetuning_warns_when_inference_reads_another_context_distribution():
    """AU1.9: fine-tuning always draws a uniform context, so a non-uniform strategy or the
    label-swapped ensemble shows the tuned weights a distribution they were never tuned on."""
    import warnings

    X, y, g = _data(n=60)
    kw = {"max_context": 30, "finetune_steps": 1, "finetune_query": 8}
    for extra in ({"context_strategy": "balanced"}, {"ensemble_label_swap": True}):
        with pytest.warns(UserWarning, match="uniform"):
            FinancialTFMClassifier(_model(), **kw, **extra).fit(X, y, groups=g)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        FinancialTFMClassifier(_model(), **kw).fit(X, y, groups=g)
        FinancialTFMClassifier(_model(), max_context=30, context_strategy="balanced").fit(X, y)


def test_gradient_clipping_is_opt_in(monkeypatch):
    """AU1.11: pretraining clips at 1.0 and fine-tuning did not clip at all. Off by default,
    because clipping changes the tuned weights and §163/§164 were measured without it."""
    calls = []
    real = torch.nn.utils.clip_grad_norm_

    def spy(parameters, max_norm, *args, **kwargs):
        calls.append(max_norm)
        return real(parameters, max_norm, *args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", spy)
    X, y, g = _data()
    kw = {"max_context": 100, "finetune_steps": 2, "finetune_query": 32}
    FinancialTFMClassifier(_model(), **kw).fit(X, y, groups=g)
    assert calls == []
    FinancialTFMClassifier(_model(), finetune_clip=1.0, **kw).fit(X, y, groups=g)
    assert calls == [1.0, 1.0]


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no MPS")
def test_mps_cache_is_released_after_finetuning(monkeypatch):
    """AU1.11: only the CUDA allocator cache was emptied before prediction."""
    calls = []
    monkeypatch.setattr(torch.mps, "empty_cache", lambda: calls.append(True))
    X, y, g = _data()
    FinancialTFMClassifier(
        _model(), device="mps", max_context=100, finetune_steps=1, finetune_query=32,
    ).fit(X, y, groups=g)
    assert calls


def test_a_failed_finetune_leaves_no_model_in_training_mode(monkeypatch):
    """AU1.11: an exception mid-fine-tuning left the model in training mode."""

    def fail(*args, **kwargs):
        raise RuntimeError("loss failed")

    monkeypatch.setattr(FinancialTFM, "loss", fail)
    X, y, g = _data()
    clf = FinancialTFMClassifier(_model(), max_context=100, finetune_steps=1, finetune_query=32)
    with pytest.raises(RuntimeError, match="loss failed"):
        clf.fit(X, y, groups=g)
    assert not clf.model.training
    assert not clf.model_.training


def test_finetuning_leaves_no_gradients_behind():
    X, y, g = _data()
    clf = FinancialTFMClassifier(
        _model(), max_context=100, finetune_steps=2, finetune_query=32,
    ).fit(X, y, groups=g)
    assert all(p.grad is None for p in clf.model_.parameters())
