"""Regression tests for openspec task 49.9's smaller defects (a), (b), (e), (f), (g), (h).

Each test fails on the code before its fix and passes after, except (h) where the code was
already fixed and the test pins the wiring (see its docstring).
"""

import numpy as np
import pytest
import torch

from fintfm.modeling.model import FinancialTFM, ModelConfig


def _tiny_model(**kw) -> FinancialTFM:
    cfg = ModelConfig(
        max_features=6, max_classes=3, d_cell=16, d_model=32, n_heads=2, n_col_layers=1,
        n_layers=1, d_ff=32, **kw,
    )
    return FinancialTFM(cfg)


# --- (a) CUDA generator states in the training state ------------------------------------


def _save_state(path, model):
    from fintfm.modeling.train import _NullSchedule, _save_training_state

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    _save_training_state(
        str(path), model, opt, _NullSchedule(1e-3), np.random.default_rng(0),
        step=1, objectives={"classification"}, total_steps=2,
    )


def test_training_state_saves_and_restores_cuda_generators(tmp_path, monkeypatch):
    """The saved state carries every CUDA generator and a resume hands them back."""
    from fintfm.modeling.train import _restore_cuda_rng

    fake = [torch.arange(4, dtype=torch.uint8), torch.arange(4, dtype=torch.uint8) + 1]
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: fake)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    restored: list = []
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", lambda s: restored.append(s))

    _save_state(tmp_path / "s.state", _tiny_model())
    state = torch.load(tmp_path / "s.state", weights_only=False)
    assert len(state["cuda_rng"]) == 2

    _restore_cuda_rng(state["cuda_rng"])
    assert len(restored) == 1
    assert all(torch.equal(a, b) for a, b in zip(restored[0], fake, strict=True))


def test_state_without_cuda_rng_still_loads(tmp_path, monkeypatch):
    """Older state files (no key) and CUDA-less machines restore nothing and do not fail."""
    from fintfm.modeling.train import _restore_cuda_rng

    _save_state(tmp_path / "s.state", _tiny_model())
    state = torch.load(tmp_path / "s.state", weights_only=False)
    assert state["cuda_rng"] is None or not torch.cuda.is_available()

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    called: list = []
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", lambda s: called.append(s))
    _restore_cuda_rng(state.get("nonexistent_key"))
    _restore_cuda_rng(None)
    assert called == []


# --- (b) seed counts per cell ------------------------------------------------------------


def test_sweeps_record_how_many_seeds_each_cell_scored(tmp_path, monkeypatch):
    """A single-class test split is still skipped, but the cell's scored count shows it."""
    from fintfm.experiments import capability

    real = capability.make_probe

    def probe(*args, **kwargs):
        Xtr, ytr, Xte, yte = real(*args, **kwargs)
        if kwargs.get("seed") == 1:
            yte = np.zeros_like(yte)
        return Xtr, ytr, Xte, yte

    monkeypatch.setattr(capability, "make_probe", probe)
    ckpt = tmp_path / "m.pt"
    FinancialTFM(ModelConfig(max_features=8, d_model=16, d_cell=8, n_layers=1,
                             n_col_layers=1, max_classes=2)).save(
        str(ckpt), trained_objectives=("classification",))

    fs = capability.feature_sweep({"p": str(ckpt)}, widths=(5,), seeds=(0, 1), max_context=100)
    assert fs["n_seeds_scored"] == [1]
    br = capability.base_rate_sweep({"p": str(ckpt)}, rates=(0.3,), n_features=5,
                                    seeds=(0, 1), max_context=100)
    assert br["n_seeds_scored"] == [1]


# --- (e) no fitted attributes before fit -------------------------------------------------


def _fitted_style(obj) -> list[str]:
    return [k for k in vars(obj) if k.endswith("_") and not k.startswith("_")]


def test_regressors_have_no_trailing_underscore_attributes_before_fit():
    """scikit-learn convention: ``name_`` attributes exist only after ``fit``."""
    from fintfm.inference.quantile_regressor import FinancialTFMQuantileRegressor
    from fintfm.inference.regressor import FinancialTFMRegressor

    binned = FinancialTFMRegressor(_tiny_model(), n_bins=3)
    quant = FinancialTFMQuantileRegressor(_tiny_model(head_type="quantile", n_quantiles=9))
    assert _fitted_style(binned) == []
    assert _fitted_style(quant) == []
    for est in (binned, quant):
        with pytest.raises(RuntimeError, match="fit must be called"):
            est.predict(np.zeros((2, 4)))

    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(40, 4)), rng.normal(size=40)
    binned.fit(X, y)
    quant.fit(X, y)
    assert {"binner_", "classifier_"} <= set(vars(binned))
    assert {"scaler_", "taus_"} <= set(vars(quant))
    assert binned.predict(X[:3]).shape == (3,)
    assert quant.predict(X[:3]).shape == (3,)


# --- (f) feature_chunk is per estimator --------------------------------------------------


def test_classifier_fit_does_not_set_feature_chunk_on_the_shared_model():
    """Two estimators on one model keep their own chunking, and the model is untouched."""
    from fintfm.inference.classifier import FinancialTFMClassifier

    model = _tiny_model()
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(40, 4)), (rng.random(40) > 0.5).astype(int)
    a = FinancialTFMClassifier(model, feature_chunk=2, max_context=30).fit(X, y)
    b = FinancialTFMClassifier(model, feature_chunk=None, max_context=30).fit(X, y)
    assert model.feature_chunk is None
    assert a.model_.feature_chunk == 2
    assert b.model_.feature_chunk is None
    assert a.model_ is not b.model_
    # chunking is an identity, so sharing parameters changes no output
    np.testing.assert_allclose(a.predict_proba(X), b.predict_proba(X), atol=1e-5)


# --- (g) prior_score's TypeError handling ------------------------------------------------


def test_prior_score_does_not_swallow_a_typeerror_raised_inside_the_sampler():
    """Only a signature without ``max_classes`` selects the plain call."""
    from fintfm.experiments.prior_score import score_prior
    from fintfm.prior.trivial import sample_trivial_task

    def buggy(rng, n_rows, max_classes=2):
        raise TypeError("a real bug inside the sampler")

    with pytest.raises(TypeError, match="real bug"):
        score_prior(buggy, n_tasks=1, n_rows=100)

    def no_max_classes(rng, n_rows):
        return sample_trivial_task(rng, n_rows)

    out = score_prior(no_max_classes, n_tasks=2, n_rows=200)
    assert "n_scored" in out


# --- (h) schedule-free warmup ------------------------------------------------------------


def test_schedulefree_receives_the_configured_warmup():
    """``warmup_steps`` reaches ``AdamWScheduleFree`` (wired in c8c3027; §146 addendum).

    Not a before/after test: the code was already fixed when task 49.9 was written, so the
    only stale part was :class:`_NullSchedule`'s docstring. This pins the wiring.
    """
    pytest.importorskip("schedulefree")
    from fintfm.modeling.train import TrainConfig, _build_optimizer

    opt, _ = _build_optimizer(
        _tiny_model(), TrainConfig(optimizer="schedulefree", warmup_steps=7)
    )
    assert opt.param_groups[0]["warmup_steps"] == 7
