"""Low-rank adapter over the frozen base (task 16.2, provenance P5).

Pinned: the base is bit-identical after adapter training and only adapter parameters carry
gradients; a zero-initialised adapter changes no output; a few CPU steps lower the training
loss on a synthetic task; the adapter round-trips through its own file with a provenance
record that says real data was used; and it can never be merged into the base.
"""

import numpy as np
import pytest
import torch

from fintfm.inference.classifier import FinancialTFMClassifier
from fintfm.modeling.adapter import (
    AdapterConfig,
    AdapterError,
    adapter_parameters,
    attach_adapter,
    base_weights_sha256,
    merge_adapter_into_base,
    read_adapter,
)
from fintfm.modeling.model import FinancialTFM, ModelConfig


def _model() -> FinancialTFM:
    torch.manual_seed(0)
    model = FinancialTFM(
        ModelConfig(
            max_features=8,
            max_classes=2,
            d_cell=16,
            d_model=32,
            n_heads=2,
            n_col_layers=1,
            n_layers=2,
            d_ff=64,
        )
    ).eval()
    model.trained_objectives = ("classification",)
    return model


def _data(n: int = 240, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6)).astype(np.float32)
    y = (X[:, 0] - X[:, 1] > 0).astype(np.int64)
    return X, y, np.repeat(np.arange(n // 3), 3)


def _cfg(**kw) -> AdapterConfig:
    base = {
        "rank": 2,
        "targets": ("attention", "head"),
        "steps": 3,
        "batch": 2,
        "query": 32,
        "context": 64,
    }
    return AdapterConfig(**{**base, **kw})


def _state(model: FinancialTFM) -> dict[str, torch.Tensor]:
    return {k: v.clone() for k, v in model.state_dict().items()}


def test_base_is_bit_identical_after_adapter_training():
    model = _model()
    before, h = _state(model), base_weights_sha256(model)
    X, y, g = _data()
    clf = FinancialTFMClassifier(model, max_context=64).fit_adapter(X, y, g, adapter=_cfg())
    after = _state(model)
    assert before.keys() == after.keys()
    assert all(torch.equal(before[k], after[k]) for k in before)
    assert base_weights_sha256(model) == h == clf.adapter_provenance_["base_weights_sha256"]
    assert clf.model_ is not model


def test_only_adapter_parameters_have_grad():
    model = _model()
    attach_adapter(model, _cfg())
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable and trainable == set(adapter_parameters(model))
    assert all(not p.requires_grad for n, p in model.named_parameters() if n not in trainable)


def test_training_leaves_gradients_on_adapter_only():
    X, y, g = _data()
    clf = FinancialTFMClassifier(_model(), max_context=64).fit_adapter(X, y, g, adapter=_cfg())
    assert all(p.grad is None for p in clf.model.parameters())
    # B starts at zero and the loss reaches it first, so a trained adapter has moved
    assert any(
        float(p.detach().abs().max()) > 0
        for n, p in adapter_parameters(clf.model_).items()
        if n.endswith("B")
    )


def test_zero_init_adapter_gives_the_base_outputs():
    model = _model()
    X, y, _ = _data()
    Xq = np.random.default_rng(1).normal(size=(20, 6)).astype(np.float32)
    plain = FinancialTFMClassifier(model, max_context=64).fit(X, y).predict_proba(Xq)
    adapted = FinancialTFMClassifier(model, max_context=64).fit(X, y)
    import copy

    m = copy.deepcopy(model)
    attach_adapter(m, _cfg(targets=("attention", "linear", "head")))
    adapted.model_ = m.eval()
    assert np.array_equal(adapted.predict_proba(Xq), plain)


def test_a_few_cpu_steps_lower_the_training_loss():
    X, y, g = _data(n=300)
    cfg = _cfg(
        rank=4,
        steps=40,
        lr=2e-2,
        batch=4,
        query=64,
        context=96,
        targets=("attention", "linear", "head"),
    )
    clf = FinancialTFMClassifier(_model(), max_context=96).fit_adapter(X, y, g, adapter=cfg)
    losses = clf.adapter_losses_
    assert len(losses) == 40
    assert np.mean(losses[-8:]) < np.mean(losses[:8])


def test_save_load_round_trip_and_provenance(tmp_path):
    model = _model()
    X, y, g = _data()
    Xq = np.random.default_rng(2).normal(size=(20, 6)).astype(np.float32)
    ckpt = tmp_path / "base.pt"
    model.save(str(ckpt), trained_objectives=("classification",))
    clf = FinancialTFMClassifier(model, max_context=64, random_state=1).fit_adapter(
        X, y, g, adapter=_cfg(), base_checkpoint=str(ckpt)
    )
    path = tmp_path / "lender.adapter.pt"
    clf.save_adapter(str(path))
    prov = read_adapter(path)["provenance"]
    for key in (
        "fitted_on_real_data",
        "base_weights_sha256",
        "base_checkpoint_sha256",
        "rank",
        "targets",
        "rows_supplied",
        "task_rows_seen",
        "merged_into_base",
    ):
        assert key in prov
    assert prov["fitted_on_real_data"] is True
    assert prov["rank"] == 2 and prov["rows_supplied"] == len(y)
    assert len(prov["base_checkpoint_sha256"]) == 64
    assert prov["merged_into_base"] is False

    fresh = FinancialTFMClassifier(FinancialTFM.load(str(ckpt)), max_context=64, random_state=1)
    fresh.fit(X, y).load_adapter(str(path))
    np.testing.assert_array_equal(fresh.predict_proba(Xq), clf.predict_proba(Xq))
    assert fresh.adapter_provenance_["fitted_on_real_data"] is True


def test_adapter_for_another_base_is_refused(tmp_path):
    X, y, g = _data()
    clf = FinancialTFMClassifier(_model(), max_context=64).fit_adapter(X, y, g, adapter=_cfg())
    path = tmp_path / "a.pt"
    clf.save_adapter(str(path))
    torch.manual_seed(5)
    other = FinancialTFM(_model().cfg).eval()
    other.trained_objectives = ("classification",)
    with pytest.raises(AdapterError, match="different base"):
        FinancialTFMClassifier(other, max_context=64).fit(X, y).load_adapter(str(path))


def test_merging_into_the_base_is_refused(tmp_path):
    X, y, g = _data()
    clf = FinancialTFMClassifier(_model(), max_context=64).fit_adapter(X, y, g, adapter=_cfg())
    with pytest.raises(AdapterError, match="never merged"):
        clf.merge_adapter()
    with pytest.raises(AdapterError, match="never merged"):
        merge_adapter_into_base(clf.model_)
    with pytest.raises(RuntimeError, match="adapter"):
        clf.model_.save(str(tmp_path / "x.pt"))


def test_adapter_and_finetune_are_exclusive_and_bad_configs_fail():
    X, y, g = _data()
    with pytest.raises(ValueError, match="both tune"):
        FinancialTFMClassifier(_model(), finetune_steps=1).fit_adapter(X, y, g, adapter=_cfg())
    with pytest.raises(ValueError):
        AdapterConfig(rank=0)
    with pytest.raises(ValueError):
        AdapterConfig(targets=("nope",))
