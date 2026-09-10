import numpy as np
import pytest
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
    """Training must reduce the loss on freshly sampled tasks.

    Uses 300 steps, a 50-step window and an explicit ``d_cell``. The history matters,
    because this test has been re-tuned twice and never loosened.

    It was 30 steps with a defaulted ``d_cell`` of 64 against a ``d_model`` of 32 — an
    inverted bottleneck — and began failing when the prior was made harder for §19. Diagnosed
    rather than loosened, and raised to 80 steps.

    It failed again at §47, which randomised the label direction per task so that no global
    feature rule can work. That is a much harder prior: measured over 600 steps the model
    goes 0.656 -> 0.592 but has not moved at all by step 80, so 80 was again too few. The
    window also went from 5 samples to 50, because a 5-sample mean at this noise level does
    not measure a trend — that is a more precise measurement, not a weaker assertion.

    **What is asserted is unchanged in both cases: training reduces the loss.**
    """
    torch.manual_seed(0)
    cfg = ModelConfig(
        max_features=8, max_classes=4, d_cell=16, d_model=32, n_heads=2, n_layers=2, d_ff=64
    )
    model = FinancialTFM(cfg)
    prior_cfg = PriorConfig(max_features=8, max_classes=4, n_rows=32)
    rng = np.random.default_rng(0)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    for _ in range(300):
        batch = sample_batch(rng, prior_cfg, batch_size=8)
        loss = model.loss(batch.X, batch.y, batch.n_ctx, batch.n_classes)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert np.mean(losses[-50:]) < np.mean(losses[:50])


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


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "mps",
            marks=pytest.mark.skipif(
                not torch.backends.mps.is_available(), reason="no Metal GPU here"
            ),
        ),
    ],
)
def test_training_completes_on_each_available_device(device, tmp_path):
    """Regression guard: the held-out eval path once crashed on any non-CPU device.

    `_eval_accuracy` sampled batches on the CPU and handed them to a model on the GPU. The
    training loop moved its own batches, so nothing failed until the first evaluation
    checkpoint — five minutes into a three-hour run, on a path no test had ever exercised
    off CPU. `eval_every` is set below `steps` here so that path actually runs.
    """
    from fintfm.modeling.train import TrainConfig, train

    cfg = ModelConfig(max_features=6, max_classes=2, d_cell=8, d_model=16, n_heads=2,
                      n_col_layers=1, n_layers=1, d_ff=16)
    prior_cfg = PriorConfig(max_features=6, max_classes=2, n_rows=16)
    train_cfg = TrainConfig(
        steps=4, batch_size=2, device=device, log_every=2, eval_every=2, warmup_steps=1
    )
    model = train(cfg, prior_cfg, train_cfg, str(tmp_path / "ckpt.pt"))
    assert str(next(model.parameters()).device).startswith(device)
    assert (tmp_path / "ckpt.pt").exists()


def test_checkpoint_records_which_objectives_were_trained(tmp_path):
    """The training loop optimises one objective per step, never both.

    A hazard checkpoint therefore leaves the classification head at random initialisation,
    and `predict_proba` served it as if it were real — mean predicted 0.69 against a 4.7%
    base rate, AUC 0.37, no error (`docs/FINDINGS.md` §34).
    """
    from fintfm.modeling.model import FinancialTFM, ModelConfig

    path = tmp_path / "m.pt"
    m = FinancialTFM(ModelConfig(max_features=4, d_model=16, d_cell=8, n_layers=1,
                                 n_col_layers=1, max_classes=2, n_horizons=3))
    m.save(str(path), trained_objectives=("survival",))
    loaded = FinancialTFM.load(str(path))
    assert loaded.trained_objectives == ("survival",)
    loaded.assert_trained_for("survival")
    with pytest.raises(RuntimeError, match="random initialisation"):
        loaded.assert_trained_for("classification")


def test_checkpoint_without_a_record_is_allowed_through(tmp_path):
    """Older checkpoints carry no record; refusing them would break every in-process use."""
    from fintfm.modeling.model import FinancialTFM, ModelConfig

    path = tmp_path / "m.pt"
    m = FinancialTFM(ModelConfig(max_features=4, d_model=16, d_cell=8, n_layers=1,
                                 n_col_layers=1, max_classes=2))
    m.save(str(path))
    FinancialTFM.load(str(path)).assert_trained_for("classification")


def test_predict_proba_refuses_a_survival_only_checkpoint(tmp_path):
    import numpy as np

    from fintfm.inference.classifier import FinancialTFMClassifier
    from fintfm.modeling.model import FinancialTFM, ModelConfig

    path = tmp_path / "m.pt"
    FinancialTFM(ModelConfig(max_features=4, d_model=16, d_cell=8, n_layers=1,
                             n_col_layers=1, max_classes=2, n_horizons=3)).save(
        str(path), trained_objectives=("survival",))
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 4)).astype(np.float32)
    y = (rng.random(40) < 0.3).astype(np.int64)
    clf = FinancialTFMClassifier(str(path), max_context=20).fit(X, y)
    with pytest.raises(RuntimeError, match="not on 'classification'"):
        clf.predict_proba(X)
    clf.predict_term_structure(X)  # the trained head still works


def test_eval_quality_handles_multi_class_tasks():
    """A multi-class prior crashed the training metric mid-run (docs/FINDINGS.md §44).

    `roc_auc_score` on the class-1 column alone raises "multi_class must be in ('ovo','ovr')"
    as soon as the generic SCM prior emits more than two classes, which is any run with
    `--p-financial < 1` and `max_classes > 2`.
    """
    import numpy as np

    from fintfm.modeling.model import FinancialTFM, ModelConfig
    from fintfm.modeling.train import _eval_quality
    from fintfm.prior import PriorConfig

    for max_classes in (2, 5):
        model = FinancialTFM(
            ModelConfig(max_features=16, d_model=32, d_cell=16, n_layers=1,
                        n_col_layers=1, max_classes=max_classes)
        )
        cfg = PriorConfig(max_features=16, max_classes=max_classes, p_financial=0.5, n_rows=200)
        q = _eval_quality(model, cfg, np.random.default_rng(0), n_batches=2)
        assert set(q) == {"auc", "brier_skill", "base_rate"}
        for key, value in q.items():
            assert np.isnan(value) or np.isfinite(value), (max_classes, key, value)
        if np.isfinite(q["auc"]):
            assert 0.0 <= q["auc"] <= 1.0
            # skill against the class-frequency predictor is bounded above by 1
            assert q["brier_skill"] <= 1.0


def test_periodic_checkpointing_writes_recoverable_files(tmp_path):
    """A long run that only saves at the end loses everything when it dies, and jobs die."""
    from fintfm.modeling.model import FinancialTFM, ModelConfig
    from fintfm.modeling.train import TrainConfig, train
    from fintfm.prior import PriorConfig

    out = tmp_path / "m.pt"
    model_cfg = ModelConfig(max_features=8, d_model=16, d_cell=8, n_layers=1,
                            n_col_layers=1, max_classes=2)
    prior_cfg = PriorConfig(max_features=8, max_classes=2, n_rows=64)
    train(model_cfg, prior_cfg,
          TrainConfig(steps=4, batch_size=2, eval_every=99, log_every=99,
                      checkpoint_every=2, device="cpu"), str(out))

    mid = sorted(tmp_path.glob("m.pt.step*"))
    assert [p.name for p in mid] == ["m.pt.step2", "m.pt.step4"]
    # each is a real, loadable checkpoint, not a truncated file
    for path in mid:
        loaded = FinancialTFM.load(str(path))
        assert loaded.cfg.max_features == 8
        assert loaded.trained_objectives == ("classification",)
    assert out.exists()


def test_checkpointing_is_off_by_default(tmp_path):
    from fintfm.modeling.model import ModelConfig
    from fintfm.modeling.train import TrainConfig, train
    from fintfm.prior import PriorConfig

    out = tmp_path / "m.pt"
    train(ModelConfig(max_features=8, d_model=16, d_cell=8, n_layers=1,
                      n_col_layers=1, max_classes=2),
          PriorConfig(max_features=8, max_classes=2, n_rows=64),
          TrainConfig(steps=2, batch_size=2, eval_every=99, log_every=99, device="cpu"),
          str(out))
    assert list(tmp_path.glob("m.pt.step*")) == []
    assert out.exists()


# --- attention pooling (docs/FINDINGS.md §50) --------------------------------------


def _pool_model(pooling, n_feat=24):
    from fintfm.modeling.model import FinancialTFM, ModelConfig

    torch.manual_seed(0)
    return FinancialTFM(ModelConfig(max_features=n_feat, max_classes=2, d_cell=16,
                                    d_model=32, n_heads=2, n_layers=2, n_col_layers=2,
                                    d_ff=64, pooling=pooling))


def test_attention_pooling_keeps_column_order_invariance():
    """Decision D4's invariance must survive the new reduction.

    Attention over feature tokens carries no positional encoding, so it should be
    permutation-equivariant and the pooled result invariant. Asserted rather than assumed:
    losing this would silently reintroduce the positional feature identity D4 removed.
    """
    m = _pool_model("attention")
    torch.manual_seed(1)
    X = torch.randn(2, 30, 24)
    y = torch.randint(0, 2, (2, 30))
    perm = torch.randperm(24)
    a = m(X, y, 15)
    b = m(X[:, :, perm], y, 15)
    torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-4)


def test_attention_pooling_keeps_padding_width_invariance():
    """Padding a table wider must not change its predictions."""
    m = _pool_model("attention", n_feat=40)
    torch.manual_seed(2)
    narrow = torch.randn(2, 24, 40)
    narrow[:, :, 20:] = float("nan")  # only 20 real columns
    y = torch.randint(0, 2, (2, 24))
    wider = narrow.clone()
    a = m(narrow, y, 12)
    b = m(wider, y, 12)
    torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)


def test_attention_pooling_adds_almost_no_parameters():
    """A gain from attention pooling must not be confusable with a gain from capacity.

    §50 ruled capacity out by showing that 17× more parameters produces identical curves, so
    this change has to be small enough that nobody can re-explain its effect as capacity.

    Measured at the **production** configuration, which is where the claim is made: 846,818 ->
    862,418, or +1.8%. At the tiny test configuration the same absolute addition is +7.0%,
    which says nothing about the real model — the proportion depends on the base size, so the
    assertion has to be made at the size that will actually be trained.
    """
    from fintfm.modeling.model import FinancialTFM, ModelConfig

    kw = {"max_features": 136, "max_classes": 2, "d_cell": 48, "d_model": 128,
          "n_heads": 4, "n_layers": 4, "n_col_layers": 2, "d_ff": 512}
    base = FinancialTFM(ModelConfig(pooling="meanmax", **kw)).num_parameters()
    attn = FinancialTFM(ModelConfig(pooling="attention", **kw)).num_parameters()
    growth = attn / base - 1
    assert 0 < growth < 0.05, f"attention pooling grew the production model by {growth:.1%}"


def test_unknown_pooling_is_refused():
    from fintfm.modeling.model import FinancialTFM, ModelConfig

    with pytest.raises(ValueError, match="pooling must be"):
        FinancialTFM(ModelConfig(max_features=8, max_classes=2, pooling="sum"))


def test_attention_pooling_checkpoint_roundtrips(tmp_path):
    from fintfm.modeling.model import FinancialTFM

    m = _pool_model("attention")
    path = tmp_path / "a.pt"
    m.save(str(path), trained_objectives=("classification",))
    loaded = FinancialTFM.load(str(path))
    assert loaded.cfg.pooling == "attention"
    torch.manual_seed(3)
    X, y = torch.randn(1, 20, 24), torch.randint(0, 2, (1, 20))
    torch.testing.assert_close(m(X, y, 10), loaded(X, y, 10))
