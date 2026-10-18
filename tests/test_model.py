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


def _ensemble_probs(model, X, y, n_ctx, n_classes=None, k=32):
    """Average predicted probabilities over ``k`` draws of the column identities.

    Column-order invariance is **distributional** once columns carry random identities
    (``docs/results/FINDINGS.md`` §54): any single draw assigns a particular tag to a particular
    column, so permuting columns changes that draw's answer. Averaging over draws recovers
    the invariance, at the 1/sqrt(k) rate measured when the mechanism was added -- 0.018 at
    k=1, 0.0056 at k=16, 0.0023 at k=64 in absolute probability.
    """
    with torch.no_grad():
        out = [
            torch.softmax(model(X, y, n_ctx, n_classes, column_id_seed=s).float(), dim=-1)
            for s in range(k)
        ]
    return torch.stack(out).mean(0)


@pytest.mark.parametrize("column_id_dim", [0, None])
def test_predictions_are_invariant_to_column_permutation(column_id_dim):
    """The property the architecture was rewritten for.

    A table has no canonical column order, so permuting features must not change any
    prediction. The previous flat-vector design failed this by construction.

    The guarantee differs by variant, and both halves are asserted here:

    * ``column_id_dim=0`` -- **exact**, pointwise. The encoder is symmetric over columns.
    * ``column_id_dim=None`` (the default) -- **distributional**, recovered by averaging over
      draws of the column identities. Exactness is what §54 shows is too strong: an encoder
      symmetric enough to guarantee it pointwise cannot represent "column j matters" at all,
      and was measured at chance on ``x_0 - x_1``. This is the trade, made deliberately.
    """
    torch.manual_seed(0)
    cfg = ModelConfig(max_features=8, max_classes=3, d_cell=16, d_model=32, n_heads=2,
                      n_col_layers=1, n_layers=2, d_ff=64, column_id_dim=column_id_dim)
    model = FinancialTFM(cfg).eval()
    X = torch.randn(2, 10, 8)
    X[0, 3, 5] = float("nan")  # keep a missing cell in the picture
    y = torch.randint(0, 3, (2, 10))
    perm = torch.randperm(8)
    ncl = torch.tensor([3, 3])

    if column_id_dim == 0:
        with torch.no_grad():
            a = model(X, y, n_ctx=6, n_classes=ncl)
            b = model(X[:, :, perm], y, n_ctx=6, n_classes=ncl)
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5)
    else:
        a = _ensemble_probs(model, X, y, 6, ncl)
        b = _ensemble_probs(model, X[:, :, perm], y, 6, ncl)
        assert (a - b).abs().max() < 0.02


@pytest.mark.parametrize("column_id_dim", [0, None])
def test_padding_width_does_not_change_predictions(column_id_dim):
    """Where the real columns sit among the padding must not change the answer.

    This is a special case of column-order invariance -- moving padding from the back to the
    front permutes the columns -- so it holds exactly without column identities and in
    distribution with them, exactly as the test above.
    """
    torch.manual_seed(0)
    narrow = ModelConfig(max_features=8, max_classes=2, d_cell=16, d_model=32, n_heads=2,
                         n_col_layers=1, n_layers=2, d_ff=64, column_id_dim=column_id_dim)
    model = FinancialTFM(narrow).eval()
    real = torch.randn(1, 8, 5)
    X8 = torch.full((1, 8, 8), float("nan"))
    X8[:, :, :5] = real
    y = torch.randint(0, 2, (1, 8))
    # same content, padding moved to the front instead of the back
    X8_shifted = torch.full((1, 8, 8), float("nan"))
    X8_shifted[:, :, 3:] = real
    ncl = torch.tensor([2])
    if column_id_dim == 0:
        with torch.no_grad():
            out8 = model(X8, y, n_ctx=5, n_classes=ncl)
            shifted = model(X8_shifted, y, n_ctx=5, n_classes=ncl)
        torch.testing.assert_close(out8, shifted, rtol=1e-4, atol=1e-5)
    else:
        a = _ensemble_probs(model, X8, y, 5, ncl)
        b = _ensemble_probs(model, X8_shifted, y, 5, ncl)
        assert (a - b).abs().max() < 0.02


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
    base rate, AUC 0.37, no error (`docs/results/FINDINGS.md` §34).
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
    """A multi-class prior crashed the training metric mid-run (docs/results/FINDINGS.md §44).

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
        assert set(q) == {"auc", "auc_per_task", "brier_skill", "base_rate"}
        # At this seed every metric is defined: both pools hold two or more classes and some
        # held-out task is binary. `isnan or isfinite` admitted the NaN this test exists to
        # catch (audit AU1.12); NaN is legitimate only when no task qualifies, which a fixed
        # seed rules out here.
        for key, value in q.items():
            assert np.isfinite(value), (max_classes, key, value)
        assert 0.0 <= q["auc"] <= 1.0
        # the per-task mean is the honest discrimination number (§57)
        assert 0.0 <= q["auc_per_task"] <= 1.0
        # skill against the class-frequency predictor is bounded above by 1
        assert q["brier_skill"] <= 1.0
        assert 0.0 < q["base_rate"] < 1.0


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


# --- attention pooling (docs/results/FINDINGS.md §50) --------------------------------------


def _pool_model(pooling, n_feat=24, column_id_dim=None):
    from fintfm.modeling.model import FinancialTFM, ModelConfig

    torch.manual_seed(0)
    return FinancialTFM(ModelConfig(max_features=n_feat, max_classes=2, d_cell=16,
                                    d_model=32, n_heads=2, n_layers=2, n_col_layers=2,
                                    d_ff=64, pooling=pooling,
                                    column_id_dim=column_id_dim))


def test_attention_pooling_keeps_column_order_invariance():
    """Decision D4's invariance must survive the new reduction.

    Attention over feature tokens carries no positional encoding, so it should be
    permutation-equivariant and the pooled result invariant. Asserted rather than assumed:
    losing this would silently reintroduce the positional feature identity D4 removed.

    Pinned to ``column_id_dim=0`` so this tests the *reduction* in isolation. The random
    column identities deliberately weaken this to a distributional guarantee, which the
    parametrized tests above cover; mixing the two here would test neither.
    """
    m = _pool_model("attention", column_id_dim=0)
    torch.manual_seed(1)
    X = torch.randn(2, 30, 24)
    y = torch.randint(0, 2, (2, 30))
    perm = torch.randperm(24)
    a = m(X, y, 15)
    b = m(X[:, :, perm], y, 15)
    torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-4)


def test_attention_pooling_keeps_padding_width_invariance():
    """Padding a table wider must not change its predictions."""
    m = _pool_model("attention", column_id_dim=0)
    torch.manual_seed(2)
    narrow = torch.randn(2, 24, 20)  # 20 real columns
    y = torch.randint(0, 2, (2, 24))
    # NaN-padded to the model's 24 features; it used to pad to 40, past the model's width,
    # and compare against an identical clone (DR1.5 refuses a table wider than the model)
    wider = torch.cat([narrow, torch.full((2, 24, 4), float("nan"))], dim=-1)
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
    torch.testing.assert_close(m.eval()(X, y, 10), loaded(X, y, 10))


@pytest.mark.parametrize("n_ctx", [32, 512])
def test_row_encoder_can_tell_its_own_columns_apart(n_ctx):
    """The regression test for ``docs/results/FINDINGS.md`` §54, the project's most expensive bug.

    Permuting the values *within each row independently* is not a column permutation: it
    destroys which value came from which column while leaving the multiset untouched. A row
    encoder that cannot notice is a symmetric function of that multiset, and such a model
    provably cannot represent a rule like ``x_0 - x_1`` -- measured at AUC 0.5097 against a
    ceiling of exactly 0.5.

    The failure was invisible for weeks because it *shrinks with context*: the only thing
    distinguishing columns in the broken encoder was sampling noise in the per-column
    normalisation statistics, which vanishes as the context grows. So this asserts at a large
    context too, where the broken version fell to 0.005.
    """
    from fintfm.modeling.model import FinancialTFM, ModelConfig

    base = {
        "max_features": 8, "max_classes": 2, "d_cell": 16, "d_model": 32, "n_heads": 2,
        "n_col_layers": 1, "n_layers": 2, "d_ff": 64,
    }
    n_rows, n_feat = n_ctx * 2, 6

    def sensitivity(model):
        g = torch.Generator().manual_seed(0)
        X = torch.randn(1, n_rows, n_feat, generator=g)
        shuffled = X.clone()
        for i in range(n_rows):
            shuffled[0, i] = X[0, i][torch.randperm(n_feat, generator=g)]
        with torch.no_grad():
            a = model.encode_rows(X, n_ctx, column_id_seed=0)
            b = model.encode_rows(shuffled, n_ctx, column_id_seed=0)
        return ((b - a).abs().mean() / a.abs().mean()).item()

    torch.manual_seed(0)
    with_ids = FinancialTFM(ModelConfig(**base)).eval()
    torch.manual_seed(0)
    without = FinancialTFM(ModelConfig(**base, column_id_dim=0)).eval()

    assert sensitivity(with_ids) > 0.05, (
        "the row encoder cannot distinguish its own columns; it is a symmetric function of "
        "the row's values and cannot learn any column-specific rule (FINDINGS §54)"
    )
    # and the pre-fix architecture is recorded as failing it, so the test is known to bite
    assert sensitivity(without) < sensitivity(with_ids)


def test_n_cell_blocks_zero_is_byte_identical_to_the_pre_39_1_architecture():
    """The regression guarantee task 39.1 depends on: n_cell_blocks=0 must change nothing.

    Every existing checkpoint was trained without this field. If it changed behaviour by
    default, every prior comparison in this project (§54-§77) would be silently invalidated.
    """
    from fintfm.modeling.model import FinancialTFM, ModelConfig

    base = {
        "max_features": 8, "max_classes": 3, "d_cell": 16, "d_model": 32, "n_heads": 2,
        "n_col_layers": 1, "n_layers": 2, "d_ff": 64,
    }
    X = torch.randn(2, 10, 8)
    y = torch.randint(0, 3, (2, 10))
    nc = torch.tensor([3, 3])

    torch.manual_seed(1)
    m_old = FinancialTFM(ModelConfig(**base)).eval()
    torch.manual_seed(1)
    m_new = FinancialTFM(ModelConfig(**base, n_cell_blocks=0)).eval()

    with torch.no_grad():
        a = m_old(X, y, 6, nc, column_id_seed=0)
        b = m_new(X, y, 6, nc, column_id_seed=0)
    torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("n_cell_blocks", [1, 2])
@pytest.mark.parametrize("cell_labels", [False, True])
def test_two_way_cell_attention_trains(n_cell_blocks, cell_labels):
    """Shapes, gradients and parameter coverage for task 39.1/39.2, before any GPU spend.

    Not a claim that this fixes anything (docs/results/FINDINGS.md §74, §76) -- only that the new
    path is mechanically sound: correct output shape, every parameter receives a finite
    gradient, nothing silently detached from the graph.
    """
    from fintfm.modeling.model import FinancialTFM, ModelConfig

    base = {
        "max_features": 8, "max_classes": 3, "d_cell": 16, "d_model": 32, "n_heads": 2,
        "n_col_layers": 1, "n_layers": 2, "d_ff": 64,
    }
    X = torch.randn(2, 10, 8)
    y = torch.randint(0, 3, (2, 10))
    nc = torch.tensor([3, 3])

    torch.manual_seed(2)
    model = FinancialTFM(
        ModelConfig(**base, n_cell_blocks=n_cell_blocks, cell_labels=cell_labels)
    )
    out = model(X, y, 6, nc)
    assert out.shape == (2, 10, 3)

    loss = model.loss(X, y, 6, nc)
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(p.grad).all(), f"{name} has a non-finite gradient"


def test_cell_labels_ignored_without_n_cell_blocks():
    """cell_labels=True with n_cell_blocks=0 must not silently build unused parameters."""
    from fintfm.modeling.model import FinancialTFM, ModelConfig

    base = {
        "max_features": 8, "max_classes": 3, "d_cell": 16, "d_model": 32, "n_heads": 2,
        "n_col_layers": 1, "n_layers": 2, "d_ff": 64,
    }
    model = FinancialTFM(ModelConfig(**base, n_cell_blocks=0, cell_labels=True))
    assert not hasattr(model, "cell_y_proj")
    assert not hasattr(model, "row_within_feature_encoder")


def test_pre_39_1_shaped_checkpoint_loads(tmp_path):
    """A checkpoint saved before n_cell_blocks/cell_labels existed must keep loading.

    Unlike column_id_dim (whose backward-compatible value is 0, not its dataclass default of
    None, so load() special-cases it), n_cell_blocks=0 and cell_labels=False ARE the correct
    defaults -- no special-casing needed. Asserted rather than assumed.
    """
    from fintfm.modeling.model import FinancialTFM, ModelConfig

    base = {
        "max_features": 8, "max_classes": 3, "d_cell": 16, "d_model": 32, "n_heads": 2,
        "n_col_layers": 1, "n_layers": 2, "d_ff": 64,
    }
    path = tmp_path / "ckpt.pt"
    FinancialTFM(ModelConfig(**base)).save(str(path), trained_objectives=("classification",))
    ckpt = torch.load(str(path), weights_only=True)
    del ckpt["config"]["n_cell_blocks"], ckpt["config"]["cell_labels"]
    torch.save(ckpt, str(path))

    model = FinancialTFM.load(str(path))
    assert model.cfg.n_cell_blocks == 0
    assert model.cfg.cell_labels is False


def test_mask_embedding_false_is_byte_identical_to_the_pre_48_10_architecture():
    """Every checkpoint trained before task 48.10 must keep producing the same output."""
    from fintfm.modeling.model import FinancialTFM, ModelConfig

    base = {
        "max_features": 8, "max_classes": 3, "d_cell": 16, "d_model": 32, "n_heads": 2,
        "n_col_layers": 1, "n_layers": 2, "d_ff": 64,
    }
    X = torch.randn(2, 10, 8)
    y = torch.randint(0, 3, (2, 10))
    nc = torch.tensor([3, 3])

    torch.manual_seed(1)
    m_old = FinancialTFM(ModelConfig(**base)).eval()
    torch.manual_seed(1)
    m_new = FinancialTFM(ModelConfig(**base, mask_embedding=False)).eval()

    with torch.no_grad():
        a = m_old(X, y, 6, nc, column_id_seed=0)
        b = m_new(X, y, 6, nc, column_id_seed=0)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert not hasattr(m_old, "mask_embed")
    assert not hasattr(m_old, "value_embed")


def test_mask_embedding_true_trains(n_cell_blocks=0):
    """Shapes, gradients and parameter coverage for task 48.10, before any GPU spend."""
    from fintfm.modeling.model import FinancialTFM, ModelConfig

    base = {
        "max_features": 8, "max_classes": 3, "d_cell": 16, "d_model": 32, "n_heads": 2,
        "n_col_layers": 1, "n_layers": 2, "d_ff": 64,
    }
    X = torch.randn(2, 10, 8)
    y = torch.randint(0, 3, (2, 10))
    nc = torch.tensor([3, 3])

    torch.manual_seed(2)
    model = FinancialTFM(ModelConfig(**base, mask_embedding=True))
    assert not hasattr(model, "cell_embed")
    out = model(X, y, 6, nc)
    assert out.shape == (2, 10, 3)

    loss = model.loss(X, y, 6, nc)
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(p.grad).all(), f"{name} has a non-finite gradient"


def test_mask_embedding_true_differs_from_the_joint_embedding():
    """The dedicated-embedding path must actually be a different computation, not a no-op
    that happens to share the same parameter count."""
    from fintfm.modeling.model import FinancialTFM, ModelConfig

    base = {
        "max_features": 8, "max_classes": 3, "d_cell": 16, "d_model": 32, "n_heads": 2,
        "n_col_layers": 1, "n_layers": 2, "d_ff": 64,
    }
    X = torch.randn(2, 10, 8)
    y = torch.randint(0, 3, (2, 10))
    nc = torch.tensor([3, 3])

    torch.manual_seed(3)
    m_joint = FinancialTFM(ModelConfig(**base, mask_embedding=False)).eval()
    torch.manual_seed(3)
    m_dedicated = FinancialTFM(ModelConfig(**base, mask_embedding=True)).eval()

    with torch.no_grad():
        a = m_joint(X, y, 6, nc, column_id_seed=0)
        b = m_dedicated(X, y, 6, nc, column_id_seed=0)
    assert not torch.allclose(a, b)


def test_pre_48_10_shaped_checkpoint_loads(tmp_path):
    """A checkpoint saved before mask_embedding existed must keep loading, defaulting to the
    joint-embedding architecture every such checkpoint actually trained with."""
    from fintfm.modeling.model import FinancialTFM, ModelConfig

    base = {
        "max_features": 8, "max_classes": 3, "d_cell": 16, "d_model": 32, "n_heads": 2,
        "n_col_layers": 1, "n_layers": 2, "d_ff": 64,
    }
    path = tmp_path / "ckpt.pt"
    FinancialTFM(ModelConfig(**base)).save(str(path), trained_objectives=("classification",))
    ckpt = torch.load(str(path), weights_only=True)
    del ckpt["config"]["mask_embedding"]
    torch.save(ckpt, str(path))

    model = FinancialTFM.load(str(path))
    assert model.cfg.mask_embedding is False
    assert hasattr(model, "cell_embed")
    X, y = torch.randn(1, 10, 8), torch.randint(0, 3, (1, 10))
    with torch.no_grad():
        model(X, y, 6, torch.tensor([3]), column_id_seed=0)  # must not raise


def test_head_type_rejects_an_unknown_value():
    """A typo in head_type must fail loudly, not fall through to the binned default."""
    from fintfm.modeling.model import ModelConfig

    with pytest.raises(ValueError, match="head_type"):
        ModelConfig(head_type="bin")


def test_head_type_quantile_rejects_n_horizons():
    """No survival-plus-quantile design exists; combining the two must not silently build a
    hazard head whose interaction with a continuous label was never designed for."""
    from fintfm.modeling.model import ModelConfig

    with pytest.raises(ValueError, match="n_horizons"):
        ModelConfig(head_type="quantile", n_horizons=6)


def test_head_type_binned_default_is_byte_identical_to_pre_48_3_architecture():
    """Every checkpoint trained before task 48.3 must keep producing the same output -- the
    default head_type is a string that did not exist in their saved config, so this also
    exercises the same path a pre-48.3 checkpoint takes through ModelConfig(**cfg)."""
    from fintfm.modeling.model import FinancialTFM, ModelConfig

    base = {
        "max_features": 8, "max_classes": 3, "d_cell": 16, "d_model": 32, "n_heads": 2,
        "n_col_layers": 1, "n_layers": 2, "d_ff": 64,
    }
    X = torch.randn(2, 10, 8)
    y = torch.randint(0, 3, (2, 10))
    nc = torch.tensor([3, 3])

    torch.manual_seed(4)
    m_old = FinancialTFM(ModelConfig(**base)).eval()
    torch.manual_seed(4)
    m_new = FinancialTFM(ModelConfig(**base, head_type="binned")).eval()

    with torch.no_grad():
        a = m_old(X, y, 6, nc, column_id_seed=0)
        b = m_new(X, y, 6, nc, column_id_seed=0)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert not hasattr(m_old, "quantile_taus")


def test_quantile_head_shapes_and_gradients(n_cell_blocks=0):
    """Shapes, gradients and parameter coverage for task 48.3, before any GPU spend."""
    from fintfm.modeling.model import FinancialTFM, ModelConfig

    base = {
        "max_features": 8, "max_classes": 3, "d_cell": 16, "d_model": 32, "n_heads": 2,
        "n_col_layers": 1, "n_layers": 2, "d_ff": 64,
    }
    n_q = 7
    X = torch.randn(2, 10, 8)
    y_cont = torch.randn(2, 10)  # continuous, context-normalised target (collate's contract)
    nc = torch.tensor([3, 3])  # ignored for a quantile head, passed to exercise that path

    torch.manual_seed(5)
    model = FinancialTFM(ModelConfig(**base, head_type="quantile", n_quantiles=n_q))
    out = model(X, y_cont, 6)
    assert out.shape == (2, 10, n_q)

    loss = model.loss(X, y_cont, 6, nc)
    assert loss.item() >= 0.0  # pinball loss is non-negative by construction
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(p.grad).all(), f"{name} has a non-finite gradient"


def test_quantile_head_taus_are_evenly_spaced_and_interior():
    """quantile_taus must avoid the two undefined endpoints 0 and 1 and cover the rest evenly,
    since the pinball loss is built directly from this tensor."""
    from fintfm.modeling.model import FinancialTFM, ModelConfig

    model = FinancialTFM(
        ModelConfig(max_features=4, d_model=8, n_heads=1, n_layers=1, d_ff=8,
                    head_type="quantile", n_quantiles=9)
    )
    taus = model.quantile_taus
    assert taus.shape == (9,)
    assert taus.min() > 0.0 and taus.max() < 1.0
    diffs = taus[1:] - taus[:-1]
    torch.testing.assert_close(diffs, diffs[0].expand_as(diffs), rtol=1e-5, atol=1e-6)


def test_quantile_head_learns_a_known_linear_target():
    """Not a contract test: trains past the point of chance, as direct evidence the pinball
    loss actually teaches the head something, before any GPU time is spent on it."""
    import numpy as np

    from fintfm.modeling.model import FinancialTFM, ModelConfig
    from fintfm.prior.base import Task, collate

    torch.manual_seed(6)
    cfg = ModelConfig(
        max_features=6, max_classes=3, d_cell=16, d_model=32, n_heads=2, n_col_layers=1,
        n_layers=1, d_ff=32, head_type="quantile", n_quantiles=9,
    )
    model = FinancialTFM(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    rng = np.random.default_rng(0)

    def make_batch(bs=8, n_rows=60):
        tasks = []
        for _ in range(bs):
            X = rng.normal(size=(n_rows, 5)).astype(np.float32)
            y_cont = (X[:, 0] * 2.0 + rng.normal(scale=0.2, size=n_rows)).astype(np.float64)
            tasks.append(
                Task(
                    X=X, y=np.zeros(n_rows, dtype=np.int64), n_classes=0,
                    is_categorical=np.zeros(5, dtype=bool), source="test", y_continuous=y_cont,
                )
            )
        return collate(tasks, n_ctx=30, max_features=6, head_type="quantile")

    losses = []
    for _ in range(80):
        batch = make_batch()
        loss = model.loss(batch.X, batch.y, batch.n_ctx, batch.n_classes)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert np.mean(losses[-10:]) < np.mean(losses[:10])


def test_quantile_head_requires_continuous_y_in_collate():
    """head_type='quantile' has no class-index path; a classification task must be refused
    rather than silently z-scored as if its 0/1 label were a continuous measurement."""
    import numpy as np

    from fintfm.prior.base import Task, collate

    task = Task(
        X=np.zeros((10, 3), dtype=np.float32), y=np.zeros(10, dtype=np.int64), n_classes=2,
        is_categorical=np.zeros(3, dtype=bool), source="financial",
    )
    with pytest.raises(ValueError, match="quantile"):
        collate([task], n_ctx=5, max_features=3, head_type="quantile")


def test_feature_chunking_is_an_identity():
    """Chunking row-within-feature attention over ``F`` must change no number.

    The whole justification for `openspec/changes/cell-attention-and-task-inference` task
    39.24 is that this is a memory/throughput knob and not an approximation -- it is what
    lets `max_context=4000` run at all (``docs/results/FINDINGS.md`` §79/§81), and if it perturbed
    predictions it would silently invalidate every comparison made across chunk sizes.

    Chunk sizes deliberately include one that does not divide the feature count (5 into 17)
    and one larger than it (64), since an off-by-one in the reshape would show up only
    there.
    """
    torch.manual_seed(0)
    cfg = ModelConfig(
        max_features=17, d_cell=16, d_model=32, n_layers=2, n_col_layers=2,
        n_heads=4, max_classes=3, n_cell_blocks=2, cell_labels=True, column_id_dim=8,
    )
    model = FinancialTFM(cfg).eval()
    X = torch.randn(3, 40, 17)
    X[0, :, 16] = float("nan")  # an entirely-absent column, so padding is exercised too
    y = torch.randint(0, 2, (3, 40))

    with torch.no_grad():
        model.feature_chunk = None
        reference = model(X, y, n_ctx=25, column_id_seed=7)
        for chunk in (1, 2, 5, 16, 17, 64):
            model.feature_chunk = chunk
            got = model(X, y, n_ctx=25, column_id_seed=7)
            assert torch.equal(got, reference), f"feature_chunk={chunk} changed the output"


def test_feature_chunk_rejects_a_nonsense_size():
    """A zero or negative chunk must fail loudly rather than silently skipping features."""
    cfg = ModelConfig(
        max_features=6, d_cell=8, d_model=16, n_layers=1, n_col_layers=1,
        n_heads=2, max_classes=2, n_cell_blocks=1,
    )
    model = FinancialTFM(cfg).eval()
    model.feature_chunk = 0
    with pytest.raises(ValueError, match="feature_chunk"):
        model(torch.randn(1, 8, 6), torch.randint(0, 2, (1, 8)), n_ctx=5)


def test_normalize_features_pads_from_context_rows_only():
    """A column absent from the whole context is padding, whatever a query row holds.

    Audit AU1.1: ``pad`` used to be computed over context *and* query rows, so a single query
    with a value in a context-empty column un-padded that column for the whole task.
    """
    X = torch.full((1, 5, 2), float("nan"))
    X[0, :, 0] = torch.arange(5.0)
    X[0, 4, 1] = 3.0  # query row 4 has a value in a column the context never saw
    _Z, _missing, pad = normalize_features(X, n_ctx=3)
    assert pad.tolist() == [[False, True]]


@pytest.mark.parametrize("n_cell_blocks", [0, 1])
def test_query_chunking_is_exact_with_a_context_empty_column(n_cell_blocks):
    """A query's output must not depend on which other queries share its forward pass.

    The classifier and quantile regressor split queries into chunks and document that as
    exact (audit AU1.1). Here query ``a`` is scored alone and beside a mate ``b`` that holds a
    value in a column the context leaves entirely NaN; ``a``'s output must not move.
    """
    torch.manual_seed(0)
    cfg = ModelConfig(max_features=6, max_classes=2, d_cell=16, d_model=32, n_heads=2,
                      n_col_layers=1, n_layers=2, d_ff=64, n_cell_blocks=n_cell_blocks,
                      cell_labels=n_cell_blocks > 0)
    model = FinancialTFM(cfg).eval()
    n_ctx = 8
    ctx = torch.randn(n_ctx, 6)
    ctx[:, 5] = float("nan")  # column 5 is absent from every context row
    a = torch.randn(1, 6)
    a[0, 5] = float("nan")
    b = torch.randn(1, 6)  # the mate has a value in column 5
    y = torch.randint(0, 2, (1, n_ctx + 2))
    ncl = torch.tensor([2])
    with torch.no_grad():
        alone = model(torch.cat([ctx, a])[None], y[:, : n_ctx + 1], n_ctx=n_ctx, n_classes=ncl)
        paired = model(torch.cat([ctx, a, b])[None], y, n_ctx=n_ctx, n_classes=ncl)
    torch.testing.assert_close(alone[:, n_ctx], paired[:, n_ctx], rtol=1e-5, atol=1e-6)


# --- Audit Phase 5 levers (openspec/changes/audit-2026-10-15 AU5.3, AU5.16, AU5.17) ---
#
# Every lever is a ModelConfig field whose default must reproduce the pre-lever model byte
# for byte. The fingerprint below was recorded by running this exact construction on the
# code before the levers existed (8b30a75): the sum over every output and its first three
# values. A drift here means the default path changed, which would invalidate every existing
# checkpoint's comparisons. AU5.9's hazard tests live in tests/test_hazard.py.

_P5_BASE = {"max_features": 6, "max_classes": 3, "d_cell": 16, "d_model": 32, "n_heads": 2,
            "n_col_layers": 1, "n_layers": 2, "d_ff": 64}
_P5_FINGERPRINT = {
    "meanmax": ({}, -33.0643186122179,
                [0.09483376145362854, -0.8273341655731201, -0.38515689969062805]),
    "attention": ({"pooling": "attention"}, 11.312919471412897,
                  [0.5645540952682495, 0.12391750514507294, -0.2492905855178833]),
    "cellblocks": ({"n_cell_blocks": 1, "cell_labels": True}, 42.728899128735065,
                   [0.2567908763885498, 0.4492500424385071, 0.5122548937797546]),
    "maskemb": ({"mask_embedding": True}, -15.284929860383272,
                [-0.3825910985469818, 0.10315485298633575, 0.26675528287887573]),
    "hazard": ({"max_classes": 2, "n_horizons": 4}, 29.38597935438156,
               [0.37285876274108887, 0.7123607397079468, 0.8926969170570374]),
}
_P5_KEYS = ("n_cell_heads", "n_summary_tokens", "value_embedding", "value_basis_size",
            "value_basis_learnable", "value_ecdf", "attn_length_scale", "attn_length_ref",
            "attn_impl", "hazard_period_context")
#: The design review DR6 levers (DR6.2, DR6.3), deleted alongside: a checkpoint saved before
#: them lacks their keys too.
_DR6_KEYS = ("cell_row_attn", "n_inducing", "readout", "n_readout", "head_mlp")


def _p5_inputs(max_classes):
    """Fixed two-task batch with a missing cell and a padded column; 9 context rows of 14."""
    g = torch.Generator().manual_seed(1)
    X = torch.randn(2, 14, 6, generator=g)
    X[0, 2, 1] = float("nan")
    X[:, :, 5] = float("nan")
    y = torch.randint(0, max_classes, (2, 14), generator=g)
    return X, y


def _p5_run(model, X, y, n_ctx=9):
    """Logits, or the term structure for a hazard model, without gradients."""
    with torch.no_grad():
        if model.cfg.n_horizons:
            return model.term_structure(X, y, n_ctx)
        return model(X, y, n_ctx)


@pytest.mark.parametrize("case", sorted(_P5_FINGERPRINT))
def test_phase5_defaults_reproduce_the_pre_phase5_fingerprint(case):
    extra, total, head = _P5_FINGERPRINT[case]
    torch.manual_seed(0)
    cfg = ModelConfig(**{**_P5_BASE, **extra})
    out = _p5_run(FinancialTFM(cfg).eval(), *_p5_inputs(cfg.max_classes))
    assert float(out.double().sum()) == pytest.approx(total, abs=1e-4)
    torch.testing.assert_close(out.flatten()[:3], torch.tensor(head), rtol=0, atol=1e-6)


@pytest.mark.parametrize("case", ["meanmax", "hazard"])
def test_checkpoint_without_phase5_keys_loads_with_identical_outputs(case, tmp_path):
    """A checkpoint saved before the levers existed has none of their keys in its config."""
    torch.manual_seed(0)
    model = FinancialTFM(ModelConfig(**{**_P5_BASE, **_P5_FINGERPRINT[case][0]})).eval()
    path = tmp_path / "old.pt"
    model.save(str(path))
    ckpt = torch.load(str(path), weights_only=True)
    for key in (*_P5_KEYS, *_DR6_KEYS):
        del ckpt["config"][key]
    torch.save(ckpt, str(path))
    loaded = FinancialTFM.load(str(path))
    X, y = _p5_inputs(model.cfg.max_classes)
    torch.testing.assert_close(_p5_run(loaded, X, y), _p5_run(model, X, y), rtol=0, atol=0)


@pytest.mark.parametrize(
    "bad, match",
    [
        ({"value_embedding": "spline"}, "value_embedding"),
        ({"value_basis_size": 0}, "value_basis_size"),
        ({"n_summary_tokens": 0}, "n_summary_tokens"),
        ({"n_summary_tokens": 4, "pooling": "attention"}, "pooling='attention'"),
        ({"n_cell_heads": 0}, "n_cell_heads"),
        ({"attn_length_ref": 1}, "attn_length_ref"),
        ({"hazard_period_context": True}, "requires n_horizons"),
        ({"attn_impl": "flash"}, "attn_impl"),
        ({"cell_row_attn": "sparse", "n_cell_blocks": 1}, "cell_row_attn"),
        ({"cell_row_attn": "induced"}, "requires n_cell_blocks"),
        ({"cell_row_attn": "induced", "n_cell_blocks": 1, "n_inducing": 0}, "n_inducing"),
        ({"readout": "summary"}, "readout"),
        ({"readout": "cls", "n_readout": 0}, "n_readout"),
        ({"readout": "cls", "n_summary_tokens": 2}, "replaces the pooling step"),
        ({"readout": "cls", "pooling": "attention"}, "replaces the pooling step"),
    ],
)
def test_phase5_config_refuses_nonsense(bad, match):
    with pytest.raises(ValueError, match=match):
        ModelConfig(**{**_P5_BASE, **bad})


def test_value_basis_rbf_peaks_at_its_centres_and_fourier_is_bounded():
    from fintfm.modeling.model import ValueBasis

    rbf = ValueBasis("rbf", 7, learnable=False)
    feats = rbf(torch.tensor([-3.0, 0.0, 3.0]))
    assert feats.shape == (3, 7)
    assert feats.argmax(dim=-1).tolist() == [0, 3, 6]  # each probe sits on one centre
    torch.testing.assert_close(feats.max(dim=-1).values, torch.ones(3))
    assert "centres" in dict(rbf.named_buffers())  # fixed, yet saved with the checkpoint
    fourier = ValueBasis("fourier", 5, learnable=True)
    out = fourier(torch.linspace(-10, 10, 50))
    assert out.shape == (50, 10) and out.abs().max() <= 1.0
    assert "freqs" in dict(fourier.named_parameters())
    # the frequency draw is private and fixed: the same every build, global RNG untouched
    state = torch.random.get_rng_state()
    again = ValueBasis("fourier", 5, learnable=True)
    assert torch.equal(torch.random.get_rng_state(), state)
    torch.testing.assert_close(again.freqs, fourier.freqs, rtol=0, atol=0)


@pytest.mark.parametrize(
    "kind, learnable, ecdf, mask_embedding",
    [("rbf", False, False, False), ("rbf", True, True, False),
     ("fourier", False, True, True), ("fourier", True, False, False),
     ("mlp", False, True, False)],
)
def test_value_embedding_variants_run_and_train(kind, learnable, ecdf, mask_embedding):
    torch.manual_seed(0)
    cfg = ModelConfig(**_P5_BASE, value_embedding=kind, value_basis_size=4,
                      value_basis_learnable=learnable, value_ecdf=ecdf,
                      mask_embedding=mask_embedding)
    model = FinancialTFM(cfg)
    X, y = _p5_inputs(3)
    loss = model.loss(X, y, 9, torch.tensor([3, 3]))
    loss.backward()
    assert torch.isfinite(loss)
    if learnable:
        name = "centres" if kind == "rbf" else "freqs"
        assert getattr(model.value_basis, name).grad.abs().sum() > 0


def test_ecdf_rank_uses_context_rows_only():
    from fintfm.modeling.model import ecdf_rank

    ctx = torch.tensor([1.0, 2.0, 2.0, float("nan"), 4.0])  # four present context values
    query = torch.tensor([2.0, 0.0, 9.0, float("nan")])
    X = torch.cat([ctx, query]).reshape(1, 9, 1)
    r = ecdf_rank(X, n_ctx=5)[0, :, 0]
    # mid-rank u = (#below + #at-or-below) / (2 * 4), reported as 2u - 1; missing -> 0
    u = torch.tensor([1 / 8, 4 / 8, 4 / 8, 0.5, 7 / 8, 4 / 8, 0.0, 1.0, 0.5])
    torch.testing.assert_close(r, 2 * u - 1)
    # moving the query rows changes no other row: they are not in the reference set
    X2 = X.clone()
    X2[0, 5:, 0] = torch.tensor([100.0, -100.0, 3.0, 2.5])
    r2 = ecdf_rank(X2, n_ctx=5)[0, :, 0]
    torch.testing.assert_close(r2[:5], r[:5], rtol=0, atol=0)
    assert r2[7].item() == pytest.approx(2 * (6 / 8) - 1)  # 3.0: above 1, 2, 2; below 4
    # a column with no context values carries no rank at all
    X3 = torch.full((1, 4, 1), float("nan"))
    X3[0, 3, 0] = 1.0
    assert ecdf_rank(X3, n_ctx=2).abs().sum() == 0


def test_value_ecdf_and_basis_keep_query_rows_independent():
    """A query's output must not move when another query's values do (no ECDF leakage)."""
    torch.manual_seed(0)
    model = FinancialTFM(ModelConfig(**_P5_BASE, value_embedding="rbf", value_ecdf=True))
    model.eval()
    X, y = _p5_inputs(3)
    X2 = X.clone()
    X2[:, 13, :5] = torch.randn(2, 5) * 50  # an extreme last query row
    with torch.no_grad():
        a, b = model(X, y, 9), model(X2, y, 9)
    torch.testing.assert_close(a[:, :13], b[:, :13], rtol=1e-5, atol=1e-6)
    assert not torch.allclose(a[:, 13], b[:, 13])


def test_length_scale_formula():
    from fintfm.modeling.model import length_scale

    assert length_scale(512, 512) == 1.0
    assert length_scale(4096, 512) == pytest.approx(12 / 9)
    assert length_scale(64, 512) == pytest.approx(6 / 9)
    assert length_scale(1, 512) == length_scale(2, 512) > 0


def test_scaled_self_attention_multiplies_the_logits():
    """Factor 1 is the stock module; factor f is the stock module with queries times f."""
    torch.manual_seed(0)
    model = FinancialTFM(ModelConfig(**_P5_BASE)).eval()
    mha = model.encoder.layers[0].self_attn
    x = torch.randn(2, 7, 32)
    mask = FinancialTFM._row_mask(7, 4, x.device)
    with torch.no_grad():
        stock = mha(x, x, x, attn_mask=mask, need_weights=False)[0]
        torch.testing.assert_close(
            model._scaled_self_attention(mha, x, ~mask, 1.0), stock, rtol=1e-5, atol=1e-6
        )
        scaled = model._scaled_self_attention(mha, x, ~mask, 2.5)
        sharper = torch.nn.MultiheadAttention(32, 2, batch_first=True).eval()
        sharper.load_state_dict(mha.state_dict())
        sharper.in_proj_weight[:32] *= 2.5  # the query block of the packed projection
        sharper.in_proj_bias[:32] *= 2.5
        ref = sharper(x, x, x, attn_mask=mask, need_weights=False)[0]
    torch.testing.assert_close(scaled, ref, rtol=1e-5, atol=1e-6)


def test_attn_length_scale_is_the_stock_model_at_the_reference_length():
    torch.manual_seed(0)
    off = FinancialTFM(ModelConfig(**_P5_BASE)).eval()
    on = FinancialTFM(ModelConfig(**_P5_BASE, attn_length_scale=True, attn_length_ref=9))
    on.load_state_dict(off.state_dict())  # the flag adds no parameter
    on.eval()
    X, y = _p5_inputs(3)
    with torch.no_grad():
        torch.testing.assert_close(on(X, y, 9), off(X, y, 9), rtol=1e-5, atol=1e-5)
        assert not torch.allclose(on(X, y, 4), off(X, y, 4), atol=1e-4)
    on.train()
    on.loss(X, y, 4, torch.tensor([3, 3])).backward()  # the training path runs too


def test_n_cell_heads_sets_the_cell_stage_only():
    cfg = ModelConfig(**_P5_BASE, n_cell_heads=4, n_cell_blocks=1, pooling="attention")
    model = FinancialTFM(cfg)
    assert model.column_encoder.layers[0].self_attn.num_heads == 4
    assert model.row_within_feature_encoder.layers[0].self_attn.num_heads == 4
    assert model.pool_attn.num_heads == 4
    assert model.encoder.layers[0].self_attn.num_heads == 2


def test_summary_tokens_replace_pooling_and_keep_column_order_invariance():
    torch.manual_seed(0)
    cfg = ModelConfig(**_P5_BASE, n_summary_tokens=4, column_id_dim=0, value_ecdf=True)
    model = FinancialTFM(cfg).eval()
    assert model.row_proj.in_features == 4 * cfg.d_cell
    X, y = _p5_inputs(3)
    perm = torch.tensor([3, 0, 5, 1, 4, 2])
    with torch.no_grad():
        a = model(X, y, 9)
        b = model(X[:, :, perm], y, 9)
    torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)
    model.train()
    model.loss(X, y, 9, torch.tensor([3, 3])).backward()
    assert model.summary_queries.grad.abs().sum() > 0


# --- AU5.2: the split row attention, an exact unmasked route to the masked one ---------


def _masked_reference(q, k, v, n_ctx, scale):
    """The masked row attention through SDPA, the definition split_row_attention matches."""
    allowed = ~FinancialTFM._row_mask(q.shape[-2], n_ctx, q.device)
    return torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=allowed, scale=scale
    )


@pytest.mark.parametrize(
    "n_rows, n_ctx, d", [(7, 4, 16), (13, 13, 8), (5, 0, 12), (40, 9, 3), (64, 63, 32)]
)
def test_split_row_attention_equals_the_masked_attention(n_rows, n_ctx, d):
    from fintfm.modeling.model import split_row_attention

    g = torch.Generator().manual_seed(n_rows * 100 + n_ctx)
    q, k, v = (torch.randn(2, 3, n_rows, d, generator=g) * 2 for _ in range(3))
    for scale in (1 / d**0.5, 2.7 / d**0.5):
        torch.testing.assert_close(
            split_row_attention(q, k, v, n_ctx, scale), _masked_reference(q, k, v, n_ctx, scale),
            rtol=1e-5, atol=1e-5,
        )


def test_split_row_attention_gradients_equal_the_masked_ones():
    from fintfm.modeling.model import split_row_attention

    g = torch.Generator().manual_seed(3)
    base = [torch.randn(2, 2, 11, 8, generator=g) for _ in range(3)]
    grads = []
    for fn in (split_row_attention, _masked_reference):
        qkv = [t.clone().requires_grad_(True) for t in base]
        (fn(*qkv, 6, 0.35) * torch.linspace(-1, 1, 8)).sum().backward()
        grads.append([t.grad for t in qkv])
    for a, b in zip(*grads):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)


_SPLIT_CASES = {
    "meanmax": {},
    "attention": {"pooling": "attention"},
    "cellblocks": {"n_cell_blocks": 2, "cell_labels": True},
    "cellblocks_chunked": {"n_cell_blocks": 1},
    "lengthscale": {"attn_length_scale": True, "attn_length_ref": 4},
    "summary": {"n_summary_tokens": 3, "n_cell_blocks": 1, "attn_length_scale": True},
    "heads": {"n_heads": 4, "n_cell_heads": 1, "n_cell_blocks": 1},
    "hazard": {"max_classes": 2, "n_horizons": 4},
}


@pytest.mark.parametrize("case", sorted(_SPLIT_CASES))
@pytest.mark.parametrize("n_ctx", [9, 3, 14])
def test_attn_impl_split_matches_masked_on_the_whole_model(case, n_ctx):
    """Same weights, both implementations: fp32 outputs equal within 1e-5, padding included
    (a missing cell and an all-NaN column in ``_p5_inputs``)."""
    torch.manual_seed(0)
    extra = _SPLIT_CASES[case]
    masked = FinancialTFM(ModelConfig(**{**_P5_BASE, **extra})).eval()
    split = FinancialTFM(ModelConfig(**{**_P5_BASE, **extra, "attn_impl": "split"})).eval()
    split.load_state_dict(masked.state_dict())  # the flag adds no parameter
    if case == "cellblocks_chunked":
        masked.feature_chunk = split.feature_chunk = 2
    X, y = _p5_inputs(masked.cfg.max_classes)
    torch.testing.assert_close(
        _p5_run(split, X, y, n_ctx), _p5_run(masked, X, y, n_ctx), rtol=1e-5, atol=1e-5
    )


def test_attn_impl_split_matches_masked_partway_through_the_row_stack():
    """The classifier's ``encoder_mid`` probe runs the first layers only, through the same
    ``_row_attention``; the two implementations agree there too."""
    torch.manual_seed(0)
    masked = FinancialTFM(ModelConfig(**_P5_BASE)).eval()
    split = FinancialTFM(ModelConfig(**_P5_BASE, attn_impl="split")).eval()
    split.load_state_dict(masked.state_dict())
    h = torch.randn(2, 14, 32)
    with torch.no_grad():
        for n_layers in (None, 1):
            torch.testing.assert_close(
                split._row_attention(h, 9, n_layers=n_layers),
                masked._row_attention(h, 9, n_layers=n_layers),
                rtol=1e-5, atol=1e-5,
            )
        one = masked._row_attention(h, 9, n_layers=1)
        assert not torch.allclose(one, masked._row_attention(h, 9))


def test_attn_impl_split_trains_to_the_masked_gradients():
    torch.manual_seed(0)
    extra = {"n_cell_blocks": 1, "cell_labels": True, "attn_length_scale": True}
    masked = FinancialTFM(ModelConfig(**{**_P5_BASE, **extra}))
    split = FinancialTFM(ModelConfig(**{**_P5_BASE, **extra, "attn_impl": "split"}))
    split.load_state_dict(masked.state_dict())
    X, y = _p5_inputs(3)
    for m in (masked, split):
        m.train()
        torch.manual_seed(5)  # training draws fresh column identities; draw the same ones
        m.loss(X, y, 9, torch.tensor([3, 3])).backward()
    for (name, a), (_, b) in zip(masked.named_parameters(), split.named_parameters()):
        if a.grad is None:
            assert b.grad is None, name
            continue
        torch.testing.assert_close(b.grad, a.grad, rtol=1e-4, atol=1e-5, msg=name)


def test_masked_checkpoint_loads_and_runs_under_split(tmp_path):
    """attn_impl changes no weight: an old checkpoint (no attn_impl key) runs either way."""
    from dataclasses import replace

    torch.manual_seed(0)
    model = FinancialTFM(ModelConfig(**_P5_BASE, n_cell_blocks=1)).eval()
    path = tmp_path / "old.pt"
    model.save(str(path))
    ckpt = torch.load(str(path), weights_only=True)
    del ckpt["config"]["attn_impl"]
    torch.save(ckpt, str(path))
    loaded = FinancialTFM.load(str(path))
    assert loaded.cfg.attn_impl == "masked"
    split = FinancialTFM(replace(loaded.cfg, attn_impl="split")).eval()
    split.load_state_dict(loaded.state_dict())
    X, y = _p5_inputs(3)
    torch.testing.assert_close(_p5_run(split, X, y), _p5_run(model, X, y), rtol=1e-5, atol=1e-5)


# --- DR5.5: half-precision attention keeps a head width the fused kernels accept --------

# Cell heads 24 / 2 = 12 channels wide, as at the published recipe (48 / 4); the row heads
# (32 / 2 = 16) are already aligned.
_DR55_BASE = {**_P5_BASE, "d_cell": 24, "n_cell_blocks": 1, "cell_labels": True}


def _record_attention_calls(monkeypatch):
    """Record (dtype, q/k/v widths, mask dtype and shape) of every SDPA call, stock or ours.

    ``nn.MultiheadAttention`` reaches the kernel through ``torch.nn.functional``'s module
    global, and ``model.py`` through ``F.``, so patching the attribute catches both.
    """
    calls = []
    real = torch.nn.functional.scaled_dot_product_attention

    def recording(q, k, v, attn_mask=None, *args, **kwargs):
        mask = None if attn_mask is None else (attn_mask.dtype, tuple(attn_mask.shape))
        calls.append((q.dtype, q.shape[-1], k.shape[-1], v.shape[-1], mask))
        return real(q, k, v, attn_mask, *args, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", recording)
    return calls


@pytest.mark.parametrize("attn_impl", ["masked", "split"])
@pytest.mark.parametrize("chunk", [None, 2])
def test_half_precision_attention_gets_fused_head_widths_and_head_broadcast_masks(
    monkeypatch, attn_impl, chunk
):
    """DR5.5's invariant, checked on CPU under bf16 autocast (the dispatch it protects is
    CUDA's): every attention call of a training step gets 16-bit q, k and v whose width is a
    multiple of 8, and a mask broadcast over heads rather than the per-head copy
    ``nn.MultiheadAttention`` builds from a key-padding mask. In fp32 the stock 12-wide calls
    are still made, so the default path is the one it always was."""
    from fintfm.modeling.model import HALF_PRECISION_HEAD_ALIGN

    torch.manual_seed(0)
    model = FinancialTFM(ModelConfig(**_DR55_BASE, attn_impl=attn_impl)).train()
    model.feature_chunk = chunk
    X, y = _p5_inputs(3)
    calls = _record_attention_calls(monkeypatch)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        model.loss(X, y, 9, torch.tensor([3, 3])).backward()
    assert calls
    for dtype, dq, dk, dv, mask in calls:
        assert dtype == torch.bfloat16
        assert dq % HALF_PRECISION_HEAD_ALIGN == dk % HALF_PRECISION_HEAD_ALIGN == 0
        assert dv % HALF_PRECISION_HEAD_ALIGN == 0
        if mask is not None:
            assert len(mask[1]) == 2 or mask[1][1] == 1, mask

    calls.clear()
    model.loss(X, y, 9, torch.tensor([3, 3])).backward()
    assert all(c[0] == torch.float32 for c in calls)
    assert any(c[1] == 12 for c in calls)  # fp32 is not padded


@pytest.mark.parametrize("shape", ["none", "rows", "keys"])
def test_head_padding_leaves_attention_and_its_gradients_unchanged(shape):
    """The padding :func:`aligned_attention` adds is exact: in fp32, attention over q, k, v
    padded with zero channels (scale kept at the true width) and sliced back equals the
    unpadded one, values and gradients, under each mask the model passes."""
    from fintfm.modeling import model as model_mod

    g = torch.Generator().manual_seed(7)
    base = [torch.randn(3, 2, 10, 12, generator=g) for _ in range(3)]
    mask = {
        "none": None,
        "rows": ~FinancialTFM._row_mask(10, 6, torch.device("cpu")),
        "keys": torch.tensor([True] * 7 + [False] * 3).expand(3, 1, 1, 10),
    }[shape]
    sdpa = torch.nn.functional.scaled_dot_product_attention
    results = []
    for padded in (False, True):
        qkv = [t.clone().requires_grad_(True) for t in base]
        if padded:
            pad = (0, model_mod.HALF_PRECISION_HEAD_ALIGN - 12 % model_mod.HALF_PRECISION_HEAD_ALIGN)
            q, k, v = (torch.nn.functional.pad(t, pad) for t in qkv)
            out = sdpa(q, k, v, attn_mask=mask, scale=12**-0.5)[..., :12]
        else:
            out = sdpa(*qkv, attn_mask=mask)
        (out * torch.linspace(-1, 1, 12)).sum().backward()
        results.append([out, *(t.grad for t in qkv)])
    for a, b in zip(*results):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
    # and fp32 input passes through aligned_attention untouched: same call, same bits
    torch.testing.assert_close(
        model_mod.aligned_attention(*base, attn_mask=mask), sdpa(*base, attn_mask=mask),
        rtol=0, atol=0,
    )


@pytest.mark.parametrize("attn_impl", ["masked", "split"])
@pytest.mark.parametrize("chunk", [None, 2])
def test_half_precision_route_matches_the_stock_encoders_in_float32(
    monkeypatch, attn_impl, chunk
):
    """The cell stages' half-precision route (the encoder layers run by ``_prenorm_stack``
    with boolean masks) computes what the stock ``nn.TransformerEncoder`` computes: forced
    on in fp32, outputs and gradients equal the stock path's within 1e-5."""
    from fintfm.modeling import model as model_mod

    torch.manual_seed(0)
    model = FinancialTFM(ModelConfig(**_DR55_BASE, attn_impl=attn_impl)).train()
    model.feature_chunk = chunk
    X, y = _p5_inputs(3)
    results = []
    for forced in (False, True):
        if forced:
            monkeypatch.setattr(model_mod, "attention_in_half_precision", lambda x: True)
        model.zero_grad()
        torch.manual_seed(5)  # training draws fresh column identities; draw the same ones
        loss = model.loss(X, y, 9, torch.tensor([3, 3]))
        loss.backward()
        results.append([loss.detach(), *(p.grad.clone() for p in model.parameters()
                                          if p.grad is not None)])
    assert len(results[0]) == len(results[1])
    for a, b in zip(*results):
        torch.testing.assert_close(b, a, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the fused-kernel dispatch is CUDA's")
@pytest.mark.parametrize("attn_impl", ["masked", "split"])
def test_fp16_peak_memory_is_at_most_fp32s_on_cuda(attn_impl):
    """DR5.5's Verify: at the recipe's cell width (48 channels, four 12-wide heads), one
    fp16-autocast training step on a fixed batch peaks at no more CUDA memory than the same
    step in fp32. Before the fix fp16 fell back to the math kernel and ran out of memory on
    a T4 where fp32 needed 6.2 GB (FINDINGS §170)."""
    torch.manual_seed(0)
    cfg = ModelConfig(max_features=64, max_classes=2, d_cell=48, d_model=128, n_heads=4,
                      n_col_layers=2, n_layers=2, d_ff=512, n_cell_blocks=1, cell_labels=True,
                      column_id_dim=16, attn_impl=attn_impl)
    model = FinancialTFM(cfg).cuda().train()
    g = torch.Generator().manual_seed(0)
    X = torch.randn(2, 512, 64, generator=g).cuda()
    y = torch.randint(0, 2, (2, 512), generator=g).cuda()
    n_classes = torch.tensor([2, 2]).cuda()
    peaks = {}
    for amp in ("fp32", "fp16"):
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp == "fp16"):
            loss = model.loss(X, y, 384, n_classes)
        loss.backward()
        torch.cuda.synchronize()
        peaks[amp] = torch.cuda.max_memory_allocated() - base
        assert torch.isfinite(loss)
    assert peaks["fp16"] <= peaks["fp32"], peaks


def test_checkpoint_records_provenance_and_load_ignores_it(tmp_path):
    """DR1.2: every save writes a provenance record, and ``load`` never reads it. The same
    weights saved with the record and with the key deleted (a pre-DR1.2 checkpoint) must
    give outputs identical to each other and to the model that wrote them."""
    import re
    from dataclasses import asdict

    from fintfm.provenance import json_safe, read_checkpoint_provenance

    torch.manual_seed(0)
    model = FinancialTFM(ModelConfig(**_P5_BASE)).eval()
    new, old = tmp_path / "new.pt", tmp_path / "old.pt"
    model.save(str(new), provenance={"tasks_seen": 16, "note": ("a", 1)})
    rec = read_checkpoint_provenance(new)
    assert rec["model_config"] == json_safe(asdict(model.cfg))
    assert rec["tasks_seen"] == 16 and rec["note"] == ["a", 1]  # tuples stored as lists
    assert rec["versions"]["torch"] == torch.__version__ and "fintfm" in rec["versions"]
    # best effort: a SHA inside a checkout, None from an installed package
    assert rec["commit"] is None or re.fullmatch(r"[0-9a-f]{40}", rec["commit"])
    assert rec["dirty"] in (None, True, False)

    ckpt = torch.load(str(new), weights_only=True)
    del ckpt["provenance"]
    torch.save(ckpt, str(old))
    assert read_checkpoint_provenance(old) is None
    X, y = _p5_inputs(3)
    reference = _p5_run(model, X, y)
    for path in (new, old):
        loaded = FinancialTFM.load(str(path))
        assert not hasattr(loaded, "provenance")
        torch.testing.assert_close(_p5_run(loaded, X, y), reference, rtol=0, atol=0)


# --- DR1.4: one inference entry point -----------------------------------------------------


def _context_model(**extra) -> FinancialTFM:
    """A tiny seeded eval-mode model for the predict_context tests."""
    torch.manual_seed(0)
    cfg = ModelConfig(max_features=8, max_classes=3, d_cell=8, d_model=16, n_heads=2,
                      n_col_layers=1, n_layers=2, d_ff=32, column_id_dim=4, **extra)
    return FinancialTFM(cfg).eval()


def _context_task(n_ctx: int = 30, n_q: int = 7, f: int = 5):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n_ctx + n_q, f)).astype(np.float32)
    y = rng.integers(0, 2, size=n_ctx + n_q).astype(np.int64)
    return X[:n_ctx], y[:n_ctx], X[n_ctx:]


def test_pack_context_pads_features_and_leaves_query_placeholders():
    from fintfm.modeling.hazard import CENSORED
    from fintfm.modeling.model import pack_context

    Xc, yc, Xq = _context_task()
    X, y, period = pack_context(Xc, yc, Xq, 8, np.int64, period=np.arange(30) % 3)
    assert X.shape == (1, 37, 8) and X.dtype == np.float32
    assert np.isnan(X[0, :, 5:]).all()
    np.testing.assert_array_equal(X[0, :30, :5], Xc)
    np.testing.assert_array_equal(X[0, 30:, :5], Xq)
    assert y.dtype == np.int64 and (y[0, 30:] == 0).all()
    assert period is not None and (period[0, 30:] == CENSORED).all()
    assert pack_context(Xc, yc, Xq, 8, np.float32)[2] is None


@pytest.mark.parametrize("pad", [True, False])
def test_predict_context_is_forward_on_the_packed_task(pad):
    """The entry point adds nothing to forward: same logits, padded or at the caller's width.

    ``pad=False`` is the capability probes' path, which must reproduce their historical
    ``model(torch.tensor(X)[None], ...)`` call bit for bit.
    """
    model = _context_model(n_cell_blocks=1, cell_labels=True)
    Xc, yc, Xq = _context_task()
    width = 8 if pad else 5
    X = torch.full((1, 37, width), float("nan"))
    X[0, :, :5] = torch.from_numpy(np.concatenate([Xc, Xq]))
    # the probes passed the query labels too; the model must never read them
    y = torch.from_numpy(np.concatenate([yc, np.ones(7, dtype=np.int64)]))[None]
    with torch.no_grad():
        expected = model(X, y, 30, torch.tensor([2]), column_id_seed=5)[0, 30:]
    got = model.predict_context(Xc, yc, Xq, n_classes=2, column_id_seed=5, pad_features=pad)
    assert torch.equal(got, expected)


def test_predict_context_representation_is_what_the_head_reads():
    model = _context_model()
    Xc, yc, Xq = _context_task()
    logits = model.predict_context(Xc, yc, Xq, column_id_seed=2)
    pre_head = model.predict_context(
        Xc, yc, Xq, column_id_seed=2, output="representation", stage="pre_head"
    )
    torch.testing.assert_close(model.head(pre_head), logits, rtol=1e-5, atol=1e-6)
    for stage in ("encode_rows", "encoder_mid"):
        h = model.predict_context(Xc, yc, Xq, output="representation", stage=stage)
        assert h.shape == (7, 16)


def test_predict_context_term_structure_takes_the_column_seed():
    model = _context_model(n_horizons=3)
    Xc, yc, Xq = _context_task()
    a = model.predict_context(Xc, yc, Xq, column_id_seed=0, output="term_structure")
    b = model.predict_context(Xc, yc, Xq, column_id_seed=4, output="term_structure")
    default = model.predict_context(Xc, yc, Xq, output="term_structure")
    assert a.shape == (7, 3)
    assert torch.equal(a, default)  # None in eval mode is seed 0, every earlier caller's draw
    assert not torch.equal(a, b)


def test_predict_context_refuses_what_it_cannot_score():
    model = _context_model()
    Xc, yc, Xq = _context_task(f=9)
    with pytest.raises(ValueError, match="at most 8 features"):
        model.predict_context(Xc, yc, Xq)
    Xc, yc, Xq = _context_task()
    with pytest.raises(ValueError, match="stage"):
        model.predict_context(Xc, yc, Xq, output="representation")
    with pytest.raises(ValueError, match="stage"):
        model.predict_context(Xc, yc, Xq, stage="pre_head")
    with pytest.raises(ValueError, match="unknown output"):
        model.predict_context(Xc, yc, Xq, output="proba")  # type: ignore[arg-type]


# --- Design review DR6 levers: DR6.2 induced cell stage, DR6.3 readout tokens and MLP head ---
#
# Defaults: test_phase5_defaults_reproduce_the_pre_phase5_fingerprint above was recorded before
# these fields existed and still passes, and tests/test_golden_outputs.py fingerprints every
# inference output; the test below adds that no default builds a DR6 module.

_DR6_INDUCED = {"n_cell_blocks": 1, "cell_labels": True, "cell_row_attn": "induced",
                "n_inducing": 4}
_DR6_CLS = {"readout": "cls", "n_readout": 4}
_DR6_CASES = {
    "induced": _DR6_INDUCED,
    "cls": _DR6_CLS,
    "induced+cls": {**_DR6_INDUCED, **_DR6_CLS, "n_cell_blocks": 2},
}


def _dr6_model(**extra):
    """A seeded tiny model on ``_P5_BASE`` plus ``extra``, in eval mode."""
    torch.manual_seed(0)
    return FinancialTFM(ModelConfig(**{**_P5_BASE, **extra})).eval()


def _record_attention_shapes(monkeypatch):
    """Record ``(head width, query length, key length)`` of every SDPA call."""
    calls = []
    real = torch.nn.functional.scaled_dot_product_attention

    def recording(q, k, v, *args, **kwargs):
        calls.append((q.shape[-1], q.shape[-2], k.shape[-2]))
        return real(q, k, v, *args, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", recording)
    return calls


def test_dr6_defaults_build_no_dr6_module():
    model = _dr6_model(n_cell_blocks=1)
    names = {n for n, _ in model.named_parameters()}
    assert not any(n.startswith(("induced_", "inducing_points", "readout_tokens")) for n in names)
    assert isinstance(model.head, torch.nn.Linear)
    assert hasattr(model, "row_within_feature_encoder")
    explicit = _dr6_model(n_cell_blocks=1, cell_row_attn="full", readout="pool", head_mlp=False)
    X, y = _p5_inputs(3)
    torch.testing.assert_close(_p5_run(explicit, X, y), _p5_run(model, X, y), rtol=0, atol=0)


@pytest.mark.parametrize("case", ["induced", "induced+cls"])
@pytest.mark.parametrize("attn_impl", ["masked", "split"])
def test_induced_stage_lets_no_query_reach_another_row(case, attn_impl):
    """DR6.2: query cells are never gathered from, so changing one query row's values (and
    every query label, which no input reads) leaves every other row's logits unchanged."""
    model = _dr6_model(**_DR6_CASES[case], attn_impl=attn_impl)
    X, y = _p5_inputs(3)
    n_ctx, row = 9, 12
    X2, y2 = X.clone(), y.clone()
    X2[:, row, :5] = 5.0 * torch.randn(2, 5, generator=torch.Generator().manual_seed(4))
    y2[:, n_ctx:] = (y2[:, n_ctx:] + 1) % 3
    base, moved = _p5_run(model, X, y, n_ctx), _p5_run(model, X2, y2, n_ctx)
    others = [r for r in range(X.shape[1]) if r != row]
    torch.testing.assert_close(moved[:, others], base[:, others], rtol=0, atol=1e-6)
    assert not torch.allclose(moved[:, row], base[:, row])  # the query itself does move


@pytest.mark.parametrize("case", sorted(_DR6_CASES))
def test_dr6_levers_score_in_query_and_feature_chunks_exactly(case):
    """Queries scored two at a time, and the cell stage run two features at a time, give the
    one-pass logits."""
    model = _dr6_model(**_DR6_CASES[case])
    X, y = _p5_inputs(3)
    Xc, yc, Xq = X[0, :9].numpy(), y[0, :9].numpy(), X[0, 9:].numpy()
    whole = model.predict_context(Xc, yc, Xq, n_classes=3, pad_features=False)
    chunked = torch.cat([
        model.predict_context(Xc, yc, Xq[i : i + 2], n_classes=3, pad_features=False)
        for i in range(0, len(Xq), 2)
    ])
    torch.testing.assert_close(chunked, whole, rtol=1e-5, atol=1e-6)
    model.feature_chunk = 2
    torch.testing.assert_close(
        model.predict_context(Xc, yc, Xq, n_classes=3, pad_features=False), whole,
        rtol=1e-5, atol=1e-6,
    )


def test_induced_stage_is_linear_in_rows(monkeypatch):
    """DR6.2: no cell-width attention is rows x rows. The cell stage's calls are k inducing
    vectors over the 9 context cells, the 14 rows over k summaries, and the 6-feature column
    attention; only the row encoder (16-wide heads) still attends 14 x 14."""
    model = _dr6_model(**_DR6_INDUCED).train()  # train: the stock layers' fast path skips SDPA
    calls = _record_attention_shapes(monkeypatch)
    X, y = _p5_inputs(3)
    model.loss(X, y, 9, torch.tensor([3, 3])).backward()
    cell = {(lq, lk) for width, lq, lk in calls if width == 8}
    row = {(lq, lk) for width, lq, lk in calls if width == 16}
    assert cell == {(4, 9), (14, 4), (6, 6)}, cell
    assert row == {(14, 14)}, row
    assert model.inducing_points.grad is not None and model.inducing_points.grad.abs().sum() > 0


@pytest.mark.parametrize("case", sorted(_DR6_CASES))
def test_dr6_levers_ignore_padded_columns(case):
    """A padded (all-NaN) column changes nothing: without column identities, a table with one
    gives the logits of the same table without it. The readout tokens never see its cells."""
    model = _dr6_model(**_DR6_CASES[case], column_id_dim=0)
    X, y = _p5_inputs(3)  # column 5 is padding
    torch.testing.assert_close(
        _p5_run(model, X[:, :, :5], y), _p5_run(model, X, y), rtol=1e-5, atol=1e-6
    )


@pytest.mark.parametrize("case", sorted(_DR6_CASES))
def test_dr6_levers_keep_column_order_invariance(case):
    """D4: readout tokens carry no position, so permuting columns permutes nothing out."""
    model = _dr6_model(**_DR6_CASES[case], column_id_dim=0)
    X, y = _p5_inputs(3)
    perm = torch.tensor([3, 0, 4, 1, 2, 5])
    torch.testing.assert_close(
        _p5_run(model, X[:, :, perm], y), _p5_run(model, X, y), rtol=1e-5, atol=1e-6
    )


def test_readout_tokens_join_column_attention(monkeypatch):
    """DR6.3: the column attention runs over the 6 cells plus 4 readout tokens, the row vector
    is their 4 x d_cell outputs, and gradients reach the tokens."""
    model = _dr6_model(**_DR6_CLS).train()
    calls = _record_attention_shapes(monkeypatch)
    X, y = _p5_inputs(3)
    model.loss(X, y, 9, torch.tensor([3, 3])).backward()
    assert {(lq, lk) for width, lq, lk in calls if width == 8} == {(10, 10)}
    assert model.readout_tokens.shape == (1, 4, 16)
    assert model.row_proj.in_features == 4 * 16
    assert model.readout_tokens.grad is not None and model.readout_tokens.grad.abs().sum() > 0
    assert not hasattr(model, "summary_queries") and not hasattr(model, "pool_query")


@pytest.mark.parametrize("head_type", ["binned", "quantile"])
def test_head_mlp_is_two_layers_at_d_ff(head_type):
    """DR6.3's head: d_model -> d_ff -> out with a GELU, for either head type."""
    extra = {"head_type": head_type, "n_quantiles": 5} if head_type == "quantile" else {}
    linear, mlp = _dr6_model(**extra), _dr6_model(**extra, head_mlp=True)
    out = 5 if head_type == "quantile" else 3
    assert [type(m) for m in mlp.head] == [torch.nn.Linear, torch.nn.GELU, torch.nn.Linear]
    assert (mlp.head[0].in_features, mlp.head[0].out_features) == (32, 64)
    assert mlp.head[2].out_features == out
    added = (32 * 64 + 64) + (64 * out + out) - (32 * out + out)
    assert mlp.num_parameters() - linear.num_parameters() == added
    X, y = _p5_inputs(3)
    y = y.float() if head_type == "quantile" else y
    assert _p5_run(mlp, X, y).shape == (2, 14, out)


@pytest.mark.parametrize("attn_impl", ["masked", "split"])
@pytest.mark.parametrize("chunk", [None, 2])
def test_dr6_levers_get_fused_head_widths_in_half_precision(monkeypatch, attn_impl, chunk):
    """DR5.5's invariant holds on the new stages: at 12-wide cell heads, every attention call
    of a bf16-autocast training step gets 16-bit q, k, v at a multiple of 8 channels."""
    from fintfm.modeling.model import HALF_PRECISION_HEAD_ALIGN

    torch.manual_seed(0)
    model = FinancialTFM(ModelConfig(
        **{**_DR55_BASE, **_DR6_INDUCED, **_DR6_CLS, "attn_impl": attn_impl}
    )).train()
    model.feature_chunk = chunk
    X, y = _p5_inputs(3)
    calls = _record_attention_calls(monkeypatch)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = model.loss(X, y, 9, torch.tensor([3, 3]))
    loss.backward()
    assert torch.isfinite(loss) and calls
    for dtype, dq, dk, dv, _mask in calls:
        assert dtype == torch.bfloat16
        assert dq % HALF_PRECISION_HEAD_ALIGN == dk % HALF_PRECISION_HEAD_ALIGN == 0
        assert dv % HALF_PRECISION_HEAD_ALIGN == 0


def test_dr6_levers_agree_across_attn_impl():
    """``attn_impl`` changes only how the row encoder computes the same attention."""
    extra = {**_DR6_INDUCED, **_DR6_CLS}
    X, y = _p5_inputs(3)
    masked, split = _dr6_model(**extra), _dr6_model(**extra, attn_impl="split")
    torch.testing.assert_close(_p5_run(split, X, y), _p5_run(masked, X, y), rtol=1e-5, atol=1e-5)


def test_induced_stage_refuses_an_empty_context():
    model = _dr6_model(**_DR6_INDUCED)
    X, y = _p5_inputs(3)
    with pytest.raises(ValueError, match="at least one context row"):
        model(X, y, 0)
