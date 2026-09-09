"""The monotonicity guarantee, and the survival likelihood that fits the whole curve.

`docs/FINDINGS.md` §11 measured 39% of firms receiving a cumulative-PD curve that fell as
the horizon grew. These tests assert the property that makes that impossible **by
construction** rather than by training, which is the difference between a design advantage
and a scale one.
"""

import numpy as np
import pytest
import torch

from fintfm.inference.classifier import FinancialTFMClassifier
from fintfm.modeling.hazard import (
    CENSORED,
    HazardHead,
    base_rate_shift,
    coherence_violations,
    shift_cumulative_pd,
)
from fintfm.modeling.model import FinancialTFM, ModelConfig


def test_cumulative_pd_is_monotone_for_random_representations():
    torch.manual_seed(0)
    head = HazardHead(d_model=16, n_horizons=6)
    pd = head.cumulative_pd(torch.randn(500, 16))
    assert pd.shape == (500, 6)
    assert coherence_violations(pd).item() == 0
    assert (pd >= 0).all() and (pd <= 1).all()


def test_monotone_even_with_adversarial_weights():
    """The guarantee must not depend on the weights being reasonable.

    A learned monotonicity penalty fails exactly here: extreme or hostile parameters break
    it. A cumulative-product parameterisation cannot break, and that is the whole point.
    """
    torch.manual_seed(0)
    head = HazardHead(d_model=8, n_horizons=8)
    with torch.no_grad():
        head.proj.weight.copy_(torch.randn_like(head.proj.weight) * 500)
        head.proj.bias.copy_(torch.linspace(-400, 400, 8))
    pd = head.cumulative_pd(torch.randn(400, 8) * 100)
    assert coherence_violations(pd).item() == 0
    assert torch.isfinite(pd).all()


def test_hazards_stay_in_the_open_unit_interval():
    torch.manual_seed(0)
    head = HazardHead(d_model=8, n_horizons=4, max_hazard=0.9)
    hz = head.hazards(torch.randn(200, 8) * 50)
    assert (hz > 0).all(), "a zero hazard makes the log-likelihood infinite"
    assert (hz <= 0.9).all(), "max_hazard must be respected so later horizons keep shape"


def test_survival_loss_prefers_the_true_default_period():
    """A model that puts its hazard mass in the right period must score better."""
    torch.manual_seed(0)
    head = HazardHead(d_model=4, n_horizons=5)
    x = torch.zeros(1, 4)  # constant input, so the bias alone sets the hazards
    with torch.no_grad():
        head.proj.weight.zero_()
        head.proj.bias.copy_(torch.tensor([-6.0, -6.0, 2.0, -6.0, -6.0]))  # mass in period 2
    correct = head.loss(x, torch.tensor([2]))
    wrong = head.loss(x, torch.tensor([0]))
    assert correct < wrong


def test_censored_rows_are_scored_on_survival_only():
    torch.manual_seed(0)
    head = HazardHead(d_model=4, n_horizons=4)
    x = torch.zeros(2, 4)
    with torch.no_grad():
        head.proj.weight.zero_()
        head.proj.bias.fill_(-5.0)  # low hazard everywhere: survival is likely
    censored = head.loss(x, torch.tensor([CENSORED, CENSORED]))
    defaulted = head.loss(x, torch.tensor([0, 0]))
    assert censored < defaulted, "low hazards should favour survival, not early default"


def test_survival_loss_decreases_under_optimisation():
    torch.manual_seed(0)
    head = HazardHead(d_model=12, n_horizons=5)
    x = torch.randn(256, 12)
    # firms with a large first feature default early; others are censored
    period = torch.where(x[:, 0] > 0.5, torch.zeros(256, dtype=torch.long),
                         torch.full((256,), CENSORED, dtype=torch.long))
    opt = torch.optim.Adam(head.parameters(), lr=0.05)
    losses = []
    for _ in range(120):
        loss = head.loss(x, period)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]
    # and the guarantee survives training
    assert coherence_violations(head.cumulative_pd(x)).item() == 0


def test_rejects_a_period_outside_the_grid():
    head = HazardHead(d_model=4, n_horizons=3)
    with pytest.raises(ValueError, match="outside grid"):
        head.loss(torch.zeros(1, 4), torch.tensor([5]))


def test_rejects_a_degenerate_horizon_grid():
    with pytest.raises(ValueError, match="n_horizons"):
        HazardHead(d_model=4, n_horizons=0)


def test_coherence_violations_detects_a_broken_curve():
    """The diagnostic must catch what §11 found, or it is not a guard."""
    good = torch.tensor([[0.01, 0.02, 0.05]])
    bad = torch.tensor([[0.05, 0.02, 0.01]])  # falls: a firm un-defaulting
    assert coherence_violations(good).item() == 0
    assert coherence_violations(bad).item() == 2


def test_model_term_structure_is_monotone_end_to_end():
    """The guarantee must survive the whole model, not just the head in isolation."""
    from fintfm.modeling.hazard import coherence_violations as viol
    from fintfm.modeling.model import FinancialTFM, ModelConfig

    torch.manual_seed(0)
    cfg = ModelConfig(max_features=8, max_classes=2, d_cell=16, d_model=32, n_heads=2,
                      n_col_layers=1, n_layers=2, d_ff=64, n_horizons=5)
    model = FinancialTFM(cfg).eval()
    X = torch.randn(2, 20, 8)
    X[0, 3, 4] = float("nan")
    y = torch.randint(0, 2, (2, 20))
    with torch.no_grad():
        ts = model.term_structure(X, y, n_ctx=12)
    assert ts.shape == (2, 8, 5)
    assert viol(ts).item() == 0
    assert torch.isfinite(ts).all()


def test_model_without_hazard_head_refuses_clearly():
    from fintfm.modeling.model import FinancialTFM, ModelConfig

    cfg = ModelConfig(max_features=6, max_classes=2, d_cell=8, d_model=16, n_heads=2,
                      n_col_layers=1, n_layers=1, d_ff=16)
    model = FinancialTFM(cfg)
    with pytest.raises(RuntimeError, match="no hazard head"):
        model.term_structure(torch.randn(1, 4, 6), torch.zeros(1, 4, dtype=torch.long), n_ctx=2)


def test_term_structure_checkpoint_roundtrip(tmp_path):
    """n_horizons must survive save/load, or a served model silently loses the head."""
    from fintfm.modeling.model import FinancialTFM, ModelConfig

    cfg = ModelConfig(max_features=6, max_classes=2, d_cell=8, d_model=16, n_heads=2,
                      n_col_layers=1, n_layers=1, d_ff=16, n_horizons=4)
    path = str(tmp_path / "hz.pt")
    FinancialTFM(cfg).save(path)
    loaded = FinancialTFM.load(path)
    assert loaded.cfg.n_horizons == 4
    assert loaded.hazard is not None


def test_prior_emits_periods_consistent_with_the_binary_label():
    """`y` must be exactly `period != CENSORED`, or the two objectives contradict."""
    import numpy as np

    from fintfm.prior.financial import sample_financial_task

    rng = np.random.default_rng(0)
    for _ in range(6):
        t = sample_financial_task(rng, 800, max_features=32, n_horizons=5)
        assert t.period is not None and t.n_horizons == 5
        assert ((t.period != CENSORED) == (t.y == 1)).all()
        assert t.period.max() < 5
        assert t.period.min() >= CENSORED


def test_prior_without_horizons_carries_no_period():
    import numpy as np

    from fintfm.prior.financial import sample_financial_task

    t = sample_financial_task(np.random.default_rng(0), 100, max_features=16)
    assert t.period is None and t.n_horizons is None


def test_collate_refuses_to_mix_survival_and_binary_tasks():
    """A padded period is indistinguishable from a real one, so refuse rather than pad."""
    import numpy as np
    import pytest as _pytest

    from fintfm.prior.base import collate
    from fintfm.prior.financial import sample_financial_task

    rng = np.random.default_rng(0)
    a = sample_financial_task(rng, 64, max_features=16, n_horizons=5)
    b = sample_financial_task(rng, 64, max_features=16)
    with _pytest.raises(ValueError, match="mixes survival and binary-only"):
        collate([a, b], n_ctx=32, max_features=16)


def test_mixture_refuses_horizons_with_a_generic_component():
    import numpy as np
    import pytest as _pytest

    from fintfm.prior import PriorConfig
    from fintfm.prior.mixture import sample_task

    cfg = PriorConfig(max_features=16, max_classes=2, p_financial=0.7, n_horizons=5)
    with _pytest.raises(ValueError, match="requires p_financial=1.0"):
        sample_task(np.random.default_rng(0), cfg, n_rows=32)


def test_survival_training_reduces_loss_and_keeps_the_guarantee():
    """The end-to-end objective must train, and coherence must survive it."""
    import numpy as np

    from fintfm.modeling.hazard import coherence_violations as viol
    from fintfm.modeling.model import FinancialTFM, ModelConfig
    from fintfm.prior import PriorConfig
    from fintfm.prior.mixture import sample_batch

    torch.manual_seed(0)
    cfg = ModelConfig(max_features=32, max_classes=2, d_cell=16, d_model=32, n_heads=2,
                      n_col_layers=1, n_layers=2, d_ff=64, n_horizons=5)
    model = FinancialTFM(cfg)
    prior = PriorConfig(max_features=32, max_classes=2, p_financial=1.0, n_rows=64, n_horizons=5)
    rng = np.random.default_rng(0)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    for _ in range(60):
        b = sample_batch(rng, prior, batch_size=8)
        loss = model.survival_loss(b.X, b.y, b.period, b.n_ctx)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert np.mean(losses[-5:]) < np.mean(losses[:5])
    with torch.no_grad():
        ts = model.term_structure(b.X, b.y, b.n_ctx)
    assert viol(ts).item() == 0


def test_censoring_is_per_row_not_global():
    """A firm observed for 2 horizons must not be scored as surviving all 5.

    Real panels are ragged: V4FinBench falls from 1,000,087 rows at h=0 to 598,832 at h=5
    because a 5-year-ahead label needs five more years of data. Scoring a short-observed
    firm as a long-run survivor biases every hazard downward.
    """
    torch.manual_seed(0)
    head = HazardHead(d_model=4, n_horizons=5)
    x = torch.zeros(1, 4)
    with torch.no_grad():
        head.proj.weight.zero_()
        head.proj.bias.fill_(-2.0)

    short = head.loss(x, torch.tensor([CENSORED]), n_observed=torch.tensor([2]))
    full = head.loss(x, torch.tensor([CENSORED]), n_observed=torch.tensor([5]))
    # surviving 5 periods is stronger evidence than surviving 2, so it costs more likelihood
    assert short < full
    # and the default (no n_observed) must equal the full-grid case
    assert torch.isclose(head.loss(x, torch.tensor([CENSORED])), full)


def test_per_row_censoring_ignores_horizons_beyond_observation():
    """Hazards past a row's observation window must not affect its likelihood."""
    torch.manual_seed(0)
    head = HazardHead(d_model=4, n_horizons=4)
    x = torch.zeros(1, 4)
    with torch.no_grad():
        head.proj.weight.zero_()
        head.proj.bias.copy_(torch.tensor([-3.0, -3.0, 5.0, 5.0]))  # huge late hazards
    observed_two = head.loss(x, torch.tensor([CENSORED]), n_observed=torch.tensor([2]))
    with torch.no_grad():
        head.proj.bias.copy_(torch.tensor([-3.0, -3.0, -5.0, -5.0]))  # tiny late hazards
    observed_two_again = head.loss(x, torch.tensor([CENSORED]), n_observed=torch.tensor([2]))
    assert torch.isclose(observed_two, observed_two_again, atol=1e-6)


# --- base-rate correction on the term structure ------------------------------------
# §28: the out-of-time harness ran the model directly and skipped the correction, so it
# reported a 12.8% default rate against a 0.47% truth and the error was diagnosed as the
# prior's fault. These guard the correction itself and the path that must apply it.


def test_base_rate_shift_is_inert_when_the_context_matches_the_population():
    assert base_rate_shift(0.2, 0.2) == pytest.approx(0.0)


def test_base_rate_shift_is_negative_for_an_over_balanced_context():
    # a balanced context against a 1.5% portfolio must pull predictions down
    assert base_rate_shift(0.5, 0.015) < -4.0


def test_base_rate_shift_refuses_a_degenerate_rate():
    for bad in (0.0, 1.0, -0.1):
        with pytest.raises(ValueError):
            base_rate_shift(bad, 0.5)
        with pytest.raises(ValueError):
            base_rate_shift(0.5, bad)


def test_shift_preserves_monotonicity_and_ranking():
    # the structural guarantee must survive the correction, at any shift magnitude
    torch.manual_seed(0)
    curve = torch.cumsum(torch.rand(64, 6) * 0.1, dim=1).clamp(1e-6, 1 - 1e-6)
    assert (curve.diff(dim=-1) >= 0).all()
    for delta in (-8.0, -4.15, -0.5, 0.5, 4.0):
        shifted = shift_cumulative_pd(curve, delta)
        assert coherence_violations(shifted).item() == 0
        # rank-preserving at every horizon, so AUC cannot move
        for k in range(curve.shape[1]):
            assert torch.equal(curve[:, k].argsort(), shifted[:, k].argsort())


def test_shift_moves_levels_towards_the_true_rate():
    curve = torch.full((256, 4), 0.5)
    shifted = shift_cumulative_pd(curve, base_rate_shift(0.5, 0.01))
    assert shifted.mean().item() < 0.02


def test_classifier_term_structure_applies_the_correction():
    """The regression guard for §28: the public path must not restate the context's rate."""
    rng = np.random.default_rng(0)
    n = 3000
    X = rng.normal(size=(n, 8)).astype(np.float32)
    y = (rng.random(n) < 0.02).astype(np.int64)
    model = FinancialTFM(ModelConfig(max_features=8, d_model=32, d_cell=16, n_layers=1,
                                     n_col_layers=1, max_classes=2, n_horizons=4))
    corrected = FinancialTFMClassifier(model, max_context=200, context_strategy="balanced",
                                       correct_prior=True).fit(X, y)
    raw = FinancialTFMClassifier(model, max_context=200, context_strategy="balanced",
                                 correct_prior=False).fit(X, y)
    assert corrected._ctx_rate > 5 * corrected._full_rate  # balancing really did distort
    c, r = corrected.predict_term_structure(X[:200]), raw.predict_term_structure(X[:200])
    assert c.mean() < r.mean()
    assert coherence_violations(torch.from_numpy(c)).item() == 0


def test_classifier_term_structure_is_chunk_exact():
    """Chunking must not change a curve, for the same reason predict_proba is exact."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(400, 6)).astype(np.float32)
    y = (rng.random(400) < 0.1).astype(np.int64)
    model = FinancialTFM(ModelConfig(max_features=6, d_model=32, d_cell=16, n_layers=1,
                                     n_col_layers=1, max_classes=2, n_horizons=3))
    kw = {"max_context": 100, "context_strategy": "balanced"}
    whole = FinancialTFMClassifier(model, query_chunk=4096, **kw).fit(X, y)
    split = FinancialTFMClassifier(model, query_chunk=37, **kw).fit(X, y)
    np.testing.assert_allclose(
        whole.predict_term_structure(X[:300]), split.predict_term_structure(X[:300]), atol=1e-5
    )


def test_classifier_term_structure_refuses_a_model_without_a_hazard_head():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(50, 5)).astype(np.float32)
    y = (rng.random(50) < 0.3).astype(np.int64)
    model = FinancialTFM(ModelConfig(max_features=5, d_model=16, d_cell=8, n_layers=1,
                                     n_col_layers=1, max_classes=2))
    clf = FinancialTFMClassifier(model, max_context=40).fit(X, y)
    with pytest.raises(RuntimeError, match="no hazard head"):
        clf.predict_term_structure(X)
