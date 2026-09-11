import numpy as np

from fintfm.prior import PriorConfig, sample_financial_task, sample_scm_task, sample_task
from fintfm.prior.mixture import sample_batch


def test_financial_task_shapes_and_labels():
    rng = np.random.default_rng(0)
    task = sample_financial_task(rng, n_rows=200, max_features=24)
    assert task.X.shape == (200, task.n_features)
    assert task.n_features <= 24
    assert set(np.unique(task.y)) <= {0, 1}
    assert task.n_classes == 2
    assert np.isnan(task.X).any() or True  # missingness is probabilistic, just must not crash


def test_scm_task_dense_labels():
    rng = np.random.default_rng(1)
    for _ in range(20):
        task = sample_scm_task(rng, n_rows=150, max_features=16, max_classes=6)
        classes = np.unique(task.y)
        assert classes.min() == 0
        assert classes.max() == len(classes) - 1
        assert task.n_classes == len(classes)


def test_mixture_batch_collates():
    rng = np.random.default_rng(2)
    cfg = PriorConfig(max_features=20, max_classes=8, n_rows=64)
    batch = sample_batch(rng, cfg, batch_size=8)
    assert batch.X.shape == (8, 64, 20)
    assert batch.y.shape == (8, 64)
    assert 1 < batch.n_ctx < 64


def test_sample_task_reproducible_with_seed():
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    cfg = PriorConfig()
    t1 = sample_task(rng1, cfg, n_rows=32)
    t2 = sample_task(rng2, cfg, n_rows=32)
    np.testing.assert_array_equal(t1.y, t2.y)


def test_base_rate_floor_scales_with_task_size():
    """Finding 26: a fixed small task size silently caps how imbalanced any task can be.

    A 256-row task cannot carry a 0.2% default rate — the expected count is 0.5 — so the
    floor must derive from the task size. The strategy targets low-default portfolios and
    the prior could not generate one.
    """
    from fintfm.prior.financial import sample_financial_task

    rates_small, rates_large = [], []
    rng = np.random.default_rng(0)
    for _ in range(24):
        rates_small.append(sample_financial_task(rng, 256, max_features=32).y.mean())
    for _ in range(24):
        rates_large.append(sample_financial_task(rng, 4096, max_features=32).y.mean())
    assert min(rates_large) < min(rates_small), "large tasks must reach lower base rates"
    assert min(rates_large) < 0.005, "must reach the low-default regime (<0.5%)"


def test_batch_task_size_varies_when_choices_given():
    from fintfm.prior import PriorConfig
    from fintfm.prior.mixture import sample_batch

    cfg = PriorConfig(max_features=16, max_classes=2, n_rows=128,
                      n_rows_choices=(64, 256, 512))
    rng = np.random.default_rng(0)
    sizes = {sample_batch(rng, cfg, batch_size=2).X.shape[1] for _ in range(12)}
    assert sizes <= {64, 256, 512}
    assert len(sizes) > 1, "task size should vary across batches"


# --- difficulty span (docs/FINDINGS.md §42) ----------------------------------------


def test_prior_spans_difficulty_from_noise_to_learnable():
    """The prior must contain tasks a model can actually learn from.

    §42's root cause: sharpness was clamped to (0.7, 2.7) so every task's label was mostly
    noise with a ceiling near 0.75. The model learned to do as well as anything can on
    irreducibly noisy data and never learned to extract a sharp boundary, scoring 0.68 on a
    clean linear task that logistic regression solves at 0.9997. A prior's job is to teach,
    not to resemble the test set.

    Asserted on the *label's* separability from the latent driver rather than by fitting a
    model, so the test is fast and does not depend on a baseline's optimiser.
    """
    import numpy as np

    from fintfm.prior.financial import _SHARPNESS_MAX, _SHARPNESS_MIN, sample_financial_task

    assert _SHARPNESS_MIN < 1.0 < _SHARPNESS_MAX
    assert _SHARPNESS_MAX / _SHARPNESS_MIN >= 20, "difficulty span is too narrow to teach"

    # over many tasks the achievable separation must vary widely: some near-noise, some sharp
    seps = []
    for seed in range(40):
        t = sample_financial_task(np.random.default_rng(seed), n_rows=600, max_features=40)
        X, y = np.asarray(t.X, dtype=np.float64), np.asarray(t.y)
        if len(np.unique(y)) < 2:
            continue
        # best single-feature separation, a floor on how learnable the task is
        best = 0.5
        for j in range(X.shape[1]):
            col = X[:, j]
            ok = np.isfinite(col)
            if ok.sum() < 100 or len(np.unique(y[ok])) < 2:
                continue
            order = np.argsort(col[ok])
            yy = y[ok][order]
            n1 = yy.sum()
            n0 = len(yy) - n1
            if n0 == 0 or n1 == 0:
                continue
            ranks = np.arange(1, len(yy) + 1)
            auc = (ranks[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n0 * n1)
            best = max(best, max(auc, 1 - auc))
        seps.append(best)

    seps = np.array(seps)
    assert len(seps) >= 20, f"too few usable tasks: {len(seps)}"
    assert seps.max() > 0.90, (
        f"no task is learnable (max single-feature AUC {seps.max():.3f}); a prior of only "
        "hard tasks is what §42 diagnosed"
    )
    assert seps.min() < 0.75, (
        f"no task is hard (min single-feature AUC {seps.min():.3f}); a prior of only easy "
        "tasks never teaches the model to abstain"
    )


def test_label_direction_varies_across_tasks():
    """The prior must not admit a global feature-to-label rule.

    §47: the driver signs were fixed, so in every task higher leverage meant riskier. A model
    could memorise one distress score and never read a context label — and measurably did not:
    shuffling the context labels left its predictions unchanged (rank correlation 0.977) and
    its AUC *higher* than with true labels. In-context learning cannot be learned from a prior
    that does not require it.
    """
    import numpy as np
    from sklearn.metrics import roc_auc_score

    from fintfm.prior.financial import sample_financial_task

    aucs = []
    for seed in range(40):
        t = sample_financial_task(np.random.default_rng(seed), n_rows=600, max_features=30)
        X, y = np.asarray(t.X, dtype=float), np.asarray(t.y)
        col = X[:, 0]
        ok = np.isfinite(col)
        if ok.sum() < 200 or len(np.unique(y[ok])) < 2:
            continue
        aucs.append(roc_auc_score(y[ok], col[ok]))

    a = np.array(aucs)
    assert len(a) >= 20, f"too few usable tasks: {len(a)}"
    up = float((a > 0.5).mean())
    assert 0.25 < up < 0.75, (
        f"the first feature points the same way in {up:.0%} of tasks; a global rule would "
        "still work and the model need never read its context"
    )


def test_trivial_prior_is_actually_trivial():
    """The control only works if a fitted linear model solves it outright.

    §53: the model scores ~0.67 on every task source, including ones with a 0.995 ceiling. The
    trivial prior asks whether the architecture can learn in-context prediction at all, and
    that question is only meaningful if the task itself is unambiguously solvable.
    """
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    from fintfm.prior.trivial import sample_trivial_task

    aucs = []
    for seed in range(12):
        t = sample_trivial_task(np.random.default_rng(seed), 1200)
        X, y = np.asarray(t.X), np.asarray(t.y)
        assert np.isfinite(X).all(), "the trivial prior must not emit missing values"
        assert len(np.unique(y)) == 2
        k = 600
        aucs.append(
            roc_auc_score(y[k:], LogisticRegression(max_iter=1000)
                          .fit(X[:k], y[:k]).predict_proba(X[k:])[:, 1])
        )
    assert np.mean(aucs) > 0.99, f"trivial tasks are not trivial: {np.mean(aucs):.4f}"


def test_trivial_prior_still_randomises_the_label_direction():
    """Even the control must not admit a global rule (§47)."""
    import numpy as np
    from sklearn.metrics import roc_auc_score

    from fintfm.prior.trivial import sample_trivial_task

    aucs = []
    for seed in range(40):
        t = sample_trivial_task(np.random.default_rng(seed), 500)
        X, y = np.asarray(t.X), np.asarray(t.y)
        aucs.append(roc_auc_score(y, X[:, 0]))
    up = float((np.array(aucs) > 0.5).mean())
    assert 0.25 < up < 0.75, f"first feature points the same way in {up:.0%} of tasks"


def test_trivial_prior_is_off_by_default():
    from fintfm.prior.mixture import PriorConfig

    assert PriorConfig().p_trivial == 0.0
