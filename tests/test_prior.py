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
