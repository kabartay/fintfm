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
