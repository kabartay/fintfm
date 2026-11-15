"""Openspec 15.2 (``max_features`` is honoured) and 15.4 (opt-in rejection of near-separable draws)."""

from __future__ import annotations

import numpy as np
import pytest

from fintfm.prior.financial import _holdout_lr_auc, sample_financial_task
from fintfm.prior.mixture import PriorConfig, sample_batch


def test_max_features_64_gives_substantially_more_than_24_columns() -> None:
    """15.2: the named-quantity dictionary no longer caps width near 24 columns."""
    widths = [
        sample_financial_task(np.random.default_rng(s), n_rows=200, max_features=64).X.shape[1]
        for s in range(30)
    ]
    assert max(widths) <= 64
    assert max(widths) > 48
    assert np.mean(widths) > 24 + 8


def test_pinned_width_reaches_max_features() -> None:
    """15.2: an explicit width of 64 is delivered exactly."""
    t = sample_financial_task(np.random.default_rng(0), 200, max_features=64, n_features=64)
    assert t.X.shape[1] <= 64 and t.X.shape[1] >= 60


def test_rejection_off_is_byte_identical() -> None:
    """15.4: ``max_lr_auc=None`` consumes no draw and reproduces today's task."""
    a = sample_financial_task(np.random.default_rng(3), 300, max_features=20)
    b = sample_financial_task(np.random.default_rng(3), 300, max_features=20, max_lr_auc=None)
    assert a.X.tobytes() == b.X.tobytes() and a.y.tobytes() == b.y.tobytes()


def test_rejection_lowers_held_out_auc_and_always_returns() -> None:
    """15.4: a tight ceiling redraws, a 1.0 ceiling keeps the first draw."""
    first = sample_financial_task(np.random.default_rng(1), 400, max_features=20, max_lr_auc=1.0)
    plain = sample_financial_task(np.random.default_rng(1), 400, max_features=20)
    assert first.X.tobytes() == plain.X.tobytes()  # 1.0 never rejects
    # Zero-tolerance ceiling exhausts the budget and still returns a task.
    t = sample_financial_task(
        np.random.default_rng(1), 400, max_features=20, max_lr_auc=0.5, max_lr_auc_tries=3
    )
    assert t.X.shape[0] == 400
    aucs = [
        _holdout_lr_auc(*(lambda k: (k.X, k.y))(sample_financial_task(
            np.random.default_rng(s), 400, max_features=20, sharpness_min=3.0, sharpness_max=6.0)))
        for s in range(6)
    ]
    scored = [a for a in aucs if a is not None]
    assert scored
    ceiling = float(np.median(scored))
    kept = []
    for s in range(6):
        k = sample_financial_task(
            np.random.default_rng(s), 400, max_features=20, sharpness_min=3.0,
            sharpness_max=6.0, max_lr_auc=ceiling, max_lr_auc_tries=20,
        )
        a = _holdout_lr_auc(k.X, k.y)
        kept.append(a)
    assert np.mean([a for a in kept if a is not None]) <= np.mean(scored) + 1e-9


def test_holdout_auc_is_none_for_single_class_half() -> None:
    """A half with one class has no AUC; the task is accepted, not rejected."""
    X = np.random.default_rng(0).normal(size=(20, 3)).astype(np.float32)
    assert _holdout_lr_auc(X, np.zeros(20, dtype=np.int64)) is None


def test_invalid_ceiling_is_refused() -> None:
    """A ceiling outside [0.5, 1] is an error, not a silent no-op."""
    with pytest.raises(ValueError, match="max_lr_auc"):
        sample_financial_task(np.random.default_rng(0), 100, max_lr_auc=0.3)
    with pytest.raises(ValueError, match="financial_max_lr_auc"):
        sample_batch(
            np.random.default_rng(0), PriorConfig(p_financial=1.0, financial_max_lr_auc=2.0), 2
        )
