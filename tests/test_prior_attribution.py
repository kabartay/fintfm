"""Ground-truth attributions from the financial prior (openspec ``zero-shot-attribution`` 12.3).

The attributions are a global per-task importance of what the generator used, so the tests pin
four things: switching them on changes no other output, they are zero where the label cannot
depend on the column, they are non-zero for the drivers, and they line up with ``X`` after the
prior's column permutation. They say nothing about any model's attribution.
"""

from __future__ import annotations

import numpy as np
import pytest

from fintfm.prior.financial import (
    _ATTRIBUTION_DRIVER,
    _rotation_attribution,
    sample_financial_task,
)

SEEDS = range(12)


def _task(seed: int, **kw):
    """One 256-row, 40-column task, attributions on unless overridden."""
    kw.setdefault("return_attributions", True)
    return sample_financial_task(np.random.default_rng(seed), 256, max_features=40, **kw)


def _attr(t) -> np.ndarray:
    """The attribution vector of a task, as an array."""
    return np.array(t.meta["attribution"], dtype=float)


@pytest.mark.parametrize("seed", SEEDS)
def test_default_outputs_are_unchanged_by_the_option(seed: int) -> None:
    """X, y and the generator's end state are byte-identical with the option on or off."""
    rng_off, rng_on = np.random.default_rng(seed), np.random.default_rng(seed)
    off = sample_financial_task(rng_off, 256, max_features=40)
    on = sample_financial_task(rng_on, 256, max_features=40, return_attributions=True)
    assert off.meta is None
    assert off.X.tobytes() == on.X.tobytes()
    assert off.y.tobytes() == on.y.tobytes()
    assert off.is_categorical.tobytes() == on.is_categorical.tobytes()
    assert rng_off.random() == rng_on.random()  # no extra draw was consumed


def test_attribution_matches_columns_after_permutation_and_every_knob() -> None:
    """One value per column of X, aligned with ``meta["columns"]``, however X was built."""
    for seed in SEEDS:
        for kw in (
            {},
            {"n_noise_features": 3, "n_correlated_block_features": 2},
            {"gate_prob": 1.0, "sign_flip_prob": 0.5, "discretize_frac": 0.3},
            {"identity_shuffle": True},
        ):
            t = _task(seed, **kw)
            a = _attr(t)
            assert a.shape == (t.X.shape[1],) == (len(t.meta["columns"]),)
            assert np.all(np.isfinite(a)) and np.all(a >= 0.0)
            for name, v in zip(t.meta["columns"], a, strict=True):
                if name not in _ATTRIBUTION_DRIVER:
                    assert v == 0.0, name


def test_non_driver_columns_are_zero_and_drivers_are_not() -> None:
    """Noise, copies and derived ratios are zero; every exposed driver quantity is positive."""
    seen_driver = seen_zero = 0
    for seed in SEEDS:
        t = _task(seed, n_noise_features=2, n_correlated_block_features=2)
        for name, v in zip(t.meta["columns"], _attr(t), strict=True):
            if name in _ATTRIBUTION_DRIVER:
                assert v > 0.0, (seed, name)
                seen_driver += 1
            else:
                assert v == 0.0, (seed, name)
                seen_zero += 1
    assert seen_driver > 0 and seen_zero > 0


def test_identity_shuffle_decouples_account_driver_columns() -> None:
    """Under the shuffle only growth, payment delay and sector can still carry the label."""
    allowed = {"growth", "payment_delay", "sector"}
    for seed in SEEDS:
        t = _task(seed, identity_shuffle=True)
        for name, v in zip(t.meta["columns"], _attr(t), strict=True):
            if name not in allowed:
                assert v == 0.0, (seed, name)


def test_gated_tasks_say_the_gate_is_not_attributed() -> None:
    """A gated task's scope names the distress score and its gate column is zero."""
    t = _task(1, gate_prob=1.0)
    assert t.meta["gate"] is not None
    assert t.meta["attribution_scope"] == "distress_score"
    for name, v in zip(t.meta["columns"], _attr(t), strict=True):
        if name.startswith("gate:"):
            assert v == 0.0
    assert _task(1).meta["attribution_scope"] == "label"


def test_zero_weight_input_receives_zero_attribution() -> None:
    """The rotation helper gives exactly 0 to an input with zero weight, and |w|*E|dx| else."""
    rng = np.random.default_rng(0)
    x = {"a": rng.normal(size=500), "b": rng.normal(size=500)}
    out = _rotation_attribution(
        lambda v: 2.0 * v["a"] + 0.0 * v["b"], x, {"a": ["a"], "b": ["b"]}, scale=1.0
    )
    assert out["b"] == 0.0
    assert out["a"] == pytest.approx(2.0 * np.mean(np.abs(x["a"] - np.roll(x["a"], 250))))
    assert _rotation_attribution(lambda v: v["a"], {"a": np.ones(1)}, {"a": ["a"]}, 1.0) == {
        "a": 0.0
    }
