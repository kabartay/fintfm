"""LendingClub is retired as a training source (AU0.1, DECISIONS D19).

It is BeyondArena's `lending_club_1m`, and this project never trains on a TabArena or
BeyondArena dataset. These tests pin the refusal at every entry point: the sampler itself, the
sequential `p_real_lendingclub` draw and a `source_weights` entry. Every test uses a small
in-memory fixture panel, never the real downloaded data.
"""

import numpy as np
import pandas as pd
import pytest

from fintfm.prior.mixture import PriorConfig, sample_batch, sample_task
from fintfm.prior.real_lendingclub import (
    RETIRED,
    RetiredSourceError,
    feature_columns,
    sample_real_lendingclub_task,
)


def _fixture_panel(n: int = 200, n_features: int = 10, seed: int = 0) -> pd.DataFrame:
    """A small, fully synthetic stand-in for a real `build_panel.py` output."""
    rng = np.random.default_rng(seed)
    data = {f"feat_{i}": rng.normal(size=n) for i in range(n_features)}
    data["loan_id"] = np.arange(n)
    data["label"] = (rng.random(n) < 0.2).astype(np.int64)
    return pd.DataFrame(data)


def test_feature_columns_excludes_bookkeeping():
    cols = feature_columns(_fixture_panel())
    assert "label" not in cols and "loan_id" not in cols
    assert len(cols) == 10


def test_the_sampler_refuses_even_a_valid_panel():
    with pytest.raises(RetiredSourceError, match="BeyondArena"):
        sample_real_lendingclub_task(
            np.random.default_rng(0), n_rows=50, max_features=20, panel=_fixture_panel()
        )
    assert issubclass(RetiredSourceError, ValueError)


def test_p_real_lendingclub_zero_never_touches_the_panel_path():
    cfg = PriorConfig(
        p_real_lendingclub=0.0, real_lendingclub_panel_path="/does/not/exist.parquet"
    )
    rng = np.random.default_rng(6)
    for _ in range(20):
        sample_task(rng, cfg, n_rows=32)  # must not raise


def test_a_positive_p_real_lendingclub_is_refused_before_any_draw(tmp_path):
    path = tmp_path / "panel.parquet"
    _fixture_panel(n=100, n_features=6).to_parquet(path)
    cfg = PriorConfig(
        p_real_lendingclub=0.1, max_features=20, real_lendingclub_panel_path=str(path), n_rows=40
    )
    with pytest.raises(RetiredSourceError, match="lending_club_1m"):
        sample_batch(np.random.default_rng(7), cfg, batch_size=4)
    with pytest.raises(RetiredSourceError):
        sample_task(np.random.default_rng(7), PriorConfig(
            p_real_lendingclub=1.0, max_features=20, real_lendingclub_panel_path=str(path),
            n_rows=40,
        ), n_rows=40)


def test_a_real_lendingclub_source_weight_is_refused():
    cfg = PriorConfig(source_weights={"financial": 1.0, "real_lendingclub": 0.5})
    with pytest.raises(RetiredSourceError, match=RETIRED[:30]):
        sample_batch(np.random.default_rng(8), cfg, batch_size=2)
