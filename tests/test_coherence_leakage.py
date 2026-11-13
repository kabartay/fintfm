"""Leakage guard for ``term_structure_coherence`` (openspec 49.8), on generated tables."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.model_selection import train_test_split

from fintfm.evaluation.datasets import CreditDataset
from fintfm.experiments import prior_ablation as pa

N, D = 200, 4


def _ds(h: int, seed: int) -> CreditDataset:
    """Build a small random table for horizon ``h``.

    Args:
        h: Horizon label.
        seed: RNG seed.

    Returns:
        A 200-row dataset with 10% positives.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(N, D)).astype(np.float32)
    y = (np.arange(N) % 10 == 0).astype(np.int64)
    return CreditDataset(X=X, y=y, name=f"fake-{h}y", horizon_years=h, licence="x", attribution="x")


@pytest.fixture
def spy(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Patch the loader and classifier; record each fitted context.

    Args:
        monkeypatch: pytest fixture.

    Returns:
        The fake data by horizon and the list of contexts the classifier was fitted on.
    """
    data = {h: _ds(h, h) for h in (1, 2, 3)}
    seen: list[np.ndarray] = []

    class Fake:
        """Stand-in classifier that records its context."""

        def __init__(self, *a: object, **k: object) -> None:
            pass

        def fit(self, X: np.ndarray, y: np.ndarray) -> Fake:
            seen.append(np.asarray(X).copy())
            return self

        def predict_proba(self, X: np.ndarray) -> np.ndarray:
            p = np.full(len(X), 0.1)
            return np.stack([1 - p, p], axis=1)

    monkeypatch.setattr(pa, "load_polish_bankruptcy", lambda h: data[h])
    monkeypatch.setattr(pa, "FinancialTFMClassifier", Fake)
    return {"data": data, "seen": seen}


def _keys(X: np.ndarray) -> set[bytes]:
    return {r.tobytes() for r in np.ascontiguousarray(X)}


def _query_rows(data: dict) -> np.ndarray:
    d = data[3]
    return train_test_split(d.X, d.y, test_size=0.3, random_state=0, stratify=d.y)[1]


def test_no_query_row_in_any_context(spy: dict) -> None:
    # plant a query row inside the horizon-2 file: it must be dropped from that context
    q = _query_rows(spy["data"])
    spy["data"][2].X[0] = q[0]
    pa.term_structure_coherence("x.pt", horizons=(1, 2, 3))
    qk = _keys(q)
    assert len(spy["seen"]) == 3
    for ctx in spy["seen"]:
        assert not (_keys(ctx) & qk)
    assert len(spy["seen"][1]) == N - 1
    assert len(spy["seen"][2]) == N - len(qk)


def test_legacy_option_reproduces_whole_panel_context(spy: dict) -> None:
    rec = pa.term_structure_coherence("x.pt", horizons=(1, 2, 3), legacy_full_panel_context=True)
    assert rec["context_selection"] == "legacy_full_panel"
    assert [len(c) for c in spy["seen"]] == [N, N, N]
    assert _keys(spy["seen"][2]) >= _keys(_query_rows(spy["data"]))


def test_default_record_names_the_selection(spy: dict) -> None:
    rec = pa.term_structure_coherence("x.pt", horizons=(1, 2, 3))
    assert rec["context_selection"] == "query_rows_excluded"
