"""Tests for the licence-cleared tabular-benchmark loader (task 37.2).

The count test reads the cached CSVs and skips when they are absent, which is always the case
in CI. The published counts come from the collection card's "Source Data" tables
(https://huggingface.co/datasets/inria-soda/tabular-benchmark/raw/main/README.md); the card's
``n_features`` excludes the target, so each file has ``n_features + 1`` columns.
"""

from __future__ import annotations

import pandas as pd
import pytest

from fintfm.evaluation.tabular_benchmark import CLEARED, load_tabular_benchmark

EXCLUDED_EXAMPLES = ["clf_num/pol", "clf_cat/albert", "clf_cat/compas-two-years", "clf_num/jannis"]


def test_exactly_ten_cleared() -> None:
    """The cleared list is the ten of docs/research/tabular_benchmark_licences.md."""
    assert len(CLEARED) == 10
    assert sum(k.startswith("clf_num/") for k in CLEARED) == 7
    assert sum(k.startswith("clf_cat/") for k in CLEARED) == 3


def test_licence_and_source_fields() -> None:
    """Every spec carries a licence, an https source and attribution text."""
    for spec in CLEARED.values():
        assert spec.licence in {"CC-BY-4.0", "OGL-UK-3.0"}
        assert spec.source_url.startswith("https://")
        assert "inria-soda/tabular-benchmark" in spec.attribution
        assert spec.url.endswith(f"{spec.key}.csv")
    assert CLEARED["clf_cat/road-safety"].licence == "OGL-UK-3.0"
    assert CLEARED["clf_num/MiniBooNE"].licence == "CC-BY-4.0"


@pytest.mark.parametrize("key", EXCLUDED_EXAMPLES + ["nonsense", "clf_num/covertype.csv"])
def test_refuses_uncleared(key: str) -> None:
    """Excluded or unknown datasets are refused before any network access."""
    with pytest.raises(ValueError, match="not licence-cleared"):
        load_tabular_benchmark(key)


@pytest.mark.parametrize("key", sorted(CLEARED))
def test_counts_match_published(key: str) -> None:
    """Row and column counts equal the collection card's published description."""
    spec = CLEARED[key]
    if not spec.cache_path.exists():
        pytest.skip(f"{spec.cache_path} not cached")
    ds = load_tabular_benchmark(key)
    assert ds.X.shape == (spec.n_samples, spec.n_features)
    assert len(ds.y) == spec.n_samples
    assert pd.read_csv(spec.cache_path, nrows=1).shape[1] == spec.n_features + 1
    assert set(ds.y) == set(range(len(set(ds.y))))
