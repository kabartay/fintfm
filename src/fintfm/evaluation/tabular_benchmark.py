"""Loader for the licence-cleared classification subset of ``inria-soda/tabular-benchmark``.

Grinsztajn, Oyallon and Varoquaux (NeurIPS 2022) published 23 classification datasets in one
Hugging Face collection that declares no licence. Task 37.1 of
``openspec/changes/diverse-eval-suite`` resolved each one from its primary source
(``docs/research/tabular_benchmark_licences.md``): ten permit commercial use, thirteen could not
be established and are excluded. This module loads exactly those ten and refuses the rest, so a
file that is merely downloadable can never enter the suite by accident.

**Evaluation only. Nothing here may be used for pretraining** (``datasets.py`` carries the
reason). The downloaded CSVs live in the gitignored ``data/cache/`` and are never committed.

The published row and feature counts below are copied from the "Source Data" tables of the
collection card,
https://huggingface.co/datasets/inria-soda/tabular-benchmark/raw/main/README.md (fetched
2026-11-15). The card's ``n_features`` excludes the target, so a file has ``n_features + 1``
columns.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from fintfm.evaluation.datasets import CACHE_DIR, _download

TABULAR_BENCHMARK_CACHE = CACHE_DIR / "tabular_benchmark"
_HF_BASE = "https://huggingface.co/datasets/inria-soda/tabular-benchmark/resolve/main"
_CARD_URL = "https://huggingface.co/datasets/inria-soda/tabular-benchmark"
_CITATION = (
    "Grinsztajn, Oyallon and Varoquaux (2022), Why do tree-based models still outperform deep "
    "learning on typical tabular data? NeurIPS Datasets and Benchmarks. "
    f"Collection: {_CARD_URL}"
)
_UCI_ATTRIBUTION = "Dataset from the UCI Machine Learning Repository, CC BY 4.0. "
_CC_BY = "CC-BY-4.0"
_OGL = "OGL-UK-3.0"


@dataclass(frozen=True)
class TabularSpec:
    """Static description of one cleared dataset.

    Attributes:
        key: Identifier ``"<folder>/<name>"``, e.g. ``"clf_num/covertype"``; also the path of
            the file inside the Hugging Face repository (without ``.csv``).
        n_samples: Published row count (collection card, "Source Data" table).
        n_features: Published feature count, target excluded (same table).
        licence: SPDX-style identifier of the licence at the primary source.
        source_url: Primary source of the licence (``tabular_benchmark_licences.md``).
        attribution: Text that must accompany any published use.
    """

    key: str
    n_samples: int
    n_features: int
    licence: str
    source_url: str
    attribution: str

    @property
    def url(self) -> str:
        """Hugging Face download URL of the CSV."""
        return f"{_HF_BASE}/{self.key}.csv"

    @property
    def cache_path(self) -> Path:
        """Where the CSV is cached under ``data/cache/``."""
        return TABULAR_BENCHMARK_CACHE / f"{self.key}.csv"


def _uci(key: str, rows: int, feats: int, page: str, name: str) -> TabularSpec:
    """Build a spec for a dataset whose licence statement is a UCI CC BY 4.0 page."""
    return TabularSpec(
        key, rows, feats, _CC_BY, f"https://archive.ics.uci.edu/dataset/{page}",
        f"{_UCI_ATTRIBUTION}{name}. {_CITATION}",
    )  # fmt: skip


_SPECS = (
    _uci("clf_num/covertype", 566602, 10, "31/covertype", "Covertype"),
    _uci("clf_num/MagicTelescope", 13376, 10, "159/magic+gamma+telescope", "MAGIC Gamma Telescope"),
    _uci("clf_num/bank-marketing", 10578, 7, "222/bank+marketing", "Bank Marketing"),
    _uci("clf_num/MiniBooNE", 72998, 50, "199/miniboone+particle+identification", "MiniBooNE"),
    _uci(
        "clf_num/default-of-credit-card-clients", 13272, 20,
        "350/default+of+credit+card+clients", "Default of Credit Card Clients",
    ),
    _uci("clf_num/Higgs", 940160, 24, "280/higgs", "HIGGS"),
    _uci(
        "clf_num/Diabetes130US", 71090, 7,
        "296/diabetes+130-us+hospitals+for+years+1999-2008", "Diabetes 130-US Hospitals",
    ),
    _uci("clf_cat/covertype", 423680, 54, "31/covertype", "Covertype"),
    _uci(
        "clf_cat/default-of-credit-card-clients", 13272, 21,
        "350/default+of+credit+card+clients", "Default of Credit Card Clients",
    ),
    TabularSpec(
        "clf_cat/road-safety", 111762, 32, _OGL,
        "https://data.gov.uk/dataset/cb7ae6f0-4be6-4935-9277-47e5ce24a11f/road-safety-data",
        "Contains public sector information licensed under the Open Government Licence v3.0 "
        f"(UK Department for Transport road safety data). {_CITATION}",
    ),
)  # fmt: skip

#: The ten datasets cleared for commercial use, keyed by ``"<folder>/<name>"``.
CLEARED: dict[str, TabularSpec] = {spec.key: spec for spec in _SPECS}


@dataclass(frozen=True)
class TabularDataset:
    """One loaded classification dataset.

    Attributes:
        X: Feature frame ``(n_samples, n_features)``; numeric columns keep their dtype and
            any non-numeric column is returned as ``category``.
        y: Integer class labels ``(n_samples,)``, classes ordered by sorted label value.
        name: The cleared key, e.g. ``"clf_num/covertype"``.
        licence: SPDX-style licence identifier of the source data.
        source_url: Primary source of that licence.
        attribution: Text that must accompany any published use of this data.
    """

    X: pd.DataFrame
    y: np.ndarray
    name: str
    licence: str
    source_url: str
    attribution: str


def load_tabular_benchmark(key: str) -> TabularDataset:
    """Load one licence-cleared classification dataset, downloading it on first use.

    Args:
        key: ``"<folder>/<name>"`` as in :data:`CLEARED`, e.g. ``"clf_cat/road-safety"``.

    Returns:
        A :class:`TabularDataset`. The target is the last CSV column.

    Raises:
        ValueError: If ``key`` is not one of the ten cleared datasets. This is the licence
            gate: the 13 excluded files are downloadable but their terms are not established,
            so they are refused here rather than loaded on request.
        RuntimeError: If the download fails.
    """
    if key not in CLEARED:
        raise ValueError(
            f"{key!r} is not licence-cleared for commercial use; see "
            f"docs/research/tabular_benchmark_licences.md. Cleared: {sorted(CLEARED)}"
        )
    spec = CLEARED[key]
    frame = pd.read_csv(_download(spec.url, spec.cache_path))
    target = frame.iloc[:, -1]
    X = frame.iloc[:, :-1].copy()
    for col in X.columns[~X.dtypes.map(pd.api.types.is_numeric_dtype)]:
        X[col] = X[col].astype("category")
    classes = sorted(target.unique())
    y = target.map({c: i for i, c in enumerate(classes)}).to_numpy(dtype=np.int64)
    return TabularDataset(X, y, key, spec.licence, spec.source_url, spec.attribution)
