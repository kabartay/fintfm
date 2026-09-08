"""Loaders for real credit-risk datasets used in evaluation only.

**Nothing here may be used for pretraining.** The model's auditability rests on the
pretraining corpus being entirely synthetic (see `docs/FINDINGS.md` §1): a benchmark number
from a model that never saw real data cannot be inflated by memorisation. These loaders exist
so that claim can be tested, not weakened.

Every dataset carries its licence and required attribution. Check both before adding one.
"""

from __future__ import annotations

import io
import shutil
import subprocess
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import numpy as np

def _repo_root() -> Path:
    """Locate the repository root by walking up to the directory holding ``pyproject.toml``.

    Counting ``parents[n]`` breaks silently when a module moves. It did: the package
    reorganisation shifted this file one level deeper, so ``parents[2]`` became ``src/`` and
    every dataset was re-downloaded into ``src/data/cache`` — 13 MB duplicated, with nothing
    failing to signal it. Anchoring on a marker file cannot break that way.
    """
    for candidate in (Path(__file__).resolve(), *Path(__file__).resolve().parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    # installed as a package with no repo around it: fall back to the working directory
    return Path.cwd()


CACHE_DIR = _repo_root() / "data" / "cache"

_POLISH_URL = "https://archive.ics.uci.edu/static/public/365/polish+companies+bankruptcy+data.zip"
_TAIWAN_URL = (
    "https://archive.ics.uci.edu/static/public/572/taiwanese+bankruptcy+prediction.zip"
)


@dataclass(frozen=True)
class CreditDataset:
    """A real corporate-default dataset.

    Attributes:
        X: Financial features ``(n_companies, n_features)``, float32, NaN for missing.
        y: Binary default label ``(n_companies,)``; 1 = bankrupt within the horizon.
        name: Short identifier used in benchmark output.
        horizon_years: Forecast horizon the label refers to.
        licence: SPDX-style licence identifier of the source data.
        attribution: Text that must accompany any published use of this data.
        has_period_labels: Whether rows carry an observation date or period. **False for
            every currently loadable panel** — both UCI sets are anonymised cross-sectional
            ratio tables (``docs/FINDINGS.md`` §7, §8). Time-based evaluation must refuse to
            run on a dataset where this is False rather than silently falling back to a
            random split, which would look like out-of-time validation and not be.
    """

    X: np.ndarray
    y: np.ndarray
    name: str
    horizon_years: int
    licence: str
    attribution: str
    has_period_labels: bool = False

    @property
    def default_rate(self) -> float:
        return float(self.y.mean())


def _download(url: str, dest: Path) -> Path:
    """Fetch ``url`` to ``dest`` unless already cached.

    Falls back to ``curl`` when the standard library cannot verify the TLS chain. A
    python.org macOS build ships without root certificates unless ``Install Certificates``
    has been run, so ``urlopen`` raises ``SSLCertVerificationError`` on a machine where
    ``curl`` fetches the same URL happily. Preferring an explicit fallback to an extra
    dependency keeps the licence surface of this project unchanged.

    Args:
        url: Source URL.
        dest: Local cache path.

    Returns:
        ``dest``.

    Raises:
        RuntimeError: If neither urllib nor curl can retrieve the file.
    """
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urlopen(url) as response:
            dest.write_bytes(response.read())
        return dest
    except (URLError, OSError) as exc:
        curl = shutil.which("curl")
        if curl is None:
            raise RuntimeError(f"could not download {url}: {exc}; curl unavailable") from exc
        result = subprocess.run(
            [curl, "-sSfL", url, "-o", str(dest)], capture_output=True, text=True, check=False
        )
        if result.returncode != 0 or not dest.exists():
            raise RuntimeError(f"could not download {url}: {result.stderr.strip()}") from exc
        return dest


def load_polish_bankruptcy(horizon_years: int = 3) -> CreditDataset:
    """Load the UCI Polish companies bankruptcy dataset.

    Corporate insolvency prediction from 64 financial ratios (profitability, liquidity,
    leverage, turnover). Bankrupt firms observed 2000-2012, operating firms 2007-2013.
    Severely imbalanced, which is representative of the real task and is why the benchmark
    reports AUC rather than accuracy.

    Args:
        horizon_years: Which forecast horizon to load, 1 to 5. The label indicates
            bankruptcy within this many years of the reported financials.

    Returns:
        A :class:`CreditDataset`.

    Raises:
        ValueError: If ``horizon_years`` is outside 1-5.
    """
    if not 1 <= horizon_years <= 5:
        raise ValueError(f"horizon_years must be 1-5, got {horizon_years}")
    from scipy.io import arff  # imported lazily; only evaluation needs it

    archive = _download(_POLISH_URL, CACHE_DIR / "polish_bankruptcy.zip")
    with zipfile.ZipFile(archive) as z:
        raw = z.read(f"{horizon_years}year.arff").decode("utf-8", errors="replace")
    records, _meta = arff.loadarff(io.StringIO(raw))
    table = np.array(records.tolist(), dtype=object)
    X = table[:, :-1].astype(np.float64).astype(np.float32)
    y = np.array([int(v) for v in table[:, -1]], dtype=np.int64)
    return CreditDataset(
        X=X,
        y=y,
        name=f"polish-bankruptcy-{horizon_years}y",
        horizon_years=horizon_years,
        licence="CC-BY-4.0",
        attribution=(
            "Polish companies bankruptcy data, Zieba, Tomczak & Tomczak, "
            "UCI Machine Learning Repository (CC BY 4.0). "
            "https://doi.org/10.24432/C5F600"
        ),
    )


def load_taiwan_bankruptcy() -> CreditDataset:
    """Load the UCI Taiwanese bankruptcy dataset.

    A second, independent panel so results do not rest on one economy, one accounting regime
    and one crisis. Taiwan Economic Journal data, 1999-2009, 95 financial ratios, no missing
    values, 3.23% bankruptcy rate.

    Like the Polish set it carries **no dates and no company identifiers** — verified by
    column inspection, not assumed — so it supports discrimination and calibration work but
    nothing temporal. See ``docs/FINDINGS.md`` §8.

    Returns:
        A :class:`CreditDataset` with ``has_period_labels=False``.
    """
    import pandas as pd

    archive = _download(_TAIWAN_URL, CACHE_DIR / "taiwan_bankruptcy.zip")
    with zipfile.ZipFile(archive) as z:
        csv_name = next(n for n in z.namelist() if n.endswith(".csv"))
        frame = pd.read_csv(io.BytesIO(z.read(csv_name)))
    # the target is the first column, named "Bankrupt?"; every other column is a ratio
    y = frame.iloc[:, 0].to_numpy(dtype=np.int64)
    X = frame.iloc[:, 1:].to_numpy(dtype=np.float32)
    return CreditDataset(
        X=X,
        y=y,
        name="taiwan-bankruptcy",
        horizon_years=1,
        licence="CC-BY-4.0",
        attribution=(
            "Taiwanese bankruptcy prediction, Liang, Lu, Tsai & Shih, "
            "UCI Machine Learning Repository (CC BY 4.0). "
            "https://doi.org/10.24432/C5004D"
        ),
        has_period_labels=False,
    )


#: Kaggle dataset slug for V4FinBench, per its repository's ``DATA_LICENSE.md`` (CC BY 4.0).
V4FINBENCH_KAGGLE = "sebastiantomczak10/v4-group-corporate-bankruptcy"

#: Horizon files, in order. ``h1`` is the paper's ``h=0`` (current-year distress) through
#: ``h6`` = ``h=5`` (five years ahead), so index ``k`` is distress by horizon ``k``.
_V4_HORIZON_FILES = tuple(f"company_years_h{i}.parquet" for i in range(1, 7))


@dataclass(frozen=True)
class SurvivalDataset:
    """A real corporate-default panel carrying a default **period**, not just a label.

    This is the object the hazard head needs (``docs/FINDINGS.md`` §20) and the reason
    V4FinBench matters: the UCI panels have no firm identifiers (§7), so no per-firm hazard
    path can be scored against them at all.

    Attributes:
        X: Features ``(n, n_features)``, float32, NaN for missing.
        y: Binary "distressed within the horizon grid" label ``(n,)``.
        period: Zero-based first horizon at which distress is observed, or
            :data:`fintfm.modeling.hazard.CENSORED` (-1) if never within the grid.
        n_horizons: Length of the horizon grid.
        year: Reporting year per row, enabling **time-based** splits — the thing no other
            panel we hold supports.
        name: Short identifier for benchmark output.
        licence: SPDX identifier of the source data.
        attribution: Text that must accompany any published use.
        feature_names: Column names, in order.
    """

    X: np.ndarray
    y: np.ndarray
    period: np.ndarray
    n_horizons: int
    year: np.ndarray | None
    name: str
    licence: str
    attribution: str
    feature_names: tuple[str, ...] = ()

    @property
    def default_rate(self) -> float:
        return float(self.y.mean())

    @property
    def has_period_labels(self) -> bool:
        """True: this is the panel that supports out-of-time validation."""
        return self.year is not None


def load_v4finbench(
    root: Path | str | None = None, id_cols: tuple[str, ...] = ("company_id", "year")
) -> SurvivalDataset:
    """Load V4FinBench and derive a default period from its six horizon files.

    1,106,879 company-year observations over the Visegrád economies, 2006-2021, 131
    features, positive rates 0.19-0.36%. Code MIT, **data CC BY 4.0** per the repository's
    separate ``DATA_LICENSE.md`` — verified, not inferred from the code licence
    (``docs/FINDINGS.md`` §8).

    The period is derived rather than read: horizon file ``k`` carries "distressed by horizon
    ``k``", so the first ``k`` whose label is 1 is the default period, and a row that is 0
    everywhere is censored. Labels are cumulative by construction, so this derivation also
    **checks** them: a row that is 1 at horizon 2 and 0 at horizon 4 is inconsistent and is
    reported rather than silently coerced.

    Args:
        root: Directory holding the parquet files. Defaults to ``data/cache/v4finbench``.
        id_cols: Columns identifying a company-year, excluded from features. Names are
            checked against the actual columns and a clear error names what was found.

    Returns:
        A :class:`SurvivalDataset`.

    Raises:
        FileNotFoundError: If the files are absent, with instructions for obtaining them.
            The data is on Kaggle and needs credentials, which only the repository owner can
            supply; nothing here attempts to fetch it silently.
    """
    import pandas as pd

    base = Path(root) if root is not None else CACHE_DIR / "v4finbench"
    missing = [f for f in _V4_HORIZON_FILES if not (base / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"V4FinBench not found under {base}. Missing: {', '.join(missing)}.\n"
            f"Obtain it one of two ways, both needing a Kaggle account:\n"
            f"  1. pip install kagglehub, then\n"
            f"     python -c \"import kagglehub; "
            f"print(kagglehub.dataset_download('{V4FINBENCH_KAGGLE}'))\"\n"
            f"     and copy the parquet files into {base}\n"
            f"  2. download manually from "
            f"https://www.kaggle.com/datasets/{V4FINBENCH_KAGGLE}\n"
            f"Data is CC BY 4.0; attribution is required and is carried on the returned "
            f"dataset."
        )

    frames = [pd.read_parquet(base / f) for f in _V4_HORIZON_FILES]
    label_col = next(
        (c for c in frames[0].columns if c.lower() in {"label", "target", "distress", "y"}),
        None,
    )
    if label_col is None:
        raise ValueError(
            f"no label column found in {_V4_HORIZON_FILES[0]}; columns are "
            f"{list(frames[0].columns)[:20]}"
        )
    keys = [c for c in id_cols if c in frames[0].columns]
    if not keys:
        raise ValueError(
            f"none of {id_cols} present; columns are {list(frames[0].columns)[:20]}"
        )

    base_df = frames[0]
    labels = np.stack(
        [f.sort_values(keys)[label_col].to_numpy().astype(np.int64) for f in frames], axis=1
    )
    base_df = base_df.sort_values(keys).reset_index(drop=True)

    # cumulative labels must be non-decreasing across horizons; report, never coerce
    inconsistent = int((np.diff(labels, axis=1) < 0).any(axis=1).sum())
    if inconsistent:
        print(
            f"    WARNING: {inconsistent} rows have non-cumulative horizon labels "
            f"({inconsistent / len(labels):.3%}); periods derived from the first positive"
        )

    any_default = labels.any(axis=1)
    period = np.where(any_default, labels.argmax(axis=1), -1).astype(np.int64)

    feature_cols = [c for c in base_df.columns if c not in {*keys, label_col}]
    X = base_df[feature_cols].to_numpy(dtype=np.float32)
    year = base_df["year"].to_numpy() if "year" in base_df.columns else None

    return SurvivalDataset(
        X=X,
        y=any_default.astype(np.int64),
        period=period,
        n_horizons=len(_V4_HORIZON_FILES),
        year=year,
        name="v4finbench",
        licence="CC-BY-4.0",
        attribution=(
            "V4FinBench, Tomczak et al., 'V4FinBench: Benchmarking Tabular Foundation "
            "Models, LLMs, and Standard Methods on Corporate Bankruptcy Prediction', "
            "arXiv:2605.10896. Data CC BY 4.0."
        ),
        feature_names=tuple(feature_cols),
    )
