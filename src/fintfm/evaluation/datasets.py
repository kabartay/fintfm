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

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache"

_POLISH_URL = "https://archive.ics.uci.edu/static/public/365/polish+companies+bankruptcy+data.zip"


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
    """

    X: np.ndarray
    y: np.ndarray
    name: str
    horizon_years: int
    licence: str
    attribution: str

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
