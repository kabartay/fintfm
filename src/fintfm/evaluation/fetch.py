"""Fetch licensed evaluation datasets that need credentials.

Kept separate from `datasets.py` on purpose: the loaders there must never reach the network
on their own, so that a benchmark run cannot silently acquire data mid-experiment. Fetching
is an explicit, human-initiated act.

**Licence obligations travel with the data**, and this module prints them rather than
assuming anyone reads the source. V4FinBench's *data* is CC BY 4.0 while its *code* is MIT —
a distinction worth respecting, since Google's TabFM ships Apache-2.0 code with
non-commercial weights and "the repo is permissive" proves nothing about what is inside it.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from fintfm.evaluation.datasets import _V4_HORIZON_FILES, CACHE_DIR, V4FINBENCH_KAGGLE

V4_ATTRIBUTION = (
    "V4FinBench, Tomczak et al., 'V4FinBench: Benchmarking Tabular Foundation Models, "
    "LLMs, and Standard Methods on Corporate Bankruptcy Prediction', arXiv:2605.10896. "
    "Data licensed CC BY 4.0."
)


def fetch_v4finbench(dest: Path | None = None) -> Path:
    """Download V4FinBench from Kaggle into the local cache.

    Args:
        dest: Target directory. Defaults to ``data/cache/v4finbench``.

    Returns:
        The directory the parquet files were placed in.

    Raises:
        SystemExit: With actionable instructions if ``kagglehub`` is missing or no Kaggle
            credentials are present. Credentials are never requested interactively and are
            never read or logged by this code.
    """
    target = dest or (CACHE_DIR / "v4finbench")
    try:
        import kagglehub
    except ImportError:
        raise SystemExit(
            "kagglehub is not installed. Run:  uv sync --extra kaggle"
        ) from None

    cred = Path.home() / ".kaggle" / "kaggle.json"
    if not cred.exists() and "KAGGLE_USERNAME" not in __import__("os").environ:
        raise SystemExit(
            f"No Kaggle credentials found at {cred}.\n"
            "Create a token at https://www.kaggle.com/settings (API -> Create New Token), "
            "then:\n"
            "  mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json "
            "&& chmod 600 ~/.kaggle/kaggle.json"
        )

    print(f"downloading {V4FINBENCH_KAGGLE} (~1.1M company-year rows, this may take a while)")
    downloaded = Path(kagglehub.dataset_download(V4FINBENCH_KAGGLE))
    target.mkdir(parents=True, exist_ok=True)

    copied = []
    for path in downloaded.rglob("*.parquet"):
        shutil.copy2(path, target / path.name)
        copied.append(path.name)
    if not copied:
        raise SystemExit(
            f"no parquet files found under {downloaded}; the dataset layout may have "
            f"changed. Expected: {', '.join(_V4_HORIZON_FILES)}"
        )

    missing = [f for f in _V4_HORIZON_FILES if not (target / f).exists()]
    print(f"copied {len(copied)} parquet files to {target}")
    if missing:
        print(f"WARNING: expected files still missing: {', '.join(missing)}")
    print(f"\nAttribution required on any published use:\n  {V4_ATTRIBUTION}")
    return target


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset", choices=["v4finbench"], help="which dataset to fetch")
    p.add_argument("--dest", type=str, default=None)
    args = p.parse_args()
    if args.dataset == "v4finbench":
        fetch_v4finbench(Path(args.dest) if args.dest else None)
    sys.exit(0)


if __name__ == "__main__":
    main()
