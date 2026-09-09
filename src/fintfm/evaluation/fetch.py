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

    # Ask kagglehub how it resolves credentials rather than reimplementing the search.
    # Kaggle now offers two mechanisms and a hand-rolled check for the legacy file alone
    # would reject a perfectly valid modern API token.
    try:
        from kagglehub.config import get_kaggle_credentials

        has_credentials = get_kaggle_credentials() is not None
    except Exception:  # noqa: BLE001 - an unknown kagglehub layout must not block the fetch
        has_credentials = (Path.home() / ".kaggle" / "kaggle.json").exists()

    if not has_credentials:
        raise SystemExit(
            "No Kaggle credentials found. Two options at "
            "https://www.kaggle.com/settings :\n"
            "\n"
            "  A) API Tokens (Kaggle's recommendation) -- 'Generate New Token', then\n"
            "       export KAGGLE_API_TOKEN=<the token>\n"
            "     Needs kagglehub >= 0.4.1; this project pins a newer one.\n"
            "\n"
            "  B) Legacy API Credentials -- downloads kaggle.json, then\n"
            "       mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json\n"
            "       chmod 600 ~/.kaggle/kaggle.json\n"
            "\n"
            "Either works. Never paste a token into a chat or commit it."
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
