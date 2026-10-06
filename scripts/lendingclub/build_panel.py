"""Build a real LendingClub panel for FinTFM-R (task C1, D17's third real source).

What this produces
-------------------

One parquet file, one row per loan, columns:

- Numeric origination-time features (`revenue`, `dti_n`, `loan_amnt`, `fico_n`,
  `emp_length_years`) plus one-hot indicators derived from `purpose` (13 categories) and
  `home_ownership_n` (4 categories) -- every column emitted is already a float, matching
  `fintfm.prior.real_edgar`/`fintfm.prior.real_mortgage`'s all-numeric, `is_categorical=False`
  precedent rather than adding an untested categorical-feature path.
- `loan_id`, `label`: bookkeeping, excluded from features by
  `fintfm.prior.real_lendingclub.feature_columns`.
- `label`: the source's own `Default` column, unchanged.

One file, already application-time-only
-----------------------------------------

Source: "Lending Club loan dataset for granting models" (Ariza-Garzon, Sanz-Guerrero, Arroyo
Gallardo, Universidad Complutense de Madrid), Zenodo record 11295916, CC-BY-4.0. 1,347,681
loans, 2007-2018. Unlike the raw Kaggle LendingClub dump, this release is **already filtered
to variables available at loan application time**, excluding post-approval fields (interest
rate, grade, payment history) that would leak the outcome -- the exact trap a naive
concatenation of LendingClub's raw columns would fall into. Absent from TabArena-v0.1, but
present in BeyondArena as `lending_club_1m` -- training on this panel forfeits clean evaluation
on that one BeyondArena task (`docs/research/data_candidates.csv`).

`addr_state`, `zip_code` (geography) and `title`/`desc` (free text) are dropped: high
cardinality or unstructured, no clear modelling value, same reasoning
`scripts/freddie_mac/build_panel.py` used to drop Property State/Postal Code/Seller Name.

This script reads real data to disk. The output file is covered by `.gitignore`
(`data/lendingclub/`) and must never be committed, matching this project's convention for
trained weights -- see `CLAUDE.md`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

#: Final feature columns this module emits, after `purpose`/`home_ownership_n` are turned
#: into one-hot indicators. Order is fixed so two panels built from the same source file
#: expose the same columns in the same positions.
FEATURE_COLUMNS: tuple[str, ...] = (
    "revenue", "dti_n", "loan_amnt", "fico_n", "emp_length_years",
    "purpose_debt_consolidation", "purpose_small_business", "purpose_home_improvement",
    "purpose_major_purchase", "purpose_credit_card", "purpose_other", "purpose_house",
    "purpose_vacation", "purpose_car", "purpose_medical", "purpose_moving",
    "purpose_renewable_energy", "purpose_wedding",
    "home_ownership_mortgage", "home_ownership_rent", "home_ownership_own",
    "home_ownership_other",
)

_PURPOSE_CATEGORIES = (
    "debt_consolidation", "small_business", "home_improvement", "major_purchase",
    "credit_card", "other", "house", "vacation", "car", "medical", "moving",
    "renewable_energy", "wedding",
)

_HOME_OWNERSHIP_CATEGORIES = ("MORTGAGE", "RENT", "OWN", "OTHER")


def _emp_length_years(series: pd.Series) -> pd.Series:
    """Parse LendingClub's `emp_length` strings into a numeric years-of-employment feature.

    Args:
        series: Raw strings like ``"10+ years"``, ``"< 1 year"``, ``"3 years"``, or ``"NI"``
            (no information -- the source's own missingness marker).

    Returns:
        A float64 series: ``10`` for ``"10+ years"``, ``0`` for ``"< 1 year"``, the parsed
        integer for ``"N years"``/``"N year"``, and NaN for ``"NI"`` -- true missingness,
        matching every other real-data source in this codebase.
    """
    out = series.str.extract(r"(\d+)", expand=False).astype(float)
    out = out.mask(series == "< 1 year", 0.0)
    return out.mask(series == "NI")


def build_panel(source_csv: Path) -> pd.DataFrame:
    """Parse the LendingClub granting-model CSV into a feature-and-label panel.

    Args:
        source_csv: Path to `LC_loans_granting_model_dataset.csv` (Zenodo 11295916).

    Returns:
        One row per loan: `loan_id`, every column in `FEATURE_COLUMNS`, and `label`.
    """
    raw = pd.read_csv(
        source_csv,
        usecols=["id", "revenue", "dti_n", "loan_amnt", "fico_n", "emp_length", "purpose",
                  "home_ownership_n", "Default"],
        dtype={"purpose": "str", "home_ownership_n": "str", "emp_length": "str"},
    )
    out = pd.DataFrame(index=raw.index)
    out["loan_id"] = raw["id"]
    out["revenue"] = raw["revenue"]
    out["dti_n"] = raw["dti_n"]
    out["loan_amnt"] = raw["loan_amnt"].astype(np.float64)
    out["fico_n"] = raw["fico_n"]
    out["emp_length_years"] = _emp_length_years(raw["emp_length"])
    for cat in _PURPOSE_CATEGORIES:
        out[f"purpose_{cat}"] = (raw["purpose"] == cat).astype(np.float64)
    for cat, name in zip(
        _HOME_OWNERSHIP_CATEGORIES, ("mortgage", "rent", "own", "other"), strict=True
    ):
        out[f"home_ownership_{name}"] = (raw["home_ownership_n"] == cat).astype(np.float64)
    out["label"] = raw["Default"].astype(np.int64)
    return out[["loan_id", *FEATURE_COLUMNS, "label"]].reset_index(drop=True)


def main() -> None:
    """Entry point. See module docstring for what this writes and why."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--source-csv", required=True,
        help="LC_loans_granting_model_dataset.csv, from Zenodo record 11295916",
    )
    p.add_argument("--out", default="data/lendingclub/panel.parquet")
    args = p.parse_args()

    panel = build_panel(Path(args.source_csv))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(out)
    print(f"wrote {len(panel)} rows, {panel['label'].mean():.4%} positive, to {out}", flush=True)


if __name__ == "__main__":
    main()
