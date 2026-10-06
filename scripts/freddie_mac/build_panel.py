"""Build a real Freddie Mac Single-Family Loan-Level panel for FinTFM-R (task B3).

What this produces
-------------------

One parquet file, one row per loan, columns:

- A fixed set of origination-time features derived from the Standard Dataset's origination
  file (`ORIGINATION_FEATURE_COLUMNS` below): numeric fields kept as floats, and a handful of
  coded/lettered fields turned into one-hot indicator columns at build time (see
  `_parse_origination`) so every value this module emits is already a float, matching
  `fintfm.prior.real_edgar`'s schema (and its `is_categorical=False` precedent) rather than
  adding an untested categorical-feature path.
- `loan_id`, `label`: bookkeeping, excluded from features by
  `fintfm.prior.real_mortgage.feature_columns`.
- `label`: 1 if the loan's monthly performance history ever recorded a Zero Balance Code of
  03 (Short Sale or Charge Off) or 09 (REO Disposition) -- the two codes that represent a
  realised credit loss, as distinct from a voluntary payoff (01), a third-party or whole-loan
  sale (02, 15, 16) or a pre-credit-event repurchase for an underwriting defect (96). Mirrors
  `scripts/edgar/build_panel.py`'s "one decisive real-world event, not a proxy" design.

Two files per vintage, joined by Loan Sequence Number
------------------------------------------------------

`sample_orig_YYYY.txt` (origination, 31 pipe-delimited columns, no header) and
`sample_perf_YYYY.txt` (monthly performance, 35 columns). Column order for both was read
directly from Freddie Mac's own "Single-Family Loan-Level Dataset General User Guide" (Release
47, July 2026), which states the attributes are listed in file order, and cross-checked against
a real downloaded sample row field-by-field before being trusted -- not assumed from an older,
possibly-stale public layout. See `docs/results/FINDINGS.md` §153 for the licence-terms read
that applies to this data, and `docs/design/DECISIONS.md` D17 for why this source exists.

This script reads real data from disk. The output file is covered by `.gitignore`
(`data/mortgage/`) and must never be committed, matching this project's convention for trained
weights -- see `CLAUDE.md`.

One obligation that binds regardless of licence tier (§153): this script and everything
downstream of it must never attempt to re-identify an individual borrower. The features kept
below are origination-time loan/property attributes, not anything that singles out a person,
and `Property State`/`Postal Code`/`Seller Name`/`Servicer Name` are dropped entirely -- partly
for cardinality, partly because there is no modelling reason to carry them closer to that line
than the panel already needs to.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

#: Column names for the origination file, in file order (31 columns, verified against the
#: current user guide and a real sample row's field count -- see module docstring).
_ORIG_COLUMNS = (
    "credit_score", "first_payment_date", "first_time_homebuyer", "maturity_date", "msa",
    "mi_pct", "num_units", "occupancy_status", "original_cltv", "original_dti", "original_upb",
    "original_ltv", "original_interest_rate", "channel", "ppm_flag", "amortization_type",
    "property_state", "property_type", "postal_code", "loan_id", "loan_purpose",
    "original_loan_term", "num_borrowers", "seller_name", "super_conforming_flag",
    "pre_harp_loan_id", "special_eligibility_program", "harp_indicator",
    "property_valuation_method", "interest_only", "vantage_score",
)

#: Column names for the monthly performance file, in file order (35 columns).
_PERF_COLUMNS = (
    "loan_id", "period", "current_actual_upb", "current_delinquency_status", "loan_age",
    "remaining_months_to_maturity", "defect_settlement_date", "modification_flag",
    "zero_balance_code", "zero_balance_effective_date", "current_interest_rate",
    "current_non_interest_bearing_upb", "ddlpi", "mi_recoveries", "net_sales_proceeds",
    "non_mi_recoveries", "total_expenses", "legal_costs", "maintenance_costs",
    "taxes_and_insurance", "misc_expenses", "actual_loss", "cumulative_modification_costs",
    "interest_rate_step_indicator", "payment_deferral_flag", "eltv",
    "zero_balance_removal_upb", "delinquent_accrued_interest", "delinquency_due_to_disaster",
    "borrower_assistance_plan", "current_period_modification_costs",
    "current_interest_bearing_upb", "mi_cancellation_indicator", "servicer_name",
    "bankruptcy_cramdown_costs",
)

#: Zero Balance Codes that represent a realised credit loss (short sale/charge-off, REO
#: disposition) -- distinct from a voluntary payoff, a performing loan/whole-loan sale, or a
#: pre-credit-event defect repurchase. See module docstring.
_DISTRESS_ZERO_BALANCE_CODES = ("03", "09")

#: Final feature columns this module emits, after the origination file's lettered/coded
#: fields are turned into one-hot indicators. Order is fixed so two panels built from the same
#: vintage range expose the same columns in the same positions (mirrors
#: `fintfm.prior.real_edgar.feature_columns`'s stability requirement).
ORIGINATION_FEATURE_COLUMNS: tuple[str, ...] = (
    "credit_score", "first_time_homebuyer", "msa", "mi_pct", "num_units",
    "occ_primary", "occ_investment", "occ_second_home",
    "original_cltv", "original_dti", "original_upb", "original_ltv", "original_interest_rate",
    "chan_retail", "chan_broker", "chan_correspondent", "chan_third_party",
    "ppm_flag", "is_arm", "original_loan_term", "num_borrowers", "super_conforming_flag",
    "special_elig_home_possible", "special_elig_hfa_advantage", "special_elig_refi_possible",
    "harp_indicator", "property_valuation_method", "interest_only", "vantage_score",
    "loan_purpose_purchase", "loan_purpose_cashout_refi", "loan_purpose_nocashout_refi",
    "loan_purpose_refi_unspecified",
    "property_type_sf", "property_type_co", "property_type_pu", "property_type_cp",
    "property_type_mh",
)


def _na(series: pd.Series, sentinel: str) -> pd.Series:
    """Numeric-parse a column, turning Freddie Mac's own "Not Available" sentinel into NaN.

    Args:
        series: A raw string column from the origination file.
        sentinel: The exact string value this field's documentation names as "Not Available"
            (e.g. ``"9999"``, ``"999"``, ``"99"``) -- true missingness, not zero-filled,
            matching every other real-data source in this codebase.

    Returns:
        A float64 series with ``sentinel`` replaced by NaN.
    """
    out = pd.to_numeric(series, errors="coerce")
    return out.mask(series == sentinel)


def _one_hot(series: pd.Series, mapping: dict[str, str], unavailable: tuple[str, ...]) -> pd.DataFrame:
    """Turn one lettered/coded column into several 0/1 indicator columns.

    Args:
        series: The raw string column.
        mapping: Raw code -> output column name, for every value that is a real category.
        unavailable: Raw values meaning "not available"/"not applicable" -- rows with one of
            these get NaN in every indicator column rather than a false all-zero reading, the
            same true-missingness convention `_na` uses.

    Returns:
        A DataFrame with one float64 column per entry in ``mapping``.
    """
    cols = {name: (series == code).astype(np.float64) for code, name in mapping.items()}
    out = pd.DataFrame(cols, index=series.index)
    out.loc[series.isin(unavailable)] = np.nan
    return out


def _parse_origination(path: Path) -> pd.DataFrame:
    """Read one origination file and derive `ORIGINATION_FEATURE_COLUMNS` plus `loan_id`.

    Args:
        path: Path to a `sample_orig_YYYY.txt` or `historical_data_YYYY.txt` file.

    Returns:
        One row per loan, `loan_id` plus every column in `ORIGINATION_FEATURE_COLUMNS`.
    """
    raw = pd.read_csv(path, sep="|", header=None, names=_ORIG_COLUMNS, dtype=str, na_filter=False)

    out = pd.DataFrame(index=raw.index)
    out["loan_id"] = raw["loan_id"]
    out["credit_score"] = _na(raw["credit_score"], "9999")
    out["first_time_homebuyer"] = raw["first_time_homebuyer"].map({"Y": 1.0, "N": 0.0})
    out["msa"] = pd.to_numeric(raw["msa"].replace("", np.nan), errors="coerce")
    out["mi_pct"] = _na(raw["mi_pct"], "999")
    out["num_units"] = _na(raw["num_units"], "99")
    out = out.join(_one_hot(
        raw["occupancy_status"],
        {"P": "occ_primary", "I": "occ_investment", "S": "occ_second_home"},
        unavailable=("9",),
    ))
    out["original_cltv"] = _na(raw["original_cltv"], "999")
    out["original_dti"] = _na(raw["original_dti"], "999")
    out["original_upb"] = pd.to_numeric(raw["original_upb"], errors="coerce")
    out["original_ltv"] = _na(raw["original_ltv"], "999")
    out["original_interest_rate"] = pd.to_numeric(raw["original_interest_rate"], errors="coerce")
    out = out.join(_one_hot(
        raw["channel"],
        {"R": "chan_retail", "B": "chan_broker", "C": "chan_correspondent", "T": "chan_third_party"},
        unavailable=("9",),
    ))
    out["ppm_flag"] = raw["ppm_flag"].map({"Y": 1.0, "N": 0.0})
    out["is_arm"] = raw["amortization_type"].map({"FRM": 0.0, "ARM": 1.0})
    out["original_loan_term"] = pd.to_numeric(raw["original_loan_term"], errors="coerce")
    out["num_borrowers"] = _na(raw["num_borrowers"], "99")
    out["super_conforming_flag"] = raw["super_conforming_flag"].map({"Y": 1.0, "N": 0.0})
    # Null here means "not available or not applicable" and the overwhelming common case is
    # "no special program" -- a real zero, not missingness, unlike the Not Available sentinels
    # above. See module docstring's categorical-handling note.
    elig = raw["special_eligibility_program"]
    out["special_elig_home_possible"] = (elig == "H").astype(np.float64)
    out["special_elig_hfa_advantage"] = (elig == "F").astype(np.float64)
    out["special_elig_refi_possible"] = (elig == "R").astype(np.float64)
    out["harp_indicator"] = raw["harp_indicator"].map({"Y": 1.0, "N": 0.0})
    out["property_valuation_method"] = _na(raw["property_valuation_method"], "7")
    out["interest_only"] = raw["interest_only"].map({"Y": 1.0, "N": 0.0})
    out["vantage_score"] = _na(raw["vantage_score"], "9999")
    out = out.join(_one_hot(
        raw["loan_purpose"],
        {
            "P": "loan_purpose_purchase", "C": "loan_purpose_cashout_refi",
            "N": "loan_purpose_nocashout_refi", "R": "loan_purpose_refi_unspecified",
        },
        unavailable=("9",),
    ))
    out = out.join(_one_hot(
        raw["property_type"],
        {
            "SF": "property_type_sf", "CO": "property_type_co", "PU": "property_type_pu",
            "CP": "property_type_cp", "MH": "property_type_mh",
        },
        unavailable=("99",),
    ))
    return out[["loan_id", *ORIGINATION_FEATURE_COLUMNS]]


def _parse_distress_labels(path: Path) -> pd.DataFrame:
    """Read one performance file and derive one label per loan.

    Reads only the two columns the label needs (`loan_id`, `zero_balance_code`), not the full
    35-column file -- the performance file is roughly 35x the origination file's size for the
    same loans, and every other column is irrelevant to this cross-sectional, origination-time
    panel.

    Args:
        path: Path to a `sample_perf_YYYY.txt` or `historical_data_YYYY.txt` performance file.

    Returns:
        One row per loan_id seen in the file, with a binary `label` column: 1 if any monthly
        record for that loan carried a distress Zero Balance Code
        (`_DISTRESS_ZERO_BALANCE_CODES`), else 0.
    """
    loan_id_idx = _PERF_COLUMNS.index("loan_id")
    zbc_idx = _PERF_COLUMNS.index("zero_balance_code")
    raw = pd.read_csv(
        path, sep="|", header=None, usecols=[loan_id_idx, zbc_idx],
        names=["loan_id", "zero_balance_code"], dtype=str, na_filter=False,
    )
    is_distress = raw["zero_balance_code"].isin(_DISTRESS_ZERO_BALANCE_CODES)
    labels = is_distress.groupby(raw["loan_id"]).any().astype(np.int64)
    return labels.rename("label").reset_index()


def build_panel(orig_path: Path, perf_path: Path) -> pd.DataFrame:
    """Join one vintage's origination features to its distress label.

    Args:
        orig_path: Path to that vintage's origination file.
        perf_path: Path to that vintage's monthly performance file.

    Returns:
        One row per loan with a performance record, `loan_id`, `label`, and every column in
        `ORIGINATION_FEATURE_COLUMNS`. Loans with no performance record at all (should not
        occur in a released vintage, since every loan has at least one reporting month) are
        dropped by the inner join rather than silently labelled 0.
    """
    features = _parse_origination(orig_path)
    labels = _parse_distress_labels(perf_path)
    panel = features.merge(labels, on="loan_id", how="inner")
    return panel.reset_index(drop=True)


def main() -> None:
    """Entry point. See module docstring for what this writes and why."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--orig-file", required=True, help="e.g. data/mortgage/sample_orig_2018.txt")
    p.add_argument("--perf-file", required=True, help="e.g. data/mortgage/sample_perf_2018.txt")
    p.add_argument("--out", default="data/mortgage/panel.parquet")
    args = p.parse_args()

    panel = build_panel(Path(args.orig_file), Path(args.perf_file))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(out)
    print(f"wrote {len(panel)} rows, {panel['label'].mean():.4%} positive, to {out}", flush=True)


if __name__ == "__main__":
    main()
