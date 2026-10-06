"""Build real-data panels for the shortlisted FinTFM-R candidates (docs/research/data_candidates.csv).

Every builder here writes `data/real/<source>/panel.parquet` with the same convention: float
feature columns plus an integer `label`, nothing else -- so one generic sampler
(`fintfm.prior.real_panel`) can read any of them, rather than a near-identical loader module per
source. Every label is a **fixed-horizon outcome fully observed for every row kept**, and every
feature is one available at origination; the specific leakage and censoring traps each source
has are named where they are handled, because each was found by opening the data, not assumed.

Sources and why each label is shaped the way it is:

- `ppdai` -- figshare 13573871, CC BY 4.0. Only `ppdaiData.csv` is read; the sibling
  `DefaultData.csv` in the same record is TabArena's `credit_card_clients_default` and must never
  be read. Label is the source's own. Features are already preprocessed by the uploader.
- `bondora` -- Kaggle mirror (CC0) of Bondora's public loan data, snapshot 2021-07-20. Label is
  default (Bondora's 60-days-past-due definition) within 12 months of the loan date, kept only for
  loans dated at least 12 months before the snapshot -- an "ever defaulted" label would be
  censored for the 2019-2020 majority of the book. Bondora's own risk outputs (`Rating*`,
  `ProbabilityOfDefault`, `ExpectedLoss`, ...) are excluded, the same choice the LendingClub
  release makes for the platform's grade: the model should learn from borrower attributes, not
  copy the platform's score. Every payment, balance, recovery and status field is post-outcome.
- `ulb_fraud` -- Kaggle `mlg-ulb/creditcardfraud`, DbCL-1.0. `Class` is the label; the 28 PCA
  components, `Time` and `Amount` are the features, unchanged.
- `sba` -- SBA 7(a) FOIA, FY2010-FY2019, U.S. Government Works (public domain), read directly from
  data.sba.gov rather than the CC-BY-SA Kaggle mirror. Label is charge-off within 60 months of
  approval: every loan in the file has at least five years observed by the 2026-06-30 snapshot,
  so the horizon is uncensored. Never-disbursed loans (`CANCLD`, `COMMIT`) are dropped. Borrower
  name/address and lender identity are dropped; charge-off amount, paid-in-full date, status and
  the secondary-market flag are post-origination and dropped. Note the source spells
  paid-in-full as ``"P I F"``.

These files hold real third-party data. Their outputs live under `data/real/`, which
`.gitignore` covers, and must never be committed.

Usage:
    uv run python scripts/candidates/build_panels.py ppdai bondora ulb_fraud sba
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REAL = ROOT / "data" / "real"


def _one_hot(series: pd.Series, prefix: str, top: int) -> pd.DataFrame:
    """One-hot the `top` most frequent values of a categorical column, as floats.

    Args:
        series: Raw categorical column.
        prefix: Output column-name prefix.
        top: How many of the most frequent values get their own column; the rest share none
            (all-zero row), keeping the feature count bounded for high-cardinality fields.

    Returns:
        A float64 indicator frame; rows where `series` is missing are NaN throughout.
    """
    values = series.value_counts().index[:top]
    out = pd.DataFrame(
        {f"{prefix}_{v}": (series == v).astype(np.float64) for v in values}, index=series.index
    )
    out.loc[series.isna()] = np.nan
    return out


def build_ppdai() -> pd.DataFrame:
    """PPDai panel: the uploader's 29 preprocessed features and binary label, as floats."""
    raw = pd.read_csv(REAL / "ppdai" / "ppdaiData.csv")
    feats = raw.drop(columns=["label"]).astype(np.float64)
    return feats.assign(label=raw["label"].astype(np.int64))


#: Bondora numeric fields known at origination.
_BONDORA_NUMERIC = (
    "Age", "AppliedAmount", "Amount", "Interest", "LoanDuration", "MonthlyPayment",
    "NrOfDependants", "IncomeFromPrincipalEmployer", "IncomeFromPension",
    "IncomeFromFamilyAllowance", "IncomeFromSocialWelfare", "IncomeFromLeavePay",
    "IncomeFromChildSupport", "IncomeOther", "IncomeTotal", "ExistingLiabilities",
    "LiabilitiesTotal", "RefinanceLiabilities", "DebtToIncome", "FreeCash",
    "NoOfPreviousLoansBeforeLoan", "AmountOfPreviousLoansBeforeLoan",
    "PreviousRepaymentsBeforeLoan", "PreviousEarlyRepaymentsBefoleLoan",
    "PreviousEarlyRepaymentsCountBeforeLoan", "ApplicationSignedHour",
    "ApplicationSignedWeekday", "BidsPortfolioManager", "BidsApi", "BidsManual",
    "MonthlyPaymentDay",
)

#: Bondora coded fields known at origination, one-hot encoded (field, top-k values kept).
_BONDORA_CATEGORICAL = (
    ("Country", 4), ("VerificationType", 5), ("Gender", 3), ("UseOfLoan", 10),
    ("Education", 6), ("MaritalStatus", 6), ("EmploymentStatus", 6), ("HomeOwnershipType", 10),
    ("EmploymentDurationCurrentEmployer", 8), ("OccupationArea", 10),
)


def build_bondora(horizon_months: int = 12) -> pd.DataFrame:
    """Bondora panel with a fixed-horizon, uncensored default label.

    Args:
        horizon_months: Default must occur within this many months of `LoanDate`; only loans
            whose full window precedes the snapshot are kept.

    Returns:
        The panel.
    """
    raw = pd.read_csv(REAL / "bondora" / "LoanData_Bondora.csv", low_memory=False)
    snapshot = pd.to_datetime(raw["ReportAsOfEOD"]).max()
    loan_date = pd.to_datetime(raw["LoanDate"])
    keep = loan_date <= snapshot - pd.DateOffset(months=horizon_months)
    raw, loan_date = raw[keep], loan_date[keep]
    default_date = pd.to_datetime(raw["DefaultDate"])
    label = (default_date.notna() & (default_date <= loan_date + pd.DateOffset(months=horizon_months)))
    out = pd.DataFrame(index=raw.index)
    for col in _BONDORA_NUMERIC:
        # "10Plus" is Bondora's own top-coding of NrOfDependants; any *other* non-numeric value
        # is unexpected and is reported rather than silently turned into missingness.
        src = raw[col].replace({"10Plus": "10"})
        out[col] = pd.to_numeric(src, errors="coerce")
        lost = int((out[col].isna() & src.notna()).sum())
        if lost:
            print(f"  bondora: {lost} non-numeric value(s) in {col} coerced to NaN", flush=True)
    out["NewCreditCustomer"] = (raw["NewCreditCustomer"].astype(str) == "True").astype(np.float64)
    out["LoanYear"] = loan_date.dt.year.astype(np.float64)
    for field, top in _BONDORA_CATEGORICAL:
        out = out.join(_one_hot(raw[field].astype("string"), field, top))
    return out.assign(label=label.astype(np.int64)).reset_index(drop=True)


def build_ulb_fraud() -> pd.DataFrame:
    """ULB card-fraud panel: `Time`, `V1`-`V28`, `Amount` and `Class` as the label."""
    raw = pd.read_csv(REAL / "ulb_fraud" / "creditcard.csv")
    return raw.drop(columns=["Class"]).astype(np.float64).assign(label=raw["Class"].astype(np.int64))


def _money(series: pd.Series) -> pd.Series:
    """Parse SBA currency strings (``"$1,234.00"`` or plain numbers) to floats."""
    return pd.to_numeric(series.astype(str).str.replace(r"[$,]", "", regex=True), errors="coerce")


def build_sba(horizon_months: int = 60) -> pd.DataFrame:
    """SBA 7(a) FY2010-FY2019 panel, label = charge-off within `horizon_months` of approval.

    Args:
        horizon_months: Charge-off horizon. 60 is the longest horizon every loan in the
            FY2010-FY2019 file has fully observed by the 2026-06-30 snapshot.

    Returns:
        The panel.
    """
    raw = pd.read_csv(
        REAL / "sba" / "FOIA_7a_FY2010_FY2019_asof_260630.csv",
        encoding_errors="replace", low_memory=False,
    )
    raw = raw[~raw["LoanStatus"].isin(["CANCLD", "COMMIT"])]
    # 10 charged-off loans in the 2026-06-30 file carry no ChargeOffDate. Their horizon outcome
    # is unknowable, and keeping them would label them 0 silently -- drop instead.
    raw = raw[~((raw["LoanStatus"] == "CHGOFF") & raw["ChargeOffDate"].isna())]
    approval = pd.to_datetime(raw["ApprovalDate"], errors="coerce")
    chargeoff = pd.to_datetime(raw["ChargeOffDate"], errors="coerce")
    label = chargeoff.notna() & (chargeoff <= approval + pd.DateOffset(months=horizon_months))
    out = pd.DataFrame(index=raw.index)
    out["GrossApproval"] = _money(raw["GrossApproval"])
    out["SBAGuaranteedApproval"] = _money(raw["SBAGuaranteedApproval"])
    out["GuaranteeShare"] = out["SBAGuaranteedApproval"] / out["GrossApproval"]
    out["InitialInterestRate"] = pd.to_numeric(raw["InitialInterestRate"], errors="coerce")
    out["TermInMonths"] = pd.to_numeric(raw["TermInMonths"], errors="coerce")
    out["JobsSupported"] = pd.to_numeric(raw["JobsSupported"], errors="coerce")
    out["ApprovalFY"] = pd.to_numeric(raw["ApprovalFY"], errors="coerce")
    out["IsVariableRate"] = (raw["FixedorVariableInterestInd"] == "V").astype(np.float64)
    out["IsRevolver"] = pd.to_numeric(raw["RevolverStatus"], errors="coerce")
    out["HasCollateral"] = (raw["CollateralInd"] == "Y").astype(np.float64)
    out["IsFranchise"] = raw["FranchiseCode"].notna().astype(np.float64)
    naics = raw["NaicsCode"].astype("string").str[:2]
    out = out.join(_one_hot(naics, "Naics2", 20))
    for field, top in (("BusinessType", 3), ("BusinessAge", 6), ("ProcessingMethod", 8)):
        out = out.join(_one_hot(raw[field].astype("string"), field, top))
    return out.assign(label=label.astype(np.int64)).reset_index(drop=True)


BUILDERS = {
    "ppdai": build_ppdai, "bondora": build_bondora, "ulb_fraud": build_ulb_fraud, "sba": build_sba,
}


def main() -> None:
    """Entry point: build the named panels and report each one's size and positive rate."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("sources", nargs="+", choices=sorted(BUILDERS))
    args = p.parse_args()
    for name in args.sources:
        panel = BUILDERS[name]()
        out = REAL / name / "panel.parquet"
        panel.to_parquet(out)
        feats = panel.shape[1] - 1
        print(f"{name}: {len(panel)} rows, {feats} features, "
              f"{panel['label'].mean():.4%} positive -> {out}", flush=True)


if __name__ == "__main__":
    main()
