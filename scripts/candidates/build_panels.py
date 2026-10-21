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
  paid-in-full as ``"P I F"``. The features come from
  `fintfm.evaluation.credit_suite.sba_frame`, which also lists every dropped field and why.
  `IsRevolver` was all-NaN in the 2026-10-06 panel (the file codes `RevolverStatus` N/Y, not
  the dictionary's 0/1) and is filled from 2026-10-20.

**`ppdai`, `bondora` and `ulb_fraud` are evaluation-only since 2026-10-19, and `sba` since
2026-10-20** (design review DR4.3): they make up the credit suite in
`fintfm.evaluation.credit_suite`, and `fintfm.prior.real_panel` refuses their panels, so their
panels here serve the feasibility script and nothing that trains.

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

from fintfm.evaluation.credit_suite import SBA_FILE, bondora_frame, sba_frame

ROOT = Path(__file__).resolve().parents[2]
REAL = ROOT / "data" / "real"


def build_ppdai() -> pd.DataFrame:
    """PPDai panel: the uploader's 29 preprocessed features and binary label, as floats."""
    raw = pd.read_csv(REAL / "ppdai" / "ppdaiData.csv")
    feats = raw.drop(columns=["label"]).astype(np.float64)
    return feats.assign(label=raw["label"].astype(np.int64))


def build_bondora(horizon_months: int = 12) -> pd.DataFrame:
    """Bondora panel with a fixed-horizon, uncensored default label.

    The definition lives in :func:`fintfm.evaluation.credit_suite.bondora_frame`, shared with
    the credit suite; the panel is that frame less its bookkeeping columns.

    Args:
        horizon_months: Default must occur within this many months of `LoanDate`; only loans
            whose full window precedes the snapshot are kept.

    Returns:
        The panel.
    """
    raw = pd.read_csv(REAL / "bondora" / "LoanData_Bondora.csv", low_memory=False)
    return bondora_frame(raw, horizon_months).drop(columns=["_user", "_loan_date"])


def build_ulb_fraud() -> pd.DataFrame:
    """ULB card-fraud panel: `Time`, `V1`-`V28`, `Amount` and `Class` as the label."""
    raw = pd.read_csv(REAL / "ulb_fraud" / "creditcard.csv")
    return raw.drop(columns=["Class"]).astype(np.float64).assign(label=raw["Class"].astype(np.int64))


def build_sba(horizon_months: int = 60) -> pd.DataFrame:
    """SBA 7(a) FY2010-FY2019 panel, label = charge-off within `horizon_months` of approval.

    The definition lives in :func:`fintfm.evaluation.credit_suite.sba_frame`, shared with the
    credit suite; the panel is that frame less its bookkeeping column.

    Args:
        horizon_months: Charge-off horizon. 60 is the longest horizon every loan in the
            FY2010-FY2019 file has fully observed by the 2026-06-30 snapshot.

    Returns:
        The panel.
    """
    raw = pd.read_csv(REAL / "sba" / SBA_FILE, encoding_errors="replace", low_memory=False)
    return sba_frame(raw, horizon_months).drop(columns=["_approval_date"])


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
