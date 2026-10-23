"""A credit evaluation suite outside V4FinBench (design review 2026-10-16, task DR4.3).

V4FinBench is the benchmark this project's claims rest on, and every arm so far was chosen on
it. A second credit suite, scored the same way, says whether a change helps on credit data the
arm was not chosen on. The suite is four public consumer-credit, card-fraud and small-business
loan tables, with a fixed split per dataset, logistic regression and LightGBM at 1,000 trees
beside FinTFM, and one CSV row per dataset, split and model.

**Evaluation only, never training.** None of the four may feed pretraining or fine-tuning of
either family: :data:`fintfm.prior.real_panel.EVAL_ONLY_SOURCES` makes the FinTFM-R sampler
refuse their panels, and ``docs/research/data_candidates.csv`` marks them ``EVAL_ONLY``. PPDai,
Bondora, ULB fraud and SBA were FinTFM-R pilot arms before this suite existed
(``docs/results/FINDINGS.md`` §157), so a checkpoint from those arms has seen one of them
and its number here is not a held-out number. SBA was also the C5 decisive arm
(``scripts/kaggle/fintfm_r_sba_s0`` and ``_s1``), so no checkpoint trained on ``real:sba`` is
held out on SBA. FinTFM (synthetic-only) never saw any of them.

Datasets (sizes MEASURED from the raw files: SBA on 2026-10-20, the others on 2026-10-19)
-----------------------------------------------------------------------------------------

=========  =======  =========  ========  ============================  ======================
key        rows     positives  features  split                         test rows per split
=========  =======  =========  ========  ============================  ======================
ppdai       55,596    7,183       28     out-of-time on ``loan_time``  8,948 / 9,307 / 10,026
bondora    148,783   43,117      100     out-of-time on ``LoanDate``,  24,791 / 24,682 / 24,961
                                         borrower-disjoint
ulb_fraud  284,807      492       30     stratified, 164 positives     94,936 / 94,936 / 94,935
                                         per test fold
sba        478,953   18,826       47     out-of-time on                79,722 / 79,887 / 79,938
                                         ``ApprovalDate``
=========  =======  =========  ========  ============================  ======================

Training rows per split: PPDai 27,315 / 36,263 / 45,570; Bondora 40,659 / 51,112 / 60,999
(after the embargo and the borrower rule); ULB 189,871 / 189,871 / 189,872; SBA 23,763 /
83,052 / 138,004 (after the embargo). SBA's test positives per split: 3,125 / 3,489 / 3,409.

**Runtime, ESTIMATED (the suite has not been run end to end).** FinTFM's cost grows with the
test rows it scores. A SMOKE-TEST of the published checkpoint (``kabartay/fintfm-binary``
``v4-cellattn-labels.pt``, sha256 ``d231b062...``) at the packaged ``inference`` section, CPU,
split 0, took 27-70 ms per test row (PPDai 67.6 s for 1,000 rows and 279.5 s for 4,000;
Bondora 27.3 s and 161.9 s) while the machine's load average stood at 200-216. At that rate the
suite's 387,522 test rows take about 3 to 7.5 hours for FinTFM, ULB alone 2.4 to 5.5 hours, and
SBA's 239,547 add about 1.8 to 4.7 hours; an unloaded machine should be faster, by an amount
not measured. LightGBM at 1,000 trees and
logistic regression are ESTIMATED at under a minute per split each, scaling §156's
feasibility timings (``runs/feasibility/*.json``: 5-fold LightGBM at 300 trees took 6.0 s on
PPDai, 12.6 s on Bondora, 15.1 s on a 200,000-row ULB sample). Fine-tuning
(``finetune_steps > 0``) is in none of these.

**PPDai** (``ppdai/ppdaiData.csv``, figshare 13573871). The uploader's 29 preprocessed columns
less ``loan_time``, which orders the split and is not a feature. ``loan_time`` is an offset
timestamp at day resolution (191 distinct days over 200 days); it is the only date. There is
no borrower id, so the split is not grouped, and the uploader does not state the label's
horizon, so no embargo is applied. Four rows are exact duplicates and are kept. Only
``ppdaiData.csv`` is read: the record's sibling ``DefaultData.csv`` is TabArena's
``credit_card_clients_default`` and must never be read.

**Bondora** (``bondora/LoanData_Bondora.csv``, Kaggle mirror snapshot 2021-07-20). The FinTFM-R
panel's label and features (:func:`bondora_frame`, which ``scripts/candidates/build_panels.py``
also uses): default within 12 months of ``LoanDate``, kept only for the 148,783 of 179,235 loans
whose window closes before the snapshot; Bondora's own risk outputs and every post-origination
field excluded. ``LoanYear`` is dropped here because the split runs on the date. ``UserName``
identifies 72,101 borrowers. Each split trains only on loans whose 12-month outcome was known
before the test window opens (a 366-day embargo) and drops training rows of borrowers who
appear in that split's test window.

**ULB card fraud** (``ulb_fraud/creditcard.csv``). ``Time``, ``V1``-``V28`` and ``Amount``,
unchanged. ``Time`` is seconds since the first transaction of a two-day capture, not a date; a
"later window" inside 48 hours measures intraday shift, not out-of-time generalisation, and
would leave about 80 positives per test window. So the split is stratified, three folds, about
164 positives each. There is no cardholder id.

**SBA 7(a)** (``sba/FOIA_7a_FY2010_FY2019_asof_260630.csv``, the SBA's FOIA release of loans
approved FY2010-FY2019, as of 2026-06-30). The FinTFM-R panel's label and features
(:func:`sba_frame`, which ``scripts/candidates/build_panels.py`` also uses): charge-off within
60 months of ``ApprovalDate``. The latest approval is 2019-09-30, so every window closes by
2024-09-30, before the snapshot, and the label is uncensored; :func:`sba_frame` refuses a file
for which that fails. Of 545,751 loans, the 66,788 never disbursed (``CANCLD``, ``COMMIT``)
and the 10 charged off with no ``ChargeOffDate`` are dropped, leaving 478,953; loans still
running (``EXEMPT``) are kept as not charged off, which the uncensored window makes exact. The
file spells paid-in-full ``"P I F"``. :data:`SBA_FEATURE_SOURCES` lists the 14 fields features
are built from, each known at approval; :data:`SBA_DROPPED` lists every other field and why,
and :func:`sba_frame` refuses a file with a field in neither. The outcome fields
(``LoanStatus``, ``ChargeOffDate``, ``GrossChargeOffAmount``, ``PaidInFullDate``), the
post-approval ones (``FirstDisbursementDate``, ``SoldSecMrktInd``), borrower name and address
(personal data) and lender identity are never features. ``RevolverStatus`` is a term or
revolver flag set by the loan's structure at approval, so it is kept; the file codes it N/Y
where the dictionary says 0/1, which left ``IsRevolver`` all-NaN in the FinTFM-R panel built on
2026-10-06 (the only difference between that panel and this frame, MEASURED 2026-10-20).
``ApprovalFY`` is dropped here, as Bondora's ``LoanYear`` is, because the split runs on the
date. There is no borrower id, so the split is not grouped. Each split trains only on loans
approved at least 1,827 days (60 months at their longest) before its test window opens, so
every training label was known by then; the cost is a small first split, 23,763 training rows
from the first seven months of FY2010. ``docs/results/FINDINGS.md`` §156 leaves one asymmetry
open: few short-term loans appear as paid in full. ``TermInMonths`` is not rewritten after
charge-off (§156), but it is the strongest single feature, so SBA's scores carry that open
question.

Split protocol
--------------

Three splits per dataset, fixed by :data:`SPLIT_SEED` and the data alone:

* **out-of-time** (a date exists): cut points at the 1/2, 2/3 and 5/6 quantiles of the date.
  Split ``k`` tests on the rows between cut ``k`` and the next one (the last includes the
  latest date) and trains on every earlier row, less the embargo and, with a borrower id,
  less the test window's borrowers. Train grows from half the rows; each test window is a
  sixth.
* **stratified** (no date): ``StratifiedKFold(3, shuffle=True, random_state=SPLIT_SEED)``,
  or ``StratifiedGroupKFold`` when a borrower id exists. No dataset uses the grouped form today.

Licence at the original source (read 2026-10-19, SBA's 2026-10-20), and which are admitted
--------------------------------------------------------------------------------------------

DR4.2's owner rule (2026-10-16) decides on the licence at the data's original source: CC BY,
CC0, public domain or MIT pass; a permissive licence granted by someone other than the data's
owner over data from a source that states no licence is excluded as unverified, and so is a
copyleft term. Applied to the four datasets:

* **PPDai**: figshare states "CC BY 4.0" (``https://api.figshare.com/v2/articles/13573871``,
  field ``license``). The record's description says the data "is obtained from customer data
  from a Chinese P2P lending platform named PaiPaiDai"; the uploader is a researcher, not the
  platform. The platform's competition page (``https://ai.ppdai.com/mirror/goToMirrorDetail?
  mirrorId=1``) did not respond; its Wayback copy of 2026-05-10 states no licence and gates the
  data behind a competition agreement (《比赛协议》).
* **Bondora**: Kaggle states "CC0: Public Domain" (``https://www.kaggle.com/api/v1/datasets/
  view/sid321axn/bondora-peer-to-peer-lending-loan-data``, field ``licenseNameNullable``);
  the uploader is an individual and the page says "The dataset has been downloaded from
  Bondora website". Bondora's page (``https://www.bondora.com/en/public-reports``) now redirects
  to ``https://www.goandgrow.eu/en/public-reports``, which offers no dataset and states no
  licence; its Wayback copy of 2021-07-24 states no licence, only "We recommend that you use
  the datasets to run your own portfolio analysis and adjust your investment strategy."
* **ULB fraud**: Kaggle states "Database: Open Database, Contents: Database Contents"
  (``https://www.kaggle.com/api/v1/datasets/view/mlg-ulb/creditcardfraud``), that is ODbL-1.0
  for the database and DbCL-1.0 for its contents, not DbCL-1.0 alone. The uploader is ULB's
  Machine Learning Group, which collected the data with Worldline, so the grant is from the
  source. ODbL carries a share-alike term on publicly used derivative databases; scores are
  "Produced Works" under ODbL, which need a notice when public. The MLG project page now
  returns 404, and its Wayback copy of 2025-04-05 states no licence.
* **SBA 7(a)**: the SBA's catalogue page (``https://data.sba.gov/dataset/7a-504-foia``, title
  "7(a) & 504 FOIA") states, in its "Additional information" table, "Publisher: U.S. Small
  Business Administration" and "License: U.S. Government Works", the licence text linking
  ``https://www.usa.gov/publicdomain/label/1.0/``; it lists this file as "FOIA - 7(a)
  (FY2010-FY2019) as of June 30, 2026 (csv)", last modified April 28, 2026. The grantor is the
  SBA, the data's producer, so the grant is from the source. The linked label now redirects to
  ``https://www.usa.gov/government-copyright``, which says "Government work is something
  created by a U.S. government officer or employee as part of their official duties", that
  "U.S. copyright laws may not protect U.S. government works outside the country" though the
  work "may be protected under the copyright laws of other jurisdictions", and that "other
  people may have rights to the work itself or how people use it. This includes publicity or
  privacy rights under state law." The last is why borrower names and addresses never leave
  the raw file.

Owner decisions: ULB is admitted and PPDai and Bondora are not (2026-10-19), and SBA is
admitted (2026-10-20), a public-domain grant by its producer, which DR4.2's rule passes and
which asks for no notice. ULB's grant comes from the data's collector, and the suite never
redistributes the database or a derivative of it; its scores are Produced Works, which
ODbL-1.0 section 4.3 allows with a notice, so the share-alike term does not apply.
:data:`ODBL_NOTICE` is that notice: :func:`run` prints it and writes it beside the results, and
any public text quoting ULB scores carries it. PPDai and
Bondora keep ``include=False``: their permissive grants come from someone other than the data's
owner over data whose source states no licence. :func:`run` refuses an unadmitted dataset, and
the command line scores the admitted ones by default. Admitting one is an owner decision
recorded by flipping its entry, not a flag.

Overlap with TabArena-v0.1 and BeyondArena (checked 2026-10-19, SBA 2026-10-20)
-------------------------------------------------------------------------------

:func:`fintfm.evaluation.overlap.name_clashes` finds nothing for ``PPDai``, ``PaiPaiDai``,
``Bondora``, ``creditcardfraud`` or ``ulb_fraud``. OpenML's name for the ULB table,
``creditcard``, contains-matches ``credit_card_clients_default`` (Taiwan, 30,000 rows, 23
features); the contents differ (284,807 rows, 30 PCA features), so it is a name collision, not
an overlap. Against the fork's metadata (``num_instances``, ``num_features``), the only row
count within 5% of a suite dataset is ``give_me_some_credit`` (150,000 rows, 10 features)
against Bondora (148,783, 100), a different table. For SBA, ``name_clashes`` finds nothing for
``sba``, ``SBA 7(a)``, ``SBA 7(a) loans (FOIA)``, ``FOIA_7a_FY2010_FY2019_asof_260630``,
``small business administration``, ``SBAnational`` (the Kaggle mirror's file) or ``should this
loan be approved or denied`` (its paper), and no benchmark in
``docs/research/benchmark_datasets.csv`` has a row count within 5% of 478,953 or of the raw
545,751. No row-hash comparison was run.

Metrics
-------

Average precision is the headline (``CLAUDE.md``: read AP first at a low base rate), ROC-AUC
beside it, and :func:`ap_top_decile`, the AP earned in the highest-scored tenth of the test
rows. Each row also carries the base rate and the mean predicted probability, so a level error
is visible beside the ranking.

Hybrid arms (task DR8.4, off unless ``--hybrid``)
-------------------------------------------------

Two more rows per split measure a **system**, FinTFM plus a booster, against the booster
alone, as :mod:`fintfm.experiments.hybrid_catboost` (AU4.4) does with CatBoost on V4FinBench:

* ``lightgbm_fintfm_feature``: LightGBM at :data:`LIGHTGBM_PARAMS` with FinTFM's logit as one
  more feature.
* ``lightgbm_fintfm_baseline``: LightGBM at :data:`LIGHTGBM_PARAMS` started from FinTFM's logit
  (``init_score``), so its trees fit what FinTFM leaves over; the test score is the sigmoid of
  the trees' raw score plus the logit.

LightGBM needs a FinTFM logit for every training row, and a context drawn from the training
rows would hold the row being scored. So :func:`oof_fintfm_logits` splits each split's training
rows into ``--inner-folds`` (default :data:`INNER_FOLDS`) stratified inner folds, grouped by
``panel.groups`` where it exists (neither ULB nor SBA has one), and scores each inner fold with
a FinTFM whose context comes only from the other inner folds. Inside an out-of-time split the
inner folds are not temporal: every inner context is still made of training rows, so the
embargo against the test window holds, but a query may be scored from later training rows.
The test rows are scored by the ``fintfm`` row's own predictions, one FinTFM fitted on all
training rows, reused rather than recomputed; when that row is skipped, so are both hybrid rows.
The inner logits condition on ``(K - 1) / K`` of the training rows and the test logits on all
of them, the usual stacking asymmetry; with a ``max_context`` far below the pool, every fit
samples the same context size. Before any inner fit, :func:`~fintfm.experiments.hybrid_catboost.
assert_no_leak` refuses an inner context holding a test row, a row it scores, or, with groups,
a borrower of either.

A hybrid row's ``seconds`` is the split's inner FinTFM fits plus that booster's fit; the test
side's FinTFM time is the ``fintfm`` row's and is not counted again.

**Hybrid runtime, ESTIMATED (not run on the real data).** The ``fintfm`` arm MEASURED 512 to
606 s per split on SBA for about 80,000 test rows (§177) and 672 to 805 s on ULB for about
94,936 (§174): 6.4 to 7.6 and 7.1 to 8.5 ms per query row at a ``max_context`` of 2,000, which
does not grow with the pool. The hybrid adds every training row once as an inner query: SBA
23,763 + 83,052 + 138,004 = 244,819 rows, 0.43 to 0.52 h; ULB 3 x 189,871 = 569,614 rows, 1.1
to 1.3 h. Two extra LightGBM fits per split add minutes. ``K`` changes the number of inner
fits, not the query rows, so it barely moves the cost.

Usage::

    uv run python -m fintfm.evaluation.credit_suite --model v4-cellattn-labels.pt \\
        --out runs/credit_suite/results.csv
    uv run python -m fintfm.evaluation.credit_suite --model v4-cellattn-labels.pt \\
        --datasets sba --hybrid --out runs/credit_suite/hybrid_sba.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np
import pandas as pd

from fintfm.evaluation.datasets import CACHE_DIR
from fintfm.evaluation.overlap import guard_selection
from fintfm.evaluation.proxy_suite import SourceLicence
from fintfm.experiments.hybrid_catboost import INNER_SEED_OFFSET, assert_no_leak, to_logit

if TYPE_CHECKING:
    from fintfm.config import Config
    from fintfm.modeling.model import FinancialTFM


class InContextClassifier(Protocol):
    """What :func:`oof_fintfm_logits` needs of a FinTFM classifier or its ``large_n`` wrapper."""

    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray | None = None) -> object:
        """Condition on a context of rows ``X`` with labels ``y``."""

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """``(n, 2)`` class probabilities for the query rows ``X``."""

#: Where the raw files live: ``data/real/<key>/``, gitignored, the FinTFM-R candidates' home.
DEFAULT_ROOT = CACHE_DIR.parent / "real"

#: Seed for the stratified folds. Fixed: the split is part of the suite, not of a run.
SPLIT_SEED = 0

#: Splits per dataset.
N_SPLITS = 3

#: Quantiles of the date at which the out-of-time test windows open; the last window closes at
#: the latest date.
OOT_CUTS = (1 / 2, 2 / 3, 5 / 6)

#: LightGBM's settings: §167 measured LightGBM's 100-tree default 0.1242 mean AP below 1,000
#: trees at learning rate 0.05 on V4FinBench, so the default is under-trained.
LIGHTGBM_PARAMS = {"n_estimators": 1000, "learning_rate": 0.05}

#: The hybrid arms (module docstring, "Hybrid arms"), in the order they are reported.
HYBRID_MODELS = ("lightgbm_fintfm_feature", "lightgbm_fintfm_baseline")

#: Default inner folds for the hybrid arms' out-of-fold FinTFM logits.
INNER_FOLDS = 3

#: The notice ODbL-1.0 section 4.3 asks of a public Produced Work, for ULB scores.
ODBL_NOTICE = (
    "ULB fraud scores are a Produced Work from the Credit Card Fraud Detection database "
    "(Machine Learning Group, Universite Libre de Bruxelles, with Worldline; "
    "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), made available under the "
    "Open Database License (https://opendatacommons.org/licenses/odbl/1-0/); its contents "
    "under the Database Contents License (https://opendatacommons.org/licenses/dbcl/1-0/)."
)

def admitted() -> tuple[str, ...]:
    """Keys of :data:`DATASETS` whose licence entry admits them, in suite order."""
    return tuple(k for k, d in DATASETS.items() if d.licence.include)


#: Date the licences and overlaps in the module docstring were checked.
CHECKED = "2026-10-19"

#: Date SBA's licence and overlap were checked, the day it joined the suite.
SBA_CHECKED = "2026-10-20"

#: CSV columns, in order.
COLUMNS = (
    "dataset", "split_kind", "split", "model", "n_train", "n_test", "n_test_positive",
    "base_rate", "mean_predicted", "average_precision", "roc_auc", "ap_top_decile", "seconds",
    "checkpoint_sha256", "skipped",
)


@dataclass(frozen=True)
class CreditPanel:
    """One dataset ready to split and score.

    Attributes:
        name: The dataset's key.
        X: ``(n, n_features)`` float features, NaN for missing.
        feature_names: Column names of ``X``.
        y: ``(n,)`` binary labels.
        groups: ``(n,)`` borrower id per row, or ``None`` when the data has none.
        time: ``(n,)`` float date per row (days or seconds, only the order and differences
            matter), or ``None`` when the data has none.
        embargo: Minimum gap, in ``time`` units, between a training row's date and the start
            of the test window: the label's outcome window, so every training label is known
            when the window opens. ``0.0`` for none.
    """

    name: str
    X: np.ndarray
    feature_names: tuple[str, ...]
    y: np.ndarray
    groups: np.ndarray | None = None
    time: np.ndarray | None = None
    embargo: float = 0.0

    @property
    def split_kind(self) -> str:
        """``"out_of_time"`` when the panel has a date, else ``"stratified"``."""
        return "out_of_time" if self.time is not None else "stratified"


@dataclass(frozen=True)
class SuiteDataset:
    """A dataset of the suite, its loader and its licence verdict.

    Attributes:
        key: Directory name under the root, and the key in
            :data:`fintfm.prior.real_panel.EVAL_ONLY_SOURCES`.
        url: Where the raw file was downloaded from.
        loader: Reads the raw file(s) under ``root / key`` into a :class:`CreditPanel`.
        licence: The licence read at the source; ``licence.include`` admits the dataset.
    """

    key: str
    url: str
    loader: Callable[[Path], CreditPanel]
    licence: SourceLicence


def one_hot(series: pd.Series, prefix: str, top: int) -> pd.DataFrame:
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


#: Bondora numeric fields known at origination.
BONDORA_NUMERIC = (
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
BONDORA_CATEGORICAL = (
    ("Country", 4), ("VerificationType", 5), ("Gender", 3), ("UseOfLoan", 10),
    ("Education", 6), ("MaritalStatus", 6), ("EmploymentStatus", 6), ("HomeOwnershipType", 10),
    ("EmploymentDurationCurrentEmployer", 8), ("OccupationArea", 10),
)


def bondora_frame(raw: pd.DataFrame, horizon_months: int = 12) -> pd.DataFrame:
    """Bondora features and a fixed-horizon, uncensored default label, with bookkeeping.

    The FinTFM-R panel builder (``scripts/candidates/build_panels.py``) and this suite share
    this one definition, so the two cannot drift apart.

    Args:
        raw: Bondora's ``LoanData_Bondora.csv`` as read by pandas.
        horizon_months: Default must occur within this many months of ``LoanDate``; only
            loans whose full window precedes the snapshot (the latest ``ReportAsOfEOD``) are
            kept.

    Returns:
        Feature columns, then ``label``, then two bookkeeping columns: ``_user`` (Bondora's
        ``UserName``) and ``_loan_date`` (``LoanDate`` as a timestamp). Index reset.
    """
    snapshot = pd.to_datetime(raw["ReportAsOfEOD"]).max()
    loan_date = pd.to_datetime(raw["LoanDate"])
    keep = loan_date <= snapshot - pd.DateOffset(months=horizon_months)
    raw, loan_date = raw[keep], loan_date[keep]
    default_date = pd.to_datetime(raw["DefaultDate"])
    label = default_date.notna() & (default_date <= loan_date + pd.DateOffset(months=horizon_months))
    out = pd.DataFrame(index=raw.index)
    for col in BONDORA_NUMERIC:
        # "10Plus" is Bondora's own top-coding of NrOfDependants; any *other* non-numeric value
        # is unexpected and is reported rather than silently turned into missingness.
        src = raw[col].replace({"10Plus": "10"})
        out[col] = pd.to_numeric(src, errors="coerce")
        lost = int((out[col].isna() & src.notna()).sum())
        if lost:
            print(f"  bondora: {lost} non-numeric value(s) in {col} coerced to NaN", flush=True)
    out["NewCreditCustomer"] = (raw["NewCreditCustomer"].astype(str) == "True").astype(np.float64)
    out["LoanYear"] = loan_date.dt.year.astype(np.float64)
    for field, top in BONDORA_CATEGORICAL:
        out = out.join(one_hot(raw[field].astype("string"), field, top))
    out = out.assign(label=label.astype(np.int64), _user=raw["UserName"], _loan_date=loan_date)
    return out.reset_index(drop=True)


#: The SBA 7(a) FOIA file read, under ``sba/``: loans approved FY2010-FY2019, as of 2026-06-30.
SBA_FILE = "FOIA_7a_FY2010_FY2019_asof_260630.csv"

#: The label's horizon: charge-off within this many months of ``ApprovalDate``.
SBA_HORIZON_MONTHS = 60

#: The out-of-time embargo for SBA, in days: 60 months at their longest (five years holding two
#: leap days), so every training label's window has closed when the test window opens.
SBA_EMBARGO_DAYS = 1827.0

#: ``LoanStatus`` values of loans never disbursed; dropped, as they carry no repayment outcome.
SBA_UNDISBURSED = frozenset({"CANCLD", "COMMIT"})

#: Every ``LoanStatus`` the data dictionary defines (``"P I F"`` is how the file spells PIF).
SBA_STATUSES = frozenset({"CANCLD", "COMMIT", "CHGOFF", "EXEMPT", "P I F", "PIF"})

#: FOIA fields the SBA features are built from, each known when the loan is approved: the
#: amounts approved and guaranteed, the rate at approval (the dictionary's own words), the term,
#: the lender's application-time estimate of jobs, the fiscal year, and the loan's structure
#: (fixed or variable, term or revolver, collateral) and borrower description (franchise,
#: industry, legal form, business age) as recorded on the application, and the SBA programme
#: it was approved under.
SBA_FEATURE_SOURCES = (
    "GrossApproval", "SBAGuaranteedApproval", "InitialInterestRate", "TermInMonths",
    "JobsSupported", "ApprovalFY", "FixedorVariableInterestInd", "RevolverStatus",
    "CollateralInd", "FranchiseCode", "NaicsCode", "BusinessType", "BusinessAge",
    "ProcessingMethod",
)

#: Every other FOIA field, and why no feature uses it. :func:`sba_frame` refuses a file with a
#: field in neither list, so a new column in a later snapshot is classified before it is used.
SBA_DROPPED: dict[str, str] = {
    "AsOfDate": "the snapshot date, one value for every row",
    "Program": "always 7A in this file",
    "ApprovalDate": "orders the out-of-time split; not a feature",
    "LoanStatus": "the outcome (current status); used only to drop undisbursed loans",
    "ChargeOffDate": "the outcome; the label is built from it",
    "GrossChargeOffAmount": "post-outcome: the balance charged off",
    "PaidInFullDate": "post-outcome: the date repaid",
    "FirstDisbursementDate": "after approval, and missing exactly for undisbursed loans",
    "SoldSecMrktInd": "after approval: the guaranteed portion sold on the secondary market",
    "BorrName": "personal data: borrower identity",
    "BorrStreet": "personal data: borrower address",
    "BorrCity": "personal data: borrower address",
    "BorrState": "personal data: borrower address",
    "BorrZip": "personal data: borrower address",
    "LocationID": "lender identity",
    "BankName": "lender identity, as currently assigned, so it can change after approval",
    "BankFDICNumber": "lender identity",
    "BankNCUANumber": "lender identity",
    "BankStreet": "lender identity",
    "BankCity": "lender identity",
    "BankState": "lender identity",
    "BankZip": "lender identity",
    "NaicsDescription": "the text of NaicsCode, which is used",
    "FranchiseName": "the name behind FranchiseCode, whose presence is used",
    "ProjectCounty": "geography, left out of the FinTFM-R panel this frame reproduces",
    "ProjectState": "geography, left out of the FinTFM-R panel this frame reproduces",
    "SBADistrictOffice": "geography, left out of the FinTFM-R panel this frame reproduces",
    "CongressionalDistrict": "geography, left out of the FinTFM-R panel this frame reproduces",
}

#: ``RevolverStatus`` codes: the dictionary documents 0/1, the 2026-06-30 file holds N/Y.
_SBA_REVOLVER = {"0": 0.0, "1": 1.0, "0.0": 0.0, "1.0": 1.0, "N": 0.0, "Y": 1.0}


def sba_money(series: pd.Series) -> pd.Series:
    """Parse SBA currency fields (``"$1,234.00"`` or plain numbers) to floats.

    Args:
        series: A raw amount column.

    Returns:
        Float amounts, NaN where a value does not parse.
    """
    return pd.to_numeric(series.astype(str).str.replace(r"[$,]", "", regex=True), errors="coerce")


def sba_frame(raw: pd.DataFrame, horizon_months: int = SBA_HORIZON_MONTHS) -> pd.DataFrame:
    """SBA 7(a) features and a fixed-horizon, uncensored charge-off label, with bookkeeping.

    The FinTFM-R panel builder (``scripts/candidates/build_panels.py``) and this suite share
    this one definition. Loans never disbursed (:data:`SBA_UNDISBURSED`) are dropped, and so
    are charged-off loans with no ``ChargeOffDate``, whose horizon outcome is unknowable and
    would otherwise be labelled 0 silently. Loans still running (``EXEMPT``) are kept: they
    have not been charged off, and the horizon check below guarantees none can still be
    charged off inside its window.

    Args:
        raw: The FOIA 7(a) CSV as read by pandas.
        horizon_months: Charge-off must occur within this many months of ``ApprovalDate``.

    Returns:
        Feature columns, then ``label``, then the bookkeeping column ``_approval_date``
        (``ApprovalDate`` as a timestamp). Index reset.

    Raises:
        ValueError: If a field is in neither :data:`SBA_FEATURE_SOURCES` nor
            :data:`SBA_DROPPED`, a ``LoanStatus`` or ``RevolverStatus`` code is unknown, an
            approval date does not parse, or a kept loan's window closes after the snapshot
            (the label would be censored).
    """
    unclassified = sorted(set(raw.columns) - set(SBA_FEATURE_SOURCES) - set(SBA_DROPPED))
    if unclassified:
        raise ValueError(f"sba: unclassified FOIA field(s) {unclassified}; add each to "
                         "SBA_FEATURE_SOURCES (known at approval) or SBA_DROPPED")
    status = raw["LoanStatus"].astype("string").str.strip()
    unknown = sorted(set(status.dropna()) - SBA_STATUSES)
    if unknown or status.isna().any():
        raise ValueError(f"sba: unknown LoanStatus value(s) {unknown}, "
                         f"{int(status.isna().sum())} missing")
    undated = (status == "CHGOFF").to_numpy(bool) & raw["ChargeOffDate"].isna().to_numpy()
    raw = raw[~status.isin(SBA_UNDISBURSED).to_numpy(bool) & ~undated]
    approval = pd.to_datetime(raw["ApprovalDate"], errors="coerce")
    if approval.isna().any():
        raise ValueError(f"sba: {int(approval.isna().sum())} ApprovalDate value(s) do not parse")
    snapshot = pd.to_datetime(raw["AsOfDate"]).max()
    window_end = approval + pd.DateOffset(months=horizon_months)
    if window_end.max() > snapshot:
        raise ValueError(f"sba: a {horizon_months}-month window closes {window_end.max():%Y-%m-%d}"
                         f", after the {snapshot:%Y-%m-%d} snapshot, so the label is censored")
    chargeoff = pd.to_datetime(raw["ChargeOffDate"], errors="coerce")
    label = chargeoff.notna() & (chargeoff <= window_end)
    revolver = raw["RevolverStatus"].astype("string").str.strip()
    odd = sorted(set(revolver.dropna()) - set(_SBA_REVOLVER))
    if odd:
        raise ValueError(f"sba: unknown RevolverStatus value(s) {odd}")
    out = pd.DataFrame(index=raw.index)
    out["GrossApproval"] = sba_money(raw["GrossApproval"])
    out["SBAGuaranteedApproval"] = sba_money(raw["SBAGuaranteedApproval"])
    out["GuaranteeShare"] = out["SBAGuaranteedApproval"] / out["GrossApproval"]
    out["InitialInterestRate"] = pd.to_numeric(raw["InitialInterestRate"], errors="coerce")
    out["TermInMonths"] = pd.to_numeric(raw["TermInMonths"], errors="coerce")
    out["JobsSupported"] = pd.to_numeric(raw["JobsSupported"], errors="coerce")
    out["ApprovalFY"] = pd.to_numeric(raw["ApprovalFY"], errors="coerce")
    out["IsVariableRate"] = (raw["FixedorVariableInterestInd"] == "V").astype(np.float64)
    out["IsRevolver"] = revolver.map(_SBA_REVOLVER).astype(np.float64)
    out["HasCollateral"] = (raw["CollateralInd"] == "Y").astype(np.float64)
    out["IsFranchise"] = raw["FranchiseCode"].notna().astype(np.float64)
    naics = raw["NaicsCode"].astype("string").str[:2]
    out = out.join(one_hot(naics, "Naics2", 20))
    for field, top in (("BusinessType", 3), ("BusinessAge", 6), ("ProcessingMethod", 8)):
        out = out.join(one_hot(raw[field].astype("string"), field, top))
    out = out.assign(label=label.astype(np.int64), _approval_date=approval)
    return out.reset_index(drop=True)


def _days(stamps: pd.Series) -> np.ndarray:
    """Timestamps as float days since the Unix epoch."""
    return (stamps - pd.Timestamp("1970-01-01")).dt.total_seconds().to_numpy() / 86_400.0


def load_ppdai(root: Path) -> CreditPanel:
    """PPDai: the uploader's features less ``loan_time``, which orders the split.

    Args:
        root: Directory holding ``ppdai/ppdaiData.csv``.

    Returns:
        The panel, out-of-time on ``loan_time``, no groups, no embargo.
    """
    raw = pd.read_csv(root / "ppdai" / "ppdaiData.csv")
    feats = raw.drop(columns=["label", "loan_time"]).astype(np.float64)
    return CreditPanel(
        name="ppdai", X=feats.to_numpy(), feature_names=tuple(feats.columns),
        y=raw["label"].to_numpy(dtype=np.int64), time=raw["loan_time"].to_numpy(np.float64),
    )


def load_bondora(root: Path) -> CreditPanel:
    """Bondora: :func:`bondora_frame` less ``LoanYear``, out-of-time on ``LoanDate``.

    Args:
        root: Directory holding ``bondora/LoanData_Bondora.csv``.

    Returns:
        The panel, grouped by borrower, with a 366-day embargo (at least the 12-month label
        window, whatever the months' lengths).
    """
    raw = pd.read_csv(root / "bondora" / "LoanData_Bondora.csv", low_memory=False)
    frame = bondora_frame(raw)
    feats = frame.drop(columns=["label", "_user", "_loan_date", "LoanYear"])
    return CreditPanel(
        name="bondora", X=feats.to_numpy(np.float64), feature_names=tuple(feats.columns),
        y=frame["label"].to_numpy(np.int64), groups=frame["_user"].to_numpy(),
        time=_days(frame["_loan_date"]), embargo=366.0,
    )


def load_ulb_fraud(root: Path) -> CreditPanel:
    """ULB card fraud: ``Time``, ``V1``-``V28`` and ``Amount`` unchanged; ``Class`` is the label.

    Args:
        root: Directory holding ``ulb_fraud/creditcard.csv``.

    Returns:
        The panel, stratified (``Time`` is an offset inside a two-day capture, not a date).
    """
    raw = pd.read_csv(root / "ulb_fraud" / "creditcard.csv")
    feats = raw.drop(columns=["Class"]).astype(np.float64)
    return CreditPanel(
        name="ulb_fraud", X=feats.to_numpy(), feature_names=tuple(feats.columns),
        y=raw["Class"].to_numpy(dtype=np.int64),
    )


def load_sba(root: Path) -> CreditPanel:
    """SBA 7(a): :func:`sba_frame` less ``ApprovalFY``, out-of-time on ``ApprovalDate``.

    Args:
        root: Directory holding ``sba/`` + :data:`SBA_FILE`.

    Returns:
        The panel, with a :data:`SBA_EMBARGO_DAYS` embargo and no groups: the file's only
        borrower identity is name and address, which are personal data and never read into a
        feature or a group.
    """
    raw = pd.read_csv(root / "sba" / SBA_FILE, encoding_errors="replace", low_memory=False)
    frame = sba_frame(raw)
    feats = frame.drop(columns=["label", "_approval_date", "ApprovalFY"])
    return CreditPanel(
        name="sba", X=feats.to_numpy(np.float64), feature_names=tuple(feats.columns),
        y=frame["label"].to_numpy(np.int64), time=_days(frame["_approval_date"]),
        embargo=SBA_EMBARGO_DAYS,
    )


_UNVERIFIED = ("owner decision needed (DR4.2 rule): the grant is the uploader's, not the data "
               "owner's, and the original source states no licence")

#: The suite, in run order. Every entry is excluded until the owner admits it; see the module
#: docstring for the text and pages behind each verdict.
DATASETS: dict[str, SuiteDataset] = {
    "ppdai": SuiteDataset(
        "ppdai", "https://ndownloader.figshare.com/files/26049599", load_ppdai,
        SourceLicence(
            "https://api.figshare.com/v2/articles/13573871", "CC BY 4.0", "CC BY 4.0", False,
            f"{_UNVERIFIED}: figshare record by a researcher over PaiPaiDai's data; "
            "ai.ppdai.com unreachable, its 2026-05-10 Wayback copy gates the data behind a "
            "competition agreement", CHECKED,
        ),
    ),
    "bondora": SuiteDataset(
        "bondora", "https://www.kaggle.com/datasets/sid321axn/bondora-peer-to-peer-lending-loan-data",
        load_bondora,
        SourceLicence(
            "https://www.kaggle.com/datasets/sid321axn/bondora-peer-to-peer-lending-loan-data",
            "CC0: Public Domain", "CC0: Public Domain", False,
            f"{_UNVERIFIED}: Kaggle mirror by an individual; bondora.com/en/public-reports "
            "(Wayback 2021-07-24) states none and now redirects to goandgrow.eu, which offers "
            "no dataset", CHECKED,
        ),
    ),
    "ulb_fraud": SuiteDataset(
        "ulb_fraud", "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud", load_ulb_fraud,
        SourceLicence(
            "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud",
            "Database: Open Database, Contents: Database Contents", "ODbL-1.0 + DbCL-1.0",
            True,
            "admitted by the owner (2026-10-19): published by ULB's Machine Learning Group, "
            "the data's collector; scores are ODbL Produced Works, public with ODBL_NOTICE, "
            "and the database is never redistributed, so the share-alike term does not apply",
            CHECKED,
        ),
    ),
    "sba": SuiteDataset(
        "sba", "https://data.sba.gov/sites/default/files/uploaded_resources/" + SBA_FILE,
        load_sba,
        SourceLicence(
            "https://data.sba.gov/dataset/7a-504-foia", "U.S. Government Works",
            "U.S. Government Works (public domain)", True,
            "admitted by the owner (2026-10-20): the SBA is the data's producer and its "
            "catalogue page states U.S. Government Works, linking "
            "https://www.usa.gov/publicdomain/label/1.0/; borrower and lender identity are "
            "never read into a feature",
            SBA_CHECKED,
        ),
    ),
}


def out_of_time_splits(panel: CreditPanel) -> list[tuple[np.ndarray, np.ndarray]]:
    """The three out-of-time splits of a dated panel (module docstring, "Split protocol").

    Args:
        panel: A panel whose ``time`` is set.

    Returns:
        ``(train, test)`` row indices per split.

    Raises:
        ValueError: If the panel has no date.
    """
    if panel.time is None:
        raise ValueError(f"{panel.name}: out-of-time splits need a date")
    t = panel.time
    cuts = [*np.quantile(t, OOT_CUTS), np.inf]
    splits = []
    for k in range(N_SPLITS):
        start, end = cuts[k], cuts[k + 1]
        test = np.flatnonzero((t >= start) & (t < end))
        train_mask = t < start - panel.embargo
        if panel.groups is not None:
            train_mask &= ~np.isin(panel.groups, panel.groups[test])
        splits.append((np.flatnonzero(train_mask), test))
    return splits


def stratified_splits(panel: CreditPanel) -> list[tuple[np.ndarray, np.ndarray]]:
    """Three stratified folds, grouped by borrower when the panel has an id.

    Args:
        panel: Any panel.

    Returns:
        ``(train, test)`` row indices per fold, seeded by :data:`SPLIT_SEED`.
    """
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

    if panel.groups is not None:
        folds = StratifiedGroupKFold(N_SPLITS, shuffle=True, random_state=SPLIT_SEED)
        return list(folds.split(panel.X, panel.y, panel.groups))
    folds = StratifiedKFold(N_SPLITS, shuffle=True, random_state=SPLIT_SEED)
    return list(folds.split(panel.X, panel.y))


def suite_splits(panel: CreditPanel) -> list[tuple[np.ndarray, np.ndarray]]:
    """The panel's fixed splits: out-of-time when it has a date, stratified otherwise.

    Args:
        panel: Any panel.

    Returns:
        ``(train, test)`` row indices per split.

    Raises:
        ValueError: If a split's train or test side lacks a class, which would make it
            unscorable; the suite fails rather than reporting a NaN row.
    """
    splits = out_of_time_splits(panel) if panel.time is not None else stratified_splits(panel)
    for k, (tr, te) in enumerate(splits):
        for side, idx in (("train", tr), ("test", te)):
            if len(np.unique(panel.y[idx])) < 2:
                raise ValueError(f"{panel.name} split {k}: {side} side has one class "
                                 f"({len(idx)} rows)")
    return splits


def ap_top_decile(y_true: np.ndarray, p: np.ndarray) -> float:
    """Average precision earned inside the highest-scored tenth of the rows.

    With rows sorted by descending score and ``k = ceil(n / 10)``, this is
    ``sum(precision@i * y_i for i <= k) / min(k, n_positive)``: AP truncated at the top
    decile and normalised so a ranking that fills the decile with positives scores 1.0 at any
    base rate. Ties are broken by row order.

    Args:
        y_true: Binary outcomes.
        p: Scores, higher meaning riskier.

    Returns:
        The value in ``[0, 1]``; NaN when there are no positives.
    """
    y_true = np.asarray(y_true).astype(np.int64)
    positives = int(y_true.sum())
    if positives == 0 or len(y_true) == 0:
        return float("nan")
    k = math.ceil(len(y_true) / 10)
    hits = y_true[np.argsort(-np.asarray(p, dtype=float), kind="stable")[:k]]
    precision = np.cumsum(hits) / np.arange(1, k + 1)
    return float((precision * hits).sum() / min(k, positives))


def score_row(y_true: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """AP, ROC-AUC, top-decile AP and the level for one model on one test split.

    Args:
        y_true: Binary outcomes of the test rows.
        p: Predicted probability of the positive class.

    Returns:
        A dict with ``average_precision``, ``roc_auc``, ``ap_top_decile``, ``base_rate`` and
        ``mean_predicted``.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    return {
        "average_precision": float(average_precision_score(y_true, p)),
        "roc_auc": float(roc_auc_score(y_true, p)),
        "ap_top_decile": ap_top_decile(y_true, p),
        "base_rate": float(np.mean(y_true)),
        "mean_predicted": float(np.mean(p)),
    }


def inner_folds(
    y_train: np.ndarray, groups_train: np.ndarray | None, n_inner: int, seed: int
) -> np.ndarray:
    """Deal a split's training rows into stratified inner folds, by borrower when grouped.

    Args:
        y_train: ``(n_train,)`` labels of the training rows.
        groups_train: ``(n_train,)`` borrower id per training row, or ``None``: plain
            ``StratifiedKFold`` then, as for ULB and SBA, which have no borrower id.
        n_inner: Inner folds, at least 2.
        seed: The folds' ``random_state``.

    Returns:
        ``(n_train,)`` inner-fold index per training row, each in ``range(n_inner)``.
    """
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

    fold = np.full(len(y_train), -1, dtype=np.int64)
    if groups_train is not None:
        dealt = StratifiedGroupKFold(n_inner, shuffle=True, random_state=seed).split(
            np.zeros(len(y_train)), y_train, groups_train
        )
    else:
        dealt = StratifiedKFold(n_inner, shuffle=True, random_state=seed).split(
            np.zeros(len(y_train)), y_train
        )
    for k, (_, query) in enumerate(dealt):
        fold[query] = k
    return fold


def oof_fintfm_logits(
    X_train: np.ndarray,
    y_train: np.ndarray,
    train_rows: np.ndarray,
    test_rows: np.ndarray,
    groups: np.ndarray | None,
    make_classifier: Callable[[int], InContextClassifier],
    n_inner: int = INNER_FOLDS,
    seed: int = 0,
) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    """Out-of-fold FinTFM logits for every training row of one split.

    Each inner fold's rows are scored by a classifier whose context is the other inner folds'
    rows; :func:`~fintfm.experiments.hybrid_catboost.assert_no_leak` checks every context
    against the test rows and against its own queries before any fit runs.

    Args:
        X_train: ``(n_train, F)`` preprocessed training features, aligned with
            ``train_rows``.
        y_train: ``(n_train,)`` training labels, aligned with ``train_rows``.
        train_rows: Panel row indices of the split's training rows.
        test_rows: Panel row indices of the split's test rows, which no context may hold.
        groups: ``(n,)`` borrower id per **panel** row, or ``None``; when given, inner folds
            keep borrowers whole and the checks refuse a shared borrower too.
        make_classifier: Builds an unfitted classifier from the inner-fold index.
        n_inner: Inner folds.
        seed: Seed for dealing the training rows to inner folds.

    Returns:
        ``(logits, plan)``: ``(n_train,)`` logits aligned with ``train_rows``, and per inner
        fold the ``(context_rows, query_rows)`` panel indices it used, for checking.

    Raises:
        LeakageError: Before any fit, if a context holds a test row or a row it scores (or,
            with groups, a borrower of either).
        ValueError: If ``n_inner`` is below 2, the inputs are misaligned, or an inner context
            holds one class.
        RuntimeError: If a training row is left without a logit.
    """
    train_rows = np.asarray(train_rows)
    if n_inner < 2:
        raise ValueError(f"n_inner must be at least 2, got {n_inner}")
    if not len(X_train) == len(y_train) == len(train_rows):
        raise ValueError("X_train, y_train and train_rows must be aligned")
    fold = inner_folds(y_train, groups[train_rows] if groups is not None else None,
                       n_inner, seed)
    plan: list[tuple[np.ndarray, np.ndarray]] = []
    for k in range(n_inner):
        ctx, qry = train_rows[fold != k], train_rows[fold == k]
        assert_no_leak(ctx, test_rows, groups, f"inner fold {k} vs the test rows")
        assert_no_leak(ctx, qry, groups, f"inner fold {k} vs its own queries")
        plan.append((ctx, qry))
    logits = np.full(len(train_rows), np.nan)
    for k in range(n_inner):
        ctx_local, qry_local = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
        if len(np.unique(y_train[ctx_local])) < 2:
            raise ValueError(f"inner fold {k}'s context holds one class; lower --inner-folds")
        clf = make_classifier(k)
        clf.fit(X_train[ctx_local], y_train[ctx_local],
                groups=groups[train_rows[ctx_local]] if groups is not None else None)
        logits[qry_local] = to_logit(clf.predict_proba(X_train[qry_local])[:, 1])
    if np.isnan(logits).any():
        raise RuntimeError("a training row received no out-of-fold logit")
    return logits, plan


def _write(rows: list[dict], out: Path) -> None:
    """Write every row so far, so a run that dies keeps what it finished."""
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def run(
    out: Path,
    model_path: str | None,
    datasets: Sequence[str] | None = None,
    root: Path = DEFAULT_ROOT,
    config_path: str | None = None,
    device: str = "cpu",
    with_lightgbm: bool = True,
    hybrid: bool = False,
    n_inner: int = INNER_FOLDS,
) -> list[dict]:
    """Score every model on every split of the named datasets and write the CSV.

    Args:
        out: CSV path; rewritten after every scored row.
        model_path: FinTFM checkpoint, or ``None`` for the baselines alone.
        datasets: Keys of :data:`DATASETS`, scored in this order; ``None`` scores the
            admitted ones (:func:`admitted`).
        root: Directory holding each dataset's raw files under ``<key>/``.
        config_path: Configuration override; its ``inference`` and ``large_n`` sections build
            the FinTFM arm, as in the V4FinBench protocol.
        device: Torch device for FinTFM.
        with_lightgbm: Fit LightGBM at :data:`LIGHTGBM_PARAMS`; off only for tests and
            machines without the ``bench`` extra.
        hybrid: Also score :data:`HYBRID_MODELS` (module docstring, "Hybrid arms"); needs
            ``model_path`` and ``with_lightgbm``.
        n_inner: Inner folds for the hybrid arms' out-of-fold FinTFM logits.

    Returns:
        The rows written, one per dataset, split and model.

    Raises:
        ValueError: On an unknown dataset, or one whose licence entry does not admit it --
            checked for every dataset before any file is read; or on ``hybrid`` without a
            checkpoint or LightGBM, or with ``n_inner`` below 2.
    """
    if hybrid and (model_path is None or not with_lightgbm):
        raise ValueError("the hybrid arms need a FinTFM checkpoint and LightGBM")
    if hybrid and n_inner < 2:
        raise ValueError(f"n_inner must be at least 2, got {n_inner}")
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    from fintfm.config import load_config
    from fintfm.modeling.model import FinancialTFM
    from fintfm.provenance import sha256_file

    datasets = admitted() if datasets is None else tuple(datasets)
    unknown = [d for d in datasets if d not in DATASETS]
    if unknown:
        raise ValueError(f"unknown dataset(s) {unknown}; the suite is {list(DATASETS)}")
    refused = [d for d in datasets if not DATASETS[d].licence.include]
    if refused:
        raise ValueError(
            "licence not admitted for " + "; ".join(
                f"{d}: {DATASETS[d].licence.reason}" for d in refused
            )
        )
    # DR4.1's one selection rule: a suite that may choose arms never holds a TabArena panel
    guard_selection(datasets, allow_reporting=False, tool="credit_suite")
    cfg = load_config(config_path)
    model = FinancialTFM.load(model_path) if model_path is not None else None
    sha = sha256_file(model_path) if model_path is not None else ""
    rows: list[dict] = []
    for key in datasets:
        panel = DATASETS[key].loader(root)
        print(f"{key}: {len(panel.y):,} rows, {panel.X.shape[1]} features, "
              f"{int(panel.y.sum()):,} positive ({panel.y.mean():.3%}), {panel.split_kind}",
              flush=True)
        for k, (tr, te) in enumerate(suite_splits(panel)):
            y_tr, y_te = panel.y[tr], panel.y[te]
            # preprocessing fitted on the training side alone, as in the V4FinBench protocol
            imputer = SimpleImputer(strategy="median").fit(panel.X[tr])
            scaler = StandardScaler().fit(imputer.transform(panel.X[tr]))
            Xtr, Xte = (scaler.transform(imputer.transform(panel.X[i])) for i in (tr, te))
            models = ["logistic_regression"] + (["lightgbm_1000"] if with_lightgbm else [])
            if model is not None:
                models.insert(0, "fintfm")
            if hybrid:
                models += list(HYBRID_MODELS)
            p_fintfm: np.ndarray | None = None  # the fintfm row's test scores, for the hybrids
            oof: tuple[np.ndarray, float] | str | None = None  # (logits, seconds) or why not
            for name in models:
                uses_checkpoint = name == "fintfm" or name in HYBRID_MODELS
                row = {
                    "dataset": key, "split_kind": panel.split_kind, "split": k, "model": name,
                    "n_train": len(tr), "n_test": len(te), "n_test_positive": int(y_te.sum()),
                    "checkpoint_sha256": sha if uses_checkpoint else "", "skipped": "",
                }
                if name == "fintfm" and model.cfg.max_features < Xtr.shape[1]:
                    row["skipped"] = (f"checkpoint takes {model.cfg.max_features} features, "
                                      f"data has {Xtr.shape[1]}")
                elif name in HYBRID_MODELS and p_fintfm is None:
                    row["skipped"] = "needs the fintfm row's test scores, which were not made"
                elif name in HYBRID_MODELS:
                    if oof is None:
                        oof = _hybrid_oof(panel, tr, te, Xtr, y_tr, model, cfg, device, k,
                                          n_inner)
                    if isinstance(oof, str):
                        row["skipped"] = oof
                    else:
                        t0 = time.perf_counter()
                        p = _predict_hybrid(name, Xtr, y_tr, Xte, oof[0], to_logit(p_fintfm))
                        row["seconds"] = round(oof[1] + time.perf_counter() - t0, 1)
                        if p is None:
                            row["skipped"] = "fit failed (printed above)"
                        else:
                            row.update(score_row(y_te, p))
                else:
                    t0 = time.perf_counter()
                    groups = panel.groups[tr] if panel.groups is not None else None
                    p = _predict(name, Xtr, y_tr, Xte, groups, model, cfg, device, seed=k)
                    row["seconds"] = round(time.perf_counter() - t0, 1)
                    if p is None:
                        row["skipped"] = "fit failed (printed above)"
                    else:
                        row.update(score_row(y_te, p))
                        if name == "fintfm":
                            p_fintfm = p
                rows.append(row)
                _write(rows, out)
                _report(row)
    _summarise(rows)
    if "ulb_fraud" in datasets:
        # ODbL-1.0 section 4.3: a public Produced Work carries this notice
        Path(out).with_suffix(".NOTICE.txt").write_text(ODBL_NOTICE + "\n")
        print(ODBL_NOTICE, flush=True)
    return rows


def _predict(
    name: str, Xtr: np.ndarray, y_tr: np.ndarray, Xte: np.ndarray, groups: np.ndarray | None,
    model: FinancialTFM | None, cfg: Config, device: str, seed: int,
) -> np.ndarray | None:
    """Fit one model on the training side and score the test side.

    Args:
        name: ``"fintfm"``, ``"logistic_regression"`` or ``"lightgbm_1000"``.
        Xtr: Imputed and scaled training features.
        y_tr: Training labels.
        Xte: Test features, transformed as ``Xtr``.
        groups: Borrower id per training row, used only by FinTFM fine-tuning; or ``None``.
        model: The loaded checkpoint, for ``"fintfm"``.
        cfg: Configuration; its ``inference`` and ``large_n`` sections build the FinTFM arm.
        device: Torch device for FinTFM.
        seed: The FinTFM classifier's ``random_state``; the split index.

    Returns:
        ``(n_test,)`` positive-class probabilities, or ``None`` when the out-of-process
        LightGBM fit failed (:func:`fintfm.evaluation.boosting.fit_predict_boosting` prints
        why).
    """
    if name == "fintfm":
        from fintfm.experiments.v4_protocol import fintfm_classifier
        from fintfm.inference.large_n import wrap_large_n

        clf = wrap_large_n(fintfm_classifier(model, cfg, device=device, random_state=seed),
                           cfg.large_n)
        clf.fit(Xtr.astype(np.float32), y_tr, groups=groups)
        return clf.predict_proba(Xte.astype(np.float32))[:, 1]
    if name == "logistic_regression":
        from sklearn.linear_model import LogisticRegression

        # the V4FinBench protocol's untuned logistic regression (`_classical_estimator`)
        return LogisticRegression(max_iter=1000).fit(Xtr, y_tr).predict_proba(Xte)[:, 1]
    if name == "lightgbm_1000":
        from fintfm.evaluation.boosting import fit_predict_boosting

        return fit_predict_boosting("lightgbm", Xtr, y_tr, Xte, params=LIGHTGBM_PARAMS)
    raise ValueError(f"unknown model {name!r}")


def _hybrid_oof(
    panel: CreditPanel, tr: np.ndarray, te: np.ndarray, Xtr: np.ndarray, y_tr: np.ndarray,
    model: FinancialTFM, cfg: Config, device: str, split: int, n_inner: int,
) -> tuple[np.ndarray, float] | str:
    """One split's out-of-fold FinTFM logits for the hybrid arms, with the cost printed.

    Each inner classifier is built exactly as the ``fintfm`` row's (``inference`` and
    ``large_n`` sections), seeded ``split + INNER_SEED_OFFSET * (inner fold + 1)`` so no inner
    seed coincides with the outer one.

    Args:
        panel: The dataset, for its borrower ids.
        tr: Panel indices of the split's training rows.
        te: Panel indices of the split's test rows.
        Xtr: Imputed and scaled training features, aligned with ``tr``.
        y_tr: Training labels, aligned with ``tr``.
        model: The loaded checkpoint.
        cfg: Configuration.
        device: Torch device for FinTFM.
        split: The split index.
        n_inner: Inner folds.

    Returns:
        ``(logits, seconds)``, or the reason both hybrid rows are skipped when an inner
        context holds one class. A :class:`~fintfm.experiments.hybrid_catboost.LeakageError`
        is never caught.
    """
    from fintfm.experiments.v4_protocol import fintfm_classifier
    from fintfm.inference.large_n import wrap_large_n

    print(f"  split {split} hybrid: {n_inner} inner FinTFM fits scoring {len(tr):,} training "
          f"rows out of fold, each context drawn from about {len(tr) * (n_inner - 1) // n_inner:,}"
          f" rows (<= {cfg.inference.max_context:,} sampled)"
          + (", grouped by borrower" if panel.groups is not None else ", no borrower id"),
          flush=True)
    t0 = time.perf_counter()
    try:
        logits, _ = oof_fintfm_logits(
            Xtr.astype(np.float32), y_tr, tr, te, panel.groups,
            lambda j: wrap_large_n(
                fintfm_classifier(model, cfg, device=device,
                                  random_state=split + INNER_SEED_OFFSET * (j + 1)),
                cfg.large_n,
            ),
            n_inner=n_inner, seed=SPLIT_SEED + INNER_SEED_OFFSET + split,
        )
    except ValueError as e:
        print(f"  split {split} hybrid: skipped, {e}", flush=True)
        return f"inner folds unusable: {e}"
    seconds = time.perf_counter() - t0
    print(f"  split {split} hybrid: out-of-fold logits in {seconds:.1f}s", flush=True)
    return logits, seconds


def _predict_hybrid(
    name: str, Xtr: np.ndarray, y_tr: np.ndarray, Xte: np.ndarray, oof_train: np.ndarray,
    logit_test: np.ndarray,
) -> np.ndarray | None:
    """Fit one hybrid arm's LightGBM out of process and score the test side.

    Args:
        name: One of :data:`HYBRID_MODELS`.
        Xtr: Imputed and scaled training features.
        y_tr: Training labels.
        Xte: Test features, transformed as ``Xtr``.
        oof_train: ``(n_train,)`` out-of-fold FinTFM logits for the training rows.
        logit_test: ``(n_test,)`` FinTFM logits for the test rows, from the ``fintfm`` row.

    Returns:
        ``(n_test,)`` positive-class probabilities, or ``None`` when the fit failed
        (:func:`fintfm.evaluation.boosting.fit_predict_boosting` prints why).

    Raises:
        ValueError: For a name outside :data:`HYBRID_MODELS`.
    """
    from fintfm.evaluation.boosting import fit_predict_boosting

    if name == "lightgbm_fintfm_feature":
        return fit_predict_boosting(
            "lightgbm", np.column_stack([Xtr, oof_train]), y_tr,
            np.column_stack([Xte, logit_test]), params=LIGHTGBM_PARAMS,
        )
    if name == "lightgbm_fintfm_baseline":
        return fit_predict_boosting(
            "lightgbm", Xtr, y_tr, Xte, params=LIGHTGBM_PARAMS,
            init_score_train=oof_train, init_score_test=logit_test,
        )
    raise ValueError(f"unknown hybrid model {name!r}")


def _report(row: dict) -> None:
    """Print one scored row, the level beside the ranking."""
    head = f"  split {row['split']} {row['model']:>24}:"
    if row["skipped"]:
        print(f"{head} skipped ({row['skipped']})", flush=True)
        return
    print(f"{head} AP {row['average_precision']:.4f}  ROC-AUC {row['roc_auc']:.4f}  "
          f"AP@decile {row['ap_top_decile']:.4f}  pred mean {row['mean_predicted']:.3%} vs "
          f"{row['base_rate']:.3%}  ({row['n_test_positive']:,}/{row['n_test']:,} positive, "
          f"{row['n_train']:,} train, {row['seconds']}s)", flush=True)


def _summarise(rows: list[dict]) -> None:
    """Print each dataset and model's mean AP, ROC-AUC and top-decile AP over its splits."""
    frame = pd.DataFrame([r for r in rows if not r["skipped"]])
    if frame.empty:
        return
    means = frame.groupby(["dataset", "model"], sort=False)[
        ["average_precision", "roc_auc", "ap_top_decile"]
    ].agg(["mean", "count"])
    print("\nmean over splits (count = splits scored):")
    print(means.round(4).to_string())


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point: ``python -m fintfm.evaluation.credit_suite --model ... --out ...``."""
    p = argparse.ArgumentParser(description="Credit evaluation suite outside V4FinBench (DR4.3).")
    p.add_argument("--model", default=None,
                   help="FinTFM checkpoint; omit to score the baselines alone")
    p.add_argument("--out", required=True, type=Path, help="CSV to write")
    p.add_argument("--datasets", default=",".join(admitted()),
                   help=f"comma-separated subset of {','.join(DATASETS)} "
                        f"(default: the admitted ones, {','.join(admitted())})")
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                   help="directory holding <dataset>/<raw file>")
    p.add_argument("--config", default=None, help="configuration override (inference section)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--no-lightgbm", action="store_true")
    p.add_argument("--hybrid", action="store_true",
                   help="also score LightGBM with FinTFM's out-of-fold logit as a feature and "
                        "as its starting score (DR8.4); needs --model")
    p.add_argument("--inner-folds", type=int, default=INNER_FOLDS,
                   help=f"inner folds for the hybrid arms' out-of-fold logits "
                        f"(default {INNER_FOLDS})")
    args = p.parse_args(argv)
    if args.hybrid and (args.model is None or args.no_lightgbm):
        p.error("--hybrid needs --model and LightGBM")
    if args.inner_folds < 2:
        p.error("--inner-folds must be at least 2")
    run(
        args.out, args.model, tuple(filter(None, args.datasets.split(","))), args.root,
        args.config, args.device, with_lightgbm=not args.no_lightgbm, hybrid=args.hybrid,
        n_inner=args.inner_folds,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
