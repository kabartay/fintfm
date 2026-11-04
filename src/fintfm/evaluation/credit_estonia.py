"""Estonian e-Business Register: firm-year bankruptcy panel for the DR4.3 credit suite.

Task DR4.3 (design review 2026-10-16). Evaluation only: the data stays under ``data/estonia/``
(gitignored), ``estonia`` is in :data:`fintfm.prior.real_panel.EVAL_ONLY_SOURCES`, and the
suite entry in :mod:`fintfm.evaluation.credit_suite` is registered with ``include=False``
(see "What the open data cannot give" below).

Source and licence
------------------
Register of Information Systems (RIK) open data, ``https://avaandmed.ariregister.rik.ee/en/
downloading-open-data``, read 2026-11-04. The pages link ``https://creativecommons.org/
licenses/by/4.0/legalcode`` as "License information" (CC BY 4.0), RIK being the register's
operator and the data's producer; the download page says it "requires acceptance of the
license terms". No sentence there reads "this dataset is licensed under CC BY 4.0", the link
is the whole statement. The files hold personal data (representatives, shareholders) which
this module never reads; GDPR applies to a re-user of it.

Files read, all from that page (about 0.6 GB zipped, 10 GB unzipped for the annual reports,
registry cards and simple data), placed unzipped in ``root/estonia/``:

* ``1.aruannete_yldandmed_kuni_30092026.csv``: one row per filed annual report.
* ``4.<year>_aruannete_elemendid_kuni_30092026.csv`` for 2019-2025: key indicators per report
  (balance sheet, income statement, labour expense, headcount), non-consolidated.
* ``ettevotja_rekvisiidid__registrikaardid.json``: registry-card entries per company, from
  which the ``Pankroti väljakuulutamiskanne`` (bankruptcy declaration) dates are read.
* ``ettevotja_rekvisiidid__lihtandmed.csv``: the date of first entry, for firm age.

What the open data cannot give
------------------------------
The download page states that the files contain only companies **currently** in the register
(statuses entered, in liquidation, in bankruptcy). Three consequences, MEASURED on the files
of 2026-11-04 (counts in ``docs/results/drafts/dr4.3-estonia_2026-11.md``):

1. A firm declared bankrupt and then deleted from the register has no registry card, so its
   bankruptcy date is absent. The cards hold only 765 declarations, 85 of them before 2019,
   and the yearly count rises toward the present, which is what deletion looks like.
2. The annual-report files do keep deleted firms (status ``Kustutatud``), but without a
   deletion date and without a reason: a firm deleted after bankruptcy, after a voluntary
   liquidation or for not filing look the same. Their label is unobservable and
   :func:`estonia_frame` drops them.
3. Positives are therefore firms whose proceedings are still open, and negatives are firms
   alive in 2026. Both are selected on 2026 state, so the label is conditioned on a future
   the model would not have had. This is survivor-conditioned, not a default label.

The panel is built anyway so the owner can see the numbers, and it is gated off. Do not
quote a score from it as a bankruptcy-prediction result.

Label
-----
One row per firm-year report that is non-consolidated, not PDF-only, and not from a firm
whose current status is deleted. ``y = 1`` when a bankruptcy declaration is dated in
``(filed, filed + 365 days]``. A report filed on or after a declaration (a firm already in
proceedings) is dropped. A report whose window ends after :data:`DATA_CUT` (2026-09-30, the
annual-report files' cut) is dropped as censored. Features are the report's own indicators
and ratios, ``log(filing date - first entry)`` and the legal form; the register's *current*
status is never a feature, since it contains the label.

Split (the suite's out-of-time protocol, :func:`fintfm.evaluation.credit_suite.
out_of_time_splits`): ``time`` is the filing date; test windows start at the 1/2, 2/3 and
5/6 quantiles; the embargo is 365 days, the label window, so every training label is known
when a test window opens; ``groups`` is the registry code, so a firm is never on both sides.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from fintfm.evaluation.credit_suite import CreditPanel

#: Label window after the filing date, days.
HORIZON_DAYS = 365
#: Last date the annual-report files cover; windows ending later are censored.
DATA_CUT = pd.Timestamp("2026-09-30")
#: Embargo for the out-of-time split: the label window.
EMBARGO_DAYS = float(HORIZON_DAYS)
#: Name of the registry-card entry that declares bankruptcy.
BANKRUPTCY_ENTRY = "Pankroti väljakuulutamiskanne"
#: Status the reports file gives a firm no longer in the register.
DELETED = "Kustutatud"

REPORTS_FILE = "1.aruannete_yldandmed_kuni_30092026.csv"
ELEMENT_FILE = "4.{year}_aruannete_elemendid_kuni_30092026.csv"
ELEMENT_YEARS = tuple(range(2019, 2026))
CARDS_FILE = "ettevotja_rekvisiidid__registrikaardid.json"
BASIC_FILE = "ettevotja_rekvisiidid__lihtandmed.csv"

#: Key indicators read from the element files (``elemendi_nimetus``).
ELEMENTS = (
    "Assets", "CurrentAssets", "NonCurrentAssets", "CashAndCashEquivalents", "Equity",
    "IssuedCapital", "CurrentLiabilities", "NonCurrentLiabilities", "Revenue",
    "EmployeeExpense", "LaborExpense", "DepreciationAndImpairmentLossReversal",
    "TotalAnnualPeriodProfitLoss", "TotalProfitLossBeforeTax", "RetainedEarningsLoss",
    "AverageNumberOfEmployeesInFullTimeEquivalentUnits",
)
#: Tables whose rows describe a liquidation or a profit-allocation proposal, not the going
#: concern's statements; their elements are skipped.
_SKIP_TABLE = re.compile(r"^(Likvideerimise|Kasumi jaotamise|Kahjumi katmise|Rahavoogude)")
#: Legal forms kept as one-hot columns; the rest fall in no column.
LEGAL_FORMS = ("OÜ", "AS", "FIE", "TÜ", "MTÜ", "SA")
_RATIOS = (
    ("equity_to_assets", "Equity", "Assets"),
    ("current_ratio", "CurrentAssets", "CurrentLiabilities"),
    ("liabilities_to_assets", "_liabilities", "Assets"),
    ("cash_to_assets", "CashAndCashEquivalents", "Assets"),
    ("net_profit_to_assets", "TotalAnnualPeriodProfitLoss", "Assets"),
    ("pretax_profit_to_assets", "TotalProfitLossBeforeTax", "Assets"),
    ("revenue_to_assets", "Revenue", "Assets"),
    ("net_margin", "TotalAnnualPeriodProfitLoss", "Revenue"),
    ("labour_to_revenue", "_labour", "Revenue"),
    ("retained_to_assets", "RetainedEarningsLoss", "Assets"),
)


def _date(col: pd.Series) -> pd.Series:
    """Parse the register's ``dd.mm.yyyy`` dates; unparseable values become NaT."""
    return pd.to_datetime(col, format="%d.%m.%Y", errors="coerce")


def read_reports(path: Path) -> pd.DataFrame:
    """Read the annual-report list: one row per filed report.

    Args:
        path: The ``1.aruannete_yldandmed`` CSV (semicolon-separated, quoted).

    Returns:
        Columns ``report_id``, ``code`` (registry code), ``form``, ``status``, ``year``,
        ``audited`` (0/1), ``filed`` and ``period_end`` (timestamps), ``consolidated``
        (bool).
    """
    raw = pd.read_csv(path, sep=";", dtype=str, usecols=[
        "report_id", "registrikood", "õiguslik vorm", "staatus", "aruandeaasta",
        "kas konsolideeritud?", "period_end", "esitatud_kpv", "kas auditeeritud?",
    ])
    return pd.DataFrame({
        "report_id": raw["report_id"].astype(np.int64),
        "code": pd.to_numeric(raw["registrikood"], errors="coerce"),
        "form": raw["õiguslik vorm"], "status": raw["staatus"],
        "year": pd.to_numeric(raw["aruandeaasta"], errors="coerce"),
        "audited": (raw["kas auditeeritud?"] == "Jah").astype(np.int8),
        "filed": _date(raw["esitatud_kpv"]), "period_end": _date(raw["period_end"]),
        "consolidated": raw["kas konsolideeritud?"] == "Jah",
    })


def read_elements(root: Path, report_ids: set[int]) -> pd.DataFrame:
    """Read the key indicators of the wanted reports into one row per report.

    Args:
        root: Directory holding the ``4.<year>_aruannete_elemendid`` CSVs.
        report_ids: Reports to keep; other rows are dropped chunk by chunk.

    Returns:
        ``report_id`` index, one float column per :data:`ELEMENTS` entry (NaN when the
        report lacks it). Where an element appears in several tables the first row wins.
    """
    parts = []
    ids = np.fromiter(report_ids, dtype=np.int64)
    for year in ELEMENT_YEARS:
        path = root / ELEMENT_FILE.format(year=year)
        if not path.exists():
            continue
        for chunk in pd.read_csv(path, sep=";", dtype=str, chunksize=2_000_000,
                                 usecols=["report_id", "tabel", "elemendi_nimetus", "vaartus"]):
            chunk = chunk[chunk["elemendi_nimetus"].isin(ELEMENTS)]
            chunk = chunk[~chunk["tabel"].str.match(_SKIP_TABLE)]
            chunk = chunk.assign(report_id=chunk["report_id"].astype(np.int64))
            chunk = chunk[chunk["report_id"].isin(ids)]
            parts.append(chunk.drop(columns="tabel"))
    long = pd.concat(parts, ignore_index=True)
    long["vaartus"] = pd.to_numeric(long["vaartus"], errors="coerce")
    long = long.drop_duplicates(["report_id", "elemendi_nimetus"], keep="first")
    wide = long.pivot(index="report_id", columns="elemendi_nimetus", values="vaartus")
    return wide.reindex(columns=list(ELEMENTS))


def read_bankruptcies(path: Path) -> pd.DataFrame:
    """Read bankruptcy-declaration dates from the registry-card JSON, line by line.

    The file is pretty-printed (about 0.6 GB), so a line scan avoids loading it as one
    object. A company's code sits at eight-space indent; its entries' dates follow their
    ``kpv`` key.

    Args:
        path: The ``registrikaardid`` JSON.

    Returns:
        Columns ``code`` and ``declared`` (timestamp), one row per declaration.
    """
    code: int | None = None
    kpv: str | None = None
    rows: list[tuple[int | None, str | None]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith('        "ariregistri_kood":'):
                code = int(line.split(":")[1].strip().rstrip(","))
            elif '"kpv":' in line:
                kpv = line.split(":", 1)[1].strip().strip(",").strip('"')
            elif BANKRUPTCY_ENTRY in line:
                rows.append((code, kpv))
    out = pd.DataFrame(rows, columns=["code", "declared"])
    out["declared"] = _date(out["declared"])
    return out.dropna()


def estonia_frame(
    reports: pd.DataFrame, elements: pd.DataFrame, bankruptcies: pd.DataFrame,
    first_entry: pd.Series, horizon_days: int = HORIZON_DAYS, cut: pd.Timestamp = DATA_CUT,
) -> pd.DataFrame:
    """Join the tables into a labelled firm-year frame (module docstring, "Label").

    Args:
        reports: :func:`read_reports` output.
        elements: :func:`read_elements` output.
        bankruptcies: :func:`read_bankruptcies` output.
        first_entry: Date of first register entry by registry code (``lihtandmed``).
        horizon_days: Label window after the filing date.
        cut: Last date covered; a window ending after it is dropped as censored.

    Returns:
        One row per kept report: ``label``, ``_code``, ``_filed``, the legal-form and ratio
        features, and the raw indicators. Earliest filing wins per firm and year.
    """
    r = reports[~reports["consolidated"] & (reports["status"] != DELETED)]
    r = r.dropna(subset=["code", "filed", "year"]).sort_values("filed")
    r = r.drop_duplicates(["code", "year"], keep="first")
    r = r.join(elements, on="report_id", how="inner").dropna(subset=["Assets"])
    first = bankruptcies.sort_values("declared").drop_duplicates("code").set_index("code")
    declared = r["code"].map(first["declared"])
    window_end = r["filed"] + pd.Timedelta(days=horizon_days)
    r = r[~(declared.reindex(r.index) <= r["filed"]) & (window_end <= cut)]
    declared = declared.reindex(r.index)
    out = pd.DataFrame(index=r.index)
    out["label"] = ((declared > r["filed"])
                    & (declared <= r["filed"] + pd.Timedelta(days=horizon_days))).astype(int)
    out["_code"], out["_filed"] = r["code"].astype(np.int64), r["filed"]
    v = {c: r[c] for c in ELEMENTS}
    both = r[["CurrentLiabilities", "NonCurrentLiabilities"]]
    v["_liabilities"] = both.sum(axis=1, min_count=1)
    v["_labour"] = r["LaborExpense"].fillna(r["EmployeeExpense"])
    for name, num, den in _RATIOS:
        out[name] = (v[num] / v[den].where(v[den].abs() > 0)).replace([np.inf, -np.inf], np.nan)
    out["log_assets"] = np.log1p(r["Assets"].clip(lower=0))
    out["log_revenue"] = np.log1p(r["Revenue"].clip(lower=0))
    out["log_age_days"] = np.log1p(
        (r["filed"] - r["code"].map(first_entry)).dt.days.clip(lower=0))
    out["audited"] = r["audited"]
    out["employees_fte"] = r["AverageNumberOfEmployeesInFullTimeEquivalentUnits"]
    for form in LEGAL_FORMS:
        out[f"form_{form}"] = (r["form"] == form).astype(np.int8)
    for c in ("Equity", "IssuedCapital", "RetainedEarningsLoss", "TotalAnnualPeriodProfitLoss"):
        out[f"{c}_sign"] = np.sign(r[c])
    return out


def load_estonia(root: Path) -> CreditPanel:
    """Estonian register firm-years: :func:`estonia_frame`, out-of-time on filing date.

    Args:
        root: Suite root; the files sit in ``root / "estonia"``.

    Returns:
        The panel, grouped by registry code, with a :data:`EMBARGO_DAYS` embargo.
    """
    from fintfm.evaluation.credit_suite import CreditPanel, _days

    d = root / "estonia"
    reports = read_reports(d / REPORTS_FILE)
    kept = reports[~reports["consolidated"] & (reports["status"] != DELETED)]
    elements = read_elements(d, set(kept["report_id"]))
    basic = pd.read_csv(d / BASIC_FILE, sep=";", dtype=str, encoding="utf-8-sig",
                        usecols=["ariregistri_kood", "ettevotja_esmakande_kpv"])
    first = pd.Series(_date(basic["ettevotja_esmakande_kpv"]).to_numpy(),
                      index=pd.to_numeric(basic["ariregistri_kood"], errors="coerce"))
    first = first[~first.index.duplicated()]
    frame = estonia_frame(reports, elements, read_bankruptcies(d / CARDS_FILE), first)
    feats = frame.drop(columns=["label", "_code", "_filed"]).astype(np.float64)
    return CreditPanel(
        name="estonia", X=feats.to_numpy(), feature_names=tuple(feats.columns),
        y=frame["label"].to_numpy(np.int64), groups=frame["_code"].to_numpy(),
        time=_days(frame["_filed"]), embargo=EMBARGO_DAYS,
    )
