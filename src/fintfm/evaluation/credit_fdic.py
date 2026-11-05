"""FDIC bank failures for the DR4.3 credit suite: will an insured bank fail within a year?

Evaluation only, never training (``docs/research/data_candidates.csv``: ``EVAL_ONLY``;
:data:`fintfm.prior.real_panel.EVAL_ONLY_SOURCES` makes the FinTFM-R sampler refuse a panel in
``data/real/fdic``). The files stay under ``data/real/fdic/`` (gitignored) and never enter this
repository. This is a corporate-distress table of US banks, a different object from the
consumer-credit tables of :mod:`fintfm.evaluation.credit_suite`.

**FinTFM-only until the SEC-filer check is done.** ``CLAUDE.md`` requires a check for overlap
with SEC EDGAR filers before any FinTFM-R number is computed on a new benchmark. Many bank
holding companies file 10-K and 10-Q with the SEC, and FinTFM-R's pretraining panels start on
EDGAR, so this dataset may share firms with them (the bank-level financials here are the Call
Report, filed with the FDIC, not the holding company's filing, but the economic entity
overlaps). That check has not been run. Until it is, score this dataset with FinTFM
(synthetic-only) and the baselines only; do not compute a FinTFM-R number on it.

Source and licence (read 2026-11-04)
------------------------------------

BankFind Suite's API (``https://api.fdic.gov/banks/docs/``, FDIC "Bank Data API (Beta)",
``/failures`` and ``/financials``). Terms as stated: the API page states no licence,
copyright, public-domain or attribution term for the data; its only usage text is that keys
"are currently not required" and that the FDIC "is not responsible for the contents of any
linked site". The FDIC's website policies (``https://www.fdic.gov/about/website-policies``)
likewise state no licence for its data; the only reuse text is that external sites "may contain
information that is copyrighted with restrictions on reuse" and that permission for such
material "cannot be obtained from the FDIC". The FDIC is the data's producer, a US federal
agency, so the work is a US government work (17 U.S.C. section 105; stated on the SBA page the
suite already uses, not on the FDIC's). DR4.2's rule is decided on what the source
states and this source states nothing, so the public-domain basis is statute, not a grant at
the source; the owner admitted it on that basis on 2026-11-05 (``include=True``), as a named
exception to the rule. The FFIEC's Call Report bulk files
are not used. Only the API's ``failures`` list and its quarterly ``financials`` summary
(about 160 fields per bank-quarter, a subset of the Call Report) are read.

Label, horizon and embargo
--------------------------

One row per bank-quarter: the ``financials`` record for a ``REPDTE`` (quarter-end report date)
from :data:`FIRST_REPDTE` (2000-03-31) on. Features are that quarter's Call Report ratios and
levels (:data:`NUMERIC_FIELDS`, bank class, state, institution age). ``label`` is 1 when the
bank's ``RESTYPE == "FAILURE"`` record in the failures list has a ``FAILDATE`` inside the
horizon window and 0 otherwise (a merger, a voluntary closure or survival all count as 0).

* **Horizon** :data:`HORIZON_QUARTERS` = 4, counted as :data:`HORIZON_DAYS` = 366 calendar days.
* **Reporting lag** :data:`LAG_DAYS` = 35: a Call Report is due 30 days after quarter end, 35
  for banks with foreign offices, so the window opens at ``REPDTE`` + 35 days, when the
  features were first public. A failure inside the lag itself is not predictable from the
  features, so the row is dropped, not labelled either way (:func:`fdic_frame` counts them).
* **Censoring**: a row whose window ends after ``as_of`` (the date the failures list was
  retrieved, ``meta.json``) is dropped, so every label is final.
* **Embargo** :data:`EMBARGO_DAYS` = 401 (lag plus horizon): a training row's outcome is known
  only after ``REPDTE`` + 401 days, so a split trains only on rows at least that far before its
  test window opens (the shared splitter of :mod:`fintfm.evaluation.credit_suite`).
* Never features: ``ACTIVE``, ``ENDEFYMD``, ``EFFDATE``, ``RISDATE``, ``INSAGNT1`` (current
  status or end dates, which give the label away), names and the failures list's own fields.

Splits, and the 2008-2012 cluster
---------------------------------

Out-of-time on ``REPDTE``, and the failures cluster: 465 of the 581 failures since 2000
(80 per cent, MEASURED 2026-11-04) fall in 2008-2012 (year counts in the draft). The two splits below answer different
questions, and the cluster is handled by saying which side it lands on, not by hiding it.

* **Shared suite splits** (``credit_suite.suite_splits``, cuts at the 1/2, 2/3 and 5/6
  quantiles of ``REPDTE``, MEASURED 2026-11-04): split 0 tests on 2010-03-31 to 2013-12-31,
  the tail of the crisis (873 positive rows), and trains on rows up to the embargo before it
  (580 positive rows, mostly 2007-2008 failures); splits 1 and 2 test on the calm years
  2014-2018 (109 positive rows) and 2019-2025 (66), with the crisis wholly in training. So
  the three splits mix two questions: split 0 is the crisis tail, splits 1 and 2 a model
  fitted through a crisis on a calm regime. Report them per split, never as one mean.
  :func:`~fintfm.evaluation.credit_suite.suite_splits` refuses a split that leaves one
  class on either side rather than scoring it.
* :func:`crisis_split` (this module): trains on rows before 2007 and tests on the crisis
  quarters 2008-2011, with the embargo (MEASURED: 254,805 training rows, 102 positive; 128,495
  test rows, 1,771 positive). It tests a model fitted on a calm regime on the crisis, the
  distribution shift that matters for a bank-failure model, with many test positives and few
  training positives.

**Institution grouping.** A bank appears in up to 105 quarters, so the borrower rule of
Bondora (drop training rows of anyone in the test window) would leave only banks that exited
before the window and a training set of mostly failures. The suite panel therefore carries no
groups, and bank identity is controlled by time instead: the label is final, the embargo
removes training rows whose outcome overlaps the test window, and a bank fails once, so a
failed bank's own failure quarters are never in training. :func:`crisis_split` takes
``grouped=True`` for the strict variant (no training bank appears in the test window), which
is reported beside it, not instead of it. The share of test failures whose bank has training
rows is MEASURED in the draft.

Usage::

    uv run python -m fintfm.evaluation.credit_fdic --download data/real/fdic
"""

from __future__ import annotations

import argparse
import json
import ssl
import time
import urllib.parse
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from fintfm.evaluation.proxy_suite import SourceLicence

if TYPE_CHECKING:
    from fintfm.evaluation.credit_suite import CreditPanel

API = "https://api.fdic.gov/banks"

#: Date the licence terms above were read at the source.
FDIC_CHECKED = "2026-11-04"

FIRST_REPDTE = "2000-03-31"
HORIZON_QUARTERS = 4
HORIZON_DAYS = 366
LAG_DAYS = 35
EMBARGO_DAYS = float(LAG_DAYS + HORIZON_DAYS)

#: Call Report fields kept as numeric features: ratios and levels known at the report date.
NUMERIC_FIELDS = (
    "ASSET", "DEP", "EQ", "EQV", "LNLSNET", "LNLSDEPR", "LNATRESR", "LNRESNCR", "NCLNLSR",
    "NTLNLSR", "ELNATRY", "ROA", "ROAPTX", "ROE", "NIMY", "INTINCY", "INTEXPY", "NONIIAY",
    "NONIXAY", "EEFFR", "IDT1CER", "RBC1AAJ", "RBCRWAJ", "P3ASSET", "P9ASSET", "P3LTOT",
    "P9LTOT", "NCRER", "NCREMULR", "NCRENRER", "NCRERESR", "NCCOMRER", "SC", "ORE", "INTAN",
    "COREDEP", "DEPUNINS", "DEPFOR", "DEPDOM", "NETINC", "NUMEMP", "OFFDOM", "OFFFOR", "TFRA",
    "NPERFV", "SZLNCON", "SZLNRES", "SZLNCI", "IDLNCORR", "NTRECOSR", "FREPO",
    "BKPREM", "NAASSET", "NALTOT",
)
CATEGORICAL_FIELDS = ("BKCLASS", "STALP")

#: Fields fetched beyond the numeric and categorical ones: keys and the two dates.
KEY_FIELDS = ("CERT", "REPDTE", "INSDATE", "ESTYMD")

#: Fields never read into a feature (module docstring); recorded so a drift is a test failure.
NEVER_FEATURES = ("ACTIVE", "ENDEFYMD", "EFFDATE", "RISDATE", "INSAGNT1", "NAME", "NAMEFULL")

LICENCE = SourceLicence(
    "https://api.fdic.gov/banks/docs/", "",
    "public domain (US government work, 17 U.S.C. section 105; none stated at the source)", True,
    "admitted by the owner (2026-11-05) as an exception to DR4.2's stated-at-source rule: the "
    "FDIC's API page and website policies state no licence for the data; the FDIC, a US federal "
    "agency, is the producer, so public domain rests on statute. FinTFM only: at least 6.72% of "
    "rows belong to SEC filers, so no FinTFM-R number is computed on it",
    FDIC_CHECKED,
)


def _get(endpoint: str, params: dict[str, str | int]) -> dict:
    """GET ``API/endpoint`` with query parameters and return the decoded JSON."""
    url = f"{API}/{endpoint}?{urllib.parse.urlencode(params)}"
    try:  # a python.org build has no system CA bundle; certifi, when installed, supplies one
        import certifi

        ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ctx = ssl.create_default_context()
    with urllib.request.urlopen(url, timeout=120, context=ctx) as resp:
        return json.load(resp)


def quarter_ends(first: str = FIRST_REPDTE, last: str | None = None) -> list[str]:
    """Quarter-end dates as ``YYYYMMDD`` strings from ``first`` to ``last`` inclusive.

    Args:
        first: First quarter end, ISO date.
        last: Last quarter end, ISO date; the latest quarter end before today when ``None``.

    Returns:
        The dates, oldest first.
    """
    end = pd.Timestamp(last) if last else pd.Timestamp.today().normalize()
    qs = pd.date_range(first, end, freq="QE")
    return [q.strftime("%Y%m%d") for q in qs]


def download(out: Path, pause: float = 0.5) -> dict:
    """Fetch the failures list and every quarter's financials into ``out``.

    Network call, never run by the tests. Writes ``failures.csv``, ``financials.csv`` (the
    :data:`NUMERIC_FIELDS`, :data:`CATEGORICAL_FIELDS` and :data:`KEY_FIELDS` of every
    bank-quarter from :data:`FIRST_REPDTE`) and ``meta.json`` (retrieval date and API index
    timestamps; ``as_of`` is the retrieval date).

    Args:
        out: Target directory, created if missing.
        pause: Seconds between requests, to be gentle with the public API.

    Returns:
        The ``meta.json`` content.
    """
    out.mkdir(parents=True, exist_ok=True)
    fails = _get("failures", {"limit": 10000, "fields": "CERT,NAME,FAILDATE,FAILYR,RESTYPE,"
                              "RESTYPE1,CHCLASS1,QBFASSET,QBFDEP,COST,PSTALP"})
    pd.DataFrame([r["data"] for r in fails["data"]]).to_csv(out / "failures.csv", index=False)
    fields = ",".join(dict.fromkeys((*KEY_FIELDS, *CATEGORICAL_FIELDS, *NUMERIC_FIELDS)))
    frames, fin_index = [], ""
    for rep in quarter_ends():
        rows: list[dict] = []
        while True:  # the API caps a page at 10,000 rows and early quarters have more
            got = _get("financials", {"limit": 10000, "offset": len(rows), "fields": fields,
                                      "filters": f"REPDTE:{rep}", "sort_by": "CERT"})
            rows += [r["data"] for r in got["data"]]
            time.sleep(pause)
            if len(rows) >= got["meta"]["total"] or not got["data"]:
                break
        if len(rows) != got["meta"]["total"]:
            raise RuntimeError(f"{rep}: read {len(rows)} of {got['meta']['total']} rows")
        fin_index = got["meta"]["index"]["createTimestamp"]
        frames.append(pd.DataFrame(rows))
    pd.concat(frames, ignore_index=True).drop(columns=["ID"], errors="ignore").to_csv(
        out / "financials.csv", index=False)
    meta = {"retrieved": pd.Timestamp.today().strftime("%Y-%m-%d"),
            "as_of": pd.Timestamp.today().strftime("%Y-%m-%d"),
            "failures_total": fails["meta"]["total"],
            "failures_index": fails["meta"]["index"]["createTimestamp"],
            "financials_index": fin_index, "api": API}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def fdic_frame(
    financials: pd.DataFrame, failures: pd.DataFrame, as_of: str,
    horizon_days: int = HORIZON_DAYS, lag_days: int = LAG_DAYS,
) -> pd.DataFrame:
    """Bank-quarters with a label and numeric features (module docstring, "Label").

    Args:
        financials: One row per bank-quarter with :data:`KEY_FIELDS`, :data:`NUMERIC_FIELDS`
            and :data:`CATEGORICAL_FIELDS` (``REPDTE`` as ``YYYYMMDD``).
        failures: The failures list, with ``CERT``, ``FAILDATE`` (``M/D/YYYY``), ``RESTYPE``.
        as_of: ISO date the failures list was current; rows whose window ends later are dropped.
        horizon_days: Window length after the reporting lag.
        lag_days: Days after ``REPDTE`` before the report is public.

    Returns:
        A frame with ``label``, bookkeeping columns ``_cert`` and ``_repdte`` (Timestamp) and
        the features. ``.attrs`` holds the counts dropped: ``dropped_lag`` (failed inside the
        lag), ``dropped_censored``, ``failures_unmatched`` (failures with a ``CERT`` never
        seen in ``financials``, counted from 2000 on) and ``positives``.

    Raises:
        ValueError: If a required column is missing or a bank has two failure records.
    """
    missing = {*KEY_FIELDS, *NUMERIC_FIELDS, *CATEGORICAL_FIELDS} - set(financials.columns)
    if missing:
        raise ValueError(f"financials lacks {sorted(missing)}")
    f = failures[failures["RESTYPE"] == "FAILURE"].copy()
    f["_fail"] = pd.to_datetime(f["FAILDATE"], format="%m/%d/%Y", errors="coerce")
    # only failures from the first report on can label a row (three pre-2000 CERTs failed twice)
    f = f.dropna(subset=["CERT", "_fail"])
    f = f[f["_fail"] >= pd.Timestamp(FIRST_REPDTE)]
    f["CERT"] = f["CERT"].astype(np.int64)
    if f["CERT"].duplicated().any():
        raise ValueError("a CERT has two FAILURE records")
    fin = financials.copy()
    fin["_repdte"] = pd.to_datetime(fin["REPDTE"].astype(str), format="%Y%m%d")
    fin["_cert"] = fin["CERT"].astype(np.int64)
    fin = fin.merge(f[["CERT", "_fail"]].rename(columns={"CERT": "_cert"}), on="_cert",
                    how="left")
    opens = fin["_repdte"] + pd.Timedelta(days=lag_days)
    closes = opens + pd.Timedelta(days=horizon_days)
    in_lag = (fin["_fail"] > fin["_repdte"]) & (fin["_fail"] <= opens)
    censored = closes > pd.Timestamp(as_of)
    pos = (fin["_fail"] > opens) & (fin["_fail"] <= closes)
    # a failure before the report date cannot have a report; keep the guard explicit
    if (fin["_fail"] <= fin["_repdte"]).any():
        fin = fin[~(fin["_fail"] <= fin["_repdte"])]
        opens, closes, in_lag, censored, pos = (s.loc[fin.index] for s in
                                                (opens, closes, in_lag, censored, pos))
    keep = ~in_lag & ~censored
    seen = set(fin["_cert"])
    unmatched = int((~f.loc[f["_fail"] >= pd.Timestamp(FIRST_REPDTE), "CERT"].isin(seen)).sum())
    fin = fin[keep].copy()
    fin["label"] = pos[keep].astype(np.int64).to_numpy()
    age = (fin["_repdte"] - pd.to_datetime(fin["INSDATE"].astype(str), format="%Y%m%d",
                                           errors="coerce")).dt.days / 365.25
    num = fin[list(NUMERIC_FIELDS)].apply(pd.to_numeric, errors="coerce").astype(np.float64)
    num["log_asset"] = np.log1p(num["ASSET"].clip(lower=0))
    num["insured_years"] = age.astype(np.float64)
    parts = [num]
    for col, top in (("BKCLASS", 12), ("STALP", 15)):
        s = fin[col].astype("string")
        vals = s.value_counts().index[:top]
        parts.append(pd.DataFrame({f"{col}_{v}": (s == v).astype(np.float64) for v in vals},
                                  index=fin.index))
    out = pd.concat(parts, axis=1)
    out["label"] = fin["label"]
    out["_cert"] = fin["_cert"]
    out["_repdte"] = fin["_repdte"]
    out.attrs = {"dropped_lag": int(in_lag.sum()), "dropped_censored": int(censored.sum()),
                 "failures_unmatched": unmatched, "positives": int(out["label"].sum())}
    return out.reset_index(drop=True)


def _days(stamps: pd.Series) -> np.ndarray:
    """Timestamps as float days since the Unix epoch."""
    return (stamps - pd.Timestamp("1970-01-01")).dt.total_seconds().to_numpy() / 86_400.0


def panel_from_frame(frame: pd.DataFrame, grouped: bool = False) -> CreditPanel:
    """Wrap :func:`fdic_frame`'s output as a :class:`~fintfm.evaluation.credit_suite.CreditPanel`.

    Args:
        frame: Output of :func:`fdic_frame`.
        grouped: Carry the bank id as ``groups`` (module docstring: off for the suite).

    Returns:
        The panel, out-of-time on ``REPDTE`` with the :data:`EMBARGO_DAYS` embargo.
    """
    from fintfm.evaluation.credit_suite import CreditPanel

    feats = frame.drop(columns=["label", "_cert", "_repdte"])
    return CreditPanel(
        name="fdic", X=feats.to_numpy(np.float64), feature_names=tuple(feats.columns),
        y=frame["label"].to_numpy(np.int64), time=_days(frame["_repdte"]),
        groups=frame["_cert"].to_numpy() if grouped else None, embargo=EMBARGO_DAYS,
    )


def load_fdic(root: Path) -> CreditPanel:
    """FDIC bank-quarters: failure within :data:`HORIZON_QUARTERS` of the report becoming public.

    Args:
        root: Directory holding ``fdic/failures.csv``, ``fdic/financials.csv`` and
            ``fdic/meta.json`` (from :func:`download`).

    Returns:
        The panel, out-of-time on ``REPDTE``, no groups, :data:`EMBARGO_DAYS` embargo.
    """
    d = root / "fdic"
    meta = json.loads((d / "meta.json").read_text())
    fin = pd.read_csv(d / "financials.csv", low_memory=False)
    fails = pd.read_csv(d / "failures.csv", low_memory=False)
    return panel_from_frame(fdic_frame(fin, fails, meta["as_of"]))


def crisis_split(
    panel: CreditPanel, train_end: str = "2006-12-31", test_start: str = "2008-01-01",
    test_end: str = "2011-12-31", grouped: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Train before the crisis, test on it (module docstring, "Splits").

    Args:
        panel: A panel with ``time`` and ``embargo``.
        train_end: Last ``REPDTE`` a training row may carry.
        test_start: First test ``REPDTE``.
        test_end: Last test ``REPDTE``.
        grouped: Also drop training rows of banks that appear in the test window.

    Returns:
        ``(train, test)`` row indices; training rows lie at least ``panel.embargo`` days before
        ``test_start``, so every training label was final when the test window opened.

    Raises:
        ValueError: If the panel has no date, or ``grouped`` is asked of a panel with no groups.
    """
    if panel.time is None:
        raise ValueError("crisis_split needs a dated panel")
    t = panel.time
    day = lambda s: (pd.Timestamp(s) - pd.Timestamp("1970-01-01")).days
    test = np.flatnonzero((t >= day(test_start)) & (t <= day(test_end)))
    train_mask = (t <= day(train_end)) & (t < day(test_start) - panel.embargo)
    if grouped:
        if panel.groups is None:
            raise ValueError("grouped=True needs a panel built with groups")
        train_mask &= ~np.isin(panel.groups, panel.groups[test])
    return np.flatnonzero(train_mask), test


def main(argv: Sequence[str] | None = None) -> int:
    """Command line: ``--download DIR`` fetches the files (network, public API)."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--download", type=Path, required=True, help="target directory")
    args = ap.parse_args(argv)
    print(json.dumps(download(args.download), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
