"""Build a real SEC EDGAR firm-quarter panel for FinTFM-R (task A5).

What this produces
-------------------

One parquet file, one row per (CIK, fiscal period end), columns:

- A fixed set of core GAAP fundamentals (`CORE_TAGS` below), each a float, NaN when a filer
  did not report that concept that period -- true missingness, not zero-filled, matching how
  every other source in this codebase represents it.
- `cik`, `period_end`: bookkeeping, excluded from features by
  `fintfm.prior.real_edgar.feature_columns`.
- `label`: 1 if this filer disclosed an 8-K Item 1.03 bankruptcy/receivership within
  `--label-horizon-quarters` quarters after this period, else 0.

Two public-domain SEC sources, joined by CIK
---------------------------------------------

Features: the Financial Statement Data Sets (`sec.gov/dera/data/financial-statement-data-sets`
, `sub.txt`/`num.txt` per quarterly zip). Labels: the full-text search API
(`efts.sec.gov/LATEST/search-index`), queried for the exact phrase "Item 1.03 Bankruptcy"
restricted to `forms=8-K`. Both are US federal government work -- public domain, no licence
gate -- see `docs/design/DECISIONS.md` D17 and `docs/results/FINDINGS.md` §151, which measured
this same join's feasibility before this script was written.

This script writes real data to disk. The output file is covered by `.gitignore`
(`data/edgar/`) and must never be committed, matching this project's convention for trained
weights -- see `CLAUDE.md`.

A descriptive User-Agent is required by SEC's own API guidance
-----------------------------------------------------------------

Without one, SEC's Akamai front end returns a 403 that reads like a licensing block and is
not one -- §151 found this the hard way. `--contact` sets it; there is no sensible default
that doesn't misattribute requests to someone else.
"""

from __future__ import annotations

import argparse
import io
import time
import zipfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

#: Broadly-populated, non-fund-specific GAAP concepts, chosen from §151's measured tag
#: frequency on one quarter (2024 Q1) of real filings -- the fifteen most common core
#: fundamentals after excluding BDC/fund-only concepts (`InvestmentOwnedAtFairValue` and
#: siblings), which are common in that quarter's data only because funds are a large filer
#: population, not because they describe a typical corporate filer.
CORE_TAGS: tuple[str, ...] = (
    "Assets",
    "AssetsCurrent",
    "Liabilities",
    "LiabilitiesCurrent",
    "LiabilitiesAndStockholdersEquity",
    "StockholdersEquity",
    "CashAndCashEquivalentsAtCarryingValue",
    "NetIncomeLoss",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "Goodwill",
    "PropertyPlantAndEquipmentNet",
    "RetainedEarningsAccumulatedDeficit",
    "AccountsPayableCurrent",
    "CommonStockSharesOutstanding",
    "OperatingLeaseRightOfUseAsset",
)

_FSDS_URL = "https://www.sec.gov/files/dera/data/financial-statement-data-sets/{q}.zip"
_FULLTEXT_URL = "https://efts.sec.gov/LATEST/search-index"


def _quarters(start: str, end: str) -> list[str]:
    """Every `YYYYqN` label from `start` to `end`, inclusive.

    Args:
        start: First quarter, e.g. `"2022q1"`.
        end: Last quarter, e.g. `"2024q4"`.

    Returns:
        Quarter labels in chronological order.
    """
    sy, sq = int(start[:4]), int(start[5])
    ey, eq = int(end[:4]), int(end[5])
    out = []
    y, q = sy, sq
    while (y, q) <= (ey, eq):
        out.append(f"{y}q{q}")
        q += 1
        if q > 4:
            q = 1
            y += 1
    return out


def _add_quarters(quarter: str, n: int) -> str:
    """`quarter` shifted forward by `n` quarter labels, e.g. `_add_quarters("2023q4", 4)` ->
    `"2024q4"`.

    Args:
        quarter: A `YYYYqN` label.
        n: Quarters to add (may be any non-negative integer).

    Returns:
        The shifted quarter label.
    """
    y, q = int(quarter[:4]), int(quarter[5])
    total = (q - 1) + n
    return f"{y + total // 4}q{total % 4 + 1}"


def _download_quarter(quarter: str, cache_dir: Path, contact: str) -> Path:
    """Download and unzip one quarter's Financial Statement Data Sets, caching on disk.

    Args:
        quarter: e.g. `"2024q1"`.
        cache_dir: Directory raw downloads and unzipped files are kept in, reused across
            calls so re-running this script does not re-download every quarter.
        contact: Descriptive User-Agent contact string SEC's API guidance asks for.

    Returns:
        Path to the unzipped quarter directory, containing `sub.txt` and `num.txt`.
    """
    qdir = cache_dir / quarter
    if (qdir / "num.txt").exists():
        return qdir
    cache_dir.mkdir(parents=True, exist_ok=True)
    resp = requests.get(
        _FSDS_URL.format(q=quarter), headers={"User-Agent": contact}, timeout=120
    )
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        zf.extractall(qdir)
    return qdir


def _load_quarter_fundamentals(qdir: Path) -> pd.DataFrame:
    """One row per (CIK, period, adsh) with `CORE_TAGS` as columns, for one quarter.

    Args:
        qdir: Directory from `_download_quarter`.

    Returns:
        A wide DataFrame; `cik` and `period_end` identify the row, one column per core tag.
    """
    sub = pd.read_csv(
        qdir / "sub.txt", sep="\t", usecols=["adsh", "cik", "form", "period"],
        dtype={"cik": "int64", "period": "str"},
    )
    sub = sub[sub["form"].isin(["10-K", "10-Q"])]
    num = pd.read_csv(
        qdir / "num.txt", sep="\t", usecols=["adsh", "tag", "qtrs", "value"],
        dtype={"qtrs": "int64"},
    )
    # qtrs=0 is an instant concept (Assets, StockholdersEquity -- a balance at one date, no
    # duration); qtrs=1 is a single-quarter flow (NetIncomeLoss, Revenue over one quarter).
    # qtrs=4 (a 10-K's full fiscal year) is deliberately *not* included here: a 10-K-only
    # filer's annual net income and a 10-Q filer's quarterly net income are not the same
    # quantity, and putting both in one column would silently corrupt the feature's scale
    # for every row of whichever type is rarer. The cost is real but the right one to pay:
    # a 10-K-only filer shows NaN for flow concepts that period, which is true missingness
    # this architecture already represents explicitly, not a scale-inconsistency bug no cell
    # -embedding can see.
    num = num[num["tag"].isin(CORE_TAGS) & num["qtrs"].isin([0, 1])]
    # A filer can report the same tag more than once in one submission (restatements,
    # dimensional breakdowns); keep the first value per (adsh, tag) rather than silently
    # summing or overwriting -- correctness over completeness for a first panel.
    num = num.drop_duplicates(subset=["adsh", "tag"], keep="first")
    wide = num.pivot(index="adsh", columns="tag", values="value")
    wide = wide.reindex(columns=list(CORE_TAGS))
    merged = sub.merge(wide, on="adsh", how="inner")
    merged["period_end"] = pd.to_datetime(merged["period"], format="%Y%m%d", errors="coerce")
    merged = merged.dropna(subset=["period_end"])
    return merged[["cik", "period_end", *CORE_TAGS]]


#: SEC's full-text search API 500s on any request with `from=100` or above -- a hard
#: pagination ceiling on this endpoint (it backs the public search box, not a bulk-export
#: API), found by actually hitting it rather than assumed from documentation that does not
#: mention it. `_bankruptcy_events` must never let one window's result count reach this.
_FULLTEXT_MAX_FROM = 100


def _get_json_with_retries(
    url: str, params: dict, contact: str, max_attempts: int = 4
) -> dict:
    """GET `url`, retrying on a transient server error with exponential backoff.

    §151's build found `efts.sec.gov` occasionally 500s on an otherwise-valid request --
    confirmed transient by retrying the exact failing URL by hand and getting a 200 back
    immediately. A 24-request-per-year pipeline hitting a free public endpoint will meet this
    at least once; failing the whole run on one blip is not the right response to it.

    Args:
        url: Request URL.
        params: Query parameters.
        contact: User-Agent contact string.
        max_attempts: Total attempts before giving up and re-raising.

    Returns:
        The parsed JSON body.

    Raises:
        requests.HTTPError: If every attempt fails, or if any attempt fails with a non-5xx
            status (a 4xx is not transient and should not be retried into silence).
    """
    for attempt in range(max_attempts):
        resp = requests.get(url, params=params, headers={"User-Agent": contact}, timeout=60)
        if resp.status_code < 500:
            resp.raise_for_status()
            return resp.json()
        if attempt == max_attempts - 1:
            resp.raise_for_status()
        time.sleep(2**attempt)
    raise AssertionError("unreachable")  # pragma: no cover


def _search_window(startdt: date, enddt: date, contact: str, sleep_s: float) -> list[dict]:
    """Every "Item 1.03 Bankruptcy" 8-K hit in one date window, paginated.

    Args:
        startdt: Window start (inclusive).
        enddt: Window end (inclusive).
        contact: User-Agent contact string.
        sleep_s: Delay between pages.

    Returns:
        `{"cik": int, "filing_date": str}` dicts, one per (filing, CIK) pair -- a filing can
        name more than one CIK for a group of affiliated debtors.

    Raises:
        RuntimeError: If this window alone has >= `_FULLTEXT_MAX_FROM` hits -- the caller
            must split the window further rather than silently lose results past the cap.
    """
    events: list[dict] = []
    frm = 0
    while True:
        data = _get_json_with_retries(
            _FULLTEXT_URL,
            params={
                "q": '"Item 1.03 Bankruptcy"', "forms": "8-K",
                "startdt": startdt.isoformat(), "enddt": enddt.isoformat(), "from": frm,
            },
            contact=contact,
        )
        hits = data["hits"]["hits"]
        if not hits:
            break
        for h in hits:
            for cik in h["_source"]["ciks"]:
                events.append({"cik": int(cik), "filing_date": h["_source"]["file_date"]})
        frm += len(hits)
        total = data["hits"]["total"]["value"]
        if frm >= total:
            break
        if frm >= _FULLTEXT_MAX_FROM:
            raise RuntimeError(
                f"{startdt} to {enddt} has {total} hits, >= the {_FULLTEXT_MAX_FROM}-result "
                "pagination ceiling this endpoint enforces -- split this window smaller "
                "(e.g. by week) and retry"
            )
        time.sleep(sleep_s)
    return events


def _month_windows(startdt: date, enddt: date) -> list[tuple[date, date]]:
    """`[startdt, enddt]` split into calendar-month `(window_start, window_end)` pairs.

    Splitting by month, not by the caller's full range, keeps each full-text search query's
    result count comfortably under `_FULLTEXT_MAX_FROM` -- §151's own measurement put a single
    quarter at 49-80 hits, so a month is a real safety margin, not an arbitrary choice.
    """
    windows = []
    y, m = startdt.year, startdt.month
    while date(y, m, 1) <= enddt:
        window_start = max(date(y, m, 1), startdt)
        next_m = m + 1
        next_y = y
        if next_m > 12:
            next_m = 1
            next_y += 1
        window_end = min(date(next_y, next_m, 1) - timedelta(days=1), enddt)
        windows.append((window_start, window_end))
        y, m = next_y, next_m
    return windows


def _bankruptcy_events(start: str, end: str, contact: str, sleep_s: float = 3.0) -> pd.DataFrame:
    """Every 8-K Item 1.03 bankruptcy/receivership disclosure in `[start, end]`.

    Args:
        start: First quarter, e.g. `"2022q1"`, converted to that quarter's first day.
        end: Last quarter, converted to that quarter's last day.
        contact: User-Agent contact string.
        sleep_s: Delay between full-text search requests -- SEC rate-limits this endpoint;
            §151 found 3 seconds sufficient, not tuned beyond that.

    Returns:
        `cik`, `filing_date` columns, one row per (filing, CIK) pair.
    """
    sy, sq = int(start[:4]), int(start[5])
    ey, eq = int(end[:4]), int(end[5])
    startdt = date(sy, 3 * (sq - 1) + 1, 1)
    # The quarter's last day is one day before the *next* quarter's first day -- computed
    # that way, not by month arithmetic on the quarter's own number, after an earlier version
    # of this got every quarter-end a month too early (Q4 -> Nov 30) and was caught by
    # actually running it against known values rather than trusting the algebra.
    next_month = 3 * eq + 1
    next_year = ey
    if next_month > 12:
        next_month -= 12
        next_year += 1
    enddt = date(next_year, next_month, 1) - timedelta(days=1)
    events: list[dict] = []
    for window_start, window_end in _month_windows(startdt, enddt):
        events.extend(_search_window(window_start, window_end, contact, sleep_s))
        time.sleep(sleep_s)
    df = pd.DataFrame(events, columns=["cik", "filing_date"])
    df["filing_date"] = pd.to_datetime(df["filing_date"])
    return df


def build_panel(
    start: str, end: str, contact: str, cache_dir: Path, label_horizon_quarters: int = 4,
) -> pd.DataFrame:
    """Assemble the full panel: fundamentals, joined to a forward-looking bankruptcy label.

    Args:
        start: First quarter of fundamentals to include, e.g. `"2022q1"`.
        end: Last quarter of fundamentals to include.
        contact: User-Agent contact string for both SEC endpoints.
        cache_dir: Where raw quarterly downloads are cached.
        label_horizon_quarters: A firm-quarter is labelled positive if a bankruptcy disclosure
            for that CIK falls within this many quarters after `period_end`. 4 (one year)
            matches the order of magnitude of V4FinBench's own shortest horizon.

    Returns:
        The assembled panel, one row per (CIK, period_end), deduplicated.
    """
    frames = []
    for q in _quarters(start, end):
        qdir = _download_quarter(q, cache_dir, contact)
        frames.append(_load_quarter_fundamentals(qdir))
        print(f"  {q}: {len(frames[-1])} firm-quarter rows", flush=True)
    panel = pd.concat(frames, ignore_index=True)
    panel = panel.drop_duplicates(subset=["cik", "period_end"], keep="first")

    # The event search must extend *past* `end` by the label horizon, or every row in the
    # panel's last `label_horizon_quarters` quarters has its forward-looking window silently
    # truncated to whatever events happen to fall before `end` -- undercounting positives
    # exactly where the panel is newest. Caught before this panel was trusted, not after.
    events_end = _add_quarters(end, label_horizon_quarters)
    events = _bankruptcy_events(start, events_end, contact)
    print(f"  {len(events)} bankruptcy disclosures in range (extended for the label horizon)", flush=True)
    horizon = pd.Timedelta(days=92 * label_horizon_quarters)
    panel["label"] = 0
    if len(events):
        events_by_cik = events.groupby("cik")["filing_date"].apply(list).to_dict()
        labels = np.zeros(len(panel), dtype=np.int64)
        for i, (cik, period_end) in enumerate(zip(panel["cik"], panel["period_end"], strict=True)):
            for fdate in events_by_cik.get(cik, []):
                if period_end < fdate <= period_end + horizon:
                    labels[i] = 1
                    break
        panel["label"] = labels
    panel = panel.drop(columns=["adsh"], errors="ignore")
    return panel.reset_index(drop=True)


def main() -> None:
    """Entry point. See module docstring for what this writes and why."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", default="2022q1", help="first quarter, e.g. 2022q1")
    p.add_argument("--end", default="2023q4", help="last quarter, e.g. 2023q4")
    p.add_argument(
        "--contact", required=True,
        help="descriptive User-Agent contact, e.g. 'fintfm-research you@example.com' -- "
        "required by SEC's own API guidance, not a credential",
    )
    p.add_argument("--cache-dir", default="data/edgar/raw", help="raw download cache")
    p.add_argument("--out", default="data/edgar/panel.parquet")
    p.add_argument("--label-horizon-quarters", type=int, default=4)
    args = p.parse_args()

    panel = build_panel(
        args.start, args.end, args.contact, Path(args.cache_dir), args.label_horizon_quarters
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(out)
    print(
        f"wrote {len(panel)} rows, {panel['label'].mean():.4%} positive, to {out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
