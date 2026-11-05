"""The DR4.3 FDIC loader: label window, reporting lag, censoring, embargo and the crisis split.

A small synthetic fixture only; no network and no FDIC file enters the repository. Covers what
would fail silently: a label that peeks inside the reporting lag, a censored row scored as a
survivor, a training row whose outcome falls in the test window, and a licence that is not
refused before any file is read.
"""

import json

import numpy as np
import pandas as pd
import pytest

from fintfm.evaluation import credit_fdic as fd
from fintfm.evaluation import credit_suite as cs
from fintfm.prior.real_panel import EVAL_ONLY_SOURCES


def _fixture(n_banks: int = 40, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Quarterly rows 2003-2012 for banks 1..n; every fifth bank fails in 2009-2010."""
    rng = np.random.default_rng(seed)
    quarters = pd.date_range("2003-03-31", "2012-12-31", freq="QE")
    rows, fails = [], []
    for cert in range(1, n_banks + 1):
        fail = pd.Timestamp("2009-06-15") + pd.Timedelta(days=int(rng.integers(0, 500))) \
            if cert % 5 == 0 else None
        for q in quarters:
            if fail is not None and q >= fail:
                break
            row = {c: float(rng.normal()) for c in fd.NUMERIC_FIELDS}
            row.update(CERT=cert, REPDTE=int(q.strftime("%Y%m%d")), INSDATE="19340101",
                       ESTYMD=19000101, BKCLASS="N" if cert % 2 else "SM",
                       STALP="CA" if cert % 3 else "NY", ASSET=1000.0 + cert)
            rows.append(row)
        if fail is not None:
            fails.append({"CERT": cert, "FAILDATE": f"{fail.month}/{fail.day}/{fail.year}",
                          "RESTYPE": "FAILURE"})
    fails.append({"CERT": 999, "FAILDATE": "1/2/2010", "RESTYPE": "ASSISTANCE"})
    fails.append({"CERT": None, "FAILDATE": "5/28/1934", "RESTYPE": "FAILURE"})
    return pd.DataFrame(rows), pd.DataFrame(fails)


def test_label_window_lag_and_censoring_by_hand():
    fin = pd.DataFrame([
        {c: 1.0 for c in fd.NUMERIC_FIELDS}
        | {"CERT": 7, "REPDTE": r, "INSDATE": "19340101", "ESTYMD": 1, "BKCLASS": "N",
           "STALP": "CA"}
        for r in (20090331, 20090630, 20090930, 20091231)
    ])
    fails = pd.DataFrame([{"CERT": 7, "FAILDATE": "4/20/2009", "RESTYPE": "FAILURE"}])
    # Only the 2009-03-31 row has a failure inside the 35-day lag (to 2009-05-05)... no: the
    # failure on 4/20 is 20 days after the report, so that row is dropped; later rows follow
    # the failure and the bank cannot report, but the fixture keeps them to test the guard.
    frame = fd.fdic_frame(fin, fails, as_of="2026-01-01")
    assert frame.attrs["dropped_lag"] == 1
    assert len(frame) == 0 or frame["_repdte"].min() > pd.Timestamp("2009-03-31")
    # a failure 6 months after the report is a positive; 2 years after it is not
    fails = pd.DataFrame([{"CERT": 7, "FAILDATE": "12/20/2009", "RESTYPE": "FAILURE"}])
    frame = fd.fdic_frame(fin.iloc[:3], fails, as_of="2026-01-01")
    labels = dict(zip(frame["_repdte"].dt.strftime("%Y%m%d"), frame["label"], strict=True))
    assert labels == {"20090331": 1, "20090630": 1, "20090930": 1}
    fails = pd.DataFrame([{"CERT": 7, "FAILDATE": "12/20/2011", "RESTYPE": "FAILURE"}])
    assert fd.fdic_frame(fin.iloc[:3], fails, as_of="2026-01-01")["label"].sum() == 0
    # censoring: with as_of before a row's window closes, the row is dropped, not a negative
    late = fd.fdic_frame(fin.iloc[:3], fails.iloc[:0].assign(CERT=[], FAILDATE=[], RESTYPE=[]),
                         as_of="2010-02-01")
    assert late.attrs["dropped_censored"] == 3 and len(late) == 0


def test_only_failure_records_label_and_unknown_certs_are_counted():
    fin, fails = _fixture()
    frame = fd.fdic_frame(fin, fails, as_of="2026-01-01")
    assert frame["label"].sum() == frame.attrs["positives"] > 0
    assert 999 not in set(frame["_cert"])  # the ASSISTANCE record labels nothing
    # no bank reports after it failed, so every positive row precedes its bank's failure
    failed = {int(c): pd.to_datetime(d, format="%m/%d/%Y")
              for c, d in zip(fails["CERT"], fails["FAILDATE"], strict=True)
              if pd.notna(c) and c != 999}
    pos = frame[frame["label"] == 1]
    assert all(r < failed[int(c)] for c, r in zip(pos["_cert"], pos["_repdte"], strict=True))
    assert frame.attrs["failures_unmatched"] == 0


def test_features_exclude_status_fields_and_the_label():
    fin, fails = _fixture()
    fin["ACTIVE"] = 0
    frame = fd.fdic_frame(fin, fails, as_of="2026-01-01")
    for never in fd.NEVER_FEATURES:
        assert never not in frame.columns
    panel = fd.panel_from_frame(frame)
    assert panel.X.shape == (len(frame), len(panel.feature_names))
    assert "label" not in panel.feature_names and "insured_years" in panel.feature_names
    assert panel.groups is None and panel.embargo == fd.EMBARGO_DAYS == 401.0
    assert fd.panel_from_frame(frame, grouped=True).groups is not None


def test_a_missing_column_or_a_duplicate_failure_is_refused():
    fin, fails = _fixture()
    with pytest.raises(ValueError, match="lacks"):
        fd.fdic_frame(fin.drop(columns=["NETINC"]), fails, "2026-01-01")
    dup = pd.concat([fails, fails.iloc[[0]]])
    with pytest.raises(ValueError, match="two FAILURE"):
        fd.fdic_frame(fin, dup, "2026-01-01")


def test_crisis_split_trains_before_and_respects_the_embargo_and_groups():
    fin, fails = _fixture(n_banks=60)
    frame = fd.fdic_frame(fin, fails, as_of="2026-01-01")
    panel = fd.panel_from_frame(frame, grouped=True)
    train, test = fd.crisis_split(panel, train_end="2006-12-31", test_start="2008-01-01",
                                  test_end="2011-12-31")
    assert len(train) and len(test) and panel.y[test].sum() > 0
    assert panel.time[train].max() < panel.time[test].min() - panel.embargo
    assert not set(train) & set(test)
    gtrain, gtest = fd.crisis_split(panel, grouped=True)
    assert not set(panel.groups[gtrain]) & set(panel.groups[gtest])
    assert set(gtest) == set(test)
    with pytest.raises(ValueError, match="groups"):
        fd.crisis_split(fd.panel_from_frame(frame), grouped=True)


def test_shared_splits_put_the_crisis_in_training(tmp_path):
    fin, fails = _fixture(n_banks=60)
    d = tmp_path / "fdic"
    d.mkdir()
    fin.to_csv(d / "financials.csv", index=False)
    fails.to_csv(d / "failures.csv", index=False)
    (d / "meta.json").write_text(json.dumps({"as_of": "2026-01-01"}))
    panel = fd.load_fdic(tmp_path)
    assert panel.split_kind == "out_of_time"
    for tr, te in cs.out_of_time_splits(panel):
        assert panel.time[tr].max() < panel.time[te].min() - panel.embargo


def test_registered_evaluation_only_and_admitted_fintfm_only():
    assert "fdic" in cs.DATASETS and "fdic" in EVAL_ONLY_SOURCES
    ds = cs.DATASETS["fdic"]
    assert ds.licence.include is True and ds.licence.checked == fd.FDIC_CHECKED
    assert "fdic" in cs.admitted() and "FinTFM only" in ds.licence.reason
    from fintfm.evaluation.proxy_suite import licence_verdict

    assert licence_verdict(ds.licence.licence) is None
    assert ds.licence.url.startswith("https://") and ds.licence.reason
    assert "FinTFM-only" in fd.__doc__ or "FinTFM-R number" in fd.__doc__


def test_quarter_ends():
    assert fd.quarter_ends("2000-03-31", "2000-12-31") == ["20000331", "20000630", "20000930",
                                                           "20001231"]
