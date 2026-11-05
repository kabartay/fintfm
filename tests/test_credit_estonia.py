"""The Estonian register panel (DR4.3): label window, dropped rows and the licence gate.

A tiny synthetic register written under ``tmp_path``; the real files never enter the repo and
no test touches the network. Covers what would fail silently: a report filed after a
declaration labelled as a positive, a deleted firm kept with a guessed label, a censored window
scored as a negative, and the entry being admitted while its label is survivor-conditioned.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from fintfm.evaluation import credit_estonia as ce
from fintfm.evaluation import credit_suite as cs

REPORT_COLS = ('"report_id";"taidetud_aruanne_report_id";"registrikood";"õiguslik vorm";'
               '"staatus";"aruandeaasta";"kas konsolideeritud?";"period_start";"period_end";'
               '"esitatud_kpv";"kas auditeeritud?"')


def _write(root: Path, specs: list[tuple]) -> Path:
    """specs: (report_id, code, status, year, filed, consolidated); codes 1.. get cards."""
    d = root / "estonia"
    d.mkdir()
    lines = [REPORT_COLS]
    el = ['"report_id";"tabel";"elemendi_label";"elemendi_nimetus";"vaartus"']
    for rid, code, status, year, filed, cons in specs:
        lines.append(f'{rid};;{code};"OÜ";"{status}";{year};"{"Jah" if cons else "Ei"}";'
                     f'"01.01.{year}";"31.12.{year}";"{filed}";"Jah"')
        for name, val in (("Assets", 100 + rid), ("Equity", 40), ("CurrentAssets", 60),
                          ("CurrentLiabilities", 30), ("NonCurrentLiabilities", 20),
                          ("Revenue", 200), ("TotalAnnualPeriodProfitLoss", -5 * (rid % 3))):
            el.append(f'{rid};"Bilanss";"x";"{name}";"{val}"')
        el.append(f'{rid};"Kasumi jaotamise otsus";"x";"Equity";"999999"')  # must be skipped
    (d / ce.REPORTS_FILE).write_text("\n".join(lines).replace('";"', '";"') + "\n")
    (d / ce.ELEMENT_FILE.format(year=2022)).write_text("\n".join(el) + "\n")
    cards = [{"ariregistri_kood": 1, "registrikaardid": [{"kanded": [
        {"kpv": "01.02.2021", "kandeliik_tekstina": "Esmakanne"},
        {"kpv": "01.06.2022", "kandeliik_tekstina": ce.BANKRUPTCY_ENTRY}]}]},
        {"ariregistri_kood": 2, "registrikaardid": [{"kanded": [
            {"kpv": "01.02.2021", "kandeliik_tekstina": "Esmakanne"}]}]}]
    (d / ce.CARDS_FILE).write_text(json.dumps(cards, indent=4, ensure_ascii=False))
    basic = ["nimi;ariregistri_kood;ettevotja_esmakande_kpv"] + [
        f"f{c};{c};01.01.2010" for c in (1, 2, 3, 4)]
    (d / ce.BASIC_FILE).write_text("﻿" + "\n".join(basic) + "\n", encoding="utf-8")
    return d


def _frame(tmp_path: Path) -> pd.DataFrame:
    specs = [
        (1, 1, "Pankrotis", 2021, "01.04.2022", False),   # filed 2 months before: positive
        (2, 1, "Pankrotis", 2022, "01.09.2022", False),   # filed after the declaration: drop
        (3, 2, "Registrisse kantud", 2021, "01.04.2022", False),  # negative
        (4, 3, "Kustutatud", 2021, "01.04.2022", False),  # deleted: label unobservable, drop
        (5, 4, "Registrisse kantud", 2021, "01.04.2022", True),   # consolidated: drop
        (6, 2, "Registrisse kantud", 2025, "01.04.2026", False),  # window past the cut: drop
        (7, 4, "Registrisse kantud", 2021, "01.04.2022", False),  # second report, code 4
    ]
    d = _write(tmp_path, specs)
    reports = ce.read_reports(d / ce.REPORTS_FILE)
    el = ce.read_elements(d, set(reports.report_id))
    bk = ce.read_bankruptcies(d / ce.CARDS_FILE)
    first = pd.Series(pd.to_datetime(["2010-01-01"] * 4), index=[1, 2, 3, 4])
    return ce.estonia_frame(reports, el, bk, first)


def test_bankruptcy_dates_are_read_from_the_cards(tmp_path):
    d = _write(tmp_path, [(1, 1, "Pankrotis", 2021, "01.04.2022", False)])
    bk = ce.read_bankruptcies(d / ce.CARDS_FILE)
    assert bk["code"].tolist() == [1] and bk["declared"].iloc[0] == pd.Timestamp("2022-06-01")


def test_label_window_and_dropped_rows(tmp_path):
    f = _frame(tmp_path)
    # kept: reports 1 (positive), 3, 7 (negatives); 2, 4, 5, 6 dropped for the stated reasons
    assert sorted(f.index.tolist()) == [0, 2, 6]
    assert f.loc[0, "label"] == 1 and f.loc[2, "label"] == 0 and f.loc[6, "label"] == 0
    assert (f["_filed"] == pd.Timestamp("2022-04-01")).all()


def test_profit_allocation_tables_are_skipped_and_ratios_use_the_balance_sheet(tmp_path):
    f = _frame(tmp_path)
    assert np.isclose(f.loc[0, "equity_to_assets"], 40 / 101)      # not 999999
    assert np.isclose(f.loc[0, "liabilities_to_assets"], 50 / 101)
    assert np.isclose(f.loc[0, "current_ratio"], 2.0)
    assert f.loc[0, "form_OÜ"] == 1 and f.loc[0, "form_AS"] == 0
    assert np.isclose(f.loc[0, "log_age_days"], np.log1p(
        (pd.Timestamp("2022-04-01") - pd.Timestamp("2010-01-01")).days))
    assert not any("status" in c for c in f.columns)  # current status contains the label


def test_load_estonia_builds_a_grouped_embargoed_panel(tmp_path):
    specs = [(i, 1 + i % 4, "Registrisse kantud", 2021, "01.04.2022", False) for i in range(1, 9)]
    _write(tmp_path, specs)
    panel = ce.load_estonia(tmp_path)
    assert panel.name == "estonia" and panel.embargo == 365.0
    assert panel.groups is not None and panel.time is not None
    assert panel.X.shape == (len(panel.y), len(panel.feature_names))
    assert not any(n.startswith("_") for n in panel.feature_names)


def test_estonia_is_registered_evaluation_only_and_admitted_with_its_caveat():
    from fintfm.prior.real_panel import EVAL_ONLY_SOURCES

    assert "estonia" in EVAL_ONLY_SOURCES and "estonia" in cs.DATASETS
    lic = cs.DATASETS["estonia"].licence
    assert lic.include is True and lic.licence == "CC BY 4.0" and "survivor" in lic.reason
    assert "estonia" in cs.admitted()
