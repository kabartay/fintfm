"""Tests for the forward-register inclusion rule (openspec task 6.3), synthetic records only."""

from __future__ import annotations

import random
from collections.abc import Iterator, Mapping
from dataclasses import replace
from datetime import date
from typing import Any

import pytest

from fintfm.register.cohort import (
    INCLUDED,
    READ_COLUMNS,
    CohortRule,
    Reason,
    cohort_hash,
    normalise_company_number,
    parse_date,
    select_cohort,
)

RULE = CohortRule(snapshot_date=date(2026, 11, 1))


def rec(number: str = "01234567", **over: Any) -> dict[str, Any]:
    """A record that passes every rule, with overrides."""
    base: dict[str, Any] = {
        "CompanyNumber": number,
        "CompanyStatus": "Active",
        "CompanyCategory": "Private Limited Company",
        "IncorporationDate": "01/03/2015",
        "DissolutionDate": "",
        "LastMadeUpDate": "31/03/2026",
        "AccountsCategory": "FULL",
        "SicText_1": "62012 - Business and domestic software development",
        "SicText_2": "",
    }
    base.update(over)
    return base


@pytest.mark.parametrize(
    ("over", "reason"),
    [
        ({}, INCLUDED),
        ({"DissolutionDate": "01/10/2026"}, Reason.ALREADY_DISSOLVED),
        ({"CompanyStatus": "Liquidation"}, Reason.STATUS_NOT_ACTIVE),
        ({"CompanyStatus": "Active - Proposal to Strike off"}, Reason.STRIKE_OFF_PROPOSED),
        ({"CompanyCategory": "Limited Partnership"}, Reason.CATEGORY_EXCLUDED),
        ({"IncorporationDate": "01/12/2023"}, Reason.TOO_YOUNG),
        ({"IncorporationDate": ""}, Reason.TOO_YOUNG),
        ({"SicText_1": "None Supplied"}, Reason.SIC_MISSING),
        ({"SicText_1": "64191 - Banks"}, Reason.SIC_EXCLUDED),
        ({"SicText_1": "62012 - x", "SicText_2": "99999 - Dormant"}, Reason.SIC_EXCLUDED),
        ({"AccountsCategory": "Dormant"}, Reason.ACCOUNTS_CATEGORY_EXCLUDED),
        ({"AccountsCategory": "NO ACCOUNTS FILED"}, Reason.ACCOUNTS_CATEGORY_EXCLUDED),
        ({"LastMadeUpDate": ""}, Reason.NO_ACCOUNTS),
        ({"LastMadeUpDate": "31/03/2023"}, Reason.ACCOUNTS_STALE),
    ],
)
def test_each_rule_gives_its_reason(over: dict[str, Any], reason: str) -> None:
    res = select_cohort([rec(**over)], RULE)
    assert res.reasons["01234567"] == reason
    assert (res.included == ("01234567",)) == (reason == INCLUDED)


def test_age_boundary_is_whole_months_to_snapshot() -> None:
    ok = select_cohort([rec(IncorporationDate="01/11/2023")], RULE)
    late = select_cohort([rec(IncorporationDate="02/11/2023")], RULE)
    assert ok.included and not late.included


def test_malformed_and_duplicate_numbers() -> None:
    res = select_cohort(
        [rec("1234567"), rec("01234567"), rec(""), rec("AB-1"), rec("sc123456")], RULE
    )
    assert res.reasons["01234567"] == Reason.DUPLICATE_NUMBER
    assert res.reasons[""] == Reason.MALFORMED_NUMBER
    assert res.reasons["AB-1"] == Reason.MALFORMED_NUMBER
    assert res.included == ("SC123456",)


def test_normalisation_and_dates() -> None:
    assert normalise_company_number(1234567) == "01234567"
    assert normalise_company_number(None) is None
    assert parse_date("31/03/2026") == date(2026, 3, 31)
    assert parse_date("2026-03-31") == date(2026, 3, 31)
    assert parse_date(float("nan")) is None


def test_hash_is_order_independent_and_reproducible() -> None:
    recs = [rec(f"{i:08d}") for i in range(1, 30)]
    shuffled = recs[:]
    random.Random(0).shuffle(shuffled)
    a, b = select_cohort(recs, RULE), select_cohort(shuffled, RULE)
    assert a.cohort_hash == b.cohort_hash == select_cohort(recs, RULE).cohort_hash
    assert a.included == b.included and len(a.cohort_hash) == 64


def test_hash_changes_with_membership_version_and_rule() -> None:
    nums = ["00000001", "00000002"]
    h = cohort_hash(nums, RULE)
    assert h != cohort_hash(nums[:1], RULE)
    assert h != cohort_hash(nums, RULE, rule_version="2")
    assert h != cohort_hash(nums, replace(RULE, min_age_months=24))
    assert h != cohort_hash(nums, replace(RULE, outcome_events=("receivership",)))
    assert h == cohort_hash(reversed(nums), RULE)


class _SpyRecord(Mapping[str, Any]):
    """Mapping that records every key read."""

    def __init__(self, data: dict[str, Any], seen: set[str]) -> None:
        self._d, self._seen = data, seen

    def __getitem__(self, k: str) -> Any:
        self._seen.add(k)
        return self._d[k]

    def get(self, k: str, default: Any = None) -> Any:
        self._seen.add(k)
        return self._d.get(k, default)

    def __iter__(self) -> Iterator[str]:
        raise AssertionError("record must not be iterated")

    def __len__(self) -> int:
        return len(self._d)


def test_no_personal_data_field_is_read() -> None:
    personal = {"Careof": "A Person", "RegAddress.AddressLine1": "1 High St", "Officers": "x"}
    seen: set[str] = set()
    res = select_cohort([_SpyRecord({**rec(), **personal}, seen)], RULE)
    assert res.included == ("01234567",)
    assert seen <= set(READ_COLUMNS)
    assert not seen & set(personal)
    # Changing personal fields cannot change the outcome or the hash.
    other = select_cohort([{**rec(), "Careof": "Someone Else"}], RULE)
    assert other.cohort_hash == res.cohort_hash
