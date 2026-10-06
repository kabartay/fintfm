"""Pure date/quarter arithmetic in `scripts/edgar/build_panel.py` (task A5).

Only the network-free logic is tested here -- `_download_quarter`, `_load_quarter_fundamentals`,
`_bankruptcy_events` and `build_panel` all call live SEC endpoints and are exercised by actually
running the script (§151, task A5's local smoke run), not by CI. The three functions below
each had a real bug caught by hand-running them against known values rather than trusting the
algebra (quarter-end date math was a month early; the bankruptcy-event search window needs to
extend past the panel's last quarter by the label horizon) -- these tests exist so a future
change cannot silently reintroduce either.
"""

import sys
from datetime import date
from pathlib import Path

import pytest

# build_panel.py imports `requests` at module level (the `real` extra, uv sync --extra real).
# CI only syncs `bench` (matching every other optional extra's test here, e.g.
# test_train.py's `pytest.importorskip("schedulefree")`) -- skip rather than fail collection
# when it is not installed, instead of making CI install every extra just for this file.
pytest.importorskip("requests")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "edgar"))

from build_panel import _add_quarters, _month_windows, _quarters


def test_quarters_spans_a_single_year():
    assert _quarters("2023q1", "2023q4") == ["2023q1", "2023q2", "2023q3", "2023q4"]


def test_quarters_spans_a_year_boundary():
    assert _quarters("2023q3", "2024q2") == ["2023q3", "2023q4", "2024q1", "2024q2"]


def test_quarters_single_quarter():
    assert _quarters("2023q1", "2023q1") == ["2023q1"]


def test_add_quarters_within_a_year():
    assert _add_quarters("2023q1", 1) == "2023q2"


def test_add_quarters_across_a_year_boundary():
    # The bug this guards: Q4 + any positive shift must land in the *next* year, not treat
    # month 13 as still-this-year.
    assert _add_quarters("2023q4", 1) == "2024q1"
    assert _add_quarters("2023q4", 4) == "2024q4"


def test_add_quarters_by_zero_is_identity():
    assert _add_quarters("2023q2", 0) == "2023q2"


def test_month_windows_covers_every_calendar_month_in_range():
    windows = _month_windows(date(2023, 1, 1), date(2023, 3, 31))
    assert windows == [
        (date(2023, 1, 1), date(2023, 1, 31)),
        (date(2023, 2, 1), date(2023, 2, 28)),
        (date(2023, 3, 1), date(2023, 3, 31)),
    ]


def test_month_windows_respects_a_partial_first_and_last_month():
    windows = _month_windows(date(2023, 1, 15), date(2023, 2, 10))
    assert windows == [
        (date(2023, 1, 15), date(2023, 1, 31)),
        (date(2023, 2, 1), date(2023, 2, 10)),
    ]


def test_month_windows_handles_a_leap_february():
    windows = _month_windows(date(2024, 2, 1), date(2024, 2, 29))
    assert windows == [(date(2024, 2, 1), date(2024, 2, 29))]


def test_month_windows_single_day():
    windows = _month_windows(date(2023, 6, 15), date(2023, 6, 15))
    assert windows == [(date(2023, 6, 15), date(2023, 6, 15))]
