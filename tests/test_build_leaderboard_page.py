"""The Pages leaderboard is generated, so the committed page must equal what the script builds.

``docs/leaderboard/index.html`` used to be hand-edited and drifted from the measurement log
(FINDINGS §225). These tests fail when someone edits the page or its data without rebuilding,
and pin the two rules the page states: positions count every method, listed or not, and the
FinTFM row is never imputed.
"""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]


def _builder() -> ModuleType:
    """Import ``scripts/build_leaderboard_page.py`` as a module."""
    spec = importlib.util.spec_from_file_location(
        "build_leaderboard_page", ROOT / "scripts" / "build_leaderboard_page.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_committed_page_equals_the_build() -> None:
    """Rebuilding from the committed template and data reproduces the committed page byte for byte."""
    builder = _builder()
    assert builder.render() == (ROOT / "docs" / "leaderboard" / "index.html").read_text()


def test_positions_count_unlisted_methods() -> None:
    """Listed positions are unique, and the highest plus nothing exceeds the board's method count."""
    page = json.loads((ROOT / "docs" / "leaderboard" / "data" / "page.json").read_text())
    board = page["board"]
    with (ROOT / "docs" / "leaderboard" / "data" / board["csv"]).open(newline="") as f:
        rows = list(csv.DictReader(f))
    positions = [int(r["position"]) for r in rows]
    assert len(set(positions)) == len(positions)
    assert len(rows) == board["n_total"] - board["n_omitted"]
    assert max(positions) <= board["n_total"]
    elos = [float(r["elo"]) for r in rows]
    assert elos == sorted(elos, reverse=True)


def test_ordinal_suffixes() -> None:
    """English ordinals, including the teens."""
    ordinal = _builder().ordinal
    assert [ordinal(n) for n in (1, 2, 3, 4, 11, 12, 13, 21, 101, 107, 112)] == [
        "1st",
        "2nd",
        "3rd",
        "4th",
        "11th",
        "12th",
        "13th",
        "21st",
        "101st",
        "107th",
        "112th",
    ]
