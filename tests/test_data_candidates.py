"""The real-data candidate registry stays consistent with the benchmark snapshot.

`scripts/candidates/check_overlap.py` is the enforcement; this runs it in CI so a registry
edit that claims a non-existent benchmark match, or adds a candidate sharing a distinctive
name token with a TabArena/BeyondArena dataset without a recorded review, fails here rather
than surfacing after a checkpoint has trained on benchmark data.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "candidates"))

from check_overlap import BENCHMARKS, REGISTRY, check


def test_registry_is_consistent_with_the_benchmark_snapshot():
    registry = pd.read_csv(REGISTRY, dtype=str, keep_default_na=False)
    registry["benchmark_match"] = registry["benchmark_match"].replace("", None)
    benchmarks = pd.read_csv(BENCHMARKS)
    assert check(registry, benchmarks) == []


def test_checker_catches_an_unrecorded_overlap():
    benchmarks = pd.DataFrame({"benchmark": ["BeyondArena"], "dataset_name": ["lending_club_1m"]})
    registry = pd.DataFrame([{
        "name": "LendingClub mirror", "verdict": "SHORTLIST", "benchmark_match": None,
        "reviewed_not_overlapping": "",
    }])
    assert any("lending_club_1m" in p for p in check(registry, benchmarks))


def test_checker_rejects_a_match_absent_from_the_snapshot():
    benchmarks = pd.DataFrame({"benchmark": ["TabArena-v0.1"], "dataset_name": ["heloc"]})
    registry = pd.DataFrame([{
        "name": "x", "verdict": "EXCLUDED_BENCHMARK", "benchmark_match": "helocc",
        "reviewed_not_overlapping": "",
    }])
    assert any("not in the snapshot" in p for p in check(registry, benchmarks))
