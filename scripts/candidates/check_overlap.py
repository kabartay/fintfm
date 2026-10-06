"""Check the real-data candidate registry against TabArena and BeyondArena, mechanically.

Why this exists: on 2026-10-06 LendingClub was checked against TabArena-v0.1's 51 datasets,
called clean, and a loader was built -- before anyone looked at BeyondArena, the TabArena
maintainers' own named next benchmark, which contains `lending_club_1m`. The check was right
about the list it used and wrong about the question that mattered. A registry nobody verifies
drifts the same way, so this script makes two things fail loudly instead of relying on memory:

1. Every `benchmark_match` the registry claims must name a dataset that actually exists in
   the benchmark snapshot (`docs/research/benchmark_datasets.csv`) -- a typo'd or stale match
   is an error, not a silent pass.
2. Every candidate not already excluded is compared token-by-token against all benchmark
   dataset names; a shared distinctive token (e.g. "bondora", "lending") with no recorded
   match is reported, so an overlap nobody noticed surfaces before training, not after. A
   collision a human has inspected and judged unrelated is recorded by name in
   `reviewed_not_overlapping` -- the judgement is written down rather than the check loosened.

Usage:
    uv run python scripts/candidates/check_overlap.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "docs" / "research" / "data_candidates.csv"
BENCHMARKS = ROOT / "docs" / "research" / "benchmark_datasets.csv"

VERDICTS = {
    "IN_USE", "SHORTLIST", "HOLD_LICENCE", "HOLD_ACCESS", "HOLD_BREADTH",
    "EXCLUDED_BENCHMARK", "EXCLUDED_DUPLICATE", "EXCLUDED_SYNTHETIC", "EXCLUDED_NOT_ROW_LEVEL",
    "EXCLUDED_NO_LABEL", "EXCLUDED_TOO_SMALL", "EXCLUDED_DETERMINISTIC", "EXCLUDED_NO_DATA",
}

#: Tokens too generic to signal an overlap on their own ("credit", "data", "prediction" appear
#: in dozens of unrelated names). Distinctive tokens -- platform, place or dataset names -- are
#: what the fuzzy check looks for.
_GENERIC = {
    "data", "dataset", "datasets", "prediction", "credit", "loan", "loans", "default", "risk",
    "uci", "kaggle", "zenodo", "the", "and", "of", "for", "in", "only", "csv", "release",
    "detection", "classification", "survival", "patient", "disease", "set", "sets", "1m",
    "fraud", "financial", "customer", "card", "public", "level", "single", "family",
}


def _tokens(text: str) -> set[str]:
    """Lower-cased alphabetic tokens of length >= 4, minus generic vocabulary.

    Args:
        text: A dataset or candidate name.

    Returns:
        The distinctive tokens in `text`.
    """
    return {t for t in re.findall(r"[a-z]+", text.lower()) if len(t) >= 4} - _GENERIC


def _flat(text: str) -> str:
    """Letters only, lower-cased, so "LendingClub" and "lending_club_1m" can be compared.

    Token sets alone miss exactly that pair -- one word on one side, two on the other -- which
    is the overlap this script was written to catch; a regression test pins it.

    Args:
        text: A dataset or candidate name.

    Returns:
        `text` with every non-letter removed.
    """
    return re.sub(r"[^a-z]", "", text.lower())


def check(registry: pd.DataFrame, benchmarks: pd.DataFrame) -> list[str]:
    """Return every problem found; an empty list means the registry is consistent.

    Args:
        registry: The candidate registry.
        benchmarks: The benchmark snapshot (`benchmark`, `dataset_name`, ...).

    Returns:
        Human-readable problem descriptions.
    """
    problems: list[str] = []
    names = set(benchmarks["dataset_name"])
    for _, row in registry.iterrows():
        if row["verdict"] not in VERDICTS:
            problems.append(f"{row['name']!r}: unknown verdict {row['verdict']!r}")
        match = row["benchmark_match"]
        if isinstance(match, str) and match and match not in names:
            problems.append(f"{row['name']!r}: benchmark_match {match!r} is not in the snapshot")
        if row["verdict"] == "EXCLUDED_BENCHMARK" and not (isinstance(match, str) and match):
            problems.append(f"{row['name']!r}: EXCLUDED_BENCHMARK with no benchmark_match")

    bench_tokens = {n: _tokens(n) for n in names}
    bench_flat = {n: _flat(n) for n in names}
    for _, row in registry.iterrows():
        reviewed = {x for x in str(row.get("reviewed_not_overlapping") or "").split(";") if x}
        for name in sorted(reviewed - names):
            problems.append(
                f"{row['name']!r}: reviewed_not_overlapping names {name!r}, not in the snapshot"
            )
        if row["verdict"].startswith("EXCLUDED") or isinstance(row["benchmark_match"], str):
            continue
        cand, cand_flat = _tokens(row["name"]), _flat(row["name"])
        hits = sorted(
            n for n, toks in bench_tokens.items()
            if n not in reviewed
            and (
                cand & toks
                or any(t in cand_flat for t in toks)
                or any(t in bench_flat[n] for t in cand)
            )
        )
        if hits:
            problems.append(
                f"{row['name']!r} ({row['verdict']}) shares a distinctive token with benchmark "
                f"dataset(s) {hits} but records no benchmark_match -- verify by hand"
            )
    return problems


def main() -> None:
    """Entry point: print a verdict summary and every problem; exit 1 if any problem."""
    registry = pd.read_csv(REGISTRY, dtype=str, keep_default_na=False)
    registry["benchmark_match"] = registry["benchmark_match"].replace("", None)
    benchmarks = pd.read_csv(BENCHMARKS)
    print(registry["verdict"].value_counts().to_string())
    problems = check(registry, benchmarks)
    for p in problems:
        print("PROBLEM:", p)
    print(f"{len(problems)} problem(s)")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
