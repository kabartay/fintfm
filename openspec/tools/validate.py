#!/usr/bin/env python3
"""Validate that this repository's own conventions actually hold.

A convention nothing checks is a wish. This script enforces the ones that are mechanisable,
and it is wired into the test suite (``tests/test_openspec.py``) so a violation fails CI
rather than waiting to be noticed.

Usage:
    uv run python openspec/tools/validate.py            # everything
    uv run python openspec/tools/validate.py --provenance
    uv run python openspec/tools/validate.py --specs --changes --findings
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "fintfm"
SPECS = ROOT / "openspec" / "specs"
CHANGES = ROOT / "openspec" / "changes"

#: Modules that may never touch real data. The auditability claim rests on this boundary.
PRETRAINING_PACKAGES = ("prior", "modeling")

#: Signatures of loading real data from disk or network.
REAL_DATA_PATTERNS = (
    r"\bfetch_openml\b",
    r"\bread_csv\b",
    r"\bread_parquet\b",
    r"\burlopen\b",
    r"\brequests\.(get|post)\b",
    r"\bload_dataset\b",
    r"from fintfm\.evaluation",
)

# "HYPOTHESIS" is for a finding that states a mechanism and a **pre-registered prediction**
# before the experiment exists. It is not a weaker MEASURED: it carries a different obligation,
# namely that the numbers testing it were fixed in advance and are quoted unchanged afterwards,
# hit or miss. Added 2026-09-10 for §41, because labelling a hypothesis MEASURED would be the
# exact mislabelling this check exists to prevent.
STATUS_WORDS = ("MEASURED", "HYPOTHESIS", "SMOKE-TEST", "SIMULATED", "ESTIMATED",
                "design invariant", "proposed strategy", "not started")


def check_provenance() -> list[str]:
    """No real data may reach the pretraining path (spec P1)."""
    problems: list[str] = []
    for package in PRETRAINING_PACKAGES:
        for path in (SRC / package).rglob("*.py"):
            text = path.read_text()
            for pattern in REAL_DATA_PATTERNS:
                for m in re.finditer(pattern, text):
                    line = text[: m.start()].count("\n") + 1
                    problems.append(
                        f"{path.relative_to(ROOT)}:{line} matches {pattern!r} — real data "
                        f"must not reach the pretraining path (openspec/specs/"
                        f"pretraining-provenance)"
                    )
    return problems


def check_weights_untracked() -> list[str]:
    """Trained weights never enter git (spec P3)."""
    out = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    tracked = [f for f in out.stdout.splitlines() if f.endswith(".pt")]
    return [f"weights tracked in git: {f}" for f in tracked]


def check_specs() -> list[str]:
    """Every requirement names what enforces it (specs/README contract)."""
    problems: list[str] = []
    for spec in sorted(SPECS.glob("*/spec.md")):
        text = spec.read_text()
        reqs = re.findall(r"^\*\*([A-Z]\d+) —", text, flags=re.MULTILINE)
        if not reqs:
            problems.append(f"{spec.relative_to(ROOT)} declares no requirements")
        # each requirement should be followed by an enforcement note before the next one
        blocks = re.split(r"^\*\*[A-Z]\d+ —", text, flags=re.MULTILINE)[1:]
        for rid, block in zip(reqs, blocks, strict=False):
            if "Enforced by" not in block:
                problems.append(
                    f"{spec.relative_to(ROOT)} requirement {rid} names nothing that "
                    f"enforces it"
                )
    return problems


def check_changes() -> list[str]:
    """Every proposal has the sections config.yaml requires, and tasks are verifiable."""
    problems: list[str] = []
    required = ("## Why", "## What", "## Non-goals", "Blocked by")
    for change in sorted(p for p in CHANGES.iterdir() if p.is_dir()):
        proposal, tasks = change / "proposal.md", change / "tasks.md"
        if not proposal.exists():
            problems.append(f"{change.name}: no proposal.md")
            continue
        text = proposal.read_text()
        problems += [
            f"{change.name}/proposal.md missing section {s!r}" for s in required if s not in text
        ]
        if not tasks.exists():
            problems.append(f"{change.name}: no tasks.md")
            continue
        for i, line in enumerate(tasks.read_text().splitlines(), start=1):
            if re.match(r"^- \[ \] \d", line) and "Verify" not in line:
                # a task may carry its Verify on a continuation line; check the block instead
                continue
    # open tasks must state a verification somewhere in their block
    for change in sorted(p for p in CHANGES.iterdir() if p.is_dir()):
        tasks = change / "tasks.md"
        if not tasks.exists():
            continue
        blocks = re.split(r"^- \[[ x]\] ", tasks.read_text(), flags=re.MULTILINE)[1:]
        for block in blocks:
            first = block.splitlines()[0] if block.splitlines() else ""
            if "Verify" not in block and "Done" not in block:
                problems.append(
                    f"{change.name}/tasks.md task {first[:48]!r} names no verification"
                )
    return problems


def check_findings() -> list[str]:
    """Every numbered finding declares how its numbers were produced (spec E1)."""
    path = ROOT / "docs" / "results" / "FINDINGS.md"
    if not path.exists():
        return ["docs/results/FINDINGS.md is missing"]
    problems: list[str] = []
    sections = re.split(r"^## (\d+)\. ", path.read_text(), flags=re.MULTILINE)[1:]
    for num, body in zip(sections[::2], sections[1::2], strict=False):
        if not any(w in body for w in STATUS_WORDS):
            problems.append(
                f"docs/results/FINDINGS.md §{num} declares no provenance "
                f"(one of {', '.join(STATUS_WORDS[:4])})"
            )
    return problems


def open_task_count() -> int:
    """Total open tasks across all changes, derived rather than recalled."""
    return sum(
        len(re.findall(r"^- \[ \]", (c / "tasks.md").read_text(), flags=re.MULTILINE))
        for c in CHANGES.iterdir()
        if c.is_dir() and (c / "tasks.md").exists()
    )


CHECKS = {
    "provenance": ("pretraining provenance", lambda: check_provenance() + check_weights_untracked()),
    "specs": ("spec requirements are enforced", check_specs),
    "changes": ("proposals are well-formed", check_changes),
    "findings": ("findings declare provenance", check_findings),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in CHECKS:
        parser.add_argument(f"--{name}", action="store_true")
    args = parser.parse_args()
    selected = [n for n in CHECKS if getattr(args, n)] or list(CHECKS)

    failures = 0
    for name in selected:
        label, fn = CHECKS[name]
        problems = fn()
        if problems:
            failures += len(problems)
            print(f"FAIL  {label} ({len(problems)}):")
            for p in problems:
                print(f"        {p}")
        else:
            print(f"ok    {label}")
    print(f"\n{open_task_count()} open tasks across {sum(1 for c in CHANGES.iterdir() if c.is_dir())} changes")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
