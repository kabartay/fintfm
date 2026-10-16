"""The DR4.2 proxy suite stays outside TabArena-v0.1 and BeyondArena and inside its licences.

Runs without a network or the TabArena fork: the candidate table, the licences and the two
benchmarks' dataset lists are all recorded in the source (`fintfm.evaluation.proxy_suite`,
`fintfm.evaluation.overlap`).
"""

from __future__ import annotations

from collections import Counter

import pytest

from fintfm.evaluation.overlap import BEYONDARENA, TABARENA_V0_1, same_dataset_name
from fintfm.evaluation.proxy_suite import (
    CANDIDATES,
    KINDS,
    SAME_SOURCE,
    SUITE,
    TARGET_COUNTS,
    Candidate,
    exclusion_reasons,
    licence_verdict,
    main,
)


def test_the_candidate_table_is_the_two_openml_suites_whole():
    """72 CC18 and 35 CTR23 tasks, each once, so a dropped row cannot pass for a filter."""
    by_suite = Counter(c.suite for c in CANDIDATES)
    assert by_suite == {"CC18": 72, "CTR23": 35}
    assert len({c.task_id for c in CANDIDATES}) == len(CANDIDATES)
    assert len({c.dataset_id for c in CANDIDATES}) == len(CANDIDATES)
    assert {c.kind for c in CANDIDATES} <= set(KINDS)
    assert all(c.kind == "regression" for c in CANDIDATES if c.suite == "CTR23")
    assert set(SAME_SOURCE) <= {c.dataset_id for c in CANDIDATES}


def test_suite_shares_no_openml_id_or_name_with_tabarena_or_beyondarena():
    """Task DR4.2: no suite task is a TabArena-v0.1 or BeyondArena dataset by id or name."""
    tabarena_ids = {i for i, _ in TABARENA_V0_1}
    names = [n for _, n in TABARENA_V0_1] + list(BEYONDARENA)
    assert SUITE, "the suite is empty"
    clashes = [(c.name, "TabArena-v0.1 id") for c in SUITE if c.dataset_id in tabarena_ids]
    clashes += [(c.name, n) for c in SUITE for n in names if same_dataset_name(c.name, n)]
    clashes += [(c.name, SAME_SOURCE[c.dataset_id]) for c in SUITE if c.dataset_id in SAME_SOURCE]
    assert clashes == []


def test_every_suite_task_carries_an_accepted_licence():
    """Only CC BY, CC0 or public domain, as recorded from the OpenML data API."""
    assert [(c.name, c.licence) for c in SUITE if licence_verdict(c.licence)] == []


@pytest.mark.parametrize(
    "licence",
    ["CC BY 4.0", "CC BY", "CC0: Public Domain", "CC 0: Public Domain", "Public Domain"],
)
def test_accepted_licences(licence: str):
    """The spellings OpenML actually records for the licences the suite admits."""
    assert licence_verdict(licence) is None


@pytest.mark.parametrize(
    "licence",
    ["Public", "public", "", "GPL (>= 2)", "Non-commercial scholarly and teaching purposes.",
     "CC BY-NC 4.0", "CC BY-SA 4.0", "CC BY-ND 4.0"],
)
def test_rejected_licences(licence: str):
    """OpenML's default ``Public`` grants research use only, so it is not a verified licence."""
    assert licence_verdict(licence)


def test_a_planted_overlap_is_excluded():
    """The filters catch an overlap by name and by id, not only on the current table."""
    planted = Candidate("CC18", 1, 46918, "credit-g", "binary", 1000, 20, 2, "CC BY 4.0")
    reasons = exclusion_reasons(planted)
    assert "TabArena-v0.1 dataset id" in reasons
    assert any("credit-g" in r for r in reasons) and any("credit_g" in r for r in reasons)


def test_suite_respects_the_target_mix_and_is_deterministic():
    """Never more than the target per kind, grouped by kind, lowest task id first."""
    counts = Counter(c.kind for c in SUITE)
    assert all(counts[k] <= TARGET_COUNTS[k] for k in KINDS)
    for kind in KINDS:
        ids = [c.task_id for c in SUITE if c.kind == kind]
        assert ids == sorted(ids)


def test_dry_run_accounts_for_every_candidate(capsys: pytest.CaptureFixture[str]):
    """Each candidate is printed once, either in the suite or with its exclusion reasons."""
    assert main(["--dry-run"]) == 0
    out = capsys.readouterr().out
    for c in CANDIDATES:
        assert out.count(f"task {c.task_id:>6}  data {c.dataset_id:>5}") == 1, c.name
    assert f"excluded: {len(CANDIDATES) - len(SUITE)} of {len(CANDIDATES)}" in out


def test_task_ids_lists_the_suite(capsys: pytest.CaptureFixture[str]):
    """``--task-ids`` is what the TabArena runner will read."""
    assert main(["--task-ids"]) == 0
    assert capsys.readouterr().out.strip() == ",".join(str(c.task_id) for c in SUITE)
