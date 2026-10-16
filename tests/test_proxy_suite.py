"""The DR4.2 proxy suite stays outside TabArena-v0.1 and BeyondArena and inside its licences.

Runs without a network or the TabArena fork: the candidate table, the licences read at each
dataset's original source and the two benchmarks' dataset lists are all recorded in the source
(`fintfm.evaluation.proxy_suite`, `fintfm.evaluation.overlap`).
"""

from __future__ import annotations

import re
from collections import Counter

import pytest

from fintfm.evaluation.overlap import BEYONDARENA, TABARENA_V0_1, same_dataset_name
from fintfm.evaluation.proxy_suite import (
    CANDIDATES,
    KINDS,
    SAME_SOURCE,
    SOURCE_LICENCES,
    SUITE,
    TARGET_COUNTS,
    Candidate,
    eligible,
    exclusion_reasons,
    licence_verdict,
    main,
    overlap_reasons,
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


def test_every_suite_task_has_a_verified_source_licence():
    """Owner decision 2026-10-16: the licence read at the original source admits a task, never
    OpenML's field. Every suite task has an entry, marked include, whose licence passes."""
    missing = [c.name for c in SUITE if c.dataset_id not in SOURCE_LICENCES]
    assert missing == []
    entries = [(c.name, SOURCE_LICENCES[c.dataset_id]) for c in SUITE]
    assert [n for n, e in entries if not e.include] == []
    assert [(n, e.licence) for n, e in entries if licence_verdict(e.licence)] == []


def test_no_excluded_source_reaches_the_suite():
    """A dataset whose source licence was rejected is excluded, whatever OpenML records."""
    rejected = {i for i, e in SOURCE_LICENCES.items() if not e.include}
    assert rejected, "no rejected source recorded; the check below would be vacuous"
    assert [c.name for c in SUITE if c.dataset_id in rejected] == []
    assert [c.name for c in eligible() if c.dataset_id in rejected] == []
    by_id = {c.dataset_id: c for c in CANDIDATES}
    for i in rejected:
        assert any(r.startswith("source licence") for r in exclusion_reasons(by_id[i])), i


def test_every_overlap_clean_candidate_was_checked_at_source():
    """The source check covers exactly the candidates the overlap filters leave, so a dataset
    cannot pass by never having been looked at, and none was checked needlessly."""
    clean = {c.dataset_id for c in CANDIDATES if not overlap_reasons(c)}
    assert set(SOURCE_LICENCES) == clean


def test_every_source_entry_is_reviewable():
    """Each entry names a page, a dated check and a reason; an include quotes a licence."""
    for i, e in SOURCE_LICENCES.items():
        assert re.match(r"https?://\S+$", e.url), i
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", e.checked), i
        assert e.reason.strip(), i
        if e.include:
            assert e.stated.strip() and e.licence.strip(), i
            assert licence_verdict(e.licence) is None, i


def test_openml_public_alone_does_not_admit_a_task():
    """A candidate with no source entry is excluded even under a licence OpenML calls CC BY."""
    planted = Candidate("CC18", 1, 99_999_999, "planted-clean", "binary", 1000, 20, 2, "CC BY 4.0")
    assert exclusion_reasons(planted) == ["source licence not verified"]


@pytest.mark.parametrize(
    "licence",
    ["CC BY 4.0", "CC BY", "CC0: Public Domain", "CC 0: Public Domain", "Public Domain", "MIT"],
)
def test_accepted_licences(licence: str):
    """The spellings OpenML and the sources record for the licences the suite admits."""
    assert licence_verdict(licence) is None


@pytest.mark.parametrize(
    "licence",
    ["Public", "public", "", "GPL (>= 2)", "Non-commercial scholarly and teaching purposes.",
     "CC BY-NC 4.0", "CC BY-SA 4.0", "CC BY-ND 4.0", "MIT-0 research only"],
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


def test_suite_counts_after_the_source_check():
    """The counts DR4.2 reports, pinned so a table edit that moves them is seen in review."""
    counts = Counter(c.kind for c in SUITE)
    assert dict(counts) == {"binary": 18, "multiclass": 8, "regression": 9}
    capped = Counter(c.kind for c in SUITE if c.n_features <= 136)
    assert dict(capped) == {"binary": 16, "multiclass": 7, "regression": 9}
    assert Counter(c.kind for c in eligible()) == {"binary": 18, "multiclass": 26, "regression": 9}


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
    for kind in KINDS:
        chosen = [c for c in SUITE if c.kind == kind]
        within = sum(c.n_features <= 136 for c in chosen)
        assert f"{kind}: {len(chosen)} of target {TARGET_COUNTS[kind]}, {within} within" in out


def test_task_ids_lists_the_suite(capsys: pytest.CaptureFixture[str]):
    """``--task-ids`` is what the TabArena runner will read."""
    assert main(["--task-ids"]) == 0
    assert capsys.readouterr().out.strip() == ",".join(str(c.task_id) for c in SUITE)
