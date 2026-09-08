"""The repository's own conventions, enforced as tests.

A convention nothing checks is a wish. These wrap `openspec/tools/validate.py` so a
violation fails the suite. When this file was first written the validator found ten real
violations of conventions this repository had already written down, including tasks with no
stated verification and findings with no provenance label.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "openspec" / "tools"))

import validate


def test_no_real_data_reaches_the_pretraining_path():
    """Spec P1. The auditability claim rests on this and it breaks silently."""
    assert validate.check_provenance() == []


def test_no_trained_weights_are_tracked_in_git():
    """Spec P3."""
    assert validate.check_weights_untracked() == []


def test_every_spec_requirement_names_what_enforces_it():
    assert validate.check_specs() == []


def test_every_proposal_is_well_formed_and_every_task_verifiable():
    assert validate.check_changes() == []


def test_every_finding_declares_how_its_numbers_were_produced():
    """Spec E1. A number without provenance cannot be acted on safely."""
    assert validate.check_findings() == []


def test_open_task_count_is_derivable():
    """The queue count must come from the files, never from memory."""
    assert validate.open_task_count() > 0


@pytest.mark.parametrize("package", validate.PRETRAINING_PACKAGES)
def test_pretraining_packages_exist(package):
    """Guard against the provenance check silently passing on a renamed package."""
    assert (ROOT / "src" / "fintfm" / package).is_dir()
