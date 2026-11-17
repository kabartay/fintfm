"""The release archive stays under Zenodo's preview limit.

GitHub builds a release's zip with ``git archive``, honouring ``export-ignore`` in
``.gitattributes``, and Zenodo archives that zip. Zenodo's previewer refuses a zip with more than
1,000 entries (files plus folders): v0.11.0 had 1,010 and its record shows "Too many files inside
the ZIP file" instead of the file tree. This test builds the same archive and counts its entries.
"""

from __future__ import annotations

import io
import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
#: Zenodo's previewer limit on zip entries.
ZENODO_PREVIEW_MAX_ENTRIES = 1000


def _archive_entries() -> int:
    """Number of entries in ``git archive --format=zip HEAD`` with the working tree's attributes."""
    out = subprocess.run(
        ["git", "archive", "--worktree-attributes", "--format=zip", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        check=True,
    ).stdout
    return len(zipfile.ZipFile(io.BytesIO(out)).namelist())


@pytest.mark.skipif(
    shutil.which("git") is None or not (ROOT / ".git").exists(), reason="needs a git checkout"
)
def test_release_archive_fits_zenodo_preview() -> None:
    """The release zip has fewer than 1,000 entries, so Zenodo can list its files."""
    n = _archive_entries()
    assert n < ZENODO_PREVIEW_MAX_ENTRIES, (
        f"release archive has {n} entries; Zenodo previews at most {ZENODO_PREVIEW_MAX_ENTRIES}. "
        "Add export-ignore paths to .gitattributes."
    )
