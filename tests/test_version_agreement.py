"""The three places a release states its version agree, on every push.

The release workflow checks this only when a tag is pushed, which is after the GitHub release is
already public: v0.7.0 was published with `.zenodo.json` still at 0.6.0, so its Zenodo record
states the wrong version and it had to be superseded by 0.7.1. Checking on every push moves the
failure to before the tag, where fixing it costs one commit.
"""

import json
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_pyproject_citation_and_zenodo_state_the_same_version():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    cff = re.search(r"^version:\s*(\S+)", (ROOT / "CITATION.cff").read_text(), re.MULTILINE)
    cited = cff.group(1).strip('"')
    deposited = json.loads((ROOT / ".zenodo.json").read_text())["version"]
    assert project == cited == deposited, (
        f"pyproject {project}, CITATION.cff {cited}, .zenodo.json {deposited}: bump all three"
    )


def test_package_and_readme_citation_state_the_same_version():
    """`fintfm.__version__` and the README's BibTeX drifted before (0.5.5 at 0.7.1, audit AU3.4)."""
    import fintfm

    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    bib = re.search(r"version\s*=\s*\{([^}]+)\}", (ROOT / "README.md").read_text())
    assert bib is not None, "README.md has no BibTeX version field"
    assert fintfm.__version__ == project, f"installed {fintfm.__version__}, pyproject {project}: re-sync"
    assert bib.group(1).strip() == project, f"README BibTeX {bib.group(1)}, pyproject {project}"
