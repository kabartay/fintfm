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
