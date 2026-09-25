"""Validate ``.zenodo.json`` before a release can mint a DOI from it.

Zenodo's GitHub integration fails *quietly*. A malformed deposit is accepted at the webhook
(HTTP 202), discarded during asynchronous processing, and then listed in Zenodo's interface as
"received" -- which is indistinguishable from success without querying the records API. v0.5.2
was lost exactly this way: ``license`` read ``Apache-2.0`` where Zenodo's vocabulary requires
the lowercase SPDX identifier ``apache-2.0``.

So the metadata is checked here, where the failure is loud, rather than discovered afterwards
by noticing that a DOI never appeared. Vocabulary-backed fields are resolved against Zenodo's
live API when the network allows and skipped when it does not, because a release must not
depend on a third party being reachable.
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Final

ZENODO_API: Final = "https://zenodo.org/api"
NETWORK_TIMEOUT_S: Final = 10

# Values Zenodo's legacy deposit schema accepts. Kept literal rather than fetched, because
# these change on a scale of years and an offline run should still catch a typo.
UPLOAD_TYPES: Final = frozenset(
    {
        "publication", "poster", "presentation", "dataset", "image", "video",
        "software", "lesson", "physicalobject", "other",
    }
)
ACCESS_RIGHTS: Final = frozenset({"open", "embargoed", "restricted", "closed"})

REQUIRED: Final = ("upload_type", "title", "description", "creators", "license", "access_right")


def _resolve_licence(licence_id: str) -> str | None:
    """Return an error string if Zenodo does not know ``licence_id``, else ``None``.

    Returns ``None`` when the vocabulary cannot be reached, so that an offline or rate-limited
    run degrades to the local checks rather than blocking a release.
    """
    url = f"{ZENODO_API}/vocabularies/licenses/{licence_id}"
    try:
        with urllib.request.urlopen(url, timeout=NETWORK_TIMEOUT_S) as response:
            if response.getcode() == 200:
                return None
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return (
                f"license {licence_id!r} is not in Zenodo's vocabulary -- it expects the "
                f"lowercase SPDX id (for example 'apache-2.0', not 'Apache-2.0')"
            )
        print(f"  note: licence vocabulary returned HTTP {exc.code}; skipping that check")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print(f"  note: licence vocabulary unreachable ({exc}); skipping that check")
    return None


def _check_creators(creators: Any) -> list[str]:
    """Validate the creator list, which is what attribution on the minted DOI depends on."""
    problems: list[str] = []
    if not isinstance(creators, list) or not creators:
        return ["creators must be a non-empty list"]
    for index, creator in enumerate(creators):
        where = f"creators[{index}]"
        if not isinstance(creator, dict) or "name" not in creator:
            problems.append(f"{where} needs a 'name'")
            continue
        if "," not in creator["name"]:
            problems.append(f"{where} name {creator['name']!r} should be 'Family, Given'")
        orcid = creator.get("orcid")
        # A URL-form ORCID is silently dropped rather than rejected, so the record would
        # publish with no identifier at all -- the failure this file exists to prevent.
        if orcid is not None and orcid.startswith("http"):
            problems.append(f"{where} orcid must be bare digits, not a URL: {orcid!r}")
    return problems


def check(path: Path) -> list[str]:
    """Return every problem found in the deposit metadata at ``path``."""
    try:
        metadata = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        return [f"{path} is not valid JSON: {exc}"]

    problems = [f"missing required field {field!r}" for field in REQUIRED if field not in metadata]

    upload_type = metadata.get("upload_type")
    if upload_type is not None and upload_type not in UPLOAD_TYPES:
        problems.append(f"upload_type {upload_type!r} not one of {sorted(UPLOAD_TYPES)}")

    access_right = metadata.get("access_right")
    if access_right is not None and access_right not in ACCESS_RIGHTS:
        problems.append(f"access_right {access_right!r} not one of {sorted(ACCESS_RIGHTS)}")

    problems.extend(_check_creators(metadata.get("creators")))

    licence_id = metadata.get("license")
    if isinstance(licence_id, str):
        if licence_id != licence_id.lower():
            problems.append(
                f"license {licence_id!r} must be lowercase -- Zenodo's ids are lowercase SPDX"
            )
        else:
            error = _resolve_licence(licence_id)
            if error:
                problems.append(error)

    return problems


def main() -> int:
    path = Path(".zenodo.json")
    if not path.exists():
        print(f"{path} absent; Zenodo would derive metadata from the GitHub repository")
        return 1
    problems = check(path)
    if problems:
        print(f"{path} would not publish:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print(f"{path} is valid Zenodo deposit metadata")
    return 0


if __name__ == "__main__":
    sys.exit(main())
