"""Fail if any installed dependency is not permissively licensed.

Why this is a CI job and not a README paragraph
------------------------------------------------

`README.md` states that every dependency is permissive and lists them by hand. A hand-written
list is a claim about the past: it was true when written and says nothing about the dependency
added last week, or about a transitive dependency nobody chose. This project's most-repeated
rule is to check a licence rather than assume one — four of the ten peer projects surveyed in
`docs/paper/RELATED_WORK.md` ship permissive code with **non-commercial weights**, a split
invisible from a repository's headline badge — and a rule enforced only by memory is a rule
that fails silently the first time someone is busy.

**This checks code licences of Python distributions only.** It cannot see a model's weights
licence, which is a separate question and the one that actually bites (`CLAUDE.md`).
"""

from __future__ import annotations

import sys
from importlib.metadata import distributions

#: Substrings that identify a permissive licence. Matched case-insensitively against whatever
#: the distribution declares, which is inconsistent across packaging generations -- some use
#: `License-Expression` (PEP 639), some a free-text `License`, some only classifiers.
PERMISSIVE: tuple[str, ...] = (
    "MIT", "BSD", "Apache", "ISC", "Zlib", "Unlicense", "HPND",
    "Python Software Foundation", "PSF", "MPL", "Mozilla Public",
)

#: Distributions whose metadata is absent or unparseable, with the licence verified by hand
#: and the reason recorded. **An entry here is a claim someone made, not one the tool
#: checked** -- keep it short, and re-verify when a major version changes.
KNOWN_GOOD: dict[str, str] = {
    # Ships no License field and no license classifier; LICENSE in the sdist is BSD-3-Clause
    # style with Facebook/NVIDIA copyright, distributed as Apache-2.0 per pytorch.org.
    "torch": "BSD-3-Clause (verified from the project's LICENSE, 2026-09-25)",
    # Declares the literal string "Dual License"; it is dual Apache-2.0 / BSD-3-Clause, both
    # permissive, so the ambiguity is in the metadata rather than the terms.
    "python-dateutil": "dual Apache-2.0 / BSD-3-Clause",
}


def declared_licence(meta) -> str:
    """Best available licence string for a distribution, across packaging generations."""
    for key in ("License-Expression", "License"):
        value = (meta.get(key) or "").strip()
        # A free-text License field sometimes holds the entire licence text; a long value is
        # not a licence name and the classifiers are more reliable in that case.
        if value and len(value) <= 60:
            return value
    classifiers = [
        c.split("::")[-1].strip()
        for c in (meta.get_all("Classifier") or [])
        if c.startswith("License")
    ]
    return "; ".join(classifiers)


def main() -> int:
    offenders: list[tuple[str, str]] = []
    unknown: list[str] = []
    for dist in distributions():
        name = dist.metadata.get("Name") or "<unnamed>"
        if name in KNOWN_GOOD:
            continue
        licence = declared_licence(dist.metadata)
        if not licence:
            unknown.append(name)
        elif not any(p.lower() in licence.lower() for p in PERMISSIVE):
            offenders.append((name, licence))

    for name, licence in offenders:
        print(f"NON-PERMISSIVE  {name}: {licence}", file=sys.stderr)
    for name in unknown:
        print(f"NO LICENCE METADATA  {name}", file=sys.stderr)

    if offenders or unknown:
        print(
            "\nEither the dependency is unacceptable, or its licence is fine and the metadata "
            "is not.\nIn the second case add it to KNOWN_GOOD **with the licence you verified "
            "and the date**,\nrather than widening PERMISSIVE -- a widened pattern silently "
            "admits the next one too.",
            file=sys.stderr,
        )
        return 1

    print(f"all dependency licences permissive ({len(KNOWN_GOOD)} verified by hand)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
