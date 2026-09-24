"""Fail if any relative Markdown link or in-page anchor does not resolve.

Grouping `docs/` into subfolders on 2026-09-25 rewrote 234 references across 135 files. Two
links were already broken before that and nobody had noticed, because a broken link in a
Markdown file costs nothing until someone follows it — and by then the document has been
wrong for weeks.

The same reorganisation broke `openspec/tools/validate.py`, which built its path from parts
(``ROOT / "docs" / "FINDINGS.md"``) while its error message was a string that *did* get
rewritten. It failed reporting a path that existed. This checker cannot catch that class —
only a link a human would click — which is worth stating so its green is not read as more
than it is.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")


def anchor_slug(heading: str) -> str:
    """GitHub's heading-to-anchor rule.

    The subtlety worth getting right: GitHub strips punctuation but does **not** collapse the
    whitespace left behind, so ``Licensing / provenance`` becomes ``licensing--provenance``
    with two hyphens. Collapsing runs of whitespace produces a single hyphen and reports a
    correct link as broken -- which this checker did on its first run, against a link that
    worked.
    """
    slug = heading.strip().lower()
    slug = re.sub(r"[^\w\s-]", "", slug)
    return re.sub(r"[\s_]", "-", slug)


def main() -> int:
    files = [
        Path(p)
        for p in subprocess.run(
            ["git", "ls-files", "*.md"], capture_output=True, text=True, check=True
        ).stdout.split()
    ]
    problems: list[str] = []
    for path in files:
        text = path.read_text()
        anchors = {anchor_slug(h) for h in re.findall(r"^#+\s+(.+)$", text, re.MULTILINE)}
        for target in LINK.findall(text):
            target = target.split()[0]  # drop an optional "title"
            if target.startswith(("http://", "https://", "mailto:")):
                continue  # network checks belong in a scheduled job, not every push
            file_part, _, anchor = target.partition("#")
            if file_part:
                if not (path.parent / file_part).resolve().exists():
                    problems.append(f"{path} -> {target} (no such file)")
            elif anchor and anchor_slug(anchor) not in anchors:
                problems.append(f"{path} -> #{anchor} (no such heading)")

    for p in problems:
        print(f"BROKEN LINK  {p}", file=sys.stderr)
    if problems:
        print(f"\n{len(problems)} broken link(s).", file=sys.stderr)
        return 1
    print(f"all relative links and in-page anchors resolve ({len(files)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
