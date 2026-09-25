"""Fail if any relative Markdown link or in-page anchor does not resolve.

Grouping `docs/` into subfolders on 2026-09-25 rewrote 234 references across 135 files. Two
links were already broken before that and nobody had noticed, because a broken link in a
Markdown file costs nothing until someone follows it — and by then the document has been
wrong for weeks.

The same reorganisation broke `openspec/tools/validate.py`, which built its path from parts
(``ROOT / "docs" / "FINDINGS.md"``) while its error message was a string that *did* get
rewritten. It failed reporting a path that existed. This checker cannot catch that class —
only a link a human would click.

It now also checks **prose references in backticks**, because 16 of those survived the
reorganisation pointing at paths that no longer existed. A reader cannot click them, so
nothing broke visibly; they simply told the reader to look somewhere empty. `README.md` is
exempt, since the repository has three of them and only a human can say which is meant.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")

#: Prose references like ``docs/results/FINDINGS.md`` inside backticks. Not clickable, so a
#: stale one is invisible until a reader follows it by hand and finds nothing.
PROSE_REF = re.compile(r"`((?:[a-z_]+/)*[A-Z_]+\.md)`")

#: Filenames the repository has more than one of. Which is meant depends on context, so these
#: are left to a human rather than guessed at.
AMBIGUOUS: frozenset[str] = frozenset({"README.md"})


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

    # Prose references: a bare or mis-pathed `NAME.md` that names a real file elsewhere.
    real = {q.name: str(q) for q in Path("docs").rglob("*.md")}
    for path in files:
        for ref in PROSE_REF.findall(path.read_text()):
            name = ref.split("/")[-1]
            if name in AMBIGUOUS or name not in real or str(path) == real[name]:
                continue
            # A reference is fine if it resolves either from the repository root or from the
            # file's own directory -- `paper/CLAIMS.md` inside docs/README.md is correct, and
            # the first version of this check flagged six such references as broken.
            if (path.parent / ref).exists() or Path(ref).exists():
                continue
            problems.append(f"{path} -> `{ref}` (stale prose reference; now {real[name]})")

    for p in problems:
        print(f"BROKEN LINK  {p}", file=sys.stderr)
    if problems:
        print(f"\n{len(problems)} broken link(s).", file=sys.stderr)
        return 1
    print(f"all relative links and in-page anchors resolve ({len(files)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
