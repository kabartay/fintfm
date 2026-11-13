"""Build the GitHub Pages leaderboard (``docs/leaderboard/index.html``) from a TabArena-Lite run.

The page used to be hand-edited, so its numbers drifted from the measurement log. It is now
generated in two steps, and ``tests/test_build_leaderboard_page.py`` fails when the committed
page differs from what ``build`` produces.

``extract`` reads a leaderboard the TabArena fork wrote (``eval/<run>/tabarena_leaderboard.csv``),
ranks every method by Elo, drops the rows matching ``--omit`` from what is listed and writes a
small CSV under ``docs/leaderboard/data/``. Omitted methods still count: every listed row keeps
its position among all methods, and the page states how many are not listed. The board's
metadata (row count, omitted count, source run, CSV name) goes into ``page.json``.

``build`` fills ``docs/leaderboard/template.html`` from that CSV and the editorial text in
``docs/leaderboard/data/page.json`` (dates, findings, the entry's description), which are
written by hand and must quote ``docs/results/FINDINGS.md``.

    uv run python scripts/build_leaderboard_page.py extract \\
        ~/proj/tabarena/eval/fintfm_linprobe_all51_1113 --omit 'REGEX'
    uv run python scripts/build_leaderboard_page.py build
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAGE_DIR = ROOT / "docs" / "leaderboard"
DATA_DIR = PAGE_DIR / "data"
PAGE_JSON = DATA_DIR / "page.json"
TEMPLATE = PAGE_DIR / "template.html"
OUTPUT = PAGE_DIR / "index.html"

#: Columns of the extracted CSV, in order.
FIELDS = ("position", "method", "elo", "elo_plus", "elo_minus", "winrate", "imputed")


def ordinal(n: int) -> str:
    """Return ``n`` with its English ordinal suffix.

    Args:
        n: A positive integer.

    Returns:
        For example ``"1st"``, ``"106th"``, ``"112th"``.
    """
    if 10 <= n % 100 <= 20:
        return f"{n}th"
    return f"{n}{ {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th') }"


def extract(run_dir: Path, omit: str | None, out_name: str | None = None) -> Path:
    """Write the listed rows of a TabArena leaderboard and record the board in ``page.json``.

    Args:
        run_dir: A fork ``eval/<run>/`` directory holding ``tabarena_leaderboard.csv``.
        omit: Regular expression (case-insensitive) for methods not to list; they still
            count in every position. ``None`` lists every method.
        out_name: File name under ``docs/leaderboard/data/``; defaults to ``<run>.csv``.

    Returns:
        The path of the CSV written.
    """
    with (run_dir / "tabarena_leaderboard.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: -float(r["elo"]))
    pattern = re.compile(omit, re.IGNORECASE) if omit else None
    listed = []
    for position, r in enumerate(rows, start=1):
        if pattern is not None and pattern.search(r["method"]):
            continue
        listed.append(
            {
                "position": position,
                "method": r["method"],
                "elo": f"{float(r['elo']):.1f}",
                "elo_plus": f"{float(r['elo+']):.1f}",
                "elo_minus": f"{float(r['elo-']):.1f}",
                "winrate": f"{float(r['winrate']):.4f}",
                "imputed": f"{float(r['imputed'] or 0):.4f}",
            }
        )
    out = DATA_DIR / (out_name or f"{run_dir.name}.csv")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(listed)
    page = json.loads(PAGE_JSON.read_text())
    page["board"] = {
        "csv": out.name,
        "source_run": run_dir.name,
        "n_total": len(rows),
        "n_omitted": len(rows) - len(listed),
    }
    PAGE_JSON.write_text(json.dumps(page, indent=2, ensure_ascii=False) + "\n")
    return out


def _js(value: object) -> str:
    """Encode ``value`` as JSON that is safe inside an inline ``<script>``.

    Args:
        value: Any JSON-serialisable value.

    Returns:
        Compact JSON with ``</`` escaped, so no string can close the script element.
    """
    return json.dumps(value, ensure_ascii=False, separators=(", ", ": ")).replace("</", "<\\/")


def render() -> str:
    """Render the page from the template, ``page.json`` and the CSV it names.

    Returns:
        The complete HTML of ``docs/leaderboard/index.html``.

    Raises:
        ValueError: If the board is not recorded, the FinTFM row is missing, or a
            placeholder is left unfilled.
    """
    page = json.loads(PAGE_JSON.read_text())
    board = page["board"]
    if not board:
        raise ValueError("page.json has no board; run `extract` first")
    with (DATA_DIR / board["csv"]).open(newline="") as f:
        rows = list(csv.DictReader(f))
    data = [
        {
            "rank": int(r["position"]),
            "method": r["method"],
            "elo": float(r["elo"]),
            "elo_hi": float(r["elo_plus"]),
            "elo_lo": float(r["elo_minus"]),
            "winrate": float(r["winrate"]),
            "imputed": float(r["imputed"]),
            "new": r["method"] == page["self_method"],
        }
        for r in rows
    ]
    selves = [d for d in data if d["new"]]
    if len(selves) != 1:
        raise ValueError(f"expected one row named {page['self_method']!r}, found {len(selves)}")
    self_row = selves[0]
    if self_row["imputed"] > 0:
        raise ValueError("the FinTFM row is imputed; the page states that it is not")
    n_omitted = board["n_omitted"]
    omitted_text = (
        f"{n_omitted} {'entry is' if n_omitted == 1 else 'entries are'} not listed on this page; "
        "they count in every Elo and every position shown. "
        if n_omitted
        else ""
    )
    cards = "\n".join(
        f'    <div class="variant{" is-self" if t["self"] else ""}">\n'
        f'      <div class="v-elo">{t["elo"]}</div>\n'
        f'      <div class="v-name">{t["name"]}</div>\n'
        f'      <div class="v-sub">{t["sub"]}</div>\n'
        "    </div>"
        for t in page["types"]
    )
    values = {
        "self_elo": f"{self_row['elo']:.0f}",
        "self_position": ordinal(self_row["rank"]),
        "n_total": str(board["n_total"]),
        "self_label": page["self_label"],
        "n_datasets": str(page["n_datasets"]),
        "finding": page["finding"],
        "date": page["date"],
        "bar_text": page["bar_text"],
        "omitted_text": omitted_text,
        "entry_text": page["entry_text"],
        "types_note": page["types_note"],
        "type_cards": cards,
        "csv_name": board["csv"],
        "source_run": board["source_run"],
        "history_text": page["history_text"],
        "data_json": _js(data),
        "anchors_json": _js(page["anchors"]),
    }
    html = TEMPLATE.read_text()
    for key, value in values.items():
        html = html.replace("{{" + key + "}}", value)
    left = re.findall(r"\{\{[a-z_]+\}\}", html)
    if left:
        raise ValueError(f"unfilled placeholders: {sorted(set(left))}")
    return html


def main(argv: list[str] | None = None) -> None:
    """Command-line entry: ``extract`` or ``build``.

    Args:
        argv: Arguments; ``None`` reads ``sys.argv``.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    ex = sub.add_parser("extract", help="write the listed rows of a fork leaderboard")
    ex.add_argument("run_dir", type=Path, help="fork eval/<run>/ directory")
    ex.add_argument("--omit", default=None, help="regex of methods not to list (they still count)")
    ex.add_argument("--out-name", default=None, help="CSV file name under docs/leaderboard/data/")
    sub.add_parser("build", help="render docs/leaderboard/index.html")
    args = parser.parse_args(argv)
    if args.command == "extract":
        print(f"wrote {extract(args.run_dir.expanduser(), args.omit, args.out_name)}")
    else:
        OUTPUT.write_text(render())
        print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
