"""The DR4.1 overlap report: every decision-suite dataset against all of TabArena, by contents.

``python -m fintfm.evaluation.overlap --report docs/results/overlap_report.json --fetch``

Candidates are the datasets arms are chosen on: the OpenML breadth suite
(:data:`fintfm.experiments.openml_breadth.SUITE`), the proxy Elo suite
(:data:`fintfm.evaluation.proxy_suite.SUITE`), and the positive control, OpenML's original
credit-g (:data:`~fintfm.evaluation.overlap.CONTROL_DATASET_ID`), which must overlap
TabArena-v0.1's 46918 or the check is not working. References are all 51 TabArena-v0.1 and all
142 BeyondArena datasets. The fingerprints, thresholds and what they miss are in
:mod:`fintfm.evaluation.overlap`.

The candidates are read first, through the suites' own loaders, and the union of their row
hashes is passed to :func:`~fintfm.evaluation.overlap.parquet_fingerprint` as ``keep_rows``, so
each reference keeps only the hashes some candidate has. References are then read one at a time
at ``os.nice(10)``, with a progress line each, because the machine is shared.

The JSON written records the date, the git commit, the thresholds, each reference's status
(``compared``, or ``missing`` with the reason) and each candidate's fingerprint summary and
best match. ``tests/test_overlap_report.py`` reads the committed report with no network: it
fails when a reference was not compared, when the suites have changed since the report was
generated, or when anything but the control overlaps.

Evaluation hashing only: nothing read here trains anything (decision D2 is untouched).

**Without --fetch** only what is already cached is compared and the rest is reported missing.
**With --fetch** the 24 TabArena-v0.1 datasets the fork never cached come through openml-python
(about 48 million cells in all) and BeyondArena's 142 parquet files through huggingface_hub,
9.74 GB at :data:`~fintfm.evaluation.overlap.BEYONDARENA_REVISION`; the opening lines print how
much of that is not yet cached. Both need the ``overlap`` extra.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fintfm.evaluation import overlap as ov

#: Where the committed report lives, relative to the repository root.
REPORT_PATH = Path("docs/results/overlap_report.json")

#: The command that regenerates the report, quoted by the report's test when it is stale.
REPORT_COMMAND = ("uv run --frozen --extra bench --extra hf --extra overlap "
                  "python -m fintfm.evaluation.overlap --report docs/results/overlap_report.json "
                  "--fetch")

#: Niceness added before the references are read (the slow part), as ``os.nice`` takes it.
NICENESS = 10


@dataclass(frozen=True)
class CandidateSpec:
    """One dataset an arm may be chosen on, and how to read it.

    Attributes:
        identity: What the report records to identify it: ``source`` (``"openml_breadth"``,
            ``"proxy_suite"`` or ``"control"``), ``name``, and ``version`` (breadth suite) or
            ``task_id`` and ``dataset_id`` (proxy suite; the control has ``dataset_id`` only).
        load: Returns the features with the target dropped, or None when the fetch fails.
    """

    identity: dict[str, Any]
    load: Callable[[], pd.DataFrame | None]


@dataclass(frozen=True)
class ReferenceSpec:
    """One TabArena-v0.1 or BeyondArena dataset, and how to find its parquet file.

    Attributes:
        label: Its label, e.g. ``"TabArena-v0.1 46918 credit-g"`` or ``"BeyondArena credit_g"``.
        suite: ``"TabArena-v0.1"`` or ``"BeyondArena"``.
        key: The OpenML dataset id (TabArena) or the ``data_foundry_uri`` (BeyondArena).
        locate: Given ``fetch``, returns ``(parquet, target)``; raises
            :class:`~fintfm.evaluation.overlap.ReferenceUnavailable` with the reason otherwise.
    """

    label: str
    suite: str
    key: int | str
    locate: Callable[[bool], tuple[Path, str]]


def candidate_specs(proxy_cache: Path) -> list[CandidateSpec]:
    """The breadth suite, the proxy suite and the control, in that order.

    Args:
        proxy_cache: The proxy runner's cache root (OpenML downloads).

    Returns:
        One :class:`CandidateSpec` per dataset, read through the suite's own loader.
    """
    from fintfm.evaluation import proxy_suite as ps
    from fintfm.experiments import openml_breadth as ob

    def breadth(name: str, version: int) -> CandidateSpec:
        """A breadth-suite dataset, read as :func:`openml_breadth.suite_features` reads it."""
        return CandidateSpec({"source": "openml_breadth", "name": name, "version": version},
                             lambda: ob.suite_features(name, version))

    def proxy(c: ps.Candidate) -> CandidateSpec:
        """A proxy-suite task's dataset, read as :func:`proxy_suite.suite_features` reads it."""
        return CandidateSpec({"source": "proxy_suite", "name": c.name, "task_id": c.task_id,
                              "dataset_id": c.dataset_id},
                             lambda: ps.suite_features(c.dataset_id, c.task_id, proxy_cache))

    control = CandidateSpec(
        {"source": "control", "name": ov.CONTROL_NAME, "dataset_id": ov.CONTROL_DATASET_ID},
        lambda: ps.suite_features(ov.CONTROL_DATASET_ID, None, proxy_cache))
    return [*(breadth(n, v) for n, v in ob.SUITE), *(proxy(c) for c in ps.SUITE), control]


def reference_specs(tabarena_cache: Path) -> list[ReferenceSpec]:
    """All 51 TabArena-v0.1 datasets, then all 142 BeyondArena datasets.

    Args:
        tabarena_cache: openml-python's dataset cache.

    Returns:
        One :class:`ReferenceSpec` per dataset.
    """
    def tabarena(dataset_id: int, name: str) -> ReferenceSpec:
        """A TabArena-v0.1 dataset in (or fetched into) the OpenML cache."""
        return ReferenceSpec(ov.tabarena_label(dataset_id, name), "TabArena-v0.1", dataset_id,
                             lambda fetch: ov.tabarena_parquet(dataset_id, tabarena_cache, fetch))

    def beyond(d: ov.BeyondArenaDataset) -> ReferenceSpec:
        """A BeyondArena dataset in (or fetched into) the Hugging Face cache."""
        return ReferenceSpec(d.label, "BeyondArena", d.uri,
                             lambda fetch: (ov.beyondarena_parquet(d, fetch), d.target))

    return [*(tabarena(i, n) for i, n in ov.TABARENA_V0_1),
            *(beyond(d) for d in ov.BEYONDARENA_DATASETS)]


def _beyondarena_uncached_bytes() -> tuple[int, int]:
    """``(files, bytes)`` of BeyondArena parquet not yet in the Hugging Face cache."""
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return len(ov.BEYONDARENA_DATASETS), sum(d.parquet_bytes for d in ov.BEYONDARENA_DATASETS)
    todo = [d for d in ov.BEYONDARENA_DATASETS
            if not isinstance(try_to_load_from_cache(ov.BEYONDARENA_REPO, d.parquet_file,
                                                     revision=ov.BEYONDARENA_REVISION,
                                                     repo_type="dataset"), str)]
    return len(todo), sum(d.parquet_bytes for d in todo)


def _git_commit() -> dict[str, Any]:
    """The checkout's commit and whether it had uncommitted changes, or Nones outside git."""
    here = Path(__file__).resolve().parent
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=here, capture_output=True,
                                text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=here, capture_output=True,
                               text=True, check=True).stdout.strip() != ""
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": dirty}


def thresholds() -> dict[str, float | int]:
    """The fingerprint and overlap thresholds in force, as the report records them."""
    return {name: getattr(ov, name) for name in (
        "SIGNIFICANT", "MIN_ROW_CELLS", "MIN_DISTINCT_NUMERIC", "MIN_DISTINCT_CATEGORIES",
        "OVERLAP_SHARE", "MIN_SHARED_COLUMNS")}


def _share(x: float) -> float | None:
    """A share for JSON: NaN (nothing to share) becomes null."""
    return None if math.isnan(x) else round(float(x), 6)


def _summary(fp: ov.ContentFingerprint) -> dict[str, Any]:
    """A candidate fingerprint's size and digests, so two reports can be compared."""
    rows = np.sort(fp.rows).astype("<u8").tobytes()
    cols = "\n".join(sorted(fp.columns)).encode()
    return {"n_rows": fp.n_rows, "n_hashed": len(fp.rows), "n_columns": len(fp.columns),
            "rows_digest": hashlib.blake2b(rows, digest_size=16).hexdigest(),
            "columns_digest": hashlib.blake2b(cols, digest_size=16).hexdigest()}


def build_report(
    proxy_cache: Path,
    tabarena_cache: Path,
    fetch: bool = False,
    only: Sequence[str] | None = None,
    print_fn: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Fingerprint every candidate and reference and compare them.

    Args:
        proxy_cache: The proxy runner's cache root.
        tabarena_cache: openml-python's dataset cache.
        fetch: Download references the caches lack (network; see the module docstring).
        only: Compare only references whose label contains one of these substrings; the
            others are recorded as ``skipped``. For smoke tests; a report made with it is
            marked incomplete and fails the report's test.
        print_fn: Where progress lines go.

    Returns:
        The report as a JSON-ready dict (see the module docstring).
    """
    specs = candidate_specs(proxy_cache)
    print_fn(f"candidates ({len(specs)}):")
    fingerprints: list[ov.ContentFingerprint | None] = []
    candidates: list[dict[str, Any]] = []
    for i, spec in enumerate(specs, 1):
        frame = spec.load()
        entry = dict(spec.identity)
        if frame is None:
            fingerprints.append(None)
            entry.update(status="not fetched", reason="the suite's loader returned nothing")
        else:
            fp = ov.content_fingerprint(frame)
            fingerprints.append(fp)
            entry.update(status="compared", **_summary(fp))
        candidates.append(entry)
        print_fn(f"  [{i:>3}/{len(specs)}] {entry['source']:<14} {entry['name']:<34} "
                 f"{entry.get('n_rows', '-'):>7} rows {entry.get('n_hashed', '-'):>7} hashed")
    hashed = [fp.rows for fp in fingerprints if fp is not None]
    keep = np.unique(np.concatenate(hashed)) if hashed else np.asarray([], dtype=np.uint64)

    n_todo, todo_bytes = _beyondarena_uncached_bytes()
    print_fn(f"\nBeyondArena: {n_todo} of {len(ov.BEYONDARENA_DATASETS)} parquet files not in the "
             f"Hugging Face cache, {todo_bytes / 1e9:.2f} GB"
             + (" to download" if fetch else " (pass --fetch to download)"))
    os.nice(NICENESS)
    refs = reference_specs(tabarena_cache)
    print_fn(f"references ({len(refs)}, at nice +{NICENESS}; {len(keep)} candidate row hashes "
             "kept):")
    reference_fps: dict[str, ov.ContentFingerprint] = {}
    references: list[dict[str, Any]] = []
    for i, ref in enumerate(refs, 1):
        entry: dict[str, Any] = {"label": ref.label, "suite": ref.suite, "key": ref.key}
        if only and not any(s in ref.label for s in only):
            references.append({**entry, "status": "skipped", "reason": "not selected by --only"})
            continue
        start = time.monotonic()
        try:
            path, target = ref.locate(fetch)
            fp = ov.parquet_fingerprint(path, target, keep)
        except (ov.ReferenceUnavailable, ValueError, OSError) as exc:
            references.append({**entry, "status": "missing", "reason": str(exc)})
            print_fn(f"  [{i:>3}/{len(refs)}] {ref.label:<58} MISSING: {exc}")
            continue
        reference_fps[ref.label] = fp
        references.append({**entry, "status": "compared", "target": target, "n_rows": fp.n_rows,
                           "n_columns": len(fp.columns), "n_kept_row_hashes": len(fp.rows)})
        print_fn(f"  [{i:>3}/{len(refs)}] {ref.label:<58} {fp.n_rows:>9} rows "
                 f"{len(fp.columns):>4} cols {len(fp.rows):>6} kept "
                 f"({time.monotonic() - start:.1f} s)")

    for entry, fp in zip(candidates, fingerprints, strict=True):
        if fp is None:
            continue
        matches = [ov.content_match(fp, r, label) for label, r in reference_fps.items()]
        best = ov.best_content_match(fp, reference_fps)
        entry.update(
            best_match=None if best is None else best.reference,
            row_share=None if best is None else _share(best.row_share),
            column_share=None if best is None else _share(best.column_share),
            shared_columns=None if best is None else best.shared_columns,
            overlaps=bool(best is not None and best.overlaps),
            overlapping_references=sorted(m.reference for m in matches if m.overlaps),
        )
    complete = (all(r["status"] == "compared" for r in references)
                and all(c["status"] == "compared" for c in candidates))
    return {
        "generated": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git": _git_commit(),
        "complete": complete,
        "command": REPORT_COMMAND,
        "thresholds": thresholds(),
        "beyondarena": {"repo": ov.BEYONDARENA_REPO, "revision": ov.BEYONDARENA_REVISION},
        "references": references,
        "candidates": candidates,
    }


def summarise(report: dict[str, Any], print_fn: Callable[[str], None] = print) -> int:
    """Print the report's verdict.

    Args:
        report: From :func:`build_report`.
        print_fn: Where lines go.

    Returns:
        0 when the report is complete, the control overlaps
        :data:`~fintfm.evaluation.overlap.CONTROL_REFERENCE` and nothing else overlaps; else 1.
    """
    refs, cands = report["references"], report["candidates"]
    compared = sum(r["status"] == "compared" for r in refs)
    print_fn(f"\n{compared} of {len(refs)} references compared; "
             f"{sum(c['status'] == 'compared' for c in cands)} of {len(cands)} candidates read")
    control = next(c for c in cands if c["source"] == "control")
    control_ok = ov.CONTROL_REFERENCE in control.get("overlapping_references", [])
    print_fn(f"control {ov.CONTROL_NAME}: "
             + ("overlaps " + ov.CONTROL_REFERENCE if control_ok else "MISSED"))
    hits = [c for c in cands if c["source"] != "control" and c.get("overlaps")]
    for c in hits:
        print_fn(f"OVERLAP {c['source']} {c['name']}: {', '.join(c['overlapping_references'])}")
    print_fn(f"{len(hits)} suite datasets overlap a reference")
    return 0 if report["complete"] and control_ok and not hits else 1


def main(argv: Sequence[str] | None = None) -> int:
    """Command line: ``--report <path.json> [--fetch]``.

    Args:
        argv: Command-line arguments; ``sys.argv[1:]`` when omitted.

    Returns:
        Process exit status, from :func:`summarise`.
    """
    import json

    from fintfm.evaluation.proxy_suite import PROXY_CACHE

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--report", type=Path, default=REPORT_PATH,
                   help=f"JSON file to write (default {REPORT_PATH})")
    p.add_argument("--fetch", action="store_true",
                   help="download the references the caches lack (network, about 10 GB)")
    p.add_argument("--only", default=None,
                   help="comma-separated label substrings: compare only those references "
                        "(smoke tests; the report is then marked incomplete)")
    p.add_argument("--cache-dir", type=Path, default=PROXY_CACHE,
                   help="proxy runner cache (OpenML downloads of the proxy suite)")
    p.add_argument("--tabarena-cache", type=Path, default=ov.TABARENA_OPENML_CACHE,
                   help="openml-python dataset cache holding TabArena-v0.1's datasets")
    args = p.parse_args(argv)
    only = [s.strip() for s in args.only.split(",") if s.strip()] if args.only else None
    report = build_report(args.cache_dir, args.tabarena_cache, fetch=args.fetch, only=only,
                          print_fn=lambda s: print(s, flush=True))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {args.report}")
    return summarise(report)
