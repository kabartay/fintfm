"""DR4.1's Verify: no decision-suite dataset shares a row hash with TabArena-v0.1 or BeyondArena.

Two parts, neither touching a network:

* :func:`~fintfm.evaluation.overlap.parquet_fingerprint`, which reads every reference, must give
  exactly :func:`~fintfm.evaluation.overlap.content_fingerprint`'s result, or the report's
  references and candidates would not be hashed alike;
* the committed report (``docs/results/overlap_report.json``, written by
  ``python -m fintfm.evaluation.overlap --report``) must have compared every one of the 193
  references, must cover exactly the suites as they are now, and must show the positive control
  overlapping and nothing else.

Until the report has been generated the report tests are expected failures (strict, so a report
that appears turns them into ordinary tests at once); they are never skipped.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from fintfm.evaluation import overlap as ov
from fintfm.evaluation import proxy_suite as ps
from fintfm.evaluation.overlap_report import REPORT_COMMAND, thresholds
from fintfm.experiments import openml_breadth as ob

REPORT = Path(__file__).resolve().parents[1] / "docs" / "results" / "overlap_report.json"

#: Marks a report test as an expected failure while the report file does not exist.
needs_report = pytest.mark.xfail(not REPORT.exists(), reason="report not generated yet",
                                 strict=True)


def _mixed_frame(n: int = 500, seed: int = 0) -> pd.DataFrame:
    """Every column kind the fingerprint distinguishes, with missing cells and a target."""
    rng = np.random.default_rng(seed)
    with_nan = rng.normal(size=n)
    with_nan[rng.random(n) < 0.1] = np.nan
    ints = pd.array(rng.integers(0, 40, n), dtype="Int64")
    ints[rng.random(n) < 0.05] = pd.NA
    text = rng.choice(["red", "green", "blue", "grey"], size=n).astype(object)
    text[rng.random(n) < 0.1] = None
    flags = rng.choice([True, False], size=n).astype(object)
    flags[rng.random(n) < 0.1] = None
    return pd.DataFrame({
        "x": rng.normal(size=n),
        "amount": rng.uniform(size=n).astype(np.float32) * 1000,
        "x_nan": with_nan,
        "count": ints,
        "num_str": [f"{v:.3f}" for v in rng.uniform(size=n)],  # numeric strings
        "level": pd.Categorical(rng.choice(list("abcde"), size=n)),
        "text": text,  # a category column with missing cells
        "flag": flags,  # bool with missing cells, read back as objects
        "binary": rng.integers(0, 2, n),  # too few values for a column fingerprint
        "empty": np.full(n, np.nan),
        "label": rng.integers(0, 2, n),  # the target
    }, index=pd.Index(np.arange(n) * 3 + 7, name="row_id"))  # stored as a column, not a feature


def _same(a: ov.ContentFingerprint, b: ov.ContentFingerprint) -> bool:
    """Whether two fingerprints agree in every field."""
    return a.n_rows == b.n_rows and np.array_equal(a.rows, b.rows) and a.columns == b.columns


@pytest.mark.parametrize("batch_cells", [ov.BATCH_CELLS, 700])
def test_parquet_fingerprint_equals_content_fingerprint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, batch_cells: int
):
    """The streaming reference path and the in-memory path give one result, in one row group
    and in several small row batches across several row groups."""
    pytest.importorskip("pyarrow")
    monkeypatch.setattr(ov, "BATCH_CELLS", batch_cells)
    path = tmp_path / "mixed.parquet"
    _mixed_frame().to_parquet(path, row_group_size=180)
    expected = ov.content_fingerprint(pd.read_parquet(path).drop(columns=["label"]))
    got = ov.parquet_fingerprint(path, "label")
    assert len(expected.rows) > 400 and len(expected.columns) >= 5  # the frame exercises both
    assert _same(got, expected)
    assert _same(got, ov.content_fingerprint(_mixed_frame().drop(columns=["label"])))


def test_parquet_fingerprint_keeps_only_the_rows_asked_for(tmp_path: Path):
    """``keep_rows`` trims the row hashes to the candidates' and leaves every share unchanged."""
    pytest.importorskip("pyarrow")
    path = tmp_path / "mixed.parquet"
    _mixed_frame(800).to_parquet(path)
    full = ov.parquet_fingerprint(path, "label")
    candidate = ov.content_fingerprint(_mixed_frame(800).iloc[:200].drop(columns=["label"]))
    stranger = ov.content_fingerprint(_mixed_frame(800, seed=5).drop(columns=["label"]))
    keep = np.concatenate([candidate.rows, stranger.rows])
    kept = ov.parquet_fingerprint(path, "label", keep_rows=keep)
    assert kept.n_rows == full.n_rows == 800 and kept.columns == full.columns
    assert set(kept.rows) == set(full.rows) & set(keep)
    for fp in (candidate, stranger):
        assert ov.content_match(fp, kept, "r") == ov.content_match(fp, full, "r")
    assert ov.content_match(candidate, kept, "r").overlaps


def test_parquet_fingerprint_refuses_a_missing_target(tmp_path: Path):
    """A target the file lacks is an error, never silently hashed as a feature."""
    pytest.importorskip("pyarrow")
    path = tmp_path / "mixed.parquet"
    _mixed_frame(50).to_parquet(path)
    with pytest.raises(ValueError, match="no target column"):
        ov.parquet_fingerprint(path, "not_a_column")


def test_beyondarena_references_are_complete():
    """142 datasets, one parquet each under its container, matching the name list."""
    data = ov.BEYONDARENA_DATASETS
    assert len(data) == 142 and len({d.uri for d in data}) == 142
    assert ov.BEYONDARENA == tuple(sorted(d.name for d in data))
    assert all(d.parquet_file == f"{d.uri}/dataset.parquet" and d.target for d in data)
    assert all(d.uri.count("/") in (1, 2) and d.parquet_bytes > 0 for d in data)
    assert round(sum(d.parquet_bytes for d in data) / 1e9, 2) == 9.74


def _report() -> dict[str, Any]:
    """The committed report; raises FileNotFoundError until it has been generated."""
    return json.loads(REPORT.read_text(encoding="utf-8"))


def _identity(c: dict[str, Any]) -> tuple[Any, ...]:
    """A candidate's identity in the report, as a comparable tuple."""
    if c["source"] == "openml_breadth":
        return ("openml_breadth", c["name"], c["version"])
    if c["source"] == "proxy_suite":
        return ("proxy_suite", c["name"], c["task_id"], c["dataset_id"])
    return (c["source"], c["name"], c["dataset_id"])


def _expected_candidates() -> set[tuple[Any, ...]]:
    """The suites as they are now, plus the control."""
    return ({("openml_breadth", n, v) for n, v in ob.SUITE}
            | {("proxy_suite", c.name, c.task_id, c.dataset_id) for c in ps.SUITE}
            | {("control", ov.CONTROL_NAME, ov.CONTROL_DATASET_ID)})


@needs_report
def test_report_compared_every_reference():
    """All 51 TabArena-v0.1 and all 142 BeyondArena datasets were read and compared."""
    report = _report()
    expected = ({ov.tabarena_label(i, n) for i, n in ov.TABARENA_V0_1}
                | {d.label for d in ov.BEYONDARENA_DATASETS})
    status = {r["label"]: r for r in report["references"]}
    assert set(status) == expected and len(expected) == 193
    not_compared = {k: f"{v['status']}: {v.get('reason')}" for k, v in status.items()
                    if v["status"] != "compared"}
    assert not_compared == {}, f"re-run with --fetch: {REPORT_COMMAND}"
    assert report["complete"] is True
    assert report["thresholds"] == thresholds(), f"thresholds changed; re-run: {REPORT_COMMAND}"
    assert report["beyondarena"]["revision"] == ov.BEYONDARENA_REVISION


@needs_report
def test_report_covers_exactly_the_current_suites():
    """A suite dataset added or removed since the report was made fails here, by name."""
    have = {_identity(c) for c in _report()["candidates"]}
    want = _expected_candidates()
    assert have == want, (
        f"the overlap report is stale: not in it {sorted(want - have)}, no longer in the "
        f"suites {sorted(have - want)}. Regenerate it: {REPORT_COMMAND}")


@needs_report
def test_report_finds_only_the_control():
    """Every candidate was read; the control overlaps TabArena's credit-g; nothing else does."""
    candidates = _report()["candidates"]
    assert [c["name"] for c in candidates if c["status"] != "compared"] == []
    control = [c for c in candidates if c["source"] == "control"]
    assert len(control) == 1 and control[0]["overlaps"]
    assert ov.CONTROL_REFERENCE in control[0]["overlapping_references"]
    hits = {c["name"]: c["overlapping_references"] for c in candidates
            if c["source"] != "control" and c["overlaps"]}
    assert hits == {}, f"suite datasets overlap TabArena-v0.1 or BeyondArena by contents: {hits}"
