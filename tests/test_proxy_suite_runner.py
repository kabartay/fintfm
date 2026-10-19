"""The DR4.2 proxy-suite runner: identical splits, cached baselines, Elo, sign test, row hash.

No network: two tiny synthetic tasks are monkeypatched in place of OpenML (``fetch_task``),
and the row-hash tests build their references in a temporary directory.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.stats import binomtest

from fintfm.evaluation import proxy_suite as ps
from fintfm.evaluation.overlap import (
    SelectionForbiddenError,
    content_fingerprint,
    content_match,
    load_reference_fingerprints,
)
from fintfm.modeling.model import FinancialTFM, ModelConfig

BINARY = ps.Candidate("CC18", 900001, 900001, "synthetic-binary", "binary", 120, 4, 2, "")
REGRESSION = ps.Candidate("CTR23", 900002, 900002, "synthetic-regression", "regression", 120,
                          3, 0, "")


def _splits(n: int, folds: int = 3) -> list[tuple[int, int, np.ndarray, np.ndarray]]:
    """Three folds of a 10-fold split, as OpenML's split file would give them."""
    order = np.random.default_rng(1).permutation(n)
    parts = np.array_split(order, 10)
    out = []
    for f in range(folds):
        test = np.sort(parts[f])
        out.append((0, f, np.setdiff1d(np.arange(n), test), test))
    return out


def _fake_fetch(candidate: ps.Candidate, cache_dir: Path, n_splits: int) -> ps.ProxyTask:
    """A synthetic stand-in for :func:`ps.fetch_task`: a numeric and a categorical column."""
    rng = np.random.default_rng(candidate.task_id)
    n = candidate.n_rows
    X = pd.DataFrame({
        "a": rng.normal(size=n), "b": rng.normal(size=n),
        "c": rng.choice(["x", "y", "z"], size=n).astype(object),
    })
    X.loc[::17, "b"] = np.nan
    signal = X["a"].to_numpy() + (X["c"] == "x").to_numpy()
    if candidate.kind == "regression":
        y = signal + 0.1 * rng.normal(size=n)
    else:
        y = (signal + 0.5 * rng.normal(size=n) > 0.3).astype(np.int64)
    return ps.ProxyTask(candidate, ps._prepare(X), y, _splits(n)[:n_splits], "synthetic")


@pytest.fixture
def tiny_checkpoint(tmp_path: Path) -> str:
    """An untrained four-class checkpoint; the runner's plumbing, not its scores, is tested."""
    torch.manual_seed(0)
    model = FinancialTFM(ModelConfig(max_features=6, max_classes=4, d_model=16, n_heads=2,
                                     n_layers=1, d_ff=32))
    path = tmp_path / "tiny.pt"
    model.save(str(path), trained_objectives=("classification",))
    return str(path)


@pytest.fixture
def synthetic_suite(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace OpenML with the two synthetic tasks."""
    monkeypatch.setattr(ps, "SUITE", (BINARY, REGRESSION))
    monkeypatch.setattr(ps, "fetch_task", _fake_fetch)


def test_parse_splits_arff():
    """OpenML's split file: train and test rows per (repeat, fold), quoted or not."""
    text = """@relation x_splits
@attribute type {TRAIN,TEST}
@attribute rowid numeric
@attribute repeat numeric
@attribute fold numeric
@data
TRAIN,2,0,0
TEST,0,0,0
TRAIN,1,0,0
'TEST',2,0,1
TRAIN,0,0,1
TRAIN,1,0,1
"""
    splits = ps.parse_splits_arff(text)
    assert set(splits) == {(0, 0), (0, 1)}
    np.testing.assert_array_equal(splits[(0, 0)][0], [1, 2])
    np.testing.assert_array_equal(splits[(0, 0)][1], [0])
    np.testing.assert_array_equal(splits[(0, 1)][1], [2])


def test_run_scores_every_method_on_identical_splits(
    synthetic_suite: None, tiny_checkpoint: str, tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    """``run`` writes one row per task, split and method, every method on the same rows."""
    out = tmp_path / "run.csv"
    code = ps.main(["--cache-dir", str(tmp_path / "cache"), "run", "--model", tiny_checkpoint,
                    "--out", str(out), "--name", "FinTFM:tiny", "--n-boot", "20"])
    assert code == 0
    frame = pd.read_csv(out)
    assert list(frame.columns) == list(ps.RESULT_COLUMNS)
    methods = {*ps.available_baselines(), "FinTFM:tiny"}
    assert set(frame["method"]) == methods
    assert len(frame) == 2 * ps.N_SPLITS * len(methods)
    assert (frame["status"] == "ok").all(), frame.loc[frame["status"] != "ok", "status"]
    assert np.isfinite(frame["error"]).all()
    # identical splits: one (n_train, n_test, fold) per (task, split), whatever the method
    per_split = frame.groupby(["task_id", "split"])[["n_train", "n_test", "fold"]].nunique()
    assert (per_split == 1).all().all()
    assert set(frame.loc[frame["kind"] == "binary", "metric"]) == {"1-roc_auc"}
    assert set(frame.loc[frame["kind"] == "regression", "metric"]) == {"rmse"}
    meta = out.with_suffix(".json").read_text()
    assert "FinTFM:tiny" in meta and "synthetic" in meta
    assert "Elo (Bradley-Terry" in capsys.readouterr().out


def test_baselines_are_cached_and_reused(
    synthetic_suite: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """A second run reads the baselines from the cache instead of refitting them."""
    cache = tmp_path / "cache"
    first = ps.run_suite(None, tmp_path / "a.csv", cache_dir=cache, log=lambda _: None)

    def refit(*args: object, **kwargs: object) -> None:
        """Fail if a baseline is built again."""
        raise AssertionError("baseline refitted despite the cache")

    monkeypatch.setattr(ps, "_baseline", refit)
    second = ps.run_suite(None, tmp_path / "b.csv", cache_dir=cache, log=lambda _: None)
    pd.testing.assert_frame_equal(first.reset_index(drop=True), second.reset_index(drop=True))


def test_run_refuses_a_selection_forbidden_task(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """DR4.1: the runner is an arm-selection tool, so a bankruptcy panel needs the flag."""
    polish = ps.Candidate("CC18", 900003, 900003, "polish_companies_bankruptcy", "binary", 120,
                          4, 2, "")
    monkeypatch.setattr(ps, "fetch_task", _fake_fetch)
    with pytest.raises(SelectionForbiddenError):
        ps.run_suite(None, tmp_path / "x.csv", tasks=[polish], cache_dir=tmp_path,
                     log=lambda _: None)
    frame = ps.run_suite(None, tmp_path / "x.csv", tasks=[polish], cache_dir=tmp_path,
                         allow_reporting=True, log=lambda _: None)
    assert len(frame)
    meta = json.loads((tmp_path / "x.json").read_text())
    assert meta["reporting_only"] == ["polish_companies_bankruptcy"]


def test_tasks_outside_the_suite_are_refused():
    """``--tasks`` cannot score an excluded task under the proxy's name."""
    excluded = next(c for c in ps.CANDIDATES if c not in ps.SUITE)
    with pytest.raises(SystemExit):
        ps._select_tasks(str(excluded.task_id))


def _results(rows: list[tuple[int, int, str, float]]) -> pd.DataFrame:
    """A results frame from (task, split, method, error) tuples."""
    frame = pd.DataFrame(rows, columns=["task_id", "split", "method", "error"])
    for col in ps.RESULT_COLUMNS:
        if col not in frame:
            frame[col] = 0
    return frame[list(ps.RESULT_COLUMNS)]


def test_elo_orders_a_dominance_chain_and_anchors_rf():
    """A beats RF beats B on every split: A > 1000 = RF > B, intervals around the point."""
    rows = [(t, s, m, e) for t in range(6) for s in range(3)
            for m, e in (("A", 0.1), ("RF", 0.2), ("B", 0.3))]
    table = ps.elo_table(_results(rows), n_boot=50).set_index("method")
    assert table.loc["RF", "elo"] == pytest.approx(1000.0)
    assert table.loc["A", "elo"] > 1000.0 > table.loc["B", "elo"]
    assert (table["lo"] <= table["elo"] + 1e-6).all() and (table["elo"] <= table["hi"] + 1e-6).all()
    assert table.loc["A", "battles"] == 6 * 3 * 2


def test_elo_rates_an_even_record_level_and_ties_as_halves():
    """Equal errors are half a win each, and a 50/50 record rates equal."""
    rows = [(t, 0, "RF", 0.2) for t in range(4)]
    rows += [(t, 0, "X", 0.2) for t in range(2)] + [(t, 0, "X", e) for t, e in ((2, 0.1), (3, 0.3))]
    table = ps.elo_table(_results(rows), n_boot=0).set_index("method")
    assert table.loc["X", "elo"] == pytest.approx(1000.0, abs=1e-6)


def test_bradley_terry_recovers_known_strengths():
    """Expected win counts from strengths 1, 2, 4 give back log-strengths 0, log 2, log 4."""
    p = np.array([1.0, 2.0, 4.0])
    W = 1000 * p[:, None] / (p[:, None] + p[None, :])
    np.fill_diagonal(W, 0)
    logp = ps.bradley_terry(W, prior=0.0)
    np.testing.assert_allclose(logp - logp[0], np.log(p), atol=1e-6)


def test_sign_test_counts_and_p_value():
    """Per task: mean error over splits; wins, losses, ties and an exact binomial p."""
    rows = []
    for t in range(10):
        cand = 0.1 if t < 7 else (0.3 if t < 9 else 0.2)
        rows += [(t, s, "cand", cand) for s in range(3)] + [(t, s, "ref", 0.2) for s in range(3)]
    test = ps.sign_test(_results(rows), "cand", "ref")
    assert (test.wins, test.losses, test.ties) == (7, 2, 1)
    assert test.p_value == pytest.approx(binomtest(7, 9, 0.5).pvalue)
    by_split = ps.sign_test(_results(rows), "cand", "ref", unit="split")
    assert (by_split.wins, by_split.losses, by_split.ties) == (21, 6, 3)


def test_compare_disambiguates_and_tests_a_against_b(
    synthetic_suite: None, tiny_checkpoint: str, tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    """``compare A B`` rates both checkpoints with the baselines and sign-tests A against B."""
    cache = str(tmp_path / "cache")
    for stem in ("a", "b"):
        ps.main(["--cache-dir", cache, "run", "--model", tiny_checkpoint, "--out",
                 str(tmp_path / f"{stem}.csv"), "--n-boot", "0"])
    capsys.readouterr()
    assert ps.main(["compare", str(tmp_path / "a.csv"), str(tmp_path / "b.csv"),
                    "--n-boot", "20"]) == 0
    out = capsys.readouterr().out
    assert "FinTFM:tiny [a]" in out and "FinTFM:tiny [b]" in out
    # the same checkpoint, settings and seed: every unit is a tie
    assert "0 wins, 0 losses, 2 ties" in out


def test_compare_refuses_runs_on_different_splits(tmp_path: Path):
    """Baselines that disagree on a shared split mean the runs did not share splits."""
    a = _results([(1, 0, "RF", 0.2), (1, 0, "FinTFM:x", 0.1)])
    b = _results([(1, 0, "RF", 0.25), (1, 0, "FinTFM:y", 0.1)])
    a.to_csv(tmp_path / "a.csv", index=False)
    b.to_csv(tmp_path / "b.csv", index=False)
    with pytest.raises(ValueError, match="identical splits"):
        ps.compare_runs(tmp_path / "a.csv", tmp_path / "b.csv", n_boot=0)


def _frame(n: int = 300, seed: int = 0) -> pd.DataFrame:
    """A mixed table: three numeric columns and a four-level category."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "x": rng.normal(size=n), "y": rng.uniform(size=n) * 1000, "z": rng.integers(0, 50, n),
        "k": rng.choice(list("abcd"), size=n),
    })


def test_row_hash_sees_through_a_reupload():
    """Shuffled rows, renamed and reordered columns, float32 cells and relabelled categories
    (TabArena's re-upload changes) leave the fingerprint matching."""
    original = _frame()
    reupload = original.sample(frac=1.0, random_state=3).rename(
        columns={"x": "feature_x", "y": "amount", "z": "count", "k": "kind"})
    reupload = reupload[["kind", "count", "feature_x", "amount"]]
    reupload["feature_x"] = reupload["feature_x"].astype(np.float32)
    reupload["kind"] = reupload["kind"].map({"a": "alpha", "b": "beta", "c": "gamma", "d": "delta"})
    match = content_match(content_fingerprint(original), content_fingerprint(reupload), "ref")
    assert match.row_share == 1.0 and match.column_share == 1.0 and match.overlaps


def test_row_hash_finds_a_subset_and_passes_a_stranger():
    """Half a dataset's rows inside a reference overlap; an unrelated table does not."""
    ref = content_fingerprint(_frame(600))
    half = content_match(content_fingerprint(_frame(600).iloc[:300]), ref, "ref")
    assert half.row_share == 1.0 and half.overlaps
    stranger = content_match(content_fingerprint(_frame(600, seed=9)), ref, "ref")
    assert stranger.row_share == 0.0 and not stranger.overlaps


def test_reference_cache_absent_is_a_clear_skip(tmp_path: Path):
    """No local TabArena copy: nothing compared, and the note says so."""
    refs, notes = load_reference_fingerprints(tmp_path / "no-such-cache")
    assert refs == {}
    assert any("row-hash check skipped" in n for n in notes)
    assert any("BeyondArena" in n for n in notes)


def test_reference_cache_reads_parquet_and_drops_the_target(tmp_path: Path):
    """openml-python's layout: ``<id>/dataset_<id>.pq`` and the target from description.xml."""
    pytest.importorskip("pyarrow")
    frame = _frame()
    frame["label"] = np.arange(len(frame)) % 2
    d = tmp_path / "46918"
    d.mkdir()
    frame.to_parquet(d / "dataset_46918.pq")
    (d / "description.xml").write_text(
        '<oml:data_set_description xmlns:oml="http://openml.org/openml">'
        "<oml:default_target_attribute>label</oml:default_target_attribute>"
        "</oml:data_set_description>")
    refs, notes = load_reference_fingerprints(tmp_path)
    assert list(refs) == ["TabArena-v0.1 46918 credit-g"]
    assert any("1 of 51 datasets read" in n for n in notes)
    match = content_match(content_fingerprint(_frame()), refs["TabArena-v0.1 46918 credit-g"], "r")
    assert match.row_share == 1.0
