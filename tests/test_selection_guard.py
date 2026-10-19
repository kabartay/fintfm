"""DR4.1: no arm-selection tool reads the Polish or Taiwanese bankruptcy panels unless told to.

Both are TabArena-v0.1 datasets, so an arm chosen on them biases the TabArena number this
project reports. `fintfm.evaluation.overlap.SELECTION_FORBIDDEN` is the one list; each tool
calls `guard_selection` and accepts `--allow-tabarena-for-reporting` for reporting only.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from fintfm.evaluation import datasets
from fintfm.evaluation.overlap import (
    BEYONDARENA,
    SELECTION_FORBIDDEN,
    TABARENA_V0_1,
    SelectionForbiddenError,
    guard_selection,
    selection_forbidden,
)
from fintfm.experiments import openml_breadth

ROOT = Path(__file__).resolve().parents[1]


def _script(rel: str) -> ModuleType:
    """Import a script under ``scripts/`` by path."""
    path = ROOT / rel
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # a script's dataclasses look their module up by name
    spec.loader.exec_module(mod)
    return mod


def test_the_forbidden_panels_are_tabarena_datasets():
    """The list names TabArena's own entries, so it cannot drift from the benchmark."""
    tabarena = {n for _, n in TABARENA_V0_1}
    assert {"polish_companies_bankruptcy", "taiwanese_bankruptcy_prediction"} <= set(
        SELECTION_FORBIDDEN) & tabarena
    assert {"polish_companies_bankruptcy", "taiwanese_bankruptcy_prediction"} <= set(BEYONDARENA)


@pytest.mark.parametrize("name", [
    "polish_companies_bankruptcy", "taiwanese_bankruptcy_prediction", "polish-bankruptcy-3y",
    "polish_bankruptcy.zip", "taiwan-bankruptcy", "taiwan_bankruptcy.zip",
])
def test_every_spelling_of_the_panels_is_forbidden(name: str):
    """TabArena's names, the loaders' names (``datasets.load_*``) and the cache archives."""
    assert selection_forbidden(name)


@pytest.mark.parametrize("name", ["v4finbench", "mammography", "credit-g", "home-credit"])
def test_other_datasets_pass(name: str):
    """Only the two panels are refused here; other overlaps are the suites' filters' job."""
    assert selection_forbidden(name) is None


def test_the_loaders_names_are_caught():
    """The names the loaders actually record, read from their source rather than retyped."""
    src = (ROOT / "src/fintfm/evaluation/datasets.py").read_text()
    assert 'name=f"polish-bankruptcy-{horizon_years}y"' in src
    assert 'name="taiwan-bankruptcy"' in src
    assert selection_forbidden("polish-bankruptcy-1y") and selection_forbidden("taiwan-bankruptcy")


def test_guard_refuses_then_flags_for_reporting():
    """Refused by default; with the override, returned so the caller labels its output."""
    with pytest.raises(SelectionForbiddenError, match="allow-tabarena-for-reporting"):
        guard_selection(["v4finbench", "taiwan-bankruptcy"], False, "tool")
    assert guard_selection(["v4finbench", "taiwan-bankruptcy"], True, "tool") == [
        "taiwan-bankruptcy"]
    assert guard_selection(["v4finbench"], False, "tool") == []


def _arm(root: Path, arm: str, dataset: str) -> None:
    """One pilot arm's score directory with fold-0 predictions and its run record."""
    d = root / f"{arm}.score"
    d.mkdir(parents=True)
    rng = np.random.default_rng(len(arm))
    y = np.r_[np.zeros(40), np.ones(10)].astype(int)
    np.savez(d / "predictions_fold0.npz", y_val=y, val_fintfm=rng.uniform(size=50),
             val_logistic_regression=rng.uniform(size=50), y_true=y,
             pred_fintfm=rng.uniform(size=50), pred_logistic_regression=rng.uniform(size=50))
    (d / "v4_protocol.json").write_text(json.dumps({"dataset": dataset}))


def test_pilot_compare_refuses_a_bankruptcy_run(tmp_path: Path):
    """An arm scored on the Polish panel stops the comparison unless the flag is given."""
    pilot = _script("scripts/candidates/pilot_compare.py")
    _arm(tmp_path, "control", "polish-bankruptcy-3y")
    _arm(tmp_path, "arm", "polish-bankruptcy-3y")
    args = ["--root", str(tmp_path), "--n-boot", "20"]
    with pytest.raises(SystemExit, match="DR4.1"):
        pilot.main(args)
    assert not (tmp_path / "comparison_val.json").exists()
    pilot.main([*args, "--allow-tabarena-for-reporting"])
    rows = json.loads((tmp_path / "comparison_val.json").read_text())
    assert rows and all(r["reporting_only"] == ["polish-bankruptcy-3y"] for r in rows)


def test_pilot_compare_runs_unchanged_on_v4finbench(tmp_path: Path):
    """The V4FinBench pilot it was written for needs no flag and gets no label."""
    pilot = _script("scripts/candidates/pilot_compare.py")
    _arm(tmp_path, "control", "v4finbench")
    _arm(tmp_path, "arm", "v4finbench")
    pilot.main(["--root", str(tmp_path), "--n-boot", "20"])
    rows = json.loads((tmp_path / "comparison_val.json").read_text())
    assert {r["arm"] for r in rows} == {"control", "arm"}
    assert all("reporting_only" not in r for r in rows)


def test_label_symmetry_probe_skips_the_panels_without_the_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
):
    """Both archives cached: neither is loaded by default, both are with the flag."""
    probe = _script("scripts/label_symmetry_probe.py")
    for archive in ("polish_bankruptcy.zip", "taiwan_bankruptcy.zip"):
        (tmp_path / archive).write_bytes(b"")
    monkeypatch.setattr(datasets, "CACHE_DIR", datasets.CACHE_DIR)  # restored after the test

    def fake(name: str) -> Callable[..., SimpleNamespace]:
        """A loader stand-in returning a two-row panel."""
        return lambda *a, **k: SimpleNamespace(name=name, X=np.zeros((2, 1)), y=np.array([0, 1]))

    monkeypatch.setattr(datasets, "load_polish_bankruptcy", fake("polish-bankruptcy-3y"))
    monkeypatch.setattr(datasets, "load_taiwan_bankruptcy", fake("taiwan-bankruptcy"))
    assert probe._cached_panels(tmp_path) == []
    out = capsys.readouterr().out
    assert out.count("allow-tabarena-for-reporting") == 2
    names = [n for n, _, _ in probe._cached_panels(tmp_path, allow_reporting=True)]
    assert names == ["polish-bankruptcy-3y", "taiwan-bankruptcy"]


def test_openml_breadth_refuses_a_planted_panel():
    """The breadth suite chooses arms too; a planted panel stops it before any fetch."""
    planted = [("taiwanese_bankruptcy_prediction", 1)]
    with pytest.raises(SelectionForbiddenError):
        openml_breadth.run("unused.pt", suite=planted)


def test_openml_breadth_suite_itself_passes_the_guard():
    """The shipped SUITE holds neither panel."""
    assert guard_selection([n for n, _ in openml_breadth.SUITE], False, "openml_breadth") == []
