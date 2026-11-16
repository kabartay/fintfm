"""``scripts/lite_same_fit.py`` keeps the field once and gives every entry its own method identity.

Only the frame-building step is tested: the fit itself needs the TabArena fork's environment.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "lite_same_fit.py"


def _load():
    """Import the script as a module."""
    spec = importlib.util.spec_from_file_location("lite_same_fit", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _entry(eval_dir: Path, run: str, error: float) -> None:
    """Write a two-row ``results_per_split.csv``: one field method and one ``[New]`` entry."""
    rows = [
        {
            "dataset": "d",
            "fold": 0,
            "method": "RF (default)",
            "metric_error": 0.3,
            "config_type": "RF",
            "ta_name": "RF",
            "ta_suite": "RF",
            "method_subtype": "default",
        },
        {
            "dataset": "d",
            "fold": 0,
            "method": "[New] TA-X (default)",
            "metric_error": error,
            "config_type": "[New] TA-X",
            "ta_name": "[New] X",
            "ta_suite": "[New] X",
            "method_subtype": "default",
        },
    ]
    (eval_dir / run).mkdir(parents=True)
    pd.DataFrame(rows).to_csv(eval_dir / run / "results_per_split.csv", index=False)


def test_field_once_and_entries_relabelled(tmp_path: Path) -> None:
    """The field comes from the first entry only; each entry gets ``[New] E<i>`` in every identity column."""
    mod = _load()
    _entry(tmp_path, "a", 0.2)
    _entry(tmp_path, "b", 0.1)
    frame = mod.combined_frame(["a", "b"], tmp_path)
    assert (frame["method"] == "RF (default)").sum() == 1
    new = frame[frame["method"].str.startswith("[New]")].set_index("method")
    assert list(new.index) == ["[New] E0 (default)", "[New] E1 (default)"]
    assert new.loc["[New] E1 (default)", "metric_error"] == 0.1
    for col in ("config_type", "ta_name", "ta_suite"):
        assert list(new[col]) == ["[New] E0", "[New] E1"]


def test_entry_without_new_rows_is_refused(tmp_path: Path) -> None:
    """An entry with no ``[New]`` rows is an error, not a silently empty arm."""
    mod = _load()
    (tmp_path / "x").mkdir()
    pd.DataFrame(
        [{"dataset": "d", "fold": 0, "method": "RF (default)", "metric_error": 0.3}]
    ).to_csv(tmp_path / "x" / "results_per_split.csv", index=False)
    with pytest.raises(ValueError, match="no \\[New\\] rows"):
        mod.combined_frame(["x"], tmp_path)
