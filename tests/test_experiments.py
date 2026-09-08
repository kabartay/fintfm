"""Guards for the Phase 1 ablation harness.

These run at a size that finishes in seconds. The point is that the harness is correct —
that compute really is matched and results really are re-derivable — not that a two-step
model learns anything.
"""

import json
from pathlib import Path

from fintfm.experiments.prior_ablation import VARIANTS, run_ablation, summarise
from fintfm.modeling.model import ModelConfig


def test_variants_isolate_exactly_one_variable():
    assert VARIANTS["financial"] == 1.0
    assert VARIANTS["generic"] == 0.0
    assert 0.0 < VARIANTS["mixed"] < 1.0


def test_ablation_matches_compute_and_writes_rederivable_results(tmp_path: Path):
    cfg = ModelConfig(max_features=8, max_classes=2, d_cell=8, d_model=16, n_heads=2,
                      n_col_layers=1, n_layers=1, d_ff=16)
    record = run_ablation(
        out_dir=tmp_path, steps=2, batch_size=2, n_rows=16, model_cfg=cfg,
        variants={"financial": 1.0, "generic": 0.0},
        horizons=(),  # training-harness test; real-data scoring is exercised separately
    )
    assert set(record["variants"]) == {"financial", "generic"}
    sizes = {v["n_parameters"] for v in record["variants"].values()}
    assert len(sizes) == 1, "compute must be matched across variants"

    written = json.loads((tmp_path / "results.json").read_text())
    assert written["config"]["steps"] == 2
    assert written["git_commit"]  # provenance is recorded, not just the numbers
    assert "created" in written


def test_summarise_reports_the_exit_condition():
    record = {
        "config": {"steps": 10},
        "variants": {
            "financial": {"n_parameters": 100, "metrics": {"d/balanced": {"roc_auc": 0.7, "ece": 0.01}}},
            "generic": {"n_parameters": 100, "metrics": {"d/balanced": {"roc_auc": 0.6, "ece": 0.02}}},
        },
    }
    text = summarise(record)
    assert "EXIT CONDITION" in text
    assert "+0.1000" in text  # financial ahead by 0.10 AUC


def test_summarise_says_so_rather_than_printing_nan_when_nothing_scored():
    """A skipped evaluation must read as "not evaluated", never as a nan verdict."""
    record = {
        "config": {"steps": 3},
        "variants": {
            "financial": {"n_parameters": 10, "metrics": {}},
            "generic": {"n_parameters": 10, "metrics": {}},
        },
    }
    text = summarise(record)
    assert "cannot be evaluated" in text
    # a bare "nan" substring also matches "fiNANcial"; check for a nan *value* instead
    assert "+nan" not in text
    assert "nan vs" not in text
    assert "delta" not in text
