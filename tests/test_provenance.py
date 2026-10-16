"""Provenance records (design review DR1.2): what they hold, and that they hold it truly.

The record is only worth having if it is right, so the realised source shares are checked
against the draw they describe, not only against arithmetic done by hand.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from fintfm.prior.mixture import PriorConfig, _draw_source, source_shares
from fintfm.provenance import estimator_params, git_state, json_safe, sha256_file


def test_realised_shares_match_the_sequential_draw():
    """``source_shares`` is computed analytically; 20,000 draws of ``_draw_source`` must land
    within binomial error of it, so a change to the draw order cannot leave the recorded
    shares describing a different mixture."""
    cfg = PriorConfig(p_tree=0.3, p_trivial=0.1, p_crossed=0.2, p_task_family=0.05,
                      p_financial=0.7)
    shares = source_shares(cfg)
    rng = np.random.default_rng(0)
    n = 20_000
    drawn = [_draw_source(rng, cfg)[0] for _ in range(n)]
    for name, share in shares["realised"].items():
        observed = drawn.count(name) / n
        assert abs(observed - share) < 4 * np.sqrt(share * (1 - share) / n) + 1e-12, name
    assert sum(shares["realised"].values()) == pytest.approx(1.0)


def test_shares_clip_weights_as_the_draw_does_and_name_each_panel():
    """A weight above 1 takes everything left, a zero weight takes nothing, and each real
    panel gets its own entry in draw order."""
    shares = source_shares(PriorConfig(p_tree=0.0, p_trivial=1.5, p_financial=0.5))
    assert shares["realised"]["trivial"] == 1.0
    assert shares["realised"]["financial"] == 0.0 and shares["realised"]["scm"] == 0.0
    panels = source_shares(PriorConfig(real_panels="a.parquet=0.5,b.parquet=0.5",
                                       p_financial=0.5))["realised"]
    assert list(panels)[-4:] == ["real_panel:a.parquet", "real_panel:b.parquet", "financial",
                                 "scm"]
    assert panels["real_panel:b.parquet"] == pytest.approx(0.25)
    assert panels["financial"] == pytest.approx(0.125)


def test_weighted_shares_are_the_normalised_weights_in_draw_order():
    """DR2.1: with ``source_weights`` the record states the weights as given and their
    normalised shares, every named source included (a zero weight too), panels by path."""
    shares = source_shares(PriorConfig(source_weights={
        "scm": 2.0, "real_panel:b.parquet": 1.0, "real_panel:a.parquet": 1.0, "tree": 0.0,
    }))
    assert shares["mode"] == "weighted"
    assert list(shares["configured"]) == ["tree", "real_panel:a.parquet",
                                          "real_panel:b.parquet", "scm"]
    assert shares["configured"]["scm"] == 2.0
    assert shares["realised"] == {"tree": 0.0, "real_panel:a.parquet": 0.25,
                                  "real_panel:b.parquet": 0.25, "scm": 0.5}
    assert source_shares(PriorConfig())["mode"] == "sequential"
    json.dumps(shares)


def test_git_state_is_none_outside_a_repository(tmp_path):
    assert git_state(tmp_path) == {"commit": None, "dirty": None}


def test_json_safe_converts_what_a_config_holds():
    out = json_safe({"t": (1, 2), "a": np.float32(0.5), "arr": np.arange(2), 3: Path("x"),
                     "obj": object})
    assert out["t"] == [1, 2] and out["a"] == 0.5 and out["arr"] == [0, 1]
    assert out["3"] == "x" and out["obj"].startswith("<class")
    json.dumps(out)
    # subclasses become their base type, or weights_only loading refuses the checkpoint
    import torch

    assert type(json_safe(torch.__version__)) is str
    assert type(json_safe(np.float64(0.5))) is float


def test_estimator_params_drop_the_model_and_keep_every_setting():
    """The model object never enters a run JSON; every scalar setting does."""
    from fintfm.inference.classifier import FinancialTFMClassifier
    from fintfm.modeling.model import FinancialTFM, ModelConfig

    model = FinancialTFM(ModelConfig(max_features=4, max_classes=2, d_cell=8, d_model=16,
                                     n_layers=1, n_col_layers=1, n_heads=2))
    clf = FinancialTFMClassifier(model, n_ensemble=3, random_state=7)
    params = estimator_params(clf)
    assert "model" not in params
    assert set(params) == set(clf.get_params()) - {"model"}
    assert params["n_ensemble"] == 3 and params["random_state"] == 7
    json.dumps(params)


def _runs_manifest_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "runs_manifest.py"
    spec = importlib.util.spec_from_file_location("runs_manifest", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_runs_manifest_lists_cited_checkpoints_by_hash_and_section(tmp_path):
    """``scripts/runs_manifest.py``: a ``runs/`` citation resolves at its path, a bare name at
    the top of runs/ or by file name below it, training state is not a citation, and a cited
    file this machine lacks is listed as missing rather than dropped."""
    mod = _runs_manifest_module()
    findings = tmp_path / "FINDINGS.md"
    findings.write_text(
        "# Findings\n\n## Where we stand\nsee `runs/a.pt`\n\n"
        "## 3. Third\n`runs/sub/b.pt` and `c.pt`, resumed from `runs/a.pt.state`\n"
        "### a subsection\nstill §3: `runs/a.pt`\n"
        "## §12 — Twelfth\n`/tmp/c.pt` and `runs/gone.pt`\n"
    )
    runs = tmp_path / "runs"
    (runs / "sub").mkdir(parents=True)
    (runs / "a.pt").write_bytes(b"aa")
    (runs / "sub" / "b.pt").write_bytes(b"bbb")
    (runs / "deep").mkdir()
    (runs / "deep" / "c.pt").write_bytes(b"c")
    out = tmp_path / "MANIFEST.json"
    mod.main(["--findings", str(findings), "--runs-dir", str(runs), "--out", str(out)])
    manifest = json.loads(out.read_text())
    by_path = {c["path"]: c for c in manifest["checkpoints"]}
    assert sorted(by_path) == ["runs/a.pt", "runs/deep/c.pt", "runs/sub/b.pt"]
    assert by_path["runs/a.pt"]["sections"] == ["Where we stand", "§3"]
    assert by_path["runs/a.pt"]["size"] == 2
    assert by_path["runs/a.pt"]["sha256"] == sha256_file(runs / "a.pt")
    assert by_path["runs/deep/c.pt"]["cited_as"] == ["/tmp/c.pt", "c.pt"]
    assert by_path["runs/deep/c.pt"]["sections"] == ["§3", "§12"]  # document order
    assert manifest["missing"] == [{"cited_as": "runs/gone.pt", "sections": ["§12"]}]
    first = out.read_text()
    mod.main(["--findings", str(findings), "--runs-dir", str(runs), "--out", str(out)])
    assert out.read_text() == first  # regenerating unchanged inputs changes nothing
