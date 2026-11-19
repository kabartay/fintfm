from __future__ import annotations

import sys
import types

import pytest

pytest.importorskip("fintfm")

from readout_support import make

from fintfm.readout.checkpoints import (
    BINARY_CHECKPOINT,
    MULTICLASS_CHECKPOINT,
    REGRESSION_CHECKPOINT,
    resolve_checkpoint,
)
from fintfm.readout.entry import entry_defaults, validated_probe
from fintfm.readout.estimator import LinearReadoutModel

#: The measured entry per problem type (fintfm FINDINGS §232, §235, §237).
ENTRY = {
    "binary": {
        "checkpoint": BINARY_CHECKPOINT,
        "partition": {"n_bags": 4},
        "probe": {"scale": True, "cv": True, "raw": True},
        "n_bins": 10,
    },
    "multiclass": {
        "checkpoint": MULTICLASS_CHECKPOINT,
        "partition": {"n_bags": 8},
        "probe": {"scale": True, "cv": True, "raw": True},
        "n_bins": 10,
    },
    "regression": {
        "checkpoint": REGRESSION_CHECKPOINT,
        "partition": {"n_bags": 4},
        "probe": {"raw": True},
        "n_bins": 10,
    },
}


def resolve(model: LinearReadoutModel) -> tuple[dict, dict]:
    params = model._get_model_params()
    entry = model._resolve_entry(params)
    return entry, params


@pytest.mark.parametrize("problem_type", sorted(ENTRY))
def test_default_is_the_entry(problem_type, tmp_path):
    entry, rest = resolve(make(problem_type, tmp_path))
    assert entry == {k: v for k, v in ENTRY[problem_type].items() if k != "checkpoint"}
    assert entry_defaults(problem_type)["checkpoint"] == ENTRY[problem_type]["checkpoint"]
    # What is left is what FinancialTFMClassifier receives.
    assert rest["feature_transform"] == "squash"
    assert rest["max_features_strategy"] == "select"
    assert rest["max_context"] == 1000
    assert rest["n_ensemble"] == 8
    assert not any(
        k.startswith(("checkpoint_", "partition_", "probe_", "model_path")) for k in rest
    )
    assert "n_bins" not in rest


def test_supports_all_three_problem_types():
    for problem_type in ("binary", "multiclass", "regression"):
        assert LinearReadoutModel(problem_type, "/unused.pt").problem_type == problem_type
    with pytest.raises(ValueError, match="problem type"):
        LinearReadoutModel("quantile", "/unused.pt")


def test_default_hyperparameter_names_and_values(tmp_path):
    binary = entry_defaults("binary")
    assert binary["partition"] == {"n_bags": 4}
    assert binary["partition_bags_max_rows"] == 5_000
    assert entry_defaults("multiclass")["partition"] == {"n_bags": 8}
    assert entry_defaults("regression")["partition"] == {"n_bags": 4}
    assert entry_defaults("regression")["probe"] == {"raw": True}
    assert binary["checkpoint"] == BINARY_CHECKPOINT
    assert binary["n_bins"] == 10
    assert entry_defaults("multiclass")["checkpoint"] == MULTICLASS_CHECKPOINT
    assert entry_defaults("regression")["checkpoint"] == REGRESSION_CHECKPOINT


def test_entry_defaults_are_fresh_copies():
    first = entry_defaults("binary")
    first["partition"]["n_bags"] = 99
    first["checkpoint"]["repo_id"] = "changed"
    assert entry_defaults("binary")["partition"] == {"n_bags": 4}
    assert entry_defaults("binary")["checkpoint"] == BINARY_CHECKPOINT


def test_entry_defaults_refuse_an_unknown_problem_type():
    with pytest.raises(ValueError, match="problem type"):
        entry_defaults("quantile")


@pytest.mark.parametrize(
    ("hyperparameters", "match"),
    [
        ({"partition_binary": {"n_bgas": 4}}, "n_bgas"),
        ({"partition_binary": {"estimator": None}}, "estimator"),
        ({"partition_binary": 4}, "JSON object or null"),
        ({"n_bins": 1}, "at least 2"),
        ({"checkpoint_binary": {"repo_id": "x"}}, "do not match"),
    ],
)
def test_unknown_or_malformed_hyperparameters_raise(hyperparameters, match, tmp_path):
    with pytest.raises(ValueError, match=match):
        resolve(make("binary", tmp_path, **hyperparameters))


def test_unknown_probe_key_raises(tmp_path):
    with pytest.raises(ValueError, match="unknown key"):
        validated_probe({"scal": True}, source="probe_binary")
    with pytest.raises(ValueError, match="no_rep needs raw"):
        validated_probe({"no_rep": True}, source="probe_binary")


def test_local_path_resolves_to_itself():
    assert resolve_checkpoint("/a/b.pt", problem_type="binary") == "/a/b.pt"


def test_published_checkpoint_checks_the_sha256(monkeypatch, tmp_path):
    fake = tmp_path / "weights.pt"
    fake.write_bytes(b"not the checkpoint")
    hub = types.SimpleNamespace(hf_hub_download=lambda **kwargs: str(fake))
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    with pytest.raises(ValueError, match="expected d231b062"):
        resolve_checkpoint(BINARY_CHECKPOINT, problem_type="binary")


#: The measured entry's levers per problem type: binary ``fintfm_lp_bin_both_kaggle`` (§254, §257),
#: multiclass ``fintfm_lp_mc_floor_cpu`` and ``fintfm_lp_mc_floor_sdss17_mac`` (§255, §265, §269),
#: regression ``fintfm_lp_reg_rawpart4_pure_1114`` (§255). Each run's recorded hyperparameters give these.
MEASURED = {
    "binary": {"scale_with_n": True, "budget": 50_000, "crossfit": 5, "floor": None},
    "multiclass": {"scale_with_n": False, "budget": 20_000, "crossfit": 0, "floor": "cv"},
    "regression": {"scale_with_n": False, "budget": 20_000, "crossfit": 0, "floor": None},
}

#: The §283 levers of the entry per problem type: multiclass one-hot indicators, regression
#: random-Fourier-feature readout capped at 5,000 training rows; binary has neither.
ONEHOT_ENTRY = {"min_count": 3, "max_columns": 256, "max_levels": 64}
MEASURED_GATED = {
    "binary": {"onehot": None, "readout": "ridge", "rff_max_rows": None},
    "multiclass": {"onehot": ONEHOT_ENTRY, "readout": "ridge", "rff_max_rows": None},
    "regression": {"onehot": None, "readout": "rff", "rff_max_rows": 5_000},
}


def levers(model: LinearReadoutModel) -> dict:
    return {
        "scale_with_n": model._scale_with_n,
        "budget": model._probe_budget_setting,
        "crossfit": model._crossfit_folds,
        "floor": model._prob_floor_setting,
    }


def gated(model: LinearReadoutModel) -> dict:
    return {
        "onehot": model._onehot_cfg,
        "readout": model._reg_readout,
        "rff_max_rows": model._reg_rff_max_rows,
    }


@pytest.mark.parametrize("problem_type", sorted(ENTRY))
def test_resolved_entry_equals_the_measured_entry(problem_type, tmp_path):
    model = make(problem_type, tmp_path)
    entry, _ = resolve(model)
    assert entry == {k: v for k, v in ENTRY[problem_type].items() if k != "checkpoint"}
    assert levers(model) == MEASURED[problem_type]


@pytest.mark.parametrize(
    ("problem_type", "hyperparameters", "expected"),
    [
        ("binary", {"partition_scale_with_n": False}, {"scale_with_n": False}),
        ("binary", {"probe_budget": 20_000}, {"budget": 20_000}),
        ("binary", {"probe_crossfit_folds": 0}, {"crossfit": 0}),
        ("binary", {"probe_prob_floor": "cv"}, {"floor": "cv"}),
        ("multiclass", {"probe_crossfit_folds": 5}, {"crossfit": 5}),
        ("multiclass", {"probe_prob_floor": None}, {"floor": None}),
        (
            "multiclass",
            {"partition_scale_with_n": True, "probe_budget": 50_000},
            {"scale_with_n": True, "budget": 50_000},
        ),
        ("regression", {"probe_crossfit_folds": 5}, {"crossfit": 5}),
        (
            "regression",
            {"partition_scale_with_n": True, "probe_budget": "all"},
            {"scale_with_n": True, "budget": "all"},
        ),
    ],
)
def test_explicit_hyperparameters_override_the_entry_levers(
    problem_type, hyperparameters, expected, tmp_path
):
    model = make(problem_type, tmp_path, **hyperparameters)
    resolve(model)
    assert levers(model) == {**MEASURED[problem_type], **expected}


@pytest.mark.parametrize("problem_type", sorted(ENTRY))
def test_gated_levers_default_to_the_entry_per_problem_type(problem_type, tmp_path):
    model = make(problem_type, tmp_path)
    resolve(model)
    assert gated(model) == MEASURED_GATED[problem_type]


@pytest.mark.parametrize(
    ("problem_type", "hyperparameters", "expected"),
    [
        ("multiclass", {"probe_onehot": None}, {"onehot": None}),
        ("multiclass", {"probe_onehot": True}, {"onehot": {**ONEHOT_ENTRY, "max_levels": None}}),
        (
            "multiclass",
            {"probe_onehot": {"max_levels": 8}},
            {"onehot": {**ONEHOT_ENTRY, "max_levels": 8}},
        ),
        ("regression", {"reg_readout": "ridge"}, {"readout": "ridge"}),
        ("regression", {"reg_rff_max_rows": 20_000}, {"rff_max_rows": 20_000}),
        ("regression", {"reg_rff_max_rows": None}, {"rff_max_rows": None}),
    ],
)
def test_explicit_hyperparameters_override_the_gated_levers(
    problem_type, hyperparameters, expected, tmp_path
):
    model = make(problem_type, tmp_path, **hyperparameters)
    resolve(model)
    assert gated(model) == {**MEASURED_GATED[problem_type], **expected}


#: The §286 gate: the binary entry's four bags apply to fits of at most 5,000 training rows.
BAGS_GATE = {"binary": 5_000, "multiclass": None, "regression": None}


def partition_for(problem_type: str, tmp_path, n_rows: int, **hyperparameters) -> dict | None:
    """The partition a fit of ``n_rows`` training rows is built with."""
    model = make(problem_type, tmp_path, **hyperparameters)
    model._n_fit_rows = n_rows  # set by _fit before the entry is resolved
    partition = resolve(model)[0]["partition"]
    # The binary entry scales the depth with the row count; that is not what these tests are about.
    return None if partition is None else {k: v for k, v in partition.items() if k != "max_depth"}


@pytest.mark.parametrize("problem_type", sorted(ENTRY))
def test_bags_gate_defaults_to_the_entry_per_problem_type(problem_type, tmp_path):
    params = make(problem_type, tmp_path)._get_model_params()
    assert params["partition_bags_max_rows"] == BAGS_GATE[problem_type]


def test_binary_entry_bags_follow_the_row_count(tmp_path):
    assert partition_for("binary", tmp_path, 5_000) == {"n_bags": 4}
    assert partition_for("binary", tmp_path, 5_001) == {}
    assert partition_for("binary", tmp_path, 100) == {"n_bags": 4}


@pytest.mark.parametrize(("problem_type", "bags"), [("multiclass", 8), ("regression", 4)])
def test_multiclass_and_regression_bags_are_never_gated(problem_type, bags, tmp_path):
    assert partition_for(problem_type, tmp_path, 1_000_000) == {"n_bags": bags}


#: FINDINGS §290, entry ``fintfm_g287_bag4_spline``: the binary entry adds spline features on the ``raw`` block
#: for fits of at most 10,000 training rows. FINDINGS §293, entry ``fintfm_g292_emr``: multiclass the same, and
#: regression at every size.
SPLINE = {"binary": True, "multiclass": True, "regression": True}
SPLINE_GATE = {"binary": 10_000, "multiclass": 10_000, "regression": None}
#: ``probe_spline=True`` resolved: ``SPLINE_N_KNOTS``, ``SPLINE_DEGREE``, ``SPLINE_MAX_COLUMNS``.
SPLINE_ON = {"n_knots": 5, "degree": 3, "max_columns": 64}


def spline_for(problem_type: str, tmp_path, n_rows: int, **hyperparameters) -> dict | None:
    """The spline settings a fit of ``n_rows`` training rows resolves to (``None``: no splines)."""
    model = make(problem_type, tmp_path, **hyperparameters)
    model._n_fit_rows = n_rows  # set by _fit before the entry is resolved
    resolve(model)
    return model._spline_cfg


@pytest.mark.parametrize("problem_type", sorted(ENTRY))
def test_spline_defaults_to_the_entry_per_problem_type(problem_type, tmp_path):
    params = make(problem_type, tmp_path)._get_model_params()
    assert params["probe_spline"] == SPLINE[problem_type]
    assert params["spline_max_rows"] == SPLINE_GATE[problem_type]


@pytest.mark.parametrize("problem_type", ["binary", "multiclass", "regression"])
def test_every_entry_has_the_raw_option_splines_need(problem_type, tmp_path):
    entry, _ = resolve(make(problem_type, tmp_path))
    assert entry["probe"]["raw"] is True


@pytest.mark.parametrize("problem_type", ["binary", "multiclass"])
def test_binary_and_multiclass_entry_splines_follow_the_row_count(problem_type, tmp_path):
    assert spline_for(problem_type, tmp_path, 100) == SPLINE_ON
    assert spline_for(problem_type, tmp_path, 10_000) == SPLINE_ON
    assert spline_for(problem_type, tmp_path, 10_001) is None
    assert spline_for(problem_type, tmp_path, 1_000_000) is None


def test_regression_entry_splines_at_every_size(tmp_path):
    for n_rows in (100, 10_000, 1_000_000):
        assert spline_for("regression", tmp_path, n_rows) == SPLINE_ON


@pytest.mark.parametrize("problem_type", ["multiclass", "regression"])
def test_multiclass_and_regression_entry_is_unchanged_by_the_spline_defaults(
    problem_type, tmp_path
):
    entry, rest = resolve(make(problem_type, tmp_path))
    assert {k: entry[k] for k in ENTRY[problem_type] if k != "checkpoint"} == {
        k: v for k, v in ENTRY[problem_type].items() if k != "checkpoint"
    }
    assert (
        "probe_spline" not in rest and "spline_max_rows" not in rest
    )  # consumed by the entry resolution


def test_spline_without_the_raw_option_is_ignored(tmp_path):
    assert spline_for("binary", tmp_path, 100, probe_binary={"scale": True}) is None


def test_published_checkpoint_without_huggingface_hub_names_the_remedy(monkeypatch):
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    with pytest.raises(ImportError, match="huggingface_hub"):
        resolve_checkpoint(BINARY_CHECKPOINT, problem_type="binary")
