"""Configuration loading, validation, and the guard against drift from code defaults.

The most important test here is
:func:`test_inference_section_matches_the_classifier_defaults`. The ``inference`` section
deliberately *mirrors* the estimator's literal defaults rather than replacing them — a library
must behave identically without reading a file from disk, and an estimator whose defaults
depend on an environment variable is not reproducible across machines. That duplication is
only safe if something fails when the two diverge, which is what that test is.
"""

import inspect
from pathlib import Path

import pytest

from fintfm.config import (
    CONFIG_ENV_VAR,
    DEFAULT_CONFIG_PATH,
    ConfigError,
    load_config,
)
from fintfm.inference.classifier import FinancialTFMClassifier


def test_packaged_default_exists_and_loads():
    assert DEFAULT_CONFIG_PATH.exists(), DEFAULT_CONFIG_PATH
    cfg = load_config()
    assert cfg.v4finbench.test_from > cfg.v4finbench.train_until
    assert cfg.context_sweep.strategies
    assert str(DEFAULT_CONFIG_PATH) in cfg.provenance()


def test_inference_section_matches_the_classifier_defaults():
    """The drift guard. If you change one, change the other."""
    cfg = load_config().inference
    sig = inspect.signature(FinancialTFMClassifier.__init__)
    for name in (
        "max_context", "context_strategy", "feature_transform", "correct_prior",
        "query_chunk", "retrieval_groups", "retrieval_min_positive",
        "prototype_minority_ratio", "n_ensemble",
    ):
        assert sig.parameters[name].default == getattr(cfg, name), (
            f"{name}: classifier default {sig.parameters[name].default!r} disagrees with "
            f"config {getattr(cfg, name)!r} — update fintfm/configs/default.yaml"
        )


def test_prior_section_matches_the_prior_constants():
    from fintfm.prior.financial import (
        _ABSOLUTE_RATE_FLOOR,
        _N_SECTORS,
        _RATE_CEILING,
        MIN_EXPECTED_POSITIVES,
    )

    cfg = load_config().prior
    assert cfg.min_expected_positives == MIN_EXPECTED_POSITIVES
    assert cfg.absolute_rate_floor == _ABSOLUTE_RATE_FLOOR
    assert cfg.rate_ceiling == _RATE_CEILING
    assert cfg.n_sectors == _N_SECTORS


def test_shipped_example_configs_all_load():
    """A broken example is worse than no example: it is copied before it is run."""
    root = Path(__file__).resolve().parents[1] / "configs"
    files = sorted(root.glob("*.yaml"))
    assert files, f"no example configs found under {root}"
    for f in files:
        cfg = load_config(f)
        assert str(f) in cfg.provenance()


def test_override_is_deep_merged_not_replaced(tmp_path):
    f = tmp_path / "o.yaml"
    f.write_text("context_sweep:\n  seeds: [0, 1, 2]\n")
    cfg = load_config(f)
    assert cfg.context_sweep.seeds == (0, 1, 2)
    # untouched keys in the same section survive
    assert cfg.context_sweep.context_sizes == load_config().context_sweep.context_sizes


def test_lists_replace_rather_than_append(tmp_path):
    f = tmp_path / "o.yaml"
    f.write_text("context_sweep:\n  strategies: [uniform]\n")
    assert load_config(f).context_sweep.strategies == ("uniform",)


def test_unknown_key_is_refused_and_named(tmp_path):
    f = tmp_path / "o.yaml"
    f.write_text("v4finbench:\n  train_untill: 2016\n")
    with pytest.raises(ConfigError, match="v4finbench.train_untill"):
        load_config(f)


def test_unknown_section_is_refused(tmp_path):
    f = tmp_path / "o.yaml"
    f.write_text("infrence:\n  max_context: 10\n")
    with pytest.raises(ConfigError, match="infrence"):
        load_config(f)


def test_overlapping_time_split_is_refused(tmp_path):
    """An overlapping split would be reported as out-of-time validation while not being it."""
    f = tmp_path / "o.yaml"
    f.write_text("v4finbench:\n  train_until: 2018\n  test_from: 2017\n")
    with pytest.raises(ConfigError, match="out-of-time"):
        load_config(f)


@pytest.mark.parametrize(
    "body, match",
    [
        ("evaluation:\n  alpha: 1.5\n", "alpha"),
        ("evaluation:\n  bootstrap_resamples: 0\n", "bootstrap_resamples"),
        ("evaluation:\n  min_rows_per_horizon: 0\n", "min_rows_per_horizon"),
        ("inference:\n  winsor_quantile: 0.7\n", "winsor_quantile"),
        ("context_sweep:\n  context_sizes: [0]\n", "context_sizes"),
        ("context_sweep:\n  seeds: []\n", "seeds"),
        ("prior:\n  absolute_rate_floor: 0.9\n  rate_ceiling: 0.3\n", "prior rates"),
    ],
)
def test_out_of_range_values_are_refused(tmp_path, body, match):
    f = tmp_path / "o.yaml"
    f.write_text(body)
    with pytest.raises(ConfigError, match=match):
        load_config(f)


def test_missing_explicit_path_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "absent.yaml")


def test_env_var_is_honoured_and_a_missing_one_is_not_ignored(tmp_path, monkeypatch):
    f = tmp_path / "o.yaml"
    f.write_text("context_sweep:\n  seeds: [7]\n")
    monkeypatch.setenv(CONFIG_ENV_VAR, str(f))
    assert load_config().context_sweep.seeds == (7,)
    # a path that does not exist must fail rather than silently fall back: the whole point of
    # setting the variable is that it takes effect
    monkeypatch.setenv(CONFIG_ENV_VAR, str(tmp_path / "gone.yaml"))
    with pytest.raises(FileNotFoundError):
        load_config()
    # an explicit argument still wins over the environment
    monkeypatch.setenv(CONFIG_ENV_VAR, str(tmp_path / "gone.yaml"))
    assert load_config(f).context_sweep.seeds == (7,)


def test_sources_cannot_be_supplied(tmp_path):
    f = tmp_path / "o.yaml"
    f.write_text("sources: [nope]\n")
    with pytest.raises(ConfigError, match="set by the loader"):
        load_config(f)


def test_non_mapping_document_is_refused(tmp_path):
    f = tmp_path / "o.yaml"
    f.write_text("- a\n- b\n")
    with pytest.raises(ConfigError, match="must be a mapping"):
        load_config(f)


def test_hazard_arms_parse_into_typed_arms():
    arms = load_config().v4finbench.hazard_arms
    assert arms and all(isinstance(a.correct_prior, bool) for a in arms)
    # the uncorrected arm is permanent by decision D8: the base-rate distortion of §28 is
    # measured beside the fix rather than assumed absent
    assert any(not a.correct_prior for a in arms)


def test_malformed_hazard_arm_is_refused(tmp_path):
    f = tmp_path / "o.yaml"
    f.write_text("v4finbench:\n  hazard_arms:\n    - {name: a, strategy: uniform, oops: 1}\n")
    with pytest.raises(ConfigError, match="oops"):
        load_config(f)
