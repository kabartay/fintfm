"""Configuration loading, validation, and the guard against drift from code defaults.

The most important test here is
:func:`test_inference_section_matches_the_classifier_defaults`. The ``inference`` section
deliberately *mirrors* the estimator's literal defaults rather than replacing them — a library
must behave identically without reading a file from disk, and an estimator whose defaults
depend on an environment variable is not reproducible across machines. That duplication is
only safe if something fails when the two diverge, which is what that test is.
"""

import inspect
from dataclasses import fields
from pathlib import Path

import pytest

from fintfm.config import (
    CONFIG_ENV_VAR,
    DEFAULT_CONFIG_PATH,
    ConfigError,
    InferenceConfig,
    load_config,
)
from fintfm.inference.classifier import FinancialTFMClassifier

#: Constructor arguments with no ``inference`` key: the network, where it runs, and the seed a
#: harness varies per fold. Every other argument must have one (design review DR1.3).
_NOT_CONFIGURATION = {"self", "model", "device", "random_state"}


def test_packaged_default_exists_and_loads():
    assert DEFAULT_CONFIG_PATH.exists(), DEFAULT_CONFIG_PATH
    cfg = load_config()
    assert cfg.v4finbench.test_from > cfg.v4finbench.train_until
    assert cfg.context_sweep.strategies
    assert str(DEFAULT_CONFIG_PATH) in cfg.provenance()


def test_inference_section_matches_the_classifier_defaults():
    """The drift guard. If you change one, change the other.

    Iterates the constructor's signature and the section's fields rather than a hand-written
    list (design review DR1.3), so a constructor argument added without a config key, or a key
    with no argument behind it, fails here instead of being silently ignored."""
    cfg = load_config().inference
    params = inspect.signature(FinancialTFMClassifier.__init__).parameters
    wanted = set(params) - _NOT_CONFIGURATION
    keys = {f.name for f in fields(InferenceConfig)}
    assert wanted - keys == set(), f"constructor arguments with no inference key: {wanted - keys}"
    assert keys - wanted == set(), f"inference keys no constructor argument reads: {keys - wanted}"
    for name in sorted(keys):
        assert params[name].default == getattr(cfg, name), (
            f"{name}: classifier default {params[name].default!r} disagrees with "
            f"config {getattr(cfg, name)!r} — update fintfm/configs/default.yaml"
        )
        assert getattr(InferenceConfig(), name) == getattr(cfg, name), name


#: A valid non-default value for every inference key; the test below refuses to run if a key
#: is missing here, so a new key must state what "changed" means for it.
_NON_DEFAULT_INFERENCE = {
    "max_context": 777,
    "context_strategy": "retrieval",
    "feature_transform": "power",
    "n_ensemble": 3,
    "ensemble_label_swap": True,
    "ensemble_feature_frac": 0.5,
    "correct_prior": False,
    "query_chunk": 99,
    "feature_chunk": 4,
    "retrieval_groups": 5,
    "retrieval_min_positive": 2,
    "prototype_minority_ratio": 0.6,
    "winsor_quantile": 0.05,
    "transform_subsample": 123,
    "finetune_steps": 2,
    "finetune_lr": 3e-4,
    "finetune_batch": 3,
    "finetune_query": 64,
    "finetune_context": 128,
    "finetune_query_group_frac": 0.5,
    "finetune_query_pos_frac": 0.2,
    "finetune_seed": 4,
    "finetune_clip": 1.0,
    "ensemble_transforms": ("rank", "squash"),
    "ensemble_svd_components": 2,
    "ensemble_class_shift": True,
    "max_features_strategy": "subsample",
    "retrieval_per_class": 3,
}


def _tiny_model():
    """An untrained network; construction never runs it."""
    from fintfm.modeling.model import FinancialTFM, ModelConfig

    return FinancialTFM(ModelConfig(max_features=8, max_classes=2, d_cell=8, d_model=16,
                                    n_heads=2, n_layers=1, n_col_layers=1, d_ff=32))


def test_every_inference_key_reaches_the_classifier(tmp_path):
    """DR1.3's verification: a file setting every inference key to a non-default value builds,
    through ``from_config`` and through the V4FinBench protocol's constructor, a classifier
    carrying every one of them. Before DR1.3 ``correct_prior``, ``retrieval_groups``,
    ``retrieval_min_positive`` and ``prototype_minority_ratio`` loaded and never arrived."""
    import yaml

    from fintfm.experiments.v4_protocol import fintfm_classifier

    keys = {f.name for f in fields(InferenceConfig)}
    assert set(_NON_DEFAULT_INFERENCE) == keys
    defaults = InferenceConfig()
    for name, value in _NON_DEFAULT_INFERENCE.items():
        assert value != getattr(defaults, name), f"{name}: {value!r} is the default"
    f = tmp_path / "every-key.yaml"
    f.write_text(yaml.safe_dump({"inference": {
        k: list(v) if isinstance(v, tuple) else v for k, v in _NON_DEFAULT_INFERENCE.items()
    }}))
    cfg = load_config(f)
    model = _tiny_model()
    for clf in (FinancialTFMClassifier.from_config(model, cfg.inference),
                fintfm_classifier(model, cfg, random_state=3)):
        got = clf.get_params(deep=False)
        for name, value in _NON_DEFAULT_INFERENCE.items():
            assert got[name] == value, f"{name}: classifier has {got[name]!r}, config {value!r}"
        assert got["model"] is model


def test_from_config_overrides_win_and_unknown_overrides_are_refused():
    model = _tiny_model()
    clf = FinancialTFMClassifier.from_config(
        model, load_config().inference, max_context=50, random_state=9, device="cpu"
    )
    assert (clf.max_context, clf.random_state) == (50, 9)
    with pytest.raises(TypeError):
        FinancialTFMClassifier.from_config(model, load_config().inference, max_contxt=50)


def test_from_config_at_the_packaged_defaults_is_the_plain_classifier():
    """Byte identity at the default: the section mirrors the constructor, so ``from_config``
    builds exactly ``FinancialTFMClassifier(model)``."""
    model = _tiny_model()
    built = FinancialTFMClassifier.from_config(model, load_config().inference)
    assert built.get_params(deep=False) == FinancialTFMClassifier(model).get_params(deep=False)


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


def test_prior_section_mirrors_the_context_class_floor_default():
    # AU1.16's flag lives in PriorConfig and is supplied from configuration by fintfm-train;
    # the two defaults must agree, and both must be 0 so no existing checkpoint's distribution
    # moves unless a run asks for it.
    from fintfm.prior import PriorConfig

    assert load_config().prior.min_ctx_per_class == PriorConfig.min_ctx_per_class == 0


def test_prior_section_mirrors_the_phase5_lever_defaults():
    # AU5.4-AU5.6's levers live in PriorConfig and are supplied from configuration by
    # fintfm-train; the defaults must agree and must all be off, so no existing checkpoint's
    # batches move unless a run asks for it.
    from fintfm.prior import PriorConfig

    cfg = load_config().prior
    assert cfg.permute_classes is PriorConfig.permute_classes is False
    assert cfg.feature_transforms == PriorConfig.feature_transforms == ()
    assert cfg.max_query_rows is PriorConfig.max_query_rows is None


def test_shipped_example_configs_all_load():
    """A broken example is worse than no example: it is copied before it is run."""
    root = Path(__file__).resolve().parents[1] / "configs"
    files = sorted(root.rglob("*.yaml"))  # configs/published/ too
    assert root / "published" / "v4-cellattn-labels.yaml" in files
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
        ("inference:\n  retrieval_per_class: 0\n", "retrieval_per_class"),
        ("large_n:\n  method: forest\n", "large_n.method"),
        ("large_n:\n  boost_weighting: square\n", "large_n.boost_weighting"),
        ("large_n:\n  partition_max_depth: 5\n", "large_n.partition_max_depth"),
        ("context_sweep:\n  context_sizes: [0]\n", "context_sizes"),
        ("context_sweep:\n  seeds: []\n", "seeds"),
        ("prior:\n  absolute_rate_floor: 0.9\n  rate_ceiling: 0.3\n", "prior rates"),
        ("prior:\n  feature_transforms: [rank, quantile]\n", "feature_transforms"),
        ("prior:\n  max_query_rows: 0\n", "max_query_rows"),
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


def test_train_section_matches_the_train_config_defaults():
    """Drift guard for the ``train`` section: it carries every ``TrainConfig`` field (design
    review DR1.3), each at ``TrainConfig``'s literal default, which is what pretraining did
    before the knob existed. Iterates both dataclasses' fields, so a field added to either
    alone fails here."""
    from fintfm.config import TrainConfigDefaults
    from fintfm.modeling.train import TrainConfig

    cfg = load_config().train
    library = TrainConfig()
    assert [f.name for f in fields(TrainConfigDefaults)] == [f.name for f in fields(TrainConfig)]
    for f in fields(TrainConfigDefaults):
        assert getattr(cfg, f.name) == getattr(library, f.name) == f.default, (
            f"train.{f.name}: config {getattr(cfg, f.name)!r} disagrees with TrainConfig "
            f"{getattr(library, f.name)!r} — update fintfm/configs/default.yaml"
        )


def test_large_n_section_matches_the_wrapper_defaults():
    """Drift guard for the ``large_n`` section (audit AU5.8): every ``partition_*`` and
    ``boost_*`` key mirrors the matching wrapper's literal default, every wrapper argument
    other than the estimator and seed has a key, and the packaged method wraps nothing."""
    from dataclasses import fields

    from fintfm.config import LargeNConfig
    from fintfm.inference.large_n import BoostedTFMClassifier, PartitionedTFMClassifier

    cfg = load_config().large_n
    assert cfg.method == "none"
    names = {f.name for f in fields(LargeNConfig)} - {"method"}
    for prefix, cls in (("partition_", PartitionedTFMClassifier),
                        ("boost_", BoostedTFMClassifier)):
        params = inspect.signature(cls.__init__).parameters
        wanted = {f"{prefix}{p}" for p in params} - {
            f"{prefix}self", f"{prefix}estimator", f"{prefix}random_state"
        }
        assert wanted == {n for n in names if n.startswith(prefix)}
        for name in wanted:
            default = params[name.removeprefix(prefix)].default
            assert getattr(cfg, name) == default, (
                f"large_n.{name}: config {getattr(cfg, name)!r} disagrees with "
                f"{cls.__name__} {default!r} — update fintfm/configs/default.yaml"
            )


def test_model_section_matches_the_model_config_defaults():
    """Drift guard for the ``model`` section: it carries every ``ModelConfig`` field (design
    review DR1.3), each at ``ModelConfig``'s default, so a field added to one side and not the
    other fails here rather than being silently ignored. ``d_ff`` is the one stated exception:
    ``None``, which ``fintfm-train`` resolves to ``4 * d_model`` as its flag always did."""
    from fintfm.config import ModelConfigDefaults
    from fintfm.modeling.model import ModelConfig

    cfg = load_config().model
    library = ModelConfig()
    assert [f.name for f in fields(ModelConfigDefaults)] == [f.name for f in fields(ModelConfig)]
    for f in fields(ModelConfigDefaults):
        if f.name == "d_ff":
            assert cfg.d_ff is None and f.default is None and library.d_ff == 512
            continue
        assert getattr(cfg, f.name) == getattr(library, f.name) == f.default, (
            f"model.{f.name}: config {getattr(cfg, f.name)!r} disagrees with ModelConfig "
            f"{getattr(library, f.name)!r} — update fintfm/configs/default.yaml"
        )


def test_prior_section_mirrors_every_prior_flag_default():
    """Each prior key ``fintfm-train`` resolves is a field of ``PriorConfig`` and of the
    section, at ``PriorConfig``'s default, and every section field is consumed: a knob, a
    lever, or the envelope (``n_sectors`` is ``PriorConfig.n_sectors_max``)."""
    from fintfm.config import PriorConfigDefaults
    from fintfm.modeling.train import _PRIOR_CONFIG_KNOBS
    from fintfm.prior import PriorConfig

    section = {f.name: f.default for f in fields(PriorConfigDefaults)}
    library = PriorConfig()
    cfg = load_config().prior
    for name in _PRIOR_CONFIG_KNOBS:
        assert name in section, f"prior.{name}: no config key"
        assert getattr(cfg, name) == section[name] == getattr(library, name), name
    envelope = {"sharpness_min", "sharpness_max", "min_expected_positives",
                "absolute_rate_floor", "rate_ceiling", "n_sectors"}
    levers = {"permute_classes", "feature_transforms", "max_query_rows"}
    assert set(section) == set(_PRIOR_CONFIG_KNOBS) | envelope | levers


def test_every_train_flag_defaults_to_none_and_names_a_config_field():
    """``CLAUDE.md``, "Configuration, not constants": every ``fintfm-train`` flag defaults to
    ``None`` so the configuration supplies the value (``--out``, a path, aside), every flag
    names a field the run consumes, and every ``ModelConfig`` and ``TrainConfig`` field has a
    flag. Iterated off the parser, so a flag added with a literal default fails here."""
    from fintfm.modeling.model import ModelConfig
    from fintfm.modeling.train import _PRIOR_CONFIG_KNOBS, TrainConfig, build_parser

    actions = {a.dest: a for a in build_parser()._actions if a.dest != "help"}
    literal = {d for d, a in actions.items() if a.default is not None}
    assert literal == {"out"}, f"flags with a literal default: {sorted(literal - {'out'})}"
    model = {f.name for f in fields(ModelConfig)}
    train = {f.name for f in fields(TrainConfig)}
    levers = {"permute_classes", "feature_transforms", "max_query_rows"}
    invocation = {"config", "out", "threads", "benchmark_steps"}
    consumed = model | train | set(_PRIOR_CONFIG_KNOBS) | levers | invocation
    assert set(actions) - consumed == set(), f"flags nothing reads: {set(actions) - consumed}"
    assert (model | train) - set(actions) == set(), (model | train) - set(actions)


@pytest.mark.parametrize(
    "body, match",
    [
        ("train:\n  amp: off\n", "train.amp"),
        ("train:\n  amp: fp8\n", "train.amp"),
        ("model:\n  value_embedding: spline\n", "model.value_embedding"),
        ("model:\n  n_heads: 0\n", "model.n_heads"),
        ("model:\n  n_head: 2\n", "n_head"),
        ("train:\n  warmup_steps: -1\n", "warmup_steps"),
        ("train:\n  muon_lr: 0\n", "muon_lr"),
        ("train:\n  decay_norm_bias: maybe\n", "decay_norm_bias"),
    ],
)
def test_train_section_values_are_validated(tmp_path, body, match):
    """``amp: off`` is the trap: YAML reads it as ``False``, not as a mode name."""
    f = tmp_path / "c.yaml"
    f.write_text(body)
    with pytest.raises(ConfigError, match=match):
        load_config(f)
