"""Typed, strictly validated configuration, layered from a packaged default.

Why this is layered rather than simply loaded
---------------------------------------------
Three layers, each overriding the one before: the **packaged default**
(``fintfm/configs/default.yaml``), then an optional **override file** given by ``--config`` or
``FINTFM_CONFIG``, then **explicit command-line flags**. An override file is deep-merged, so
it need only carry the keys it changes — a sweep that differs from the default in one year
should be one line, not a copy of the whole file that silently freezes every other value at
the moment it was copied.

Why unknown keys are an error
-----------------------------
A misspelled key in a silently-tolerant loader is the worst kind of configuration bug: the run
completes, reports numbers, and used the default. This repository has already lost a
pretraining run to a value that was quietly not what it appeared to be
(``docs/results/FINDINGS.md`` §28), so :func:`load_config` refuses unknown keys and names the path of
the offender.

Why library defaults are duplicated rather than moved
-----------------------------------------------------
:class:`~fintfm.inference.classifier.FinancialTFMClassifier` keeps its own literal defaults.
A library must behave identically without reading a file from disk, and an estimator whose
defaults depend on an environment variable is not reproducible across machines. The
``inference`` section therefore *mirrors* those defaults for experiments to read, and
``tests/test_config.py`` asserts the two agree — the test is the drift guard, so changing one
without the other fails the suite rather than diverging quietly. The same holds for the
``model`` and ``train`` sections against ``ModelConfig`` and ``TrainConfig``.

Why a harness builds its classifier from the whole section
----------------------------------------------------------
:meth:`FinancialTFMClassifier.from_config` passes every ``inference`` field. Before it, each
harness copied a keyword list by hand, and four keys (``correct_prior``, ``retrieval_groups``,
``retrieval_min_positive``, ``prototype_minority_ratio``) loaded, validated, and never reached
the V4FinBench protocol's classifier: the failure shape a strict loader exists to prevent,
one step later (design review DR1.3).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """A configuration file is malformed, unknown, or internally inconsistent.

    A subclass of :class:`ValueError` so existing handlers still catch it, and its own type so
    a caller can distinguish "the config is wrong" from "an argument is wrong" — which matters
    for a CLI that should tell the user which file to edit.
    """


#: Environment variable naming an override file, used when no ``--config`` is given.
CONFIG_ENV_VAR = "FINTFM_CONFIG"

#: The packaged default, always the base layer.
DEFAULT_CONFIG_PATH = Path(__file__).with_name("configs") / "default.yaml"


@dataclass(frozen=True)
class InferenceConfig:
    """Mirror of :class:`FinancialTFMClassifier`'s defaults, one field per constructor argument.

    Every constructor argument except ``model``, ``device`` and ``random_state`` (the network,
    where it runs, and the seed a harness varies per fold) has a field here, and
    :meth:`FinancialTFMClassifier.from_config` passes every field, so a key set in a file
    reaches the classifier rather than being read by some harnesses and not others (design
    review DR1.3). ``tests/test_config.py`` iterates the constructor's signature and these
    fields, so an argument added to one and not the other fails the suite.
    """

    max_context: int = 2000
    context_strategy: str = "uniform"
    feature_transform: str = "rank"
    n_ensemble: int = 1
    ensemble_label_swap: bool = False
    ensemble_feature_frac: float = 1.0
    correct_prior: bool = True
    query_chunk: int = 2048
    feature_chunk: int | None = 16
    retrieval_groups: int = 64
    retrieval_min_positive: int = 8
    prototype_minority_ratio: float = 0.3
    winsor_quantile: float = 0.01
    transform_subsample: int = 20_000
    finetune_steps: int = 0
    finetune_lr: float = 1e-5
    finetune_batch: int = 2
    finetune_query: int = 512
    finetune_context: int | None = None
    finetune_query_group_frac: float = 0.3
    finetune_query_pos_frac: float | None = None
    finetune_seed: int = 0
    finetune_clip: float | None = None
    ensemble_transforms: tuple[str, ...] | None = None
    ensemble_svd_components: int = 0
    ensemble_class_shift: bool = False
    max_features_strategy: str = "raise"
    retrieval_per_class: int | None = None


@dataclass(frozen=True)
class EvaluationConfig:
    """Thresholds that decide what counts as a scoreable result."""

    bootstrap_resamples: int = 2000
    alpha: float = 0.05
    ece_bins: int = 10
    min_rows_per_horizon: int = 100


@dataclass(frozen=True)
class HazardArm:
    """One configuration of the hazard head, scored as its own arm."""

    name: str
    strategy: str
    correct_prior: bool


@dataclass(frozen=True)
class V4FinBenchConfig:
    """The out-of-time protocol on V4FinBench."""

    max_rows: int = 120_000
    train_until: int = 2016
    test_from: int = 2017
    max_context: int = 2000
    hazard_arms: tuple[HazardArm, ...] = ()


@dataclass(frozen=True)
class ContextSweepConfig:
    """The context-strategy sweep."""

    strategies: tuple[str, ...] = ("balanced", "hybrid", "uniform", "retrieval")
    context_sizes: tuple[int, ...] = (1000, 2000, 4000)
    retrieval_groups: int = 64
    seeds: tuple[int, ...] = (0,)


@dataclass(frozen=True)
class PriorConfigDefaults:
    """The pretraining task distribution ``fintfm-train`` builds its ``PriorConfig`` from.

    The default-rate envelope, the context class floor (AU1.16) and the audit Phase 5
    pretraining levers (AU5.4-AU5.6), plus one field per ``fintfm-train`` prior flag (design
    review DR1.3), so a recipe can be stated in a file. Each mirrors the ``PriorConfig`` field
    of the same name (``n_sectors`` mirrors ``n_sectors_max``); ``tests/test_config.py`` is
    the drift guard. ``max_features``, ``max_classes``, ``n_horizons`` and ``head_type`` are
    not here: the prior takes them from the ``model`` section, so the two cannot disagree.
    """

    sharpness_min: float = 0.3
    sharpness_max: float = 12.0
    min_expected_positives: float = 2.0
    absolute_rate_floor: float = 0.001
    rate_ceiling: float = 0.30
    n_sectors: int = 12
    min_ctx_per_class: int = 0
    permute_classes: bool = False
    feature_transforms: tuple[str, ...] = ()
    max_query_rows: int | None = None
    pad_to_batch_width: bool = False
    n_rows: int = 256
    n_rows_choices: tuple[int, ...] | None = None
    p_financial: float = 0.7
    p_tree: float = 0.0
    p_trivial: float = 0.0
    p_crossed: float = 0.0
    p_task_family: float = 0.0
    p_regression: float = 0.0
    p_learnability_filter: float = 0.0
    scm_legacy: bool = False
    scm_reuse_graph: int = 1
    identity_shuffle: bool = False
    discretize_frac: float = 0.0
    n_noise_features: int = 0
    n_correlated_block_features: int = 0
    label_noise_rate: float = 0.0
    p_real_edgar: float = 0.0
    real_edgar_panel_path: str = "data/edgar/panel.parquet"
    p_real_mortgage: float = 0.0
    real_mortgage_panel_path: str = "data/mortgage/panel.parquet"
    p_real_lendingclub: float = 0.0
    real_lendingclub_panel_path: str = "data/lendingclub/panel.parquet"
    real_panels: str = ""
    source_weights: dict[str, float] | None = None
    financial_gate_prob: float = 0.0
    feature_ratio_bands: tuple[tuple[float, float, float], ...] = ()
    p_binary: float | None = None
    imbalance_draws: bool = False
    categorical_mode: str = "ordered"
    missing_mode: str = "mcar"
    tie_column_prob: float = 0.0
    irrelevant_frac_max: float = 0.0
    duplicate_frac_max: float = 0.0
    label_flip_max: float = 0.0


@dataclass(frozen=True)
class TrainConfigDefaults:
    """Every :class:`~fintfm.modeling.train.TrainConfig` field, for ``fintfm-train`` to read.

    Mirrors ``TrainConfig``'s literal defaults, for the same reason the ``inference`` section
    mirrors the classifier: ``train()`` must behave the same without reading a file. Each
    field is also a ``fintfm-train`` flag defaulting to ``None``, so a flag given wins and a
    flag omitted takes this section's value (design review DR1.3). ``tests/test_config.py``
    is the drift guard, both ways. Every default reproduces pretraining as it ran before these
    knobs existed.
    """

    steps: int = 20_000
    batch_size: int = 64
    lr: float = 3e-4
    warmup_steps: int = 500
    log_every: int = 50
    eval_every: int = 500
    checkpoint_every: int = 0
    seed: int = 0
    device: str = "cpu"
    run_steps: int | None = None
    resume: str | None = None
    init_from: str | None = None
    feature_chunk: int | None = None
    optimizer: str = "adamw"
    task_dump: str | None = None
    task_dump_allow_code_mismatch: bool = False
    amp: str = "none"
    decay_norm_bias: bool = True
    muon_lr: float | None = None
    muon_scale: str = "rms_match"
    loader: str = "sync"
    loader_workers: int = 0
    loader_prefetch: int = 2
    ddp: bool = False
    schedule: str = "cosine"
    decay_frac: float = 0.1
    lr_min: float = 0.0
    eval_set_seed: int | None = None
    eval_batches: int = 5


@dataclass(frozen=True)
class ModelConfigDefaults:
    """Every :class:`~fintfm.modeling.model.ModelConfig` field, for ``fintfm-train`` to read.

    Mirrors ``ModelConfig``'s literal defaults, which stay authoritative because a checkpoint
    rebuilds its model from them without reading a file. Each field is also a
    ``fintfm-train`` flag defaulting to ``None`` (design review DR1.3); ``tests/test_config.py``
    is the drift guard, both ways. **One deliberate difference:** ``d_ff`` is ``None``, which
    ``fintfm-train`` resolves to ``4 * d_model``, the transformer convention its ``--d-ff``
    flag has always applied; ``ModelConfig``'s own 512 is what a bare ``ModelConfig()``
    builds and is not what the command line ever built. Every other default builds the
    architecture as it was before these knobs existed.
    """

    max_features: int = 24
    max_classes: int = 10
    d_cell: int = 64
    d_model: int = 192
    n_heads: int = 4
    n_col_layers: int = 2
    n_layers: int = 6
    d_ff: int | None = None
    dropout: float = 0.0
    pooling: str = "meanmax"
    column_id_dim: int | None = None
    n_horizons: int | None = None
    n_cell_blocks: int = 0
    cell_labels: bool = False
    mask_embedding: bool = False
    head_type: str = "binned"
    n_quantiles: int = 99
    n_cell_heads: int | None = None
    n_summary_tokens: int = 1
    value_embedding: str = "mlp"
    value_basis_size: int = 16
    value_basis_learnable: bool = False
    value_ecdf: bool = False
    attn_length_scale: bool = False
    attn_length_ref: int = 512
    attn_impl: str = "masked"
    hazard_period_context: bool = False


@dataclass(frozen=True)
class LargeNConfig:
    """The large-table wrapper ``fintfm-v4protocol`` puts around its fintfm arm (audit AU5.8).

    The ``partition_*`` and ``boost_*`` fields mirror the literal defaults of
    :class:`~fintfm.inference.large_n.PartitionedTFMClassifier` and
    :class:`~fintfm.inference.large_n.BoostedTFMClassifier`, with the prefix dropped;
    ``tests/test_config.py`` is the drift guard. ``method="none"`` wraps nothing.
    """

    method: str = "none"
    partition_max_depth: int = 2
    partition_min_samples_leaf: float = 0.05  # an int is a row count, as in sklearn
    partition_n_bags: int = 1
    partition_bag_frac: float = 0.6
    partition_min_leaf_rows: int = 200
    partition_min_class_rows: int = 2
    boost_n_rounds: int = 8
    boost_context_rows: int = 512
    boost_learning_rate: float = 0.5
    boost_weighting: str = "abs"
    boost_max_step: float = 4.0
    boost_pool_rows: int | None = None


@dataclass(frozen=True)
class Config:
    """The whole configuration, with the path it was loaded from for the run record."""

    inference: InferenceConfig = field(default_factory=InferenceConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    v4finbench: V4FinBenchConfig = field(default_factory=V4FinBenchConfig)
    context_sweep: ContextSweepConfig = field(default_factory=ContextSweepConfig)
    prior: PriorConfigDefaults = field(default_factory=PriorConfigDefaults)
    train: TrainConfigDefaults = field(default_factory=TrainConfigDefaults)
    model: ModelConfigDefaults = field(default_factory=ModelConfigDefaults)
    large_n: LargeNConfig = field(default_factory=LargeNConfig)
    sources: tuple[str, ...] = ()

    def provenance(self) -> str:
        """The layers this configuration was built from, newest last.

        Returns:
            A string for a run record, so a result can be traced to the values that made it.
        """
        return " <- ".join(self.sources) if self.sources else "built-in defaults"


def _read_yaml(path: Path) -> dict[str, Any]:
    """Parse a YAML mapping, refusing anything else.

    Args:
        path: File to read.

    Returns:
        The parsed mapping, or an empty dict for an empty file.

    Raises:
        FileNotFoundError: If the file is absent.
        ConfigError: If the document is not a mapping.
    """
    import yaml

    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    loaded = yaml.safe_load(path.read_text()) or {}
    if not isinstance(loaded, dict):
        raise ConfigError(
            f"{path}: top level must be a mapping, got {type(loaded).__name__}"
        )
    return loaded


def _deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    """Merge ``over`` onto ``base``, recursing into nested mappings.

    Lists replace rather than concatenate: a sweep asking for two strategies means those two,
    not those two appended to the default four.

    Args:
        base: The lower-priority mapping.
        over: The higher-priority mapping.

    Returns:
        A new merged mapping; neither input is modified.
    """
    out = dict(base)
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _build(cls: type, data: Any, path: str) -> Any:
    """Instantiate a dataclass from parsed data, rejecting unknown keys.

    Args:
        cls: Target dataclass.
        data: Parsed mapping, or a value for a non-dataclass field.
        path: Dotted key path, used in error messages.

    Returns:
        An instance of ``cls``.

    Raises:
        ConfigError: If ``data`` is not a mapping where one is required, or carries a key the
            dataclass does not define. The message names the full path and the valid keys,
            because a configuration error should be fixable without reading this module.
    """
    if not is_dataclass(cls):
        return data
    if not isinstance(data, dict):
        raise ConfigError(
            f"{path or 'config'}: expected a mapping, got {type(data).__name__}"
        )
    valid = {f.name: f for f in fields(cls)}
    unknown = sorted(set(data) - set(valid))
    if unknown:
        where = f"{path}." if path else ""
        raise ConfigError(
            f"unknown config key(s) {', '.join(where + u for u in unknown)}; "
            f"valid keys here are {', '.join(sorted(valid))}"
        )
    kwargs: dict[str, Any] = {}
    for name in valid:
        if name not in data:
            continue
        sub = f"{path}.{name}" if path else name
        value = data[name]
        if name == "hazard_arms":
            if not isinstance(value, list):
                raise ConfigError(f"{sub}: expected a list of arms")
            kwargs[name] = tuple(_build(HazardArm, v, f"{sub}[]") for v in value)
        elif isinstance(value, list):
            kwargs[name] = tuple(value)
        else:
            kwargs[name] = value
    return cls(**kwargs)


def _validate(cfg: Config) -> None:
    """Reject configurations that are internally inconsistent.

    Checked here rather than at use time, so a bad value fails before a run starts instead of
    hours in. An overlapping time split is the important one: it would be reported as
    out-of-time validation while not being it.

    Args:
        cfg: The assembled configuration.

    Raises:
        ConfigError: Naming the offending key and why it is wrong.
    """
    v = cfg.v4finbench
    if v.test_from <= v.train_until:
        raise ConfigError(
            f"v4finbench.test_from ({v.test_from}) must exceed train_until ({v.train_until}); "
            "an overlapping split is not out-of-time validation"
        )
    if not 0.0 < cfg.evaluation.alpha < 1.0:
        raise ConfigError(f"evaluation.alpha must lie in (0, 1), got {cfg.evaluation.alpha}")
    if cfg.evaluation.bootstrap_resamples < 1:
        raise ConfigError("evaluation.bootstrap_resamples must be at least 1")
    if cfg.evaluation.min_rows_per_horizon < 1:
        raise ConfigError("evaluation.min_rows_per_horizon must be at least 1")
    if not 0.0 < cfg.inference.prototype_minority_ratio <= 1.0:
        raise ConfigError(
            "inference.prototype_minority_ratio must lie in (0, 1], got "
            f"{cfg.inference.prototype_minority_ratio}"
        )
    if not 0.0 <= cfg.inference.winsor_quantile < 0.5:
        raise ConfigError("inference.winsor_quantile must lie in [0, 0.5)")
    if cfg.inference.transform_subsample < 1:
        raise ConfigError("inference.transform_subsample must be at least 1")
    if not 0.0 < cfg.inference.ensemble_feature_frac <= 1.0:
        raise ConfigError(
            "inference.ensemble_feature_frac must lie in (0, 1], got "
            f"{cfg.inference.ensemble_feature_frac}"
        )
    if cfg.inference.finetune_clip is not None and cfg.inference.finetune_clip <= 0:
        raise ConfigError("inference.finetune_clip must be null or positive")
    if cfg.inference.max_features_strategy not in ("raise", "subsample"):
        raise ConfigError(
            "inference.max_features_strategy must be 'raise' or 'subsample', got "
            f"{cfg.inference.max_features_strategy!r}"
        )
    if cfg.inference.retrieval_per_class is not None and cfg.inference.retrieval_per_class < 1:
        raise ConfigError("inference.retrieval_per_class must be null or at least 1")
    if cfg.inference.ensemble_svd_components < 0:
        raise ConfigError("inference.ensemble_svd_components must be at least 0")
    if cfg.large_n.method not in ("none", "partition", "boost"):
        raise ConfigError(
            f"large_n.method must be 'none', 'partition' or 'boost', got {cfg.large_n.method!r}"
        )
    if cfg.large_n.boost_weighting not in ("abs", "exp"):
        raise ConfigError(
            f"large_n.boost_weighting must be 'abs' or 'exp', got {cfg.large_n.boost_weighting!r}"
        )
    if not 1 <= cfg.large_n.partition_max_depth <= 3:
        raise ConfigError("large_n.partition_max_depth must lie in [1, 3]")
    if any(n < 1 for n in cfg.context_sweep.context_sizes):
        raise ConfigError("context_sweep.context_sizes must all be at least 1")
    if not cfg.context_sweep.seeds:
        raise ConfigError("context_sweep.seeds must not be empty")
    p = cfg.prior
    if not 0.0 < p.absolute_rate_floor < p.rate_ceiling <= 1.0:
        raise ConfigError(
            f"prior rates must satisfy 0 < absolute_rate_floor ({p.absolute_rate_floor}) < "
            f"rate_ceiling ({p.rate_ceiling}) <= 1"
        )
    valid_transforms = ("none", "rank", "power", "winsor")
    bad = [k for k in p.feature_transforms if k not in valid_transforms]
    if bad:
        raise ConfigError(
            f"prior.feature_transforms has unknown name(s) {bad}; valid names are "
            f"{list(valid_transforms)}"
        )
    if p.max_query_rows is not None and p.max_query_rows < 1:
        raise ConfigError(f"prior.max_query_rows must be at least 1, got {p.max_query_rows}")
    if p.source_weights is not None:
        # names and weights are checked by the prior itself, so a file and a flag are refused
        # by the same code; imported here because it pulls in every prior module
        from fintfm.prior.mixture import REAL_PANEL_PREFIX, SOURCE_NAMES

        if not isinstance(p.source_weights, dict) or not all(
            isinstance(k, str) and isinstance(w, (int, float)) and not isinstance(w, bool)
            for k, w in p.source_weights.items()
        ):
            raise ConfigError(
                "prior.source_weights must be null or a mapping of source name to weight, "
                f"got {p.source_weights!r}"
            )
        unknown = [
            k for k in p.source_weights
            if k not in SOURCE_NAMES and not k.startswith(REAL_PANEL_PREFIX)
        ]
        if unknown:
            raise ConfigError(
                f"prior.source_weights has unknown source(s) {unknown}; valid names are "
                f"{list(SOURCE_NAMES)} or '{REAL_PANEL_PREFIX}<path>'"
            )
    if p.categorical_mode not in ("ordered", "unordered"):
        raise ConfigError(
            f"prior.categorical_mode must be 'ordered' or 'unordered', got {p.categorical_mode!r}"
        )
    if p.missing_mode not in ("mcar", "mixed"):
        raise ConfigError(f"prior.missing_mode must be 'mcar' or 'mixed', got {p.missing_mode!r}")
    if not all(isinstance(b, (list, tuple)) and len(b) == 3 for b in p.feature_ratio_bands):
        raise ConfigError(
            "prior.feature_ratio_bands must be a list of [weight, lo, hi] entries, got "
            f"{p.feature_ratio_bands!r}"
        )
    unit = {
        "financial_gate_prob": p.financial_gate_prob,
        "p_binary": 0.0 if p.p_binary is None else p.p_binary,
        "tie_column_prob": p.tie_column_prob,
        "irrelevant_frac_max": p.irrelevant_frac_max,
        "duplicate_frac_max": p.duplicate_frac_max,
        "label_flip_max": p.label_flip_max,
    }
    off = [k for k, v in unit.items() if not 0.0 <= v <= 1.0]
    if off:
        raise ConfigError(f"prior.{off[0]} must lie in [0, 1], got {unit[off[0]]}")
    t = cfg.train
    if t.amp not in ("none", "bf16", "fp16"):
        # YAML reads a bare `off`/`no` as False, which would otherwise surface as a confusing
        # type error deep inside the training loop
        raise ConfigError(f"train.amp must be 'none', 'bf16' or 'fp16', got {t.amp!r}")
    if t.warmup_steps < 0:
        raise ConfigError(f"train.warmup_steps must be at least 0, got {t.warmup_steps}")
    if not isinstance(t.decay_norm_bias, bool):
        raise ConfigError(f"train.decay_norm_bias must be true or false, got {t.decay_norm_bias!r}")
    if t.muon_lr is not None and t.muon_lr <= 0:
        raise ConfigError(f"train.muon_lr must be positive or null, got {t.muon_lr}")
    if t.muon_scale not in ("rms_match", "keller"):
        raise ConfigError(
            f"train.muon_scale must be 'rms_match' or 'keller', got {t.muon_scale!r}"
        )
    if t.loader not in ("sync", "stream"):
        raise ConfigError(f"train.loader must be 'sync' or 'stream', got {t.loader!r}")
    if t.loader_workers < 0:
        raise ConfigError(f"train.loader_workers must be at least 0, got {t.loader_workers}")
    if t.loader_prefetch < 1:
        raise ConfigError(f"train.loader_prefetch must be at least 1, got {t.loader_prefetch}")
    if not isinstance(t.ddp, bool):
        raise ConfigError(f"train.ddp must be true or false, got {t.ddp!r}")
    if t.schedule not in ("cosine", "wsd"):
        raise ConfigError(f"train.schedule must be 'cosine' or 'wsd', got {t.schedule!r}")
    if not 0.0 <= t.decay_frac <= 1.0:
        raise ConfigError(f"train.decay_frac must be in [0, 1], got {t.decay_frac}")
    if t.lr_min < 0:
        raise ConfigError(f"train.lr_min must be at least 0, got {t.lr_min}")
    if t.eval_batches < 1:
        raise ConfigError(f"train.eval_batches must be at least 1, got {t.eval_batches}")
    m = cfg.model
    # ModelConfig validates the rest when the run builds its model; these two are the ones a
    # hand-edited file most plausibly gets wrong, so they fail at load instead
    if m.value_embedding not in ("mlp", "rbf", "fourier"):
        raise ConfigError(
            f"model.value_embedding must be 'mlp', 'rbf' or 'fourier', got {m.value_embedding!r}"
        )
    if m.n_heads < 1:
        raise ConfigError(f"model.n_heads must be at least 1, got {m.n_heads}")
    if m.attn_impl not in ("masked", "split"):
        raise ConfigError(f"model.attn_impl must be 'masked' or 'split', got {m.attn_impl!r}")


#: Section name to dataclass. Declared explicitly rather than read off ``Config``'s
#: annotations, because ``from __future__ import annotations`` turns ``field.type`` into a
#: string and resolving it back would be indirection for its own sake.
_SECTIONS: dict[str, type] = {
    "inference": InferenceConfig,
    "evaluation": EvaluationConfig,
    "v4finbench": V4FinBenchConfig,
    "context_sweep": ContextSweepConfig,
    "prior": PriorConfigDefaults,
    "train": TrainConfigDefaults,
    "model": ModelConfigDefaults,
    "large_n": LargeNConfig,
}


def load_config(path: str | os.PathLike[str] | None = None) -> Config:
    """Load configuration, layering an override over the packaged default.

    Args:
        path: Override file. Falls back to ``$FINTFM_CONFIG``, then to no override at all.

    Returns:
        A validated :class:`Config` recording the layers it came from.

    Raises:
        FileNotFoundError: If a path was given explicitly and does not exist. A *missing*
            ``FINTFM_CONFIG`` is also an error rather than a silent fallback, because the
            whole point of setting it is that it takes effect.
        ConfigError: If a key is unknown, mistyped, or fails validation.
    """
    data = _read_yaml(DEFAULT_CONFIG_PATH)
    sources = [str(DEFAULT_CONFIG_PATH)]
    override = path if path is not None else os.environ.get(CONFIG_ENV_VAR)
    if override:
        over_path = Path(override)
        data = _deep_merge(data, _read_yaml(over_path))
        sources.append(str(over_path))
    if "sources" in data:
        raise ConfigError("config key 'sources' is set by the loader and cannot be provided")
    unknown = sorted(set(data) - set(_SECTIONS))
    if unknown:
        raise ConfigError(
            f"unknown top-level config section(s) {', '.join(unknown)}; valid sections are "
            f"{', '.join(sorted(_SECTIONS))}"
        )
    cfg = Config(
        **{name: _build(cls, data[name], name) for name, cls in _SECTIONS.items() if name in data},
        sources=tuple(sources),
    )
    _validate(cfg)
    return cfg
