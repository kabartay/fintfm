"""Lever combinations: the refused pairs, and a pairwise covering array of the rest (DR1.5).

Each lever has its own test where it was added; what nothing tested was the levers together.
The first half asserts that every combination known to misbehave is refused with a message
naming both levers (or, for fine-tuning beside an ensemble axis, warned about). The second
half runs two pairwise covering arrays, one over :class:`ModelConfig` levers and one over
:class:`FinancialTFMClassifier` levers, generated deterministically by a greedy cover of
the valid configurations: every pair of lever values that some valid configuration holds
appears in at least one row. Each row must run, give finite outputs and score exactly the
same in chunks of three queries as in one pass; model rows must pass a gradient to every
parameter, classifier rows must return valid probabilities.
"""

from __future__ import annotations

import itertools
import warnings
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import pytest
import torch

from fintfm.inference.classifier import FinancialTFMClassifier
from fintfm.inference.large_n import BoostedTFMClassifier, PartitionedTFMClassifier
from fintfm.inference.regressor import FinancialTFMRegressor
from fintfm.modeling.hazard import CENSORED
from fintfm.modeling.model import FinancialTFM, ModelConfig
from fintfm.modeling.train import TrainConfig, check_model_matches_prior, train
from fintfm.prior import PriorConfig

# --- a deterministic pairwise covering-array generator --------------------------------------


def pairwise_cover(
    levers: Mapping[str, Sequence[Any]], valid: Callable[[dict[str, Any]], bool]
) -> list[dict[str, Any]]:
    """A greedy pairwise covering array over the valid configurations.

    Enumerates the full product of ``levers`` in order, keeps the rows ``valid`` accepts,
    and repeatedly takes the first row covering the most still-uncovered value pairs, until
    every pair held by some valid row is covered. Deterministic: no random draw, ties broken
    by product order.

    Args:
        levers: Lever name to its values, in a fixed order.
        valid: Whether a full configuration is one the code accepts.

    Returns:
        The chosen configurations, in the order picked.
    """
    names = list(levers)
    sizes = [len(levers[n]) for n in names]
    rows = np.array(
        [
            idx
            for idx in itertools.product(*(range(s) for s in sizes))
            if valid({n: levers[n][i] for n, i in zip(names, idx, strict=True)})
        ]
    )
    # one integer id per (lever i, value a, lever j, value b), i < j
    pair_cols, offset = [], 0
    for i, j in itertools.combinations(range(len(names)), 2):
        pair_cols.append(offset + rows[:, i] * sizes[j] + rows[:, j])
        offset += sizes[i] * sizes[j]
    pairs = np.stack(pair_cols, axis=1)
    uncovered = np.zeros(offset, dtype=bool)
    uncovered[np.unique(pairs)] = True
    chosen: list[dict[str, Any]] = []
    while uncovered.any():
        best = int(np.argmax(uncovered[pairs].sum(axis=1)))
        uncovered[pairs[best]] = False
        chosen.append({n: levers[n][i] for n, i in zip(names, rows[best], strict=True)})
    return chosen


def _all_pairs_covered(
    rows: list[dict[str, Any]],
    levers: Mapping[str, Sequence[Any]],
    valid: Callable[[dict[str, Any]], bool],
) -> bool:
    """Check a cover independently of how it was built: every valid pair appears in a row."""
    def key(row: dict[str, Any]) -> set[tuple[str, str, str, str]]:
        return {(a, repr(row[a]), b, repr(row[b])) for a, b in itertools.combinations(levers, 2)}

    needed: set[tuple[str, str, str, str]] = set()
    for values in itertools.product(*levers.values()):
        row = dict(zip(levers, values, strict=True))
        if valid(row):
            needed |= key(row)
    have: set[tuple[str, str, str, str]] = set()
    for row in rows:
        have |= key(row)
    return needed <= have


def test_the_generator_covers_every_valid_pair_and_only_valid_rows():
    levers = {"a": (0, 1, 2), "b": (0, 1), "c": (0, 1), "d": ("x", "y")}

    def valid(r: dict[str, Any]) -> bool:
        return not (r["a"] == 2 and r["d"] == "y")

    rows = pairwise_cover(levers, valid)
    assert all(valid(r) for r in rows)
    assert _all_pairs_covered(rows, levers, valid)
    assert rows == pairwise_cover(levers, valid)  # deterministic
    assert len(rows) < 3 * 2 * 2 * 2  # a cover, not the product


# --- helpers ---------------------------------------------------------------------------------


def _raises_naming(fn: Callable[[], object], *levers: str) -> None:
    """``fn`` raises ValueError and the message names every lever in ``levers``."""
    with pytest.raises(ValueError) as info:
        fn()
    for lever in levers:
        assert lever in str(info.value), f"{lever!r} missing from: {info.value}"


def _clf_model(**extra: Any) -> FinancialTFM:
    """A tiny seeded classifier checkpoint, recorded as trained for classification."""
    torch.manual_seed(0)
    cfg = ModelConfig(max_features=8, max_classes=3, d_cell=8, d_model=16, n_heads=2,
                      n_col_layers=1, n_layers=1, d_ff=32, **extra)
    model = FinancialTFM(cfg).eval()
    if cfg.head_type == "binned" and cfg.n_horizons is None:
        model.trained_objectives = ("classification",)
    return model


def _table(n: int = 120, n_feat: int = 5, n_classes: int = 2, seed: int = 0):
    """Training rows, labels with a learnable rule, and nine query rows."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n + 9, n_feat)).astype(np.float32)
    X[3, 1] = np.nan
    score = X[:, 0] + 0.5 * X[:, n_feat - 1]
    y = (
        (score > 0.6).astype(np.int64)
        if n_classes == 2
        else np.digitize(score, [-0.5, 0.5]).astype(np.int64)
    )
    return X[:n], y[:n], X[n:]


# --- refused combinations ------------------------------------------------------------------


def test_finetuning_a_quantile_checkpoint_is_refused():
    X, y, _ = _table()
    model = _clf_model(head_type="quantile", n_quantiles=5)
    _raises_naming(
        lambda: FinancialTFMClassifier(model, max_context=40, finetune_steps=1).fit(X, y),
        "finetune_steps", "head_type",
    )


def test_finetuning_a_hazard_checkpoint_is_refused():
    """Its classification head was never trained, and fine-tuning reaches the encoder."""
    X, y, _ = _table()
    model = _clf_model(n_horizons=3)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    _raises_naming(
        lambda: FinancialTFMClassifier(model, max_context=40, finetune_steps=1).fit(X, y),
        "finetune_steps", "n_horizons",
    )
    # recorded as survival-only: refused on the record as well
    model.trained_objectives = ("survival",)
    _raises_naming(
        lambda: FinancialTFMClassifier(model, max_context=40, finetune_steps=1).fit(X, y),
        "finetune_steps", "trained_objectives",
    )
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    # without fine-tuning the hazard checkpoint still fits, for its term structure
    FinancialTFMClassifier(model, max_context=40).fit(X, y)
    # and a hazard checkpoint that did train classification may be fine-tuned
    model.trained_objectives = ("classification", "survival")
    FinancialTFMClassifier(
        model, max_context=40, finetune_steps=1, finetune_query=8
    ).fit(X, y)


def test_class_shift_with_label_swap_is_refused_on_a_binary_task_only():
    X, y, _ = _table()
    model = _clf_model()
    kw = {"max_context": 40, "n_ensemble": 2, "ensemble_class_shift": True,
          "ensemble_label_swap": True}
    _raises_naming(
        lambda: FinancialTFMClassifier(model, **kw).fit(X, y),
        "ensemble_class_shift", "ensemble_label_swap",
    )
    # on three classes the swap is inert and the shift acts, so the pair is allowed
    X3, y3, Xq3 = _table(n_classes=3)
    p = FinancialTFMClassifier(model, **kw).fit(X3, y3).predict_proba(Xq3)
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-5)


def test_svd_components_with_no_room_left_are_refused():
    """Every member already at the checkpoint's width would silently carry none."""
    model = _clf_model()  # max_features 8
    Xw, yw, _ = _table(n_feat=11)
    _raises_naming(
        lambda: FinancialTFMClassifier(
            model, max_context=40, n_ensemble=2, ensemble_svd_components=2,
            max_features_strategy="subsample",
        ).fit(Xw, yw),
        "ensemble_svd_components", "max_features_strategy",
    )
    X8, y8, _ = _table(n_feat=8)
    _raises_naming(
        lambda: FinancialTFMClassifier(
            model, max_context=40, n_ensemble=2, ensemble_svd_components=2
        ).fit(X8, y8),
        "ensemble_svd_components", "max_features",
    )
    # a narrower member leaves room again: 11 * 0.6 rounds to 7 of 8
    clf = FinancialTFMClassifier(
        model, max_context=40, n_ensemble=2, ensemble_svd_components=2,
        ensemble_feature_frac=0.6, max_features_strategy="subsample",
    ).fit(Xw, yw)
    member = clf._clone_with_seed(1, Xw[:, clf._member_columns[1]], svd_components=2)
    assert member.svd_components_ == 1


def test_binned_regressor_refuses_class_shift_on_ordinal_bins(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 5)).astype(np.float32)
    y = X[:, 0] + 0.1 * rng.normal(size=120)
    params = {"max_context": 40, "n_ensemble": 2, "ensemble_class_shift": True}
    _raises_naming(
        lambda: FinancialTFMRegressor(_clf_model(), n_bins=3, classifier_params=params).fit(X, y),
        "ensemble_class_shift", "permute_classes",
    )
    # two bins are a label swap, which is allowed
    FinancialTFMRegressor(_clf_model(), n_bins=2, classifier_params=params).fit(X, y)
    # a checkpoint whose recorded prior permuted classes is allowed, from its file
    path = tmp_path / "permuted.pt"
    _clf_model().save(str(path), provenance={"prior_config": {"permute_classes": True}})
    reg = FinancialTFMRegressor(str(path), n_bins=3, classifier_params=params).fit(X, y)
    assert np.isfinite(reg.predict(X[:9])).all()
    plain = tmp_path / "plain.pt"
    _clf_model().save(str(plain), provenance={"prior_config": {"permute_classes": False}})
    _raises_naming(
        lambda: FinancialTFMRegressor(str(plain), n_bins=3, classifier_params=params).fit(X, y),
        "ensemble_class_shift", "permute_classes",
    )


@pytest.mark.parametrize(
    ("prior_change", "model_change", "field"),
    [
        ({"max_features": 10}, {}, "max_features"),
        ({"max_classes": 4}, {"max_classes": 3}, "max_classes"),
        ({"head_type": "quantile", "p_regression": 1.0}, {}, "head_type"),
        ({"n_horizons": 3, "p_financial": 1.0}, {}, "n_horizons"),
        ({}, {"n_horizons": 3}, "n_horizons"),
    ],
)
def test_train_refuses_a_model_that_disagrees_with_its_prior(
    prior_change, model_change, field, tmp_path
):
    """Both configurations are named, and nothing is written: it fails before any step."""
    model_cfg = ModelConfig(**{"max_features": 8, "max_classes": 2, "d_cell": 8, "d_model": 16,
                               "n_heads": 2, "n_col_layers": 1, "n_layers": 1, "d_ff": 32,
                               **model_change})
    prior_cfg = PriorConfig(**{"max_features": 8, "max_classes": 2, "n_rows": 32,
                               **prior_change})
    out = tmp_path / "m.pt"
    _raises_naming(
        lambda: train(model_cfg, prior_cfg, TrainConfig(steps=1, batch_size=2), str(out)),
        f"PriorConfig.{field}", f"ModelConfig.{field}",
    )
    assert not out.exists()


def test_train_accepts_a_prior_narrower_than_the_model():
    """Padding is masked, so fewer features or classes than the model holds is legitimate."""
    check_model_matches_prior(
        ModelConfig(max_features=8, max_classes=3),
        PriorConfig(max_features=6, max_classes=2),
    )


def test_the_model_refuses_a_table_wider_than_max_features():
    model = _clf_model()
    X = torch.randn(1, 12, 9)
    y = torch.randint(0, 2, (1, 12))
    for call in (
        lambda: model(X, y, 8),
        lambda: model.loss(X, y, 8, torch.tensor([2])),
    ):
        _raises_naming(call, "max_features")
    hazard = _clf_model(n_horizons=3)
    _raises_naming(lambda: hazard.term_structure(X, y, 8), "max_features")
    period = torch.full((1, 12), CENSORED)
    _raises_naming(lambda: hazard.survival_loss(X, y, period, 8), "max_features")
    # exactly max_features wide is accepted
    assert torch.isfinite(model(X[..., :8], y, 8)[:, 8:]).all()


@pytest.mark.parametrize(
    ("extra", "lever"),
    [
        ({"context_strategy": "balanced"}, "context_strategy"),
        ({"ensemble_label_swap": True}, "ensemble_label_swap"),
        ({"n_ensemble": 2, "ensemble_class_shift": True}, "ensemble_class_shift"),
        ({"n_ensemble": 2, "ensemble_transforms": ("rank", "squash")}, "ensemble_transforms"),
        ({"n_ensemble": 2, "ensemble_svd_components": 2}, "ensemble_svd_components"),
        ({"n_ensemble": 2, "ensemble_feature_frac": 0.6}, "ensemble_feature_frac"),
    ],
)
def test_finetuning_beside_an_ensemble_axis_warns_naming_both(extra, lever):
    X, y, _ = _table(n=60)
    kw = {"max_context": 30, "finetune_steps": 1, "finetune_query": 8}
    with pytest.warns(UserWarning) as record:
        FinancialTFMClassifier(_clf_model(), **kw, **extra).fit(X, y)
    message = " ".join(str(w.message) for w in record)
    assert "finetune_steps" in message and lever in message
    if lever == "ensemble_class_shift":
        # on a binary task the shift is a label swap, and the warning says so
        assert "label swap" in message
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        FinancialTFMClassifier(_clf_model(), **kw).fit(X, y)
        FinancialTFMClassifier(_clf_model(), max_context=30, **extra).fit(X, y)


@pytest.mark.parametrize("wrapper", ["partition", "boost"])
def test_the_large_n_wrappers_refuse_before_fitting_anything(wrapper):
    X, y, _ = _table()
    inner = FinancialTFMClassifier(
        _clf_model(), max_context=40, n_ensemble=2, ensemble_class_shift=True,
        ensemble_label_swap=True,
    )
    est = (
        PartitionedTFMClassifier(inner, max_depth=1, min_leaf_rows=20)
        if wrapper == "partition"
        else BoostedTFMClassifier(inner, n_rounds=2, context_rows=40)
    )
    _raises_naming(lambda: est.fit(X, y), "ensemble_class_shift", "ensemble_label_swap")
    assert not hasattr(est, "trees_") and not hasattr(est, "estimators_")


# --- covering array: model levers ------------------------------------------------------------

_MODEL_LEVERS: dict[str, tuple[Any, ...]] = {
    "value_embedding": ("mlp", "rbf", "fourier"),
    "value_ecdf": (False, True),
    "attn_length_scale": (False, True),
    "attn_impl": ("masked", "split"),
    "n_summary_tokens": (1, 2, 3),
    "n_cell_heads": (None, 1, 4),
    "n_cell_blocks": (0, 1, 2),
    "cell_labels": (False, True),
    "pooling": ("meanmax", "attention"),
    # none, a hazard head, or a hazard head whose context carries periods
    "hazard": ("none", "plain", "period"),
}


def _model_valid(r: dict[str, Any]) -> bool:
    """ModelConfig's own refusals, so every row is one it builds."""
    return not (r["n_summary_tokens"] > 1 and r["pooling"] == "attention")


_MODEL_ROWS = pairwise_cover(_MODEL_LEVERS, _model_valid)


def test_model_cover_is_pairwise_complete():
    assert _all_pairs_covered(_MODEL_ROWS, _MODEL_LEVERS, _model_valid)
    assert 9 <= len(_MODEL_ROWS) <= 25


def _build_model(row: dict[str, Any]) -> FinancialTFM:
    """The row's tiny model; ``attn_length_ref`` is set near the test context length."""
    torch.manual_seed(0)
    levers = {k: v for k, v in row.items() if k != "hazard"}
    return FinancialTFM(
        ModelConfig(
            max_features=6, max_classes=3, d_cell=8, d_model=16, n_heads=2, n_col_layers=1,
            n_layers=2, d_ff=32, attn_length_ref=8,
            n_horizons=None if row["hazard"] == "none" else 3,
            hazard_period_context=row["hazard"] == "period", **levers,
        )
    )


@pytest.mark.parametrize("row", _MODEL_ROWS, ids=lambda r: ",".join(f"{v}" for v in r.values()))
def test_model_cover_row_runs_finite_with_gradients_and_exact_chunks(row):
    model = _build_model(row)
    n_ctx, n_q = 12, 9
    g = torch.Generator().manual_seed(1)
    X = torch.randn(2, n_ctx + n_q, 6, generator=g)
    X[0, 2, 1] = float("nan")
    X[:, :, 5] = float("nan")  # a padded column
    period = torch.randint(-1, 3, (2, n_ctx + n_q), generator=g)
    y = (period != CENSORED).long()

    # gradients reach every parameter: classification loss, plus survival on a hazard head
    model.train()
    loss = model.loss(X, y, n_ctx, torch.tensor([2, 2]))
    if model.hazard is not None:
        loss = loss + model.survival_loss(X, y, period, n_ctx)
    assert torch.isfinite(loss)
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"no gradient reached {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite gradient in {name}"
    assert sum(float(p.grad.abs().sum()) for p in model.parameters()) > 0

    # scoring: finite, and the same in chunks of three queries as in one pass
    model.eval()
    Xc, Xq = X[0, :n_ctx].numpy(), X[0, n_ctx:].numpy()
    yc, pc = y[0, :n_ctx].numpy(), period[0, :n_ctx].numpy()
    outputs = ["logits"] + ([] if model.hazard is None else ["term_structure"])
    for output in outputs:
        def score(q: np.ndarray, output: str = output) -> np.ndarray:
            return model.predict_context(
                Xc, yc, q, n_classes=2, column_id_seed=3, output=output, period=pc
            ).numpy()

        whole = score(Xq)
        chunked = np.concatenate([score(Xq[i : i + 3]) for i in range(0, n_q, 3)])
        assert np.isfinite(whole[:, :2] if output == "logits" else whole).all()
        np.testing.assert_allclose(chunked, whole, rtol=1e-5, atol=1e-6)


# --- covering array: classifier levers -------------------------------------------------------

_CLF_LEVERS: dict[str, tuple[Any, ...]] = {
    "context_strategy": ("uniform", "balanced", "hybrid", "retrieval", "prototype"),
    "n_ensemble": (1, 3),
    "ensemble_label_swap": (False, True),
    "ensemble_class_shift": (False, True),
    "ensemble_transforms": (None, ("rank", "squash")),
    "ensemble_svd_components": (0, 2),
    "ensemble_feature_frac": (1.0, 0.6),
    # the table's width against the checkpoint's 8 features, with the strategy for it
    "width": ("narrow-raise", "narrow-subsample", "wide-subsample"),
    "finetune_steps": (0, 2),
    "retrieval_per_class": (None, 4),
    "large_n": ("none", "partition", "boost"),
    "n_classes": (2, 3),
}


def _clf_valid(r: dict[str, Any]) -> bool:
    """The combinations the estimators accept; each refusal here is asserted above."""
    wide = r["width"] == "wide-subsample"
    if r["ensemble_svd_components"] and (
        r["n_ensemble"] < 2
        or r["context_strategy"] == "retrieval"
        or (wide and r["ensemble_feature_frac"] == 1.0)  # no member would have room
    ):
        return False
    if r["retrieval_per_class"] is not None and r["context_strategy"] != "retrieval":
        return False
    if wide and r["finetune_steps"]:
        return False
    if r["ensemble_class_shift"] and r["ensemble_label_swap"] and r["n_classes"] == 2:
        return False
    return not (r["large_n"] == "boost" and r["n_classes"] != 2)


_CLF_ROWS = pairwise_cover(_CLF_LEVERS, _clf_valid)


def test_classifier_cover_is_pairwise_complete():
    assert _all_pairs_covered(_CLF_ROWS, _CLF_LEVERS, _clf_valid)
    assert 15 <= len(_CLF_ROWS) <= 35


@pytest.fixture(scope="module")
def clf_model() -> FinancialTFM:
    """One checkpoint for every classifier row; estimators never modify it."""
    return _clf_model()


def _build_estimator(row: dict[str, Any], model: FinancialTFM):
    """The row's estimator: a classifier, or a large-n wrapper around one."""
    params = {k: v for k, v in row.items() if k not in ("width", "large_n", "n_classes")}
    clf = FinancialTFMClassifier(
        model, max_context=40, query_chunk=10_000, retrieval_groups=4, finetune_query=8,
        max_features_strategy=row["width"].split("-")[1], **params,
    )
    if row["large_n"] == "partition":
        return PartitionedTFMClassifier(clf, max_depth=1, min_leaf_rows=30)
    if row["large_n"] == "boost":
        return BoostedTFMClassifier(clf, n_rounds=2, context_rows=40)
    return clf


def _set_query_chunk(est: Any, n: int) -> None:
    """Set ``query_chunk`` on a fitted estimator and every classifier inside it."""
    if isinstance(est, PartitionedTFMClassifier):
        inner = [est.global_estimator_] + [
            e for leaves in est.leaf_estimators_ for e in leaves.values() if e is not None
        ]
    elif isinstance(est, BoostedTFMClassifier):
        inner = list(est.estimators_)
    else:
        inner = [est]
    for clf in inner:
        clf.query_chunk = n


@pytest.mark.parametrize("row", _CLF_ROWS, ids=lambda r: ",".join(f"{v}" for v in r.values()))
def test_classifier_cover_row_runs_with_valid_probabilities_and_exact_chunks(row, clf_model):
    n_feat = 11 if row["width"] == "wide-subsample" else 5
    X, y, Xq = _table(n=120, n_feat=n_feat, n_classes=row["n_classes"])
    est = _build_estimator(row, clf_model)
    with warnings.catch_warnings():
        # fine-tuning beside an ensemble axis warns by design; asserted above
        warnings.simplefilter("ignore", UserWarning)
        est.fit(X, y)
    whole = est.predict_proba(Xq)
    assert whole.shape == (9, row["n_classes"])
    assert np.isfinite(whole).all()
    assert (whole >= 0).all() and (whole <= 1).all()
    np.testing.assert_allclose(whole.sum(axis=1), 1.0, atol=1e-5)
    if row["context_strategy"] != "retrieval":
        # retrieval groups queries per chunk, so its chunking is documented as inexact
        _set_query_chunk(est, 3)
        np.testing.assert_allclose(est.predict_proba(Xq), whole, rtol=1e-5, atol=1e-6)
