"""The mixture sampler's own contracts (`prior/mixture.py`), as distinct from any one prior's.

What is checked here fails silently when it breaks: a mixture whose realised source shares
drift from its configured weights still trains, still logs a falling loss, and only shows up
as a comparison that measured two changes at once (audit AU1.13).
"""

from dataclasses import replace

import numpy as np
import pytest
import torch

from fintfm.prior.base import Task, TaskBatch, ensure_context_classes
from fintfm.prior.mixture import (
    SOURCE_NAMES,
    PriorConfig,
    _draw_source,
    _sample_tasks_reusing_graphs,
    _sequential_sources,
    parse_source_weights,
    sample_batch,
    sample_task,
    source_shares,
)

#: Every synthetic source switched on at once, so the fingerprint below exercises each branch
#: of the source draw. Real sources are left out: their samplers are FinTFM-R only and are
#: allowed to change.
_ALL_SYNTHETIC = PriorConfig(
    max_features=12, max_classes=5, n_rows=48, p_tree=0.15, p_regression=0.1,
    p_trivial=0.1, p_crossed=0.1, p_task_family=0.15, p_financial=0.5,
)


def test_reuse_one_draws_the_same_stream_as_before_the_source_split():
    # AU1.13 split sample_task into a source draw and a dispatch. At scm_reuse_graph=1 that
    # must be invisible: the same sources, shapes and class counts in the same order, and the
    # generator left at the same position. Recorded from the code before the split; the
    # trailing rng.random() pins the number of draws consumed, which is what byte-identity of
    # every later batch rests on.
    rng = np.random.default_rng(123)
    got = [(t.source, t.X.shape[1], t.n_classes) for t in (
        sample_task(rng, _ALL_SYNTHETIC) for _ in range(40)
    )]
    assert got == [
        ("scm-regression", 11, 5), ("scm-regression", 5, 5), ("tree", 3, 2), ("tree", 10, 4),
        ("tree", 4, 2), ("scm", 5, 4), ("trivial", 6, 2), ("financial", 9, 2),
        ("trivial", 6, 2), ("family:sparse", 12, 2), ("financial", 9, 2), ("tree", 9, 2),
        ("scm", 7, 3), ("tree", 3, 2), ("scm", 5, 5), ("financial", 7, 2),
        ("family:linear", 12, 2), ("scm", 11, 2), ("financial", 11, 2),
        ("family:composition_composed", 12, 2), ("trivial", 4, 2), ("family:xor", 12, 2),
        ("financial", 12, 2), ("scm", 10, 5), ("financial", 6, 2), ("scm", 8, 4),
        ("financial", 12, 2), ("scm", 3, 3), ("scm_x_fin_y", 12, 2), ("scm", 6, 3),
        ("tree", 4, 3), ("family:xor", 12, 2), ("financial", 11, 2), ("tree", 5, 2),
        ("scm", 3, 4), ("scm_x_fin_y", 2, 2), ("scm", 11, 5), ("family:interaction", 12, 2),
        ("scm", 3, 2), ("scm", 12, 2),
    ]
    assert rng.random() == 0.10348447068500588

    # DR2.1: source_weights=None, stated explicitly, is the same sequential stream
    for cfg in (_ALL_SYNTHETIC, replace(_ALL_SYNTHETIC, source_weights=None)):
        rng = np.random.default_rng(7)
        batches = [sample_batch(rng, cfg, batch_size=4) for _ in range(3)]
        assert [(b.n_ctx, b.n_classes.tolist()) for b in batches] == [
            (19, [4, 3, 3, 2]), (43, [2, 5, 2, 5]), (29, [2, 5, 4, 3]),
        ]
        assert rng.random() == 0.0627829080239598


def _kind(source: str) -> str:
    """Collapse the task-family sources to one name, since they are one mixture weight."""
    return "family" if source.startswith("family:") else source


def test_source_shares_agree_at_reuse_one_and_four():
    # AU1.13: at p_financial 0.7, p_tree 0.2, p_task_family 0.2 the SCM share was measured at
    # 0.198 at reuse 1 against 0.558 at reuse 4. Each SCM task's share must be the mixture's
    # (0.8 x 0.8 x 0.3 = 0.192) at both values, within binomial error.
    base = {"max_features": 8, "max_classes": 4, "n_rows": 32,
            "p_financial": 0.7, "p_tree": 0.2, "p_task_family": 0.2}
    n_batches, batch_size = 60, 8
    n = n_batches * batch_size
    shares = {}
    for reuse in (1, 4):
        cfg = PriorConfig(**base, scm_reuse_graph=reuse)
        rng = np.random.default_rng(11)
        sources: list[str] = []
        for _ in range(n_batches):
            if reuse == 1:
                tasks = [sample_task(rng, cfg) for _ in range(batch_size)]
            else:
                tasks = _sample_tasks_reusing_graphs(rng, cfg, cfg.n_rows, batch_size)
            assert len(tasks) == batch_size
            sources += [_kind(t.source) for t in tasks]
        shares[reuse] = {k: sources.count(k) / n for k in ("scm", "tree", "family", "financial")}

    expected = {"scm": 0.8 * 0.8 * 0.3, "tree": 0.2, "family": 0.8 * 0.2,
                "financial": 0.8 * 0.8 * 0.7}
    for kind, p in expected.items():
        sd = np.sqrt(p * (1 - p) / n)
        for reuse in (1, 4):
            assert abs(shares[reuse][kind] - p) < 4 * sd, (kind, reuse, shares[reuse][kind], p)
        # and the two arms against each other, two-sample
        assert abs(shares[1][kind] - shares[4][kind]) < 4 * np.sqrt(2) * sd, (kind, shares)


def _query_only_classes(batch: TaskBatch) -> int:
    """Count tasks in a batch with a query class that never appears in the context."""
    bad = 0
    y = batch.y.numpy()
    for i in range(y.shape[0]):
        if set(y[i, batch.n_ctx :].tolist()) - set(y[i, : batch.n_ctx].tolist()):
            bad += 1
    return bad


def test_every_query_class_is_in_the_context_with_a_class_floor():
    # Audit AU1.16: random cut-points leave rare classes that can land only in the query. Over
    # 1,000 256-row SCM and tree tasks the floor must leave none; without it some exist, so
    # the check is not vacuous.
    base = {"max_features": 8, "max_classes": 10, "n_rows": 256, "p_financial": 0.0,
            "p_tree": 0.5}
    counts = {}
    for floor in (0, 1):
        cfg = PriorConfig(**base, min_ctx_per_class=floor)
        rng = np.random.default_rng(5)
        counts[floor] = sum(_query_only_classes(sample_batch(rng, cfg, 8)) for _ in range(125))
    assert counts[0] > 0
    assert counts[1] == 0


def test_class_floor_moves_whole_rows_and_keeps_the_class_balance():
    n = 40
    y = np.zeros(n, dtype=np.int64)
    y[-3:] = [1, 2, 2]  # classes 1 and 2 only in the last rows, so only in the query
    X = np.arange(n, dtype=np.float32)[:, None].repeat(2, axis=1)
    task = Task(X=X, y=y, n_classes=3, is_categorical=np.zeros(2, dtype=bool))
    out = ensure_context_classes(np.random.default_rng(0), task, n_ctx=20, min_per_class=2)
    assert np.bincount(out.y).tolist() == np.bincount(y).tolist()
    np.testing.assert_array_equal(y[out.X[:, 0].astype(int)], out.y)  # rows moved whole
    ctx = out.y[:20]
    assert (ctx == 1).sum() == 1  # class 1 has one row, and it is now context
    assert (ctx == 2).sum() == 2
    assert 1 not in out.y[20:].tolist()


def test_class_floor_zero_changes_nothing():
    cfg = PriorConfig(max_features=8, max_classes=6, n_rows=64, p_financial=0.3)
    a = sample_batch(np.random.default_rng(3), cfg, 4)
    b = sample_batch(np.random.default_rng(3), replace(cfg, min_ctx_per_class=0), 4)
    assert torch.equal(a.y, b.y)
    assert torch.equal(torch.nan_to_num(a.X), torch.nan_to_num(b.X))


# --- DR2.1: normalised source weights ------------------------------------------------------

#: Unnormalised (they sum to 2) and written out of draw order, so the test covers both.
_WEIGHTS = {"scm": 0.8, "tree": 0.5, "financial": 0.2, "task_family": 0.2, "crossed": 0.1,
            "trivial": 0.1, "regression": 0.1}


def test_weighted_draw_realises_the_normalised_weights_at_10000_draws():
    """Each source's share over 10,000 draws lies within binomial error (4 sd) of its weight
    over the total, and every draw costs exactly one ``rng.random()``."""
    cfg = PriorConfig(source_weights=_WEIGHTS)
    n = 10_000
    rng = np.random.default_rng(3)
    drawn = [_draw_source(rng, cfg)[0] for _ in range(n)]
    shares = source_shares(cfg)
    assert shares["mode"] == "weighted"
    assert list(shares["realised"]) == ["tree", "regression", "trivial", "crossed",
                                        "task_family", "financial", "scm"]
    for name, w in _WEIGHTS.items():
        p = w / 2.0
        assert shares["realised"][name] == pytest.approx(p)
        assert abs(drawn.count(name) / n - p) < 4 * np.sqrt(p * (1 - p) / n), name
    position = np.random.default_rng(3)
    position.random(n)
    assert rng.random() == position.random()


def test_weighted_draw_ignores_mapping_order_and_never_draws_a_zero_weight():
    a = PriorConfig(source_weights=_WEIGHTS)
    b = PriorConfig(source_weights=dict(reversed(list(_WEIGHTS.items()))))
    ra, rb = np.random.default_rng(1), np.random.default_rng(1)
    assert [_draw_source(ra, a) for _ in range(500)] == [_draw_source(rb, b) for _ in range(500)]
    only = PriorConfig(source_weights={"tree": 0.0, "financial": 1.0, "scm": 0.0})
    rng = np.random.default_rng(0)
    assert {_draw_source(rng, only)[0] for _ in range(2000)} == {"financial"}
    panel = PriorConfig(source_weights={"real_panel:data/x=y.parquet": 1.0})
    assert _draw_source(np.random.default_rng(0), panel) == ("real_panel", "data/x=y.parquet")


def test_weighted_shares_agree_at_reuse_one_and_four():
    """Graph reuse groups SCM tasks after every source is drawn, so in weighted mode too the
    task shares follow the weights at reuse 1 and 4 alike."""
    weights = {"tree": 0.2, "task_family": 0.2, "financial": 0.2, "scm": 0.4}
    base = {"max_features": 8, "max_classes": 4, "n_rows": 32, "source_weights": weights}
    n_batches, batch_size = 60, 8
    n = n_batches * batch_size
    shares = {}
    for reuse in (1, 4):
        cfg = PriorConfig(**base, scm_reuse_graph=reuse)
        rng = np.random.default_rng(11)
        sources: list[str] = []
        for _ in range(n_batches):
            if reuse == 1:
                tasks = [sample_task(rng, cfg) for _ in range(batch_size)]
            else:
                tasks = _sample_tasks_reusing_graphs(rng, cfg, cfg.n_rows, batch_size)
            assert len(tasks) == batch_size
            sources += [_kind(t.source) for t in tasks]
        shares[reuse] = {k: sources.count(k) / n for k in ("scm", "tree", "family", "financial")}
    for kind, p in {"scm": 0.4, "tree": 0.2, "family": 0.2, "financial": 0.2}.items():
        sd = np.sqrt(p * (1 - p) / n)
        for reuse in (1, 4):
            assert abs(shares[reuse][kind] - p) < 4 * sd, (kind, reuse, shares[reuse][kind], p)
    batch = sample_batch(np.random.default_rng(0), PriorConfig(**base, scm_reuse_graph=4), 8)
    assert batch.X.shape[0] == 8


def test_weighted_names_follow_the_sequential_chain():
    """One list of names: the weighted draw's order is the chain's, then financial and scm."""
    assert SOURCE_NAMES == (*(n for n, _ in _sequential_sources(PriorConfig())), "financial",
                            "scm")


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"source_weights": {}}, "empty"),
        ({"source_weights": {"trees": 1.0}}, "unknown source"),
        ({"source_weights": {"real_panel:": 1.0}}, "unknown source"),
        ({"source_weights": {"tree": -0.1, "scm": 1.0}}, "non-negative"),
        ({"source_weights": {"tree": float("nan")}}, "non-negative"),
        ({"source_weights": {"tree": 0.0, "scm": 0.0}}, "sum to zero"),
        ({"source_weights": {"tree": 1.0}, "p_tree": 0.3}, "p_tree"),
        ({"source_weights": {"tree": 1.0}, "p_financial": 1.0}, "p_financial"),
        ({"source_weights": {"tree": 1.0}, "real_panels": "a.parquet=0.5"}, "real_panels"),
        ({"source_weights": {"financial": 1.0, "scm": 0.1}, "n_horizons": 3}, "time axis"),
    ],
)
def test_bad_source_weights_are_refused_before_any_draw(overrides, match):
    cfg = PriorConfig(max_features=8, n_rows=32, **overrides)
    rng = np.random.default_rng(0)
    for call in (lambda: sample_batch(rng, cfg, 2), lambda: source_shares(cfg),
                 lambda: sample_task(rng, cfg)):
        with pytest.raises(ValueError, match=match):
            call()


def test_horizon_run_may_weight_the_sources_that_carry_a_period():
    cfg = PriorConfig(max_features=8, n_rows=64, n_horizons=3,
                      source_weights={"financial": 1.0, "scm": 0.0})
    batch = sample_batch(np.random.default_rng(0), cfg, 2)
    assert batch.period is not None


def test_parse_source_weights():
    assert parse_source_weights("") is None and parse_source_weights(" , ") is None
    assert parse_source_weights("tree=0.25, scm=0.4,real_panel:a=b.parquet=1") == {
        "tree": 0.25, "scm": 0.4, "real_panel:a=b.parquet": 1.0,
    }
    for spec, match in (("tree", "source=weight"), ("=1", "source=weight"),
                        ("tree=x", "not a number"), ("tree=1,tree=2", "twice")):
        with pytest.raises(ValueError, match=match):
            parse_source_weights(spec)
