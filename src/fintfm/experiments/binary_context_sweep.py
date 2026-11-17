"""Context strategies on the binary path, per cell, with Holm-adjusted paired comparisons.

Openspec ``revisit-context-strategy`` tasks 32.3, 32.4 and 32.6. ``fintfm-ctxsweep`` measures
the strategies on the survival path; ``fintfm-bench --credit`` compares three of them once, on
one split, and prints the result without keeping it. This harness is the binary-path
counterpart that keeps everything: for each panel, context size, seed and strategy it fits the
checkpoint on a stratified train part, scores the held-out part, and stores AUC, average
precision, ECE, the **slice default rate** (the held-out part's positive rate) and the
context's own positive share. Strategies are compared on the **same held-out rows** within a
(panel, context size, seed), so every pair is paired.

What it records, and why:

* **Average precision beside ROC-AUC.** At a 3% to 5% base rate AUC compresses the usable
  range (``CLAUDE.md``, "Read average precision first"); both go in every cell.
* **The slice default rate and the pool.** ``max_context`` is a fraction of the pool it
  samples from, so every cell carries ``n_train``. A size at or above ``n_train`` gives every
  strategy the same rows and is skipped with that reason, never scored as a tie.
* **Holm-adjusted p-values** over every strategy pair in the record, one family per metric
  (AUC, AP), using :func:`~fintfm.evaluation.metrics.paired_auc_difference` and
  :func:`~fintfm.evaluation.metrics.holm_adjusted_p`. Rows are drawn independently here (no
  company ids exist on the UCI panels), so the bootstrap resamples rows.
* **Coverage** of cells attempted, scored and skipped, with a reason per skip.

Verification against held-out labels of observed bankruptcies is validation of discrimination
on those panels, not coverage against a solver. A SMOKE-TEST run of this command proves the
pipeline executes and says nothing about the strategies.
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

from fintfm.evaluation.coverage import Coverage
from fintfm.evaluation.datasets import (
    CreditDataset,
    load_polish_bankruptcy,
    load_taiwan_bankruptcy,
)
from fintfm.evaluation.metrics import evaluate_binary, holm_adjusted_p, paired_auc_difference
from fintfm.inference.classifier import FinancialTFMClassifier
from fintfm.modeling.model import FinancialTFM

#: Strategies of §35 (balanced, hybrid, uniform, retrieval) plus ``prototype`` (task 32.6).
DEFAULT_STRATEGIES = ("uniform", "balanced", "hybrid", "prototype", "retrieval")
#: Context sizes; §35 used 2,000. Smaller sizes make the rebalancing more or less extreme.
DEFAULT_SIZES = (500, 1000, 2000)
DEFAULT_SEEDS = (0, 1, 2)
#: Panel names accepted by ``--panels``: Polish at 1, 3 and 5 years (as §35) and Taiwan.
PANEL_NAMES = ("polish1", "polish3", "polish5", "taiwan")
OUT_FILE = "binary_context_sweep.json"
METRICS = ("auc", "ap")
#: Holm floors at or above this refuse to run (see :func:`check_bootstrap_resolution`).
MAX_HOLM_FLOOR = 0.005
#: Default bootstrap resamples. The pre-registered family of task 32.7 is ``uniform`` against
#: ``prototype`` and ``retrieval`` (2 pairs) over 4 panels x 3 sizes x 3 seeds = 72
#: comparisons per metric, and a Holm floor below :data:`MAX_HOLM_FLOOR` needs
#: ``n_boot >= 200 * 72 = 14,400``. 20,000 clears that with margin (floor 0.0036) and
#: also covers up to 100 comparisons per metric. Bootstrap cost is linear in this number
#: (roughly an hour for one Polish panel at the default), so a larger family needs a
#: larger value and a smaller one may lower it.
DEFAULT_N_BOOT = 20000


def load_panels(names: Sequence[str]) -> list[CreditDataset]:
    """Load the named binary panels.

    Args:
        names: Entries of :data:`PANEL_NAMES`.

    Returns:
        The datasets, in the order given.

    Raises:
        ValueError: For an unknown name.
    """
    out = []
    for n in names:
        if n == "taiwan":
            out.append(load_taiwan_bankruptcy())
        elif n.startswith("polish") and n[6:].isdigit():
            out.append(load_polish_bankruptcy(int(n[6:])))
        else:
            raise ValueError(f"unknown panel {n!r}; choose from {', '.join(PANEL_NAMES)}")
    return out


def comparison_pairs(
    strategies: Sequence[str], compare_against: Sequence[str] | None = None
) -> list[tuple[str, str]]:
    """The strategy pairs of one (panel, size, seed) group that belong to the family.

    Args:
        strategies: Strategies scored in the group, in order.
        compare_against: Baseline strategies. ``None`` keeps every pair. Otherwise a pair is
            kept when at least one member is a baseline, and is oriented with the baseline
            first (``a - b``), so ``["uniform"]`` yields ``uniform - x`` for every other ``x``.

    Returns:
        The ``(a, b)`` pairs.
    """
    pairs = []
    for a, b in itertools.combinations(strategies, 2):
        if compare_against is None or a in compare_against:
            pairs.append((a, b))
        elif b in compare_against:
            pairs.append((b, a))
    return pairs


def check_bootstrap_resolution(
    n_boot: int, family_size: int, allow_coarse: bool = False
) -> int | None:
    """Refuse a bootstrap whose Holm floor is at or above :data:`MAX_HOLM_FLOOR`.

    Args:
        n_boot: Bootstrap resamples per comparison.
        family_size: Comparisons per metric in the Holm family.
        allow_coarse: When true, never refuse (the summary still warns, FINDINGS §288).

    Returns:
        The smallest ``n_boot`` that would pass when the check fails but was overridden;
        ``None`` when it passes.

    Raises:
        ValueError: If the floor is too high and ``allow_coarse`` is false; the message names
            the minimum ``n_boot``.
    """
    if n_boot <= 0 or family_size <= 0 or holm_floor(n_boot, family_size) < MAX_HOLM_FLOOR:
        return None
    need = round(family_size / MAX_HOLM_FLOOR)
    while holm_floor(need, family_size) >= MAX_HOLM_FLOOR:
        need += 1
    if allow_coarse:
        return need
    raise ValueError(
        f"n_boot={n_boot} over a Holm family of {family_size} comparisons per metric has an "
        f"adjusted-p floor of {holm_floor(n_boot, family_size):.4f} (limit {MAX_HOLM_FLOOR}); "
        f"use n_boot >= {need}, shrink the family (--compare-against), or pass "
        f"--allow-coarse-bootstrap to accept the coarse test"
    )


def paired_cell_comparisons(
    scores: dict[tuple[str, int, int, str], tuple[np.ndarray, np.ndarray]],
    n_boot: int = DEFAULT_N_BOOT,
    coverage: Coverage | None = None,
    compare_against: Sequence[str] | None = None,
    family_size: int | None = None,
    allow_coarse: bool = False,
) -> list[dict]:
    """Paired bootstrap of the strategy pairs in the family, Holm-adjusted per metric.

    Args:
        scores: ``(panel, max_context, seed, strategy) -> (y_test, p_test)``. Strategies of one
            (panel, max_context, seed) must have scored the same rows in the same order.
        n_boot: Bootstrap resamples per comparison.
        coverage: Optional tracker; one unit per (panel, size, seed, pair, metric).
        compare_against: Baseline strategies restricting the family (see
            :func:`comparison_pairs`); ``None`` compares every pair.
        family_size: Comparisons per metric of the intended (merged) family, when larger than
            this call's own; used only for the resolution check.
        allow_coarse: Skip the refusal of :func:`check_bootstrap_resolution`.

    Returns:
        One row per pair and metric with ``panel``, ``max_context``, ``seed``, ``a``, ``b``,
        ``metric``, ``delta`` (``a - b``), ``ci_lo``, ``ci_hi``, ``p_raw``, ``p_holm``,
        ``n_family`` and ``significant`` (Holm p below 0.05). Empty when no group has a pair.

    Raises:
        ValueError: If two strategies of one group scored different labels, or the bootstrap
            is too coarse for the family (checked before any resampling).
    """
    groups: dict[tuple[str, int, int], list[str]] = {}
    for panel, size, seed, strategy in scores:
        groups.setdefault((panel, size, seed), []).append(strategy)
    plan = [
        (key, pair)
        for key, strategies in groups.items()
        for pair in comparison_pairs(strategies, compare_against)
    ]
    check_bootstrap_resolution(n_boot, family_size or len(plan), allow_coarse)
    rows: list[dict] = []
    for (panel, size, seed), (a, b) in plan:
        y_a, p_a = scores[(panel, size, seed, a)]
        y_b, p_b = scores[(panel, size, seed, b)]
        if not np.array_equal(y_a, y_b):
            raise ValueError(f"{panel} ctx={size} seed={seed}: {a} and {b} scored other rows")
        for metric in METRICS:
            unit = f"{panel}|{size}|{seed}|{a}-{b}|{metric}"
            if coverage is not None:
                coverage.attempt(unit)
            delta, (lo, hi), p = paired_auc_difference(
                y_a,
                p_a,
                p_b,
                n_boot,
                seed,
                metric=average_precision_score if metric == "ap" else None,
            )
            if coverage is not None:
                if np.isfinite(p):
                    coverage.scored(unit)
                else:
                    coverage.skip(unit, "single-class test part or too few bootstrap draws")
            rows.append(
                {
                    "panel": panel,
                    "max_context": size,
                    "seed": seed,
                    "a": a,
                    "b": b,
                    "metric": metric,
                    "delta": delta,
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "p_raw": p,
                }
            )
    apply_holm(rows)
    return rows


def _method_text(n_boot: int | None, compare_against: Sequence[str] | None, where: str) -> str:
    """The ``comparison_method`` sentence.

    Args:
        n_boot: Bootstrap resamples.
        compare_against: Baselines restricting the family, or ``None`` for every pair.
        where: Where the Holm family spans, e.g. ``"this record"``.

    Returns:
        The sentence.
    """
    family = (
        "every strategy pair"
        if compare_against is None
        else "pairs against " + ", ".join(compare_against)
    )
    return (
        f"paired row bootstrap on the shared held-out rows, n_boot={n_boot}; family: {family}; "
        f"Holm-adjusted across every comparison in {where}, separately for AUC and AP"
    )


_NUM_LIST = re.compile(r"\[\s*(-?\d[^\[\]{}\"]*?)\s*\]")


def dump_record(record: dict) -> str:
    """Serialise a record as indented JSON with each list of numbers on one line.

    ``json.dumps(indent=2)`` would put every stored probability on its own line.

    Args:
        record: A sweep record.

    Returns:
        The JSON text.
    """
    text = json.dumps(record, indent=2)
    return _NUM_LIST.sub(lambda m: "[" + re.sub(r"\s+", " ", m.group(1)) + "]", text)


def apply_holm(rows: list[dict]) -> None:
    """Set ``p_holm``, ``n_family`` and ``significant`` on every row, one family per metric.

    Any value already in those keys is overwritten, so this is also how stale adjustments are
    dropped when records are merged.

    Args:
        rows: Comparison rows with ``metric`` and ``p_raw``; modified in place.
    """
    for metric in METRICS:
        idx = [i for i, r in enumerate(rows) if r["metric"] == metric]
        adj = holm_adjusted_p([rows[i]["p_raw"] for i in idx])
        for i, p_holm in zip(idx, adj, strict=True):
            rows[i]["p_holm"] = p_holm
            rows[i]["n_family"] = len(idx)
            rows[i]["significant"] = bool(np.isfinite(p_holm) and p_holm < 0.05)


def run_sweep(
    model: FinancialTFM | str,
    panels: Sequence[CreditDataset],
    strategies: Sequence[str] = DEFAULT_STRATEGIES,
    sizes: Sequence[int] = DEFAULT_SIZES,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    n_boot: int = DEFAULT_N_BOOT,
    n_ensemble: int = 8,
    test_size: float = 0.3,
    max_rows: int | None = None,
    retrieval_groups: int = 64,
    out_dir: Path | None = None,
    compare_against: Sequence[str] | None = None,
    family_size: int | None = None,
    allow_coarse_bootstrap: bool = False,
) -> dict:
    """Run every strategy at every context size and seed on every panel.

    Rank transform is on and the base-rate correction is on in every arm, as in §35 and §29.
    The split seed and the classifier's ``random_state`` are both the cell's seed, and the
    split is identical across strategies and sizes.

    Args:
        model: Checkpoint path or a model.
        panels: Binary panels to score.
        strategies: Context strategies (``ContextStrategy`` names).
        sizes: ``max_context`` values; a size at or above the training pool is skipped.
        seeds: Split and classifier seeds.
        n_boot: Bootstrap resamples per comparison; 0 skips the comparisons. See
            :data:`DEFAULT_N_BOOT` for why the default is what it is.
        n_ensemble: Ensemble members per fit (``fintfm-bench`` uses 8).
        test_size: Held-out share of each panel.
        max_rows: Optional stratified subsample of each panel (smoke tests only).
        retrieval_groups: Query groups for ``retrieval``.
        out_dir: When given, ``binary_context_sweep.json`` is written there.
        compare_against: Pre-registered baselines restricting the comparison family (each must
            be one of ``strategies``); ``None`` compares every pair.
        family_size: Comparisons per metric of the **merged** family this run belongs to. A
            per-panel kernel sees only its own slice, but Holm will span all of them, so the
            resolution check must use the full number. ``None`` uses an upper bound for this
            run (panels x sizes x seeds x pairs).
        allow_coarse_bootstrap: Run even when ``n_boot`` leaves the Holm floor at or above
            :data:`MAX_HOLM_FLOOR`; otherwise that is refused **before any cell is scored**.

    Returns:
        ``{"config", "labels", "cells", "comparisons", "coverage", "comparisons_coverage",
        "comparison_method"}``. ``labels`` maps ``"<panel>|<seed>"`` to the held-out labels
        shared by every strategy and size of that split. Each cell holds ``p`` (its held-out
        probabilities, rounded to 6 decimals; the comparisons use these rounded values so a
        recomputation is exact) and ``panel``, ``max_context``, ``seed``,
        ``strategy``, ``n_train``, ``n_test``, ``slice_default_rate``, ``train_default_rate``,
        ``context_rate``, ``n_context``, ``n_context_positive``, ``auc``, ``ap``, ``ece``,
        ``brier``, ``mean_predicted`` and ``seconds``.

    Raises:
        ValueError: If a ``compare_against`` entry is not in ``strategies``, or the bootstrap
            is too coarse for the family and ``allow_coarse_bootstrap`` is false.
    """
    if compare_against is not None:
        unknown = [c for c in compare_against if c not in strategies]
        if unknown:
            raise ValueError(f"--compare-against {unknown} not among strategies {list(strategies)}")
    if n_boot > 0:
        n_pairs = len(comparison_pairs(list(strategies), compare_against))
        upper = len(panels) * len(sizes) * len(seeds) * n_pairs
        check_bootstrap_resolution(n_boot, family_size or upper, allow_coarse_bootstrap)
    cov = Coverage()
    labels: dict[str, list[int]] = {}
    cells: list[dict] = []
    scores: dict[tuple[str, int, int, str], tuple[np.ndarray, np.ndarray]] = {}
    max_features = (
        FinancialTFM.load(model).cfg.max_features
        if isinstance(model, str)
        else model.cfg.max_features
    )
    for ds in panels:
        X, y = ds.X, ds.y
        if max_rows is not None and len(y) > max_rows:
            keep, _ = train_test_split(
                np.arange(len(y)), train_size=max_rows, random_state=0, stratify=y
            )
            X, y = X[keep], y[keep]
        print(
            f"\n=== {ds.name}: {len(y)} rows, {X.shape[1]} features, "
            f"default rate {float(np.mean(y)):.3%} ===",
            flush=True,
        )
        for seed in seeds:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=test_size, random_state=seed, stratify=y
            )
            labels[f"{ds.name}|{seed}"] = [int(v) for v in y_te]
            for size, strategy in itertools.product(sizes, strategies):
                unit = f"{ds.name}|{size}|{seed}|{strategy}"
                cov.attempt(unit)
                if X.shape[1] > max_features:
                    cov.skip(unit, f"data has {X.shape[1]} features, model takes {max_features}")
                    continue
                if size >= len(y_tr):
                    cov.skip(unit, f"context size {size} covers the whole pool of {len(y_tr)} rows")
                    continue
                t0 = time.time()
                try:
                    clf = FinancialTFMClassifier(
                        model,
                        max_context=size,
                        context_strategy=strategy,  # type: ignore[arg-type]
                        feature_transform="rank",
                        correct_prior=True,
                        random_state=seed,
                        n_ensemble=n_ensemble,
                        retrieval_groups=retrieval_groups,
                    )
                    clf.fit(X_tr, y_tr)
                    p = clf.predict_proba(X_te)[:, 1]
                except (ValueError, RuntimeError) as exc:
                    cov.skip(unit, f"fit or predict failed: {exc}")
                    print(f"  SKIPPED {unit}: {exc}", flush=True)
                    continue
                m = evaluate_binary(y_te, p)
                p_kept = np.round(np.asarray(p, dtype=float), 6)
                s = clf.context_summary_
                cell = {
                    "panel": ds.name,
                    "max_context": size,
                    "seed": seed,
                    "strategy": strategy,
                    "n_train": len(y_tr),
                    "n_test": len(y_te),
                    "slice_default_rate": float(np.mean(y_te)),
                    "train_default_rate": float(np.mean(y_tr)),
                    "context_rate": float(s.n_positive / s.n_context),
                    "n_context": int(s.n_context),
                    "n_context_positive": int(s.n_positive),
                    "auc": float(m.roc_auc),
                    "ap": float(m.average_precision),
                    "ece": float(m.ece),
                    "brier": float(m.brier),
                    "mean_predicted": float(m.mean_predicted),
                    "seconds": float(time.time() - t0),
                    "p": p_kept.tolist(),
                }
                cells.append(cell)
                scores[(ds.name, size, seed, strategy)] = (y_te, p_kept)
                cov.scored(unit)
                print(
                    f"  {strategy:>9} ctx={size:<5} seed={seed} "
                    f"AP={cell['ap']:.4f} AUC={cell['auc']:.4f} ECE={cell['ece']:.4f} "
                    f"rate={cell['slice_default_rate']:.3%} "
                    f"ctx_rate={cell['context_rate']:.2%} {cell['seconds']:.0f}s",
                    flush=True,
                )
    cmp_cov = Coverage()
    comparisons = (
        paired_cell_comparisons(
            scores, n_boot, cmp_cov, compare_against, family_size, allow_coarse_bootstrap
        )
        if n_boot > 0
        else []
    )
    record = {
        "config": {
            "model": model if isinstance(model, str) else "<in-memory model>",
            "panels": [d.name for d in panels],
            "strategies": list(strategies),
            "sizes": list(sizes),
            "seeds": list(seeds),
            "n_boot": n_boot,
            "n_ensemble": n_ensemble,
            "test_size": test_size,
            "max_rows": max_rows,
            "feature_transform": "rank",
            "correct_prior": True,
            "compare_against": list(compare_against) if compare_against is not None else None,
            "family_size": family_size,
        },
        "labels": labels,
        "cells": cells,
        "coverage": cov.to_dict(),
        "comparisons": comparisons,
        "comparisons_coverage": cmp_cov.to_dict(),
        "comparison_method": _method_text(n_boot, compare_against, "this record"),
    }
    print(cov.summary("cells"), flush=True)
    if n_boot > 0:
        print(cmp_cov.summary("comparisons"), flush=True)
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / OUT_FILE).write_text(dump_record(record))
    return record


#: Config keys that may differ between slices of one sweep; every other key must match.
SLICE_KEYS = ("panels", "seeds")


def _cell_key(c: dict) -> tuple[str, int, int, str]:
    """The identity of a cell: ``(panel, max_context, seed, strategy)``."""
    return (c["panel"], c["max_context"], c["seed"], c["strategy"])


def _merge_coverage(parts: Sequence[dict], what: str) -> dict:
    """Add up coverage dicts of disjoint slices.

    Args:
        parts: ``Coverage.to_dict()`` outputs, one per slice.
        what: Name of the coverage block, for the error message.

    Returns:
        One coverage dict over the union.

    Raises:
        ValueError: If one unit is skipped in more than one slice.
    """
    skipped: dict[str, str] = {}
    for part in parts:
        for unit, reason in part["skipped"].items():
            if unit in skipped:
                raise ValueError(f"duplicate {what} unit {unit!r} skipped in more than one record")
            skipped[unit] = reason
    n_att = sum(p["n_attempted"] for p in parts)
    n_sco = sum(p["n_scored"] for p in parts)
    return {
        "n_attempted": n_att,
        "n_scored": n_sco,
        "n_skipped": len(skipped),
        "coverage": n_sco / n_att if n_att else None,
        "skipped": skipped,
    }


def merge_records(records: Sequence[dict], sources: Sequence[str] | None = None) -> dict:
    """Merge sweep records of disjoint (panel, seed) slices into one record.

    The Holm family must span every slice (FINDINGS §273), so ``p_holm``, ``n_family`` and
    ``significant`` of the inputs are discarded and recomputed per metric from ``p_raw`` over
    the union of the comparison rows.

    Args:
        records: Records written by :func:`run_sweep`, each for a slice of the same sweep.
        sources: Optional label per record (file path, kernel name or commit), kept in
            ``provenance``. Defaults to ``"record <i>"``.

    Held-out labels and per-cell predictions are carried through when present (records from
    before task 32.7 have none and merge as before, but cannot be recomputed); labels of the
    same ``panel|seed`` must be identical.

    Returns:
        A record that :func:`summarise` accepts. ``config`` has the union of ``panels`` and
        ``seeds`` (first-seen order); ``provenance`` lists each source with its own panels and
        seeds, and keeps the provenance of records that were themselves merged.

    Raises:
        ValueError: If no record is given, ``sources`` has another length, configs differ in
            any key other than ``panels`` and ``seeds`` (the error names the key), two records
            hold different labels for one split, hold the same (panel, max_context, seed, strategy) cell, or two comparison rows
            share a (panel, max_context, seed, a, b, metric) key.
    """
    if not records:
        raise ValueError("no records to merge")
    if sources is not None and len(sources) != len(records):
        raise ValueError(f"{len(sources)} sources for {len(records)} records")
    names = list(sources) if sources is not None else [f"record {i}" for i in range(len(records))]
    base = records[0]["config"]
    for label, rec in zip(names, records, strict=True):
        cfg = rec["config"]
        for key in sorted((set(base) | set(cfg)) - set(SLICE_KEYS)):
            if base.get(key, "<missing>") != cfg.get(key, "<missing>"):
                raise ValueError(
                    f"config field {key!r} differs: {base.get(key, '<missing>')!r} in "
                    f"{names[0]} but {cfg.get(key, '<missing>')!r} in {label}"
                )
    config = dict(base)
    for key in SLICE_KEYS:
        config[key] = list(dict.fromkeys(v for r in records for v in r["config"].get(key, [])))
    cells: list[dict] = []
    labels: dict[str, list[int]] = {}
    seen_cells: dict[tuple, str] = {}
    comparisons: list[dict] = []
    seen_rows: dict[tuple, str] = {}
    provenance: list[dict] = []
    for label, rec in zip(names, records, strict=True):
        for lkey, lval in rec.get("labels", {}).items():
            if labels.setdefault(lkey, lval) != lval:
                raise ValueError(f"held-out labels for {lkey!r} differ in {label}")
        for c in rec["cells"]:
            key = _cell_key(c)
            if key in seen_cells:
                raise ValueError(f"duplicate cell {key} in {label} (already in {seen_cells[key]})")
            seen_cells[key] = label
            cells.append(dict(c))
        for r in rec["comparisons"]:
            rkey = (r["panel"], r["max_context"], r["seed"], r["a"], r["b"], r["metric"])
            if rkey in seen_rows:
                raise ValueError(
                    f"duplicate comparison {rkey} in {label} (already in {seen_rows[rkey]})"
                )
            seen_rows[rkey] = label
            row = {k: v for k, v in r.items() if k not in ("p_holm", "n_family", "significant")}
            comparisons.append(row)
        inherited = rec.get("provenance")
        if inherited:
            provenance.extend(inherited)
        else:
            provenance.append(
                {
                    "source": label,
                    "panels": list(rec["config"].get("panels", [])),
                    "seeds": list(rec["config"].get("seeds", [])),
                }
            )
    apply_holm(comparisons)
    n_boot = config.get("n_boot")
    return {
        "config": config,
        **({"labels": labels} if labels else {}),
        "cells": cells,
        "coverage": _merge_coverage([r["coverage"] for r in records], "cell"),
        "comparisons": comparisons,
        "comparisons_coverage": _merge_coverage(
            [r["comparisons_coverage"] for r in records], "comparison"
        ),
        "comparison_method": _method_text(
            n_boot,
            config.get("compare_against"),
            f"this merged record ({len(records)} slices)",
        ),
        "provenance": provenance,
    }


def holm_floor(n_boot: int | None, n_family: int) -> float:
    """Smallest Holm-adjusted p a bootstrap of ``n_boot`` resamples can produce.

    :func:`~fintfm.evaluation.metrics.paired_auc_difference` never returns a p below
    ``1 / (n_boot + 1)``, and Holm multiplies the smallest raw p by the family size, so a
    family larger than about ``0.05 * (n_boot + 1)`` can never show a significant pair however
    large the differences are. FINDINGS §288 is the run that hit this.

    Args:
        n_boot: Bootstrap resamples per comparison; ``None`` or 0 when none were drawn.
        n_family: Comparisons in the Holm family.

    Returns:
        The floor, capped at 1.0; 0.0 when ``n_boot`` is unknown, so no warning is raised on
        a record that does not say.
    """
    if not n_boot or n_family <= 0:
        return 0.0
    return min(1.0, n_family / (n_boot + 1))


def summarise(record: dict) -> str:
    """Seed-mean table per panel, size and strategy, then the Holm table, AP first.

    Args:
        record: The dict :func:`run_sweep` returns.

    Returns:
        Printable text.
    """
    head = (
        f"{'panel':>12} {'ctx':>5} {'strategy':>10} {'seeds':>5} {'slice%':>7} "
        f"{'AP':>7} {'AUC':>7} {'ECE':>7} {'ctx%':>6}"
    )
    lines = [head]
    keys = sorted({(c["panel"], c["max_context"], c["strategy"]) for c in record["cells"]})
    for panel, size, strat in keys:
        g = [
            c
            for c in record["cells"]
            if (c["panel"], c["max_context"], c["strategy"]) == (panel, size, strat)
        ]
        lines.append(
            f"{panel:>12} {size:>5} {strat:>10} {len(g):>5} "
            f"{100 * np.mean([c['slice_default_rate'] for c in g]):>7.3f} "
            f"{np.mean([c['ap'] for c in g]):>7.4f} {np.mean([c['auc'] for c in g]):>7.4f} "
            f"{np.mean([c['ece'] for c in g]):>7.4f} "
            f"{100 * np.mean([c['context_rate'] for c in g]):>6.2f}"
        )
    for metric in METRICS:
        cmps = [c for c in record["comparisons"] if c["metric"] == metric]
        if not cmps:
            continue
        lines += ["", f"paired {metric.upper()} comparisons ({len(cmps)} in the Holm family):"]
        floor = holm_floor(record["config"].get("n_boot"), len(cmps))
        if floor >= 0.05:
            lines.append(
                f"WARNING: n_boot={record['config'].get('n_boot')} resolves p no lower than "
                f"1/(n_boot+1), so Holm over {len(cmps)} comparisons cannot go below "
                f"{floor:.4f}: no comparison can reach 0.05, and 'not significant' here is the "
                f"test's resolution, not evidence of no difference (FINDINGS §288)"
            )
        lines.append(
            f"{'panel':>12} {'ctx':>5} {'a - b':>22} {'seeds':>5} {'mean delta':>10} "
            f"{'min Holm p':>11} {'Holm<.05':>9}"
        )
        for panel, size, a, b in sorted(
            {(c["panel"], c["max_context"], c["a"], c["b"]) for c in cmps}
        ):
            g = [
                c
                for c in cmps
                if (c["panel"], c["max_context"], c["a"], c["b"]) == (panel, size, a, b)
            ]
            ps = [c["p_holm"] for c in g if np.isfinite(c["p_holm"])]
            lines.append(
                f"{panel:>12} {size:>5} {a + ' - ' + b:>22} {len(g):>5} "
                f"{np.mean([c['delta'] for c in g]):>+10.4f} "
                f"{(min(ps) if ps else float('nan')):>11.4f} {sum(p < 0.05 for p in ps):>9}"
            )
    return "\n".join(lines)


def merge_files(paths: Sequence[Path], out_dir: Path) -> dict:
    """Merge sweep record files and write the merged record.

    Args:
        paths: ``binary_context_sweep.json`` files of the slices.
        out_dir: Directory that receives the merged ``binary_context_sweep.json``.

    Returns:
        The merged record (see :func:`merge_records`); sources are the given paths.
    """
    records = [json.loads(Path(p).read_text()) for p in paths]
    merged = merge_records(records, sources=[str(p) for p in paths])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / OUT_FILE).write_text(dump_record(merged))
    return merged


def recompare_record(
    record: dict,
    n_boot: int | None = None,
    compare_against: Sequence[str] | None = None,
    allow_coarse_bootstrap: bool = False,
) -> dict:
    """Redo the paired comparisons of a record from its stored predictions, without scoring.

    The bootstrap seed of each comparison is its cell's seed, as in :func:`run_sweep`, so the
    same ``n_boot`` and family reproduce the original rows exactly.

    Args:
        record: A record whose cells all hold ``p`` and whose ``labels`` cover every cell.
        n_boot: New resamples per comparison; ``None`` keeps the record's.
        compare_against: New family restriction; ``None`` keeps the record's.
        allow_coarse_bootstrap: Skip the refusal of :func:`check_bootstrap_resolution`.

    Returns:
        A new record with the same cells, labels and coverage and fresh comparisons, config
        ``n_boot`` and ``compare_against``, and ``comparison_method``; ``provenance`` is kept
        and ``recomputed_from`` notes the old ``n_boot``.

    Raises:
        ValueError: If any cell has no stored predictions or its labels are missing or of
            another length, or the bootstrap is too coarse for the family.
    """
    labels = record.get("labels", {})
    scores: dict[tuple[str, int, int, str], tuple[np.ndarray, np.ndarray]] = {}
    for c in record["cells"]:
        y = labels.get(f"{c['panel']}|{c['seed']}")
        if "p" not in c or y is None:
            raise ValueError(
                "record has no stored predictions (written before task 32.7?); "
                "the cells must be rescored to compare again"
            )
        if len(y) != len(c["p"]):
            raise ValueError(f"{_cell_key(c)}: {len(c['p'])} predictions for {len(y)} labels")
        scores[_cell_key(c)[:3] + (c["strategy"],)] = (np.asarray(y), np.asarray(c["p"]))
    cfg = dict(record["config"])
    old_n_boot = cfg.get("n_boot")
    n_boot = old_n_boot if n_boot is None else n_boot
    against = cfg.get("compare_against") if compare_against is None else list(compare_against)
    strategies = {c["strategy"] for c in record["cells"]}
    unknown = [a for a in against or [] if a not in strategies]
    if unknown:
        raise ValueError(f"--compare-against {unknown} not among the record's strategies")
    cmp_cov = Coverage()
    comparisons = (
        paired_cell_comparisons(
            scores, n_boot, cmp_cov, against, allow_coarse=allow_coarse_bootstrap
        )
        if n_boot
        else []
    )
    cfg.update(n_boot=n_boot, compare_against=against, family_size=None)
    prov = record.get("provenance")
    where = "this merged record" if prov else "this record"
    out = {
        **record,
        "config": cfg,
        "comparisons": comparisons,
        "comparisons_coverage": cmp_cov.to_dict(),
        "comparison_method": _method_text(n_boot, against, where),
        "recomputed_from": {"n_boot": old_n_boot},
    }
    return out


def recompare_file(path: Path, out_dir: Path, **kwargs) -> dict:
    """Recompute the comparisons of a record file and write the new record.

    Args:
        path: A ``binary_context_sweep.json`` with stored predictions.
        out_dir: Directory that receives the new ``binary_context_sweep.json``.
        **kwargs: Passed to :func:`recompare_record`.

    Returns:
        The new record.
    """
    new = recompare_record(json.loads(Path(path).read_text()), **kwargs)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / OUT_FILE).write_text(dump_record(new))
    return new


def main() -> None:
    """Command-line entry point (``fintfm-binctx``).

    Three modes. Without ``--merge`` or ``--recompare`` it runs the sweep and needs
    ``--model``. With ``--merge a.json b.json ...`` it merges those slice records,
    recomputing the Holm family over all of them, writes ``binary_context_sweep.json`` to
    ``--out`` and prints the summary. With ``--recompare RECORD.json`` it redoes the
    bootstrap from the stored predictions (``--n-boot`` and ``--compare-against`` default to
    the record's), writes a new record to ``--out`` and prints the summary.
    """
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--merge", nargs="+", metavar="JSON", help="merge these slice records")
    p.add_argument("--model")
    p.add_argument("--out", type=str, default="runs/binary-context-sweep")
    p.add_argument("--panels", type=str, default=",".join(PANEL_NAMES), help="comma-separated")
    p.add_argument("--strategies", type=str, default=",".join(DEFAULT_STRATEGIES))
    p.add_argument("--sizes", type=str, default=",".join(map(str, DEFAULT_SIZES)))
    p.add_argument("--seeds", type=str, default=",".join(map(str, DEFAULT_SEEDS)))
    p.add_argument("--recompare", metavar="JSON", help="recompute comparisons from stored p")
    p.add_argument(
        "--compare-against",
        action="append",
        metavar="STRATEGY",
        help="baseline of the pre-registered family; repeat for several (default: all pairs)",
    )
    p.add_argument(
        "--n-boot",
        type=int,
        default=None,
        help=f"bootstrap resamples (default {DEFAULT_N_BOOT}; with --recompare, the record's)",
    )
    p.add_argument(
        "--family-size",
        type=int,
        default=None,
        help="comparisons per metric of the merged family a per-panel run belongs to",
    )
    p.add_argument(
        "--allow-coarse-bootstrap",
        action="store_true",
        help=f"run although n_boot leaves the Holm floor >= {MAX_HOLM_FLOOR}",
    )
    p.add_argument("--n-ensemble", type=int, default=8)
    p.add_argument("--max-rows", type=int, default=None, help="subsample each panel (smoke only)")
    p.add_argument("--retrieval-groups", type=int, default=64)
    a = p.parse_args()
    if a.merge:
        merged = merge_files([Path(f) for f in a.merge], Path(a.out))
        print(summarise(merged))
        return
    try:
        if a.recompare:
            new = recompare_file(
                Path(a.recompare),
                Path(a.out),
                n_boot=a.n_boot,
                compare_against=a.compare_against,
                allow_coarse_bootstrap=a.allow_coarse_bootstrap,
            )
            print(summarise(new))
            return
        if not a.model:
            p.error("--model is required unless --merge or --recompare is given")
        _run_cli(a)
    except ValueError as exc:
        p.error(str(exc))


def _run_cli(a: argparse.Namespace) -> None:
    """Run the sweep from parsed command-line arguments and print the summary.

    Args:
        a: Parsed arguments of :func:`main`.
    """
    record = run_sweep(
        a.model,
        load_panels([s for s in a.panels.split(",") if s]),
        strategies=[s for s in a.strategies.split(",") if s],
        sizes=[int(s) for s in a.sizes.split(",") if s],
        seeds=[int(s) for s in a.seeds.split(",") if s],
        n_boot=DEFAULT_N_BOOT if a.n_boot is None else a.n_boot,
        n_ensemble=a.n_ensemble,
        max_rows=a.max_rows,
        retrieval_groups=a.retrieval_groups,
        out_dir=Path(a.out),
        compare_against=a.compare_against,
        family_size=a.family_size,
        allow_coarse_bootstrap=a.allow_coarse_bootstrap,
    )
    print("\n" + summarise(record))


if __name__ == "__main__":
    main()
