"""Schema variation at fixed task simplicity (openspec task 39.12).

Why this exists
---------------
A model that has learned "3 of these columns matter" should not have its signal diluted by a
table's width, column order, units, column types or missing cells. This harness holds the
*task* fixed (same signal columns, same Bayes-optimal AUC, same labels) and varies only the
schema around it, then scores the checkpoint(s), an untrained control of the same
architecture (the floor; CLAUDE.md "Every evaluation needs a floor"), logistic regression and
histogram gradient boosting on each schema over seeds. It also recovers a per-column
permutation-importance ranking and compares it to the true relevant set (the task's Verify).

Design
------
- Task: ``S`` signal columns (default 3), class 1 shifted by ``mu / sqrt(S)`` in each, unit
  variance, 50% class rate, labels exactly balanced in the context and in the query set. The
  true ``logit P(y=1|x) = sum_j mu_j x_j - sum_j mu_j^2 / 2`` is closed form, so the Bayes AUC
  and Bayesian regret (:func:`fintfm.experiments.capability.bayesian_regret`) are exact and
  *computed from the base table*, so they are identical across schema variants by
  construction (a test pins it). Target Bayes AUC is :data:`TARGET_AUC`.
- Schema variants (:class:`Schema`) are transforms of one base table of the same seed:
  column count (padding with independent N(0,1) columns, prefix-stable across widths), the
  number of signal columns at fixed width (useful-vs-useless ratio), column order (signal
  first, signal last, shuffled: exact permutations), scaling and units (per-column affine map
  ``a*x+b``, ``a`` spanning 1e-3..1e3: exact), duplicated columns (copies of signal and noise
  columns), constant columns, categorical columns (noise columns binned to 4 integer codes
  at fixed normal quantiles) and missingness (NaN in a share of the noise cells). Only noise
  columns are ever replaced, never signal ones, so the Bayes AUC is unchanged.
- Width above a checkpoint's ``max_features`` is not run for that checkpoint: a recorded skip
  (``skipped`` list in the record), not a crash. The baselines always run.
- Arms: each supplied checkpoint, ``untrained_control`` (seeded weights), logistic regression
  (median impute + standardise, so units do not handicap the floor) and gradient boosting.
- Importance: per-column permutation importance (AUC drop on the query rows), then recall of
  the true relevant set in the top-``|relevant|`` columns. Duplicates of a signal column
  count as relevant but share its information, so a permutation can under-credit each copy;
  read the duplicate schema with that in mind.

Nothing here trains. Numbers are MEASURED only when produced by the command in the openspec
Progress note on real checkpoints; the unit tests are SMOKE-TEST only.
"""

from __future__ import annotations

import argparse
import json
import zlib
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from fintfm.experiments.capability import _bayes_optimal_mu, bayesian_regret
from fintfm.experiments.irrelevant_features import UNTRAINED_SEED

if TYPE_CHECKING:
    from fintfm.modeling.model import FinancialTFM

#: Target Bayes-optimal AUC of the fixed task (the medium tier of the 39.16 grid).
TARGET_AUC = 0.85

#: Number of signal columns of the base task.
N_SIGNAL = 3

#: Column counts of the width sweep named by the task.
WIDTHS: tuple[int, ...] = (5, 10, 50, 200)

#: Width at which the non-width schema properties are varied.
FIXED_WIDTH = 50

#: Context rows, query rows and rows used for permutation importance.
N_CONTEXT = 512
N_QUERY = 500
N_IMPORTANCE = 200

#: Edges that cut N(0,1) into four equal-mass levels (categorical columns).
_QUARTILES = np.array([-0.6744897501960817, 0.0, 0.6744897501960817])


@dataclass(frozen=True)
class Schema:
    """One table layout around the fixed task.

    Attributes:
        name: Unique label.
        width: Total column count, at least ``n_signal``.
        n_signal: Signal columns (the task's useful columns).
        order: ``"signal_first"``, ``"signal_last"`` or ``"shuffled"``.
        scaled: Apply a per-column affine map with scales across six decades.
        n_dup: Noise columns replaced by copies of signal/noise columns.
        n_const: Noise columns replaced by a constant.
        n_cat: Noise columns replaced by 4-level integer codes.
        missing: Share of plain-noise cells set to NaN, in ``[0, 1)``.
    """

    name: str
    width: int
    n_signal: int = N_SIGNAL
    order: str = "signal_first"
    scaled: bool = False
    n_dup: int = 0
    n_const: int = 0
    n_cat: int = 0
    missing: float = 0.0


def default_schemas(
    widths: tuple[int, ...] = WIDTHS, fixed_width: int = FIXED_WIDTH
) -> list[Schema]:
    """The sweep the task names: widths, then one property at a time at ``fixed_width``.

    Args:
        widths: Column counts of the width sweep.
        fixed_width: Width for the single-property variants (at least 20).

    Returns:
        Schemas with unique names, the plain-width ones first.
    """
    q = fixed_width // 5
    out = [Schema(f"w{d}", d) for d in widths]
    out += [
        Schema(f"w{fixed_width}_signal_last", fixed_width, order="signal_last"),
        Schema(f"w{fixed_width}_shuffled", fixed_width, order="shuffled"),
        Schema(f"w{fixed_width}_scaled", fixed_width, scaled=True),
        Schema(f"w{fixed_width}_dup{q}", fixed_width, n_dup=q),
        Schema(f"w{fixed_width}_const{q}", fixed_width, n_const=q),
        Schema(f"w{fixed_width}_cat{2 * q}", fixed_width, n_cat=2 * q),
        Schema(f"w{fixed_width}_missing30", fixed_width, missing=0.3),
        Schema(f"w{fixed_width}_signal1", fixed_width, n_signal=1),
        Schema(f"w{fixed_width}_signal10", fixed_width, n_signal=10),
    ]
    return out


def _check(s: Schema) -> None:
    """Refuse a schema whose edits do not fit its noise columns."""
    if s.width < s.n_signal or s.n_signal < 1:
        raise ValueError(f"{s.name}: width {s.width} must be >= n_signal {s.n_signal} >= 1")
    if s.order not in ("signal_first", "signal_last", "shuffled"):
        raise ValueError(f"{s.name}: unknown order {s.order!r}")
    if s.n_dup + s.n_const + s.n_cat > s.width - s.n_signal:
        raise ValueError(f"{s.name}: dup+const+cat exceed the {s.width - s.n_signal} noise columns")
    if not 0.0 <= s.missing < 1.0:
        raise ValueError(f"{s.name}: missing must be in [0, 1)")


def _stable(name: str) -> int:
    """Process-independent integer for a schema name (``hash`` is salted per process)."""
    return zlib.crc32(name.encode())


def bayes_probability(X_signal: np.ndarray, mu_total: float) -> np.ndarray:
    """True ``P(y=1 | x)`` from the signal columns at a 50% class rate.

    Each of the ``S`` signal columns carries shift ``mu_total / sqrt(S)``.

    Args:
        X_signal: ``(n, S)`` signal columns of the *base* table.
        mu_total: Combined separation (Bayes AUC is ``Phi(mu_total / sqrt(2))``).

    Returns:
        ``(n,)`` probabilities.
    """
    mu_j = mu_total / np.sqrt(X_signal.shape[1])
    logit = mu_j * X_signal.sum(axis=1) - 0.5 * X_signal.shape[1] * mu_j * mu_j
    return 1.0 / (1.0 + np.exp(-logit))


def make_base(
    seed: int, width: int, n_signal: int, n_rows: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Base table: signal columns first, independent N(0,1) noise after.

    Noise column ``j`` has its own stream, so a wider table's first columns equal a narrower
    one's. Rows are ``n_rows[0]`` context then ``n_rows[1]`` query, each exactly balanced.

    Args:
        seed: Task seed; the same arguments give bit-identical output.
        width: Total columns, at least ``n_signal``.
        n_signal: Signal columns.
        n_rows: ``(context rows, query rows)``.

    Returns:
        ``(X, y, p_bayes)`` over all rows (float64 table, int64 labels, true probability).
    """
    mu = _bayes_optimal_mu(TARGET_AUC)
    mu_j = mu / np.sqrt(n_signal)
    lab = np.random.default_rng(np.random.SeedSequence([seed, 7, 0]))
    y = np.concatenate([lab.permutation(np.arange(m) % 2) for m in n_rows]).astype(np.int64)
    n = int(sum(n_rows))
    X = np.empty((n, width))
    for j in range(width):
        col = np.random.default_rng(np.random.SeedSequence([seed, 7, 1, j])).normal(size=n)
        X[:, j] = col + (mu_j * y if j < n_signal else 0.0)
    return X, y, bayes_probability(X[:, :n_signal], mu)


def apply_schema(
    X_base: np.ndarray, schema: Schema, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Turn a base table into the schema's layout, using only data-independent parameters.

    Order of edits: constants, duplicates and categorical codes replace noise columns, NaNs
    are placed in plain noise, the affine scaling is applied, and the columns are permuted.

    Args:
        X_base: ``(n, schema.width)`` base table from :func:`make_base`.
        schema: Layout to apply.
        seed: Seed for the schema's own parameters (scales, permutation, NaN mask).

    Returns:
        ``(X, relevant, perm)``: the table, sorted indices of the information-bearing columns
        in it (signal columns plus copies of them), and ``perm`` with ``X[:, k]`` derived from
        base column ``perm[k]`` (edits applied before the permutation).
    """
    _check(schema)
    s, d = schema.n_signal, schema.width
    rng = np.random.default_rng(np.random.SeedSequence([seed, 11, _stable(schema.name)]))
    X = X_base.astype(np.float64).copy()
    relevant = list(range(s))
    end = d
    for i in range(schema.n_dup):  # copies: signal columns round-robin, noise column s between
        end -= 1
        src = i // 2 % s if i % 2 == 0 else s
        X[:, end] = X[:, src]
        if i % 2 == 0:
            relevant.append(end)
    for _ in range(schema.n_const):
        end -= 1
        X[:, end] = 3.5
    for _ in range(schema.n_cat):
        end -= 1
        X[:, end] = np.searchsorted(_QUARTILES, X[:, end]).astype(np.float64)
    if schema.missing > 0.0 and end > s:
        mask = rng.random(size=(X.shape[0], end - s)) < schema.missing
        X[:, s:end] = np.where(mask, np.nan, X[:, s:end])
    if schema.scaled:
        a = 10.0 ** rng.uniform(-3.0, 3.0, size=d)
        X = X * a + a * rng.normal(scale=10.0, size=d)
    if schema.order == "signal_first":
        perm = np.arange(d)
    elif schema.order == "signal_last":
        perm = np.concatenate([np.arange(s, d), np.arange(s)])
    else:
        perm = rng.permutation(d)
    pos = np.empty(d, dtype=np.int64)
    pos[perm] = np.arange(d)
    return X[:, perm], np.sort(pos[np.array(relevant)]), perm


def recall_at_k(importance: np.ndarray, relevant: np.ndarray) -> float:
    """Share of the true relevant set among the ``len(relevant)`` most important columns.

    Args:
        importance: ``(d,)`` per-column importance; ties are broken by column index.
        relevant: Indices of the true relevant columns.

    Returns:
        Recall in ``[0, 1]``.
    """
    k = len(relevant)
    top = np.argsort(-np.asarray(importance, dtype=float), kind="stable")[:k]
    return float(len(set(top.tolist()) & set(relevant.tolist())) / k)


def _fit_predictors(
    models: dict[str, FinancialTFM], schema_width: int, X: np.ndarray, y: np.ndarray, seed: int
) -> tuple[dict[str, Callable[[np.ndarray], np.ndarray]], list[dict]]:
    """Fit every arm once on the context; return predict functions and recorded skips."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from fintfm.inference.classifier import FinancialTFMClassifier

    fns: dict[str, Callable[[np.ndarray], np.ndarray]] = {}
    skipped: list[dict] = []
    for name, m in models.items():
        if schema_width > m.cfg.max_features:
            skipped.append(
                {
                    "arm": name,
                    "width": schema_width,
                    "reason": f"width {schema_width} exceeds max_features {m.cfg.max_features}",
                }
            )
            continue
        clf = FinancialTFMClassifier(
            m,
            max_context=len(y),
            context_strategy="uniform",
            feature_transform="rank",
            random_state=seed,
        ).fit(X, y)
        fns[name] = lambda Z, c=clf: c.predict_proba(Z)[:, 1]
    lr = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        StandardScaler(),
        LogisticRegression(max_iter=1000),
    ).fit(X, y)
    fns["logistic_regression"] = lambda Z: lr.predict_proba(Z)[:, 1]
    gb = HistGradientBoostingClassifier(max_iter=100, random_state=seed).fit(X, y)
    fns["gradient_boosting"] = lambda Z: gb.predict_proba(Z)[:, 1]
    return fns, skipped


def permutation_importance(
    predict: Callable[[np.ndarray], np.ndarray], X: np.ndarray, y: np.ndarray, seed: int
) -> np.ndarray:
    """AUC drop on ``(X, y)`` when each column is permuted in turn, per column.

    Args:
        predict: Positive-class probability for a table.
        X: Query table.
        y: Query labels (both classes present).
        seed: Seed for the permutations.

    Returns:
        ``(d,)`` drops (baseline AUC minus permuted AUC).
    """
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    base = roc_auc_score(y, predict(X))
    drops = np.empty(X.shape[1])
    for j in range(X.shape[1]):
        Xp = X.copy()
        Xp[:, j] = Xp[rng.permutation(len(Xp)), j]
        drops[j] = base - roc_auc_score(y, predict(Xp))
    return drops


def run(
    model_paths: dict[str, str],
    out_dir: Path | None = None,
    schemas: list[Schema] | None = None,
    seeds: tuple[int, ...] = (0, 1, 2),
    n_context: int = N_CONTEXT,
    n_query: int = N_QUERY,
    importance: bool = True,
    n_importance: int = N_IMPORTANCE,
    models: dict[str, FinancialTFM] | None = None,
) -> dict:
    """Score every arm on every schema over seeds, and recover the importance ranking.

    Args:
        model_paths: ``{name: checkpoint path}``; one untrained control is built from the
            first one's architecture. Ignored when ``models`` is given.
        out_dir: If given, ``schema_variation.json`` is written there.
        schemas: Layouts to run; default :func:`default_schemas`.
        seeds: Task seeds.
        n_context: Context rows.
        n_query: Query rows scored per schema.
        importance: Also compute permutation importance and recall of the relevant set.
        n_importance: Query rows used for the importance (first rows of the query set).
        models: Already-built models by name (used by tests); the control is added.

    Returns:
        The record: config, per-(arm, schema) results, and ``skipped`` (arm, width, reason).

    Raises:
        ValueError: On duplicate schema names or an invalid schema.
    """
    import torch
    from sklearn.metrics import average_precision_score

    from fintfm.modeling.model import FinancialTFM

    schemas = default_schemas() if schemas is None else schemas
    if len({s.name for s in schemas}) != len(schemas):
        raise ValueError("schema names must be unique")
    for s in schemas:
        _check(s)
    loaded = (
        dict(models)
        if models is not None
        else {name: FinancialTFM.load(p) for name, p in model_paths.items()}
    )
    if loaded:
        torch.manual_seed(UNTRAINED_SEED)  # seeded: an unseeded floor moves by 0.2 (§42)
        loaded["untrained_control"] = FinancialTFM(next(iter(loaded.values())).cfg)

    acc: dict[tuple[str, str], dict[str, list]] = {}
    skipped: dict[tuple[str, int], dict] = {}
    n_imp = min(n_importance, n_query)
    for seed in seeds:
        for sc in schemas:
            Xb, y, pb = make_base(seed, sc.width, sc.n_signal, (n_context, n_query))
            X, relevant, _ = apply_schema(Xb, sc, seed)
            Xtr, ytr, Xte, yte = X[:n_context], y[:n_context], X[n_context:], y[n_context:]
            fns, skips = _fit_predictors(loaded, sc.width, Xtr, ytr, seed)
            for k in skips:
                skipped[(k["arm"], k["width"])] = k
            for arm, fn in fns.items():
                p = fn(Xte)
                r = bayesian_regret(yte, pb[n_context:], p)
                c = acc.setdefault(
                    (arm, sc.name),
                    {k: [] for k in ("auc", "ap", "b", "r", "mp", "rate", "rec")},
                )
                c["auc"].append(r["auc_model"])
                c["ap"].append(float(average_precision_score(yte, p)))
                c["b"].append(r["auc_bayes"])
                c["r"].append(r["regret"])
                c["mp"].append(float(np.mean(p)))
                c["rate"].append(float(np.mean(yte)))
                if importance:
                    imp = permutation_importance(fn, Xte[:n_imp], yte[:n_imp], seed)
                    c["rec"].append(recall_at_k(imp, relevant))
        print(f"  seed {seed} done", flush=True)

    def _m(v: list) -> float:
        return float(np.mean(v)) if v else float("nan")

    results = []
    for (arm, name), c in acc.items():
        sc = next(s for s in schemas if s.name == name)
        results.append(
            {
                "arm": arm,
                "schema": name,
                "width": sc.width,
                "n_signal": sc.n_signal,
                "auc_mean": _m(c["auc"]),
                "auc_std": float(np.std(c["auc"], ddof=1)) if len(c["auc"]) > 1 else None,
                "ap_mean": _m(c["ap"]),
                "auc_bayes_mean": _m(c["b"]),
                "regret_mean": _m(c["r"]),
                "mean_pred": _m(c["mp"]),
                "observed_rate": _m(c["rate"]),
                "recall_at_k_mean": _m(c["rec"]) if c["rec"] else None,
                "n_seeds": len(c["auc"]),
            }
        )
    record = {
        "task": f"{N_SIGNAL} signal columns, two Gaussians, Bayes AUC {TARGET_AUC}; schema varies",
        "config": {
            "models": model_paths,
            "schemas": [asdict(s) for s in schemas],
            "seeds": list(seeds),
            "n_context": n_context,
            "n_query": n_query,
            "importance": importance,
            "n_importance": n_imp,
            "target_auc": TARGET_AUC,
        },
        "results": results,
        "skipped": list(skipped.values()),
    }
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "schema_variation.json").write_text(json.dumps(record, indent=2))
    return record


def summarise(record: dict) -> str:
    """Render one table per arm: AUC, regret, importance recall and calibration per schema.

    Args:
        record: Output of :func:`run`.

    Returns:
        A printable multi-line string, gaps (skips) first.
    """
    lines: list[str] = []
    for k in record["skipped"]:
        lines.append(f"SKIPPED {k['arm']} at width {k['width']}: {k['reason']}")
    if record["skipped"]:
        lines.append("")
    names = [s["name"] for s in record["config"]["schemas"]]
    rows = record["results"]
    for arm in sorted({r["arm"] for r in rows}):
        lines.append(f"{arm}: AUC / regret / recall@|relevant| / mean pred vs rate")
        for n in names:
            h = [r for r in rows if r["arm"] == arm and r["schema"] == n]
            if not h:
                lines.append(f"  {n:<22} (skipped)")
                continue
            r = h[0]
            rec = "-" if r["recall_at_k_mean"] is None else f"{r['recall_at_k_mean']:.2f}"
            lines.append(
                f"  {n:<22} {r['auc_mean']:.3f} / {r['regret_mean']:+.3f} / {rec:>4} / "
                f"{r['mean_pred']:.3f} vs {r['observed_rate']:.3f}"
            )
        lines.append("")
    lines += [
        "Reading it: the task and its Bayes AUC are identical in every row, so any change is the",
        "schema. Logistic regression is near the Bayes family here; the untrained control is the",
        "floor. Recall is the share of the true relevant columns in the top-|relevant| columns.",
    ]
    return "\n".join(lines)


def main() -> None:
    """Command-line entry point: ``python -m fintfm.experiments.schema_variation``."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", type=str, default="", help="name=path pairs, comma separated")
    p.add_argument("--out", type=str, default="runs/schema_variation")
    p.add_argument("--seeds", type=str, default="0,1,2")
    p.add_argument("--widths", type=str, default=",".join(str(w) for w in WIDTHS))
    p.add_argument("--fixed-width", type=int, default=FIXED_WIDTH)
    p.add_argument("--n-context", type=int, default=N_CONTEXT)
    p.add_argument("--n-query", type=int, default=N_QUERY)
    p.add_argument("--no-importance", action="store_true")
    args = p.parse_args()
    paths = dict(pair.split("=", 1) for pair in args.models.split(",") if pair)
    record = run(
        paths,
        Path(args.out),
        schemas=default_schemas(tuple(int(w) for w in args.widths.split(",")), args.fixed_width),
        seeds=tuple(int(s) for s in args.seeds.split(",")),
        n_context=args.n_context,
        n_query=args.n_query,
        importance=not args.no_importance,
    )
    print("\n" + summarise(record))


if __name__ == "__main__":
    main()
