"""Posterior task uncertainty on ambiguous contexts (openspec task 39.19).

Why this exists
---------------
Predictive uncertainty and *task* uncertainty are different things. Given a context that is
consistent with two or more data-generating hypotheses, a model with a real task posterior should
hedge on queries where the hypotheses disagree. A model that arbitrarily commits to one
hypothesis is confidently wrong half the time. This harness builds such contexts, with the Bayes
posterior over hypotheses in closed form as the reference, and measures how the model's
confidence moves as disambiguating rows are added (CLAUDE.md "Every evaluation needs a floor").

Design
------
- Hypotheses ``h_j: y = 1[x_j > 0]`` for ``j = 0 .. K-1`` over standard-normal columns. Labels
  are noiseless, so a context is *exactly* consistent with a hypothesis when every row's label
  equals ``h_j(x)``; :func:`consistent_hypotheses` checks this.
- **Ambiguous rows** have the same sign in all ``K`` of the first columns, so every hypothesis
  labels them identically. **Disambiguating rows** have mixed signs, so the hypotheses disagree
  on them. A context of ``n_ctx`` rows holds ``n_dis`` disambiguating rows and ``n_ctx - n_dis``
  ambiguous ones, labelled by the true hypothesis ``t``. ``n_dis = 0`` is fully ambiguous.
- **Matched difficulty:** every point of the curve uses the same context size, the same columns,
  the same hypothesis set and the *same query set* (mixed-sign rows, where the hypotheses
  disagree). Only the number of disambiguating rows changes, so the ambiguous and unambiguous
  contexts differ in task information and nothing else.
- **Bayes reference:** with a uniform prior and label-flip likelihood ``eps`` (default 0,
  noiseless) the posterior is proportional to
  ``prod_i (1-eps)^[h(x_i)=y_i] * eps^[h(x_i)!=y_i]``. At ``eps = 0`` it is uniform over the
  consistent hypotheses. The Bayes predictive on a query is the posterior-weighted vote.

Reported per arm and ``n_dis``, over seeds and tasks
----------------------------------------------------
- ``entropy``: mean binary predictive entropy in bits on the query rows (1 = fully hedged).
- ``posterior_entropy``: entropy (bits) of the *implied* task posterior, ``w_h`` the mean
  probability the arm assigns to ``h``'s label on the queries, normalised over ``h``. The Bayes
  arm's value is its exact posterior entropy.
- ``abs_err_bayes``: mean ``|p - p_Bayes|`` on the queries.
- ``commit``: the largest fraction of queries on which the arm's hard label agrees with one
  single hypothesis. Near ``1/K`` (``0.5`` for ``K = 2``) means no single hypothesis is picked;
  near 1 means arbitrary commitment on the ambiguous contexts.
- ``accuracy``: against the true hypothesis' labels.
- Arms: each checkpoint, an untrained control (seeded weights, the floor), logistic regression
  fitted on the same context, and the Bayes reference.

Nothing here trains. Numbers are labelled MEASURED only when produced by the command in the
openspec Progress note on real checkpoints; the unit tests are SMOKE-TEST only.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from fintfm.experiments.capability import UNTRAINED_SEED
from fintfm.experiments.sign_inference import _score_in_context, _score_logreg

if TYPE_CHECKING:
    from fintfm.modeling.model import FinancialTFM

#: Counts of disambiguating context rows on the curve (clipped to the context size).
N_DIS_GRID: tuple[int, ...] = (0, 1, 2, 4, 8, 16, 32)


@dataclass
class PosteriorPoint:
    """One arm at one number of disambiguating rows, summarised over seeds and tasks."""

    arm: str
    n_dis: int
    entropy: float
    posterior_entropy: float
    abs_err_bayes: float
    commit: float
    accuracy: float
    n_tasks: int


def hypothesis_labels(X: np.ndarray, k_hyp: int) -> np.ndarray:
    """Each hypothesis' label for every row.

    Args:
        X: Features ``(n, >= k_hyp)``.
        k_hyp: Number of hypotheses; ``h_j = 1[x_j > 0]``.

    Returns:
        Int array ``(k_hyp, n)``.
    """
    return (np.asarray(X)[:, :k_hyp] > 0).T.astype(np.int64)


def consistent_hypotheses(X: np.ndarray, y: np.ndarray, k_hyp: int) -> np.ndarray:
    """Indices of the hypotheses that label every row of ``(X, y)`` exactly.

    Args:
        X: Context features.
        y: Context labels.
        k_hyp: Number of hypotheses.

    Returns:
        Sorted int array of hypothesis indices.
    """
    return np.flatnonzero((hypothesis_labels(X, k_hyp) == np.asarray(y)[None, :]).all(axis=1))


def bayes_posterior(X: np.ndarray, y: np.ndarray, k_hyp: int, eps: float = 0.0) -> np.ndarray:
    """Closed-form posterior over the hypotheses under a uniform prior.

    Args:
        X: Context features.
        y: Context labels.
        k_hyp: Number of hypotheses.
        eps: Label-flip probability of the likelihood; 0 gives a hard consistency filter.

    Returns:
        Probability vector of length ``k_hyp``.

    Raises:
        ValueError: If ``eps`` is outside ``[0, 0.5)`` or no hypothesis has nonzero likelihood
            (at ``eps = 0``, a context no hypothesis explains).
    """
    if not 0.0 <= eps < 0.5:
        raise ValueError("eps must be in [0, 0.5)")
    wrong = (hypothesis_labels(X, k_hyp) != np.asarray(y)[None, :]).sum(axis=1)
    n = len(y)
    if eps == 0.0:
        like = (wrong == 0).astype(np.float64)
    else:
        logl = (n - wrong) * np.log1p(-eps) + wrong * np.log(eps)
        like = np.exp(logl - logl.max())
    if like.sum() == 0:
        raise ValueError("no hypothesis is consistent with the context")
    return like / like.sum()


def bayes_predictive(posterior: np.ndarray, Xq: np.ndarray, eps: float = 0.0) -> np.ndarray:
    """Posterior-weighted probability of class 1 on the query rows.

    Args:
        posterior: Probability vector over hypotheses.
        Xq: Query features.
        eps: Label-flip probability (a hypothesis predicts class 1 with ``1 - eps`` if it labels
            the row 1, ``eps`` otherwise).

    Returns:
        ``(m,)`` probabilities.
    """
    lab = hypothesis_labels(Xq, len(posterior))
    return posterior @ np.where(lab == 1, 1.0 - eps, eps)


def entropy_bits(p: np.ndarray) -> np.ndarray:
    """Binary entropy in bits, elementwise (0 at p = 0 or 1).

    Args:
        p: Probabilities.

    Returns:
        Entropies, same shape.
    """
    p = np.clip(np.asarray(p, dtype=np.float64), 0.0, 1.0)
    q = 1.0 - p
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -(np.where(p > 0, p * np.log2(p), 0.0) + np.where(q > 0, q * np.log2(q), 0.0))
    return h


def _dist_entropy_bits(w: np.ndarray) -> float:
    """Shannon entropy in bits of a probability vector (0 log 0 = 0)."""
    w = w[w > 0]
    return float(-(w * np.log2(w)).sum())


def implied_posterior(p: np.ndarray, Xq: np.ndarray, k_hyp: int) -> np.ndarray:
    """The task posterior an arm's predictions imply on the query rows.

    Args:
        p: The arm's class-1 probabilities on the queries.
        Xq: Query features.
        k_hyp: Number of hypotheses.

    Returns:
        ``w_h`` = mean probability the arm gives ``h``'s label, normalised to sum to 1.
    """
    lab = hypothesis_labels(Xq, k_hyp)
    w = np.where(lab == 1, p[None, :], 1.0 - p[None, :]).mean(axis=1)
    return w / w.sum()


def make_contexts(
    seed: int,
    true_h: int,
    k_hyp: int,
    n_ctx: int,
    n_query: int,
    d: int,
    n_dis_grid: tuple[int, ...] = N_DIS_GRID,
) -> tuple[dict[int, tuple[np.ndarray, np.ndarray]], np.ndarray, np.ndarray]:
    """Build the matched contexts for one task.

    The context for ``n_dis`` is the first ``n_dis`` disambiguating rows and the first
    ``n_ctx - n_dis`` ambiguous rows of two fixed pools, shuffled with a fixed permutation, so
    contexts along the grid differ only in how many rows carry task information.

    Args:
        seed: Random seed; the same arguments give bit-identical output.
        true_h: Index of the hypothesis that labels the data.
        k_hyp: Number of hypotheses (``2 <= k_hyp <= d``).
        n_ctx: Context rows.
        n_query: Query rows (all disambiguating).
        d: Total columns.
        n_dis_grid: Disambiguating-row counts; each is clipped to ``n_ctx``.

    Returns:
        ``(contexts, Xq, yq)``: ``contexts[n_dis] = (X, y)`` and the shared query set with the
        true hypothesis' labels.

    Raises:
        ValueError: On invalid ``k_hyp`` or ``true_h``.
    """
    if not 2 <= k_hyp <= d:
        raise ValueError("need 2 <= k_hyp <= d")
    if not 0 <= true_h < k_hyp:
        raise ValueError("true_h out of range")
    rng = np.random.default_rng(np.random.SeedSequence([seed, k_hyp, true_h]))
    need_amb, need_dis = n_ctx, n_ctx + n_query
    amb, dis = [], []
    n_amb = n_dis_rows = 0
    # Chunked rejection sampling; the chunk size is fixed so the stream is deterministic.
    for _ in range(10_000):
        X = rng.normal(size=(1024, d)).astype(np.float32)
        s = X[:, :k_hyp] > 0
        same = s.all(axis=1) | (~s).all(axis=1)
        amb.append(X[same])
        dis.append(X[~same])
        n_amb += int(same.sum())
        n_dis_rows += int((~same).sum())
        if n_amb >= need_amb and n_dis_rows >= need_dis:
            break
    amb_pool = np.concatenate(amb)[:need_amb]
    dis_pool = np.concatenate(dis)[:need_dis]
    Xq = dis_pool[n_ctx:]
    yq = hypothesis_labels(Xq, k_hyp)[true_h]
    perm = rng.permutation(n_ctx)
    contexts: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for nd in n_dis_grid:
        nd = min(nd, n_ctx)
        Xc = np.concatenate([dis_pool[:nd], amb_pool[: n_ctx - nd]])[perm]
        contexts[nd] = (Xc, hypothesis_labels(Xc, k_hyp)[true_h])
    return contexts, Xq, yq


def _metrics(p: np.ndarray, Xq: np.ndarray, yq: np.ndarray, k_hyp: int, pb: np.ndarray) -> dict:
    """Per-task metrics of one arm's query probabilities (see the module docstring)."""
    lab = hypothesis_labels(Xq, k_hyp)
    hard = (p >= 0.5).astype(np.int64)
    return {
        "entropy": float(entropy_bits(p).mean()),
        "posterior_entropy": _dist_entropy_bits(implied_posterior(p, Xq, k_hyp)),
        "abs_err_bayes": float(np.abs(p - pb).mean()),
        "commit": float((lab == hard[None, :]).mean(axis=1).max()),
        "accuracy": float((hard == yq).mean()),
    }


def run(
    model_paths: dict[str, str],
    out_dir: Path | None = None,
    seeds: tuple[int, ...] = (0, 1, 2),
    k_hyp: int = 2,
    tasks_per_seed: int | None = None,
    n_ctx: int = 200,
    n_query: int = 200,
    d: int = 8,
    n_dis_grid: tuple[int, ...] = N_DIS_GRID,
    eps: float = 0.0,
    max_context: int = 1000,
    models: dict[str, FinancialTFM] | None = None,
) -> dict:
    """Score every arm on ambiguous versus disambiguated contexts.

    Args:
        model_paths: ``{name: checkpoint path}``; one untrained control is built from the first
            one's architecture. Ignored when ``models`` is given.
        out_dir: If given, ``task_posterior.json`` is written there.
        seeds: Task seeds.
        k_hyp: Number of hypotheses.
        tasks_per_seed: Tasks per seed; the true hypothesis cycles through ``0..k_hyp-1``.
            Defaults to ``k_hyp`` so every hypothesis is the truth once per seed.
        n_ctx: Context rows (fixed across the curve).
        n_query: Query rows.
        d: Total columns.
        n_dis_grid: Disambiguating-row counts; must include 0 (the ambiguous context).
        eps: Likelihood label-flip rate of the Bayes reference.
        max_context: Context rows handed to the in-context arms.
        models: Already-built models by name (used by tests); the untrained control is added.

    Returns:
        The record: config, per-arm per-``n_dis`` results and the ambiguous-versus-resolved
        entropy comparison.

    Raises:
        ValueError: If ``d`` exceeds a model's ``max_features`` or ``n_dis_grid`` lacks 0.
    """
    import torch

    from fintfm.modeling.model import FinancialTFM

    if 0 not in n_dis_grid:
        raise ValueError("n_dis_grid must include 0 (the fully ambiguous context)")
    tasks_per_seed = k_hyp if tasks_per_seed is None else tasks_per_seed
    loaded = (
        dict(models)
        if models is not None
        else {name: FinancialTFM.load(p) for name, p in model_paths.items()}
    )
    if loaded:
        torch.manual_seed(UNTRAINED_SEED)  # seeded: an unseeded floor moves by 0.2 (§42)
        loaded["untrained_control"] = FinancialTFM(next(iter(loaded.values())).cfg)
    for name, m in loaded.items():
        if d > m.cfg.max_features:
            raise ValueError(f"d={d} exceeds {name}'s max_features {m.cfg.max_features}")

    acc: dict[tuple[str, int], dict[str, list[float]]] = {}
    for seed in seeds:
        for t in range(tasks_per_seed):
            task_seed = seed * 1000 + t
            contexts, Xq, yq = make_contexts(
                task_seed, t % k_hyp, k_hyp, n_ctx, n_query, d, n_dis_grid
            )
            for nd, (Xc, yc) in contexts.items():
                if len(np.unique(yc)) < 2:
                    continue  # a one-class context has no in-context signal to score
                post = bayes_posterior(Xc, yc, k_hyp, eps)
                pb = bayes_predictive(post, Xq, eps)
                preds = {"bayes": pb, "logistic_regression": _score_logreg(Xc, yc, Xq)}
                for name, m in loaded.items():
                    preds[name] = _score_in_context(m, Xc, yc, Xq, max_context, task_seed)
                for arm, p in preds.items():
                    met = _metrics(p, Xq, yq, k_hyp, pb)
                    if arm == "bayes":
                        met["posterior_entropy"] = _dist_entropy_bits(post)
                    a = acc.setdefault((arm, nd), {k: [] for k in met})
                    for k, v in met.items():
                        a[k].append(v)
        print(f"  seed {seed} done", flush=True)

    results = [
        PosteriorPoint(
            arm=arm,
            n_dis=nd,
            n_tasks=len(a["entropy"]),
            **{k: float(np.mean(v)) for k, v in a.items()},
        )
        for (arm, nd), a in sorted(acc.items())
    ]
    comparison = {}
    for arm in sorted({r.arm for r in results}):
        pts = {r.n_dis: r for r in results if r.arm == arm}
        if 0 in pts and len(pts) > 1:
            hi = pts[max(pts)]
            comparison[arm] = {
                "entropy_ambiguous": pts[0].entropy,
                "entropy_resolved": hi.entropy,
                "n_dis_resolved": hi.n_dis,
                "hedging_gain": pts[0].entropy - hi.entropy,
            }
    record = {
        "task": "ambiguous vs disambiguated contexts, hypotheses h_j: y = 1[x_j > 0]",
        "config": {
            "models": model_paths,
            "seeds": list(seeds),
            "k_hyp": k_hyp,
            "tasks_per_seed": tasks_per_seed,
            "n_ctx": n_ctx,
            "n_query": n_query,
            "d": d,
            "n_dis_grid": list(n_dis_grid),
            "eps": eps,
            "max_context": max_context,
        },
        "results": [asdict(r) for r in results],
        "comparison": comparison,
    }
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "task_posterior.json").write_text(json.dumps(record, indent=2))
    return record


def summarise(record: dict) -> str:
    """Render the per-arm table over ``n_dis`` and the hedging comparison.

    Args:
        record: Output of :func:`run`.

    Returns:
        A printable multi-line string.
    """
    lines = [
        (
            f"{'arm':>20} {'n_dis':>6} {'entropy':>8} {'post_H':>7} {'|p-Bayes|':>10} "
            f"{'commit':>7} {'acc':>6} {'tasks':>6}"
        )
    ]
    for r in record["results"]:
        lines.append(
            f"{r['arm']:>20} {r['n_dis']:>6} {r['entropy']:>8.3f} {r['posterior_entropy']:>7.3f} "
            f"{r['abs_err_bayes']:>10.3f} {r['commit']:>7.3f} {r['accuracy']:>6.3f} "
            f"{r['n_tasks']:>6}"
        )
    lines += ["", "Entropy on the discriminating queries, ambiguous (n_dis=0) vs most resolved:"]
    for arm, c in record["comparison"].items():
        lines.append(
            f"{arm:>20}  {c['entropy_ambiguous']:.3f} -> {c['entropy_resolved']:.3f} bits "
            f"(n_dis={c['n_dis_resolved']}), gain {c['hedging_gain']:+.3f}"
        )
    lines += [
        "",
        "Reading it: the Bayes row is the reference (1 bit and uniform posterior when ambiguous,",
        "0 once one disambiguating row is in, at eps=0). A model with a task posterior has",
        "entropy and post_H high at n_dis=0 and falling as rows are added, commit near 1/K",
        "when ambiguous. High confidence with commit near 1 on ambiguous contexts is arbitrary",
        "commitment. The untrained control is the floor; logistic regression is a point-estimate",
        "learner and shows what a single fitted rule does.",
    ]
    return "\n".join(lines)


def main() -> None:
    """Command-line entry point: ``python -m fintfm.experiments.task_posterior``."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", type=str, required=True, help="name=path pairs, comma-separated")
    p.add_argument("--out", type=str, default="runs/task_posterior")
    p.add_argument("--seeds", type=str, default="0,1,2")
    p.add_argument("--k-hyp", type=int, default=2)
    p.add_argument("--tasks-per-seed", type=int, default=None)
    p.add_argument("--n-ctx", type=int, default=200)
    p.add_argument("--n-query", type=int, default=200)
    p.add_argument("--d", type=int, default=8)
    p.add_argument("--n-dis", type=str, default=",".join(map(str, N_DIS_GRID)))
    p.add_argument("--eps", type=float, default=0.0)
    p.add_argument("--max-context", type=int, default=1000)
    args = p.parse_args()
    paths = dict(pair.split("=", 1) for pair in args.models.split(",") if pair)
    record = run(
        paths,
        Path(args.out),
        seeds=tuple(int(s) for s in args.seeds.split(",")),
        k_hyp=args.k_hyp,
        tasks_per_seed=args.tasks_per_seed,
        n_ctx=args.n_ctx,
        n_query=args.n_query,
        d=args.d,
        n_dis_grid=tuple(int(s) for s in args.n_dis.split(",")),
        eps=args.eps,
        max_context=args.max_context,
    )
    print("\n" + summarise(record))


if __name__ == "__main__":
    main()
