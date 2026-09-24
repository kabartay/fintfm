"""Phase 1: does a financial prior beat a generic one at matched compute?

This is the experiment that decides whether the project's core bet is real
(``docs/roadmap/STRATEGY.md``). Everything else — the architecture, the calibration work, the
positioning — is downstream of its answer.

**The design.** Train several models that differ in *one* variable, the pretraining task
distribution, holding architecture, parameter count, optimiser, step count, batch size, seed
and evaluation identical. Then score them all on the same held-out real credit panels.

    financial   p_financial = 1.0   only synthetic company financials
    mixed       p_financial = 0.7   the library default
    generic     p_financial = 0.0   only structural causal models, no finance at all

**The exit condition, stated before the run so it cannot be moved afterwards:** the
financial variant must beat the generic variant on real credit data, on calibration as well
as ranking. If it does not, domain-specific pretraining buys nothing here, and the honest
response is to publish that and drop to the validation layer, which needs no model of our
own. A null result is a result; it is cheaper to learn it now than after raising money on it.

**Results are written to JSON, not just printed**, with the config and git commit that
produced them, so any number in a document can be re-derived rather than recalled.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from fintfm.evaluation.datasets import load_polish_bankruptcy
from fintfm.evaluation.metrics import evaluate_binary, holm_bonferroni, paired_auc_difference
from fintfm.inference.classifier import ContextStrategy, FinancialTFMClassifier
from fintfm.modeling.model import FinancialTFM, ModelConfig
from fintfm.modeling.train import TrainConfig, train
from fintfm.prior import PriorConfig

#: The one variable under test. Everything else is held fixed across variants.
VARIANTS: dict[str, float] = {
    "financial": 1.0,
    "mixed": 0.7,
    "generic": 0.0,
}


@dataclass
class VariantResult:
    """Outcome for one pretraining variant.

    Attributes:
        name: Variant key from :data:`VARIANTS`.
        p_financial: Probability of drawing a financial task during pretraining.
        n_parameters: Model size, recorded to prove compute was matched.
        train_seconds: Wall-clock training time.
        final_loss: Mean training loss over the last logging window.
        steps: Gradient steps actually taken. Zero identifies the untrained control.
        metrics: Real-data scores keyed by ``"<dataset>/<context_strategy>"``.
    """

    name: str
    p_financial: float
    n_parameters: int
    train_seconds: float
    final_loss: float
    steps: int = 0
    metrics: dict[str, dict] = field(default_factory=dict)


def _git_commit() -> str:
    """Return the current commit, or ``"unknown"`` outside a repository."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        )
        return out.stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def evaluate_on_credit(
    model_path: str,
    horizons: tuple[int, ...] = (1, 3, 5),
    strategies: tuple[ContextStrategy, ...] = ("uniform", "balanced"),
    seed: int = 0,
) -> dict[str, dict]:
    """Score one checkpoint on the real corporate-default panels.

    Args:
        model_path: Checkpoint to evaluate.
        horizons: Bankruptcy forecast horizons, in years.
        strategies: Context-construction strategies to try.
        seed: Split seed, identical across variants so the comparison is paired.

    Returns:
        Mapping of ``"<dataset>/<strategy>"`` to a :class:`CreditMetrics` as a dict. A panel
        wider than the model is skipped with a printed note rather than raising: a narrow
        model is a legitimate configuration, and one unusable panel must not discard the
        results for every other.
    """
    out: dict[str, dict] = {}
    cfg = FinancialTFM.load(model_path).cfg
    for horizon in horizons:
        ds = load_polish_bankruptcy(horizon)
        if ds.X.shape[1] > cfg.max_features:
            print(f"    skipping {ds.name}: needs max_features >= {ds.X.shape[1]}, "
                  f"model has {cfg.max_features}")
            continue
        X_train, X_test, y_train, y_test = train_test_split(
            ds.X, ds.y, test_size=0.3, random_state=seed, stratify=ds.y
        )
        for strategy in strategies:
            clf = FinancialTFMClassifier(model_path, context_strategy=strategy).fit(
                X_train, y_train
            )
            proba = clf.predict_proba(X_test)[:, 1]
            metrics = evaluate_binary(y_test, proba)
            cell = asdict(metrics)
            # keep predictions and labels so variants can be compared as PAIRED samples
            # afterwards; a win count across cells is not a result (docs/results/FINDINGS.md §9).
            cell["_y_true"] = y_test.tolist()
            cell["_proba"] = proba.tolist()
            out[f"{ds.name}/{strategy}"] = cell
    return out


def run_ablation(
    out_dir: Path,
    steps: int,
    batch_size: int,
    n_rows: int,
    model_cfg: ModelConfig,
    seed: int = 0,
    variants: dict[str, float] | None = None,
    threads: int | None = None,
    horizons: tuple[int, ...] = (1, 3, 5),
    device: str = "cpu",
    include_untrained_control: bool = True,
) -> dict:
    """Train one model per variant at matched compute and score them identically.

    Args:
        out_dir: Directory for checkpoints and ``results.json``.
        steps: Gradient steps per variant. Identical across variants by construction.
        batch_size: Tasks per step.
        n_rows: Rows per synthetic task.
        model_cfg: Architecture, shared by every variant.
        seed: Seed for training and evaluation splits.
        variants: Override :data:`VARIANTS`.
        threads: Cap on torch CPU threads. Set this on a shared machine.
        horizons: Bankruptcy horizons to evaluate on. Pass ``()`` to train only.
        device: Training device. ``"mps"`` uses the Apple GPU, which is ~3.5x faster than
            CPU here *and* leaves the CPU cores to whatever else shares the machine — see
            ``docs/infra/COMPUTE.md`` for measured step times.
        include_untrained_control: Also evaluate a **randomly initialised, untrained** model.
            Without this control the ablation is uninterpretable when the priors tie: "the
            financial prior adds nothing over a generic one" and "no pretraining adds
            anything at all" are very different conclusions and only the control separates
            them. This is Marconi's Transfer-Gain Test (arXiv:2507.07296) and it costs no
            training time.

    Returns:
        The results record that was written to ``out_dir / "results.json"``.
    """
    if threads is not None:
        torch.set_num_threads(threads)
    variants = dict(variants or VARIANTS)
    out_dir.mkdir(parents=True, exist_ok=True)
    # steps per variant: every prior gets the same budget; the control gets none
    budgets = {name: steps for name in variants}
    if include_untrained_control:
        variants["untrained"] = 0.7  # prior is irrelevant at zero steps
        budgets["untrained"] = 0

    record: dict = {
        "created": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(),
        "platform": platform.platform(),
        "torch_threads": torch.get_num_threads(),
        "device": device,
        "config": {
            "steps": steps,
            "batch_size": batch_size,
            "n_rows": n_rows,
            "seed": seed,
            "model": asdict(model_cfg),
        },
        "variants": {},
    }

    for name, p_financial in variants.items():
        label = "UNTRAINED CONTROL" if budgets[name] == 0 else f"p_financial={p_financial}"
        print(f"\n=== variant {name!r} ({label}) ===")
        prior_cfg = PriorConfig(
            max_features=model_cfg.max_features,
            max_classes=model_cfg.max_classes,
            p_financial=p_financial,
            n_rows=n_rows,
        )
        variant_steps = budgets[name]
        train_cfg = TrainConfig(
            steps=variant_steps, batch_size=batch_size, seed=seed, device=device
        )
        ckpt = out_dir / f"{name}.pt"
        started = time.time()
        model = train(model_cfg, prior_cfg, train_cfg, str(ckpt))
        elapsed = time.time() - started

        result = VariantResult(
            name=name,
            p_financial=p_financial,
            n_parameters=model.num_parameters(),
            train_seconds=elapsed,
            final_loss=float("nan"),
            steps=variant_steps,
            metrics=evaluate_on_credit(str(ckpt), horizons=horizons, seed=seed),
        )
        record["variants"][name] = asdict(result)

    sizes = {v["n_parameters"] for v in record["variants"].values()}
    if len(sizes) > 1:
        raise RuntimeError(f"compute was not matched across variants: parameter counts {sizes}")

    (out_dir / "results.json").write_text(json.dumps(record, indent=2))
    print(f"\nwrote {out_dir / 'results.json'}")
    return record


def summarise(record: dict) -> str:
    """Render an ablation record as a comparison table.

    Args:
        record: Output of :func:`run_ablation`.

    Returns:
        A printable summary, including the verdict against the pre-stated exit condition.
    """
    variants = record["variants"]
    keys = sorted({k for v in variants.values() for k in v["metrics"]})
    lines = [
        (
            f"parameters: {next(iter(variants.values()))['n_parameters']:,} "
            f"(identical across variants) | steps: {record['config']['steps']}"
        ),
        "",
        (f"{'dataset/strategy':38s} " + " ".join(f"{n:>22s}" for n in variants)),
    ]
    for key in keys:
        cells = []
        for name in variants:
            m = variants[name]["metrics"].get(key)
            cells.append("".ljust(22) if m is None else f"AUC {m['roc_auc']:.4f} ECE {m['ece']:.4f}")
        lines.append(f"{key:38s} " + " ".join(f"{c:>22s}" for c in cells))

    if "financial" in variants and "generic" in variants:
        fin, gen = variants["financial"]["metrics"], variants["generic"]["metrics"]
        shared = [k for k in keys if k in fin and k in gen]
        if not shared:
            lines += [
                "",
                "No panel was scored, so the exit condition cannot be evaluated. Either every",
                "panel was skipped (model narrower than the data) or horizons was empty.",
            ]
            return "\n".join(lines)
        auc_wins = sum(fin[k]["roc_auc"] > gen[k]["roc_auc"] for k in shared)
        ece_wins = sum(fin[k]["ece"] < gen[k]["ece"] for k in shared)
        mean_fin = float(np.mean([fin[k]["roc_auc"] for k in shared]))
        mean_gen = float(np.mean([gen[k]["roc_auc"] for k in shared]))
        lines += [
            "",
            (
                f"financial vs generic: AUC better on {auc_wins}/{len(shared)}, "
                f"ECE better on {ece_wins}/{len(shared)} (win counts only, NOT a result)"
            ),
            (
                f"mean AUC: financial {mean_fin:.4f} vs generic {mean_gen:.4f} "
                f"(delta {mean_fin - mean_gen:+.4f})"
            ),
        ]

        # Paired bootstrap per cell, then family-wise correction. Without this a sweep over
        # panels and strategies produces a "winner" by chance roughly one time in twenty.
        rows, pvals = [], []
        for k in shared:
            y = fin[k].get("_y_true")
            if y is None or gen[k].get("_proba") is None:
                continue
            delta, (lo, hi), p = paired_auc_difference(
                np.asarray(y), np.asarray(fin[k]["_proba"]), np.asarray(gen[k]["_proba"])
            )
            rows.append((k, delta, lo, hi, p))
            pvals.append(p)
        if rows:
            survives = holm_bonferroni(pvals)
            lines += [
                "",
                (
                    "paired AUC difference (financial - generic), 95% bootstrap CI, "
                    "Holm-corrected across cells:"
                ),
            ]
            for (k, delta, lo, hi, p), ok in zip(rows, survives, strict=True):
                mark = "significant" if ok else "not significant"
                lines.append(
                    f"  {k:38s} {delta:+.4f}  [{lo:+.4f}, {hi:+.4f}]  p={p:.4f}  {mark}"
                )
            n_sig = sum(survives)
            n_sig_fav = sum(
                ok and delta > 0 for (_k, delta, _lo, _hi, _p), ok in zip(rows, survives, strict=True)
            )
            lines += [
                "",
                (
                    f"{n_sig}/{len(rows)} cells significant after correction; "
                    f"{n_sig_fav} favour the financial prior."
                ),
            ]

        lines += [
            "",
            "EXIT CONDITION (docs/roadmap/STRATEGY.md Phase 1): the financial prior must beat the",
            "generic one on real credit data, and the difference must survive the paired test",
            "above with family-wise correction. Baesens et al. (arXiv:2605.18147) found only",
            "22 of 406 pairwise comparisons significant in this domain, so expect small",
            "effects and do not read a win count as a verdict. A null result says",
            "domain-specific pretraining buys nothing here and the thesis needs revising.",
        ]
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=str, default="runs/phase1")
    p.add_argument("--steps", type=int, default=20_000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--n-rows", type=int, default=256)
    p.add_argument("--max-features", type=int, default=64)
    p.add_argument("--d-cell", type=int, default=64)
    p.add_argument("--d-model", type=int, default=192)
    p.add_argument("--n-layers", type=int, default=6)
    p.add_argument("--n-col-layers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--threads",
        type=int,
        default=None,
        help="cap torch CPU threads; set this on a shared machine (see CLAUDE.md)",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="cpu, mps (Apple GPU, ~3.5x faster and spares the CPU cores), or cuda",
    )
    p.add_argument(
        "--sample-efficiency",
        type=str,
        default=None,
        metavar="CHECKPOINT",
        help="skip training; run the sample-efficiency probe on an existing checkpoint",
    )
    p.add_argument(
        "--coherence",
        type=str,
        default=None,
        metavar="CHECKPOINT",
        help="skip training; check PD term-structure monotonicity on an existing checkpoint",
    )
    args = p.parse_args()
    if args.coherence:
        print("=== PD term-structure coherence: is cumulative PD monotone in horizon? ===")
        record = term_structure_coherence(args.coherence)
        out = Path(args.out) / "coherence.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(record, indent=2))
        print(f"\n{json.dumps(record, indent=2)}")
        print(f"wrote {out}")
        return
    if args.sample_efficiency:
        print("=== sample-efficiency probe: where does the TFM beat gradient boosting? ===")
        record = sample_efficiency_probe(args.sample_efficiency)
        out = Path(args.out) / "sample_efficiency.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(record, indent=2))
        print(f"\ncrossover (largest n where fintfm still leads): {record['crossover_size']}")
        print(f"wrote {out}")
        return

    model_cfg = ModelConfig(
        max_features=args.max_features,
        max_classes=2,
        d_cell=args.d_cell,
        d_model=args.d_model,
        n_col_layers=args.n_col_layers,
        n_layers=args.n_layers,
    )
    record = run_ablation(
        out_dir=Path(args.out),
        steps=args.steps,
        batch_size=args.batch_size,
        n_rows=args.n_rows,
        model_cfg=model_cfg,
        seed=args.seed,
        threads=args.threads,
        device=args.device,
    )
    print("\n" + summarise(record))


if __name__ == "__main__":
    main()


def sample_efficiency_probe(
    model_path: str,
    train_sizes: tuple[int, ...] = (100, 250, 500, 1000, 2000, 4000),
    horizon: int = 3,
    seed: int = 0,
    context_strategy: ContextStrategy = "balanced",
) -> dict:
    """Measure where the in-context model beats gradient boosting as training data shrinks.

    **This is the experiment the benchmark was missing.** `bench.py` evaluates on the full
    panel, which for the UCI sets is 6,000-10,500 rows. Baesens et al.
    (arXiv:2605.18147) find the TFM advantage grows as data shrinks and put the LGD
    crossover near **8,000 observations**, with substantial gains below 1,000 — so
    evaluating only at full size measures the regime the literature says we lose, and never
    the one the thesis depends on. Marconi (arXiv:2507.07296) calls this probe a
    Sample-Efficiency Probe.

    The test set is held **fixed** across all training sizes, so the only thing varying is
    how much data each model gets to learn from.

    Args:
        model_path: Checkpoint to evaluate.
        train_sizes: Training-set sizes to sweep, ascending.
        horizon: Which bankruptcy horizon to use.
        seed: Seed for the split and the subsampling.
        context_strategy: Context construction for the in-context model.

    Returns:
        A record mapping each training size to metrics for the TFM and for gradient
        boosting, plus the crossover size if one is observed.
    """
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline

    ds = load_polish_bankruptcy(horizon)
    X_pool, X_test, y_pool, y_test = train_test_split(
        ds.X, ds.y, test_size=0.3, random_state=seed, stratify=ds.y
    )
    rng = np.random.default_rng(seed)
    out: dict = {"dataset": ds.name, "test_rows": len(y_test), "sizes": {}}

    for size in train_sizes:
        if size > len(y_pool):
            continue
        # stratified subsample so tiny training sets still contain defaults at all
        pos, neg = np.flatnonzero(y_pool == 1), np.flatnonzero(y_pool == 0)
        n_pos = max(1, round(size * ds.default_rate))
        n_pos, n_neg = min(n_pos, len(pos)), min(size - n_pos, len(neg))
        idx = np.concatenate(
            [rng.choice(pos, n_pos, replace=False), rng.choice(neg, n_neg, replace=False)]
        )
        Xs, ys = X_pool[idx], y_pool[idx]
        if len(np.unique(ys)) < 2:
            continue

        tfm = FinancialTFMClassifier(model_path, context_strategy=context_strategy).fit(Xs, ys)
        arms: dict[str, object] = {
            "fintfm": evaluate_binary(y_test, tfm.predict_proba(X_test)[:, 1])
        }

        def _gbm():
            return make_pipeline(
                SimpleImputer(strategy="median"),
                GradientBoostingClassifier(random_state=seed),
            )

        arms["gboost"] = evaluate_binary(y_test, _gbm().fit(Xs, ys).predict_proba(X_test)[:, 1])

        # The calibrated arms are the honest comparison. Post-hoc calibration is cheap and
        # standard, and it is the obvious rebuttal to our calibration advantage. Its
        # *failure* at small n is itself the finding: Platt and isotonic both need held-out
        # rows containing events, which is exactly what a low-default portfolio lacks.
        for method in ("sigmoid", "isotonic"):
            key = f"gboost_{method}"
            try:
                cal = CalibratedClassifierCV(_gbm(), method=method, cv=3)
                cal.fit(Xs, ys)
                arms[key] = evaluate_binary(y_test, cal.predict_proba(X_test)[:, 1])
            except Exception as exc:  # noqa: BLE001 - the failure is data, not a defect
                arms[key] = None
                print(f"      {key}: could not be fitted ({type(exc).__name__})")

        cell: dict = {"n_train": len(ys), "n_positive_train": int(ys.sum())}
        for name, m in arms.items():
            cell[name] = asdict(m) if m is not None else None
        cell["auc_delta"] = arms["fintfm"].roc_auc - arms["gboost"].roc_auc
        out["sizes"][str(size)] = cell

        print(f"  n={len(ys):>5} ({int(ys.sum())} defaults):")
        for name, m in arms.items():
            if m is None:
                continue
            print(
                f"      {name:<18} AUC={m.roc_auc:.4f} ECE={m.ece:.4f} "
                f"Brier={m.brier:.4f} predmean={m.mean_predicted:.3%}"
            )

    # the crossover is the largest size at which the TFM still leads
    leads = [int(k) for k, v in out["sizes"].items() if v["auc_delta"] > 0]
    out["crossover_size"] = max(leads) if leads else None
    out["leads_at"] = sorted(leads)
    return out


def term_structure_coherence(
    model_path: str,
    query_horizon: int = 3,
    horizons: tuple[int, ...] = (1, 2, 3, 4, 5),
    seed: int = 0,
    context_strategy: ContextStrategy = "balanced",
) -> dict:
    """Is the predicted PD term structure monotone across horizons?

    Cumulative default probability **must not decrease** with the horizon: a firm that has
    defaulted by year 3 has defaulted by year 5. A model claiming otherwise is incoherent,
    and IFRS 9 lifetime expected credit loss consumes exactly this curve, so incoherence is
    a product defect rather than a curiosity. See ``openspec/changes/pd-term-structure``.

    **Design, and the constraint that forced it.** The UCI panels carry no company
    identifiers (``docs/results/FINDINGS.md`` §7), so a firm cannot be followed across horizon
    files. Instead the *query rows are held fixed* — one panel's held-out rows — and only
    the labelled context varies, taking each horizon's data in turn. The same firms are
    therefore scored under contexts meaning "defaults within 1 year" through "within 5
    years", and their predicted PD should rise.

    **The confound, stated rather than hidden:** the horizon files are different samples of
    firms, so contexts differ in composition as well as in label meaning. Observed base
    rates do rise with horizon (3.86%, 4.71%, 6.94% at 1, 3 and 5 years), so the expected
    direction is unambiguous even though the samples are not nested.

    Args:
        model_path: Checkpoint to evaluate.
        query_horizon: Which panel supplies the fixed query rows.
        horizons: Horizons whose labelled data supplies the context, ascending.
        seed: Split seed.
        context_strategy: Context construction.

    Returns:
        A record with mean predicted PD per horizon, the per-row violation rate, and the
        fraction of rows whose whole curve is monotone.
    """
    query_ds = load_polish_bankruptcy(query_horizon)
    _pool, X_query, _y_pool, _y_query = train_test_split(
        query_ds.X, query_ds.y, test_size=0.3, random_state=seed, stratify=query_ds.y
    )

    curves, used = [], []
    for h in horizons:
        ctx = load_polish_bankruptcy(h)
        if ctx.X.shape[1] != X_query.shape[1]:
            continue
        clf = FinancialTFMClassifier(model_path, context_strategy=context_strategy).fit(
            ctx.X, ctx.y
        )
        pd_h = clf.predict_proba(X_query)[:, 1]
        curves.append(pd_h)
        used.append(h)
        print(
            f"  context horizon {h}y (base rate {ctx.default_rate:.3%}): "
            f"mean predicted PD {pd_h.mean():.4%}"
        )

    if len(curves) < 2:
        return {"error": "fewer than two horizons evaluable", "horizons": used}

    matrix = np.stack(curves, axis=1)  # (n_query, n_horizons)
    diffs = np.diff(matrix, axis=1)
    violations = diffs < 0
    record = {
        "query_dataset": query_ds.name,
        "n_query_rows": int(matrix.shape[0]),
        "horizons": used,
        "mean_pd_by_horizon": {str(h): float(matrix[:, i].mean()) for i, h in enumerate(used)},
        "violation_rate_per_step": float(violations.mean()),
        "fully_monotone_row_fraction": float((~violations.any(axis=1)).mean()),
        "mean_pd_is_monotone": bool(
            all(
                matrix[:, i].mean() <= matrix[:, i + 1].mean() + 1e-12
                for i in range(matrix.shape[1] - 1)
            )
        ),
    }
    return record
