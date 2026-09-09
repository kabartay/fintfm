# Figures and tables

What a paper needs, and the command that produces each. **A figure with no command behind it
does not go in.** Every entry states whether the underlying run exists today.

## Table 1 — Dataset

V4FinBench per-horizon instance and positive counts. **Exists**: §22 and §36 (corroborated
against the paper's own Table 1 — 1,000,087 rows at horizon 0 falling to 598,832 at horizon 5).
Cite theirs; do not re-derive.

## Figure 1 — The defect

Cumulative PD curves from per-horizon models that *decrease* with horizon, against hazard-head
curves that cannot. Per-firm, with the portfolio aggregate overlaid to show it hiding the
violations. **The paper's most important figure**, because it makes a 39% failure rate visible
that a portfolio-level view conceals.

Data exists (§11, §26, §32); the plot does not. Needs a small plotting script over
`runs/v4-oot-*/v4_out_of_time.json`.

## Table 2 — Out-of-time headline

Arms: hazard head (best configuration), uncorrected context (the permanent distortion arm),
per-horizon logistic regression, LightGBM, CatBoost, XGBoost. Columns: mean AUC, per-horizon
AUC, ECE, coherence violations, fully-monotone fraction. **The accuracy gap appears in this
table, not in later prose.**

```bash
uv run fintfm-v4oot --model runs/v4-hazard-ldp.pt --config configs/retrieval-best.yaml
```

Exists for our arms and logistic regression. **Missing: the boosters on the out-of-time split**
— `evaluation/boosting.py` runs them out-of-process (§25) but the out-of-time harness does not
call them. That is a gap and it is the comparison a reviewer wants most.

## Figure 2 — Context construction ablation

Mean AUC against context default rate, for balanced / hybrid / uniform / prototype / retrieval,
at 1,000 / 2,000 / 4,000 context rows, three seeds with error bars. Shows the monotone
relationship with the context's *rate* and the absence of one with its positive count.

```bash
uv run fintfm-ctxsweep --model runs/v4-hazard-ldp.pt --config configs/context-sweep-3seed.yaml
```

Exists (§29, §33) except the `prototype` arm, which is implemented but not yet in the sweep's
default strategy list.

## Figure 3 — Feature conditioning

Two panels: the tail diagnostic (distribution of standard-deviation-to-IQR ratio across 136
features, median 240, with 110 of 136 above 10×), and mean AUC for none / winsor / rank across
strategies and panels. **The most transferable result in the paper** (Claim 3), and the
diagnostic panel is what makes it obviously right rather than merely measured.

Data exists (§35); needs a plotting script.

## Figure 4 — Grouping approximation

Deviation from exact per-query retrieval against queries per shared context: mean and
**maximum** absolute deviation, Spearman, and AUC delta. The maximum is the point — it reaches
0.64 while the mean stays at 0.003.

```bash
uv run fintfm-retrgroup --model runs/v4-hazard-ldp.pt --n-positives 150 --n-negatives 250
```

Exists (§37), single seed, on a 37.5%-positive subsample. **Needs task 17.7** at a realistic
base rate before publication.

## Table 3 — Prior ablation

Financial prior against generic SCM prior against an untrained control of the same
architecture, at matched compute. **Exists** (§14, §15) — and note the untrained control still
ranks at 0.726, which must be reported, because it shows pretraining buys calibration far more
than ranking.

## Table 4 — Reproduction

Entry points, `configs/`, and the `config_sources` recorded in every run's output JSON. Cheap
to produce and disproportionately persuasive to the model-risk reader this project targets.

## Not a figure: the failure log

`docs/POSTMORTEM.md`. Consider an appendix. Five wrong diagnoses in one day, each caught by
measurement — unusual to publish, and the strongest available evidence that the numbers were
checked adversarially rather than defended.
