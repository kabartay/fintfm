# `src/`

A [src layout](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/):
the package lives in [`fintfm/`](fintfm/) rather than at the repository root, so tests run
against the **installed** package and cannot accidentally import from the working tree. That is
what makes the clean-install check in CI meaningful — a wheel missing its packaged config would
still pass a flat-layout test run.

The package is `fintfm` (lowercase, per PEP 8); **FinTFM** is the project name. Same split as
PyTorch/`torch` and scikit-learn/`sklearn`.

This file orients someone browsing the directory;
[`docs/design/ARCHITECTURE.md`](../docs/design/ARCHITECTURE.md) explains how the model
works, and the docstring in [`__init__.py`](fintfm/__init__.py) is the same map for someone in a
Python session.

Layout follows the pipeline, in the order data moves through it:

| package | what lives here |
| --- | --- |
| [`prior/`](fintfm/prior/) | Synthetic task generation — **the only source of pretraining data**. `financial.py` is the domain story, `scm.py` a generic causal-graph prior, `tree.py` axis-aligned structure, and `trivial.py`/`crossed.py` are diagnostic-only. |
| [`modeling/`](fintfm/modeling/) | The architecture (`model.py`), the monotone PD head (`hazard.py`), and the pretraining loop (`train.py`). |
| [`inference/`](fintfm/inference/) | In-context prediction. The sklearn-facing estimators, context construction, the base-rate correction, categorical target statistics and the binned regression path. |
| [`evaluation/`](fintfm/evaluation/) | Real datasets, calibration-aware metrics, and the benchmark harnesses. |
| [`experiments/`](fintfm/experiments/) | Designed experiments with pre-stated exit conditions, each behind a `fintfm-*` console script. |
| [`config.py`](fintfm/config.py) | The layered configuration loader. Values live in [`configs/default.yaml`](fintfm/configs/default.yaml), not scattered through the code. |

Three things are worth knowing before reading any of it:

- **`fit()` takes no gradient steps.** It stores the table as context. Everything that would be
  training in another estimator happened during pretraining, on synthetic data.
- **The prior is the asset, not the architecture.** The model is reproducible from the public
  TabPFN/TabICL literature in days; what is distinctive is the generative story in `prior/` and
  the fact that it never saw a real table.
- **Comments cite measurements.** A `§N` in this package points at entry N of
  [`docs/results/FINDINGS.md`](../docs/results/FINDINGS.md), which records how that number was
  produced. Where a default looks arbitrary, the citation is usually the reason it is not.
