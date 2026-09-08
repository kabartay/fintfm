# PD and attribution from one in-context model

## Why

Baesens et al. (arXiv:2605.18147) name the incumbent this project must displace, and it is
not gradient boosting alone: the quasi-standard in credit risk is **gradient boosting paired
with SHAP explainers**. They also name the comparison as their own future work — *"comparing
feature attributions derived from PFN and GBM pipelines using SHAP"*. It is open.

Explanations are not a nicety here. Credit scorecards are logistic regression *because*
regulators demand interpretability (`docs/DECISIONS.md` D3), adverse-action notices require a
reason per decision in several jurisdictions, and a model-risk reviewer will not accept a
score without one. A model that beats GBM on AUC but cannot say why loses to GBM+SHAP anyway.

Fonseca & Stoyanovich's ExplainerPFN (arXiv:2601.23068) shows the mechanism exists: a
TabPFN-based model pretrained on synthetic datasets **labelled with Shapley values** predicts
attributions zero-shot, with no model access, no gradients and no example explanations,
competitive with few-shot surrogates using 2-10 SHAP examples.

## What

- Emit per-feature attributions alongside PD from the same in-context forward pass, by
  extending the prior to carry attribution targets computable in closed form from the
  generative process. **Our prior knows the true data-generating weights**, so it can label
  synthetic tasks with ground-truth attributions rather than with an approximation of another
  model's behaviour — an advantage a real-data-pretrained competitor does not have.
- Validate against SHAP on gradient boosting over the real panels: agreement where both are
  applicable, and cost when they are not.
- Report attribution stability across seeds and context draws. An explanation that moves when
  the context is resampled is not an explanation a regulator can use.

## Non-goals

- Not claiming to be "true to the model". ExplainerPFN is explicit that several models can
  share predictions yet differ in Shapley decomposition, so zero-shot attribution is *true to
  the data*. Say so; the distinction is exactly the kind a reviewer catches.
- Not counterfactual or recourse explanations in v1.
- Not a UI.

## Falsified by

If zero-shot attributions disagree materially with SHAP-on-GBM where both apply, and no
stability advantage compensates, the honest answer is to ship SHAP over our own predictions
instead and drop the zero-shot ambition. That fallback is cheap and always available, which
caps the downside of trying.

## Blocked by / blocks

- **Blocked by** `phase1-prior-ablation`. Attribution from a model that cannot discriminate
  is meaningless.
- **Blocks** any serious credit-risk sales conversation, since the incumbent bundle includes
  explanations and ours currently does not.
