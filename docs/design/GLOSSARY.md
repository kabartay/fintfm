# Glossary — ubiquitous language

The vocabulary this project must use precisely. Credit risk terminology is **legally loaded**:
several of these terms name specific regulatory objects, and using one loosely in a document
a supervisor reads is a product defect, not a style slip (`openspec/specs/evaluation-honesty`).

**A spec or proposal that introduces a domain concept adds it here**, and corrects it if this
file is wrong. Entries marked **[verify]** are ones this project has not confirmed against a
primary regulatory source and must not be asserted to a customer or regulator until checked.

## The quantities

**PD — Probability of Default.** The probability that an obligor defaults within a stated
horizon. **The horizon is part of the definition**: a "PD of 2%" is meaningless without it.
This project's target object is the *term structure* of PD, not a single horizon.

**LGD — Loss Given Default.** The fraction of exposure expected to be lost if default
occurs. A regression target, typically bimodal and zero-one-inflated. **Not modelled here
yet**; Baesens et al. benchmark it and this repository does not.

**EAD — Exposure at Default.** The amount outstanding when default occurs. Not modelled here.

**ECL — Expected Credit Loss.** Broadly PD × LGD × EAD, discounted. Requires all three
quantities, so this project currently supplies only one input to it.

**Hazard rate.** In discrete time, the probability of default in period *t* **given survival
to** *t*. A hazard *path* (h₁ … h_K) is the object `changes/pd-term-structure` proposes to
predict; cumulative PD follows from it and must be non-decreasing in the horizon.

**Cumulative PD.** Probability of default at any point up to horizon *k*. **Must not
decrease** as *k* grows — a firm defaulted by year 3 has defaulted by year 5. Measured
violation rate in this repository: 11.0% per step, with only 60.6% of firms fully monotone
(`docs/results/FINDINGS.md` §11).

## The regulatory objects

**IFRS 9.** The accounting standard governing provisioning for credit losses. It requires
**expected** credit loss rather than incurred loss, and — the part that matters here —
**lifetime** ECL for exposures whose credit risk has increased significantly, which makes a
*term structure* of PD mandatory rather than optional. A 12-month classifier cannot satisfy
it. **[verify]** the precise stage-transition mechanics before describing them to a customer.

**SICR — Significant Increase in Credit Risk.** The IFRS 9 trigger moving an exposure from
12-month to lifetime ECL measurement. **[verify]** the operational definitions, which are
partly entity-specific.

**Basel framework.** The bank capital framework. Its internal-ratings-based approaches use
PD, LGD and EAD as inputs, which is why those three quantities have regulatory definitions
rather than merely statistical ones. **[verify]** which Basel version and which approach
before citing specifics.

**LDP — Low Default Portfolio.** A portfolio with too few observed defaults to estimate PD
reliably by any method. A **named supervisory category**, not merely a small dataset, and
supervisors expect conservatism and explicit uncertainty in response. This is the project's
target segment, and note that the field's own authoritative benchmark *named* the
low-default case as promising and **did not test it** — their datasets average a 22% default
rate (`docs/results/FINDINGS.md` §9). **[verify]** the specific supervisory expectations text.

**Point-in-time (PIT) versus through-the-cycle (TTC) PD.** A PIT PD reflects current
economic conditions; a TTC PD averages across the cycle. They are materially different
numbers for the same obligor and are used for different purposes. **Never write "PD" in a
customer-facing document without saying which**, and note that this project's prior samples a
macro regime, which makes its output conceptually closer to PIT. **[verify]** before making
any claim about which one the model produces.

**Scorecard.** The traditional credit model: usually logistic regression on binned features.
The incumbent to displace, and it is logistic regression **because** regulators demand
interpretability, which is why the accuracy bar is low and the validation bar is high.

**Model risk management / model validation.** The independent function that decides whether a
model may be used. It tests calibration, stability, documentation and out-of-time
performance. The certificate is aimed at this reader.

## The measurement vocabulary

**Discrimination.** The ability to rank defaulters above non-defaulters. AUC, Gini, KS. **Says
nothing about whether a stated probability is correct.**

**Calibration.** Agreement between stated probabilities and observed frequencies. If the model
says 2% for a thousand obligors, about twenty should default. Measured by Brier score and
expected calibration error. **This is where this project's measured advantage lies** — 2.3× to
11.7× better than gradient boosting at every dataset size (`docs/results/FINDINGS.md` §12).

**Brier score.** Mean squared error of a predicted probability. A **proper scoring rule**,
meaning it is minimised only by honest probabilities.

**ECE — Expected Calibration Error.** Average gap between predicted probability and observed
frequency, weighted by bin population. Sensitive to binning, so report the reliability bins
alongside it, not the single number alone.

**Proper scoring rule.** A loss minimised only by reporting true probabilities. Cross-entropy
and Brier are proper; accuracy and AUC are not, which is why training uses cross-entropy.

**Conformal prediction.** A distribution-free method giving a coverage guarantee under
exchangeability. **Never claim novelty on the mathematics** — it is published prior art. The
contribution is the protocol, the pre-registration and the certificate.

**Coverage, and interval width.** The fraction of true outcomes inside a stated interval.
**Never report coverage without width**: an interval from 0% to 100% has perfect coverage and
no information.

**Kupiec test / Christoffersen test.** Supervisory backtests of interval coverage, standard in
market-risk validation. Kupiec tests unconditional coverage; Christoffersen additionally tests
whether violations are independent rather than clustered — which matters in credit, where
defaults cluster in recessions.

**Escalation / refusal.** Declining to score an obligor unlike anything in the context.
**Correct behaviour, reported as such**, not a failure. A certificate that cannot say "no" is
not evidence.

## This project's own terms

**Prior.** The generative distribution of synthetic tasks the model is pretrained on. Here a
*financial prior* (company balance sheets, P&L, macro regime, default labels) mixed with a
generic structural-causal-model prior.

**In-context learning.** Prediction by conditioning on labelled examples supplied at
inference, with no gradient step. `fit()` stores the table; it does not train.

**Context.** The labelled rows handed to the model at inference. **Context construction
matters more than architecture choice** on imbalanced credit data (`docs/results/FINDINGS.md` §5).

**Context strategy.** How the context is subsampled when the table exceeds `max_context`:
`uniform`, `balanced`, or `hybrid`.

**Base-rate correction.** The logit shift undoing the class-balance distortion that
resampling the context introduces. Exact under label shift and provably ranking-preserving
(`docs/results/FINDINGS.md` §6).

**Crossover.** The training-set size at which gradient boosting overtakes the in-context
model on ranking. Measured here around 100-250 rows on an under-trained checkpoint; Baesens
et al. report roughly 8,000 for LGD. **Do not quote ours as the product's crossover** until a
properly trained model has been measured.

**Sample-efficiency probe.** The evaluation sweeping training-set size against a fixed test
set, to find that crossover.

**Term structure.** The curve of cumulative PD across horizons. The object this project aims
to predict, and what IFRS 9 lifetime ECL consumes.
