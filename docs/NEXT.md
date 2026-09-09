# The queue

Ordered by urgency in tiers. Work it in tier order unless the conversation overrides that,
and prefer the cheapest item that unblocks others. Proposals live in
[`openspec/changes/`](../openspec/changes/); this file is the ordering.

**Derive the count, do not recall it:**

```bash
grep -c "^- \[ \]" openspec/changes/*/tasks.md | awk -F: '{s+=$2} END {print s" open tasks"}'
```

As of 2026-09-09: **78 open tasks across 19 changes.** Expect the count to rise — finishing
one item often discovers two, and that is the queue working.

---

## Tier 0 — the top of the queue, reordered 2026-09-09

Reordered after §28-§30. The out-of-time failure that §26 blamed on the prior was our own
missing base-rate correction; fixing it cut fourth-horizon calibration error 32×, left AUC
untouched, and revealed that the context default rate — not the architecture, not the prior —
was driving the ranking. The queue follows that.

1. **`retrieval-context`** — §31 decomposed the out-of-time gap and **84% of it is
   horizon-independent**: a constant 0.132 AUC deficit against a baseline fitted on 72,622
   rows while the model sees a 2,000-row context. That is §16's crossover on real data. It
   cannot be closed by supplying more rows — uniform context peaks at 2,000 and *falls* at
   4,000, because pretraining used 256-1,024-row tasks — and pretraining on bigger tasks is
   priced out at 32× per step for 2,048 rows (`docs/COMPUTE.md`). **Choosing which 2,000 rows
   is therefore the only remaining lever on the dominant term.**

2. **`revisit-context-strategy`** — §29 reversed a published default by 10-12 AUC points on
   one seed. Task 32.1 (three seeds) is cheap and either promotes the finding or closes the
   proposal; the class default should not stay in a state one experiment contradicts. It also
   feeds item 1: §29 says the context's composition dominates, and retrieval is a claim about
   composition.

3. **`survival-context-labels`** — **demoted the same day it was written** (§31). Its honest
   ceiling is the 0.024 of excess horizon decay, not the 0.03+ its pre-registration claimed,
   and its cheap premise test came back confounded. Worth doing, second-order.

## Tier 1 — the thesis now rests on these

Reordered 2026-09-08 after the day's measurements. Accuracy and calibration were measured
small; coherence was measured large. The queue follows the evidence.

2. **`pd-term-structure` tasks 11.2-11.6** — hazard-path output and a permanent coherence
   diagnostic. §11 measured 39% of firms getting a self-contradicting curve, an order of
   magnitude larger than any accuracy edge here, and structural rather than incremental:
   nothing in the loss forbids a violation, so training cannot fix it. **This is now the
   product.**
3. **`prior-width-and-fidelity` task 15.5** — retrain at matched compute against the old
   prior. The width fix (§19) is a fidelity improvement with **unmeasured** transfer impact,
   and until this runs we do not know whether width was the binding constraint.
4. **`forward-prediction-register` task 6.1** — establish a licensed data source with
   observable outcomes. Pure licence reading, no compute. **Out of tier order deliberately:
   start now**, because the asset is elapsed time and delay is the only way to lose it.

## Tier 2 — blocks any external claim

5. **`second-credit-panel` task 7.5** — ingest V4FinBench. The only licensed panel with dates,
   so it is the only route to temporal validation. **Not** the accuracy target (`FINDINGS` §9).
6. **`time-based-evaluation`** — unblocked by the above. Every current split is random, which
   for credit is optimistic in exactly the way a supervisory reviewer looks for.
7. **`ci-and-release-gates`** — there is no CI, and `v0.1.0` was cut from a locally-green tree
   carrying files the repository does not.

## Tier 3 — the product

8. **`pd-term-structure` tasks 11.2-11.6** — hazard-path output. IFRS 9 requires lifetime ECL,
   so this is the object a regulated lender must buy rather than merely like.
9. **`fitted-calibration`** — the analytic base-rate correction has a known counterexample in
   its own evidence (`FINDINGS` §6, 1-year horizon).
10. **`conformal-pd-certificate`** — what is actually sold. Add the supervisory coverage tests
    the finance literature uses: Kupiec, Christoffersen.
11. **`zero-shot-attribution`** — the incumbent is gradient boosting **plus SHAP**, so half a
    replacement loses.

## Tier 4 — science, once direction is settled

12. **`temporal-financial-prior`** — largest known gap between prior and task; needs
    V4FinBench.
13. **`scaling-curve`** — **demoted.** Beyond IID says scale does not fix the non-IID regime,
    so this no longer justifies renting NVIDIA and is not on the critical path.

---

## Found while working

Discoveries append here rather than into a tier. Triage happens when Tier 0 empties.

- **The training loop's eval metric is uninformative.** `_eval_accuracy` reports plain
  accuracy on tasks whose base rates are 1-30%, so majority-class prediction scores 0.70-0.99.
  Replace with AUC or balanced accuracy. Cheap; do not trust the training log meanwhile.
- **Horizon independence is unverifiable.** The UCI files have no identifiers, so we cannot
  tell whether the same firm appears across horizons (`FINDINGS` §7). Any external
  presentation of "three horizons" must say the independence is assumed, not shown.
- **`max_context` default may be too low.** Tanna et al. report 5,000-10,000 as the sweet spot;
  the default here is 2,000, chosen for CPU cost before Metal was in use.
- **Two papers are known only from abstracts** and both bear on scope: Baesens et al. for the
  exact crossover and which TFMs, ExplainerPFN for how attribution targets are generated.
- **Regression head for LGD.** Expected credit loss needs PD *and* LGD; the model is
  classification-only. No proposal yet, and Baesens et al. benchmark LGD too.
- **Distillation for serving.** In-context inference carries the context on every request.
  Kumo's published answer is a two-stage distil. Only matters once latency is a requirement.
- **GraphPFN convergence risk.** If tabular foundation models can be turned into graph ones,
  Kumo's structural advantage becomes reproducible from open weights, which changes the
  competitive picture (`LANDSCAPE.md`).
- **Homogenisation** belongs in the certificate's own caveats rather than as a separate change.
