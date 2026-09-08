# The queue

Ordered by urgency in tiers. Work it in tier order unless the conversation overrides that,
and prefer the cheapest item that unblocks others. Proposals live in
[`openspec/changes/`](../openspec/changes/); this file is the ordering.

**Derive the count, do not recall it:**

```bash
grep -c "^- \[ \]" openspec/changes/*/tasks.md | awk -F: '{s+=$2} END {print s" open tasks"}'
```

As of 2026-09-08, after the literature review: **12 changes.** Expect the count to rise —
finishing one item often discovers two, and that is the queue working.

---

## Tier 0 — in flight

1. **`phase1-prior-ablation`** — the run deciding whether the central bet is real. Started
   14:36 on Metal, ~3 h for three variants plus an untrained control. Tasks 1.4-1.7.

## Tier 1 — cheap, and they decide direction

These need **no GPU and no new training**, which is why they come before everything else.

2. **`pd-term-structure` task 11.1** — check whether cumulative PD across the five UCI
   horizons is even monotone. Hours of work, no training. A high violation rate founds the
   whole term-structure direction; a low one deflates it to an efficiency claim. **This is
   the single highest value-per-hour item in the queue.**
3. **`sample-efficiency-regime` tasks 13.3-13.4** — run the probe the moment a checkpoint
   exists. Every measurement so far was taken above the crossover, so this tests the regime
   the thesis needs. Implemented already; only the run remains.
4. **`forward-prediction-register` task 6.1** — establish a licensed data source with
   observable outcomes. Pure licence reading. **Out of tier order deliberately: start now**,
   because the asset is elapsed time and delay is the only way to lose it.

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
