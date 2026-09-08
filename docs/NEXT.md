# The queue

Ordered by urgency in tiers. Work it in tier order unless the conversation overrides that,
and prefer the cheapest item that unblocks others. Proposals live in
[`openspec/changes/`](../openspec/changes/); this file is the ordering, not the content.

**Report the remaining count after finishing anything, and derive it rather than recalling
it:**

```bash
grep -c "^- \[ \]" openspec/changes/*/tasks.md | awk -F: '{s+=$2} END {print s" open tasks"}'
```

As of 2026-09-08: **42 open tasks across 9 changes.** It is expected to go *up* sometimes —
finishing one item often discovers two, and that is the queue working rather than failing.

---

## Tier 0 — in flight

1. **`phase1-prior-ablation`** — the experiment that decides whether the project has a
   reason to exist. 5,000-step × 3-variant run on Metal, started 2026-09-08 14:36.
   Tasks 1.4-1.7 open. Everything below is contingent on its result.

## Tier 1 — blocks any external claim

2. **`time-based-evaluation`** — every existing number uses a random split, which for credit
   data is optimistic in exactly the way a validation committee looks for. Do not publish a
   headline AUC before this.
3. **`second-credit-panel`** — all real-data results rest on one dataset, one economy, one
   crisis. A single-panel result is a result about that panel.
4. **`ci-and-release-gates`** — there is no CI. `v0.1.0` was cut from a locally-green tree
   that carries files the repository does not.

## Tier 2 — the product

5. **`fitted-calibration`** — the analytic base-rate correction has a known counterexample in
   its own evidence (1-year horizon, `FINDINGS` §6). Fix before building intervals on top.
6. **`conformal-pd-certificate`** — this is what is being sold. A port of proven machinery
   rather than research, but blocked on the two above being trustworthy first.

## Tier 3 — compounding, so start early

7. **`forward-prediction-register`** — the only asset capital cannot buy, and its entire
   value is elapsed time. **Deliberately out of tier order: start task 6.1 now**, in
   parallel, because every week of delay is a week of the asset.

## Tier 4 — scale and science

8. **`temporal-financial-prior`** — the largest known gap between the prior and the real
   task. Blocked on Phase 1: do not enrich a prior before knowing it transfers.
9. **`scaling-curve`** — Phase 2. Justifies renting NVIDIA. Blocked on Phase 1.

---

## Found while working

Discoveries append here rather than into a tier. A defect found mid-task looks more urgent
than it is because it is the thing in front of you, and stopping to re-rank the queue is how
a session ends with the queue reordered and nothing built. Triage happens when Tier 0 empties.

- **Model size is below the stated target.** `docs/STRATEGY.md` Phase 1 says 10-50M
  parameters; the current config is 2.2M. Reaching ~12M needs `--d-model 384 --n-layers 8`,
  about 1.5-2 days on Metal (ESTIMATED by scaling, not measured). Covered by task 1.6.
- **Regression head for LGD.** The model is classification-only. Loss given default is a
  regression problem, and PD alone does not give expected loss. No proposal yet.
- **Distillation for serving.** In-context inference carries the context table on every
  request and attention is quadratic in it. Kumo's published answer is a two-stage distil to
  a light serving model. Only matters once latency is a customer requirement; noted so it is
  not discovered under one.
- **Homogenisation is a real risk of the category.** If many lenders score with one model,
  their failures correlate, which is systemic risk (Bommasani et al.). Say it to a regulator
  before they say it to us. Belongs in the certificate's own caveats, so likely folds into
  `conformal-pd-certificate` rather than becoming its own change.
- **`max_context` default may be too low.** Tanna et al. report 5,000-10,000 as the sweet
  spot on credit data; the default here is 2,000, chosen for CPU cost before Metal was in
  use. Untested at the higher value.
