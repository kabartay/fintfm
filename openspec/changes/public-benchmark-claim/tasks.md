# Tasks

- [x] 33.1 Read V4FinBench's published protocol and baseline table into `docs/results/FINDINGS.md`:
      split construction, metric definitions, which models they ran, and the numbers. Verify:
      a finding recording them with the paper cited, **before** any comparison is attempted,
      so our protocol is not quietly tuned to flatter the comparison.
      **Done 2026-09-09 (`docs/results/FINDINGS.md` §36), and it changed this proposal.** Their
      protocol is **5-fold company-grouped stratified cross-validation, not out-of-time**;
      their horizon tasks are built on **different rows** than ours; their inference context
      is 10,000 rows; and their TabPFN is **fine-tuned on V4FinBench**. Our numbers are not
      comparable to theirs for four independent reasons, so 33.2 is not optional polish — it
      is the whole comparison. They release fold indices, which makes it tractable.
      **Also found: their best method pre-empts our §29 mechanism.** Prototype undersampling
      (MiniBatchKMeans over the majority class, keeping the real observation nearest each
      centroid) is the same insight, published in May 2026. What survives as distinct is that
      ours is query-conditioned rather than global. That goes at the top of any claim.
- [x] 33.2 Implement their protocol as a second arm beside the out-of-time one. Verify: both
      reported by the same command, with the split difference stated in the output.
      **Done 2026-09-09**: `fintfm-v4protocol`, reimplemented from their
      `docs/benchmark_protocol.md` and `src/v4finbench/data/folds.py` (MIT) rather than
      vendored. Reproduces the fold algebra exactly — one `RandomState(42)` consumed across
      countries in sorted order, companies shuffled within country and dealt round-robin,
      validation `fold` / test `(fold+1)%5` / train the rest — plus their 11-column drop list,
      per-fold median imputation and standardisation fitted on train only, and F1 thresholds
      calibrated on validation. Thirteen tests in `tests/test_v4_protocol.py`, including the
      off-by-one horizon mapping (**their h=0 is `company_years_h1.parquet`**) which would
      shift every result silently.
      **Reported as a separate command rather than an arm of `fintfm-v4oot`**, because the two
      protocols score different rows and merging them into one table would invite exactly the
      comparison §36 says cannot be made.
- [x] 33.2a **Done (2026-11-15, evidence): the blocker no longer holds. Classification checkpoints at `--max-features 136` exist (`runs/v4-cellattn-labels.pt`, the published recipe, 885K parameters) and `fintfm-v4protocol --no-boosting --horizon 0 --folds 0,1,2,3,4` runs a fintfm arm on the full 1,000,087-row panel with them: FINDINGS §124 (depth arms, `max_features=136`), §141 (control AP 0.2113 / 0.2050 / 0.1786 / 0.1810 / 0.2172 per fold) and §163-§164 (zero-shot mean AP 0.1986). The route taken was a new 136-feature classification checkpoint, not the 34.x joint objective. Not closed by this: 33.3-33.6.** **Blocker: no checkpoint can run this protocol.** The wide checkpoint
      (`v4-hazard-ldp.pt`, 136 features) is survival-only, so §34's guard correctly refuses
      its untrained classification head; the classification checkpoints are 120 features
      against this protocol's 130. Pretrain a classification checkpoint at
      `--max-features 136`, matched in size to `v4-hazard-ldp.pt` for comparability, or land
      `joint-objective-training` (34.x) which removes the dichotomy. Verify: `fintfm-v4protocol`
      runs a `fintfm` arm rather than printing "skipped".
- [ ] 33.3 Three seeds on every number intended for publication, with the paired bootstrap and
      Holm correction. Verify: adjusted p-values beside every difference quoted.
- [ ] 33.4 **Progress (2026-11-15, code and synthetic tests only; no real run, so the Verify
      is not met and the task stays open).** `fintfm-v4protocol --coherence-dirs` measures
      coherence from finished single-horizon runs, for every arm (fintfm and each
      per-horizon baseline) by one rule, per fold beside that arm's per-horizon AP and
      ROC-AUC. Derivation (their rows differ by horizon): a company's score at horizon *h*
      is the max (or mean, `--coherence-reduce`) of its test-row scores in horizon *h*'s test
      fold; the curve is those scores in horizon order, over companies in the test fold at
      every horizon (count reported as `n_companies`). The definition is
      `hazard.coherence_violations` plus the fully-monotone fraction of
      `term_structure.summarise_curves`, reused and pinned by test. Mean predicted and
      observed rate per horizon are printed beside it. Runs now save `company_name_test` in
      `predictions_fold*.npz`; runs made before this lack it and must be repeated. Command,
      after running each horizon on the published checkpoint with the §141 settings
      (`--horizon H` for H in 0..5, `--out runs/v4-protocol-hH`, all five folds, boosters on):
      `uv run fintfm-v4protocol --coherence-dirs
      0=runs/v4-protocol-h0,1=runs/v4-protocol-h1,2=runs/v4-protocol-h2,3=runs/v4-protocol-h3,4=runs/v4-protocol-h4,5=runs/v4-protocol-h5
      --out runs/v4-coherence` (writes `v4_coherence.json`). Tests:
      `tests/test_v4_coherence.py`.
      Add the coherence measurement under their protocol, since their table lacks it.
      Note their horizon tasks are on different rows, so a cumulative-PD curve is not
      directly defined under their construction — state how it is derived, or say plainly
      that coherence cannot be measured on their protocol and must be reported on ours.
      Verify: violation rate and fully-monotone fraction for every arm, ours and the
      baselines we can reproduce.
- [ ] 33.5 Write the report, leading with the accuracy gap rather than burying it. Verify: a
      reviewer reading only the first paragraph learns that we lose on AUC and win on
      coherence. If that is not true of the draft, the draft is wrong.
- [ ] 33.6 Only then decide the venue. Verify: a decision recorded in `docs/design/DECISIONS.md`
      with the alternatives considered, including "do not publish yet".
