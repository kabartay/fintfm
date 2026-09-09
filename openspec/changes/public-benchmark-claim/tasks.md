# Tasks

- [ ] 33.1 Read V4FinBench's published protocol and baseline table into `docs/FINDINGS.md`:
      split construction, metric definitions, which models they ran, and the numbers. Verify:
      a finding recording them with the paper cited, **before** any comparison is attempted,
      so our protocol is not quietly tuned to flatter the comparison.
- [ ] 33.2 Implement their protocol as a second arm beside the out-of-time one. Verify: both
      reported by the same command, with the split difference stated in the output.
- [ ] 33.3 Three seeds on every number intended for publication, with the paired bootstrap and
      Holm correction. Verify: adjusted p-values beside every difference quoted.
- [ ] 33.4 Add the coherence measurement under their protocol, since their table lacks it.
      Verify: violation rate and fully-monotone fraction for every arm, ours and the
      baselines we can reproduce.
- [ ] 33.5 Write the report, leading with the accuracy gap rather than burying it. Verify: a
      reviewer reading only the first paragraph learns that we lose on AUC and win on
      coherence. If that is not true of the draft, the draft is wrong.
- [ ] 33.6 Only then decide the venue. Verify: a decision recorded in `docs/DECISIONS.md`
      with the alternatives considered, including "do not publish yet".
