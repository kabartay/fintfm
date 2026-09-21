# Tasks

- [ ] 41.1 **Wait on `cell-attention-and-task-inference` task 39.7's result.** Verify: this
      proposal's first pretraining task (41.3) cites the probe-separation finding that
      justifies starting it, not merely a calendar date.
- [ ] 41.2 **Design the explicit task token** as a config-gated addition
      **Prior art, found 2026-09-21: TabSwift's register tokens** (`github.com/LAMDA-Tabular/
      TabSwift`) are a published implementation — learnable tokens prepended to the ICL
      sequence, carrying dataset-level information "without interfering with the data tokens",
      discarded after the final layer so only data positions decode. Design against that rather
      than from scratch, and note it is a *dataset*-level slot, orthogonal to `column_id_dim`
      (which carries column identity and which §104 closed at 16).
      (`ModelConfig.task_token: bool = False`, matching the established additive-change pattern
      of `pooling`, `column_id_dim`, `n_cell_blocks`), pooling across context rows into one
      `z_task` that query predictions condition on. Verify: `task_token=False` reproduces every
      existing test byte-for-byte, matching how `identity_shuffle=False` and `n_cell_blocks=0`
      were regression-tested.
- [ ] 41.3 **Pretrain with and without the task token**, same protocol as every other T4
      comparison this session. Verify: task 39.7's DGP-classification probe run on both,
      reporting whether the explicit token measurably improves task-family separability, and
      the §74 Bayes-ceiling probe run on both to check it does not regress raw discrimination.
- [ ] 41.4 **Mixture-of-priors routing**, only if 41.3 shows the task token carries real
      information. Verify: `p(k|D)` is inspectable per prediction, and a held-out mixture of
      known families is scored with the routing weights reported alongside the prediction.
- [ ] 41.5 **Posterior task uncertainty on ambiguous contexts.** Verify: constructed contexts
      consistent with two task families are scored, and `p(k|D)`'s entropy is compared against
      unambiguous contexts of matched difficulty — it should be higher on the ambiguous ones.
- [ ] 41.6 **Retrieval over mechanisms**, reframing `retrieval-context`'s existing work. Verify:
      only started after 41.4 exists to retrieve into; the finding states explicitly whether
      this out-performs `retrieval-context`'s row-level retrieval or merely duplicates it.
