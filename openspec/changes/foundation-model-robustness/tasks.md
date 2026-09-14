# Tasks

- [ ] 42.1 **Irrelevant-feature robustness curve.** No dependency; can start now. Fixed task
      `y=f(x0,x1)`, noise dimensions added up to d=100, on existing checkpoints first (no new
      training). Verify: AUC-vs-d curve plotted for at least the fin00/fin10 pair from §74, to
      see whether the same architectural split shows up here too.
- [ ] 42.2 **Schema-variation sweep**, blocked on `mechanism-diverse-prior` task 40.2's
      task-family generators. Verify: recovered feature-importance ranking compared against
      the true relevant set as width grows from 5 to 200.
- [ ] 42.3 **Semantic column information, mode A vs B vs C.** No hard dependency, though most
      informative once real-panel evaluation (§73/§75) is stable. Verify: A/B/C compared on the
      same checkpoint family and the same real panels §73/§75 already measured, since this is
      specifically about transfer to tables with meaningful column names.
- [ ] 42.4 **Active context selection**, blocked on `cross-domain-coverage` task 38.13
      (retrieval's mechanism). Verify: an active-selection strategy is compared against uniform
      and against whatever retrieval fix task 38.13 produces, on the same fold, same paired-
      bootstrap methodology as §69/§71.
