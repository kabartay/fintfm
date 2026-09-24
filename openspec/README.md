# openspec

Proposals for work not yet done, so that an idea found mid-session survives the session.
Every insight worth acting on becomes a change here **before** it is implemented, and the
proposal records why, what, what is explicitly out of scope, and what would falsify the
assumption it rests on.

```
openspec/
  config.yaml          project context and the rules a proposal/tasks file must satisfy
  changes/<slug>/      what we INTEND to do
    proposal.md        Why / What / Non-goals / Blocked by-blocks / Falsified by
    tasks.md           half-day chunks, each with the command that verifies it
  specs/<capability>/  what is CURRENTLY TRUE — the living source of truth
    spec.md            numbered requirements, each naming what enforces it
  tools/
    validate.py        enforces these conventions; wired into tests/test_openspec.py
```

**`changes/` versus `specs/`** is the distinction that makes this work. `changes/` is
intent; `specs/` is current state. **A change that lands updates the spec it touches in the
same commit** — a change that leaves the spec stale has not really landed, and the specs stop
being trustworthy the first time that is allowed.

**Workflow.** An idea becomes a proposal. A proposal becomes tasks. A task is ticked only
when its stated verification passes, **with the result recorded next to it** rather than a
bare `[x]`. When a change lands, its findings go to `docs/results/FINDINGS.md` and its decisions to
`docs/design/DECISIONS.md`; the proposal keeps a `DONE` header rather than being deleted, so the
reasoning stays readable.

**The queue lives in [`docs/roadmap/NEXT.md`](../docs/roadmap/NEXT.md)**, tiered by urgency. Report the
remaining count after finishing anything, and derive it rather than recalling it:

```bash
grep -c "^- \[ \]" openspec/changes/*/tasks.md | awk -F: '{s+=$2} END {print s" open tasks"}'
```

A queue nobody counts only grows.
