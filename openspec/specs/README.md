# specs/ — what the system guarantees now

`changes/` describes what we *intend*. This directory describes what is **currently true**,
and it is the source of truth for that. When a change lands, the spec it touches is updated
in the same commit; a change that does not update a spec has not really landed.

Each capability is one directory with a `spec.md` stating requirements in testable form.
**Every requirement names the test or command that enforces it**, because a requirement
nothing checks is a wish. `tools/validate.py` verifies that property across all specs.

| capability | why it exists |
| --- | --- |
| [pretraining-provenance](pretraining-provenance/spec.md) | the auditability claim; breakable silently by a change that improves benchmarks |
| [model-invariances](model-invariances/spec.md) | column order and padding must not change an answer |
| [calibrated-prediction](calibrated-prediction/spec.md) | a stated probability must mean what it says |
| [evaluation-honesty](evaluation-honesty/spec.md) | in an ML repo a wrong number does not crash, it looks like a result |
| [inference-options](inference-options/spec.md) | thirty off-by-default options must not leak into the published outputs |
| [experiment-decisions](experiment-decisions/spec.md) | a rule moved after the results turns a null into a candidate |
| [credit-evaluation-data](credit-evaluation-data/spec.md) | a wrong licence makes a result unpublishable; an overlap or a time leak makes it wrong |

[`METHODS.md`](METHODS.md) lists every method built, tested or evaluated from 2026-10-10 to
2026-11-10 with its status (default, adopted, lead, Lite-only, null, harmful, prepared,
deferred) and the FINDINGS section behind it. It is a register, not a spec.
