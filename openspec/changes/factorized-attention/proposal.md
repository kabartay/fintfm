# Replace O(n²d) cell attention with a factorized three-stage encoder

## Why

Three separate walls this project has hit are the same wall.

`docs/results/FINDINGS.md` §78/§91 established that two-way cell attention is load-bearing: the
row-attention-within-feature stage plus per-cell labels is worth +0.0486 AP on real data
(§80), and removing the labels alone drops the model **below chance** (§91). The mechanism
works. It is also the most expensive possible way to get it.

That stage attends across all `N` rows **for every feature**, giving an attention tensor of
`(batch*F, heads, N, N)` — cost `O(n²d)` in rows and columns. Measured consequences:

| finding | wall |
| --- | --- |
| §79 | ~16 GB at N=2024, estimated ~63 GB at N=4048, with a measured 92x performance cliff between them |
| §84 | `max_context=4000` unreachable, forcing the architecture comparison to run at context 1000 |
| §94 | a 5M-parameter model cannot train at `n_rows=1024` on a 24 GB card, at any `feature_chunk` |

§81's chunking helps at inference and **not at all** in training (§94), because autograd
retains every chunk. So the cost is structural, not an implementation detail.

**An external account says the field routed around this.** Neuralk's Seldon technical post
(relayed 2026-09-19) describes the current generation — Seldon, TabPFN v3, TabICL v2 — as
factorizing attention into three stages: a **column transformer** building a latent per
column, a **row transformer** compressing each row to a single embedding, and an
**in-context block** attending across row embeddings. They state this cuts cost from naive
cell attention's `O(n²d + nd²)` to roughly `O(n² + nd²)`, "which is what makes million-row
contexts feasible". Treat the claim as a hypothesis to test rather than a fact to adopt --
but it converges with three independent measurements of our own.

## Amendment, 2026-09-19 (§95): the memory argument above is Mac-specific

The three measurements cited are all from this Mac, on CPU and MPS, where attention scores are
materialised and memory grows quadratically in N. **On CUDA, flash SDPA is active and memory
is linear in N** — measured at x1.97/x1.98/x1.99 per doubling. So §79's ~63 GB estimate and its
92x cliff do not transfer to a GPU, and the case for factorizing cannot rest on them.

What survives, and is still a real bound: memory scales with the **feature factor**
(`B * F * N * d_cell * layers`), which is why 136 features at `n_rows=1024` needs ~24.5 GB
against an L4's 23.7 and OOMs by exactly that margin (§94). And `O(n²d)` remains correct for
**compute**, so the factorization's benefit on GPU is throughput and feature scaling rather
than quadratic memory relief. Task 44.1 is now mandatory rather than a formality: the cost
model must be derived per-platform, and must retrodict linear CUDA growth as well as the
quadratic MPS behaviour.

## What changes

Restructure the encoder so per-column semantics are built **without** attending across all
rows for every feature. The target is to keep what §91 proved necessary — a cell knowing both
its column's identity and its row's label — while paying `O(n²)` rather than `O(n²d)`.

This is explicitly **not** a proposal to delete cell attention. §91 measured the unlabelled
variant at below chance, so the mechanism cannot simply be removed; it has to be reproduced
more cheaply.

## Non-goals

- **Not a claim that the factorization is better.** It is cheaper by the cited analysis. Its
  accuracy against the current architecture is unmeasured here and must be measured, not
  assumed, on the same five folds and the same Bayes-ceiling probe.
- **Not adopting any third-party implementation.** `CLAUDE.md`'s boundary stands: the
  architecture is described in a public post and may be reimplemented from the description,
  and no code or weights from Seldon, TabPFN or TabICL may enter this repository.
- **Not a prior change.** `mechanism-diverse-prior` is a separate axis and is independently
  the leading hypothesis for the accuracy deficit after the breadth result.

## Blocked by

- Nothing technically. Sequencing is the question: the breadth diagnostic (§95, pending)
  measures whether the ~0.22 AP deficit is credit-specific or general. If general, prior
  mechanism coverage competes with this for priority, and this proposal is the one that
  unblocks *scale* rather than the one that closes the gap.
