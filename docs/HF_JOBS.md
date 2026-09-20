# GPU pretraining on Hugging Face Jobs

**Status: the recipe below has been executed.** Following the sibling
`finkele-axiom/scripts/hf_job_recipe.md`, every line here was run rather than inferred, and the
failures are kept because each one cost a probe and would otherwise be rediscovered. Six probes
established the working configuration; the three training runs are in flight.

## Why GPU at all

Every accuracy number in this project comes from a checkpoint of **846,818 parameters** trained
for 6,000 steps at batch 8. `docs/STRATEGY.md`'s Phase 1 target is 10-50M. That is 12-60× under
size and has never been tested, so "the model is small" is currently an untested explanation
for the 0.047 out-of-time AUC gap rather than a measured one.

There is recorded counter-evidence — `docs/FINDINGS.md` §9, that TFMs lose on large/wide
non-IID data regardless of scale — so this is a real experiment with a real chance of a null
result, not a formality.

## What the run produces

Three **classification** checkpoints at `max_features=136`, differing only in size, at matched
step count:

| name | d_cell / d_model / layers / col-layers / d_ff | parameters |
| --- | --- | --- |
| `small` | 48 / 128 / 4 / 2 / 512 | 846,818 |
| `medium` | 64 / 256 / 6 / 3 / 1024 | 4,878,146 |
| `large` | 96 / 384 / 8 / 3 / 1536 | 14,506,466 |

`small` is size-matched to `runs/v4-hazard-ldp.pt`, so it is the control for the scaling
comparison **and** the checkpoint that unblocks `public-benchmark-claim` task 33.2a — no
existing checkpoint can run V4FinBench's published protocol, because the wide one is
survival-only (§34) and the classification ones are 120 features against the protocol's 130.

**Matched steps, not matched FLOPs.** Each model sees the same number of synthetic tasks; the
larger ones consume more compute per step. That is the standard shape for a scaling curve and
it is not the same claim as §14's "matched compute", so do not conflate them when writing this
up.

## Verified so far

| line | result |
| --- | --- |
| token with no scopes | `403 ... missing permissions: job.read` |
| scopes `repo.access.read`, `repo.content.read`, `repo.write`, `job.write` | `hf jobs ps` works |
| private model repo + `hf upload` of the wheel | works; repo reports `private: True` |
| `--flavor t4-small` | `Tesla T4`, 14.74 GiB. **Scheduled immediately, every time.** |
| `--flavor l4x1` | `NVIDIA L4`, 23034 MiB. **Queued 30+ minutes** on one attempt; cancelled and re-run on T4 instead |
| `huggingface_hub[cli]` | **does not exist** as of hub 1.30: `does not provide the extra 'cli'`. The `hf` command ships in the base package |
| `requires-python = ">=3.13"` | **killed the first probe**: the image ships Python 3.12.3, so `pip install` refused the wheel. Lowered to `>=3.12`, with a CI matrix so the floor is exercised |
| `--no-deps` for our wheel | keeps the image's CUDA-matched `torch 2.12.1+cu126`; `torch.cuda.is_available()` is `True` |
| packaged config from the installed wheel | loads at `/usr/local/lib/python3.12/dist-packages/fintfm/configs/default.yaml` |
| training throughput, small (847K), T4 | **0.64 s/step** at batch 8, against 1.06 s/step on Apple Metal — about 1.7× |
| training throughput, medium (4.9M), T4 | **0.98 s/step** at batch 8 |

## Two memory lessons, both of which cost a run

**A short probe with sampled task sizes does not test the worst case.** `--n-rows-choices
256,512,1024` draws the task size *per batch*, so a 200-step probe passed and the 6,000-step
run at identical settings died of OOM six minutes in. Memory is driven by batch × **max** task
size, and the probe simply never drew 1,024. Probe the worst case explicitly:

```bash
--steps 12 --n-rows 1024 --n-rows-choices 1024      # pins every batch to the maximum
```

**Most of the apparent shortage was fragmentation.** The failing run reported **5.38 GiB
"reserved but unallocated"** — the error message names the fix and it works:

```bash
-e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

With it, every configuration fits on a T4 at the worst case that previously failed, including
small at batch 8 and large at batch 4. This is a non-secret value, so `-e` is correct here;
see the secrets section for why the distinction matters.

## Memory scales with batch × rows × features², not with parameters

The column-attention stage reshapes to ``(B·N, F, d_cell)`` and attends across **features**, so
a task of 1,024 rows at batch 8 is 8,192 sequences of 136 tokens. That, not parameter count, is
what fills the card:

| model | parameters | T4 (14.74 GiB), 1,024-row tasks | L4 (23 GiB) |
| --- | --- | --- | --- |
| small | 846,818 | fits at batch 8 | — |
| medium | 4,878,146 | fits at batch 8 | — |
| large | 14,506,466 | fits at batch **4**; batch 8 OOMs by ~1.2 GiB with only 26 MiB unallocated, so this one is real capacity rather than fragmentation | used for batch 8, to keep the scaling curve matched |

The practical consequence: **wide tables are memory-bound before they are compute-bound here**,
and a scaling study has to hold batch size constant across sizes or it is comparing two things
at once.

## The `--d-ff` trap

The training CLI had no `--d-ff` flag, so the feed-forward width stayed pinned at its 512
default while `d_model` grew. The "medium" and "large" configurations came out at 3.3M and 8.2M
parameters instead of 4.9M and 14.5M, and the FFN silently became a bottleneck as the model
widened. `--d-ff` now exists and defaults to `4 × d_model`, the transformer convention.

Nothing failed. The run completed and reported a parameter count, which is the only reason it
was caught — the same shape as `docs/FINDINGS.md` §28, where the wrong number was computed,
stored and simply not looked at.

## Scopes the token needs

**[verified]** A token with no scopes fails with `403 ... missing permissions: job.read`.
Required, on the fine-grained token:

```text
Jobs         > Start and manage Jobs
Repositories > Read contents of your repos
Repositories > Write contents/settings of your repos
Repositories > Read contents of public gated repos you can access
```

Same set as `finkele-axiom`. Note this makes the token spend-capable, which is a deliberate
change from its original dataset-read-only purpose.

## `--detach`, or a loop launches exactly one job

`hf jobs run` **attaches and streams the job's logs** unless given `-d`/`--detach`. A shell
loop that launches several runs therefore blocks on the first one's log stream forever, and
the remaining iterations never execute. Measured on 2026-09-20: a three-job `column_id_dim`
sweep registered `fintfm-colid20` and nothing else, with no error — the loop was simply still
tailing. Always pass `--detach` when launching more than one job, then poll with `hf jobs ps`.

## Secrets: `--secrets`, never `-e`

Carried over from `finkele-axiom` (their FINDINGS 1.58), where it cost a revoked token:

`-e HF_TOKEN=...` stores the value as **plaintext in the job's metadata**, and
`hf jobs inspect <id>` prints it back in full. There is no `hf jobs delete`. Use:

```bash
hf jobs run --secrets HF_TOKEN ...     # taken from the environment, encrypted server-side
```

`-e` is for values that are not secret — an output path, a run name, a step count.

## Getting our code into the container

`github.com/kabartay/fintfm` is **private**, so the job cannot `pip install git+https://...`
without a GitHub credential. The route used here is an HF repo instead:

1. `uv build --wheel` — 111 KB, pure Python, and the packaged `configs/default.yaml` is
   inside it **[verified]**.
2. Upload the wheel to a **private** HF model repo **[unverified]**.
3. Mount it read-only in the job with `-v hf://...:/build` **[unverified]**.
4. `pip install --no-deps /build/fintfm-*.whl` so the image's own torch is not replaced
   **[unverified]**.

**The repo must be private.** `CLAUDE.md` records the trained weights and the mature prior as
the private asset — the moat — and this uploads both the generator and, afterwards, the
weights to a third party. That is an acceptable trade for rented compute and it is a decision,
not a detail.

## Run command **[corrected 2026-09-17 — the earlier draft cost two 3-hour runs]**

**The `pip install` line MUST include `huggingface_hub`.** The draft below originally omitted
it, and every command in it ends with `hf upload`. A job then trains for three hours, saves the
checkpoint into the container, and dies on `bash: line 1: hf: command not found` — the
checkpoint is lost with the container, the job is billed in full, and the log's last useful
line is a successful save. Two runs were lost this way (`docs/FINDINGS.md` §90).

**Fail fast on the upload path.** Put `hf --version || exit 1` immediately after the installs,
so a broken upload costs seconds rather than the whole run. Anything a job needs *at the end*
should be checked at the *start*.

```bash
set -a; . ./.env; set +a
hf jobs run --flavor l4x1 --timeout 4h --name fintfm-clf-small \
  --secrets HF_TOKEN \
  -v hf://kabartay/fintfm-build:/build \
  pytorch/pytorch:2.12.1-cuda12.6-cudnn9-devel \
  bash -c 'pip install --break-system-packages --no-deps /build/fintfm-*.whl; pip install --break-system-packages numpy pandas scikit-learn scipy pyyaml huggingface_hub; hf --version || exit 1; fintfm-train --steps 6000 --batch-size 8 --n-rows 512 --n-rows-choices 256,512,1024 --d-cell 48 --d-model 128 --n-layers 4 --n-col-layers 2 --max-features 136 --max-classes 2 --p-financial 1.0 --device cuda --out /tmp/v4-clf-small.pt; hf upload kabartay/fintfm-build /tmp/v4-clf-small.pt v4-clf-small.pt'
```

Carried lessons from `finkele-axiom`, each of which cost a probe there:

| choice | why |
| --- | --- |
| `--timeout 4h` | Jobs default to **30 minutes** and are then killed, having been paid for |
| one-line `bash -c` | a multi-line string is read as a **filename** and dies with `File name too long` (exit 126) |
| `pytorch/pytorch:2.12.1-cuda12.6-cudnn9-devel` | Docker Hub; ships Python 3.12.3. The `2.6.0-cuda12.4` tag ships 3.11 and produces a "no matching version" error that reads like a missing package |
| `--break-system-packages` | PEP 668 marks the image's Python externally managed; the container is disposable |
| `--no-deps` for our wheel | otherwise pip may replace the image's CUDA-matched torch |
| Docker Hub image | `nvcr.io` is not an accepted registry |
| credits, not Pro | Jobs need a positive credit balance, not a subscription |

## Probe first

Run `small` alone before the other two, on the cheapest flavour that has a GPU, and read
**seconds per step** off the log. The Metal reference is 6,372 s for 6,000 steps at this size
(`docs/COMPUTE.md`). Size `medium` and `large` from the measured rate rather than from a guess
— `docs/COMPUTE.md` already records that step cost here is worse than quadratic in task size,
so extrapolation across configurations has burned this project once.

## Cost

`l4x1` is $0.80/h. If `small` matches Metal's 1h 46m, the three runs land in the region of $6-10
total. Confirm against the probe before launching the larger two.
