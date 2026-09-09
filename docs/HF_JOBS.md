# GPU pretraining on Hugging Face Jobs

**Status: setup verified up to the point of the token scope; the run recipe below is drafted
and NOT yet executed.** Lines are marked **[verified]** or **[unverified]** — the sibling
`finkele-axiom/scripts/hf_job_recipe.md` earns its authority by recording only lines that were
actually run, and this file will not claim more than it has done. Delete the unverified marks
as each line executes.

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

## Drafted run command **[unverified]**

```bash
set -a; . ./.env; set +a
hf jobs run --flavor l4x1 --timeout 4h --name fintfm-clf-small \
  --secrets HF_TOKEN \
  -v hf://kabartay/fintfm-build:/build \
  pytorch/pytorch:2.12.1-cuda12.6-cudnn9-devel \
  bash -c 'pip install --break-system-packages --no-deps /build/fintfm-*.whl; pip install --break-system-packages numpy pandas scikit-learn scipy pyyaml; fintfm-train --steps 6000 --batch-size 8 --n-rows 512 --n-rows-choices 256,512,1024 --d-cell 48 --d-model 128 --n-layers 4 --n-col-layers 2 --max-features 136 --max-classes 2 --p-financial 1.0 --device cuda --out /tmp/v4-clf-small.pt; hf upload kabartay/fintfm-build /tmp/v4-clf-small.pt v4-clf-small.pt'
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
