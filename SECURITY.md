# Security policy

## Reporting a vulnerability

Open a [private security advisory](https://github.com/kabartay/fintfm/security/advisories/new),
or email **mukharbek.organokov@gmail.com**. Please do not open a public issue for anything
exploitable.

This is a research codebase, not a deployed service, so the realistic surface is narrow:

- **Loading a checkpoint executes `torch.load`.** It is called with `weights_only=True`, which
  is what stops a crafted file running arbitrary code — treat any change to that call as
  security-relevant. Only load checkpoints you trust the origin of.
- **Dependencies.** `scripts/check_licences.py` runs in CI, but it checks *licences*, not
  vulnerabilities. Dependabot handles updates.
- **No network calls, no credentials, no user data.** The package reads local files and
  tensors. The only credential anywhere is a Hugging Face token used by the GPU recipe, which
  lives in `.env` (gitignored) and is passed as `--secrets`, never `-e` — see
  `docs/infra/HF_JOBS.md` for why that distinction cost someone a revoked token.

## What is not a vulnerability

A model producing a wrong or badly calibrated prediction. That is a measurement question, and
the honest answer is in [`docs/paper/CLAIMS.md`](docs/paper/CLAIMS.md): this model places 93rd
of 95 on TabArena and loses to tuned gradient boosting on discrimination. **Do not deploy it
for a lending decision** without your own validation.
