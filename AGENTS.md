# AGENTS.md

Quick guide for agents running RFD4-Proteina inference, search, and evaluation.
Use the workflow docs and task skills for details; do not duplicate model internals here.

## Work safely

- Run from the repository root (`.project-root` and `pyproject.toml`).
- Check `git status --short` before edits and preserve user changes. Do not discard, stage,
  commit, stash, reset, or switch branches without explicit instruction.
- Keep generated runs, checkpoints, datasets, caches, credentials, and local environment
  files out of source changes unless the user has asked for a named output directory to
  be tracked. Do not publish packages or public releases without authorization.
- Preserve existing outputs. Record inputs, effective config, code/checkpoint identity,
  seed, output directory, and job IDs without recording secrets.
- Before every commit or push, follow [Before every commit or push](#before-every-commit-or-push).

## Before every commit or push

Do this before every `git commit` and before every push to an online remote. Stop when
personal data or a secret is still in the content that would leave this machine.

- Read the staged diff before a commit. Read every commit that the push would send before
  a push. Search file names and file text.
- Remove personal data. That includes home directories, user-profile paths, machine names,
  account names, and absolute paths that identify this computer: `/home/<name>`,
  `C:\Users\<name>`, `/mnt/<letter>/Users/<name>`, and this checkout's absolute path.
  A path inside the checkout is stored relative to the repository.
- Remove secrets. That includes API keys, access tokens, passwords, private keys, and
  authorization headers, including values that start with `sk-`, `xai-`, `nvapi-`, `hf_`,
  `ghp_`, `github_pat_`, or `AKIA`, and any `Authorization: Bearer` token.
- Do not commit or push `api_keys.json`, `master.key`, `Humanitys-Last-Disease.log`,
  secret `.env` files, or anything under `~/.config/Humanitys-Last-Disease`.
- Text saved for a later upload goes through `scrub_for_upload`. A path written into a
  tracked file goes through `repository_relative`.
- A fake key that an existing test already uses may stay in that test. A real key may not.
- Do not push unless the user asks. Preparing or creating a commit does not authorize a push.
  Publishing the production report documents to the destination named below is
  authorized. Nothing else is.

## Online destination for the disease work

Online updates for this disease work go to one repository and no other:

- GitHub: `https://github.com/AiCanada/Humanitys-Last-Disease`

**That repository is public.** Hugging Face is not a destination. Do not upload the
disease work anywhere else.

Because it is public, anything pushed there is permanently readable. It is also a fork
of a public upstream, so pushed objects stay fetchable from that fork network even if
the repository is later made private. Treat every push as irreversible publication and
apply [Before every commit or push](#before-every-commit-or-push) to all of it.

Authorized: the production report documents. Not authorized without asking: the disease
source, run outputs, checkpoints, credentials, and anything else in this checkout.

This checkout's `origin` is `https://github.com/AiCanada/RFD4-Proteina.git`. Do not use
that remote for the disease work, and do not add a remote to this checkout. Clone the
destination into its own directory outside the checkout.

## Choose the workflow

Read the relevant skill under `skills/` before acting:

- `rfd4-proteina-setup`: environment, checkpoint availability, and preflight.
- `rfd4-proteina-annotated-cif`: creating or editing conditioned structure inputs.
- `rfd4-proteina-generate`: generation, partial diffusion, and inverse folding.
- `rfd4-proteina-search`: reward-guided branching and selection.
- `rfd4-proteina-evaluate`: refolding, metrics, aggregation, and structural clustering.

Start with `docs/quickstart.rst` and `docs/installation.rst`; task semantics are in
`docs/how-to/input_prep/index.md`, `docs/how-to/run-the-model.md`, `docs/how-to/search.md`, and
`docs/how-to/evaluate-with-metrics.md`.

## Environment and smoke test

Use the existing configured environment. For a new installation, follow the setup skill;
`make setup` can use the internal cluster installer, while `bash setup.sh --native`
explicitly selects a native `.venv`. Do not reinstall a working environment unnecessarily.

Local runs read `.env.local` (fallback `.env`); named clusters read `.env.<cluster>`.
Files are not merged; exported settings take precedence. `DATA_PATH` holds native
checkpoints and optional assets; HF bundles use `HF_HOME`.

Before a large run, pass the setup preflight and generate one representative example:
```bash
source .venv/bin/activate
bash skills/rfd4-proteina-setup/scripts/preflight.sh
rfd4-generate inputs=data/benchmarks/monomer/monomer-short.cif.zst num_replicates=1 out_dir=./runs/smoke
```
Use a fresh output directory. For the actual task, test its longest input at batch size 1
and verify required oracle metrics before scaling.

## Configuration and inputs

- `rfd4-generate` and `rfd4-evaluate` use Hydra `key=value` arguments, not `--key value`.
  Their roots are `configs/inference.yaml` and `configs/evaluate.yaml`; search also uses generation.
- `configs/inference.yaml` selects the release checkpoint pair on the Hugging Face Hub. A non-null `hf_ckpt_path`
  overrides native checkpoint paths (`hf_ckpt_path=null` selects them); preserve the matching AE and pin HF revisions.
- Use `configs/inference_experiment/TEMPLATE.yaml` for reusable inference configs.
  Preserve defaults ordering and `_self_` precedence; verify composition after edits.
- Prefer benchmark examples or validated annotated CIFs. Preserve bonds, annotations,
  charges, and intended conditioning; do not silently repair incomplete residues or atomize inputs.
  Check designed/fixed regions and chain of interest. Fixing sidechains also fixes identity;
  use backbone-only conditioning when redesigning sequence.
- For PPI, default to target backbone-only internal coordinates (`C_CRDGRP`), full
  target sequence (`C_SEQ`), and requested hotspots. Preserve explicit user conditions.
- For search, explicitly select metric providers needed by rewards and pass criteria.
- Keep requested sampling, recycle, conditioning, and metric settings explicit.
  Do not substitute another preset just to make a run succeed.
- Select `recycle_strategy` from the [task schedule table](docs/how-to/run-the-model.md#recycling-schedules).
  Use that task's medium preset unless another budget was requested. PPI uses front-loaded
  120/180/300 passes; MCSA/small molecules use midpeak 120/150/180; DNA uses front-loaded
  Low/Medium and flat5 High. `thinking_*` aliases are front-loaded, not task-aware defaults.

## Scheduler submissions

- Read the launcher and `scripts/.user_info_<cluster>` before submitting. Use
  `scripts/launch_generate.sh` or `scripts/launch_evaluate.sh` for batch inference/evaluation.
- Preserve `PARTITION`, `CPU_PARTITION`, `ACCOUNT`, `GRES`, `RUNTIME`, `MEM`,
  `CPUS_PER_TASK`, and the exact `SLURM_EXTRA_OPTIONS` unless explicitly overridden.
  Explicit supported user launcher flags take precedence; never silently change QoS or priority.
- State cluster, partition, account, GPU allocation, time limit, and extra options before
  submission. Inspect generated SBATCH directives when available before submitting.

## Verify results

- Outputs are under `inference_samples/` and `inference_metrics/` (or `evaluation_samples/`
  and `evaluation_metrics/`). Use `verbose=true` for diagnostics; use fresh output directories
  when changing settings so existing structures are not silently reused.
- Check generated structures and required metrics, not just job exit status. Report missing
  outputs, failed designs, and skipped oracle checks; retain failures in pass-rate denominators.
- Refolding needs the selected oracle's environment and weights. For structural clustering,
  enable `save_chain_of_interest=true` and use the evaluation skill.
- Keep input/target identity when aggregating results. Distinguish ordinary pass rates from
  clustered pass rates and record the criteria used.
- `compress_outputs=true/false` controls CIF compression. Use supported structure IO;
  never rename compressed bytes to `.cif`.
- For code/config edits, run focused tests and config-composition checks. Never weaken
  regression assertions or regenerate baselines without explicit user approval.
