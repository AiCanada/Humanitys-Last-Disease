# Production report 3 — Computational Backend

**Program:** `Humanitys-Last-Disease`, the disease-binder design flow built on RFD4-Proteina.
**Scope:** what actually computes, where it runs, and what is not present.
**Status:** implemented. Not medically verified. A clear screen is a computational result;
it is not experimental validation.

## Where the code lives

Paths are relative to the repository root of the RFD4-Proteina fork carrying the disease
flow, which is not in this repository yet. The generation and evaluation entry points are
upstream and are present here.

| Component | Path |
|---|---|
| Generation entry point | `src/rfproteina/generate.py` |
| Evaluation entry point | `src/rfproteina/evaluate.py` |
| Reward-guided search | `src/rfproteina/search/` |
| Pass criteria | `configs/metrics/pass_criteria/all_pass_criteria.yaml` |
| Host sharding and command building | `src/rfproteina/disease/shards.py` |
| Job files | `src/rfproteina/disease/jobs.py` |
| Subprocess execution and cancellation | `src/rfproteina/disease/dispatch.py` |
| Remote host discovery | `src/rfproteina/disease/machines.py` |
| Declared criteria reader | `src/rfproteina/disease/refold.py` |

## Plan, then execute — a deliberate split

The disease flow never computes. It plans, writes the plan to disk, and then runs a child
process. The stages are separable and each one is inspectable before the next:

1. **Shard.** Selected structures are dealt round-robin across available hosts.
2. **Write the job.** A job file is written beside the annotated input, recording the
   class, the binder kind, the target and `generation_started: false`. It also records
   `search: null` — see below.
3. **Build commands.** Generation argv lists are constructed. The module's own comment is
   *"Nothing is executed."*
4. **Choose the model.** See *Which weights* below. The binder kind can be switched per
   run, which rebuilds a non-persistent buffer and loads no different weights.
5. **Execute.** One function spawns children, local and remote concurrently. Every child is
   registered so it can be terminated, which is what makes the MCP cancellation hook real
   rather than advisory.

**The disease package itself never touches a GPU.** The accelerator is configured for the
generation program, which the disease flow invokes as a subprocess.

### Remote hosts

Candidate hosts are offered only if they already appear in the user's own SSH
configuration and known-hosts file. **Nothing is probed over the network to discover a
host.** Transport is ordinary `ssh` and `scp`.

## Which weights

This program ships no weights of its own. It runs one of two checkpoints, chosen by whether
`--modified-ckpt` points at a file that exists.

**The intended model is a modified disease checkpoint, and it is not yet installed.** It is
a checkpoint trained with the combination projection — the class conditioning that carries
the liability axes, at twice the number of axes in the table. When such a file is named, the
overrides select the disease-binder inference experiment, clear the hub checkpoint path, and
enable the per-run binder kind. The flag's own help states the position plainly: *"Until that
file exists, commands use the original release model."*

**The current fallback is the stock upstream release**, and it is what every run to date has
used:

```
hf_ckpt_path:     nvidia/RFD4-Proteina
hf_ckpt_revision: a06e2509c7033b010fc43614e9fd840303588502   (pinned)
inference_experiment: inference_on_user_inputs
```

A metrics table from a finished run records it as such: `ckpt_path = nvidia/RFD4-Proteina`,
`ckpt_name = model.safetensors`. With `hf_ckpt_path` set to null the native pair under
`$DATA_PATH` is used instead.

### What that means for a reader of these reports

The stock release weights were **not trained with the combination projection**, and the
function that would apply it returns its input unchanged when no projection module is
present. So on the weights used so far, the disease conditioning is inert: generation is
ordinary RFD4-Proteina, and everything disease-specific in this program is in the
**screening that happens after generation** — reports 4 through 8.

A result produced today is therefore a stock-model design that has been screened, not a
design produced by a disease-conditioned model. The distinction is recorded per run in the
job file and the metrics table rather than left to be inferred.

## Reward-guided search exists, and is off for this flow

The search package implements beam search with reward models, running through the same
generation entry point and selected by an inference-experiment config.

For the disease flow it is **not used**. The job file records `search: null`, and the
disease-binder experiment config states in its own header comment: *"Beam search stays
off."* This is worth saying explicitly because the capability is present in the repository
and a reader could reasonably assume it is in the path.

## What stands in for binding assessment

This is the part most likely to be over-read, so it is stated in full.

**There is no docking.** A search of the source tree for docking programs returns nothing.
The MCP layer declares `docking` as `not_performed` with the note *"No docking program is
connected."*

**There is no predicted binding constant.** No affinity, no dissociation constant, no free
energy of binding is produced for a design. What exists is:

- **Contact molecular surface** and a **clash fraction**, computed geometrically from the
  generated structure. This is a distance-weighted interfacial surface area in the Lawrence
  and Colman sense. It is a geometry, not an affinity.
- **Declared pass criteria**, read off the per-design metrics table. The repository declares
  eleven named criteria; for a protein–protein interface the relevant one bounds a predicted
  TM-score for the chain of interest at a floor, a minimum interface predicted aligned error
  at a ceiling, and an aligned binder RMSD at a ceiling. These are refolding-oracle
  confidence numbers, not measurements of binding.

The criteria and their cuts are **read from the config file at screen time, never copied
into code**, so a recorded verdict cannot drift from the bar that produced it.

### Physics code that is present and unwired

The backend ships an explicit interface ddG implementation, a block-pair metric with a
binder/target preset, a charge-complementarity metric, Rosetta SAP scoring, and
PoseBusters checks.

**None of it is invoked by the disease flow.** Searching the disease package for any of
these returns two hits, and both are comments saying so — one noting that Rosetta SAP is
not run there, one noting which criterion bounds it. The honest summary is: the capability
is in the repository and is not in this path.

## Candidate ranking

Four different things in this program sort, and only one of them ranks designs:

| What | Where | Ranks |
|---|---|---|
| Design ranking | `mcp_layers.py` | **Designs.** Intersects the confidence metrics present on *all* traces, takes the first shared one, sorts by it. With no shared metric it falls back to filename order and records the metric as `not_recorded` — it refuses to invent a score. |
| Structure ranking | `src/rfproteina/disease/focus.py` | Input structures and targets, by role — not designs. |
| Target ranking | `src/rfproteina/disease/research.py` | Targets, by a fixed-weight combination of record-derived evidence and a model rating. |
| Follow-up | `src/rfproteina/disease/follow_up.py` | Nothing. It is **triage**: a design with no measured contact gets more replicates; a design with contact but a failed sequence check gets an inverse-folding spec. It builds argv and starts nothing. |

Design diversity is measured separately and **changes no verdict** — it reports how many
distinct designs a run really produced, so a batch of near-duplicates is not read as that
many independent shots.

## What this layer does not do

- No docking, at any stage.
- No binding affinity, dissociation constant, or binding free energy for a design.
- No reward-guided search in the disease path.
- No GPU work initiated from the disease package itself.
- No host discovered over the network.
