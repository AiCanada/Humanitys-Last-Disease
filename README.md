DRAFT 

Atomic level Structure AI Powered Binding Affinity 

Generative protein design with rich conditioning.

Goal: e/acc Simplified... Computation to Wet Lab. Help Guide Norms.

- Looking for Ideas or Feedback.
- Current Investigation Work Flow:
- In the programs Alpha Fold(2&3) there is an intermediary step that calculates MSA multiple sequence alignment. Many other programs use this important metric in there own code making it High Signal.
- this program uses approx. 85 different screening methods (global list of over 1500)
- Currently identifying any MSA type metric, anticipating multiple runs to gather data(weeks)
- turning the knobs on the intermediary screening metrics
- GPT-6.2-Astra ultra 5 hours for 1 test run, i-9HX, 4060, 128GB RAM
- Less Screening methods installed increases speed, Generation only takes 5-10 Min. depending on residue size.

Prepping program for entry into Drug Discovery and Development Benchmark
https://dddbench.insilico.com/?tab=core

https://huggingface.co/datasets/AICanada/humanitys_last_disease/tree/main (datasets, weights)

https://github.com/AiCanada/Humanitys-Last-Disease (Code)

Not Medically Verified.

INPUT=Disease

Then Confirm selection of Disease, and select pathway to investigate.
- Disease.
- Relevant biological pathway.
- Potential therapeutic mechanism.
- Optional target protein or gene.
The system searches for relevant biological targets and presents them in a ranked selection interface. on the choose structure page 

This is important because a disease can involve multiple pathways, proteins, and mechanisms.
For example, targeting a protein involved in disease progression is not necessarily equivalent to targeting a protein responsible for initiating the disease.

Total of (selected) diseased structures provided by program searching 8 databases is displayed with rankings of instances found:
UniProt, 
NCBI Gene, 
STRING, 
IntAct, 
Human Protein Atlas, 
PRIDE, 
PeptideAtlas, 
Therapeutic Target Database,

Any Additional diseased structures found via LLM search added to list 

Each structure is then ranked by LLM/Bot for Binding 


Post-generation screening options include:

Protein and peptide
• Immunogenicity, Calis 2013 class I 9-mer
• Proteome off-target, exact distinct 9-mer
• MHC class I, from a local predictor table
• MHC class II, from a local predictor table
• ADA, Paul et al. 2015 DR-panel breadth
• Paralog, UniRef50, with a co-evolution note beside it
• Plasma abundance, Human Protein Atlas
• Plasma hub, STRING physical score 700
• Plasma fold, from plasma_fold.tsv plus a named study
• Sequence developability, GRAVY and instability index
• Chemical liability: deamidation (NG, NS), isomerization (DG), N-glycosylation (NXS, NXT), methionine, tryptophan, free cysteine, isoelectric point, and net charge at pH 7.4
• Furin cleavage (RXKR, RXRR)
• Asp-Pro acid-labile bond
• N-terminal glutamine
• Nonstandard residues, by position
• Longest same-residue run
• Longest hydrophobic run
• Aliphatic index
• Average molecular weight

Peptide, added to the protein list
• Membrane evidence from the peptide screen or a documented caller-supplied result

Nucleic
• Off-target hybridization against a configured reference
• Innate immune motif findings, with a curated justification required for a clear
• GC fraction
• CpG count
• Longest G run

Ligand
• hERG
• Nav1.5
• CYP3A4
• CYP2D6
• CYP2C9
• CYP1A2
• CYP2C19
• P-gp
• BCRP
• BSEP
• Formula weight, from the cell or from the chemical formula

**Choosing screens for a finished candidate.** The picker recommends installed methods using the candidate’s inputs, the available model assets and the expected workload. Choose individual methods or select all installed screens. The results give a reason for each skipped method. MCP clients can inspect the same recommendations with `plan_screening` and execute them with `run_selected_screens`. [Screen selection guide](docs/reference/screen-inputs.md#choose-scientific-screens-for-a-finished-candidate).

Version 5 and later screen records distinguish incomplete refolds, interface measurements without calibrated cuts, and ligand readings by assay. The [screening reference](docs/reference/screen-inputs.md) describes the current record. The [local verification notes](docs/screening-implementation-handoff.md) give the tested scope and program sizes.

Structure and binder visual presentation after generation

OUTPUT=Target Binder, identifying promising candidates for experimental validation and potentially therapeutic development.

An MCP server (Model Context Protocol server) is a lightweight program that acts as a universal bridge, allowing AI assistants and agents (like Claude, Cursor, or custom LLMs) to securely connect to external tools, data sources, and workflows.

For better results across all frontier models mount Via Terminal: XAI, Claude, or OpenAI to download directory running program for additional support or guidance.

Mirror/Control Access to Program from android or I Phone phone

Program Architecture: 
- A. AI Interface
- OpenAI
- Claude
- xAI
- Other compatible models
- B. MCP Integration Layer
- Database retrieval.
- Protein structure retrieval.
- Sequence processing.
- Program option and structure-selection discovery.
- Generative design execution.
- Screening pipeline execution.
- Results retrieval.
- Bounded output-artifact listing and reading.
- C. Computational Backend
- Modified RFD4.
- GPU processing.
- Structural prediction.
- Docking.
- Binding assessment.
- Candidate ranking.
  
- RFD4 as base details below: (modified version out by year end)



<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/RosettaCommons/RFD4-Proteina/main/docs/_static/wordmark-dark.webp">
    <img src="https://raw.githubusercontent.com/RosettaCommons/RFD4-Proteina/main/docs/_static/wordmark.webp" width="520" alt="RFD4-Proteina">
  </picture>
</p>

<p align="center"><em>A flow-matching foundation model for biomolecular design — Baker Lab × NVIDIA GenAIR</em></p>

<p align="center">
  <a href="https://github.com/RosettaCommons/RFD4-Proteina/actions/workflows/ci.yaml"><img src="https://github.com/RosettaCommons/RFD4-Proteina/actions/workflows/ci.yaml/badge.svg?branch=main" alt="CI"></a>
  <a href="https://pypi.org/project/rfd4-proteina/"><img src="https://img.shields.io/pypi/v/rfd4-proteina?logo=pypi&amp;logoColor=white" alt="PyPI"></a>
  <a href="https://github.com/RosettaCommons/RFD4-Proteina/blob/main/pyproject.toml"><img src="https://img.shields.io/badge/python-3.12-blue?logo=python&amp;logoColor=white" alt="Python 3.12"></a>
  <a href="https://rfd4-proteina.org/latest/"><img src="https://img.shields.io/badge/docs-read-blue" alt="Documentation"></a>
  <a href="https://github.com/RosettaCommons/RFD4-Proteina/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-BSD--3--Clause-blue.svg" alt="BSD 3-Clause License"></a>
</p>

<p align="center">
  <a href="https://huggingface.co/nvidia/RFD4-Proteina"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-model-yellow" alt="Hugging Face model"></a>
  <a href="https://huggingface.co/datasets/nvidia/RFD4-Proteina"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-dataset-yellow" alt="Hugging Face dataset"></a>
  <a href="https://huggingface.co/nvidia/RFD4-Proteina-Diffusers"><img src="https://img.shields.io/badge/%F0%9F%A7%A8%20Diffusers-pipeline-orange" alt="Diffusers pipeline"></a>
  <a href="https://huggingface.co/spaces/nvidia/rfd4-proteina-zerogpu"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Space-ZeroGPU-blue" alt="ZeroGPU Space"></a>
  <a href="https://huggingface.co/spaces/nvidia/rfd4-proteina-space"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Space-dedicated%20GPU-blue" alt="Dedicated GPU Space"></a>
</p>

<p align="center">
  <a href="https://rfd4-proteina.org/">Project page</a> ·
  <a href="https://www.biorxiv.org/content/10.64898/2026.11.12.766037v1">Paper</a> ·
  <a href="#try-it-now">Try it now</a> ·
  <a href="#system-requirements">Requirements</a> ·
  <a href="#install">Installation</a> ·
  <a href="#run-the-model">Quick start</a> ·
  <a href="#more-workflows">Workflows</a> ·
  <a href="#tutorials">Tutorials</a> ·
  <a href="#screening-tools-licences-and-citations">Screening licences</a> ·
  <a href="#citation">Citation</a> ·
  <a href="https://rfd4-proteina.org/latest/">Documentation</a>
</p>

<picture>
  <source media="(max-width: 600px)" srcset="assets/readme/design-gallery-mobile.webp">
  <img src="assets/readme/design-gallery.webp" width="1050" alt="Nanobody complex, protein–DNA binder, RNA pseudoknot, ligand pocket, glycan complex, and protein assembly.">
</picture>

## Try it now

No install needed: design in your browser with the Hugging Face Studio Spaces.

- [RFD4-Proteina Studio (ZeroGPU)](https://huggingface.co/spaces/nvidia/rfd4-proteina-zerogpu): runs on your own ZeroGPU quota.
- [RFD4-Proteina Studio (dedicated GPUs)](https://huggingface.co/spaces/nvidia/rfd4-proteina-space): a shared dedicated GPU pool.

Both require a Hugging Face sign-in, and both screen designs with SecureDNA.

RFD4-Proteina designs proteins and biomolecular complexes with sequence–structure co-design,
motif and binding-site conditioning, partial diffusion, and reward-guided search.

Key features:

- **Many design tasks from one model:** monomers, protein and DNA binders, antibodies and nanobodies, enzymes and motif scaffolding, small-molecule pockets.
- **Rich conditioning:** annotated CIF inputs or text-based YAML specifications (the quickest way to start) for fixed motifs, hotspots, burial, hydrogen bonds, and more.
- **Reward-guided search:** beam search over generation paths scored by structure-prediction oracles and custom metrics.
- **Partial diffusion** to refine or diversify existing structures.
- **Fast inference:** `torch.compile` by default, with an optional BioIR backend (`inference_backend=bioir`).
- **Hugging Face release and Python API** alongside the `rfd4-generate`, `rfd4-evaluate` and training CLIs.

## System requirements

- Linux x86_64, Python 3.12, and an NVIDIA GPU with a CUDA 13-compatible driver (CUDA 12 and CPU-only containers are also available).
- Tested GPUs: see the [supported hardware table](https://rfd4-proteina.org/latest/installation.html#supported-hardware). Check yours with `nvidia-smi`.
- macOS and Windows are not supported.

## Install

**Start with `make setup`.** On a machine that meets the requirements above:

```bash
git clone https://github.com/RosettaCommons/RFD4-Proteina.git
cd RFD4-Proteina
make setup
source .venv/bin/activate
```

The repository history is large because of past notebook outputs. For a fast clone use
`git clone --depth 1` (no history) or `git clone --filter=blob:none` (full history, blobs fetched on demand).
Notebooks are committed without outputs; run `pre-commit install` to enforce this with `nbstripout`.

Setup installs the environment, caches the released Hugging Face model bundle
(about 1.7 GB, including its autoencoder), and configures `DATA_PATH` for optional data.
The cache location follows `HF_HOME`. See the
[installation guide](https://rfd4-proteina.org/latest/installation.html)
for CUDA build prerequisites, optional metrics, and cluster setup.

Setup also asks whether to add an AlphaFold 2 or AlphaFold 3 / OpenFold3 environment for
reward-guided search and refolding metrics (and, if so, your CPU architecture and CUDA version).
Skip that with `bash setup.sh --skip-search-envs`; run it later with `bash scripts/setup_search_envs.sh`.

Other options (CUDA 12, CPU-only, Docker, Apptainer, pip) are listed in the
[deployment options](https://rfd4-proteina.org/latest/installation.html#deployment-options).
For example, a container with CUDA 13 (use `RFD4_ACCELERATOR=cu12` or `cpu` for the others):

```bash
docker build -t rfd4-proteina:cu13 .
docker run --rm --gpus all rfd4-proteina:cu13 rfd4-generate --help
```

If access to the Hugging Face repositories is restricted, supply `HF_TOKEN` or use `hf auth login`
with an account that has access. For interactive login on a fresh checkout, run
`bash setup.sh --skip-weights`, activate `.venv`, log in, then run `bash setup.sh --native`.
Public repositories need no authentication.

### Quick start with pip

On Linux x86_64 with Python 3.12, an NVIDIA GPU and a CUDA 13-compatible driver. Older pip releases still
trust the compromised eMudhra/emSign CA (October 2026), so the first command installs pip 26.2.2 from a
hash-pinned URL before `uv` is downloaded:

```bash
pip install "pip @ https://files.pythonhosted.org/packages/3f/06/3e4420302363fbb71be3c7f35ce042f54bfca330e3e66b3a2e3c4cca579f/pip-26.2.2-py3-none-any.whl#sha256=fd7f4dfa8190e014013b9f08f3e973f4e4c26ffd179a204895533a7254813a70"
pip install "uv>=0.13.4"            # or: curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.12 && source .venv/bin/activate
uv pip install "rfd4-proteina[all-cuda13]"   # CUDA 12: all-cuda12; the base package alone is "rfd4-proteina"

# a binder to PD-L1: download the target, describe the design in YAML, generate
curl -LO https://raw.githubusercontent.com/RosettaCommons/RFD4-Proteina/main/docs/examples/inputs/5o45_cropped.cif.zst
cat > binder.yaml <<'EOF'
binder:
  input: 5o45_cropped.cif.zst
  contig: "50-120, /0, A17-131"
  conditions: {C_CRDGRP: {1: [{select: "sel('A/*/*/[N,CA,C,O]')"}]},
               C_SEQ: {true: [{select: "sel('A')"}]}}
EOF
rfd4-generate inference_experiment=quickstart inputs=binder.yaml out_dir=./runs/first
```

The `quickstart` preset skips scoring and the `torch.compile` warm-up so a first run finishes quickly; drop it for larger runs.

The environment is about 7 GB, mostly PyTorch and the CUDA libraries. `uv` downloads in parallel and caches wheels, so it installs in
about two minutes on a fresh cluster node where plain `pip` takes about ten; `pip install` works too, only more slowly.

The model can also be run as a Hugging Face Diffusers pipeline. The [quickstart](https://rfd4-proteina.org/latest/quickstart.html) covers all three routes in order: GitHub clone, PyPI and Diffusers.

The CUDA extra also installs the fused cuEquivariance kernels the model loads, so use it even if you skip metrics. Add `inference_backend=bioir` to use the BioIR runtime (included in `all-cuda13`). `all-cuda12` and the plain package
use the native runtime. If `pip` cannot see a very recent release, retry with `--no-cache-dir`.

For the single-worker Docker/Kubernetes API, see the [deployment guide](deploy/README.md).

## Run the model

After setup, describe a design in a small YAML [condition spec](https://rfd4-proteina.org/latest/how-to/input_prep/yaml-json.html).
This one asks for a 50-120 residue binder to PD-L1, using an input included in the repository.
Save it as `binder.yaml` in the repository root:

```yaml
binder:
  input: docs/examples/inputs/5o45_cropped.cif.zst
  contig: "50-120, /0, A17-131"
  conditions: {C_CRDGRP: {1: [{select: "sel('A/*/*/[N,CA,C,O]')"}]},
               C_SEQ: {true: [{select: "sel('A')"}]}}
```

and run it:

```bash
rfd4-generate inputs=binder.yaml num_replicates=2 out_dir=./runs/first '~metrics'
```

JSON with the same keys works too. Designs are written to `runs/first/inference_samples/`.
`'~metrics'` skips scoring, so no refolding environment is needed (quote it so zsh keeps the `~`).
Add `compile_lengths=null` to a first smoke test to skip the `torch.compile` warm-up.
Next: [design a binder in 10 lines](https://rfd4-proteina.org/latest/how-to/binder-in-10-lines.html),
the [cookbook](https://rfd4-proteina.org/latest/how-to/cookbook.html) with a recipe per task (monomers,
binders, enzymes, nanobodies, small molecules, partial diffusion), or your own
[annotated input](https://rfd4-proteina.org/latest/how-to/input_prep/index.html).

### Hugging Face and Python

To load the model from the Hugging Face Hub, with no local checkpoints and no `DATA_PATH`, pass
`hf_ckpt_path`; use a different Hub repo or a local exported bundle the same way:

```bash
rfd4-generate hf_ckpt_path=nvidia/RFD4-Proteina \
    inference_experiment=quickstart inputs=design.yaml out_dir=./runs/hf
```

Install it with `uv pip install rfd4-proteina` on a
supported CUDA/Python system. The same API works from an installed development checkout:

```python
import rfd4_proteina as rfd4

model = rfd4.Proteina.from_pretrained("nvidia/RFD4-Proteina")
rfd4.generate(model, overrides=["inference_experiment=quickstart", "inputs=design.yaml", "out_dir=runs/hf"])
```

Use an annotated CIF or condition-spec YAML as `inputs`. Add `revision="<commit>"`
to `from_pretrained` (or `hf_ckpt_revision=<commit>` to the CLI) for reproducibility.
HF caches downloaded weights; `rfd4.generate` reuses the loaded model.
For a native local checkpoint, set `hf_ckpt_path=null`,
`ckpt_path=/path/to/checkpoints`, `ckpt_name=model.ckpt`, and
`autoencoder_ckpt_path=/path/to/ae.ckpt`.
Training datasets are separate from model downloads: see the
[training guide](https://rfd4-proteina.org/latest/contributing/developer-guide.html).

- [Running the model](https://rfd4-proteina.org/latest/how-to/run-the-model.html): binders, motif scaffolding, partial diffusion, and sampling settings.
- [Input preparation](https://rfd4-proteina.org/latest/how-to/input_prep/index.html) · [Text-based design specifications](https://rfd4-proteina.org/latest/how-to/input_prep/yaml-json.html)
- [Scoring and refolding](https://rfd4-proteina.org/latest/how-to/evaluate-with-metrics.html) · [Reward-guided search](https://rfd4-proteina.org/latest/how-to/search.html)
- [Training and fine-tuning](https://rfd4-proteina.org/latest/contributing/developer-guide.html) · [Worked examples](https://rfd4-proteina.org/latest/examples/index.html)

## More workflows

- **Binders and motif scaffolding:** condition on an annotated CIF or a YAML design spec; see [Running the model](https://rfd4-proteina.org/latest/how-to/run-the-model.html).
- **Reward-guided search:** select a search `inference_experiment` config to branch and prune candidates against an oracle; see [Reward-guided search](https://rfd4-proteina.org/latest/how-to/search.html).
- **Partial diffusion:** re-noise and refine an existing design; see [Partial diffusion](https://rfd4-proteina.org/latest/how-to/partial-diffusion.html).
- **Scoring and refolding:** run `rfd4-evaluate` on generated structures; see [Scoring and refolding](https://rfd4-proteina.org/latest/how-to/evaluate-with-metrics.html).
- **Inference backends:** `native` (default, compiled or eager) or `bioir` via `inference_backend=`; see [Inference backends](https://rfd4-proteina.org/latest/how-to/advanced/backends.html).

## Tutorials

- [Python API tutorial](https://github.com/RosettaCommons/RFD4-Proteina/blob/main/docs/examples/python_api_tutorial.ipynb): build and validate conditioned structure inputs.
- [Enzyme design tutorial](https://github.com/RosettaCommons/RFD4-Proteina/blob/main/docs/examples/enzyme_design_tutorial.ipynb): motif scaffolding and catalytic-site metrics, end to end.
- [PPI specificity tutorial](https://github.com/RosettaCommons/RFD4-Proteina/blob/main/docs/examples/ppi_specificity_tutorial.ipynb): protein-binder design and specificity checks.
- [Antibody campaign examples](https://github.com/RosettaCommons/RFD4-Proteina/blob/main/examples/antibody_design/README.md): cropped PMVK/MZB1 VHH inputs, hotspot sampling, high-early recycling, and partial refinement.
- [Agent skills](https://github.com/RosettaCommons/RFD4-Proteina/blob/main/skills/README.md): task-specific guides for setup, generation, search, and evaluation.

## Screening tools: licences and citations

The screens that run after generation wrap third-party tools, models and reference data.
Each tool is fetched by a script in `scripts/` into its own directory under `DATA_PATH`;
DockQ, MHCflurry, Commec, ThermoMPNN and CrossAbSense each get their own virtual
environment there, because their pinned dependencies conflict with this project's. Two
tools' sources are vendored rather than fetched — CrossAbSense and ThermoMPNN, under
[`screening subpackage/`](screening%20subpackage/README.md), each pinned to an upstream
commit and byte-identical to it. No weights are vendored anywhere. A screen that cannot find its tool records `untested` with a blocker
naming what is missing — it never substitutes a guess.

Every licence below was read from the tool's own distribution: a shipped `LICENSE` file, a
`License-Expression` field, or an OSI classifier in the installed package metadata. Where a
project's PyPI metadata and its repository `LICENSE` disagree, the repository file governs
and the disagreement is recorded. Per-screen method, thresholds and threshold provenance
are in [`docs/screening-implementation-handoff.md`](docs/screening-implementation-handoff.md),
install routes in [`docs/installation.rst`](docs/installation.rst), and the machine-readable
records — including the date each licence was read and the quote it was read from — in
`configs/metrics/`.

### Tools installed and exercised

| Tool | Version | Licence | Role in the screens |
| --- | --- | --- | --- |
| [ANARCII](https://github.com/oxpig/ANARCII) | 2.0.8 | BSD-3-Clause | IMGT numbering, which every antibody screen rests on |
| [DeepSP](https://github.com/Lailabcode/DeepSP) | published weights | MIT | 30 sequence-derived spatial aggregation and charge descriptors |
| [DeepViscosity](https://github.com/Lailabcode/DeepViscosity) | 102-model ensemble | MIT | viscosity class at 150 mg/mL (≤ 20 cP or > 20 cP) |
| [CrossAbSense](https://huggingface.co/SimonCrouzet/CrossAbSense) | v0.9 | Apache-2.0 | HIC, polyreactivity, AC-SINS, Tm2 and titer oracles |
| [promb](https://github.com/MSDLLCPapers/promb) | 1.0.2 | MIT | the OASis 9-mer peptide set behind the humanness score |
| [AbLang-2](https://github.com/oxpig/AbLang2) | 0.2.1 | BSD-3-Clause | antibody pseudo-log-likelihood — naturalness, not humanness |
| [MHCflurry](https://github.com/openvax/mhcflurry) | 2.3.30 | Apache-2.0 | MHC class I presentation; own venv at `DATA_PATH/mhcflurry` |
| [DockQ](https://github.com/bjornwallner/DockQ) | 2.1.3 | MIT | interface quality against a reference; own venv at `DATA_PATH/dockq` |
| [ProLIF](https://github.com/chemosim-lab/ProLIF) | 2.1.0 | Apache-2.0 | typed protein–ligand interaction fingerprints |
| [RDKit](https://pypi.org/project/rdkit/) | 2025.9.6 | BSD-3-Clause | cheminformatics throughout the ligand screens |
| [PoseBusters](https://github.com/maabuu/posebusters) | 0.6.5 | MIT | pose validity checks |
| [ADMET-AI](https://github.com/swansonk14/admet_ai) | 2.0.1 | MIT | seven of the ten ligand-safety panel axes |
| [Chemprop](https://pypi.org/project/chemprop/) | 2.3.1 | MIT | the D-MPNN inside ADMET-AI, and the scaffold-balanced splitter |
| [ThermoMPNN](https://github.com/Kuhlman-Lab/ThermoMPNN) | commit `370f76e` | MIT | ΔΔG for point mutations |
| [MMseqs2](https://github.com/soedinglab/MMseqs2) | commit `6b89b5c` | MIT | alignment-based proteome off-target search |
| [Commec](https://github.com/ibbis-bio/common-mechanism) | 2.1.0 | MIT | biosecurity screening; own venv at `DATA_PATH/commec` |
| [biotite](https://pypi.org/project/biotite/) | 1.6.0 | BSD-3-Clause | structure parsing in the interface and pose screens |

The `deepsp`, `deepviscosity` and `ligand-regressor` extras add `h5py` 3.16.0, `joblib`
1.6.0 and `scikit-learn` 1.9.1, all BSD-3-Clause.

**PoseBusters is a worked example of why the repository file governs.** Its PyPI classifier
is `License :: OSI Approved :: BSD License`, but the `LICENSE` shipped inside the installed
0.6.5 wheel is the MIT licence, "Copyright (c) 2025 Martin Buttenschoen". MIT is recorded
above because that is the file that travels with the code.

**One copyleft dependency arrives indirectly.** ProLIF is Apache-2.0, but it requires
MDAnalysis, which is **LGPL-3.0-or-later** (`MDAnalysis` 2.11.0, from its own classifier),
so installing the `prolif` extra puts an LGPL library in the environment. The interaction
screen keeps a native fallback and does not require ProLIF.

**MMseqs2 was relicensed from GPL-3.0 to MIT in November 2024.** The installed build
postdates that change; an older MMseqs2 would not carry the MIT terms recorded here.

Of the refolding oracles configured under `configs/metrics/refolding_oracles/`, two are
permissive end to end and record their own terms in the config header: Boltz is MIT, and
Chai-1 is Apache-2.0 for **both code and model weights**, commercial use included — the
least licence-encumbered AF3-class oracle available. Both were read from their repositories
on 2026-11-16.

### Reference data

| Dataset | Licence | Used for |
| --- | --- | --- |
| [IEDB](https://www.iedb.org/) class II DR panel export | CC BY 4.0 | training the class II binding predictor fitted in this repository |
| [TEPITOPE matrices](https://github.com/BenPorebski/tepitope) | The Unlicense | class II DR pocket-profile scoring |
| [Octant CYP inhibition and reactivity release](https://huggingface.co/datasets/openadmet/Octant_CYP_inhibition_reactivity_blog_release) | CC BY 4.0 | measured CYP readings for the ligand panel |
| OASis v1 ([Zenodo 5164685](https://doi.org/10.5281/zenodo.5164685)) | CC BY 4.0 | the human 9-mer peptide set, reached through `promb`'s packaged subset |
| UniProt human proteome `UP000005640_9606` | CC BY 4.0 | the off-target search target |
| UniProt sequences `P02768`, `P01857`, `P0CG47` | CC BY 4.0 | serum albumin, the IgG1 constant region and ubiquitin-B: the tolerated-protein baseline the MHC class I and class II axes measure a design's peptide density against |
| Commec biorisk and benign databases | MIT | biosecurity screening |
| [BindingDB](https://www.bindingdb.org/) | BindingDB terms | training data for the three panel axes ADMET-AI has no endpoint for |
| ChEMBL-derived panel readings | CC BY-SA 3.0 | measured ligand-panel activities |

IEDB declares CC BY 4.0 as `dcterms:license` in the JSON-LD metadata of its export page
rather than in prose. That is the source relied on, read 2026-11-17, and the quote is kept
on disk beside the downloaded table. BindingDB and ChEMBL rows carry their licence with
them, so a record built from either names its per-row reference rather than leaning on
this note.

### Documented but not installed

| Tool | Licence | Why it is not installed |
| --- | --- | --- |
| [Foldseek](https://github.com/steineggerlab/foldseek) | GPL-3.0 | the structural novelty screen is written against it, but the binary and its databases (1.5 GB for Swiss-Prot up to 114 GB for AlphaFold/UniProt50) are operator-fetched |
| [BioPhi](https://github.com/Merck/BioPhi) | MIT | conda-only and a 21.5 GB database; the OASis score is reached through `promb` instead |
| [BayeshERG](https://github.com/GIST-CSBL/BayeshERG) | **not verified** | not installed, and its licence has not been read at source — it must be before the hERG route is used |

### Excluded on licence grounds

These were evaluated and deliberately left out. Each licence was read from the project's own
file; the date and the quote are recorded in
`configs/metrics/antibody_developability/antibody_developability.yaml`.

| Tool | Licence | Note |
| --- | --- | --- |
| NetMHCIIpan 4.3 | academic-use download; commercial use by separate arrangement | this is the consensus scale the ADA breadth threshold of 20 is actually defined on, so that threshold is not applied to a substitute predictor |
| MixMHC2pred | academic non-commercial | — |
| MHCnuggets | JHU Academic Software License, non-commercial | its PyPI metadata says "Apache License", contradicting the repository `LICENSE`. The repository file governs. A licence read off PyPI alone would have wired this in by mistake |
| IgFold | academic use only | ImmuneBuilder (BSD-3-Clause) is the structure-prediction replacement |
| DeepSCM | CC BY-NC | DeepViscosity covers the same endpoint without the restriction |

A permissive wrapper is not a permissive predictor: `epytope` (BSD-3-Clause) and `mhctools`
(Apache-2.0) are permissive frameworks that call the tools above, so the bundled predictor's
terms still apply and they are not a route around this list. For the same reason the
numbering tool here is ANARCII (BSD-3-Clause) and not the older ANARCI from the same group,
which is GPL-3.0.

### Cite these when a screen is reported

Cite the tools whose screens you report. Author lists, journals and DOIs below were
verified against PubMed on 2026-11-18, except ANARCII's, which is a preprint and is quoted
from the citation block in the installed package's own metadata.

- **DeepSP** — Kalejaye LA, Wu I-E, Terry T, Lai P-K. *DeepSP: Deep learning-based spatial properties to predict monoclonal antibody stability.* Comput Struct Biotechnol J 23:2220–2229 (2024). [10.1016/j.csbj.2024.05.029](https://doi.org/10.1016/j.csbj.2024.05.029)
- **DeepViscosity** — Kalejaye LA, Chu J-M, Wu I-E, et al., Lai P-K. *Accelerating high-concentration monoclonal antibody development with large-scale viscosity data and ensemble deep learning.* mAbs 17(1):2483944 (2025). [10.1080/19420862.2025.2483944](https://doi.org/10.1080/19420862.2025.2483944)
- **ANARCII** — Greenshields-Watson A, Agarwal P, Robinson SA, et al., Deane CM. *ANARCII: A generalised language model for antigen receptor numbering.* bioRxiv 2025.04.16.648720. [10.1101/2025.04.16.648720](https://doi.org/10.1101/2025.04.16.648720)
- **AbLang-2** — Olsen TH, Moal IH, Deane CM. *Addressing the antibody germline bias and its effect on language models for improved antibody design.* Bioinformatics 40(11) (2024). [10.1093/bioinformatics/btae618](https://doi.org/10.1093/bioinformatics/btae618)
- **BioPhi and OASis** — Prihoda D, Maamary J, Waight A, et al., Bitton DA. *BioPhi: A platform for antibody design, humanization, and humanness evaluation based on natural antibody repertoires and deep learning.* mAbs 14(1):2020203 (2022). [10.1080/19420862.2021.2020203](https://doi.org/10.1080/19420862.2021.2020203)
- **MHCflurry 2.0** — O'Donnell TJ, Rubinsteyn A, Laserson U. *MHCflurry 2.0: Improved pan-allele prediction of MHC class I-presented peptides by incorporating antigen processing.* Cell Syst 11(1):42–48.e7 (2020). [10.1016/j.cels.2020.06.010](https://doi.org/10.1016/j.cels.2020.06.010)
- **IEDB** — Vita R, Mahajan S, Overton JA, et al., Peters B. *The Immune Epitope Database (IEDB): 2018 update.* Nucleic Acids Res 47(D1):D339–D343 (2019). [10.1093/nar/gky1006](https://doi.org/10.1093/nar/gky1006)
- **TEPITOPE** — Sturniolo T, Bono E, Ding J, et al., Hammer J. *Generation of tissue-specific and promiscuous HLA ligand databases using DNA microarrays and virtual HLA class II matrices.* Nat Biotechnol 17(6):555–561 (1999). [10.1038/9858](https://doi.org/10.1038/9858)
- **DockQ v2** — Mirabello C, Wallner B. *DockQ v2: improved automatic quality measure for protein multimers, nucleic acids, and small molecules.* Bioinformatics 40(10) (2024). [10.1093/bioinformatics/btae586](https://doi.org/10.1093/bioinformatics/btae586)
- **ProLIF** — Bouysset C, Fiorucci S. *ProLIF: a library to encode molecular interactions as fingerprints.* J Cheminform 13(1):72 (2021). [10.1186/s13321-021-00548-6](https://doi.org/10.1186/s13321-021-00548-6)
- **PoseBusters** — Buttenschoen M, Morris GM, Deane CM. *PoseBusters: AI-based docking methods fail to generate physically valid poses or generalise to novel sequences.* Chem Sci 15(9):3130–3139 (2024). [10.1039/d3sc04185a](https://doi.org/10.1039/d3sc04185a)
- **ADMET-AI** — Swanson K, Walther P, Leitz J, et al., Zou J. *ADMET-AI: a machine learning ADMET platform for evaluation of large-scale chemical libraries.* Bioinformatics 40(7) (2024). [10.1093/bioinformatics/btae416](https://doi.org/10.1093/bioinformatics/btae416)
- **Chemprop** — Heid E, Greenman KP, Chung Y, et al., McGill CJ. *Chemprop: A machine learning package for chemical property prediction.* J Chem Inf Model 64(1):9–17 (2024). [10.1021/acs.jcim.3c01250](https://doi.org/10.1021/acs.jcim.3c01250)
- **ThermoMPNN** — Dieckhaus H, Brocidiacono M, Randolph NZ, Kuhlman B. *Transfer learning to leverage larger datasets for improved prediction of protein stability changes.* Proc Natl Acad Sci USA 121(6):e2314853121 (2024). [10.1073/pnas.2314853121](https://doi.org/10.1073/pnas.2314853121)
- **MMseqs2** — Steinegger M, Söding J. *MMseqs2 enables sensitive protein sequence searching for the analysis of massive data sets.* Nat Biotechnol 35(11):1026–1028 (2017). [10.1038/nbt.3988](https://doi.org/10.1038/nbt.3988)
- **Foldseek** — van Kempen M, Kim SS, Tumescheit C, et al., Steinegger M. *Fast and accurate protein structure search with Foldseek.* Nat Biotechnol 42(2):243–246 (2024). [10.1038/s41587-023-01773-0](https://doi.org/10.1038/s41587-023-01773-0)
- **BayeshERG** — Kim H, Park M, Lee I, Nam H. *BayeshERG: a robust, reliable and interpretable deep learning model for predicting hERG channel blockers.* Brief Bioinform 23(4) (2022). [10.1093/bib/bbac211](https://doi.org/10.1093/bib/bbac211)

@article{confrover2025,
  title={Simultaneous Modeling of Protein Conformation and Dynamics via Autoregression},
  author={Shen, Yuning and Wang, Lihao and Yuan, Huizhuo and Wang, Yan and Yang, Bangji and Gu, Quanquan},
  journal={arXiv preprint arXiv:2505.17478},
  year={2025}

Published thresholds carry their own citations, and are reported as their authors defined
them: the same-fold TM-score cut of 0.5 (Xu & Zhang, Bioinformatics 26(7):889–895, 2010),
the DR-panel ADA breadth rule (Paul et al., J Immunol Methods 422:28–34, 2015,
[10.1016/j.jim.2015.03.022](https://doi.org/10.1016/j.jim.2015.03.022)), the class I 9-mer
immunogenicity score (Calis et al., PLoS Comput Biol 2013,
[10.1371/journal.pcbi.1003266](https://doi.org/10.1371/journal.pcbi.1003266)), shape
complementarity (Lawrence & Colman 1993), the instability index (Guruprasad et al. 1990),
and Lipinski's rule of five (1997). Thresholds this program chose for itself are labelled
`suggested` in both the code and the configs, and are never presented as published values.

## Citation

If you use RFD4-Proteina in your research, please cite
[*Biomolecular design through latent reasoning with RFD4-Proteina*](https://www.biorxiv.org/content/10.64898/2026.11.12.766037v1):

```bibtex
@article{corley2026rfd4proteina,
  title        = {Biomolecular design through latent reasoning with {RFD4-Proteina}},
  author       = {Corley, Nathaniel and Didi, Kieran and Kazman, Maxwell W. and Brent, Rafael I. and others},
  year         = {2026},
  journal      = {bioRxiv},
  doi          = {10.64898/2026.11.12.766037},
  url          = {https://www.biorxiv.org/content/10.64898/2026.11.12.766037v1}
}
```
}
See the [project page](https://rfd4-proteina.org/#citations) for citation updates.

[Documentation](https://rfd4-proteina.org/latest/) · [Contributing](https://rfd4-proteina.org/latest/contributing/index.html) · [BSD 3-Clause license](https://github.com/RosettaCommons/RFD4-Proteina/blob/main/LICENSE)

<p align="center">
  <img src="assets/readme/dna-binder-generation.webp" width="360" alt="Protein-binder generation conditioned on a DNA duplex.">
</p>
