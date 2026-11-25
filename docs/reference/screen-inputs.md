# Screen inputs

The disease panel (`human`) screens a finished design against a set of liability axes and
writes the verdict to `screen.json` beside the designs. Most of those axes read a file
**you** supply, because the measurement needs a tool or a dataset this program will not run
or download for you.

This page is the contract for those files: where each one goes, which columns it needs, and
what each refusal means. Nothing here is optional in the sense of being unimportant — an
axis with no input stays `untested`, and a bar with an untested axis never clears.

**Contents**

- [Where the files go](#where-the-files-go)
- [PRODIGY protein-pair affinity](#prodigy-protein-pair-affinity)
- [OpenAntigens human target evidence](#openantigens-human-target-evidence)
- [P2Rank receptor pockets](#p2rank-receptor-pockets)
- [AutoDock Vina fixed-pose score](#autodock-vina-fixed-pose-score)
- [sPyRMSD redocking controls](#spyrmsd-redocking-controls)
- [ESP-Sim prealigned ligand comparison](#esp-sim-prealigned-ligand-comparison)
- [Badapple 2 historical scaffold assay evidence](#badapple-2-historical-scaffold-assay-evidence)
- [SCScore synthetic complexity](#scscore-synthetic-complexity)
- [CYPstrate substrate profiling](#cypstrate-substrate-profiling)
- [EPA httk steady-state exposure screen](#epa-httk-steady-state-exposure-screen)
- [PK-Sim human organ exposure](#pk-sim-human-organ-exposure)
- [CellProfiler chromogenic tissue and chip imaging](#cellprofiler-chromogenic-tissue-and-chip-imaging)
- [Small-molecule aqueous solubility](#small-molecule-aqueous-solubility)
- [fastsolv organic-solvent solubility](#fastsolv-organic-solvent-solubility)
- [Jazzy hydration and hydrogen-bond descriptors](#jazzy-hydration-and-hydrogen-bond-descriptors)
- [Small-molecule Ames mutagenicity prediction](#small-molecule-ames-mutagenicity-prediction)
- [Human liver microsomal clearance](#human-liver-microsomal-clearance)
- [Small-molecule Caco-2 permeability and efflux](#small-molecule-caco-2-permeability-and-efflux)
- [Small-molecule drug-induced liver injury](#small-molecule-drug-induced-liver-injury)
- [DELPHI paired-antibody polyreactivity](#delphi-paired-antibody-polyreactivity)
- [Therapeutic peptide red-cell assay evidence](#therapeutic-peptide-red-cell-assay-evidence)
- [MolProbity whole-model geometry](#molprobity-whole-model-geometry)
- [Dimorphite-DL ligand protonation states](#dimorphite-dl-ligand-protonation-states)
- [CREST ligand conformer sampling](#crest-ligand-conformer-sampling)
- [PROPKA structure-dependent pKa](#propka-structure-dependent-pka)
- [QupKake microscopic pKa](#qupkake-microscopic-pka)
- [SyGMa human metabolite predictions](#sygma-human-metabolite-predictions)
- [MetFrag measured spectrum comparison](#metfrag-measured-spectrum-comparison)
- [Therapeutic Nanobody Profiler](#therapeutic-nanobody-profiler)
- [How a verdict is reached](#how-a-verdict-is-reached)
- [Class I and class II predictor tables](#class-i-and-class-ii-predictor-tables)
- [ImmuScope study benchmarks](#immuscope-study-benchmarks)
- [Reference tables](#reference-tables)
- [The proteome](#the-proteome)
- [Plasma fold change](#plasma-fold-change)
- [Toxicology panel](#toxicology-panel)
- [Paralog co-evolution](#paralog-co-evolution)
- [Chemical liability](#chemical-liability)
- [Drug-type reports](#drug-type-reports)
- [Solubility and aggregation](#solubility-and-aggregation)
- [Design diversity](#design-diversity)
- [Supplied scores and curated evidence](#supplied-scores-and-curated-evidence)
- [Command-line flags](#command-line-flags)
- [Every blocker, and what to do about it](#every-blocker-and-what-to-do-about-it)

```{important}
These axes are **not** [pass criteria](metrics.md#pass-criteria). They have no YAML config,
no `metrics_tags` gate and no `.pass` column, and each is tri-state — `hit`, `clear` or
`untested` with a named blocker — rather than boolean. They borrow the shape of that table
and none of its machinery.
```

## Where the files go

Every file is read from the generation's own output directory: `<out-dir>/<group>_<entry>_<chain>/`
for a live run, or the archived folder under `<out-dir>/generations/<disease>/<target>/`.

A table may be `.tsv` or `.csv`; `.tsv` is tried first. Every file is read as UTF-8 with or
without a byte-order mark — the one Excel's "CSV UTF-8" writes by default — and column names
are matched case-insensitively.

A table may also open with a banner, which is skipped: a leading block of blank lines and
lines beginning `#`, `---`, `//` or `;;`, as a predictor's export often does. An *unmarked*
banner line is not skipped, because it is indistinguishable from a header; the table then
reports `predictor_unreadable`. `axis_evidence.json` is JSON, so it has no banner: the
object must start the file.

| File | Axis it feeds |
|---|---|
| `mhc_i.tsv` | `mhc_i`, and restricts `immunogenicity` |
| `mhc_ii.tsv` | `mhc_ii` and `ada` |
| `mhc_i_reference.tsv` | the bar `mhc_i` is judged against |
| `mhc_ii_reference.tsv` | the bar `mhc_ii` is judged against |
| `immunogenicity.tsv` | overrides the computed immunogenicity score |
| `ada.tsv` | overrides the computed ADA measurement |
| `plasma_fold.tsv` | `plasma_fold` |
| `toxicology.tsv` | the ligand safety panel (`herg`, `nav1_5`, `cyp3a4`, `cyp2d6`, `cyp2c9`, `cyp1a2`, `cyp2c19`, `pgp`, `bcrp`, `bsep`) |
| `ligand_candidates.tsv` or `ligand_smiles.txt` | ADMET-AI aqueous solubility for the nominated small molecules in a ligand run |
| `fastsolv.json` and nominated ligands | optional fastsolv solubility predictions at declared pure organic solvent and temperature conditions |
| `ligand_candidates.tsv` or `ligand_smiles.txt` | optional Jazzy hydration free energy and hydrogen-bond strengths for nominated neutral ligands |
| `dimorphite.json` and either `ligand_candidates.tsv` or `ligand_smiles.txt` | Dimorphite-DL ligand protonation states for a specified pH window |
| `crest.json` and a nominated three-dimensional ligand SDF | optional conformer ensemble for one explicitly prepared molecular state |
| `propka.json` and a nominated PDB | optional structure-dependent pKa estimates for ionizable protein residues and recognized ligand groups |
| `qupkake.json` and a nominated ligand input | QupKake microscopic pKa for individual proton transfers in a ligand run |
| `sygma.json` and a nominated ligand input | SyGMa ranked human metabolite structures and their predicted formation pathways |
| `metfrag.json`, a product-ion TSV and a nominated ligand input | optional MetFrag comparison of candidate structures with one measured MS/MS spectrum |
| `ligand_candidates.tsv` or `ligand_smiles.txt` | optional ADMET-AI Ames probability for a nominated small-molecule lead |
| `ligand_candidates.tsv` or `ligand_smiles.txt` | optional ADMET-AI human liver microsomal intrinsic clearance prediction for nominated small molecules |
| `ligand_candidates.tsv` or `ligand_smiles.txt` | optional AstraZeneca Caco-2 permeability and efflux model scores for nominated small molecules |
| `ligand_candidates.tsv` or `ligand_smiles.txt` | optional ADMET-AI drug-induced liver injury classifier scores for nominated small molecules |
| `delphi_pairs.tsv` | optional DELPHI PSR three-column nominations of paired antibody variable domains |
| `peptide_rbc_evidence.tsv` | optional previously measured red-cell assay observations for identified peptide candidates and formulations |
| generated `*_generated.cif`, `.cif.gz`, or `.cif.zst` | optional local MolProbity backbone, rotamer, and all-atom geometry for a finished protein or peptide model |
| `solubility.tsv` | supplied solubility and aggregation scores, reported and not judged |
| `seq_prediction.txt` | native Protein-Sol report for the matching sample sequences |
| `prodigy_partners.json` | optional protein-pair affinity estimates for each generated structure |
| regular job record or `openantigens.json` | optional human target topology and public evidence availability for the canonical UniProt accession |
| `p2rank.json` and a nominated receptor PDB or mmCIF | optional small-molecule pocket predictions for a supplied protein receptor |
| `vina_poses.json` | optional empirical score of an explicitly selected small-molecule pose in each finished structure |
| `spyrmsd_controls.json` | symmetry-corrected ligand pose RMSD for nominated redocking controls |
| `espsim.json` and two prealigned ligand SDFs | optional shape and electrostatic comparison of a nominated candidate and reference pose |
| `ligand_candidates.tsv` or `ligand_smiles.txt` | optional Badapple 2 historical assay evidence for nominated ligand scaffolds |
| `scscore_comparison.json` and `ligand_candidates.tsv` | synthetic-complexity tie-break for an explicitly comparable set of small molecules |
| `cypstrate.json` and molecular nominations | human CYP substrate classifications for the isoforms named for each systemic small molecule |
| `httk.json` and `ligand_candidates.tsv` | adult human plasma steady-state estimates at a fixed reference intake for substances covered by EPA httk |
| `pksim.json` and molecular nominations | individual human concentration–time profiles from a registered PK-Sim project |
| `cellprofiler.json`, RGB fields and region masks | regional DAB intensity in candidate-stained tissue or chip images and matched reagent controls |
| `paralog_coevolution.tsv` | a co-evolution note stored on `paralog` |
| `axis_evidence.json` | any axis, as a curated judgement |

## Small-molecule aqueous solubility

For a finished ligand run, place one SMILES in `ligand_smiles.txt` and run
`human --screen RUN --binder-kind ligand`. A fixed conditioning ligand receives one
prediction across all generated pockets. To nominate several compounds, supply a
UTF-8 `ligand_candidates.tsv` with this exact tab-separated header and a unique ID
per nomination. When both files exist, this screen reads the TSV. The single-ligand
file accepts an optional title after the SMILES. The TSV requires a plain SMILES
in each structure cell. Candidate IDs describe the supplied list independently of
the pocket files.

```tsv
candidate_id	smiles
aspirin	CC(=O)Oc1ccccc1C(=O)O
caffeine	Cn1c(=O)c2c(ncn2C)n(C)c1=O
```

For a standalone check with the installed `ligand-safety` extra, run
`python scripts/screen_extras.py admet-solubility 'CC(=O)Oc1ccccc1C(=O)O'`.
The CPU worker loads the bundled ADMET-AI v2 ensemble locally. It checks the
endpoint definition and the five regression checkpoint hashes against the installed
distribution. Each result names the method, endpoint and unit.

`screen.json.ligand_solubility` reports the native `Solubility_AqSolDB` prediction
as `predicted_log_s`, the base-10 logarithm of aqueous molar solubility. For example,
−2 on this scale means 0.01 mol/L. The model takes the molecular graph as its input.
The AqSolDB measurements come from nine sources with differing experimental conditions.
Solubility at a specified pH or temperature, the behavior of a formulation and
prospective prediction error require separate evidence. This measurement has no
pass threshold. [AqSolDB dataset](https://doi.org/10.7910/DVN/OVHAW8) ·
[ADMET-AI endpoint definitions](https://github.com/swansonk14/admet_ai/blob/main/admet_ai/resources/data/admet.csv)

`inputs` maps every nomination to its compound ID or a blocker. Equivalent isomeric
SMILES share a result, including equivalent atom orderings and atom-map annotations.
Stereochemistry, isotopes, charge and salt fragments are retained. The local
`structures` map identifies the graph behind each compound ID. The ID contains the
SHA-256 of its canonical isomeric SMILES and is stable across input orderings made
with the same canonicalizer. The service exposes
the IDs, predictions and coverage. Invalid molecules, repeated candidate IDs and
unavailable predictions remain visible. `partial` means some nominations or compounds
lack results. The standalone command exits with status 2 for partial or untested output.

## fastsolv organic-solvent solubility

Use fastsolv to choose which pure organic solvent and temperature conditions
to test for the saturation solubility of a neutral small-molecule solid. Use
conditions relevant to the planned experiment and verify the solubility in
that solvent and solid form. Aqueous samples, pH effects, solvent mixtures,
and formulated products require measurements under their own conditions.
[fastsolv study](https://doi.org/10.1038/s41467-025-62717-7)

In a finished ligand run, nominate structures in `ligand_candidates.tsv` with
`candidate_id` and `smiles` columns, or put one SMILES in `ligand_smiles.txt`.
A screen accepts up to 16 selected solutes. Use a single connected, neutral
molecule containing carbon and 4–50 heavy atoms. Supported elements are H, C,
N, O, F, P, S, Cl, Br and I. Isotopes and radicals are outside the input scope.

Place `fastsolv.json` in the same directory. Specify 1–8 distinct solvent and
temperature pairs using `temperature_k` from 295 to 323 K inclusive. The
permitted solvent IDs are `acetone`, `acetonitrile`, `dimethyl_sulfoxide`,
`dimethylformamide`, `ethanol`, `ethyl_acetate`, `isopropanol`, `methanol`,
`tetrahydrofuran`, and `toluene`. This range is the workflow's input limit.

```json
{"version": 1, "conditions": [{"solvent": "ethanol", "temperature_k": 312}]}
```

Use the project's Linux CPU Python environment with Torch 2.6 or newer and
the `fastsolv` extra for descriptors. The four verified checkpoints total
about 0.90 GB. Add `--checkpoints DIRECTORY` to use local copies. Predictions
run offline with the installed checkpoints.
[Official fastsolv checkpoints](https://doi.org/10.5281/zenodo.13943074)

```sh
uv pip install ".[fastsolv]"
python scripts/install_fastsolv.py
human --screen finished-run --binder-kind ligand --screen-method fastsolv
python scripts/screen_extras.py fastsolv finished-run
```

For one candidate, add `--candidate ID` to the standalone command and use an
exact ID from `ligand_candidates.tsv`. The panel saves results in `screen.json`
under `method_selection.prepared_measurements.fastsolv`. Each predicted
candidate and condition has `log_s` and `ensemble_sd`, both in `log10(mol/L)`.
These report the four-member mean and population standard deviation.
In the published solute-held-out tests, the authors evaluated FASTPROP before
forming the four-member ensemble. Its RMSE was 0.83 log units on SolProp and
0.95 on Leeds. Respectively, 78.1% and 73.8% of predictions were within a
factor of ten of the measurement. The spread between the four members has no
calibrated interval coverage. Measure saturation solubility in the intended
solid form when choosing between solvents with similar estimates.
[Benchmark and model methods](https://doi.org/10.1038/s41467-025-62717-7)

The published training used organic-solvent observations from BigSolDB after
excluding water. Its source records express solubility as a mole fraction,
which the authors converted to molar solubility before training. The aqueous
ADMET-AI model uses AqSolDB and addresses aqueous solubility.
[BigSolDB training source](https://doi.org/10.5281/zenodo.6984601) ·
[fastsolv training preparation](https://github.com/JacksonBurns/fastsolv/blob/main/paper/data/krasnov.py)

## Jazzy hydration and hydrogen-bond descriptors

Use Jazzy to compare how related neutral ligands interact with water and to
interpret changes in hydrogen-bond donor and acceptor strength. The hydration
free energy describes transfer of the nominated molecular state from gas to
water. Saturation solubility also depends on the solid form and experimental
conditions. Measurements at a chosen pH and temperature are needed for
decisions specific to those conditions.
[Jazzy study](https://doi.org/10.1038/s41598-023-30089-x)

In a finished ligand run, put candidates in `ligand_candidates.tsv` with
`candidate_id` and `smiles` columns, or put one SMILES in `ligand_smiles.txt`.
Up to 32 selected candidates can be screened. Use one connected molecule
containing carbon, with 2–40 heavy atoms and with all atomic formal charges
zero. Supported elements are H, C, N, O, F, P, S, Cl, Br and I. Specify stable
stereochemistry in the nominated microstate. Salts, isotopes and radicals
are outside the supported input scope.

Install Jazzy 0.1.4 from the active Linux x86-64 project environment. The
installer builds an isolated Python 3.12 CPU environment with RDKit 2024.3.1
and Kallisto 1.0.10. The screen generates a single 3D conformer using ETDG
with seed 11 and at most 64 embedding attempts, then requires MMFF94 minimization to
converge within 200 iterations.
[Jazzy release](https://pypi.org/project/jazzy/0.1.4/)

```sh
python scripts/install_jazzy.py
human --screen finished-run --binder-kind ligand --screen-method jazzy
python scripts/screen_extras.py jazzy finished-run
```

Add `--candidate ID` to the standalone command for one exact ID from
`ligand_candidates.tsv`. The panel saves the result in `screen.json` under
`method_selection.prepared_measurements.jazzy`.

The total hydration free energy and its apolar, polar and interaction terms
use kJ/mol. Molecular strengths for C–H donors, X–H donors with X other than
carbon, and acceptors are dimensionless sums. The model was fitted to Gerber's experimental
hydration data. Its external test on 3,313 Guthrie compounds reported 9.19
kJ/mol RMSE and 5.89 kJ/mol MAE, with larger errors for some flexible,
internally hydrogen-bonded or aggregation-prone compounds. Use the results
as descriptors when reviewing comparable molecules.
[Jazzy benchmark](https://doi.org/10.1038/s41598-023-30089-x) ·
[Jazzy descriptor definitions](https://jazzy.readthedocs.io/en/latest/cookbook.html)

## Small-molecule Ames mutagenicity prediction

Use the optional Ames screen to prioritize bacterial reverse-mutation testing for
a nominated small-molecule lead. It can help select compounds for an experimental
panel. It does not represent a measured Ames result, establish a mutation rate, or
rule out genotoxicity at a low score. Skip it for biologics and for unsupported
forms such as mixtures, disconnected salts, or molecules with unresolved atoms
or radicals. The screen neither removes fragments nor changes protonation or
stereochemistry for a nomination. [OECD Test Guideline 471](https://doi.org/10.1787/9789264071247-en)

In a finished ligand run, supply `ligand_candidates.tsv` with the exact
`candidate_id<TAB>smiles` header or one molecule in `ligand_smiles.txt` for the
whole run. The TSV takes precedence, including when invalid. Use a regular
UTF-8 file no larger than 1 MiB and no more than 256 candidate rows. Use unique,
non-secret candidate identifiers of up to 128 characters, starting with an
ASCII letter or digit and containing only letters, digits, periods, underscores,
and hyphens. Keep structures in the SMILES field, not in the identifier. A
run-level molecule cannot stand in for a specifically selected candidate.
Each SMILES is limited to 4,096 characters and each parsed graph to 256 atoms.

```bash
human --screen finished-run --binder-kind ligand --screen-method ligand_ames
```

The browser's screening picker and MCP `plan_screening` can recommend it for
eligible ligand inputs when the required local assets verify. Use
`run_selected_screens` with `methods: ["ligand_ames"]` for a supplemental result
without changing the stored verdict. A normal run without a screen selection does
not start this prediction automatically.

The isolated CPU worker reuses the installed ADMET-AI v2 classification ensemble
from the `ligand-safety` extra. It verifies the endpoint and all five checkpoint
hashes against the installed distribution before loading and does not download
weights, reference data, or a new runtime. The worker uses no GPU VRAM. Apply the
project's resource checks for the less-than-25 GB disk and at-most-96 GB VRAM
limits to the environment and scratch space you run it in.
[ADMET-AI v2.0.1 endpoint definitions](https://github.com/swansonk14/admet_ai/blob/v_2.0.1/admet_ai/resources/data/admet.csv)

`screen.json.ligand_ames` reports `predicted_ames_probability` on a 0–1 scale,
with the method `admet_ai_ames_v2`, endpoint `AMES`, release identity and training
corpus declaration. Equivalent canonical isomeric graphs share one prediction.
`inputs` maps nominations to hashed compound IDs or named blockers, and
`compounds` gives the score or blocker for each ID. No raw graphs are stored in
this report or returned in its MCP view. `partial` means a nomination or compound
could not be scored. `untested` means no attributable prediction was produced.
No cutoff or change to the existing verdict is applied.

## Human liver microsomal clearance

Use this optional screen to prioritize human liver microsomal stability assays
for nominated small-molecule leads, or to compare leads on the same predicted
intrinsic clearance scale. Higher predicted values mean faster intrinsic
clearance in that assay. The regression uses the human AstraZeneca microsomal
endpoint `Clearance_Microsome_AZ` and reports its native linear value in
`uL/min/mg` of microsomal protein. It does not convert the value to a log scale
or use the separate hepatocyte endpoint. Do not use it for biologics, to
identify metabolites, or to infer whole-body clearance, dose, or half-life.
Measured microsomal results take precedence over predictions.
[ADMET-AI v2.0.1 endpoint definitions](https://github.com/swansonk14/admet_ai/blob/v_2.0.1/admet_ai/resources/data/admet.csv)
· [AstraZeneca dataset](https://tdcommons.ai/single_pred_tasks/adme/#clearance-astrazeneca)

Nominate molecules in `ligand_candidates.tsv` with the exact
`candidate_id<TAB>smiles` header, or nominate one molecule for the whole run in
`ligand_smiles.txt`. A present TSV always wins, even when invalid. Supply a
regular UTF-8 file no larger than 1 MiB with at most 256 nominations. Use unique
non-secret IDs of up to 128 characters, starting with an ASCII letter or digit
and otherwise containing only letters, digits, periods, underscores, or hyphens.
Each SMILES is limited to 4,096 characters and each parsed graph to 256 atoms.
Use connected, closed-shell organic graphs. No mixture, disconnected salt,
unresolved atom, or radical is accepted. The nominated charge and stereochemistry
are retained. A run-level molecule cannot stand in for a selected candidate.

```bash
human --screen finished-run --binder-kind ligand --screen-method ligand_microsomal_clearance
```

The browser screening picker and MCP `plan_screening` list the method and check
its assets and input prerequisites. Confirm charged nominations and select large
batches explicitly. MCP `run_selected_screens` accepts
`methods: ["ligand_microsomal_clearance"]`. Ordinary runs without a screen
selection do not start it automatically.

The isolated CPU worker reuses the five installed ADMET-AI v2 regression model
files from the `ligand-safety` extra. It verifies endpoint metadata and the
installed distribution's hashes, and does not download model or reference data.
Apply the project's less-than-25 GB disk and at-most-96 GB VRAM resource checks
to the installed environment and scratch space. This screen uses no GPU VRAM.

`screen.json.ligand_microsomal_clearance` records the method
`admet_ai_microsomal_clearance_v2`, endpoint, unit, provider and corpus
provenance, candidate coverage and `predicted_clint_ul_min_mg` for each unique
compound. Equivalent canonical isomeric graphs share one prediction and an
opaque compound ID. Raw graphs are not copied into this report or its MCP view.
Zero remains a real numeric prediction. A negative, nonfinite or unattributed
value does not become a measurement. `partial` means some nominations or
compounds lack predictions. `untested` means none can be reported. No threshold
or change to the existing verdict is applied.

## Small-molecule Caco-2 permeability and efflux

Select **AstraZeneca Caco-2 permeability and efflux** for a small-molecule
candidate when planning an experimental Caco-2 study, particularly when
intestinal permeability or efflux matters to the intended route. With a declared
oral route, the picker recommends this screen for a small eligible batch.
An unknown route or explicitly local exposure leaves the choice manual. Do not use this screen for proteins, other
biologics, or leads for which these assays are not relevant.

Nominate molecules in `ligand_candidates.tsv` with the exact
`candidate_id<TAB>smiles` header, or one run-level ligand in
`ligand_smiles.txt`. A present TSV always takes precedence. A run-level
ligand cannot stand in for an individual candidate. Use the same nominations
as for [human liver microsomal clearance](#human-liver-microsomal-clearance):
unique IDs, a regular UTF-8 input no larger than 1 MiB, at most 256
nominations, at most 4,096 characters per SMILES, and at most 256 atoms
per graph. Use connected, closed-shell organic molecules. The nominated
charge and stereochemistry are retained. Mixtures and disconnected salts
are not accepted.

```bash
human --screen finished-run --binder-kind ligand --screen-method ligand_caco2
```

MCP `plan_screening` checks the same prerequisites, and
`run_selected_screens` accepts `methods: ["ligand_caco2"]`. The isolated
worker uses the installed `ligand-safety` environment on CPU, with no GPU
VRAM or downloads during screening. An administrator can place the
`az_gnn_mtl_v1.pt` file from the [AstraZeneca v1 release](https://zenodo.org/records/16948542)
at `DATA_PATH/caco2/az_gnn_mtl_v1.pt` or set `RFD4_CACO2_CHECKPOINT` to its
path. The expected size is 1,283,341 bytes. The pinned SHA-256 is
`c979bd0dab17a324e2c0c52c3da0bbe25d93415b7c6b49d62f0e01f6f2a487de`.

`screen.json.ligand_caco2` records `az_gnn_mtl_caco2_v1`, the provider and
corpus provenance, and the per-compound `efflux_ratio` (`Caco2ER`) and
`apparent_permeability` (`Caco2Papp`) heads. The
[model paper](https://doi.org/10.1021/acsomega.5c04861) describes
log-transformed assay labels. The release does not establish a physical-unit
conversion for the checkpoint's outputs, so the record reports signed
`model_score` values only, with scale `native_model_score`. Zero is a valid
score. The heads are two outputs of the same model, not independent opinions.
Do not combine them, compare them to a laboratory threshold, or use them to
assign a transporter, estimate human oral bioavailability, or replace an assay.
The MDCK outputs from the same checkpoint are not reported here.

Equivalent canonical isomeric graphs share one prediction and an opaque
compound ID. The report and MCP view do not include raw graphs.
`unique_predicted` counts distinct compounds with at least one attributable
finite score and `unique_complete` counts those with both. A missing or
unusable head is `untested` with a blocker. A report with scores but incomplete
coverage is `partial`. No threshold or change to the existing verdict applies.

## Small-molecule drug-induced liver injury

Use **ADMET-AI drug-induced liver injury** to prioritize follow-up on nominated
small-molecule therapeutic leads when liver exposure is relevant. For a series
of related leads it can flag compounds to review first, on the same model scale.
It is not useful for biologics, when liver exposure is not part of the question,
or as a substitute for hepatotoxicity studies. It does not assess dose, route,
metabolites, individual susceptibility, BSEP inhibition, or clearance. A low
score, an unsupported compound, and an untested compound do not establish safety.

The verified [ADMET-AI v2 classifier](https://github.com/swansonk14/admet_ai/blob/v_2.0.1/admet_ai/resources/data/admet.csv)
exposes the human `DILI` head. It scores the positive class of the
[binary TDC DILI task](https://tdcommons.ai/single_pred_tasks/tox/#dili-drug-induced-liver-injury),
which documents 475 drugs. It is not a calibrated clinical probability or an
injury severity scale. For an already known drug, consult
[FDA DILIrank 2.0](https://www.fda.gov/science-research/liver-toxicity-knowledge-base-ltkb/drug-induced-liver-injury-rank-dilirank-20-dataset)
separately after confirming the drug's identity. That reference reports
label-based categories, not model scores. This screen does not query it, infer
safety from an absent match, or assume it is independent of the training corpus.

Use the same bounded, named small-molecule nominations as for
[human liver microsomal clearance](#human-liver-microsomal-clearance).
`ligand_candidates.tsv` takes precedence over `ligand_smiles.txt`, even when
the TSV cannot be read. The former is required when selecting one named
candidate. The picker and MCP recommend a small eligible batch with declared
systemic exposure. A bare molecular nomination leaves this screen available for manual selection.

```bash
human --screen finished-run --binder-kind ligand --screen-method ligand_dili
```

MCP `plan_screening` checks the same prerequisites. `run_selected_screens`
accepts `methods: ["ligand_dili"]`. This screen reuses the installed five-model
ADMET-AI v2 classification ensemble from the `ligand-safety` extra. The worker
checks the endpoint and installed wheel hashes before loading the weights, runs
on CPU, and downloads no data. The installed distribution is a trusted local
dependency. A matching wheel manifest checks installation consistency, not
publisher identity. Ordinary screening does not run this optional prediction.

`screen.json.ligand_dili` records `admet_ai_dili_v2`, `DILI`, provider and
training-corpus provenance, and per-compound `predicted_dili_probability` from
0 to 1. Equivalent canonical isomeric graphs share a prediction and an opaque
compound ID. Raw graphs are not in the report or its MCP view. An incomplete
report is `partial` when some scores exist and `untested` when none exist.
There is no cutoff, pass/fail classification, or change to the existing verdict.

## DELPHI paired-antibody polyreactivity

Use **DELPHI antibody polyreactivity (PSR)** to prioritize experimental PSR
measurements for conventional antibodies with a confirmed VH/VL pair. As a
secondary use, compare related paired leads on the same PSR model scale, or
review it beside a separate CHO-lysate polyreactivity result. Keep the two
readouts separate. The PSR training sequences are private, so overlap with
the GDPa1 corpus behind CrossAbSense cannot be verified. Do not use DELPHI for
VHHs, unpaired or swapped chains, or to infer target specificity, serum
behavior, clinical safety, or monomer purity. Measured evidence for the
assay has priority. SEC is a separate readout and is not run by this screen.

In a finished protein run, put the intended **complete variable domains only**
in a UTF-8 `delphi_pairs.tsv` with the exact tab-separated header
`candidate_id<TAB>vh<TAB>vl`. Use the run's example ID for each nomination.
A nomination is the user's statement of pairing, not a pair inferred from the
structure or row order. Nominate the original VH and VL together, without Fc,
constant domains, linkers, tags, gaps, or ambiguous amino acids. The file is
limited to 64 rows and 1 MiB. Each domain fits a nontruncating 135-residue
window. Local ANARCII numbering must recover a complete heavy domain and
kappa or lambda light domain, and the HCDR3 at IMGT 105–117 including
insertions must fit 1–25 residues. The leading framework Cys is not part of
this input. A Cys at the first CDR3 position is declined rather than passing
it through the upstream pipeline's ambiguous leading-Cys shortcut.

```bash
human --screen finished-run --binder-kind protein --screen-method delphi_psr
```

The browser and MCP `plan_screening` check for the pinned local model and a
nomination in scope. `run_selected_screens` accepts `methods: ["delphi_psr"]`.
The picker can preselect an eligible single nomination. Review larger or
incomplete inputs before selecting them. Legacy screening without an explicit
selection does not start this tool.

Install only `FINAL_psr_filter_onehot_transformer_onehot_ipi_psr_trainset.pt`
from the [IPI DELPHI v4 release](https://zenodo.org/records/21823887), not the
full archive. It is 7,204,447 bytes with SHA-256
`5bcd6accfd832c88f582b8206a27b5930f3cbf3a442eba0c59695ebb6f6d8b81`.
Run `python scripts/install_delphi_psr.py path/to/the_downloaded_checkpoint.pt`
from the installed project environment. The default destination is
`DATA_PATH/delphi/FINAL_psr_filter_onehot_transformer_onehot_ipi_psr_trainset.pt`.
An administrator may also set `RFD4_DELPHI_PSR_CHECKPOINT` to a local copy of
the same checkpoint. The worker rechecks the digest and loads weights only
on CPU. It does not download models or use a protein language model.

`screen.json.delphi_psr` reports per-pair `predicted_psr_pass_probability`,
the model's 0–1 score for its PSR pass class. Higher means lower predicted
polyreactivity for the IPI PSR label. It is not a pass decision, measured
assay value, or calibrated probability outside that assay population.
It carries the method, release, coverage, corpus provenance, and the digest
of the exact nomination file read without outputting protein sequences.
Failed pairs keep a named blocker. No
threshold is applied and the run's verdict stays unchanged.

## Therapeutic peptide red-cell assay evidence

Use **Peptide red-cell assay evidence** when a finished peptide candidate has
measured hemolysis results and you need to establish which formulation, species,
and assay conditions the results actually cover. Use human observations for
human red-cell context and animal observations as separate supporting evidence.
Do not substitute animal results for human data, transfer a result to a different
formulation, or treat even a qualified in-vitro result as a clinical safety
determination. Species and test conditions can affect measurements. See this
[experimental comparison of hemolysis assays](https://pmc.ncbi.nlm.nih.gov/articles/PMC9917735/).

Place a regular UTF-8 `peptide_rbc_evidence.tsv` beside the finished peptide
run. Use this exact header with tab separators:

```text
candidate_id	observation_id	formulation_id	source_id	species	matrix	context_id	concentration	concentration_unit	exposure_minutes	hemolysis_percent	controls_qc
```

- `candidate_id` is the exact example ID of an unambiguous finished candidate,
  not the opaque MCP selection key. `observation_id` must be unique within that
  candidate. The two together identify a row. Replicate observations require
  separate IDs and are not averaged.
- `formulation_id`, `source_id`, and `context_id` refer to the exact tested
  formulation or lot, originating report, and documented assay context. Use
  opaque identifiers up to 96 characters, starting with an ASCII letter or
  digit and otherwise containing only letters, digits, periods, underscores,
  or hyphens. Do not put patient or donor identifiers, sequences, or file paths
  into these fields. Candidate matching does not independently verify the
  chemical identity or authenticate the report.
- `species` is one of `human`, `mouse`, `rat`, `rabbit`, `dog`, `pig`, `sheep`,
  `cow`, or `cynomolgus`. `matrix` is `washed_rbc` or `whole_blood`.
- `concentration` and `exposure_minutes` are the positive, finite values stated
  by the source, with the latter expressed in minutes. The allowed concentration
  units are `uM`, `ug/mL`, and `mg/mL`. The screen does not convert units or
  combine different contexts.
- `hemolysis_percent` is a finite, exact reported percentage on the 0–100
  scale, without a `%` symbol. Numeric values must round-trip without a change
  to their stated value. A reported zero stays zero. An inequality,
  censored result, concentration-at-effect endpoint, or ambiguous range is not
  treated as an exact percentage. Keep it in the source report for manual review.
- `controls_qc` is `accepted` only when the originating assay documents
  acceptable positive and negative controls. `failed`, `not_reported`, or a
  blank value keeps the observation untested. There is no built-in assay QC
  rule or safety cutoff.

The file is limited to 256 rows and 256 KiB, including for candidate-scoped
review. The browser and MCP `plan_screening` can recommend a complete set of
qualified human observations. Animal or mixed records are for manual selection.
MCP `run_selected_screens` accepts `methods: ["peptide_rbc_evidence"]`.

```bash
human --screen finished-run --binder-kind peptide --screen-method peptide_rbc_evidence
```

`screen.json.peptide_rbc_evidence` keeps each observation and its provenance
separate. `reported` means the supplied rows met the input contract, not that
the candidate passed an assay. `partial` means only some submitted rows are
reportable. `untested` retains a named blocker and does not imply zero
hemolysis. The digest identifies the exact input bytes consumed. This screen
does not infer results from sequences, estimate potency, rank variants, set
a cutoff, or change the verdict. It needs no weights, downloads, or GPU.

## MolProbity whole-model geometry

Use **MolProbity geometry** to check the coordinates of a protein or peptide
finalist before spending effort on experimental validation. It can also
compare geometry across alternative models of the same construct, provided
you compare the same readout and inspect its denominator. Do not use a model
with good local geometry as evidence of binding, correct folding, or safety.
The screen evaluates the **whole submitted model**, including any target
chains. Its numbers are not attributed to the designed chain alone.

This screen runs CCTBX Ramalyze and Rotalyze on the original coordinates.
They report favored, allowed, and outlier counts on their own eligible
denominators. Short, discontinuous, or incomplete models may have fewer
eligible torsions, and their counts are not interchangeable. The independent
clashscore counts Probe2 serious clashes per 1,000 evaluated atoms. Reduce2
rebuilds hydrogens in temporary scratch with side-chain flips disabled. It does
not modify the source file or replace the native distance-only clash screen.
There is no combined MolProbity score, cutoff, or change to the verdict.
[MolProbity reference distributions](https://pmc.ncbi.nlm.nih.gov/articles/PMC5734394/)
· [CCTBX v2025.11 native validators](https://github.com/cctbx/cctbx_project/tree/v2025.11/mmtbx/validation)

Choose the existing finished candidate model. The screen reads the canonical
generated mmCIF file, including gzip or Zstandard variants, without
reconverting it to PDB first. It accepts single coordinate models containing
the standard amino acids and optional waters. It does not remove ligands or
other unsupported residue types to produce a deceptively clean score. The
clash component also needs complete, occupied heavy atoms and identifiers
compatible with the pinned contact implementation. Backbone-only structures
may still have useful backbone torsions, but they cannot report all-atom
clashscore. An empty denominator is `untested`, never zero outliers.

Each model is limited to 8 MiB encoded, 16 MiB expanded, 16,000 input atoms,
and a 1,000 Å extent along any axis. A run-wide selection may include up to
eight models. For larger runs, choose a single finalist in the browser or
through the candidate-scoped MCP tool. The picker recommends one eligible
finished model when the pinned runtime is installed, and leaves multi-model
selections manual. MCP `run_selected_screens` accepts
`methods: ["molprobity_geometry"]`. The optional `screen.json.molprobity_geometry`
record keeps metric-specific blockers and a SHA-256 digest of the exact expanded
coordinates measured. A `partial` result means at least one metric or model
could not be reported.

```bash
human --screen finished-run --binder-kind protein --screen-method molprobity_geometry
```

Install `cctbx-base==2025.11` with `numpy==1.26.4` in a separate CPU Python
3.12 environment. The default discovery location is
`DATA_PATH/molprobity/venv/bin/python` on Linux or WSL. A trusted administrator
can set `RFD4_MOLPROBITY_PYTHON` to another compatible interpreter. Keep the
main RFD4 environment unchanged. Use the pinned
[CCTBX chemistry archive](https://github.com/cctbx/cctbx_project/releases/tag/v2025.11)
and install only its protein validation references with the project's normal
Python environment:

```bash
python scripts/install_molprobity_data.py chem_data-2025.11-pyhe0d8492_0.conda \
  --python "$DATA_PATH/molprobity/venv/bin/python"
```

The installer verifies the publisher's SHA-256, leaves any existing chemistry
library intact, and builds the reference caches offline. With only the CCTBX
wheel installed, unavailable components remain `untested`; the wheel alone
does not contain the full chemistry references. Screening is entirely local
and makes no web submission.

## Dimorphite-DL ligand protonation states

Use Dimorphite-DL to prepare plausible protonation states for a small molecule
when the pH window for an assay or docking protocol is known. For a secondary
comparison, run the same ligand in separate run folders with the pH window
chosen for each condition. The method applies empirical substructure rules.
Use QupKake for microscopic pKa estimates and dedicated workflows for tautomer
search and docking. Measuring populations or validating a candidate requires
separate evidence.
[Method](https://doi.org/10.1186/s13321-019-0336-9) ·
[Dimorphite-DL 2.1.0](https://pypi.org/project/dimorphite-dl/2.1.0/)

In a finished ligand run, supply `dimorphite.json` with all three fields. The
example window of pH 6.8–7.4 suits a near-neutral protocol when it matches the
planned conditions.

```json
{"version": 1, "ph_min": 6.8, "ph_max": 7.4}
```

Choose finite values with `0 ≤ ph_min ≤ ph_max ≤ 14`. Equal limits request a
single pH. The screen has no default window and uses the release's precision
factor of 1.0. Name candidates in `ligand_candidates.tsv` with the header
`candidate_id<TAB>smiles`, or supply one SMILES in `ligand_smiles.txt`. The table
takes precedence. Supply single-component, closed-shell organic molecules as
plain SMILES up to 512 characters and 80 heavy atoms. Include the intended formal
charge and known stereochemistry. Supported elements are H, B, C, N, O, F, Si,
P, S, Cl, Br and I. Isotopically labeled molecules, radicals and mixtures leave
the candidate `untested`. Select up to 32 entries per run. A molecule with more
than 32 possible states under the rules stays `untested` with
`variant_budget_exceeded`.

Install the optional `dimorphite` extra in the project environment and select
the method for the finished run.

```bash
uv pip install ".[dimorphite]"
human --screen finished-run --binder-kind ligand --screen-method dimorphite_dl
```

To run the stand-alone wrapper, use
`python scripts/screen_extras.py dimorphite finished-run`. Add `--candidate ID`
to evaluate one named entry from the TSV. The wrapper prints JSON and exits
with code 2 for `partial` or `untested` results. The finished-run screen saves
results in `screen.json` under `method_selection.prepared_measurements.dimorphite_dl`.
Each candidate's state set contains canonical SMILES and total formal charge.
The record includes the verified release's SMARTS-table checksum. MCP output
marks structure fields withheld when the existing sequence redaction applies.
The returned structures preserve connectivity and defined stereochemistry while
allowing protonation and charge to vary. A molecule with no recognized ionizable
sites is `untested` with `no_recognized_sites`. Treat the state set as preparation
for the next experiment or calculation.

## CREST ligand conformer sampling

Use CREST to prepare several low-energy geometries of one ligand state for
geometry review or downstream calculations. Choose the protonation, tautomer
and stereochemical state before running this screen. It uses the reduced
`--quick` iMTD-GC search at the gas-phase GFN2-xTB level on the CPU.
[CREST sampling study](https://doi.org/10.1039/C9CP06869D) ·
[CREST 3.0 methods](https://doi.org/10.1063/5.0197592)

Place `crest.json` and the prepared SDF in the finished ligand run directory.
Use a local ASCII filename and give all three manifest fields.

```json
{"version": 1, "candidate_id": "lead_A", "structure": "lead_A.sdf"}
```

The SDF must contain one 3D record with the exact candidate ID as its title.
Use one neutral, closed-shell organic molecule with explicit hydrogens and
fully specified stable stereochemistry. The screen retains the input atom
order and checks the chemical identity against a matching `ligand_candidates.tsv`
row when one is available. The SDF limit is 256 KiB, with 4–32 heavy atoms and at most 96 total
atoms. Supported elements are H, C, N, O, F, P, S, Cl, Br and I. Salts, mixtures,
isotopes and unresolved stereochemistry are outside this workflow's scope.

Install the pinned CREST 3.0.2 release from the active Linux x86-64 project
environment. Add `--archive PATH` to use a local copy of the official archive.
Each search uses two CPU threads and has a 240-second time limit.

```sh
uv pip install ".[crest]"
python scripts/install_crest.py
human --screen finished-run --binder-kind ligand --screen-method crest
python scripts/screen_extras.py crest finished-run --candidate lead_A
```

The standalone command also accepts the run directory alone. The panel saves
the local coordinate ensemble in `screen.json` under
`method_selection.prepared_measurements.crest`. It reports native total
electronic energy in Hartree and relative energy in kcal/mol from the lowest
validated conformer in that run. The output contains up to 32 validated
conformers and flags truncation. Compare the reported energies only within
that ensemble. Solution populations, the bound pose and experimental potency
require their own evidence.
[CREST 3.0.2 release](https://github.com/crest-lab/crest/releases/tag/v3.0.2) ·
[CREST ensemble format](https://crest-lab.github.io/crest-docs/page/documentation/coords.html)

## PROPKA structure-dependent pKa

Use PROPKA to identify ionizable groups whose predicted pKa in a prepared protein
or protein–ligand structure could affect an assay at its planned pH. For an apo
and bound comparison, prepare each structure separately and match the reported
groups by identity. Treat differences as hypotheses for experimental follow-up.
[PROPKA protein method](https://doi.org/10.1021/ct100578z) ·
[PROPKA ligand and coupling method](https://doi.org/10.1021/ct200133y)

Place `propka.json` and the prepared PDB in the finished run directory. The
manifest requires these two fields.

```json
{"version": 1, "structure": "candidate.pdb"}
```

The optional `assay_ph` field accepts a finite number from 0 to 14 and records
the pH for interpretation. Add `candidate_id` with the exact candidate ID when
selecting a named candidate. The manifest links this PDB to the candidate.
Supply one explicitly prepared PDB model with no
alternate locations or insertion codes, at least 10 distinct amino-acid
residues, at most 40,000 atoms, and at most 8 MiB. Use distinct chain IDs and
explicit PDB `TER` records between protein chains. Heavy-atom occupancy must
be greater than zero and at most one. Missing terminal atoms make the result
partial. The screen reads the supplied coordinates without
rewriting them. A sequence, SMILES string, or isolated ligand does not supply
the required protein environment.

Install the optional `propka` extra for PROPKA 3.5.1 and Gemmi in the project
environment. Select `propka` in the picker or use the corresponding CLI command.

```sh
uv pip install ".[propka]"
human --screen finished-run --binder-kind protein --screen-method propka
python scripts/screen_extras.py propka finished-run
```

Add `--candidate ID` to the standalone command when the manifest names that
candidate. The panel saves results in `screen.json` under
`method_selection.prepared_measurements.propka`.
Each group retains its structural identity. The folded pKa estimate and the
native reference model pKa are distinct values. Review any incomplete-input or
coupling flags alongside the estimate. A disulfide group is reported as unscored.
Reported ligand groups describe the nominated complex. A free-ligand pKa or
an assay decision requires evidence for that experiment.
[PROPKA 3.5.1 release](https://pypi.org/project/propka/3.5.1/)

## QupKake microscopic pKa

Place `qupkake.json` containing `{}` beside the finished ligand run, then run
`human --screen RUN --binder-kind ligand`. Supply `ligand_candidates.tsv` with
the header `candidate_id<TAB>smiles`, or one structure in `ligand_smiles.txt`.
The candidate table takes precedence when both exist. To use an SDF, set
`{"input":"sdf"}` and supply `ligand_candidates.sdf`. The record title identifies
each candidate. An untitled record receives an ID from its file ordinal.
One distinct molecular graph is scored once across all pocket designs.

The default profiles the supplied protonation state and tautomer. Setting
`"tautomerize": true` asks the upstream workflow to enumerate tautomers and rank
them with GFN2-xTB in implicit water when several exist. Both input formats
regenerate coordinates with the recorded RDKit seed. The output names the input
and selected states, including when the tautomer search changes the structure.
Submit a single connected, closed-shell species with explicit charge and the
intended stereochemistry. Submit each desired salt component or protomer separately.

Each value in `screen.json.qupkake` describes one aqueous microscopic pKa,
on the dimensionless acid-dissociation scale. A `basic` nomination adds a proton
to the selected state. An `acidic` nomination removes one. Each site identifies
the protonated and deprotonated structures, the actual atom that gains or loses
hydrogen and the atoms nominated by the site model. Atom maps start at one.
For an individual nomination, `source_atom_indices_by_map[map - 1]` gives the
zero-based atom ordinal in its original SMILES or SDF, including any preceding
explicit hydrogens. The local record contains the atom-mapped structures. The MCP
views return their identifiers, site values and coverage.

A compound with two successful site-head evaluations and zero nominations is
recorded as `no_sites`. An unscored nomination carries a blocker. Site coverage
can be incomplete even for a compound with several predictions. A `complete`
batch means that every input and every model nomination was processed. The screen
supplies no macroscopic pKa, pH-dependent species fractions or pass threshold.
The authors used ChEMBL structures, CREST assignments and predicted pKa labels
for pretraining, followed by experimental pKa fine-tuning. Their pretraining
assigned each most acidic or most basic estimate to the lowest-energy conjugate.
Additional transitions in polyprotic compounds have weaker support. Aliphatic
carbon acids were absent from training. Screening accepts the represented
elements H, C, N, O, S, P, F, Cl, Br and I. SAMPL6 was one of the authors'
published benchmarks. [Method and training](https://doi.org/10.1021/acs.jctc.4c00328) ·
[Release data](https://github.com/Shualdon/QupKake/tree/f7a294e91929e4e4f0a7100ad5cf5a64a34a7d4f/data)

### Local CPU runtime

The reviewed backend uses the released Linux x86-64 xTB 6.4.1 executable and a
separate Python 3.10 environment. On Linux or WSL, the following commands prepare
a new installation with `uv` and Git. Run them once from the project root.

```bash
qupkake_root="$HOME/.local/share/rfproteina/qupkake-f7a294e-cpu"
mkdir -p "$qupkake_root"
git clone https://github.com/Shualdon/QupKake.git "$qupkake_root/source"
git -C "$qupkake_root/source" checkout --detach f7a294e91929e4e4f0a7100ad5cf5a64a34a7d4f
uv venv --python 3.10 "$qupkake_root/env"
qpython="$qupkake_root/env/bin/python"
UV_CACHE_DIR="$qupkake_root/cache" uv pip install --python "$qpython" \
  'torch @ https://download.pytorch.org/whl/cpu/torch-2.1.2%2Bcpu-cp310-cp310-linux_x86_64.whl' \
  torch-geometric==2.4.0 pytorch-lightning==2.1.0 torchmetrics==1.2.1 rdkit==2022.9.5 \
  numpy==1.26.4 pandas==2.1.4 scipy==1.11.4 scikit-learn==1.3.2 'setuptools<81'
qpackages=$("$qpython" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
printf '%s\n' "$qupkake_root/source" > "$qpackages/qupkake-release.pth"
chmod u+x "$qupkake_root/source/qupkake/xtb-641/bin/xtb"
.venv/bin/python scripts/screen_extras.py qupkake 'c1ccncc1'
```

Set `QUPKAKE_PYTHON` to the worker interpreter for a custom installation.
The worker verifies the scientific modules, the three checkpoints, the xTB binary
and its GFN2 parameter file before scoring. Each distinct compound gets its own
scratch directory, a single CPU thread and a ten-minute timeout. The standalone
CLI also accepts `--sdf FILE` and `--tautomerize`. It exits with code 2 for a
partial or untested batch and retains per-input results in its JSON output.

## SyGMa human metabolite predictions

SyGMa proposes structures to check in a human metabolite-identification experiment.
Install the project’s `sygma` extra to obtain RDKit with InChI support. From the
project environment, install the release rule tables and try one parent compound.

```bash
python scripts/install_sygma.py
python scripts/screen_extras.py sygma 'CC(=O)Nc1ccc(O)cc1'
```

The installer checks the SHA-256 of the official SyGMa 1.1.0 source archive.
For an offline installation use `--from-archive SyGMa-1.1.0.tar.gz`.
Set `SYGMA_RULE_ARCHIVE` to select a custom archive location. Each worker verifies
the two rule tables before use. Inference runs on CPU. The release contains 148
phase I and 27 phase II transformations. Ridder and Wagener used human metabolism
records in the MDL Metabolite Database to assign empirical rule scores.
Training membership for an individual compound is
unknown. [Method](https://pubmed.ncbi.nlm.nih.gov/18311745/) · [Release](https://pypi.org/project/SyGMa/1.1.0/)

For a finished ligand run, create `sygma.json` containing `{}` beside the
generated structures and run `human --screen RUN --binder-kind ligand`. Provide
`ligand_candidates.tsv` with columns `candidate_id` and `smiles`, or one
structure in `ligand_smiles.txt`. The table takes precedence. To nominate
`ligand_candidates.sdf`, set `"input": "sdf"` in the manifest. The standalone
command accepts the same file through `--sdf FILE`. Each nomination retains its
ID or SDF title. An untitled SDF record receives a file ordinal. Aliases of one
isomeric structure share one calculation across all pockets. Submit individual
connected, closed-shell organic molecules in their intended protonation and
stereochemical states. A disconnected salt receives a named blocker.

The default runs one phase I cycle followed by one phase II cycle. Set
`phase1_cycles` and `phase2_cycles` to nonnegative integers totalling one to
four. For example, `{"phase1_cycles":0,"phase2_cycles":1}` requests direct
phase II products. The corresponding CLI options are `--phase1-cycles` and
`--phase2-cycles`. For each product the screen reports the permitted pathway
with the largest product of empirical rule scores. Ranking starts at one for
each parent. Equal scores share a rank. Intermediates can skip a phase or cycle.
Each reported structure retains more than 15% of the input’s heavy atoms,
traced across transformations. The unchanged parent is excluded from the count.

Read `screen.json.sygma.parents[].products` for the rank, score, rule-by-rule
pathway, standard InChIKey, formula, exact mass in Da for the represented charge,
and number of unspecified stereoelements. Look up each `product_id` in the
local `structures` dictionary for isomeric SMILES. The local MCP service returns
the ranked identifiers and pathways and keeps the SMILES dictionary on disk.
SyGMa can propose a glucuronide while leaving the sugar stereochemistry
unspecified. Check the full stereochemical structure as well as connectivity
when comparing with an authentic standard. Scores alone establish neither
metabolite abundance nor toxicity.

A completed parent with no eligible product reports `no_products`. Missing or
damaged rules, unsupported inputs, and exceeded computation limits report a
named blocker. Limits include 256 parent heavy atoms, 120 seconds per parent,
2,048 distinct combinations of a structure and its surviving parent atoms, 1,024 matches per rule application
and 25,000 cumulative reaction matches per parent. Incomplete parents receive
no ranking. Failed nominations remain in the reported input count.

## MetFrag measured spectrum comparison

Use MetFrag to compare nominated small molecules against one measured tandem
mass spectrum, for example to prioritize tentative metabolite assignments for
reference-standard follow-up. Supply a centroided product-ion list for a single
measured precursor in positive ion mode with the `[M+H]+` adduct. Confirming
identity requires evidence from an authentic standard under comparable
experimental conditions.
[MetFrag method](https://doi.org/10.1186/s13321-016-0115-9) ·
[Identification confidence](https://doi.org/10.1021/es5002105)

Place `metfrag.json` and the nominated TSV in the finished ligand run directory.
Use a local TSV filename and include all seven fields.

```json
{
  "version": 1,
  "peaks": "product_ions.tsv",
  "adduct": "[M+H]+",
  "precursor_mz": 195.08879,
  "precursor_ppm": 10,
  "fragment_ppm": 10,
  "fragment_da": 0.001
}
```

Use the instrument's monoisotopic precursor ion m/z, from 20 to 1,500. The
precursor window is 0.5–50 ppm around the measured ion m/z. Candidate neutral
exact masses are converted to `[M+H]+` before this comparison. Product-ion
matching adds the configured 0–0.05 Da and 0–100 ppm tolerances. At least one
product tolerance must be positive. Choose tolerances that match the acquisition.
[MetFrag CLI parameters](https://ipb-halle.github.io/MetFrag/projects/metfragcl/)

Use the exact tab-separated `mz` and `intensity` header. Supply 3–500 distinct
centroid product ions with m/z at least 1 and below the measured precursor,
and finite positive intensities up to 10¹⁵. The TSV limit is 64 KiB. These three
rows illustrate the format using an excerpt from a measured caffeine spectrum.
[MassBank caffeine record MSBNK-mFam-MC23_000485](https://massbank.eu/MassBank/RecordDisplay?id=MSBNK-mFam-MC23_000485&raw=)

```tsv
mz	intensity
83.0601	226
108.0511	342
138.0649	776
```

Nominate candidates in `ligand_candidates.tsv` with `candidate_id` and `smiles`
columns, or put one SMILES in `ligand_smiles.txt`. A screen accepts up to 32
selected candidates. Use neutral, connected organic molecules with 4–40 heavy
atoms and elements H, C, N, O, F, P, S, Cl, Br and I. Isotopes, radicals and
atom-mapped SMILES are outside this workflow's scope.

Install the verified MetFrag 2.6.11 jar in the active Linux or WSL project
environment. The offline calculation uses an existing Java 21 runtime in an
isolated network namespace. Add `--jar PATH` to the installer to use a local
copy of the official jar. [MetFrag 2.6.11 release](https://github.com/ipb-halle/MetFragRelaunched/releases/tag/v2.6.11)

```sh
python scripts/install_metfrag.py
human --screen finished-run --binder-kind ligand --screen-method metfrag
python scripts/screen_extras.py metfrag finished-run
```

For one candidate, add `--candidate ID` to the standalone command and use a
unique exact ID from `ligand_candidates.tsv`. The panel saves results under
`method_selection.prepared_measurements.metfrag` in `screen.json`. Read the
raw `FragmenterScore` with the explained and used peak counts. Compare scores
for candidates tested against the same spectrum with the same settings.
Each distinct eligible structure is fragmented once. Multiple candidate IDs can
refer to the same canonical isomeric SMILES and each ID keeps its result.
Stereoisomers and distinct tautomeric graphs are evaluated separately. The
`comparison` field counts tested nominations, distinct tested structures and
distinct structures sharing the highest positive raw score. If every tested
structure explains zero peaks, the count of top-scoring structures is zero.
The board derives these counts from the candidate rows when opening older results.
A candidate outside the precursor mass window is
`untested`. When no product ion matches a candidate, the result records
`no_matches`.

## OpenAntigens human target evidence

Use OpenAntigens when planning experimental validation for a binder against a
human cell-surface or secreted protein. The lookup records the target's
secreted, GPI-anchored, single-pass or multipass classification and whether the
release has AlphaFold, PDB, InterPro, family or assembly context. A PDB flag
calls for checking the deposited structure. An unset flag indicates a gap
in that source's coverage for this snapshot. The result describes the run
target. Binding, expression and cross-reactivity require evidence for the
individual candidate and assay.
[OpenAntigens methods](https://openantigens.org/methods.html)

Install the pinned compact human index and accession mapping in the project
environment. The lookup then runs offline against the 2026-11-01 snapshot.
The OpenAntigens paper describes the earlier 2026-07-23 release.
[November release metadata](https://openantigens.org/portal_metadata.json) ·
[OpenAntigens paper](https://doi.org/10.1093/nar/gkag1051) ·
[Data terms](https://openantigens.org/terms.html)

The finished-run lookup reads the exact canonical UniProt accession from
the archived generation record or a regular job record. For a standalone
directory or a run without a recorded target, supply `openantigens.json`
with these two fields. The nomination and any recorded targets must agree.

```json
{"version": 1, "target_accession": "P00533"}
```

```sh
python scripts/install_openantigens.py
human --screen finished-run --binder-kind protein --screen-method openantigens
python scripts/screen_extras.py openantigens finished-run
```

The panel saves the lookup in `screen.json` under
`method_selection.prepared_measurements.openantigens`.

## P2Rank receptor pockets

Use P2Rank to nominate small-molecule binding sites when the site on a protein
receptor is uncertain. For a secondary comparison, review whether a predicted
pocket overlaps a known site in the supplied receptor's coordinate frame. The
screen ranks pockets on one receptor for the run. Use its centers and residue
identifiers to choose sites for structural review or a subsequent docking study.
Ligand ranking, binding evidence and candidate validation require separate methods.
[P2Rank method](https://doi.org/10.1186/s13321-018-0285-8)

In the finished ligand run directory, place a regular PDB, CIF or mmCIF receptor
file beside `p2rank.json` and give all three manifest fields.

```json
{"version": 1, "receptor": "receptor.pdb", "structure_source": "xray"}
```

Use a single model with at least 20 amino-acid residues, at most 40,000 atoms,
and identifiable chain and residue IDs. The receptor file limit is 8 MiB. Declare
the structure source that matches the recorded method when the file contains
method metadata. Use `xray` for the default P2Rank model, which uses B-factors.
At least one amino-acid heavy atom must have a positive B-factor for this profile.
Use `alphafold`, `predicted`, `cryoem`, or `nmr` for the `alphafold` profile,
which omits B-factors.

Use Java 17–24 and Gemmi in the screening environment. The screen looks for
Java under `share/p2rank/java` in the current Python environment, then under
`JAVA_HOME` and on `PATH`. Prediction uses one CPU thread, a 2 GiB Java heap
limit and a two-minute timeout. Install the verified P2Rank 2.5.1 release,
then select the screen for the run.

```bash
python scripts/install_p2rank.py
human --screen finished-run --binder-kind ligand --screen-method p2rank
```

For an offline installation pass `--archive FILE` to the installer with the
pinned release archive. The installer checks the archive checksum and records
the installed file hashes in the project environment. Run the standalone screen
with `python scripts/screen_extras.py p2rank finished-run`.
[P2Rank 2.5.1 release](https://github.com/rdk/p2rank/releases/tag/2.5.1)

Results are saved in `screen.json` under
`method_selection.prepared_measurements.p2rank`. Each pocket has a native score,
a probability specific to the selected model, a center in Ångströms in the
receptor frame, and residue IDs from the supplied structure. An accepted
header-only report records `complete` with an empty pocket list.

## AutoDock Vina fixed-pose score

For a finished ligand run, put `vina_poses.json` beside `inference_samples/`. Give each
design its exact sample ID, the mmCIF `label_asym_id` for each partner, the ligand component
ID, and the paths to its prepared PDBQT files relative to the manifest. A single author
chain can contain both a protein and several ligand copies. The label IDs identify which
copy supplied the coordinates. This example selects the mouse Abl kinase domain labelled
A and the imatinib copy labelled G in the deposited structure. [PDB 1IEP](https://www.rcsb.org/structure/1IEP)

```json
{
  "samples": {
    "binder_0": {
      "receptor": "prepared/receptor.pdbqt",
      "ligand": "prepared/ligand.pdbqt",
      "receptor_chains": ["A"],
      "ligand_chain": "G",
      "ligand_component": "STI",
      "center": [15.190, 53.903, 16.917],
      "size": [20, 20, 20]
    }
  }
}
```

Install the project's `vina` Python extra and the upstream AutoDock Vina 1.2.7 CPU executable.
Set `AUTODOCK_VINA_EXEC` to that executable, place it under `autodock-vina/` in the local
screening tools directory, or put `vina` on `PATH`. On Windows the binary name is `vina.exe`.
Both commands use the same scorer.

```bash
python scripts/screen_pose_safety.py vina complex.cif --manifest vina_poses.json --sample binder_0
human --screen finished-run --binder-kind ligand
```

The receptor comprises the selected standard protein chains. The ligand is a single
noncovalent organic component. Select a single model and resolve alternative conformers
before preparation. Supply the rigid receptor and one unwrapped PDBQT pose in the same
coordinate frame as the finished complex. Protonation, charges, donor and acceptor types,
and the torsion tree come from that preparation. Water, cofactors and other chains lie
outside this protein-only receptor model. A declared covalent or metal link to the ligand
blocks a score.

Every observed heavy atom must match exactly once by element and position to within
0.000501 Å per Cartesian axis, the rounding allowance for PDBQT. Receptor residue and
atom names must also agree. The report identifies whether an embedded component atom
list, a molecular formula, or only the observed coordinates supported the ligand inventory.
Where the CIF supplies bond data, it also reports the number of aromatic/aliphatic carbon
types and heavy-atom torsion bonds checked against the component.

The `center` and `size` arrays use Å in x, y, z order. The supplied box must contain
the ligand heavy atoms. The executable uses `--score_only`, the original Vina function,
one CPU thread, seed 1 and 0.375 Å grid spacing. The adapter limits each PDBQT file to
32 MiB, grid allocations to four million points and execution to 120 seconds per pose.
It records the score in kcal/mol, preparation and executable hashes, and the version.
The original function was fitted on PDBbind affinities. Interpreting its score as a
measured affinity or applying a selection cutoff requires separate assay calibration.
[Vina method](https://doi.org/10.1002/jcc.21334) · [Upstream fixed-pose example](https://autodock-vina.readthedocs.io/en/latest/docking_python.html) · [Executable release](https://github.com/ccsb-scripps/AutoDock-Vina/releases/tag/v1.2.7)

`screen.json` stores per-design results under `vina_pose.samples`. Each discovered
structure counts once, including unreadable files and designs missing settings.
`unmatched_manifest_ids` lists manifest entries with no corresponding structure.
Rows with a blocker carry `status: untested`. These descriptive scores have no
pass/fail threshold and do not enter the screening verdict.

## CellProfiler chromogenic tissue and chip imaging

Use `cellprofiler` to quantify hematoxylin-counterstained, DAB-developed brightfield
images from a human frozen tissue panel or a human chip. A reviewer outlines the
anatomical region in each field. CellProfiler 4.2.8 separates the two stains with
its built-in hematoxylin and DAB presets and measures the DAB channel inside
each mask. This preserves signal from small annotated regions that a whole-slide
mean could obscure. The source images, tissue identity, reagent assignments and
control performance still need review by the study pathologist.
[UnmixColors](https://cellprofiler-manual.s3.amazonaws.com/CPmanual/UnmixColors.html)
and [MeasureImageIntensity](https://cellprofiler-manual.s3.amazonaws.com/CPmanual/MeasureImageIntensity.html).

Install the host image reader and pull the pinned, CPU-only CellProfiler release
into the local Docker engine once. The container uses the released code and
requires no model weights. The unpacked container filesystem measured 3.27 GB.
The native worker reached 702 MB peak resident memory while processing four
2,048 × 2,048-pixel fields in one batch. The worker container has a 3 GiB memory limit.

```bash
uv pip install ".[cellprofiler]"
docker pull --platform linux/amd64 cellprofiler/cellprofiler@sha256:fec440caa2b44edf80f9bd440a3d8c6b214d72437332ad2586a32d79e2eae2a4
```

Place `cellprofiler.json` in the finished run. This example compares renal
cortex on candidate and isotype-control sections from the same coded specimen.
Use the run's candidate ID and record reagent lots, staining batches and
acquisition settings under stable lab identifiers. The manifest links those
identifiers to the images. Keep the specimen key and acquisition log in the lab.

```json
{
  "stains": "hematoxylin_dab",
  "fields": {
    "cortex_test": {
      "role": "candidate", "candidate_id": "antibody_A", "reagent_id": "test_lot_17",
      "material": "human_frozen_tissue", "specimen_id": "specimen_01",
      "tissue": "Kidney", "region": "Cortex", "assay_batch": "batch_01",
      "acquisition_id": "scan_profile_01",
      "image": "fields/test.png", "mask": "fields/test_cortex.png"
    },
    "cortex_isotype": {
      "role": "isotype_control", "reagent_id": "isotype_lot_03",
      "material": "human_frozen_tissue", "specimen_id": "specimen_01",
      "tissue": "Kidney", "region": "Cortex", "assay_batch": "batch_01",
      "acquisition_id": "scan_profile_01",
      "image": "fields/isotype.tif", "mask": "fields/isotype_cortex.png"
    }
  },
  "comparisons": [
    {"comparison_id": "kidney_cortex_01", "candidate_id": "antibody_A",
     "test": "cortex_test", "negative": "cortex_isotype"}
  ]
}
```

Use an independently acquired `isotype_control` or `no_primary_control` appropriate
to the experiment. The test and negative fields must declare the same material,
specimen, tissue, region, staining batch and acquisition settings. A separate
known-positive field can be attached using `"positive": "field_id"` and
`"role": "positive_control"`. It can come from another specimen or from an
`assay_control` such as a cell pellet. Its staining batch and acquisition settings
must match the test field. The report states whether a positive control was
measured. The reviewer determines whether it stained as expected.

Each image is one lossless 8-bit RGB field in PNG or classic TIFF format. Supply a
separate, nonempty binary mask on that image's own pixel grid, using 1-bit pixels
or 8-bit grayscale with values 0/1 or 0/255. Use top-left orientation and
run-relative file paths. A mask may enclose different numbers of pixels on the two
sections. The person preparing each mask verifies that it overlays its original
field. Measurements use each region's mean and median. Export one field per
specimen from whole-slide scans or tissue microarrays before
running the screen. Each file is limited to 20 MiB and each field to 4,194,304 pixels.
The manifest supports 1,024 field definitions and 512 comparisons. Across eligible
fields, a request can process up to 4 GiB of encoded image and mask data and
1,073,741,824 image pixels. A field used in several comparisons counts once.

```bash
python scripts/screen_extras.py cellprofiler finished-run --candidate antibody_A
human --screen finished-run --binder-kind protein --screen-method cellprofiler
```

The picker and candidate MCP expose the same method. Finished-run results are in
`screen.json → method_selection → prepared_measurements → cellprofiler → comparisons`.
Use the field ID, image hash and mask hash to locate the exact region for review.
The signed `test_minus_negative_mean_dab` is the difference between two regional
means on CellProfiler's normalized intensity scale. Retain negative differences
and report each donor and anatomical region separately. Reuse a field by its ID
when it legitimately serves several comparisons. A candidate-specific execution
checks for contradictory use of the same pixels in other eligible fields, and
returns measurements only for the selected candidate.

Use `human_chip` for identically stained chip images. A reagent-control contrast
measures staining background. Establishing a treatment effect requires biological
comparators. These image measurements alone cannot establish antibody specificity,
tissue injury, or absence of cross-reactivity in an unexamined tissue.

## PK-Sim human organ exposure

Use `pksim` when a small molecule has a parameterized human PBPK model and a
dosing regimen relevant to the exposure question. The screen runs the native
PK-Sim 12.3.173 solver on CPU. One nomination selects one individual simulation
and one dosed compound. Outputs can include a separately modelled, undosed
analyte when it is present in that scenario. In a linked-chip study, compare
each prediction with measurements for the same compound, compartment, regimen
and time window. Concentration alone cannot establish organ injury.
[PK-Sim simulations](https://docs.open-systems-pharmacology.org/working-with-pk-sim/pk-sim-documentation/pk-sim-simulations)

Install the project extra and the pinned portable Windows release. WSL uses
Windows executable interop and a cache on a Windows-accessible drive.
The portable release is 109 MB to download and occupies 237 MB extracted.
The cimetidine control below used CPU only. Recording four concentration curves
for that scenario gave an observed peak solver working set of 541 MB on Windows. Memory
use depends on the model and the number of requested outputs.
Model registration is a local operator action. Review the PK-Sim project
snapshot and verify its compound identity before registering it. The registry
binds the exact snapshot bytes, compound name and full standard InChIKey.
PK-Sim compound names are modeler-defined. The screen independently derives
the candidate's InChIKey from its submitted SMILES.

```bash
uv pip install ".[pksim]"
python scripts/install_pksim.py
python scripts/install_pksim.py --model Cimetidine-Model.json --compound Cimetidine --inchikey AQIXAKUUQRKLND-UHFFFAOYSA-N
```

This example uses the original OSP cimetidine project and its individual IV
simulation. The registered digest printed by the installer must match the
digest in the manifest. Re-exporting a project can change its digest.
[Exact reference snapshot](https://github.com/Open-Systems-Pharmacology/Cimetidine-Model/blob/ee1d8f6e933298a79f8a0dd7be47fffa6e02cd7c/Cimetidine-Model.json)

```json
{
  "samples": {
    "cimetidine_control": {
      "model_sha256": "314c835349119455bc5c60c16febbf301b55de97eb0f3e4279313949b4b6f8aa",
      "simulation": "iv 100 mg (5 min), Grahnen 1979, n=3",
      "compound": "Cimetidine",
      "outputs": ["Organism|PeripheralVenousBlood|Cimetidine|Plasma (Peripheral Venous Blood)"]
    }
  }
}
```

Save this as `pksim.json`. Put the matching `candidate_id` and `smiles` in
`ligand_candidates.tsv`. For this control the two tab-separated columns are
shown below. [Cimetidine identity](https://pubchem.ncbi.nlm.nih.gov/compound/2756)

```tsv
candidate_id	smiles
cimetidine_control	CC1=C(N=CN1)CSCCNC(=NC)NC#N
```

A run containing only `ligand_smiles.txt` can use
`run_ligand` for a run-level request. For gut, hepatic or renal exposure, select
the exact concentration observers in the model, export and register the updated
project, and name those paths in `outputs`.

```bash
python scripts/screen_extras.py pksim finished-run --candidate cimetidine_control
human --screen finished-run --binder-kind ligand --screen-method pksim
```

The picker and MCP use the same method. The finished-run report stores each
case in `screen.json → method_selection → prepared_measurements → pksim → samples`.
Each curve retains its native concentration and time units, exact output path,
time points, and simulated values. `grid_cmax` is the highest sampled
concentration and `grid_tmax` is the first sampled time attaining it.
`auc_trapezoidal` integrates between `auc_start` and `auc_end` on the native
grid. Blood, whole tissue, interstitial fluid and lumen are distinct quantities.
The native export and project each have a SHA-256 digest. The screen reports
unusable scenarios individually and leaves the safety verdict to its existing
assays.

A run accepts up to eight candidate scenarios, eight concentration paths per
scenario and 8,192 concentration values per scenario. Population simulations,
scenario-local building-block overrides, coadministration, amounts, fractions
and excretion rates are outside the supported individual concentration endpoint. Each model execution has a
180-second timeout. [PK-Sim release](https://github.com/Open-Systems-Pharmacology/PK-Sim/releases/tag/v12.3.173)

## EPA httk steady-state exposure screen

Choose **EPA httk steady-state exposure** for batch prioritization of small
molecules that already have usable human hepatocyte clearance and plasma
protein binding measurements in the package. It shows how modeled plasma
concentrations vary across an adult population under the same reference
scenario. As a secondary use, compare the estimate with more detailed PK
work for the *same substance* and a compatible exposure scenario. Do not
pool these values with concentration–time curves, clearance, bioactivity, or
hazard scores. [EPA httk overview](https://www.epa.gov/comptox-tools/high-throughput-toxicokinetics-httk-r-package)

Supply up to 32 named small molecules per batch. In this example,
`ligand_candidates.tsv` contains two tab-separated columns.

```tsv
candidate_id	smiles
caffeine	Cn1c(=O)c2c(ncn2C)n(C)c1=O
acetaminophen	CC(=O)Nc1ccc(O)cc1
```

Save the matching nominations in `httk.json`.

```json
{
  "scenario": "human_oral_unit_intake",
  "candidates": {
    "caffeine": {"cas": "58-08-2", "dtxsid": "DTXSID0020232"},
    "acetaminophen": {"cas": "103-90-2", "dtxsid": "DTXSID2020006"}
  }
}
```

RDKit derives the full standard InChIKey from each submitted structure. Its
CAS number, EPA substance ID and InChIKey must match one record in the pinned
DSSTox release. EPA httk must independently resolve both identifiers to one
curated row. A mismatch, ambiguous association, or missing reference leaves
the candidate untested. The compact index covers 14,980 CAS/DTXSID pairs
with an unambiguous standard InChIKey in the intersection of the two releases.
Further assay and model requirements determine which of these can be simulated.
[EPA DSSTox release](https://doi.org/10.23645/epacomptox.5588566.v8).

The fixed model is EPA httk 2.7.4 `calc_mc_css` with `3compartmentss`, a
standardized oral intake of **1 mg/kg/day**, and the `mg/L` total-plasma
readout. It uses 1,000 seeded direct HTTK-Pop resamples of human adults ages
18–79, holding the in-vitro measurements fixed. The 5th percentile, median,
and 95th percentile summarize simulated variation across adults under this
reference intake. The model retains EPA's oral-absorption and other physiological
assumptions. Each reported member also records the unadjusted assay central values
for human intrinsic clearance (µL/min per million hepatocytes) and plasma fraction
unbound (unitless), with the package's source reference for each measurement.
[EPA httk model documentation](https://cran.r-project.org/web/packages/httk/refman/httk.html#calc_mc_css)

Skip the screen for novel structures without a matching reference,
non-oral scenarios, or candidates for which the reference input is missing,
unusable, or outside the model's applicability checks. It does not substitute
ADMET-AI microsomal clearance in `uL/min/mg` for hepatocyte clearance or
construct binding data from a predicted score. It adds no cutoff and does not
change the screening verdict.

Install the pinned EPA R package and its dependencies into a trusted R library.
The CRAN 2.7.4 source archive is 4,160,963 bytes with SHA-256
`e186171a90d97fe456e7836ec3096719feb49354afd71860df2d2a5c8c4c51b2`.
[CRAN httk](https://cran.r-project.org/web/packages/httk/index.html).

```r
install.packages("remotes", repos = "https://cloud.r-project.org")
remotes::install_version("httk", version = "2.7.4", repos = "https://cloud.r-project.org", upgrade = "never")
```

Configure
`RFD4_HTTK_RSCRIPT` with that runtime's `Rscript` executable, use
`DATA_PATH/httk/bin/Rscript` on Linux, or place the trusted `Rscript` on `PATH`.
Install the project extra and build the local identity index once. The setup
command streams the 290 MB EPA ZIP archive, verifies its SHA-256, and installs a
792,035-byte index under `DATA_PATH/.cache/httk/2.7.4` (the default is the checkout's
`data` directory). For an existing downloaded archive use
`--from-file DSSTox_CCD_dump_12092025_CSVs.zip`. Screening runs offline on CPU.
The tested R 4.5.3 environment occupied 1.55 GB. A batch of 32 distinct medicines
finished in 21.45 seconds with a sampled peak of 543 MB resident memory across
Python and R. It returned 26 estimates and recorded six candidates with unusable
human assay inputs. The package uses no GPU or model weights.

```bash
uv pip install ".[httk]"
python scripts/install_httk_identity.py
python scripts/screen_extras.py httk finished-run --candidate caffeine
human --screen finished-run --binder-kind ligand --screen-method epa_httk
```

The browser and MCP offer the screen for explicit selection. MCP `run_selected_screens` accepts
`methods: ["epa_httk"]`. In a CLI screening record the results are stored under
`method_selection.prepared_measurements.epa_httk` with per-candidate coverage
and no raw structures.

## CYPstrate substrate profiling

Use this screen when metabolism by a particular human CYP is relevant to a systemic
small molecule. Declare the exposure and name each candidate's isoforms in
`cypstrate.json`. Candidate IDs refer to the `candidate_id` and `smiles` columns in
`ligand_candidates.tsv`. A run with only `ligand_smiles.txt` can use the single
manifest ID `run_ligand` for a run-level request.

```json
{
  "systemic": true,
  "prediction_mode": "best_performance",
  "candidates": {
    "binder_0": ["CYP1A2", "CYP3A4"],
    "binder_1": ["CYP2D6"]
  }
}
```

Supported isoforms are CYP1A2, CYP2A6, CYP2B6, CYP2C8, CYP2C9, CYP2C19,
CYP2D6, CYP2E1 and CYP3A4. The default `best_performance` mode uses the authors'
hard-voting classifiers. Each requested pair can receive a substrate classification,
a non-substrate classification or an abstention. Select `full_coverage` explicitly
to use the single classifier for each isoform. Both modes share the development
corpus of 1,831 compounds curated from Hunt and Tian. [CYPstrate study](https://doi.org/10.3390/molecules26154678)

Install the released Linux/AMD64 image once on the local Docker engine. The
project's CYPstrate extra supplies RDKit for reading nominations. The image
contains the authors' Python, preprocessing code and fitted models in about
425 MB of compressed layers. Inference runs on CPU with networking disabled.

```bash
uv pip install ".[cypstrate]"
docker pull --platform linux/amd64 ghcr.io/molinfo-vienna/cypstrate@sha256:587227598961627192f5c007757df46369d2a2960e004b726a95eb1bf3774d5b
python scripts/screen_extras.py cypstrate finished-run
human --screen finished-run --binder-kind ligand --screen-method cypstrate
```

The picker and MCP accept `cypstrate`. The standalone command also accepts
`--candidate binder_0` to return that candidate's nominated isoforms. Finished-run results appear under
`screen.json → method_selection → prepared_measurements → cypstrate`.
Each row retains its method, isoform and model status. `nearest_training_maccs_tanimoto`
is molecular similarity to the nearest training example for that isoform. The
submitted and standardized molecular forms have separate identity digests because
the upstream preprocessing can change salts, tautomers and stereochemistry.
The screen accepts up to 256 nominated candidates, a 64 KiB manifest, a 1 MiB
molecular table and 256 atoms per submitted form.

Substrate status identifies a metabolism hypothesis for assay planning. The CYP
inhibition panel reports its own observations. Clearance rates and clinical
interaction risk require separate evidence. [Authors' model and output definitions](https://github.com/molinfo-vienna/cypstrate/tree/c153960de7fba9b9b07047b7f20e55277a88fb41)

## Badapple 2 historical scaffold assay evidence

Use Badapple 2 when deciding which assay-interference controls or orthogonal
assays to include for a cyclic small-molecule lead. For a secondary review,
compare the scaffold records for a short analogue series using the same snapshot.
The local HierS workflow looks up each scaffold in the publisher's historical
PubChem assay summary. The dump snapshot is `2025-08-08`. The publisher lists
`2024-11-26` as the data cutoff. Treat each pScore as scaffold-level historical
evidence. Establish candidate activity, selectivity and safety with the relevant
experiments. [Badapple 2 study](https://doi.org/10.1021/acs.jcim.5c02297) ·
[Publisher data description](https://chiltepin.health.unm.edu/badapple2/about.html)

Supply `ligand_candidates.tsv` with the header `candidate_id<TAB>smiles`, or
one plain SMILES in `ligand_smiles.txt`. The table takes precedence. Each screen
accepts up to 32 selected entries. Use a single-component cyclic organic
molecule with up to 80 heavy atoms and a SMILES up to 512 characters. The
worker caps enumeration at five ring systems and 128 scaffold keys per molecule.

### Local CPU setup

In the project environment, install the pinned data with the PostgreSQL client
tool `pg_restore` available on `PATH`. The installer verifies the publisher's
slim archive, streams its scaffold table into SQLite, and records the snapshot
and file digest. This uses the client to read the dump offline. Candidate
screening reads the local SQLite database and uses the isolated HierS worker.
Run these commands once from the project root to set up the worker on Linux or WSL.

```bash
badapple_worker="$(python -c 'import sys; from pathlib import Path; print(Path(sys.prefix) / "share" / "badapple2" / "chem-env")')"
uv venv --python 3.12 "$badapple_worker"
uv pip install --python "$badapple_worker/bin/python" rdkit==2024.3.3 scaffoldgraph==1.1.1 numpy==1.26.4
python scripts/install_badapple2.py
```

Use `--dump FILE` for a local copy of the slim archive and `--pg-restore PATH`
for a PostgreSQL client outside `PATH`. The installed snapshot lives at
`share/badapple2/2025-08-08/scaffolds.sqlite` under the Python environment.
[Publisher archive instructions](https://github.com/unmtransinfo/Badapple2#easy-setup)

```bash
human --screen finished-run --binder-kind ligand --screen-method badapple2
python scripts/screen_extras.py badapple2 finished-run --candidate lead_A
```

The stand-alone `--candidate` option selects a named TSV entry. The finished-run
screen saves evidence in `screen.json` under
`method_selection.prepared_measurements.badapple2`. Read each pScore alongside
the scaffold's active and tested assays, compounds, substances and samples.
`in_drug` records whether the scaffold appears in the publisher's drug set.
A numeric zero is a reported score. A matched scaffold with a null pScore is
`unscored`. Missing matches are `not_in_corpus` and multiple matches are
`ambiguous`. When no scaffold has a reported score, the candidate remains
`untested`.

## SCScore synthetic complexity

Use SCScore before chemistry review when other lead-selection criteria leave two or
more small molecules otherwise comparable. Declare that set in `scscore_comparison.json`.
The names refer to the `candidate_id` and `smiles` columns in the run's
`ligand_candidates.tsv`.

```json
{"candidates": ["binder_0", "binder_1"]}
```

If a verified route and procurement costs settle this comparison, record the decision
and its two source identifiers in the same manifest.

```json
{
  "candidates": ["binder_0", "binder_1"],
  "decision_evidence": {
    "verified_route_ref": "ELN-418",
    "procurement_costs_ref": "QUOTES-27",
    "resolves_comparison": true
  }
}
```

The reviewer attests that those records resolve the named comparison. The screen
then records a skip even when the checkpoint is absent. An individual reference,
or two references with the decision still open, leaves the comparison eligible.
A claim of resolution missing either source returns a blocker.

```bash
uv pip install ".[scscore]"
python scripts/install_scscore.py
human --screen finished-run --binder-kind ligand --screen-method scscore
```

The installer verifies the authors' 6,178,760-byte compressed JSON checkpoint for
the 1024-bit Boolean model. An offline install accepts `--from-file FILE`. Screening
uses local RDKit chiral Morgan fingerprints and NumPy on CPU. The model learned
relative complexity from about 12 million Reaxys reactions. Its dimensionless
scores range from 1 to 5. Higher values indicate greater predicted synthetic
complexity. Establishing route feasibility requires chemistry review. The RDKit
synthetic-accessibility score has its own method and scale.
[Authors' implementation and training description](https://github.com/connorcoley/scscore/tree/37090a6aa8220408f572de20224c33c221312d16)

Results appear under `screen.json → method_selection → prepared_measurements → scscore`.
Each scored member has its candidate ID, molecular identity digest and unrounded
score. A complete comparison receives ranks from lowest complexity upward, with
equal scores sharing a rank. A partial comparison reports the individual results
and blockers. A candidate-scoped request measures only that member and omits its
rank. The same inputs work outside a generated run with
`python scripts/screen_extras.py scscore DIRECTORY`.

The comparison accepts up to 256 named nominations from a table of at most 1 MiB,
with one connected organic molecule per nomination. Supply the intended chemical
form. Duplicate IDs, salts and mixtures require an unambiguous nomination.
Equivalent SMILES identify one compound for inference. A comparison needs at least
two distinct supported molecules.

## sPyRMSD redocking controls

Use this screen to measure how closely a redocked ligand recovers its experimentally
observed binding pose. Place `spyrmsd_controls.json` and the two ligand files in the
finished run directory. Each entry names the generated candidate and its redocked pose.
Omit candidates without a matched experimental reference.

```json
{
  "samples": {
    "binder_0": {
      "pose": "binder_0_redocked.sdf",
      "reference": "crystal_ligand.sdf",
      "experimental_reference_id": "crystal23:model1:A:LIG:401",
      "pose_frame": "crystal23.model1.prep1",
      "reference_frame": "crystal23.model1.prep1"
    }
  }
}
```

Supply two distinct, single-record 3D SDF or MOL files with bond information and
coordinates in ångströms. Identify the experimental structure, model, chain and
ligand instance in `experimental_reference_id`. Copy each frame ID from the
corresponding coordinate preparation. These IDs start with a letter or digit and
accept letters, digits, periods, underscores, colons, plus signs and hyphens, up
to 128 characters. If the receptors started in different frames,
apply the receptor alignment to the ligand coordinates before export. The screen
checks equality of the declared IDs. Ligand coordinates alone cannot verify that
alignment or establish the origin of the reference.

A control requires the same connected chemical form, including bond orders,
formal charges, isotopes and stereochemistry. The score is the smallest heavy-atom
RMSD over element- and heavy-isotope-labelled graph mappings. Both poses retain their
supplied coordinates throughout, so displacement within the pocket contributes to the
result. The calculation has no fitted model or pass threshold. PoseBusters supplies
the separate physical plausibility checks. [sPyRMSD method](https://doi.org/10.1186/s13321-020-00455-2)

Install the project's `spyrmsd` extra and select the screen on a finished ligand run.

```bash
uv pip install ".[spyrmsd]"
human --screen finished-run --binder-kind ligand --screen-method spyrmsd
```

For a standalone redocking experiment, use the same manifest and ligand files in a
control directory. The sample names then identify the controls in that experiment.

```bash
python scripts/screen_pose_safety.py spyrmsd controls --sample binder_0
```

The standalone command prints JSON. The finished-run screen saves the result under
`screen.json → method_selection → prepared_measurements → spyrmsd`.
Each measured sample records `rmsd_angstrom`, the experimental reference and frame IDs,
the heavy-atom count and both input hashes. The candidate picker and MCP tool
`run_selected_screens` also accept `spyrmsd`. A missing match or incompatible frame
supplies a skip reason. In a mixed run, failed nominations retain their blockers and
count toward `attempted`. The interactive limits are 128 nominated controls, a 64 KiB
manifest, 1 MiB per ligand file, 256 heavy atoms and 4,096 atom mappings per control.
A control exceeding the mapping limit returns `untested`. [sPyRMSD release](https://pypi.org/project/spyrmsd/0.9.0/#files)

## ESP-Sim prealigned ligand comparison

Use ESP-Sim to compare a candidate pose with a relevant reference pose
already placed in the same coordinate frame. The supplied coordinates define
the comparison. The scores can guide inspection of shape and electrostatic
differences. Binding and potency require target-specific experimental evidence.
[ESP-Sim study](https://doi.org/10.1021/acs.jcim.1c01535)

Place `espsim.json` and two separate SDF files in the finished ligand run
directory. Use distinct local SDF filenames and distinct candidate and
reference IDs. Give all six manifest fields.

```json
{
  "version": 1,
  "candidate_id": "lead_A",
  "reference_id": "reference_A",
  "candidate_structure": "lead_A.sdf",
  "reference_structure": "reference_A.sdf",
  "prealigned": true
}
```

Each SDF must contain one valid 3D record with explicit hydrogens and a title
that exactly matches its manifest ID. Use connected, uncharged organic
molecules containing carbon, with 3–48 heavy atoms and at most 128 atoms per
record. Supported elements are H, C, N, O, F, P, S, Cl, Br and I. Isotopes
and radicals are outside the supported input scope. Express both poses in
ångströms in one common frame. The combined span must be at most 48 Å along
each coordinate axis. If `ligand_candidates.tsv` includes the nominated
candidate ID, its molecular graph must agree with the candidate SDF.

Install the verified ESP-Sim 0.0.1 scoring source in the active project
environment. The comparison uses RDKit and NumPy on the CPU. Add
`--archive ARCHIVE` to use a local copy of the official source archive.
[ESP-Sim 0.0.1 release](https://pypi.org/project/espsim/0.0.1/)

```sh
python scripts/install_espsim.py
human --screen finished-run --binder-kind ligand --screen-method espsim
python scripts/screen_extras.py espsim finished-run
```

Add `--candidate ID` to the standalone command to select the exact candidate
named in the manifest. The panel saves results in `screen.json` under
`method_selection.prepared_measurements.espsim`.

The heavy-atom RDKit shape Tanimoto score is dimensionless and ranges from
0 to 1. The electrostatic Carbo score comes from the analytic Gaussian
overlap of the two potentials using Gasteiger charges computed in 12 iterations.
Its raw dimensionless range is −1 to 1. A negative Carbo value denotes
opposing potentials in the supplied frame. Read the two scores independently.
An available shape score can accompany an unscored electrostatic result with
its blocker recorded.
[ESP-Sim scoring implementation](https://github.com/hesther/espsim) ·
[RDKit shape documentation](https://www.rdkit.org/docs/source/rdkit.Chem.rdShapeHelpers.html)

## PRODIGY protein-pair affinity

Install the project's `prodigy` extra in the local screening environment. To score a finished
complex once, supply both protein partners. A partner can contain several chains.

```bash
python scripts/screen_extras.py prodigy complex.cif --binder H,L --target A
```

For a finished protein or peptide run, save the following as `prodigy_partners.json` in its
run directory, then run `human --screen <run-directory>`. The screen uses those exact chain
groups for each sample. Other chains stay outside both the contact and surface calculations.

```json
{"binder_chains":["H","L"],"target_chains":["A"],"chain_namespace":"author","temperature_c":25}
```

Author chain identifiers are the default. Use `chain_namespace` set to `label` for mmCIF label
identifiers. Supply a one-based `model_index` for a multi-model file and an `altloc` identifier
when a residue has multiple conformers. The standalone options are `--chain-namespace`,
`--model-index`, `--altloc` and `--temperature-c`. PDB and mmCIF inputs may use gzip or Zstandard.

`screen.json` records each estimate under `prodigy_affinity.samples`, with its method, partner
groups, structure digest, software versions, model ordinal and residue counts. The model estimates
binding free energy in kcal/mol. It derives a dissociation constant in mol/L at the specified
temperature. This empirical result has no pass threshold. The report counts every attempted
design and names the blocker for each untested one.
The local MCP `read_verdict` tool exposes these samples as a list with an `example_id` on each
row. Its residue counts use a list of chain names and counts.

The calculation uses standard protein heavy atoms. It excludes waters and nonpolymer components
and preserves insertion-coded residues and multi-character chain IDs through a temporary internal
renumbering. A missing heavy atom, a zero-occupancy atom, an unsupported polymer residue or a
pair without contacts yields an untested result. The source structure is read-only.
For Zstandard inputs the reader verifies every frame, including the final one. An interrupted
file yields an untested sample even when the first frame contains readable coordinates.
Generated coordinates may omit crystallographic B factors, which the predictor never reads.
The residue check covers the atoms of residues present in the coordinate file. An unresolved
segment can still affect the estimate. On the deposited durvalumab fragment (5XJ4) the model
returned 142.8 nM at 25 °C. The authors measured 0.667 nM by surface plasmon resonance.
[Durvalumab structure and assay](https://pmc.ncbi.nlm.nih.gov/articles/PMC5777972/)
[PRODIGY method](https://doi.org/10.7554/eLife.07454) · [Upstream software](https://github.com/haddocking/prodigy)

## Therapeutic Nanobody Profiler

TNP profiles an explicitly nominated therapeutic VHH domain after generation. Give each
sample its mmCIF label chain and the SHA-256 of its preselected, complete domain sequence.
Compute the digest over uppercase one-letter ASCII residues without spaces, a FASTA
header or a trailing newline. Determine it from the intended sequence independently of
the structure being checked. IMGT heavy-chain numbering alone cannot establish that a
domain is a VHH. The caller makes that declaration.

Place `tnp_domains.json` beside the finished protein run's `inference_samples/` directory.
Replace the placeholder with the 64 hexadecimal digits of the intended domain's digest.

```json
{
  "samples": {
    "binder_0": {
      "domain_type": "VHH",
      "label_asym_id": "H",
      "sequence_sha256": "<SHA-256 of the intended full VHH sequence>"
    }
  }
}
```

Run `human --screen RUN_DIRECTORY --binder-kind protein`. For a single finished mmCIF,
use `python scripts/screen_extras.py tnp DESIGN.cif --chain H --sequence-sha256 DIGEST`.
Both entry points accept gzip and Zstandard files. The adapter checks polymer declarations,
observed residue identities and domain coverage. Coordinates can be incomplete when the
full polymer sequence is independently declared and verified. Fusions, partial numbering
and ambiguous polymer ordinals receive a named blocker. The output records the full
chain length and the number of residues with at least one finite atom coordinate triple.
Atoms with an explicit zero occupancy contribute no observed residue.

Install the project's `tnp` extra for the mmCIF reader. The native engine runs in a
separate Linux CPU environment. The reviewed package combination can be installed with
`micromamba create -n tnp-cpu -c conda-forge -c bioconda python=3.12 tnp=0.1.0 immunebuilder=1.2 dssp=4.6.1 'pytorch=*=cpu*'`.
Set `TNP_PYTHON` to that environment's absolute `bin/python`. Before screening, install
the four released `nanobody_model_1` through `nanobody_model_4` files in the provider's
`ImmuneBuilder/trained_model/` directory. The worker verifies their published checksums
before invoking the model. It also verifies IMGT numbering and the native DSSP annotation.
[TNP release](https://github.com/oxpig/TNP/releases/tag/v0.1.0) · [NanoBodyBuilder2 checkpoints](https://zenodo.org/records/7258553)

The `tnp_profile` block in `screen.json` reports total CDR length, CDR3 length, CDR3
compactness, and three CDR-vicinity surface patch scores for hydrophobicity, positive
charge and negative charge. The inputs to the structural calculations are TNP's own
NanoBodyBuilder2 coordinates. The native CLI exposes no seed. Fresh CPU models of the
same reference domain gave different patch scores, so use the recorded model SHA-256
when comparing runs. Each measurement carries its native flag. Missing
compactness with a native grey flag yields a partial profile with five other readings.
The standalone command exits 0 for a full profile and 2 for a partial or untested sample.
The finished-run report retains every attempted sample and the IDs of manifest entries
that had no matching design. These descriptors have no vote in the screening verdict.

The flag ranges were calibrated on 36 clinical-stage VHHs. Those 36 also contributed to
the 108-domain experimental panel. One of the 72 proprietary candidates failed to
express. The assays used Fc fusions. On the
archived caplacizumab VHH1 structure, TNP 0.1.0 reproduced both published loop lengths.
The published positive-charge score was 0.5694 (amber under the released thresholds).
Recalculation on the archived coordinates gave 0.0000 (green). Keep the native release,
model hash and reference table with any interpretation of the flags.
[TNP study](https://doi.org/10.1038/s42003-026-09594-y) · [Authors' reference data](https://github.com/oxpig/TNP/tree/ea619d119b9e9da00b0d40961769d7166ce7677f/paper/paper_data)

## How a verdict is reached

A design is screened against a **bar** — a named set of axes that must *all* be clear. For a
protein binder that is `immunogenicity`, `proteome_offtarget`, `mhc_i`, `mhc_ii`, `ada` and
`paralog`. For a ligand binder the default bar is coverage: `herg`, `pgp`, and `bcrp`.
One axis reading `hit` makes the call `axis_hit` regardless of the bar; every axis
clear makes it `screened_clear`; anything else stays `untested`.

New screens record `screen_version: 19`. The optional `method_selection` block records the
preset, selected methods, attempted methods, reasons for skips and method-specific results.
Version 11 added optional SyGMa metabolite predictions.
Version 10 added QupKake microscopic pKa.
Version 9 added compound-level aqueous solubility.
Version 8 added the optional TNP domain profile.
Version 7 added the optional Vina fixed-pose score. Version 6 added the per-design PRODIGY affinity report.
Version 5 distinguishes incomplete refold results,
unapplied interface reference thresholds and assay-specific ligand readings. Version 4
added the interface and ligand report blocks. Version 3 added `refold` and its criterion bounds.
Version 2 is
the shape that can account for its own verdict: it adds `cuts`, `bar_axes`, `metrics_file`,
`metrics_blocker`, `chain_blocker`, `inputs`, `code`, `replaced` and `nucleic_liability` at
the top level, and within the record a sample carries `row_blocker` and a named
`interface_blocker` and `sequence_blocker`, an axis carries `curated` when a curated state
decided it, and a table-backed axis carries `covered_examples`. A screen written at 1 has
none of these and nothing in it says which of the two it is, which is the whole reason the
number exists.

Raising it is not optional bookkeeping, and it is not left to memory: `screen.py` declares
`SCREEN_REQUIRED_KEYS` and `SCREEN_OPTIONAL_KEYS`, and
`tests/test_disease_screen_shape.py` pins both sets as literals beside the version number.
Adding a key fails that test, and the fix puts `SCREEN_VERSION` in front of whoever is
editing it. Required keys are present in every screen whatever the kind; optional keys are
absent rather than null, so a reader can tell "this run has no ligand panel" from "this run
has an empty one".

Every screen also records `cuts`, every
number the verdict was taken against: `gravy_max`, `instability_max`,
`clash_distance_angstrom`, the four MHC rank and IC50 cuts, `ada_median_rank_max`,
`ligand_panel_nm_max`, and the three `--plasma-*-max` values, which are `null`
when you named none. A verdict is only reproducible against the cuts that
produced it — edit one of those constants later and every stored screen silently
changes meaning. The recorded values are the constants themselves, not copies, so
what is written cannot drift from what was applied.

`metrics_file` records the selected table relative to the run. The reader tries
the generation primary, its standalone re-evaluation, then the offline evaluator's
primary. Pending generation jobs also block the offline fallback.

`metrics_blocker` distinguishes a missing table (`metrics_not_configured`) and
an empty one (`metrics_not_reported`) from a read error (`metrics_unreadable`).
The reader checks the 16 MiB limit, repeated columns used by the screen or trace,
CSV quoting and row widths. The corresponding blockers are `metrics_too_large`,
`metrics_columns_repeated`, `metrics_row_too_long` and `metrics_row_too_short`.
A refused table leaves its flags absent.

`metrics_publication_pending` identifies an unfinished publication.
`metrics_aggregation_required` identifies job files whose inclusion is unverified.
`metrics_provenance_mismatch` identifies a primary table whose digest differs from
the recorded one. `metrics_provenance_unreadable` identifies a receipt or source
that could not be verified. [Recovery and export behavior](../how-to/advanced/offline-metrics.md#retrying-and-exporting-reports)

The screen matches structures by their recorded path, relocated run-relative path,
offline evaluator ID or historical bare name. `metrics_row_absent` means no row
matched a discovered design. `metrics_row_ambiguous` identifies conflicting rows
for one design. `metrics_row_shared` means the same evidence matches multiple
files. Their per-design traces withhold the disputed values.

Run coverage counts each discovered design once, includes unmeasured designs and
retains table-only observations. Identical rows for one structure count once.
A known failure remains visible. A whole-run pass requires complete applicable
coverage. Offline tables can also name standalone inputs outside the generated-CIF
layout. When the screen reads only their stored measurements, its structure blocks
report `evaluation_structure_table_only`.

`chain_blocker` is `chain_disagreement` when the table names more than one
`chain_of_interest`. A design's own row still wins; the fallback across the
table stands in only while the table is of one mind, so a design whose own row
is silent is left unmeasured rather than measured against a neighbour's chain,
and its `sequence_blocker` and `interface_blocker` say `chain_disagreement`.
`designed_chain` returns nothing for such a run, so nothing re-folding a design
is handed a contested chain either.

`bar_axes` is the list of axes that bar required, read from `BARS` at the moment of the
verdict. The bar's *name* was stored from the beginning and that is not enough: add an axis
to `BARS["protein"]["sequence"]` and every stored `screened_clear` silently starts claiming
more than it was ever checked for, while removing one makes it claim less. This is the same
hazard `cuts` exists for, and the same answer — the list written is the list applied. A
panel target's bar is `none` and a binder of kind `other` has no bars, so both record
`absent`.

The bar can be chosen. `BARS` declares thirteen named strategies and `judge_combination`
has always taken one, but nothing passed it, so nine of the thirteen could not be selected
at all. `--bar NAME` on `--screen` picks one, and `screen_generation(bar=...)` is the same
choice in code:

| Kind | Default | Also available |
|---|---|---|
| protein | `sequence` | `plasma`, `all` |
| peptide | `sequence` | `plasma`, `all` |
| ligand | `coverage` | `strict`, `plasma`, `all` |
| nucleic | `oligo` | `all` |

A bar that kind does not have is refused by name, with the ones that exist listed, and it
is refused *before* anything is measured — a misspelling should not cost a proteome scan
first. Bars are not ordered from weak to strong: `plasma` is three axes and says nothing
about immunogenicity. That is exactly why `bar_axes` is recorded beside the name.

`inputs` lists every file supplied beside the run — each predictor table, the toxicology
panel, the co-evolution notes, the solubility scores, `axis_evidence.json` — with its size
and a short digest, whether or not an axis read it. A table that was present and refused is
as much a part of the record as one that decided an axis. Two inputs already carried a
digest, the proteome and the off-target reference, and the reason given there holds for all
of them: a swapped or rebuilt table is otherwise invisible between two runs. Only the run
folder's own files; the designs and the metrics table are named elsewhere in the record. A
file past 256 MiB is recorded by name and size with `digest: "file_too_large"`.

A supplied file that is a symlink is listed like any other, with its digest, and carries
`"link": true`. It was skipped, while every reader here follows a link, so a linked table or
a linked `axis_evidence.json` decided the verdict and the record denied it existed — and the
next `replaced` then said the verdict had moved with nothing on disk moving. The link's
target is not written down: it is a path outside the run, and the digest already says
whether the bytes moved.

`replaced` now also carries `changed_inputs` when a re-screen reaches a different verdict
*and* the supplied files differ, naming each file as `added`, `removed` or `edited`. The
replacement record said only that the verdict had changed. Which input moved is the part a
reader can act on: a verdict that moved because a predictor table was swapped is a
different event from one that moved with nothing on disk touched.

A run folder holding more than 64 files is past what one run supplies, and the list is then
withheld whole — `inputs` is `[{"file": "inputs_truncated", "files": N}]` — rather than cut
at the limit. A truncated list reads as a complete one, so every file past the cut would
look absent and the next re-screen would report them all as removed, a change that never
happened. `changed_inputs` says nothing at all when either screen withheld its list, because
the only alternative to saying nothing there is saying everything moved.

`code` is the commit this checkout sat on, read out of `.git` without running git so a
screen starts no subprocess. A verdict is reproducible against four things — its inputs,
the `cuts` it was taken against, the `bar_axes` it had to clear, and the code that applied
them — and the fourth was the one the screen did not keep. The gap is plainest in the line
above: a re-screen that changes the verdict while naming no changed input is pointing at
the code, and until now the record could not say which code it was pointing at. `tree_state`
is always `not_checked` and says so, because telling a clean tree from a dirty one means
asking git, and a commit recorded as though it described the working tree would be worse
than one that admits it is only the base that tree sat on. A checkout with no `.git` — an
installed copy, a container image — records `not_recorded` rather than inventing a commit.
It is the same helper the research traces use, shared rather than written twice: two copies
are how the two records come to disagree about one checkout.

`replaced` appears when a re-screen reaches a different verdict from the one it
overwrote, and carries that verdict's `call` and a digest of it. `--screen`
rewrites `screen.json` outright, so a table dropped in afterwards could turn an
`axis_hit` into a `screened_clear` with nothing on disk remembering the hit. It
is written only when the verdict actually changed, which is also what keeps
screening the same run twice byte-identical.

`job_mismatch` names what the job file recorded when it disagrees with the kind or
the target this screen was run against — the two then describe different runs. The
caller's values stay authoritative, because they are what the measurement used, so
the record only says what the file said. It is now on the printed line as `Job file
disagrees: ...` and on the closing page, because a verdict about a run the job file
calls something else is not a bare verdict.

A run with no design to measure says so as well: `sequence_blocker` and
`interface_blocker` are `designs_absent` when the run wrote none. The printed
line says `No design measured: <reason>.`

`*_generated.cif`, `*_generated.cif.gz`, `*_generated.cif.gzip` and
`*_generated.cif.zst` are all read,
so a run written with `compress_outputs=true` is screened like any other. Where
both spellings of one design sit side by side — what a decompress-in-place
leaves behind — the plain file is the one measured, and the design is counted
once. `.cif.gz` goes through the standard library. `.cif.zst` needs
`zstandard`, which arrives with the model stack rather than with this package;
where it is absent, such a design is reported as `designs_compressed` instead of
being counted and measuring nothing. A binary `*_generated.bcif` is always
`designs_compressed`: it is a different format rather than a compressed one, and
this screen parses CIF text.

Everything downstream of the screen reads them the same way: the follow-up plan
writes an inverse-folding spec from a compressed design and points the command
at the compressed file, which the model reads, and the structure viewer
decompresses before serving the CIF to the browser. A condition spec's own
`input` may be compressed too — `inputs=…/monomer-short.cif.zst` is the
documented form — and the target's accession is resolved from it either way.

The designed sequence is read from `_entity_poly` in **both** spellings mmCIF
allows. A category with several rows is a `loop_`; the same category with one row
is written as plain `_name.item value` lines, and the model's own writer emits
both — so a structure with exactly one polymer entity (a monomer, an
inverse-folding output, an enzyme design, a motif scaffold, a small-molecule
binder) carries no `entity_poly` loop at all. Reading loops alone found no
sequence in any of them and reported `sequence_absent` for a sequence the file
was holding, which left every sequence-liability axis untested rather than
measured. A multi-line value in a semicolon-delimited block is one sequence, not
one per line. Where a file somehow carries both spellings, the loop is the one
read.

The target's accession is read from a condition spec's deposit the same way, and
from the same two spellings. It joins `_struct_ref` to the fixed chain through
`_struct_asym` (the label chain), `_struct_ref_seq` (the author strand) and
`_entity_poly`, and a deposit with a single UniProt reference writes all four
flat. Reading loops alone found no accession in such a file, so the fixed chains
then "disagreed" and the target resolved to `target_accession_ambiguous` — for a
deposit naming its protein in plain sight. Reading both spellings does not make
the reader decide more than it did: a deposit whose fixed chains genuinely name
different proteins is still refused, which is the common case for an antibody or
receptor complex and is why that blocker is still the usual answer there.

A design whose numbers are missing says which miss it was rather than leaving
the same silence for every cause. `sequence_blocker` is one of `chain_not_named`,
`structure_unreadable`, `structure_unparsed`, `sequence_absent`, `metrics_failed`,
`not_a_polypeptide`, `designs_absent` or `designs_compressed`. `not_a_polypeptide` is a
nucleic or small-molecule design: GRAVY and the instability index are defined over amino
acids, so they are not computed and `sequence_pass` reads `absent` rather than true or
false. The instability rule is ExPASy's, which calls a protein stable below 40, so a value
of exactly 40 does not pass. The GRAVY bar of 0.1 beside it is this program's own triage
convention and not a published cut; every stored verdict records both cuts and names which
is which under `gravy_max_source` and `instability_max_source`. `structure_unparsed` is a loop whose values
do not fill its rows — most often a trailing comment or an unclosed quote on a data
line — which is a file to fix and not a chain that held no sequence.
`interface_blocker` is one of `chain_not_named`, `structure_unreadable`,
`coordinates_absent` (a sequence-only file, which is the ordinary result of an
inverse-folding run), `coordinates_unusable`, `binder_chain_absent` or
`interface_failed`. Both sit on the design's own sample and on the top-level
block beside the nulls they explain, and the closed page prints each one with
what to do about it.

One axis cannot clear from a measurement at all, and two more clear only against a cut you
name:

- **`paralog`** reports `hit` when another human reviewed protein shares the target's
  UniRef50 cluster — positive evidence of a family. The absence of one is **not** evidence
  of absence: a UniRef50 cluster is built around a representative and does not gather a
  protein's near-identical relatives, so KRAS and EGFR both come back singletons while
  having exactly the families a binder would cross-react with. Measured against the live
  services, that understates it — KRAS reports `paralogs=[]` while Compara puts HRAS at
  85.6% identity to it and NRAS at 84.1%.

  So a singleton is handed to **Ensembl Compara**, asked over REST for that one gene
  rather than downloaded as 11.23 GB, which answers with per-pair identity. The closest
  paralog's identity becomes the measurement, judged against `--paralog-identity-max`.
  **The lookup happens only once that cut is named**, and until then the axis answers
  exactly as it did before, having touched no network: without a cut it cannot judge, and
  two requests per target that buy an unjudgeable answer are the difference between a
  screen and a rate limit on a real campaign. No published cross-reactivity identity
  threshold exists, so none is assumed. A UniRef50 hit keeps
  precedence and nothing here can downgrade it; if Compara cannot answer, the old
  singleton blocker is kept and the trouble recorded beside it as `compara_blocker`.

  Two things it deliberately does not do. It does not judge on a **count**: Compara holds
  every paralog in its gene trees at any identity — 53 for ERBB2, median identity 17.4% —
  so a count against a cut would call ERBB2 a liability on the strength of fifty proteins
  that share no surface with it. And it reads identity in **both directions**: Compara's
  `perc_id` is identity over the length of its own sequence, so ERBB2 and EGFR are 50.7%
  identical over EGFR and 48.9% over ERBB2. Reading only one direction made the verdict
  depend on which protein the run named, and against the live service it suppressed EGFR's
  hit outright — 48.9% directional became 51.9% once the pair was read whole.

  A protein or peptide screen also reads `paralog_coevolution.tsv` from the output
  directory and stores that note on this record. It does not change the state.
- **`plasma_abundance`** and **`plasma_hub`** measure real numbers, but no published
  threshold turns either into a liability. They report and stay `untested` until
  `--plasma-abundance-max` or `--plasma-hub-max` names a cut, after which they `hit` or
  `clear` against it like any other axis.

Chemical liability is a report on the designed chain of a protein or peptide binder.
It sits beside sequence developability. It is not an axis. A motif leaves the call
as the named bar decided it. Ligand and nucleic screens leave the scan unread.

The same screen writes `modality_screen` for the binder kind. A protein or peptide
report lists furin cleavage motifs, Asp-Pro bonds, N-terminal glutamine,
nonstandard letters, the longest hydrophobic and same-residue runs, the
Ikai aliphatic index, the Shannon entropy of the residue composition, the
average molecular weight, the Pace 1995 extinction at 280 nm, the N-end rule
half-life, the residue and atomic composition, Kyte-Doolittle spans, and an
AGGRESCAN 2007 normalized sum. A nucleic
report lists GC fraction, CpG count, and the longest G run. A ligand report lists
a formula weight when the CIF names one non-polymer component. A missing weight
is calculated from the chemical formula when every element is known, and left
out when that formula does not parse. None of these is an axis,
and none of them changes the call.

## Class I and class II predictor tables

Produce these by running a standalone predictor **on this machine** — NetMHCpan /
NetMHCIIpan from DTU, or the IEDB standalone tools — over the designed chain. Nothing in this
program contacts a prediction service, and no designed sequence leaves the machine.

```{warning}
This cannot be enforced. If you paste output from a web server, your design sequence has
left the machine and the code has no way to tell.
```

| Column | Accepted names | Needed for |
|---|---|---|
| allele | `allele`, `mhc`, `hla` | `ada`; allele comparison on the reference |
| peptide | `peptide`, `sequence` | everything except the legacy rank-only shape |
| rank | `consensus_percentile_rank`, `consensus_percentile`, `percentile_rank`, `%rank`, `rank_el`, `el_rank`, `%rank_el`, `rank_ba`, `ba_rank`, `rank` | a binding call |
| affinity | `ic50_nm`, `ic50`, `affinity_nm`, `ba_affinity`, `aff_nm` | a binding call, on its own or alongside a rank — either number at or under its cutoff makes the row binding |
| unit | `unit`, `units`, `uom`, `standard_unit`, `standard_units` | reading `ic50` or `ba_affinity`, which name no unit of their own |
| method | `method`, `predictor` | `ada`, which needs the consensus scale |
| position | `start`, `begin`, `first`, `pos` / `end`, `stop`, `last` | a reference table's protein length |

Allele spelling is normalised, so `HLA-DRB1*03:01`, `DRB1_0301` and `drb1*03:01` are one
allele.

A row counts as binding at **percentile rank ≤ 1** or **IC50 ≤ 500 nM** for class I, and
**rank ≤ 10** or **IC50 ≤ 1000 nM** for class II.

The affinity cutoffs are **nanomolar**, and a micromolar figure is never rescaled onto them.
`ic50_nm`, `affinity_nm` and `aff_nm` state the unit in their own name and need no unit cell.
`ic50` and `ba_affinity` name only the analyte, so they are read only where the row's unit
cell says nM (`nm`, `nanomolar`, `nmol/L`), and are otherwise refused with `ic50_not_nm` —
read as nanomolar, a 2 µM non-binder under a bare `ic50` header was published as a `hit` at
2.0, and the same cell in a reference table counts as a 2 nM binder and inflates the bar. A
unit cell naming some other unit overrides even a column name that says nM, because the two
then contradict each other. A column left unread is named in `not_nm_columns` on the axis,
and it refuses a `clear` without touching a `hit`: a binder measured on the numbers that
could be read is not deleted by a sibling column nobody can read.

```{note}
A percentile rank written as a **fraction of one** rather than as a percentage cannot be
told from a very small percentage — `0.005` means 0.5 % in one export and is read as
0.005 % here — and nothing published distinguishes the two, so no floor is invented and no
cell is rescaled. A rank above 1 cannot be a fraction, so a table whose every rank is at or
below 1 is the ambiguous case. On a reference table that is recorded as
`reference_rank_scale_ambiguous` beside the verdict, because the percentage reading calls
more rows binding and so sets a higher bar, and a bar that is too high clears designs. On a
design table the same reading calls more rows binding and so errs toward a `hit`, which this
screen reports rather than refuses. It is a limit of every rank column read here, the `ada`
panel's included.
```

Scan the **whole chain**: every window, not a filtered or truncated export. Finding binders
is positive evidence whatever else was skipped, so a density above zero is taken as it
stands. But "no binding peptide anywhere", and a cleared `immunogenicity`, are claims about
every window, and a file that did not look at every window does not support them — a scan
covering 40 of a chain's 84 windows reported zero binding peptides per 100 windows as a
measurement and cleared, with the binders sitting at windows 61 to 64, outside the file.
Those two cases are refused with `scan_incomplete`.

Coverage is counted against the chain's **distinct** windows, so a low-complexity design is
not penalised for repeating one, and it must hold at **every** peptide width the table
scores. One width is not evidence for another: a file of 85 8-mers plus 4 9-mers covers its
8-mers completely, covers 4 of the chain's 84 9-mers, and must not buy a measured zero with
the cheap half. Widths outside 8–11 and 13–25 are ignored, since no binding cutoff is defined
for them — IEDB calibrates class I at 8–11 and class II at 13–25, so **12** falls between the
two. A table holding nothing but widths like that supports no binding call at all, and the
density axis refuses it with `width_without_cutoff` rather than as an incomplete scan.

A cell reading `-`, `n/a` or `n.d.` says the predictor declined to score that window. Such a
window is not a non-binder and does not count as scanned; `unscored_peptides` on the axis
records how many there were. A cell that is neither a number nor one of those spellings is
refused with `cell_unparsed`, because silently skipping it always loses a binder and never
invents one.

Whenever a reference table is present, the design and the reference must agree about the
questions they asked: the same alleles, the same peptide widths, and the same tool. A design
scored on one allele against a reference scored on seven is a count over one set compared
with a count over seven, and the same is true of peptide width — the density counts binders
of every width and divides by one length, so four widths against one inflates the bar about
fourfold. See `reference_allele_unknown`, `reference_allele_mismatch` and
`reference_width_mismatch`.

Tool identity is the one of the four taken on trust, because neither documented predictor
writes a `method` column by default. One table declaring a method while the other does not is
refused (`reference_method_unknown`) — that asymmetry is what hid a real mismatch. Neither
declaring is measured, with `reference_method_undeclared` recorded on the axis, so the verdict
says tool identity was assumed rather than checked.

### ada: class II allele breadth

`ada` applies the scheme of Paul et al., *J Immunol Methods* 422:28-34 (2015), blind-validated
on fifteen antigens. A peptide that binds across the DR panel is the accepted computational
anti-drug-antibody signal, because a promiscuous binder finds a restricting allele in most
people.

The panel is the method, not a setting — the three DRB1 alleles at ≥ 12 % population
frequency plus four DRB3/4/5, with DP deliberately excluded:

```
DRB1*03:01  DRB1*07:01  DRB1*15:01
DRB3*01:01  DRB3*02:02  DRB4*01:01  DRB5*01:01
```

The table must cover **all seven** for at least one peptide, and must declare a consensus
method, because the ≤ 20 threshold belongs to the IEDB consensus rank scale. A NetMHCIIpan
eluted-ligand rank is calibrated differently — its published bands are 1 % strong and 5 %
weak — so reading a median of 20 as permissive there would invert the verdict.

A peptide whose **median rank across the seven is ≤ 20** is promiscuous, and makes the axis
a `hit`. Every peptide above it clears.

```{important}
This axis reads the **rank** column only — the threshold is a percentile, so an affinity
cannot stand in for it. A class II table carrying affinities and no ranks reports
`rank_absent`, even with all seven alleles covered.
```

### immunogenicity is restricted by the class I table

The Calis et al. 2013 class I 9-mer score was fit on predicted MHC ligands, not on every
window of a protein. Scanning a whole chain asks the model a question it was not fitted to
answer: a real binder had 359 windows above zero and could never clear.

So when `mhc_i.tsv` names peptides, only the 9-mers it calls presented are scored. One above
zero is a `hit`; none above zero is a `clear`. Without such a table the whole-chain maximum
is still reported, as information, and the axis records `presentation: not_configured`.

The paper masks the **anchor positions**, and it makes them per HLA molecule: the excluded
set is "P1, P2 and P9 for most HLA molecules, but e.g. P2, P5 and P9 for HLA-B\*0801". Its
published position importances cover the default masking only — P1, P2 and P9 read
"NA (anchor)" there — so no weight exists for scoring P1 under any other mask, and none is
invented here. The default mask is what is applied, and the record says so: `calis_mask`
lists the masked positions, `calis_mask_allele_specific` is `false`, and
`restricting_alleles` names what presented the ligands. A ligand restricted by HLA-B\*08:01,
the one exception the paper names, cannot `clear` this axis — the blocker is
`calis_mask_unpublished_for_allele` — while a positive ligand is still a `hit`, because a
positive score found under the default mask is still positive evidence.

### Which designs a supplied table covers

A run holds many designs and a supplied predictor table is one scan. `mhc_i`, `mhc_ii`,
`ada` and a restricted `immunogenicity` all check that the table belongs to this run by
finding the design its peptides are windows of — the *best-matching* design, which is the
right test for a table inherited from an earlier design in the same folder. The verdict
was then recorded for the whole run, with nothing saying whose chain it had been taken
from, so a run of eight designs could read `mhc_ii: clear` on a scan of one and seven
designs no predictor had ever seen were covered by that word.

Those axes now record `covered_examples`, the designs the table is a scan of by the same
share that decides provenance, and `uncovered_examples` when any design is not among them.
The closed page prints the uncovered designs under the axis. A table that was refused
outright records no coverage, because there is no claim to scope.

**This records the scope of the claim; it does not narrow the claim.** The state is still
the one the table gave, for the whole run. Whether an `mhc_ii` clear taken from one design
should clear a run containing designs that were never scored is a decision about what the
verdict means, and it belongs to whoever runs the screen: tightening it would make the
protein `sequence` bar unreachable for any multi-design run, which is the normal shape of
a run here. The evidence is on the record so that call can be made deliberately — and
`axis_evidence.json` with a stated `reason` is how to make it.

## Where the tables come from

Each table is a file you supply. `rfproteina.disease.tables` is the registry of
where to get them, and `scripts/fetch_screening_tables.py` downloads them:

```bash
python scripts/fetch_screening_tables.py --list        # every source, query and refusal
python scripts/fetch_screening_tables.py               # fetch the human tables
python scripts/fetch_screening_tables.py --have        # which downloads are on disk
python scripts/fetch_screening_tables.py --stage disease_runs/my_run --disease "Breast cancer"
python scripts/fetch_screening_tables.py --coverage disease_runs/my_run
```

**A download is not a screen table.** Three directories are in play and they
are not interchangeable: downloads keep the names their servers give them, a
run reads tables out of its own output directory under the stems above, and
`--stage` is the step between. `--have` reports the first, `--coverage` the
second.

| Source | On disk as | Reaches the screen as | Licence |
|---|---|---|---|
| Human Protein Atlas disease blood | `blood_pea_disease_de.tsv` | `plasma_fold.tsv`, by conversion | CC BY-SA 3.0 |
| UniProt human reference proteome | `UP000005640_9606.fasta.gz` | `--proteome`, and the gene→accession map `--stage` needs | CC BY 4.0 |
| IEDB MHC ligand assays | `mhc_ligand_full.tsv` | **nothing** | CC BY 4.0 |
| IEDB T cell assays | `tcell_full_v3.tsv` | **nothing** | CC BY 4.0 |
| IDC anti-drug antibody incidence | `aggregated_ada_values.csv` | **nothing** | CC BY 4.0 |

### Why three of them reach nothing

`mhc_i`, `mhc_ii`, `immunogenicity` and `ada` read a scan **of this design's
own chain** — `_table_belongs` requires the table's peptides to be windows of
the design it is matched to. The IEDB exports are censuses of measured epitopes
across every organism IEDB holds, and the IDC export is clinical anti-drug
antibody rates for drugs that exist. They answer different questions, and no
rename or allele split converts one into the other; a predictor run over the
designed sequence does, and none is wired up here.

Nor do their columns match. Both IEDB exports use a two-row header of
group-and-subname pairs — `Epitope`/`Name` for the peptide, a name under `MHC
Restriction` for the allele, `Assay`/`Quantitative measurement` for the number
— and of the 112 and 161 columns, none is `peptide`, `allele` or
`consensus_percentile_rank`. (An earlier version of this page said the exports
"drop in without conversion". They do not, and the stems are not destinations.)

Renaming one to its stem is worse than leaving it where it is:

| Renamed to | What happens |
|---|---|
| `ada.csv` | None of its 57 columns is a score the reader takes, so the axis reports `score_absent` — **and a supplied table takes precedence, so the DR-breadth measurement `mhc_ii.tsv` would have produced is displaced and never run.** An axis that could have been a hit reads untested. |
| `immunogenicity.tsv` | Same shape: `predictor_unreadable`, displacing the Calis computation `mhc_i` drives. |
| `mhc_i.tsv` | `predictor_unreadable`, and at 10.26 GB this reader — unlike the fold reader beside it — bounds no file size and catches no `MemoryError` before reading the whole file into memory. |

`--list` prints each of these as a `HAZARD` line beside the source.

### Staging a run

`--stage` converts what can be converted and names everything that cannot:

```bash
python scripts/fetch_screening_tables.py --stage disease_runs/my_run \
    --disease "Breast cancer" --gene ERBB2 --gene TP53 --release v26.0
```

`--disease` is required and not defaulted: a fold table holds one disease
because the axis asks how far the target moves *in that disease*, so a guess
answers a question the run did not ask. `--gene` narrows it; without narrowing
the whole-proteome conversion exceeds the reader's 1 MB file bound and is
written but refused.

`--gene` takes a gene symbol or a UniProt accession in any mixture, because the
accession is what a caller downstream of generation actually holds.

HPA reports several rows per gene per disease — one per assay binder — and the
conversion keeps all of them rather than collapsing them. The reader then
judges the **largest sourced magnitude** among a target's rows, which is the
conservative reading: it cannot suppress a hit that one binder saw and another
missed. The per-gene row cap still applies, so a pathological number of rows
for one protein is reported rather than silently truncated.

If a narrowed conversion comes back empty, the blocker says which filter
emptied it — `disease_not_in_table` names the diseases the file does hold,
`targets_not_in_table` names genes the disease does report. These were one
blocker until it was found reporting `disease_not_in_table` for a disease that
was present, because the gene filter was the one that matched nothing.

Staging also writes an `accession` column beside `gene`, read from the UniProt
proteome FASTA. This matters more than it sounds: HPA keys its rows by gene
symbol, the screen resolves a target to a UniProt accession whenever a
condition spec names one, and `_names_target` compares the two literally — by
design, so no row belonging to another protein is ever read. Without the
mapping, a run asking for `P04626` gets `fold_target_not_in_table` for a
protein that is in the table as `ERBB2`. `--stage` reports that as a shortfall
and does not call the run equipped.

### Staging as part of a generation run

`--stage-tables` on the generation command does the same thing per run
directory, just before that run is screened, narrowed to that run's own
resolved target and the disease being worked on:

```bash
python -m rfproteina.disease.cli --disease "Breast cancer" --stage-tables --plasma-fold-max 2.0
```

It is opt-in: a run that did not ask for it gets no files written into its
output directory. It never fails a run either — a missing download, a disease
HPA does not report, or an absent proteome each print what happened and the
screen proceeds, because every one of those is a thinner screen rather than a
broken one. `--plasma-fold-max` still has to be set for the axis to reach a
verdict; without a threshold a staged table measures the fold and reports
`threshold_not_set`.

The same flag also points `proteome_offtarget` at the downloaded human
reference proteome, unless `--proteome` names one — an explicit `--proteome`
always wins. The FASTA is read where it lies rather than copied into the run,
so it is reported separately from the staged tables. Which FASTA a clear was
measured against is part of the measurement and the axis records only its
blocker, so the substitution is printed.

Two things had to be fixed for that axis to work from a download at all, and
both were silent:

* UniProt ships the proteome gzipped and the registry fetches
  `UP000005640_9606.fasta.gz` unchanged, but the reader called `read_text` on
  it — so the one shape this program downloads was the one shape it could not
  open, and the axis reported `proteome_unreadable`. It now reads `.gz` and
  plain alike, decided on the extension, and the guards that matter are
  unchanged: a gzipped coding FASTA is still caught as `proteome_not_protein`,
  and a proteome under a million distinct 9-mers is still `proteome_too_small`.
  Both of those would otherwise read as a clear against nothing.
* Nothing told the screen where the download was, so the axis reported
  `proteome_not_configured` on a machine holding the file.

**Only human subsets are fetched, and there is a 10 GB limit per download.**
Every one of these sits beside a whole-database file under a similar name:
UniProt's all-species archive is 355 GB, STRING's all-organisms links 134 GB,
the HLA Ligand Atlas raw spectra deposit 1.39 TB, and Ensembl's all-species
homologies 11.23 GB — which its own FTP listing rounds to "10G", so it reads as
though it fits. Those are refused by name, each pointing at the smaller file or
the live query that answers the same question.

These axes are **not** tables and are asked online on every run, whatever is on
disk. This is why the 134 GB and 355 GB files are never needed: the screen asks
about one target at a time.

| Axis | Endpoint | Narrowed to |
|---|---|---|
| `paralog` | `rest.uniprot.org` UniRef50 + UniProtKB | one accession, then `organism_id:9606`, `reviewed:true` |
| `paralog` (singleton + `--paralog-identity-max`) | `rest.ensembl.org/homology/symbol` Compara | one gene symbol, `type=paralogues`, `target_taxon=9606` |
| `plasma_abundance` | `proteinatlas.org/api/search_download.php` | one accession, concentration columns |
| `plasma_hub` | `string-db.org/api/json/interaction_partners` | one identifier, `species=9606`, confidence floor |

`--coverage` reports what a run has on disk and what it will still ask for. A
run with every table present is **not** offline: those three lookups remain, and
the report says so rather than letting a network error be the first news of it.

## ImmuScope study benchmarks

Run `python scripts/screen_extras.py immuscope-study` to read the published
benchmarks for all three ImmuScope models. The same JSON record is available
under `reference_studies.immuscope` in the MCP `declare_corpora` response.
Both entry points work offline and take no study inputs.

| Model | Evidence in the study | Reported performance |
|---|---|---|
| ImmuScope-EL | Eluted MHC-II ligands and decoys, evaluated across alleles | Precision–recall AUC 0.856, positive predictive value 0.796 |
| ImmuScope-CD4 | Identification of known CD4 epitopes within their source proteins for a specified MHC-II allele | Mean per-triplet ROC AUC 0.825 |
| ImmuScope-IM | Peptide-level immunogenicity on 17,897 held-out records, including sampled negatives | ROC AUC 0.909 (95% CI 0.901–0.918) |

Each value describes performance on its own study cohort. Measuring a
therapeutic's clinical anti-drug antibody incidence requires patient data.
The Figure 2 source workbook reproduces the EL metrics from 58 allele rows
and the CD4 mean from 842 triplets. Its caption gives 824 for the paired
statistical comparison. The Figure 5 workbook omits the pooled IM ROC panel.
Integrating the curve in the published figure gives 0.90936. The confidence
interval comes from the authors.
[Shen et al., Figures 2 and 5 and Methods](https://doi.org/10.1038/s42256-025-01073-z)

A local CPU test ran the released ten-checkpoint ensemble for each endpoint on
one fixed reference, native human insulin B9–23 (UniProt P01308) presented by
HLA-DQ8. The raw ensemble scores were 0.21929 for EL, 0.27916 for CD4, and
0.95919 for IM. Each model has its own scale. Human islet-derived CD4 T-cell
receptors from two donors with type 1 diabetes responded to this peptide in
a published experiment. The exact peptide–HLA pair occurs in three rows of
the released CD4 evaluation set and one row of the IM training set. The
high IM score is therefore an in-sample result. This control verifies model
execution on a known epitope. [Human assay, Figure 4](https://doi.org/10.2337/db16-1025)

## Reference tables

`mhc_i` and `mhc_ii` cannot clear against an absolute cutoff: those cutoffs were calibrated
to pick candidate epitopes out of a peptide library, and across every window of a protein
they are met by serum albumin, by IgG and by ubiquitin. A rule saying "clear = no binding
window anywhere" would be a claim no human protein satisfies.

Instead the design's **binding peptides per 100 windows** is compared with the same count
over proteins people tolerate at high systemic exposure. Score these once per campaign, with
the same tool, version and alleles as the design:

| Protein | Accession |
|---|---|
| Human serum albumin | `P02768` |
| IgG1 constant region | `P01857` |
| Ubiquitin | `P0CG47` |

A reference table adds a `protein` column (`protein`, `accession`, `entry`, `id`) and the
protein's **length**, either as a `length` / `protein_length` column or implicitly from a
complete scan — one that starts at the protein's first residues, so the furthest peptide end
is the length.

```{important}
A binders-only export is the common trap. Filtered to its binders, an albumin scan ends at
the last binder rather than residue 609, and the density computed from it measured **6.25**
where the complete scan of the same protein measured **0.49**. An inflated bar always errs
toward clearing the design, so a scan that cannot be shown to be complete is refused with
`reference_length_unknown`.
```

A length that *was* read and is shorter than every peptide width the reference was scored at
is a different fault with a different remedy: the protein holds no window of that width, so
it has no density, and it is refused with `reference_too_short_for_width`, which names the
proteins and their stated lengths. Where only some candidates drop that way, the ones that
did are listed in `reference_proteins_too_short` beside the bar the others set.

The design clears when its density is at or below the highest reference density.

Only the three proteins above may set the bar. A label has to *be* one of them, after an
isoform (`-1`) or version (`.2`) suffix is dropped — `P02768`, `P02768-1`,
`sp|P02768|ALBU_HUMAN`, `sp|P02768.2| ALBU_HUMAN (Homo sapiens)` and the bare entry name
`ALBU_HUMAN` all name albumin, while `P02768-FRAGMENT-1-76` and
`sp|Q9Y6K9|TNF_HUMAN,P02768` name something else and are refused with
`reference_not_tolerated_protein`. Rows are then merged by accession, so one protein under
two spellings is one bar candidate rather than two — written as two, a 76-residue "fragment"
at 96.05 per 100 windows sat beside the honest albumin row at 0.4967 and the higher one won. A
row for any other protein is dropped rather than counted, so a stray entry cannot lift the
bar. Rows are also checked against the length
they claim: a protein of length *L* holds *L − w + 1* windows of width *w*, and a peptide
starting at *s* ends at *s + w − 1*. An export that pooled several proteins under one
accession breaks both and is refused with `reference_rows_exceed_protein` — such a table
divided three molecules' binders by one molecule's residues and raised the measured bar
3.46×, which turned a hit into a clear.

## The proteome

`proteome_offtarget` counts exact distinct 9-mer matches between the designed chain and a
protein FASTA. Point it at one with `--proteome`, or set `HUMAN_PROTEOME`:

```bash
curl -o UP000005640_9606.fasta \
  'https://rest.uniprot.org/uniprotkb/stream?query=proteome:UP000005640&format=fasta'
```

Any 9-mer shared with the proteome is a `hit`. This is exact identity only — a 9-mer
differing by one residue is a real cross-reactivity risk and reads as clear here.

It must be a whole protein proteome. A `clear` here means "no shared 9-mer in this file",
so every 9-mer the file does not hold is an off-target the axis cannot see. The human
reference carries about 10.8 million distinct 9-mers; a file an order of magnitude below
that is refused with `proteome_too_small`, and a coding FASTA with `proteome_not_protein`.
Both are easy to reach by accident — a UniProtKB download with the **Sequence** column
unticked parses as one record of about a hundred 9-mers, and `cds.all.fa` sits beside
`pep.all.fa`. The axis records `proteome_kmers` so a clear always says what it was
measured against.

## Plasma fold change

`plasma_fold` is how far the target's circulating level moves between disease and health. A
target whose level swings with the disease moves the antigen sink with it, so the dose that
works in one patient does not in another.

No public per-target API reports this, so it comes from a table you supply:

```
fold	source
3.4	Sun 2023 UKB-PPP
```

A linear fold column is `fold` or `fold_change`. `log2_fold_change` is converted with
`2 ** abs(value)` before that comparison; the stored measurement stays the number in the cell.
A provenance column is one of `source`, `study`, `reference`, `doi`. **Both are required** — a
fold change without a provenance is a rumour. The verdict uses magnitude, so a target that falls
in disease counts the same as one that rises, against `--plasma-fold-max`.

A ratio is symmetric about 1, not 0: in a `fold` or `fold_change` column a fivefold fall is
written `0.2` and a fivefold rise `5`, and both are measured as five. In a `log2_fold_change`
column the symmetric point is 0, and `-2.32` and `2.32` are both about fivefold. Where a table
holds several rows, the row judged is the one with the largest magnitude on that scale, so a
downward row is not passed over for a smaller upward one. The stored measurement is always the
number in the cell. `fold_column` identifies its input scale. `linear_fold_magnitude` records
the magnitude used for comparison when it can be represented as a finite number.

## Toxicology panel

A ligand binder (`ligand`, `small_molecule`, or `non_polymer`) reads `toxicology.tsv`, or
`toxicology.csv` when no `.tsv` is present. Protein, peptide, and nucleic screens leave that
file unread, so a panel column there does not enter their evidence.

The cells are concentrations in nM. A reading at or below 10_000 nM (10 µM) is a hit on that
axis. A higher reading is clear. The cut is the panel rule already used by the combination
judge. An axis with no cell stays untested, and the coverage bar stays untested until `herg`,
`pgp`, and `bcrp` are all clear. A hit on any panel axis, including `nav1_5`, `bsep`, or a CYP
outside that bar, makes the call `axis_hit`.

Long form names the axis and the concentration:

```
axis	nm
herg	50
pgp	20000
bcrp	20000
```

`axis`, `protein`, and `panel` are the axis column. `nm`, `ic50_nm`, `value_nm`, `reading_nm`,
`concentration_nm`, and `aff_nm` are the concentration. Wide form uses one column per axis id,
optionally with an `_nm` suffix. A column whose own name says nM (`herg_nm`, `herg (nM)`,
`herg_ic50_nm`) needs no unit cell; a column named only for the panel protein (`herg`,
`bcrp`, `nav1.5`) is read only when a unit cell reads nM, and is refused with
`reading_not_nm` otherwise, because a micromolar export under that header was judged a
thousandfold too strictly. The axis ids are `herg`,
`nav1_5`, `cyp3a4`, `cyp2d6`, `cyp2c9`, `cyp1a2`, `cyp2c19`, `pgp`, `bcrp`, and `bsep`.
`Nav1.5` matches `nav1_5`, and `P-gp` matches `pgp`. Where a row carries more than one axis
column, every one is read, so a table naming the gene in `protein` and the panel id in `panel`
resolves on the `panel` cell. Within one method, source, endpoint and assay, the reader
uses the most potent determinate reading and records `n_readings`. Across groups it
retains each opinion and any disagreement. Exact-ligand exports carry the InChIKey. A
compound mismatch or unverified identity in an automatically written Octant table leaves
the panel untested. A blank cell, `n.a.`, `true`, or `false` is no reading.
`0` is a reading, and at or below 10_000 nM it is a hit. A banner of blank lines and lines
beginning `#`, `---`, `//`, or `;;` is skipped, and the file is UTF-8 with or without a
byte-order mark.

A `unit`, `units`, `uom`, `standard_unit`, or `standard_units` cell that names anything other
than nM (`nm`, `nanomolar`, `nmol/L`) leaves that axis untested. The number is not converted
from µM or from a percent. A blank unit cell leaves a column whose own name says nM as
nM; a column named only for the panel protein is refused with `reading_not_nm`.

Each axis is judged against its own cut, recorded on the axis as `threshold` with
`threshold_source` naming where that cut came from. Every axis currently carries 10_000 nM
and the source `program_convention_10uM`: 10 µM is this program's convention across the
panel and is not a published bar for any one of these ten proteins.

### Bounded readings

An assay writes `>10000` for a compound that showed no activity and `<100` for one that
saturated, either in the cell itself or in a `relation` / `standard_relation` column beside the
number. A bound is read, and it decides the axis when every value it admits falls on one side
of the cut:

| Cell | Verdict |
|---|---|
| `>10000`, `>=10001`, `>30000` | clear — above the cut whatever the figure is |
| `<100`, `<=10000` | hit — at or below the cut whatever the figure is |
| `>=10000` | `reading_not_decisive` — 10000 itself is a hit, above it is a clear |
| `>5000`, `<50000` | `reading_not_decisive` — the interval straddles the cut |

`measurement` stays the number the table wrote and `qualifier` records the bound, so a tab
never shows a figure the file does not contain. A cell's own bound wins over a relation column.
A relation this does not understand, such as `~`, makes a **bare** number not a reading — but it
does not delete a bound the cell carries. That distinction was once missing, and the asymmetry it
produced was the giveaway: `<100` beside a flatly contradicting `>` was a hit at 100 nM, while
`<100` beside a `~`, which claims nothing at all, was deleted outright. It cost readings in both
directions — a bound a hundredfold inside the cut lost its hit, and one well clear of it lost its
clear.

A negative cell, a non-finite cell, or a cell that is neither a number nor blank leaves that
axis untested. A comma decimal such as `10,5` is `cell_unparsed`, and `unparsed_columns`
names the column. **An axis that already measured a hit keeps it**, even when another cell for
that axis could not be read: a reading the table made is not deleted by one beside it, because
that deletion turns a panel hit into a `screened_clear` headline.

The file is refused in full, as `toxicology_unreadable`, when a column this reader uses is
repeated, when the header holds none of the columns it reads at all, when a row carries more
fields than the header has columns, or when a concentration sits on a row that names no protein.
The last two move one protein's number onto another protein's name. A row that names a protein
outside the panel is dropped rather than refused, and a repeated notes column is ignored.
`axis_evidence.json` wins over the table on that axis and keeps the table's own state as
`measured_state`.

This screen does not launch a predictor and does not contact a service. A clear coverage bar
is the supplied table against the 10_000 nM cut.

## Paralog co-evolution

A protein or peptide screen reads `paralog_coevolution.tsv`, or `paralog_coevolution.csv`
when no `.tsv` is present. The note is stored on the `paralog` record. It does not change
the paralog state, which stays whatever UniRef50 membership or the Compara identity
measured. A curated clear in `axis_evidence.json` is still needed where Compara cannot
answer, or where no `--paralog-identity-max` was named.

The columns are `same_shape` (`yes` or `no`), `co_evolution` (`high` or `low`;
`co-evolution` and `coevolution` are the same column), and `toxicity`
(`toxic` or `non-toxic`; `other_species_toxicity` is the same column).
`species`, `organism`, `taxon`, and `other_species` name the organism and are
optional. A blank cell, `n.a.`, or `inconclusive` is no reading. `true` and
`false` name the shape. A trailing period and a non-breaking space are ignored.
A number in the co-evolution column stays unclassified: it is not turned into
high or low. Two alias columns that name different classes leave the note unset,
and `unparsed_columns` names them. Two different organism names on one row are
omitted. A pasted sequence in an organism cell is omitted.

For a shared shape, the note identifies the supplied co-evolution level and the
reported outcome in the other species. The human outcome is recorded as
`human_toxicity_state: untested`. The historical `coevolution_signal` values
identify these four combinations for readers of earlier records.

Several agreeing rows store `coevolution_rows` and the species names. Rows that name
different notes stay untested with `coevolution_disagreement`, which records both notes
and no row count: those rows agreed on nothing. A row with no same shape leaves the note
unset.

The file is refused in full, as `coevolution_unreadable`, when a column this reader uses
is repeated, when the header holds none of the columns it reads at all, when a row
carries more fields than the header has columns, or when a quote is left open. The last
two are the ones that change an answer rather than withhold it: a surplus field reads
every later cell under its neighbour's column name, which moved a row stating high and
toxic onto `low` and flipped the note to the opposite one of the four; and an open quote
swallows the rows beneath it, so a table whose second row disagreed came back as a single
confident note. A repeated notes column is not a surplus field, and a trailing empty cell
is not one either.

More than 256 data rows is `coevolution_truncated`. A file over 1 MiB is
`coevolution_too_large`. Either bound withholds the whole table, so an earlier row cannot
stand in for a later one. A one-word organism label spelled only from the residue alphabet
is dropped as a pasted sequence when it runs to twelve characters or carries a capital
after its first letter; `salamander`, `killifish` and `Canis familiaris` are kept. This
note is the supplied table read with those four rules.

## Chemical liability

A protein or peptide screen reads the designed chain from the generated CIF already
in the run. The record is `chemical_liability`, next to `sequence_developability`.
The sequence is not stored, and neither is a file path. Ligand and nucleic screens
omit the key. `axis_evidence.json` does not gain a chemical axis.

The motifs are Asn-Gly and Asn-Ser (deamidation), Asp-Gly (isomerization), and the
N-glycosylation sequon Asn-X-Ser/Thr where X is not Pro. Methionine and tryptophan
are counted. Each site stores its residue number, counted from 1 on that chain, and
the motif name (`NG`, `NS`, `DG`, `NXS`, or `NXT`).

When `_struct_conn` names a disulfide, a cysteine with a disulfide partner is not
free and a cysteine with no partner is counted. A disulfide is `conn_type_id`
`disulf`, or a `covale` link between two cysteine SG atoms. A loop row and a
single non-loop `_struct_conn` category both count. A loop whose values do not
fill a row does not.

A bond of any other kind is not an answer. A file whose only bonds are metal
coordination leaves free cysteine untested, not counted: the parse preset this
program uses keeps `covale` and `metalc` and drops `disulf`, so a design against
a metalloprotein carries coordination rows and no disulfide rows, and a count
taken from it would be a hard number from a file that recorded nothing about
disulfides. A CIF with no disulfide row leaves free cysteine untested and still
reports the sequence motifs. An untested free-cysteine read stores the state
and no count. A number the screen did not store is left off the closed page,
and a gauge is drawn only for a stored number.

The isoelectric point and the net charge at pH 7.4 use the Bjellqvist 1994 pKa set
(Electrophoresis 15:529-539), including the residue-specific terminal values used by
ExPASy Compute pI/Mw. The stored name is `bjellqvist_1994`. No pass band applies to
the counts or to either charge number. The measured line names a hotspot only when
its count is above zero, and it names only the motif names `NG`, `NS`, `DG`, `NXS`,
and `NXT`. The closed page uses those same names. A count stays when the name is
omitted. The closed page draws the report, including the zeros, on the
sequence-developability area. A chemical read that cannot be completed omits this
record and leaves the sequence metrics in place.

## Drug-type reports

`modality_screen` sits beside `chemical_liability`. The binder kind chooses the
checks. The sequence is not stored, and neither is a file path. `axis_evidence.json`
does not gain an axis for this report.

A protein or peptide chain is read from the generated CIF. Whitespace in the
chain is ignored. Furin sites are Arg-X-Lys/Arg-Arg, stored as `RXKR` or `RXRR`.
Overlapping windows are counted separately. Asp-Pro bonds are stored as `DP`.
N-terminal glutamine is true when the chain starts with Gln. A letter outside
the twenty standard amino acids is counted by its 1-based position and is not
stored as a letter. The rest of that chain is still scanned. The longest run of
one residue is stored as a length, a single residue letter, and a start number.
The longest hydrophobic run uses the Kyte-Doolittle residues with positive
hydropathy, the same scale as GRAVY: Ala, Cys, Phe, Ile, Leu, Met, and Val.
The aliphatic index is the Ikai 1980 mole-percent formula for Ala, Val, Ile,
and Leu. For a partly identified sequence, the reported lower bound uses the
identified residues and the full chain length. A wholly unidentified sequence
has no reported index. `sequence_entropy_bits` is the Shannon entropy of the residue
composition in bits, so the scale is fixed: a chain drawing evenly on all
twenty residues approaches 4.3219 and one residue alone is 0.
`most_common_residue` and `most_common_fraction` name the residue that takes
the largest share, with the earlier letter chosen when two tie, and
`distinct_residues` counts how many letters appear. These four catch the
low-complexity a run length cannot: `AVAVAVAVAV` has a longest run of one and
an entropy of 1 bit. No cut applies to any of them, and they stay off the
measured line. The closed page draws the entropy, the commonest residue, and
the distinct-residue count. The molecular weight is the sum of the free amino-acid masses from
the IUPAC atomic weights below, minus one water for each peptide bond. A letter
outside the twenty standard amino acids leaves that mass out. The measured
line names a furin site, an Asp-Pro bond, N-terminal glutamine, or
a nonstandard letter only when that count is above zero. The index, the mass,
and the reports below stay off that line. A residue number below
1 stays off the line and off the closed page. The count stays. Run lengths stay
on the closed page.

The molar extinction at 280 nm follows Pace et al., Protein Sci. 1995, 4:2411:
tyrosine 1490, tryptophan 5500, and cystine 125. `extinction_280_cystine` counts
every pair of cysteine as one cystine, and `extinction_280_reduced` leaves that
term at zero. An odd cysteine is not a cystine. The absorbance of a 1 g/L
solution is that extinction divided by the stored molecular weight, rounded to
four decimal places, and it is stored only when the extinction is above zero.
A letter outside the twenty leaves both numbers out. The source is `pace_1995`.

The N-end rule half-life is the ExPASy ProtParam table (Bachmair, Finley and
Varshavsky, Science 1986, 234:179; Gonda et al., J. Biol. Chem. 1989, 264:16700;
Tobias et al., Science 1991, 254:1374). The words are stored for mammalian
reticulocytes, yeast, and E. coli. Proline has no E. coli estimate, so that
field is omitted. A nonstandard N-terminus omits the estimate. The source is
`n_end_rule`. The closed page prints a phrase only when it is one of those
published words.

`residue_composition` counts the twenty standard letters that occur, and leaves
the key out when none do. A nonstandard letter stays in `nonstandard_count` and
is not stored as a letter. `atom_counts` sums the same residue formulas as the
molecular weight, in the order C, H, N, O, S, then removes H2 O once for each
peptide bond. A zero count is omitted. A letter outside the twenty, or a count
that would go negative, leaves the atoms out.

A Kyte and Doolittle 1982 span is a run of 19-residue windows whose mean
hydropathy is greater than 1.6. Overlapping windows merge into one span, which
stores the 1-based ends and the highest window mean in that span, rounded to
four decimal places. A chain shorter than 19 residues, or one with no window
above 1.6, stores a count of zero and an empty span list. A letter with no
hydropathy value leaves the report out. The method is `kyte_doolittle_1982`.
A mean equal to 1.6 is not a span. This count is separate from the longest
positive hydrophobic run, and it does not set the membrane axis.

AGGRESCAN follows Conchillo-Solé et al., BMC Bioinformatics 2007, 8:65. The
residue scale is Sánchez de Groot et al., BMC Struct. Biol. 2005, 5:18, Table 1.
The window is 5 residues at a length of 75 or below, 7 at 175 or below, 9 at
300 or below, and 11 above that. One virtual residue is added at each end: the
N-terminal value is the mean of Lys and Arg, and the C-terminal value is the
mean of Asp and Glu. Residues that sit outside a centered window copy the first
or last window center. `na4vss` is the sum of the window averages divided by
the length and multiplied by 100. A hot spot is five or more consecutive
residues whose window average is strictly greater than -0.02 and that contain
no proline. A chain shorter than its window, or a chain with a letter outside
the twenty, leaves `aggrescan` out. The method is `aggrescan_2007`. This number
does not change the call, and it is not a row of `solubility.tsv`.

No pass band applies to any of these numbers. They stay off the measured line.
The closed page draws them, including a zero extinction and a zero span count.

A nucleic chain uses A, C, G, T, and U. GC fraction is the share of G and C.
CpG positions are 1-based starts of the dinucleotide CG. A letter outside that
alphabet leaves the oligonucleotide report out. The measured line names CpG only
when the count is above zero. A position below 1 stays off the line.

A ligand report reads `_chem_comp` rows whose type is `non-polymer`, from a loop
or from a single non-loop category. A value may sit on the tag line or on the
next non-blank line. A quoted token is read without the quotes, and a quote
that contains spaces stays one cell, so a component name or formula does not
move the formula weight. The name and the formula are not stored. Water
component ids are left out. One remaining component stores its formula weight
when that cell is a positive finite number. When the cell is missing, zero, or
not finite, the weight is the sum of the IUPAC conventional atomic weights for
the chemical formula — C, H, N, O, P, S and most of the table are the 2005
values (Wieser, Pure Appl. Chem. 2006, 78, 2051), and about twenty heavier
elements carry a later revision, among them Pt 195.084 and Au 196.966569 —
rounded to
three decimal places, and the source is stored as `iupac_conventional`. A
trailing charge token is not part of the mass. The formula string is not stored.
A formula with an unknown element, a zero count, parentheses, or a dot hydrate
is left out, and no partial sum is stored. Two or more components store
`composition: ambiguous` and the count, and no weight. A loop whose values do
not fill a row adds no component. An unclosed quote is not a cell. The measured
line does not name the weight.
A component id of nine or more standard amino acids is omitted. No pass band
applies to any of these numbers. The closed page draws the report on the
sequence-developability area, under a heading that names the binder kind.

## Membrane, and the two oligo axes

Three axes sat in `AXES_BY_KIND` and in `BARS` from the day the classes were defined
and were measured by nothing. A peptide's `sequence` bar requires `membrane`; a nucleic
binder's `oligo` bar is `off_target_hybridization` and `innate_immune` and nothing else.
Until these were implemented neither bar could be reached by measurement at all: a
peptide could not clear its own bar however hydrophilic it was, and a nucleic design came
back `untested` with no blocker and no file anyone could supply to change it.

**`membrane`** is measured for a peptide only, from the Kyte-Doolittle 1982 scan that was
already being computed for the drug-type report and then thrown away. A span is a
19-residue window whose mean hydropathy is strictly above 1.6, merged with its
overlapping neighbours; one span is a `hit` and the spans are stored with their peak
hydropathy. No span, over a chain long enough to hold a window, is a `clear`. A chain
shorter than 19 residues is `chain_shorter_than_window` and stays undecided — the
criterion says nothing about an 8-mer, which is not the same as clearing it — and a chain
carrying a letter with no hydropathy value is `hydropathy_unknown`. The judge refuses the
axis for any other kind, so a protein binder's record does not carry it.

**`off_target_hybridization`** mirrors `proteome_offtarget`. Supply a nucleotide FASTA as
`transcriptome.fa` (or `.fasta`, `.fna`, `.txt`, and each of those `.gz`) beside the run, or name one in
`HUMAN_TRANSCRIPTOME`. Every exact 15-base seed of the design **and of its reverse
complement** is counted against it: an antisense oligo hybridizes where a transcript
carries its reverse complement, and a sense or passenger strand hybridizes as itself, so
both orientations are searched. U is read as T throughout, so an RNA design and a DNA
reference compare. One seed occurring in the file is a `hit`; none is a `clear`, and
because that clear is a statement about one file the file's base count, digest and name
are stored beside it. A reference under 1,000,000 bases is `transcriptome_too_small` —
the same guard the proteome reader applies, because a clear against a two-record FASTA
would read like a clear against a reference. A file whose letters do not read as bases is
`transcriptome_not_nucleic`, the mirror of `proteome_not_protein`: a protein FASTA is a
quarter ACGTN letters and holds only 423 windows of fifteen consecutive ACGT letters (in
105 low-complexity tracts of UP000005640_9606), so searching one would clear every
design but a poly-A, poly-G or poly-T one, against a file
that cannot hold a transcript.

The seed is 15 bases rather than 7, but 15 does not make a perfect match rare. 4^15 is
1.07e9 against the roughly 3e8 bases of a human transcriptome, so any one 15-mer occurs
0.28 times in it by chance, while 4^7 is 16384 and every 7-mer occurs thousands of times
over. This is the same arithmetic behind the >= 15-nt perfect-complementarity rule used
for antisense off-target triage, and it is why a seed-region match alone is not screened
here. A design shorter than one seed is `chain_shorter_than_seed`.

Measured against a 3e8-base reference, fifteen random designs of 18 to 80 nt and the
published ODN 2006 each matched at least one seed by chance -- 0.18 to 0.42 of their own
seeds -- and only a poly-A design cleared. So a hit is a pointer into the reference for
someone to read rather than a count of off-targets, and only a clear is a statement
about the whole file. The 1,000,000-base floor is the proteome reader's, and a clear
against a file that size is a clear against 0.3% of a transcriptome, which is why the
record stores that file's base count and digest.

A gzipped reference is read directly, which matters because that is the only form
Ensembl and GENCODE distribute cDNA in — without it the file this seed width is argued
for was never found beside a run, and the axis stayed untested on a machine that had it.
The byte bound is argued in uncompressed FASTA bytes, so a compressed reference is held
to a quarter of it rather than admitting four times the content the limit was chosen for.
A byte-order mark is tolerated, as it is in every other reader here.

**`innate_immune`** reads two published TLR9 CpG motifs: the murine consensus
5'-purine-purine-CG-pyrimidine-pyrimidine-3', whose optimal hexamer is GACGTT (Krieg et
al., *Nature* 1995, 374:546), and the optimal **human** hexamer 5'-GTCGTT-3' (Hartmann et
al., *J Immunol* 2000, 164:1617). GTCGTT is **not** an instance of the murine consensus —
its second base is a pyrimidine — and Hartmann reports that ODN built on the murine
formula show no or only weak activity in human cells, so both rules are read separately
and neither stands for the other. A design carrying a hexamer either rule reads is a
`hit`, with the count, the one-based positions and the motif that fired at each position
stored. CpG present but in neither motif keeps its count and stays `cpg_not_consensus`.

A design with no CpG dinucleotide at all is **not** a clear. It is
`cpg_free_not_evidence`, and that is deliberate: it cannot present the CpG motif, but
TLR9 is not the only innate sensor an oligo meets — RIG-I, TLR3, TLR7 and TLR8 and the
backbone chemistry are all outside what a sequence consensus can answer. A clear here is
curated into `axis_evidence.json` by someone who can speak for the rest. This was once the
same refusal `paralog` made for a singleton cluster; `paralog` now has a measured route
through Compara identity, and this axis has no equivalent.

All three roll up across a run by the rule the proteome axis has always used, now written
once in `rollup_axis`: one design hitting makes the axis a `hit` and is named in `failed`,
every design clear makes it `clear`, and anything else stays `untested` carrying the first
undecided design's own blocker. A hit's measurement is the worst of the hits, so a rollup
cannot read milder than the designs it summarizes.

## The declared pass criteria

`configs/metrics/pass_criteria/all_pass_criteria.yaml` declares eleven named criteria —
`unconditional`, `ppi`, `antibody`, `dna`, `motif_scaffold`, `internal_coords`,
`sm_binder`, `ame`, `ame_no_ligand`, `disco` and `developability` — each a set of metric
columns with a bound on every one. `rfd4-evaluate` evaluates them and writes a
`<name>.pass` column for each into `per_example_metrics.csv`.

The screen reads each criterion and its companion `<name>.all_metrics_present` column.
The separate `pre_oracle_pass` flag records the earlier filtering step.

`refold` now carries every declared criterion:

```json
"refold": {
  "criteria": [
    {"name": "ppi", "state": "fail",
     "thresholds": [
       {"metric": "oracle_ppi.ptm_coi", "bound": "min", "cut": 80.0, "measured": 86.2, "met": true},
       {"metric": "oracle_ppi.min_ipae_coi", "bound": "max", "cut": 1.5, "measured": 2.9, "met": false}
     ],
     "failed_thresholds": ["oracle_ppi.min_ipae_coi"]}
  ],
  "declared": 11, "evaluated": 1,
  "criteria_file": "configs/metrics/pass_criteria/all_pass_criteria.yaml"
}
```

The stored pass flag belongs to the evaluated run. The diagnostic bounds come from the
checkout at screen time, identified by `threshold_source: repository_defaults`. A run
with overrides can legitimately disagree with those comparisons. The reader resolves
the same-file threshold reference used by `internal_coords`. It supports inclusive and
exclusive ranges and per-design `min_of`, `max_of` and `mean_of` reductions. Each operand
must be finite. The reported measurement is the least favorable value among applicable
designs for each bound.

`rows_measured` counts determinate pass flags and `rows_expected` counts applicable
rows. A measured failure sets the criterion to `fail`. A whole-run `pass` requires
a measured pass on every applicable design. A partial evaluation reports
`incomplete`, including a stored zero with `all_metrics_present=0`. Legacy tables
without a presence column can still supply a binary flag. Missing rows cannot
establish that a bound held. A run without flags reports `not_evaluated`, and an
unreadable criteria file reports `criteria_not_found`. These entries are evidence
alongside the separately chosen screening bar.

## Folding a design somewhere else

Every criterion above needs a refold, and the oracle that produces one may not be
runnable on the machine that did the designing. AF3's weights are licence-gated, a
container runtime may be missing, and the AlphaFold Server has no API at all: it wants a
sign-in and hands a download to a browser. In all three the fold is obtainable — it just
does not arrive through a subprocess.

So the route is files in both directions. `src/rfproteina/disease/refold_export.py` writes
the request; `src/rfproteina/metrics/oracles/imported_refolds.py` files the answer at the
paths every oracle already looks in. Nothing in between is automated, and nothing is
uploaded: a request sits on disk until someone sends it.

```bash
# write the requests, and say what is outstanding
human --refold disease_runs/generations/<disease>/<target>/<run>

# file what came back, and say what is still missing
human --refold <run> --refold-import ~/Downloads/folds
```

### What the export writes

Two files per run. One `<design>.fasta` each, which is what any folder driven by sequence
takes — including a service whose submission is a person pasting text. And one
`refold_requests.json` recording what was asked for, so the manual return leg has
something to be checked against.

Each request carries every polymer chain, not just the binder, with the designed one
marked `|design` and the rest `|context`. Refolding a binder alone answers whether the
binder folds; it cannot answer whether the **complex** forms, which is what `ptm_coi`,
`min_ipae_coi` and the aligned binder RMSD are about. A request that omitted the target
would have changed the question without saying so.

The chain marked `|design` comes from the metrics table, which records it as a chain
*instance* id — `A_1` for the first instance of chain `A` — while `entity_poly` names the
strand `A`. The screen never had to reconcile those because it matches against a parsed
structure's `chain_iid`, where `A_1` is right. The export reads `entity_poly`, so it drops
one trailing instance number when the raw value matches no strand. A value matching
neither spelling leaves the chain unmarked rather than guessed: a request marking the
wrong chain as the design is worse than one marking none, because the mark is what a
reader trusts.

`--refold-dir` moves where the requests live; it defaults to `refold_requests` inside the
run.

**The hosted AlphaFold Server is not this route.** Its JSON dialect, allowlists, token
budget and daily quota belong to `rfd4-af3server` — see
[the AlphaFold Server guide](../how-to/af3-server.md). Use `rfd4-af3server prepare` and
`rfd4-af3server retrieve` for that service; this mode is for folders that take a
sequence.

### What the import accepts

A file is matched to a design when its stem holds that design's id at a token boundary —
containment, not a prefix, because tools put their own word first. The AlphaFold Server
returns `fold_binder_0_model_0.cif`, which no prefix rule matches. The boundary is what
keeps that from going too far: `binder_1` does occur inside `binder_10`, and what follows
it there is a digit, so one design can never claim another's fold.

Subdirectories are searched, so an unpacked download can be handed over as it is. Where a
tool returns several models for one design, the lowest rank in the filename is kept —
rank 0 is the best-scoring model in both AF3's and the server's numbering — and the rest
are reported as `superseded_files`. Importing all five in turn would have left whichever
sorted last in place, which for a ranked set is the worst of them. A file matching no
design is reported in `unmatched_files` rather than skipped, because an import that
silently did nothing is the likeliest failure here.

A confidence JSON beside the structure is used for that design, under any of four
spellings: `<stem>_confidences.json`, `<stem>.json`, or the summary file AF3 and the
server write (`<job>_summary_confidences.json`, numbered per model). Missing it is not a
loud failure — the fold imports with its confidence recorded absent, and the criteria that
needed those numbers then report bounds they could not check rather than bounds that
failed. Every imported record carries `imported_refold: true`, which a supplied file
cannot clear, so a reader can tell these numbers did not come from the oracle named on
the file.

A refusal writes nothing at all. The confidence JSON is what marks a design refolded, so
one written beside a structure that cannot be read would make the design look done and
measure nothing.

### Knowing what is left

Both directions end by reporting the round trip, because the leg between them is manual
and the server caps jobs per day: a batch comes back over several sittings.

```json
"status": {"requested": 22, "refolded": 15, "outstanding": ["binder_16", "binder_17"]}
```

`unrequested` appears when a design is refolded but absent from the manifest — a design
generated after the export was written, or folded by a local oracle. Not an error, and not
counted against the batch, but said, so a count here cannot quietly disagree with the
directory.

A design's refold is filed under its own base directory, which is the parent of its own
folder rather than the run root: a run nests designs below `inference_samples/`, and
`design_output_dir` requires a design folder to be a direct child of the base it is given.
Taking the ids by listing the run's directories instead produced `inference_samples` as a
design name and reported every real design as never having come back.

`--refold-oracle` sets the name the refolds are filed under. It appears in every filename,
so it has to match the oracle the metrics are configured with; it defaults to `af3`.

### What this does not do

It does not sign in, submit, or download. It also does not decide whether you are
permitted to use a particular fold: a service's terms are between you and that service,
and the AlphaFold Server's output terms in particular restrict what its predictions may be
used for. This reads and writes files you already have.

## Solubility and aggregation

A run folder accepts supplied scores from CamSol, Aggrescan3D, Protein-Sol
and AGGRESCAN in `solubility.tsv`, with `solubility.csv` as a fallback.
The drug-type report includes its own sequence AGGRESCAN calculation.

For native Protein-Sol output, put the authors' `seq_prediction.txt` in the
run folder. For an ordinary FASTA header, also supply the predictor input as
`protein_sol.fasta`. The full header must match the sample ID and its sequence
must match the resolved chain. A report with the FASTA header
`>sha256:<lowercase SHA-256 of the uppercase sequence>` can establish that
identity directly. Both inputs are limited to 1 MiB each.

The `protein_sol.samples` records preserve the native `scaled-sol` score,
the population reference, predicted percent solubility and predicted pI in
separate fields. A missing or ambiguous match carries a blocker for the
sample. The population value describes the authors' reference dataset.
These predicted properties have no pass threshold in the screen.
[University of Manchester software and paper](https://protein-sol.manchester.ac.uk/software)

Each row needs three things: a score (`score`, `solubility`, `solubility_score`,
`aggregation`, `aggregation_score` or `value`), the method that produced it
(`method`, `tool` or `predictor`), and a provenance (`source`, `study`,
`reference`, `doi` or `version`). A row missing any of them is not stored, and
`incomplete_rows` counts those. A `unit` and a `note` are kept when present.

```
score	method	source
0.72	CamSol intrinsic	Sormanni 2015
-1.8	Aggrescan3D a3d	Kuriata 2019
```

Several rows are kept in file order, so one design can carry a score from each
tool it was run through.

**Nothing here is judged.** The method is required because a solubility number
has no meaning without it — 0.4 is a poor CamSol score and a low AGGRESCAN one,
and the two mean opposite things — and for that same reason no cut turns a score
into a hit or a clear: the scales are not comparable to each other and no
published threshold spans them. This is a report, like the chemical liabilities,
not an axis. It is written for every binder kind and the screen's call does not
move.

The file is refused in full, as `solubility_unreadable`, when a column this
reader uses is repeated, when the header holds none of them, when a row carries
more fields than the header has columns, or when a quote is left open. More than
32 rows is `solubility_truncated` and a file over 1 MiB is `solubility_too_large`;
either withholds the whole table.

## Design diversity

`design_diversity` is the one record about the set rather than about a design.
Every other screen reads one chain, so a batch whose replicates came back as the
same molecule passes each of them once per copy and reads as that many separate
shots.

It is written whenever two or more designed chains could be read, for any binder
kind, and is left out otherwise: one design has nothing to differ from, and a
record saying so would read as a measurement. When any chains could not be
read, `unreadable_designs` counts them separately. The total attempted is
`designs` plus `unreadable_designs`.

`designs` and `distinct` are how many chains were read and how many of them are
different strings. Identity is counted position by position between two chains of
the same length only. A pair of different lengths is not aligned: an alignment
would have to choose a gap penalty, and the number it produced would be that
choice as much as the designs, so the pair is counted under `uncompared_pairs`
instead.

That is the common case, not the rare one. The binder contig this flow writes is
a range — 60 to 110 residues — so one run's designs routinely differ in length
and no pair of them can be compared. Such a run records `pairwise: not_compared`,
and no `groups`, `largest_group`, `max_identity` or `mean_identity` is written,
because a grouping built from zero comparisons would report every design as its
own and read as a spread that was never measured.

`max_similarity` and `mean_similarity` answer that run anyway. They are the share
of `kmer_width`-residue windows two designs have in common, as a Jaccard index
over every pair whatever their lengths: 1 when the two draw on exactly the same
set of windows, 0 when they share none. It needs no alignment and no common
length, so it is the number to read when `pairwise` is `not_compared`. A chain
shorter than one window is left out of it rather than scored 0.

It is reported and never grouped on. The cut that would turn a similarity into
"the same design" is not one this program gets to choose, so the grouping below
stays on positional identity, where the threshold has a plain meaning.

When at least one pair could be compared, `max_identity` and `mean_identity` are
over those pairs, and `groups` is how many designs the run really produced, under
single linkage at `threshold` (0.9): a chain of near-identical designs is one
design however it is reached, and a design with no comparable partner is its own
group. `largest_group` is the biggest of them. The threshold is a reading aid for
that count, not a pass band — no call turns on any of these numbers.

Past 200 designs the counts are still reported, because they cost one pass, and
the pairwise block is left out with `pairwise: not_performed` and the bound under
`limit`, because that work grows with the square of the batch.

## Supplied scores and curated evidence

### A curated call records what it rests on

Some axes have no measured clear at all. `innate_immune` refuses one for a CpG-free oligo.
`paralog` used to be in this list and no longer is: with `--paralog-identity-max` set and
Compara reachable it has a measured clear, and a curated one is needed only where Compara
cannot answer. For those, a curated clear is the only way the axis ever clears — and a `screened_clear` resting on one that recorded
no reason cannot be checked by anyone who was not in the room.

An entry in `axis_evidence.json` may still be the bare state it has always been:

```json
{"paralog": "clear"}
```

or it may carry the basis with it:

```json
{
  "paralog": {
    "state": "clear",
    "reason": "No reviewed human paralog shares the binding face.",
    "source": "UniProt 2026_03 review, ticket RFD-412"
  }
}
```

`reason` and `source` are both optional text, each at most 300 characters — a sentence and
a citation, not a document. The axis record then carries them under `curated`, and the
closed page prints them under the curated-call line. An entry with no basis records
`curated: "basis_not_stated"`, and the page says so: a curated call with nothing behind it
is a fact about the record, not an absence to pass over.

A field that is not `state`, `reason` or `source` is refused by name rather than ignored,
because a misspelled `reason` that is silently dropped leaves exactly the record the field
was added to prevent. So is a state that is not `hit` or `clear`, an entry with no state, a
blank reason and one past the bound.

An axis named twice in one file is refused the same way — `axis_evidence.json names paralog
twice. Each name is given once.` JSON keeps the last value for a repeated key and says
nothing, so appending a clear after a curated hit dropped the hit, with the file's own bytes
digested in `inputs` as though they said one thing. This is the refusal the metrics reader
already makes for a repeated column, and this is the one file that can set a state outright.

A file that cannot be read at all — a folder of that name, a dangling link, a permission
error — refuses that one directory with `axis_evidence.json could not be read.` It used to
raise out of the sweep, which abandoned every other run in the tree, and a dangling link was
read as though no evidence had been supplied.


`immunogenicity.tsv` and `ada.tsv` take a single `score` column and override the computed
measurement. A supplied score is reported but never becomes `hit` or `clear`, and it is
tagged `supplied_table` rather than named after a formula this program implements.

`axis_evidence.json` supplies a curated judgement for any axis **of this run's binder kind**:

```json
{"paralog": "clear"}
```

For a protein binder those are `immunogenicity`, `proteome_offtarget`, `mhc_i`, `mhc_ii`,
`ada`, `paralog`, `plasma_abundance`, `plasma_hub` and `plasma_fold`. Only `hit` and `clear`
are accepted as values.

An axis belonging to another kind, or a misspelled name, **refuses the screen for that
directory** rather than being ignored — a typo that silently did nothing would leave you
waiting for a verdict that was never going to change.

A supplied state overrides a measurement, and the record keeps what the measurement said
under `measured_state` and `measured_blocker`, with `source: axis_evidence` — so a curated
call is never mistaken for a measured one.

Every axis block prints that: the printed line names the axis, `source axis_evidence`, and
`measured <state>` when a measurement was overridden, whether the axis is a sequence axis, a
target axis, a plasma axis, a nucleic axis or the panel. Only the sequence axes were
rendered, so a curated clear over a measured `innate_immune` or `paralog` hit printed
nothing at all and the line read as a plain `screened_clear` — the axes that can only ever
clear by hand were the ones the sentence left out.

## Command-line flags

Use **Validate output files** in the finished-run browser to review files for experimental
handoff. The panel starts with recommendations for the first finished run. Choosing a different
run or **All finished runs** refreshes the recommended selection. Use **Refresh recommendations**
to pick up files added to the same run. The panel preselects installed checks that match the
available files and shows their on-disk sizes where available. **Select all installed** also
selects optional checks such as SHA-256 checksums. **Run selected checks** starts the checks
after you review or adjust the selection. The Zstandard check requires the local decoder.

For a single run, use **Candidate** to scope the structure checks and recommendations to one
sample. A bench-evidence check is recommended when an entry matches that sample or the record has a
run-level status. Shared metrics-table checks and synthesis-receipt checks are optional in this view.

**Therapeutic experimental evidence** checks the format of an attached
`experimental_validation.json` and reports how many entries have a recorded status.
**DNA-synthesis review receipt** checks for a separate `synthesis_review.json` supplied by the
reviewer. The panel reports the receipt's presence and leaves the review decision with the reviewer.

Open the same panel with `--validate-output` and repeat the flag to add another run.

```bash
human --validate-output disease_runs/gene_1ABC_1
human --validate-output disease_runs/gene_1ABC_1 --validate-output disease_runs/gene_2ABC_1
```

### Choose scientific screens for a finished candidate

Open **Post-generation screening** on the results page. For multiple runs, choose a run
first. Choose a candidate and refresh to see its own recommendations. The initial checks
use the run's binder kind, the candidate's nominated inputs, and the local installation.
Model-backed methods check their required software and assets. Each method shows its
workload and explains which inputs it needs. A prepared QupKake, SyGMa, Vina, PRODIGY
or nanobody manifest requests that method when the inputs match.

Uniquely named rows in `ligand_candidates.tsv` can be selected before coordinates
exist. **Select recommended screens** uses the chosen candidate's inputs, stated
experimental objective and workload. **Select all installed screens** also checks
installed methods that need more inputs. The execution report explains each skip.
Saved, attributable observations remain readable if the original producer was removed.

For small molecules, optionally place `screening_context.json` in the run directory.
Use the same candidate IDs as the molecular nominations.

```json
{
  "candidates": {
    "oral-lead": {"route": "oral", "systemic": true},
    "topical-lead": {"route": "topical", "systemic": false}
  }
}
```

Each entry accepts a route, a boolean declaration of systemic exposure, or both.
Supported routes are `oral`, `intravenous`, `subcutaneous`, `intramuscular`,
`inhaled`, `topical` and `other`. A route alone makes no claim about systemic
exposure. A valid `cypstrate.json` can also supply the systemic declaration for
its named candidates. Partial or conflicting evidence leaves the affected
recommendation manual. At run scope, the chosen batch must meet the method's
criteria. Selecting one candidate evaluates just that nomination.

Every option shows **Validation performance**. **Evidence basis** tells you how
that performance was assessed. For example, PRODIGY's quoted error comes from
its development set, while sPyRMSD was compared with another RMSD implementation.
Agreement between programs verifies the calculation. Predicting recovery of an experimental
pose requires a matched redocking control. [PRODIGY method](https://wenmr.science.uu.nl/prodigy/method) ·
[sPyRMSD study](https://doi.org/10.1186/s13321-020-00455-2)

A published estimate identifies the endpoint, metric and evaluation population with its source. Sensitivity
and specificity describe two different outcomes at a given cutoff, and RMSE
measures error in the endpoint's units. For a measurement, the row names the
controls or reference needed to assess error. A model without a verified
estimate for the configured release says so. **What this means** explains how
much weight to give the result for an experimental decision. QupKake's published
RMSE of 0.5 to 0.8 pKa units matters most when the expected pKa is near the assay
pH. The Caco-2 study's external macrocycle Spearman correlations of 0.40 for permeability
and 0.19 for efflux support experimental follow-up, particularly for efflux.
The picker identifies that study's model and the subsequent retraining of the
released weights. [QupKake study](https://doi.org/10.1021/acs.jctc.4c00328) ·
[Caco-2 supplement, Figure S4](https://doi.org/10.1021/acsomega.5c04861.s001)

These annotations are available offline, are returned by MCP `plan_screening`
in each method's `performance` object, and accompany selected methods in
`screen.json` under `method_selection.choices`. A locally trained BindingDB
regressor can report held-out errors by target from its configured model manifest.
Each target's error describes that fit. Source links let a reviewer check
published endpoint-specific values, including those distributed with
the retrained release of ADMET-AI. [ADMET-AI v2.0.1 endpoint table](https://github.com/swansonk14/admet_ai/blob/v_2.0.1/admet_ai/resources/data/admet.csv)

The browser captures each selected method's evidence before it runs. Open the
result to see what was available at execution time, even after local model
files change. A result saved without a snapshot keeps its original measurement.
MCP callers can request the same per-result evidence with `include_performance: true`
on `run_selected_screens`. Omitting that flag returns the usual compact results.

**Badapple 2 historical scaffold assay evidence** can be recommended when the
local snapshot and chemistry worker are available and every selected nomination
is a supported cyclic small molecule. Mixed batches remain selectable. Review
the experimental coverage beside each reported scaffold score when choosing counterscreens.

**RDKit assay-interference alerts and descriptors** are checked by default for a ligand
whose nomination identifies one complete, connected molecular graph. The picker verifies
the descriptor software and the PAINS A/B/C, BRENK and NIH catalogues before recommending
the screen. A salt or mixture can be screened manually as supplied. A run containing both
usable and unresolved nominations also requires manual selection. Choosing an individual
candidate checks only its own nomination. Duplicate IDs, wildcard atoms and malformed SMILES
cannot identify a testable molecule. A single titled record in `ligand_smiles.txt` belongs
to the run. A result for an individual candidate requires a row in `ligand_candidates.tsv`.
Catalogue hits can inform the choice of experimental assay controls.
[RDKit filter catalogue documentation](https://www.rdkit.org/docs/GettingStartedInPython.html#filtering-unwanted-substructures)

**ADMET-AI aqueous solubility** is checked for an explicitly nominated, connected,
carbon-containing molecule with zero net charge. The default covers one CPU batch
of up to 64 distinct molecular inputs. Equivalent SMILES and atom-map labels share
one input. Larger batches, salts, mixtures, net-charged forms and carbon-free
structures remain available for manual selection. For a salt or ion, confirm the
chemical form and planned assay conditions before interpreting the estimate.
The picker verifies the published endpoint definition and all five local regression
checkpoint hashes. Selection reads the molecular graph without running inference.
The result estimates aqueous logS across the conditions represented in the training
data. A pH-specific solubility claim needs a measurement at that pH.
[AqSolDB study](https://www.nature.com/articles/s41597-019-0151-1)

**fastsolv organic-solvent solubility** can be recommended when the local
model is installed, `fastsolv.json` specifies supported conditions, and the
selected solutes all meet the chemistry requirements. The picker shows how many
solutes qualify and the total predictions across the requested conditions.
Select a partially eligible batch manually or select one eligible solute.

**Jazzy hydration and hydrogen-bond descriptors** can be recommended when
the isolated runtime is installed and all selected molecular forms meet the
neutrality and stereochemistry requirements. A mixed batch remains available
for manual screening of its eligible members.

Badapple 2, MetFrag, fastsolv and Jazzy preserve the source of a saved
measurement. A single `ligand_smiles.txt` supplies one result row labelled
`run_ligand`. A named candidate requires a nomination in `ligand_candidates.tsv`.

**Supplied solubility and aggregation measurements** are checked when every row in
scope has a finite score, a method, a source, and an explicit unit or scale. Use
`unit` or `units` for a physical unit or a named predictor scale. A readable score
with an unspecified scale, or a table with some incomplete rows, can be reviewed
manually. An optional `candidate_id` column matches rows to the exact example ID.
A named observation whose ID is absent from the run requires manual review.
A table without that column can be selected for the whole run. Each observation
keeps its candidate, method, source and original value. The reader accepts at most
32 rows and 1 MiB per file.

**Protein-Sol result reader** is checked when the selected protein's resolved
designed chain matches one native prediction, using its exact sequence hash or
the original matching FASTA entry. The prediction can inform a cell-free expression
experiment. A partial run, a peptide selection, or a scope with more than 16 MiB of
structure files requires manual selection. The reader verifies sequence identity
before returning any score. An invalid report, missing match, ambiguous match, or
unresolved chain leaves the affected candidate unchecked. [Protein-Sol study](https://doi.org/10.1093/bioinformatics/btx345)

**Dimorphite-DL ligand protonation states** can be recommended for a finished
ligand run with a valid `dimorphite.json` and eligible molecular input. Set the
pH window from the assay or docking conditions before selecting the screen.

**CREST ligand conformer sampling** can be recommended after the picker verifies
the local runtime and the prepared SDF explicitly nominated in `crest.json`.
For a named candidate, the manifest and SDF title must identify that candidate.

**PROPKA structure-dependent pKa** can be recommended when the installed runtime
and the explicitly prepared PDB named in `propka.json` are eligible for the
selection. For a named candidate the manifest must identify that candidate.
Set `assay_ph` when the predicted sites will inform a specific assay pH.
A PDB with incomplete termini remains available for manual screening. Review
the missing atoms before interpreting a partial result.

**QupKake microscopic pKa** is checked for an expressly configured ligand run
with one supported, identifiable molecular state. Each distinct state starts its own
xTB calculation. Separate candidate IDs describing the same graph share that calculation.
Multiple distinct states, partially usable nominations, unnamed SDF records and
hydrocarbons stay available for manual selection. The latter require an explicit
site-specific assay question. The picker uses the provider's own graph and atom-map
checks and verifies the released code, checkpoints, xTB executable and parameters.
It can make this decision before enumerating protonation sites or running xTB.
[QupKake method](https://doi.org/10.1021/acs.jctc.4c00328)

**SyGMa human metabolite prediction** is checked when a ligand run requests
metabolite ranking for one identifiable organic parent with resolved stereochemistry.
The default budget permits one iteration of each requested phase. Multiple parents,
repeated reaction phases, or unspecified parent stereochemistry warrant a manual
choice. Equivalent nominations share one ranking. An invalid graph, a disconnected
salt, or an ambiguous ID cannot identify a parent. The picker verifies both release
rule tables and leaves reaction enumeration to the selected run. The ranking
proposes structures for analytical confirmation. [SyGMa release](https://pypi.org/project/SyGMa/1.1.0/)

**MetFrag measured spectrum comparison** can be recommended when the local
runtime is available, `metfrag.json` names an eligible measured product-ion
list, and at least one nominated ligand matches the measured precursor window.
The picker shows how many nominations matched and how many peaks were supplied.
It checks how many distinct eligible molecular structures those nominations represent.
A lone compatible structure can explain fragments, but distinguishing alternative
identities requires additional candidates or a reference standard.

**P2Rank receptor binding pockets** can be recommended after the picker verifies the local
runtime and an eligible receptor nominated in `p2rank.json`. Create the manifest
when the site is uncertain or a predicted-site comparison would inform review.
All candidates in the run share the nominated receptor's result.

**ESP-Sim prealigned ligand comparison** can be recommended when the local
runtime is installed and `espsim.json` identifies an eligible candidate and
reference pair in the same frame.

**OpenAntigens human target evidence** can be recommended when the local snapshot
is installed and contains the run's exact canonical human UniProt accession.
The lookup provides context shared by the run's candidates.

**PoseBusters** is checked for one uniquely named ligand with a concrete molecular
graph and a finite 3D conformation in the nominated SDF. The picker keeps a
distorted pose eligible so the run can report failing geometry. Separate poses
require separate energy checks and a manual choice. A 2D drawing or ambiguous
pose ID cannot supply coordinates for a candidate. When reviewing stored results,
named boolean checks and complete published pass rates can be preselected.
Coverage counts alone and partial published runs remain available for review.
The native SDF option uses the ligand-only checklist. [PoseBusters documentation](https://posebusters.readthedocs.io/en/latest/)

**Ligand component composition** is checked when each selected structure names
one non-water ligand component with a positive formula weight or a complete
interpretable formula. The result identifies the component and the source of its
mass in daltons. When the supplied and calculated masses disagree at their shared
precision, both are reported and the method stays manual. A partly readable run
also stays manual. The 16 MiB interactive budget counts decompressed structure
text. [wwPDB formula-weight definition](https://mmcif.wwpdb.org/dictionaries/mmcif_pdbx_v50.dic/Items/_chem_comp.formula_weight.html)

**Protein composition and mass** is checked for an attributable designed protein
or peptide chain whose standard amino-acid sequence is fully resolved. A nucleotide
chain or conflicting assignments for the same strand cannot provide this endpoint.
The reported average mass is for the unmodified linear chain. Modifications and
annotated disulfides prompt manual review before comparing it with an intact-mass
measurement. A partially resolved cohort can be inspected per candidate.

**Protein charge and isoelectric point** is checked for a fully identified standard
polypeptide when the Bjellqvist model brackets a zero crossing between pH 0 and 14.
The result reports the modeled charge at pH 7.4 and the pKa set used. When the
curve has no crossing in range, the charge can still be selected manually and the
pI carries its own blocker. Annotated modifications and disulfides also prompt
manual review of this unmodified-chain estimate. [ExPASy pI documentation](https://web.expasy.org/compute_pi/pi_tool-doc.html)

**GRAVY mean hydropathy** is checked when every residue in each assigned protein
or peptide chain has a value on the Kyte–Doolittle scale. The score is a residue
average. Both high and low values are reportable. Partial cohorts and chains
with annotated modifications can be selected manually. [Kyte and Doolittle](https://pubmed.ncbi.nlm.nih.gov/7108955/)

**Instability index** is available for manual comparison when an assigned protein
or peptide has at least two adjacent standard residues and its complete sequence
is known. The empirical index was derived from 12 unstable and 32 stable
proteins. Its numerical value does not select a candidate or a screening
checkbox. [Guruprasad and colleagues](https://pubmed.ncbi.nlm.nih.gov/2075190/)

**Sequence diversity** is checked for a run containing two to 200 identified
designed polypeptides when at least one pair supports positional identity or
five-residue window comparison. Exact replicates are eligible. Selecting one
candidate cannot establish diversity. Partial or modified cohorts and cohorts
that support only distinct counts remain manual. A partial report includes
the number of unreadable designs.

**AGGRESCAN sequence score** is checked for an assigned standard protein or
peptide sequence long enough to fill the published window. The shortest window
is five residues. The result names the method, window size and normalized
aggregation score. Annotated modifications and partially readable cohorts
require manual selection. [AGGRESCAN study](https://pmc.ncbi.nlm.nih.gov/articles/PMC1828741/)

| Screen | When the picker recommends it |
| --- | --- |
| Stereochemistry | Each selected candidate has a recognized reference-comparison endpoint on its declared scale. The producer's `percent_correct` columns contain fractions from 0 to 1. An attached percentage needs an explicit percentage unit. |
| P-SEA | The selected protein chain has at least six consecutive observed C-alpha residues. Short traces can be reviewed manually with the coverage warning. |
| Steric clashes | The parsed structure has finite atom coordinates and at least one nonbonded pair eligible for the native check. |
| Bond lengths | The parsed structure contains bonds with finite coordinates at both ends. |
| Octant and BindingDB exact lookups | The nominated molecular form has a determinate measurement in the local reference. Multiple distinct compounds or a reference above the interactive byte budget prompt manual selection. |
| BindingDB fingerprint and GtoPdb | Valid named compounds can be selected for comparison with reference compounds. These comparisons are optional. |
| BindingDB ExtraTrees and Chemprop | An installed, compatible model and a valid molecular nomination make model inference available for manual selection. |
| OpenFE | The proposed edge and both simulation phases agree across completed files. A single repeat in either phase leaves the uncertainty unresolved and the reader manual. |
| pmx | Completed work inputs and a configured PyMBAR runtime make the method available for manual selection. Selecting it fits BAR and checks the supplied estimate and analytical uncertainty. |
| BayeshERG | A nominated connected molecule and the verified local model make an optional cardiac ion-channel screen available. Each distinct molecule requires 30 stochastic draws. |
| Supplied toxicology | At least one supplied assay has a readable concentration, a named source and endpoint, and an identity matching the nominated compound. The table is reviewed at run scope. Distinct assays retain separate readings. |
| Therapeutic Nanobody Profiler | One explicitly nominated VHH matches the structure's chain, full sequence digest and observed residues. |
| AutoDock Vina supplied pose | One prepared ligand and receptor match the selected structure, and the ligand is within the specified scoring box. |
| sPyRMSD redocking pose recovery | Each nominated control in scope has a matched experimental ligand, two readable 3D poses and matching receptor-frame IDs. |
| SCScore synthetic complexity | An explicitly nominated comparison contains distinct eligible small molecules and remains open for chemistry review. |
| CYPstrate substrate profiling | Systemic exposure and at least one supported human CYP isoform are declared for the selected small molecule. |
| PK-Sim human organ exposure | A nominated molecule matches a registered individual human small-molecule model, regimen and concentration observers. |
| CellProfiler regional DAB intensity | Candidate and negative-control fields have matched specimen and acquisition metadata, with a named mask for each field. |
| PRODIGY | Both configured protein partners are complete and share an interpartner contact in the chosen coordinate model. Several prepared candidates at run scope require manual selection. |

For BayeshERG, place the released source and checkpoint under `DATA_PATH/bayesherg`
and the compatible Linux interpreter at `DATA_PATH/bayesherg-runtime/bin/python`.
`BAYESHERG_ROOT` and `BAYESHERG_PYTHON` can select an explicitly configured
installation. The authors specify noncommercial terms for the weights and prediction
results. [BayeshERG release terms](https://github.com/GIST-CSBL/BayeshERG)

During interactive generation, the browser offers these choices when the run files
are ready, before the optional screens start. An explicit CLI selection applies directly.

**Select all installed screens** includes optional methods and readers. Each selected
method returns a separate result. A method whose inputs belong to another candidate or
are missing returns its reason and skips the calculation. **Select recommended screens**
restores the defaults for the current scope. **Clear screen selection** allows an empty
selection. **Run selected screens** processes the selected methods serially.

Calculations produce a fresh measurement. Local result readers inspect the named analytical
files. The previously recorded observations section displays attributable values from the
saved screen, published metrics or a supplied observation file. A database client in this
section reads attached observations locally. The picker makes no remote database request.
When generated files are present, only observations naming a candidate in the run are
returned. Mixed or unverifiable candidate attribution leaves the stored review manual.
DockQ scalar observations stay manual so the reviewer can identify the reference
complex and chain mapping. [DockQ documentation](https://github.com/bjornwallner/DockQ)
The main screen keeps its required verdict criteria.

```bash
human --screen finished-run --screen-selection recommended
human --screen finished-run --screen-method ligand_solubility --screen-method sygma
human --screen finished-run --screen-selection all_installed
```

An optional `screening_observations.json` maps a method ID to rows containing
`candidate_id`, `endpoint`, `unit` and a finite numeric `value`. Use the exact example ID
from the run. The panel returns up to 512 values for each method and reports how many additional
observations it found. It preserves each method's scale and the attachment's attribution.

PoseBusters accepts named 3D records in `ligand_candidates.sdf` for molecule-mode checks.
RDKit alerts, Octant, BindingDB and GtoPdb accept the same SMILES nominations as ADMET-AI.
BindingDB exact measurements, fingerprint neighbours and model predictions have separate
options. The
toxicology reader accepts a run-level `toxicology.tsv` or `toxicology.csv`. To verify
completed free-energy outputs, supply one of the following manifests with its source files
directly in the run directory. The named candidate identifies the comparison being reviewed.

| Manifest | Required fields |
| --- | --- |
| `openfe_inputs.json` | `candidate_id`, `ligand_a`, `ligand_b`, and `results` containing the complex and solvent result filenames |
| `pmx_inputs.json` | `candidate_id`, `report`, `forward`, and `reverse` containing the pmx BAR report and integrated-work filenames |

```bash
# screen a finished run again, after dropping tables in
human --screen disease_runs/gene_1ABC_1

# every finished run beneath a directory
human --screen disease_runs/generations
```

One invocation screens at most 200 runs, found no more than four directory levels below
`DIR`, and searches at most 400 directories. It says so when it stops at either limit and
exits non-zero, so a larger archive is screened a subtree at a time rather than silently in
part.

The search does not follow a directory symlink and does not read a folder whose name begins
with `.`, and it now says so too: when such a pruned subtree holds a finished run, the sweep
reports itself partial instead of answering "No finished generation was found" and exiting
0. An archive kept on another volume and linked in, or filed under `.archive`, is screened
by naming that path directly.

That check is a bounded look, not a second search. A link to a folder the sweep reads anyway,
such as `current` pointing at a run beside it, is not a gap and is not reported as one. A
pruned folder is read for at most 60 folders, and at most 40 pruned folders are looked into
per sweep. A run buried deeper than that is not reported, so name such an archive directly
rather than relying on the warning.

| Flag | Effect |
|---|---|
| `--screen DIR` | Re-screen a finished directory, or every run beneath it. Generates and downloads nothing. Rewrites `screen.json` and refreshes the screening block of every research trace beside it. |
| `--proteome PATH` | The FASTA for `proteome_offtarget`. Defaults to `$HUMAN_PROTEOME`. |
| `--bar NAME` | Which named set of axes must all be clear. Defaults to the kind's own bar. A bar that kind does not have is refused by name, before anything is measured. |
| `--lookup-targets` | Look the target up in UniProt, the Human Protein Atlas and STRING. Without it the three target axes report `network_not_permitted`. |
| `--plasma-abundance-max` | Blood concentration in pg/L at or above which the target is an antigen sink. |
| `--plasma-hub-max` | High-confidence interaction partners above which the target is a hub. |
| `--plasma-fold-max` | Fold change, by magnitude, at or above which the target is a liability. |
| `--follow-up` | Write and print the generation commands the screen selects for a next round. Starts nothing. |
| `--binder-kind` | Which bar applies. A screen-only run reads it from the run's own files unless you type it. |

## Every blocker, and what to do about it

| Blocker | What it means | What to do |
|---|---|---|
| `unsupported_caco2_structure` | the submitted graph is outside the local Caco-2 screen's structure scope | nominate the intended connected, closed-shell organic molecule |
| `caco2_input_limit_exceeded` | the Caco-2 input exceeds the file or nomination limit | use a nomination file within the 1 MiB and 256-row limits |
| `caco2_checkpoint_missing` | the pinned AstraZeneca Caco-2 checkpoint was not found | ask the administrator to install the published v1 checkpoint |
| `caco2_checkpoint_invalid` | the local checkpoint could not be read or does not match the pinned release | check file access and verify the documented SHA-256 |
| `caco2_runtime_unavailable` | the local Caco-2 prediction runtime could not be loaded | check the installed `ligand-safety` environment and checkpoint-read support |
| `caco2_output_contract_mismatch` | the model output does not match the pinned Caco-2 head contract | check the checkpoint and runtime versions before retrying |
| `dili_structure_unsupported` | the graph is outside this DILI screen's supported structure scope | nominate the intended connected, closed-shell organic molecule |
| `dili_inputs_over_limit` | the DILI nomination source exceeds 1 MiB or 256 rows | provide a bounded candidate file |
| `dili_resources_unavailable` | the local DILI ensemble exceeds the verified artifact limit | check the installed ADMET-AI v2 classification distribution |
| `predictor_not_configured` | no table for this axis | add `mhc_i.tsv` / `mhc_ii.tsv` |
| `predictor_unreadable` | the file is not parseable as a table | check the encoding and delimiter |
| `peptide_column_absent` | ranks with no peptide named | add a `peptide` column |
| `allele_column_absent` | no allele column in the class II table | add an `allele` column |
| `allele_panel_incomplete` | the DR panel is not fully covered | score the alleles named in `missing_alleles` |
| `rank_absent` | the panel is covered but no row carries a rank | the median is a percentile; add a rank column |
| `rank_scale_unknown` | the table does not declare a consensus method | re-run with the IEDB consensus method, or add a `method` column |
| `score_absent` | a table with no usable number | check the rank or affinity column |
| `table_not_from_this_design` | the peptides are not windows of this design | the table is another run's scan; re-score this chain |
| `reference_not_configured` | no reference table | score the three reference proteins |
| `reference_unreadable` | the reference file is not parseable | as above |
| `reference_length_unknown` | the reference scan was filtered, or states a peptide length | re-export the complete scan, or add a real `length` |
| `reference_too_short_for_width` | the reference length was read, and is shorter than every width the table scored | score the reference over a width the protein holds; `reference_proteins`, `stated_length` and `reference_widths` are on the record |
| `ic50_not_nm` | an affinity column states no unit, or states one that is not nM | the cutoffs are nanomolar and nothing is rescaled: rename the column `ic50_nm`, or add a `unit` cell reading `nM`. `not_nm_columns` names the column |
| `reference_ic50_not_nm` | the same, in the reference table | as above. The bar is what a `clear` is taken against, so an affinity of unknown unit cannot set one |
| `width_without_cutoff` | every peptide in the table is of a width no binding cutoff is defined for | score 8–11-mers for class I or 13–25-mers for class II; 12 falls between the two published ranges |
| `calis_mask_unpublished_for_allele` | a presented ligand is restricted by HLA-B\*08:01, whose anchors the paper puts at P2/P5/P9 | no position weight is published for that mask, so the score cannot be computed for it and no anchor set is invented. `mask_unpublished_alleles` names the allele |
| `reference_method_mismatch` | design and reference came from different tools | score both with the same tool |
| `reference_allele_mismatch` | design and reference used different alleles | score both over the same allele set |
| `cell_unparsed` | a rank, affinity or score cell is neither a number nor blank | most often a comma decimal from a German or French Excel export; re-save with a dot. `unparsed_columns` names the column |
| `reference_cell_unparsed` | the same, in the reference table | as above |
| `scan_incomplete` | the scan covers too little of the chain for a density of zero, or for a clear | export the complete scan — every window, at every width scored — not a truncated or filtered one |
| `unscored_windows` | a density of zero, but some window of the chain was never scored | the rank cells reading `-` or `n.d.` are windows the predictor declined; score them, or accept that absence cannot be claimed. `unscored_peptides` counts them |
| `reference_density_implausible` | a reference protein has more than a quarter of its windows binding | the table contradicts its own cutoff — rank ≤ 1 is the top 1% of peptides, and albumin measures 0.49 per 100 residues. Re-export the scan |
| `reference_method_unknown` | exactly one of the two tables declares a `method` | add a `method` column to both, or to neither. Neither tool writes one by default, so a verdict with neither is measured and records `reference_method_undeclared` — one side declaring and the other not is the case that hid a real mismatch |
| `reference_width_mismatch` | design and reference scored different peptide lengths | score both over the same widths; the density counts binders of every width and divides by one length, so four widths against one inflates the bar about fourfold |
| `reference_length_inconsistent` | one contiguous scan whose stated `length` is wrong | correct the length; `stated_length` and `observed_extent` are on the record |
| `reference_allele_unknown` | one of the two tables declares no `allele` | add an `allele` column to both, over the same alleles |
| `reference_not_tolerated_protein` | the reference names no protein from the table above | score albumin, IgG1 or ubiquitin; `reference_proteins` names what was found instead |
| `reference_rows_exceed_protein` | more rows than one protein of that length can hold, or a row ending past it | the export pooled several proteins under one accession; keep each protein's own accession and length |
| `no_presented_ligand` | the class I table presented nothing | expected for a design with no predicted ligand |
| `no_nine_mer_ligand` | it presented ligands, but none of length 9 | include 9-mers in the scan |
| `presented_not_in_design` | the presented peptides are not in this chain | re-score this chain |
| `proteome_not_configured` | no FASTA | pass `--proteome` or set `HUMAN_PROTEOME` |
| `proteome_unreadable` / `proteome_empty` / `proteome_too_large` | the FASTA could not be used | check the file is a protein FASTA of reasonable size |
| `proteome_not_protein` | the records read as DNA or RNA | `cds.all.fa` where `pep.all.fa` was meant |
| `proteome_too_small` | far fewer distinct 9-mers than a proteome holds | a UniProtKB export with the **Sequence** column unticked, or a single protein. Every 9-mer the file lacks is an off-target the axis cannot see, so a clear against it is not a clear |
| `sequence_absent` | no designed sequence to measure | check the run produced designs |
| `structure_unparsed` | a loop in the generated file does not divide into rows | check the data lines for a trailing comment or an unclosed quote |
| `not_a_polypeptide` | the designed chain is not a protein or peptide, so GRAVY and the instability index do not apply | expected for a nucleic or small-molecule design; read its own triage report instead |
| `designs_absent` | the run holds no generated design | check the run produced designs |
| `evaluation_structure_table_only` | The evaluator recorded inputs and the screen read their table. | Read the per-design measurements in the evaluator CSV. |
| `designs_compressed` | every design in the run is one this screen cannot read: a binary `.bcif`, a `.cif.zst` on a machine without `zstandard`, or an archive that does not decompress — most often a plain CIF renamed rather than compressed | `.cif` and `.cif.gz` are always read and `.cif.zst` is read where `zstandard` is installed; otherwise decompress the designs beside the run and screen it again. An archive whose header is good and whose tail is truncated is a per-design `structure_unreadable` instead, because that one really is a damaged compressed design |
| `no_pass_cut` | a number with no threshold that applies to it | informational |
| `cpg_not_consensus` | the design holds CpG and none of it is one of the two consensus hexamers | a cut does apply on this axis — one hexamer is a hit — so this is not `no_pass_cut`. Zero hexamers is not an absence of innate-immune signal, because the other sensors are not scored here. Curate the call if you want a state |
| `cluster_singleton_not_evidence` | the target is alone in its UniRef50 cluster | curate the call into `axis_evidence.json` |
| `cluster_not_found` / `members_not_listed` / `members_truncated` | the paralog lookup could not answer | retry, or curate |
| `target_not_accession` | the target is a gene symbol and no condition spec resolved it | `paralog` and `plasma_abundance` need an accession |
| `target_accession_ambiguous` | the fixed chains of one deposit name different proteins, or two condition specs in the campaign name different accessions | name the target explicitly |
| `target_not_named` | the run names no target at all | supply `generation.json` or a job file that does |
| `target_not_reported` / `concentration_not_reported` | the atlas holds no blood concentration | nothing to do |
| `partners_not_listed` / `degree_censored` | the partner lookup could not answer | retry |
| `threshold_not_set` | a measurement with no cut named | pass the matching `--plasma-*-max` |
| `fold_not_configured` / `fold_not_reported` | no fold table, or none with a usable number | add `plasma_fold.tsv` |
| `fold_unreadable` | the fold table is present but not parseable | check the encoding and the delimiter |
| `fold_too_large` | the fold table is past 1 MiB or holds more than 256 data rows | a fold table is a few rows per target; a file this size is a different file. Same bounds the co-evolution and solubility readers keep |
| `fold_target_not_in_table` | the fold table labels its rows with an accession or a gene and holds none for this target | add a row for this target, or remove the label column so the file is read as this run's own. An unlabelled table is still read as this run's |
| `source_not_named` | the fold table names no study, or its largest swing is on a row that names none | add a `source` column. A sourced row clearing the cut does not answer for a louder unsourced row beside it, so the verdict is withheld and the unsourced number reported |
| `toxicology_not_configured` | No toxicology table was found in the output folder. | Add toxicology.tsv to the output folder. |
| `toxicology_unreadable` | The toxicology table could not be read. | Check the encoding and the delimiter. |
| `toxicology_not_reported` | The toxicology table has no usable nM reading for this check. | Add an nM column for this panel protein. |
| `solubility_not_configured` | No solubility table was found in the output folder. | Add solubility.tsv to the output folder. |
| `solubility_unreadable` | The solubility table could not be read. | Check the encoding and the delimiter. |
| `solubility_not_reported` | The solubility table has no row naming a score, a method and a study. | Give each row a score, the method that produced it, and a study. |
| `solubility_truncated` | The solubility table has more rows than this screen will read. | Keep one row per method. The limit is 32 rows. |
| `solubility_too_large` | The solubility file is larger than this screen will read. | Keep the file within 1 MiB. |
| `reading_not_decisive` | The cell bounds the reading without placing it on one side of the cut. | Record the measured concentration, or a bound that falls clear of the cut. |
| `reading_not_nm` | The cell is not a non-negative nM reading. | Write the reading in nM. |
| `coevolution_not_configured` | No co-evolution table was found in the output folder. | Add paralog_coevolution.tsv to the output folder. |
| `coevolution_unreadable` | The co-evolution table could not be read. | Check the encoding and the delimiter. |
| `coevolution_not_reported` | The co-evolution table has no usable row for this check. | Name same shape, co-evolution, and other-species toxicity. |
| `coevolution_unparsed` | A co-evolution cell could not be read as yes or no, high or low, or toxic or non-toxic. | Use yes or no, high or low, and toxic or non-toxic. |
| `coevolution_not_classified` | Co-evolution is not named high or low. | Write high or low in the co-evolution column. |
| `toxicity_not_classified` | Other-species toxicity is not named toxic or non-toxic. | Write toxic or non-toxic. |
| `same_shape_absent` | No row names the same shape in another species. | The four transfer notes apply when the same shape is present. |
| `coevolution_disagreement` | Rows with the same shape name different notes. | Keep one class in the table. |
| `coevolution_truncated` | The co-evolution table has more data rows than this screen will judge. | Keep one row per species. The limit is 256 data rows. |
| `coevolution_too_large` | The co-evolution file is larger than this screen will read. | Keep the file within 1 MiB. |
| `metrics_not_configured` | No per-design metrics table was found in the output folder. | Evaluate the run to write `inference_metrics/per_example_metrics.csv`. |
| `metrics_not_reported` | The metrics table holds no rows. | Check the run wrote a row per design. |
| `metrics_publication_pending` | A publication is unfinished. | Let the writer finish or rerun the in-place writer to recover it. |
| `metrics_aggregation_required` | Inclusion of the current job files is unverified. | Aggregate the source directory. |
| `metrics_provenance_mismatch` | The primary table differs from its recorded digest. | Restore the verified table and receipt or rebuild from the original jobs in a fresh directory. |
| `metrics_provenance_unreadable` | The receipt or a source file could not be verified. | Check both and recover from verified copies. |
| `metrics_unreadable` | The metrics table could not be read. | Check the encoding and the delimiter. |
| `metrics_too_large` | The metrics file is larger than this screen will read. | Keep the file within 16 MiB. |
| `metrics_columns_repeated` | The metrics table names a column this screen reads more than once. | Keep one copy of each column and screen again. |
| `metrics_row_too_long` | A metrics row carries more fields than its own header. | Export the table again without the stray separator. |
| `metrics_row_too_short` | A metrics row stops short of its own header. | Export the table again with every row the width of its header. |
| `metrics_row_absent` | No row of the metrics table names this design. | Check the table names this design in `example_id`. |
| `metrics_row_ambiguous` | Two metrics rows name this design and disagree about it. | Screen one job's table at a time, or drop the row that is out of date. |
| `metrics_row_shared` | The same evidence matches multiple structure files. | Export a separate row with its structure path for each design. |
| `chain_disagreement` | The metrics table names more than one chain of interest. | Screen one target at a time, or name the chain in every row. |
| `chain_shorter_than_window` | the chain is shorter than the 19-residue window | a span cannot be claimed or ruled out below 19 residues |
| `hydropathy_unknown` | the chain carries a letter with no hydropathy value | check the designed chain is a standard protein sequence |
| `transcriptome_not_configured` | no reference for the off-target seed count | add `transcriptome.fa` or `transcriptome.fa.gz` to the output folder, or set `HUMAN_TRANSCRIPTOME` |
| `transcriptome_unreadable` | the reference is not parseable | check the file is a nucleotide FASTA of reasonable size |
| `transcriptome_empty` | the reference holds no sequence | as above |
| `transcriptome_too_large` | the reference is larger than this screen will read | keep the file within 512 MiB, or 128 MiB compressed |
| `transcriptome_too_small` | the reference is too small to support a clear | use a reference of at least 1,000,000 bases |
| `transcriptome_not_nucleic` | the reference does not read as DNA or RNA | this is a protein FASTA; supply the transcript sequences |
| `sequence_not_nucleic` | the designed chain is not a DNA or RNA sequence | check the run wrote a nucleic chain for this design |
| `chain_shorter_than_seed` | the design is shorter than one 15-base seed | a seed match cannot be claimed below 15 bases |
| `design_windows_unscannable` | every seed window of the design spans an ambiguity code, so nothing was searched | resolve the degenerate positions, or accept that this design cannot be searched |
| `cpg_free_not_evidence` | no CpG at all, which is not a clear for the other innate sensors | curate the call into `axis_evidence.json` |
| `not_evaluated` | no declared pass criterion was evaluated for this run | run the refolding oracle to evaluate them |
| `criteria_not_found` | the declared pass criteria could not be read | expected at `configs/metrics/pass_criteria/all_pass_criteria.yaml` |
| `criteria_not_declared` | the pass criteria file declares none | check the file is the one this repository ships |
| `thresholds_not_resolved` | a criterion borrows its bounds in a form this reader does not resolve | its own `<name>.pass` column still stands |
| `network_not_permitted` | no lookup was asked for | pass `--lookup-targets` |
| `lookup_unavailable` | a service did not answer | retry |
| `empty_command` | a screening tool was asked to run with no command | no input causes this; report it as a defect in the screen |
| `interface_quality_failed` | the interface quality measurement did not complete for this design | check the design and the target are both in the generated file |
| `module_unavailable` | the module this check runs could not be imported | reinstall this project's environment; the module ships in-tree, so what is missing is a dependency of it |
| `synthesis_failed` | the construct synthesis screen returned no reading for this sequence | check the designed chain is a plain DNA or RNA sequence |
| `no_commec_verdict` | the biosecurity screen ran and its flag table names no recommendation for this design | re-run the screen so every record gets a row, or screen this design on its own |
| `no_numbered_chains` | no chain was IMGT-numbered, so there are no antibody regions to scan | give `--heavy`, `--light`, or both, and check ANARCII accepted them |
| `oracle_unreadable` | the developability oracle's table could not be read | check the encoding and delimiter |
| `oracle_too_large` | the oracle's table is larger than this screen will read | keep the file within 1 MiB |
| `oracle_columns_absent` | the table defines none of the columns this oracle reports, so it is not that oracle's output | supply the oracle's own CSV, with its property columns named in the header |
| `oracle_truncated` | the oracle's table holds more rows than this screen will judge | keep one row per design; the limit is 64 rows |
| `oracle_not_reported` | the oracle's table holds no row carrying a usable score | check the property columns carry numbers |
| `no_scorable_peptides` | the domain yields no 9-mer of standard residues, so no human peptide was looked up | supply a variable domain of at least nine standard residues |
| `no_paired_chains_supplied` | antibody-likeness is scored over a heavy and light pair, and none was supplied | pass `--likeness` with both `--heavy` and `--light` |
