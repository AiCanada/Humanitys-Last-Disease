# DDD Benchmark readiness: a gap audit

The [Insilico Drug Discovery and Development Benchmark](https://dddbench.insilico.com/)
(launched 30 July 2026) evaluates frontier and foundation models served through a
standard chat-completions API, scoring their output against expert reference
baselines. It has two suites: *Drug Discovery Foundations*, 300+ evaluations built
from decontaminated public data and proprietary out-of-distribution sets, and *Drug
Candidate Essentials*, which follows an end-to-end programme from hit identification
to preclinical candidate nomination. Coverage spans medicinal chemistry, chemical
synthesis, disease biology, clinical development and longevity.

**This document is a gap audit, not a submission plan.** This program is not a
frontier model: it is a CLI, an MCP server, a generative design pipeline and a
screening layer. It cannot be submitted as it stands, and the useful thing the
benchmark offers here is its taxonomy — a map of what a drug-discovery system is
expected to cover, against which this program's coverage can be read honestly.

## 1. Domain coverage

Measured against the source tree, not estimated.

| DDD domain | coverage | what exists | what does not |
| --- | --- | --- | --- |
| **Disease biology** | **Strong** | Target identification across 8 databases; Reactome pathway selection; `liability.py` immunogenicity (Calis 2013), proteome off-target, MHC class I; `target_liability.py` paralog, plasma abundance, plasma hub, plasma fold; `coevolution.py` | Mechanistic disease modelling; perturbation prediction |
| **Medicinal chemistry** | **Partial** | `ligand_alerts.py` (Lipinski, QED, SA score, PAINS, BRENK); `ligand_safety.py` 10-axis panel via ADMET-AI; `ligand_regressor.py` for the 3 axes ADMET-AI lacks; `octant.py` measured CYP; `toxicology.py` | No SAR reasoning, no lead optimisation loop, no generative small-molecule chemistry, no potency prediction against a novel target |
| **Chemical synthesis** | **Absent as the benchmark means it** | `synthesis.py` scores *gene* synthesis manufacturability: local GC windows, long repeats, hairpins — whether a DNA construct can be ordered | Zero modules for retrosynthesis, reaction routes, or organic synthesis planning |
| **Clinical development** | **Absent** | — | Zero modules for trial design, phases, endpoints, regulatory strategy |
| **Longevity** | **Absent** | — | Zero modules |

Verified by search across `src/rfproteina/disease/`: `clinical trial` 0, `phase i` 0,
`regulatory` 0, `longevity` 0, `retrosynthes` 0, `reaction route` 0, `SAR` 0.

### The modality mismatch underneath

*Drug Candidate Essentials* runs hit identification to PCC nomination, which is a
small-molecule arc. This program is biologics-first: protein binders, antibodies,
nanobodies, enzymes, nucleic acids. Its ligand side is real but is the less finished
half. Three of five domains scoring zero is not a defect to fix — it is a statement
that the two things are built for different jobs.

## 2. Contamination audit

The benchmark decontaminates public data against its test items. This program
touches public data in two distinct ways, and **the second is the one easily
missed**.

### 2a. Lookup tables read at runtime — lower risk

| source | read by |
| --- | --- |
| IEDB (`mhc_ligand_full`, `tcell_full_v3`) | 8 modules |
| UniProt (proteome, reference sequences) | 19 modules |
| PDB / AFDB | 12 modules |
| ChEMBL | 5 modules |
| TEPITOPE matrices | 5 modules |
| BindingDB | 4 modules |
| Reactome | 4 modules |
| Commec biorisk / benign | 3 modules |
| OpenADMET Octant CYP | 2 modules |
| UKB-PPP blood PEA | 2 modules |
| OASis v1 | 1 module |
| STRING (plasma-hub axis) | 1 axis |
| Human Protein Atlas | plasma-abundance axis |

A runtime lookup only contaminates if a benchmark item asks about a row in the exact
table a screen reads. That is possible but narrow.

### 2b. Training corpora of embedded models — higher risk

A model trained on a public corpus can answer from memory even when the program
never reads that corpus at runtime.

| model | trained on | overlap risk |
| --- | --- | --- |
| **ADMET-AI** | Therapeutics Data Commons ADMET suite | **HIGH** — TDC is *the* public ADMET benchmark; ADMET-AI tops its leaderboard |
| **Chemprop** (inside ADMET-AI) | the same TDC datasets | **HIGH** — same corpus |
| **`ligand_regressor`** (fitted in this repo) | BindingDB, scaffold-split | **HIGH** — public and heavily reused |
| MHCflurry 2.3 | IEDB + mass-spec ligand sets | MEDIUM |
| ThermoMPNN | Megascale stability dataset | MEDIUM |
| `class_ii` (fitted in this repo) | IEDB CC BY rows | MEDIUM |
| DeepSP | 20,530 sequences, MD-derived labels | LOW — labels are simulated, not a QA corpus |
| DeepViscosity | 229 measured mAbs | LOW |
| AbLang-2 | Observed Antibody Space | LOW |
| CrossAbSense v0.9 | GDPA1 assay data | LOW |

**Three HIGH entries must be declared before any evaluation**, and all three are on
the small-molecule side — the same side where coverage is weakest. An ADMET or
potency score from this program would be the least trustworthy number it produces
and the most likely to look good for the wrong reason.

## 3. The question to settle before anything is built

**This has not been sent. It is the user's to send.**

> The DDD Benchmark scores model output against expert reference baselines. Our
> system is built to decline rather than guess: on a representative run, 7 of 10
> screening axes returned `untested`, each with a machine-readable reason —
> `rank_scale_unknown` (the published threshold is defined on a rank scale our
> installed predictor does not produce), `threshold_not_set` (the bar is
> caller-supplied and unset), `cluster_singleton_not_evidence` (a one-member cluster
> is absence of evidence), `reference_density_implausible` (the reference table
> cannot support a bar). Every one of those refusals is correct.
>
> 1. How does the benchmark score a calibrated refusal that states its reason —
>    as a wrong answer, as a non-response, or is there provision for it?
> 2. Does the Core Score distinguish a confident wrong answer from a declined one?
> 3. For decontamination, do you need us to declare the *training corpora of
>    embedded models* (ADMET-AI on TDC, an in-repo regressor on BindingDB), or only
>    the datasets read at inference?
>
> The answer to (1) determines whether an evaluation would measure our system or
> penalise its main design property.

If refusals score as wrong answers, the benchmark would rank this program below one
that guesses — and the resulting number would mislead whoever reads it. That is worth
knowing before, not after.

## 4. What this audit changed in the code

The audit surfaced defects; they are fixed rather than listed.

- **The MHC reference scan no longer repeats per design.** `prepare_class_i_table`
  rescanned all three tolerated proteins on every pass, because the reference table
  is written per run directory while the proteins never change. A single
  `screen_directory` call on a one-design fixture was measured starting
  `mhcflurry-predict` eight times. The tail path now reuses an existing
  `mhc_i_reference.tsv`, as the early-return path already did, and records
  `reference_reused`.
- **Fourteen more thresholds carry provenance**, closing the gap this document would
  otherwise have had to report. The registry now covers 31: the 12 verdict cuts, the
  5 interface bars, and 14 bars living inside individual screens. Of those 14, five
  are `published` (Lipinski's four rule-of-five terms, and Xu & Zhang's same-fold
  TM-score of 0.5), seven are `convention`, two are `reported_only`.
- **A name collision was caught by the guard**, not by review: `evalue_max` is a bar
  in both `novelty` and `offtarget`, with the same value and different reasoning, so
  that family is keyed `<module>.<bar>`. The guard also found an unregistered bar,
  `ligand_alerts.sa_score`, on its first run.

## 5. Where this program would place honestly

Strong, and defensible against an expert baseline:

- **Disease biology to target triage.** Demonstrated end to end: cystic fibrosis →
  MONDO_0009061 → Reactome pathways including "Defective CFTR causes cystic
  fibrosis" → 421 ranked structures.
- **Biologics developability.** DeepSP → DeepViscosity → humanness → CrossAbSense,
  licence-clean throughout, with the viscosity port validated against the published
  probabilities to a worst difference of 4.6e-07.
- **Class I immunogenicity**, measured against a tolerated-protein baseline rather
  than asserted.
- **Structure-level design and interface scoring**, with DockQ supplying published
  CAPRI bands.
- **Biosecurity screening**, which most pipelines omit.
- **Auditability.** Code identity, the cuts a verdict was taken against, and what
  kind of thing each threshold is. If a benchmark ever scored whether a result can be
  checked, this is where the program wins outright.

Weak, and it would show:

- Anything requiring synthesis planning, clinical reasoning or longevity biology.
- Small-molecule potency and ADMET, where the numbers exist but rest on the three
  HIGH-contamination models above.
- Any item needing a bar this program has not calibrated: of 31 registered
  thresholds, only 7 are `published`.
- Throughput. One screen of one design took 736 s before the reference-scan fix.
