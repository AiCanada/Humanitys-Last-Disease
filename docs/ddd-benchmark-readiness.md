# DDD Benchmark readiness: a gap audit

The [Insilico Drug Discovery and Development Benchmark](https://dddbench.insilico.com/)
scores a system's output against expert reference baselines.

**Read from the site on 2026-11-18** (it reports "last updated Nov 16 2026"):
**7 domains, 308 benchmarks**, of which 61 are the Core subset — Target Biology (12),
Medicinal Chemistry (239), Chemical Synthesis (5), Clinical Trials (28), Longevity (16),
Therapeutic Biomolecules (6) and Structural Biology (2). The last two are
**Antibody Developability** and **ResidueBench**, and both are this program's own
subject matter.

**Two premises of the first version of this audit were wrong**, and are corrected here:

- It described five domains, from the launch announcement rather than the live site,
  and so missed the two that matter most to this program.
- It said the program "cannot be submitted as it stands". Submission is by endpoint —
  "Share model details → Connect a secure endpoint → Receive your private report. We
  query only the endpoint you provide" — and specialist, non-generalist systems are
  ranked alongside chat models. A pipeline can be evaluated.

What the program does and does not cover is now **in code**, not in this document:
`src/rfproteina/disease/ddd_benchmark.py` carries the 20 items in scope with the
blocker that declines each one it cannot answer, and
`src/rfproteina/disease/corpus_provenance.py` carries what every embedded model was
trained on. Both are tested. This document keeps the argument; the data lives where
it can be checked.

## 1. Domain coverage

Measured against the source tree, not estimated.

The domain names below are the seven the site now uses. Two of them were missing
from the first version of this table.

| DDD domain | coverage | what exists | what does not |
| --- | --- | --- | --- |
| **Target Biology** (12) | **Strong for 9 of 12** | Target identification across 8 databases via `sources.py` (Open Targets associations, ranked, which is what Precision@K scores); Reactome pathway selection; `liability.py` immunogenicity (Calis 2013), proteome off-target, MHC class I; `target_liability.py` paralog, plasma abundance, plasma hub, plasma fold | The 3 Single-Cell Perturbation sets: no single-cell module, no perturbation prediction |
| **Structural Biology** (2) | **1 of 2** | ResidueBench *Mutations* — `stability.py` site-saturation ddG through the vendored ThermoMPNN, verified at 194 positions / 3,686 mutations | ResidueBench *Evolutionary pressure*: nothing computes per-residue conservation. `coevolution.py` reads a caller-supplied paralog table and infers none |
| **Therapeutic Biomolecules** (6) | **0 of 6 today** | `developability.py` reads a CrossAbSense or DeepSP CSV; `deepsp.py` and `deepviscosity.py` run here | None of the six assays is produced. CrossAbSense is read rather than run, its weights are over the size cap (except PR_CHO), and SMAC, ovalbumin polyreactivity and HAC have no head at all |
| **Medicinal chemistry** | **Partial** | `ligand_alerts.py` (Lipinski, QED, SA score, PAINS, BRENK); `ligand_safety.py` 10-axis panel via ADMET-AI; `ligand_regressor.py` for the 3 axes ADMET-AI lacks; `octant.py` measured CYP; `toxicology.py` | No SAR reasoning, no lead optimisation loop, no generative small-molecule chemistry, no potency prediction against a novel target |
| **Chemical synthesis** | **Absent as the benchmark means it** | `synthesis.py` scores *gene* synthesis manufacturability: local GC windows, long repeats, hairpins — whether a DNA construct can be ordered | Zero modules for retrosynthesis, reaction routes, or organic synthesis planning |
| **Clinical Trials** (28) | **Absent** | — | Zero modules for trial design, phases, endpoints, regulatory strategy |
| **Longevity** (16) | **Absent** | — | Zero modules |

Verified by search across `src/rfproteina/disease/`: `clinical trial` 0, `phase i` 0,
`regulatory` 0, `longevity` 0, `retrosynthes` 0, `reaction route` 0, `SAR` 0.

**Ten of the twenty items in scope are answerable today**, and that number is
asserted in `tests/test_disease_ddd_benchmark.py` so it moves only on purpose.

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

**This table used to live here and was wrong.** It is now
`src/rfproteina/disease/corpus_provenance.py`, with a guard test that reads the
vendored CrossAbSense tree for GDPa1's assay columns rather than restating them,
so a vendor bump fails a test instead of silently ageing this page. Ask it
directly:

```bash
python -m rfproteina.disease.cli --declare-corpora --corpora-against HIC
```

The error worth recording, because it is the one this design prevents: the old
table rated **CrossAbSense LOW**. CrossAbSense is fitted on **GDPa1** (Ginkgo,
242 IgGs), whose columns are `HIC, AC-SINS_pH7.4, Titer, Tm2, PR_CHO, HAC, Tm1,
Tonset, Purity, SEC_%Monomer, AC-SINS_pH6.0` — the same assay panel the
benchmark's Antibody Developability column scores. Against those assays it is
the *highest*-risk model in the program, in the one domain where it would
otherwise look strongest.

The mistake was structural, not clerical: **the table stored a severity per
model.** Overlap is not a property of a model. It is a property of the pair
(model, what you are scoring it on) — AbLang-2 on the Observed Antibody Space is
genuinely unaligned with a developability assay and would be a label corpus
against a humanness benchmark. The registry therefore stores **mechanisms**,
which do not drift, and derives the severity for a pair on request.

Still true, and now machine-readable: ADMET-AI and the Chemprop model inside it
rest on Therapeutics Data Commons, which ADMET-AI tops the leaderboard of, and
the in-repo ligand regressor rests on BindingDB. All three are on the
small-molecule side, which is why the 239 Medicinal Chemistry benchmarks are
deliberately out of scope: that is where this program would score well for the
wrong reason.

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
