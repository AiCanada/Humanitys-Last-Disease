# Production report 4 — Screening: protein and peptide binders

**Program:** `Humanitys-Last-Disease`, the disease-binder design flow built on RFD4-Proteina.
**Scope:** every check a finished protein or peptide design is put through after generation.
**Status:** implemented and tested. Not medically verified. Every number below is a
computational prediction; none is an experimental result.

## Where the code lives

Paths are relative to the repository root of the RFD4-Proteina fork carrying the disease
flow, which is not in this repository yet.

| Component | Path |
|---|---|
| Screen record writer | `src/rfproteina/disease/screen.py` |
| Sequence liability axes | `src/rfproteina/disease/liability.py` |
| Target liability axes | `src/rfproteina/disease/target_liability.py` |
| Chemical liability | `src/rfproteina/disease/chemical.py` |
| Drug-type report | `src/rfproteina/disease/modality.py` |
| Paralog co-evolution note | `src/rfproteina/disease/coevolution.py` |
| Reader reference | `docs/reference/screen-inputs.md` |

## How a verdict is formed

Each check is an **axis** with one of three states: `hit`, `clear`, or `untested`. A design
is screened against a named **bar** — a set of axes that must *all* be clear. One axis
reading `hit` makes the call `axis_hit` regardless of the bar; every axis in the bar clear
makes it `screened_clear`; anything else stays `untested`.

The default bar for a protein or peptide is the sequence bar: immunogenicity, proteome
off-target, MHC class I, MHC class II, ADA, paralog — and, for a peptide, membrane.

**`untested` is a real answer, not a failure to answer.** Where a check cannot decide, the
record names *why* with a blocker from a fixed vocabulary, and the reference documents what
to supply for each one. This is the single most important property of the screen: it does
not fill a gap with a guess, and it does not let an absence read as a pass.

## Immunogenicity — Calis 2013 class I 9-mer

Scored from the sequence itself with the published position weights, the
position mask, and the documented exception allele. A clear requires **every** 9-mer of the
chain to score at or below zero. A single positive window is stored and the axis stays
`untested` — a positive score is not treated as a hit, because the formula predicts
immunogenicity of a presented peptide, not that the peptide is presented.

A supplied immunogenicity table overrides the formula. A restricted table — one listing
only what a predictor said was presented — is handled separately and must cover the chain.

## Proteome off-target — exact distinct 9-mer identity

Every distinct 9-mer of the designed chain is matched against a supplied protein FASTA. Any
shared 9-mer is a `hit`; none is a `clear`.

Two things travel with the verdict, because a clear here means *"no shared 9-mer in **this**
file"*:

- the number of k-mers the reference held, and
- a digest of that k-mer set.

A reference below one million k-mers cannot support a clear. The guard exists because a
two-record FASTA shares nothing with anything, and a clear against it would read exactly
like a clear against a whole proteome.

Per design, and rolled up for the run: one design hitting makes the axis a hit for the run
and that design is named.

## MHC class I and class II — from a local predictor table

Read from tables the user generates with their own predictor and drops beside the run.

| Axis | Rank cut | IC50 cut |
|---|---|---|
| MHC class I | ≤ 1.0 | ≤ 500 nM |
| MHC class II | ≤ 10.0 | ≤ 1000 nM |

The judgement differs by binder kind, deliberately:

- A **peptide** is judged by the window rule directly.
- A **protein** is judged by *binding density against reference proteins* — serum albumin,
  the IgG1 constant region, and ubiquitin — because the absolute cutoffs mark essentially
  every protein, so an absolute threshold would make the axis useless. The frame is epitope
  content relative to tolerated proteins rather than an absolute cut.

A density of zero is only accepted from a scan that looked at every window. A table whose
rank cells read `-` or `n.d.` is declaring that it declined to score those windows, and
counting them as non-binders would turn four unscored 9-mers into a false clear.

**A table is checked against the designs.** Its peptides must be windows of a design, by a
share rather than by a single match — an inherited table from an earlier design in the same
folder shares only incidental windows, and one shared window was once enough to produce a
confident verdict about a chain nothing had scanned. The record also names which designs
the table covers and which it does not, so a run of eight screened on a scan of one is
visible as such.

## ADA — Paul 2015 DR-panel breadth

Median percentile rank across the seven-allele DR panel. A peptide binds an allele at a
consensus rank of 20 or better, and the blind-validated rule is a median of 20 or better
across the seven; both numbers are the paper's, on the IEDB consensus rank scale.

The table must declare a consensus method. A rank from a different tool means something
else on a different scale, so an undeclared scale leaves the axis `untested` rather than
judged. The panel must also be fully covered; a partial panel names the alleles that are
missing.

## Paralog — UniRef50, with a co-evolution note

`hit` when another reviewed human protein shares the target's UniRef50 cluster.

**There is no measured clear.** The absence of a cluster-mate is not evidence of absence: a
UniRef50 cluster is built around a representative and does not gather a protein's
near-identical relatives, so well-known targets come back as singletons while having exactly
the families a binder would cross-react with. A singleton is recorded as *not evidence* and
a clear here must be curated by a person — and a curated clear records the reason and source
it rests on.

Beside the axis sits a **co-evolution note**, read from a supplied table: whether the same
shape exists in another species, whether co-evolution is high or low, and whether the
other-species protein is toxic or non-toxic. The four combinations give four different
transfer strengths. The note does not change the paralog state; it qualifies it.

## Plasma axes — properties of the target, not the design

| Axis | Source | Cut |
|---|---|---|
| Plasma abundance | Human Protein Atlas blood concentration, pg/L | only against a cut you name |
| Plasma hub | STRING physical interaction partners at high confidence | only against a cut you name |
| Plasma fold | a supplied fold-change table | only against a cut you name |

These three report their number and stay `untested` unless a threshold is named, because
there is no published universal cut for "too abundant" or "too connected". The fold table
must name the study it came from: a fold change without a provenance is a rumour.

These axes run outside the sequence gate, since they say nothing about the designed chain.

## Sequence developability

| Measure | Cut | Basis |
|---|---|---|
| GRAVY | ≤ 0.1 | program convention |
| Instability index | ≤ 40.0 | ExPASy ProtParam, Guruprasad 1990 |

Both are computed from the designed chain. This is **not** the repository's
`developability` pass criterion, which additionally bounds Rosetta SAP and hydrophobic
exposed surface area — those are not run here, and the screen says so.

## Chemical liability

Reported for a protein or peptide from the designed chain. **These are counts and numbers,
not an axis: none of them clears or fails a verdict.**

- **Deamidation** motifs `NG`, `NS`
- **Isomerization** motif `DG`
- **N-glycosylation** sequons `NXS`, `NXT`
- **Methionine** and **tryptophan** counts (oxidation-prone)
- **Free cysteine**, counted only when the structure file names its disulfide partners —
  without that, a cysteine's state is unknown rather than free
- **Isoelectric point**, Bjellqvist 1994, bisected to a residual far inside the fourth
  stored decimal
- **Net charge at pH 7.4**

A chain that keeps one charge sign across the whole pH range has no isoelectric point, and
the record says so rather than reporting a bracket endpoint.

## Drug-type report

Also reported, also not an axis:

- **Furin cleavage** sites `RXKR`, `RXRR`
- **Asp-Pro** acid-labile bond
- **N-terminal glutamine**
- **Nonstandard residues**, by position
- **Longest same-residue run**
- **Longest hydrophobic run**
- **Aliphatic index**
- **Average molecular weight**
- Extinction at 280 nm (Pace 1995), N-end rule half-life, residue and atomic composition,
  sequence entropy, a Kyte-Doolittle hydropathy span, and a normalised AGGRESCAN sum

## Interface measurements

Taken from the generated structure when it has coordinates: a contact molecular surface and
a clash fraction at 1.5 Å. Where either cannot be measured, the record names which of six
reasons applies — including the ordinary case of a sequence-only file from an
inverse-folding run, which is not a failure.

## What this report does not claim

- No binding affinity, dissociation constant, or binding free energy is produced.
- No docking is performed.
- A `screened_clear` means every axis in the named bar was clear **against the inputs
  supplied**, recorded alongside the cuts, the axis list, the input digests and the code
  commit that produced it.
- A clear screen is a computational result. It is not experimental validation.
