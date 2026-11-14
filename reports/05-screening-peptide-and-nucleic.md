# Production report 5 — Screening: peptide addition and nucleic binders

**Program:** `Humanitys-Last-Disease`, the disease-binder design flow built on RFD4-Proteina.
**Scope:** the one axis a peptide adds to the protein set, and the three checks a nucleic
binder gets.
**Status:** implemented and tested. Not medically verified.

## Where the code lives

Paths are relative to the repository root of the RFD4-Proteina fork carrying the disease
flow, which is not in this repository yet.

| Component | Path |
|---|---|
| Membrane span scan | `src/rfproteina/disease/modality.py` |
| Nucleic axes | `src/rfproteina/disease/oligo.py` |
| Nucleic drug-type report | `src/rfproteina/disease/modality.py` |
| Axis declarations and bars | `src/rfproteina/disease/combinations.py` |

A note on history worth recording: all three of these axes were **declared in the axis
table and in the bars from the start, and measured by nothing**. A peptide could not reach
`screened_clear` on its own bar however hydrophilic it was, and a nucleic design came back
`untested` with no blocker and no file anyone could supply to change that. The strategies
were declared and unimplemented. They are implemented now.

---

# Peptide: the membrane axis

A peptide is screened on everything a protein is screened on, plus one axis.

**Measurement.** Kyte-Doolittle mean hydropathy over a 19-residue window. A window whose
mean is **strictly above 1.6** is a span; overlapping spans merge, and each stored span
carries its peak hydropathy.

| Outcome | Condition |
|---|---|
| `hit` | at least one span |
| `clear` | no span, over a chain long enough to hold a window |
| `untested` — `chain_shorter_than_window` | chain shorter than 19 residues |
| `untested` — `hydropathy_unknown` | a residue with no hydropathy value |

A chain shorter than the window is **refused, not cleared**. The criterion is about
spanning a bilayer; it says nothing about an 8-mer, and that is not the same as clearing it.
This matters here more than it looks: a peptide binder in this program is at most 20
residues, so at most two windows exist and the axis is often legitimately undecidable.

The number itself was already being computed for the drug-type report and then discarded.
Making it an axis added no new computation.

The axis is refused for any other binder kind, so a protein's record does not carry it.

---

# Nucleic binders

## Off-target hybridization

Exact **15-base seeds** of the design **and of its reverse complement**, counted against a
nucleotide FASTA supplied beside the run or named in the environment.

Both orientations are searched because an antisense oligo hybridizes where a transcript
carries its reverse complement, while a sense or passenger strand hybridizes as itself. `U`
is read as `T` throughout, so an RNA design and a DNA reference compare.

**Why 15 and not a seed-region 7.** Fifteen is the width at which a perfect match stops
being an accident: 4¹⁵ is 1.07 × 10⁹, past the roughly 3 × 10⁸ bases of a human
transcriptome read on both strands, whereas 4⁷ is 16 384 and every 7-mer occurs in such a
file thousands of times over. A 7-mer screen would mark every oligo and mean nothing. This
is the same arithmetic behind the ≥ 15-nt perfect-complementarity rule used for antisense
off-target triage, and it is why a seed-region match alone is not screened here.

| Outcome | Condition |
|---|---|
| `hit` | one or more seeds occur in the reference |
| `clear` | none do |
| `untested` | no reference configured, unreadable, empty, too large, or **too small to support a clear** (below one million bases) |
| `untested` | the design is shorter than one seed |

A clear is a statement about one file, so the file's base count, digest and name are stored
beside it. The scan is blocked with a seed's worth of overlap carried between blocks, so a
match straddling a block boundary is still found and the whole reference is never held in
memory.

## Innate immune

The **TLR9 CpG consensus**: purine–purine–CG–pyrimidine–pyrimidine (Krieg 1995; the optimal
human hexamer of Hartmann 2000 is one instance of it).

| Outcome | Condition |
|---|---|
| `hit` | the design carries a consensus hexamer; the count and one-based positions are stored |
| `untested` — `no_pass_cut` | CpG present, never in the consensus; the CpG count is kept |
| `untested` — `cpg_free_not_evidence` | **no CpG at all** |

A CpG-free oligo is deliberately **not** a clear. It cannot present the CpG motif — but
TLR9 is not the only innate sensor an oligo meets; RIG-I, TLR3, TLR7 and TLR8 and the
backbone chemistry are all outside what a sequence consensus can answer. This is the same
refusal the paralog axis makes for a singleton cluster, and a clear here is curated by
someone who can speak for the rest, with a recorded reason.

## Nucleic drug-type report

Reported, not judged, and it moves no verdict:

- **GC fraction**
- **CpG count**, with every position
- **Longest G run**, with its start

## The oligo bar

The nucleic bar is these two axes and nothing else. With a reference supplied, off-target
hybridization clears from a measurement; innate immune is curated, exactly as paralog is for
a protein. Without the reference the bar cannot be reached even with the curated clear in
place — which is the point: it is reachable, not given away.

## Run-level roll-up

All of these axes apply one rule across a run: one design hitting makes the axis a hit and
that design is named; every design clear makes it clear; anything else stays untested
carrying the first undecided design's own reason. A hit's reported measurement is the worst
of the hits, so a summary cannot read milder than the designs it summarises.
