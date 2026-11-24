# Production report 5 — Screening: peptide addition and nucleic binders

**Program:** `Humanitys-Last-Disease`, the disease-binder design flow built on RFD4-Proteina.
**Scope:** the one axis a peptide adds to the protein set, and the three checks a nucleic
binder gets.
**Status:** implemented and tested. Not medically verified.

## Where the code lives

The paths below are relative to this checkout.

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

**Why 15 and not a seed-region 7.** Fifteen is the ≥ 15-nt perfect-complementarity width
used for antisense off-target triage, adopted as that convention. Fifteen does **not** make
a perfect match rare: 4¹⁵ is 1.07 × 10⁹ against the roughly 3 × 10⁸ bases of a human
transcriptome, so any one 15-mer is expected 0.28 times in it and a design presents
2 × (len − 14) seeds. The seed is not shorter because 4⁷ is 16 384 and every 7-mer occurs in such a
file thousands of times over — a 7-mer screen would mark every oligo and mean nothing — and
it is why a seed-region match alone is not screened here.

| Outcome | Condition |
|---|---|
| `hit` | one or more seeds occur in the reference — a pointer to read, not an off-target count |
| `clear` | none do |
| `untested` | no reference configured, unreadable, empty, too large, or **too small to support a clear** (below one million bases) |
| `untested` | the design is shorter than one seed |

A hit is a pointer into the reference for someone to read, not a count of off-targets. A
clear is a statement about one file, so the file's base count, digest and name are stored
beside it. The scan is blocked with a seed's worth of overlap carried between blocks, so a
match straddling a block boundary is still found and the whole reference is never held in
memory.

## Innate immune

The **two published TLR9 CpG motifs**, either of which is a hit: the murine consensus
purine–purine–CG–pyrimidine–pyrimidine, whose optimal hexamer is `GACGTT` (Krieg 1995), and
the optimal **human** hexamer `GTCGTT` (Hartmann 2000). `GTCGTT` is **not** an instance of
the murine consensus — its second base is a pyrimidine — and ODN built on the murine formula
show no or only weak activity in human cells, so neither rule stands for the other and the
two are never merged.

| Outcome | Condition |
|---|---|
| `hit` | the design carries either consensus hexamer; the count, the one-based positions, and the motif that fired at each are stored |
| `untested` — `cpg_not_consensus` | CpG present, none of it in either consensus hexamer; the CpG count is kept |
| `untested` — `cpg_free_not_evidence` | **no CpG at all** |

A CpG-free oligo is deliberately **not** a clear. It cannot present either motif — but
TLR9 is not the only innate sensor an oligo meets; RIG-I, TLR3, TLR7 and TLR8 and the
backbone chemistry are all outside what a sequence consensus can answer. This is the same
refusal the paralog axis makes for a singleton cluster Compara cannot resolve, and a clear
here is curated by someone who can speak for the rest, with a recorded reason.

## Nucleic drug-type report

Reported, not judged, and it moves no verdict:

- **GC fraction**
- **CpG count**, with every position
- **Longest G run**, with its start

## The oligo bar

The nucleic bar is these two axes and nothing else. With a reference supplied, off-target
hybridization clears from a measurement; innate immune is curated, as paralog is for a
protein wherever Compara cannot resolve a singleton. Without the reference the bar cannot be reached even with the curated clear in
place — which is the point: it is reachable, not given away.

## Run-level roll-up

All of these axes apply one rule across a run: one design hitting makes the axis a hit and
that design is named; every design clear makes it clear; anything else stays untested
carrying the first undecided design's own reason. A hit's reported measurement is the worst
of the hits, so a summary cannot read milder than the designs it summarises.
