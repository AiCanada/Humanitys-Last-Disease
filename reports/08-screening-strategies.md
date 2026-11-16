# Production report 8 — Screening strategies

**Program:** `Humanitys-Last-Disease`, the disease-binder design flow built on RFD4-Proteina.
**Scope:** the named sets of axes a design can be judged against, and the repository's own
declared pass criteria.
**Status:** implemented and tested. Not medically verified.

## Where the code lives

Paths are relative to the repository root of the RFD4-Proteina fork carrying the disease
flow, which is not in this repository yet. The criteria file is upstream and is present
here.

| Component | Path |
|---|---|
| Axis table, bars and the judge | `src/rfproteina/disease/combinations.py` |
| Declared criteria reader | `src/rfproteina/disease/refold.py` |
| Declared criteria | `configs/metrics/pass_criteria/all_pass_criteria.yaml` |

## What a bar is

A **bar** is a named set of axes that must *all* be clear. One axis reading `hit` makes the
call `axis_hit` whatever the bar; every axis in the bar clear makes it `screened_clear`,
unless the target is itself on the safety panel, which is recorded as `target_is_panel`
instead; anything else stays `untested`.

Twelve bars are declared:

| Kind | Default | Also available |
|---|---|---|
| protein | `sequence` — 6 axes | `plasma` (3), `all` (9) |
| peptide | `sequence` — 7 axes | `plasma` (3), `all` (10) |
| ligand | `coverage` — 3 axes | `strict` (8), `plasma` (3), `all` (13) |
| nucleic | `oligo` — 2 axes | `all` (2) |

A bar can be chosen per screen, by name. Until recently it could not: the judge had always
accepted a bar argument and nothing ever passed one, so only each kind's default was
reachable and eight of the twelve were declared but unselectable. A bar a kind does not have
is refused **by name, with the valid ones listed, before anything is measured** — a
misspelling should not cost a proteome scan first.

### Bars are not ordered from weak to strong

This is the part worth stating plainly, because the names invite the opposite reading.
`plasma` is three axes describing the *target* and says nothing whatever about
immunogenicity. A protein screened on `plasma` can therefore reach `screened_clear` with
every sequence axis untested.

That is exactly why the stored verdict records **`bar_axes`**, the axis list itself, beside
the bar's name. A name that reads reassuring is not evidence; the list is.

## Axes are fixed by the trained model

The axis table has 22 entries and the model's class-conditioning dimension is twice that.
The dimension is baked into the trained checkpoint, so **an axis cannot be added to the
table** without retraining. Every check added to this program since has therefore been
added either as a report block — which records numbers and moves no verdict — or as a
measurement for an axis that was already declared and previously unimplemented.

All 22 axes now have a measurement path, and all 12 bars are reachable by measurement. One
axis refuses a measured clear by design — innate immune (report 5) — and clears only by a
curated, reasoned declaration. Paralog's UniRef50 clustering refuses one too, but when the
caller names an identity cut the singleton goes to Ensembl Compara, which can clear it from
a measurement (report 4); without a cut it still clears only by declaration.

## The repository's declared pass criteria

Separately from the bars, the repository declares **eleven** named pass criteria, each a set
of metric columns with a bound on every one. The evaluation program evaluates them and
writes a per-criterion pass column into the per-design metrics table.

| Criterion | Bounds |
|---|---|
| `unconditional` | 1 |
| `ppi` | 3 |
| `antibody` | 2 |
| `dna` | 2 |
| `motif_scaffold` | 3 |
| `internal_coords` | 3 (borrowed from `motif_scaffold`) |
| `sm_binder` | 4 |
| `ame` | 4 |
| `ame_no_ligand` | 2 |
| `disco` | 2 |
| `developability` | 4 |

**This screen read one of them.** The interface criterion became a recorded flag, and a
pre-oracle filter flag — not one of the eleven — was read directly; the other ten were
evaluated, written into the file this screen opens, and discarded.

`developability` is the pointed one: it bounds Rosetta SAP and hydrophobic exposed surface
area *alongside* the same GRAVY and instability index this screen recomputes from the
sequence. The screen was recomputing half of a criterion while throwing away the whole of
it.

A bare interface `false` also said nothing about **which** of its three bounds was missed —
a predicted TM-score floor, an interface predicted-aligned-error ceiling, and an aligned
binder RMSD ceiling decide very different things about a design, and a reader choosing
between re-folding and redesigning needs to know which.

### What is recorded now

Every declared criterion, with its stored verdict, each bound, the worst value the run
reached against that bound, whether it held, and which bounds were missed.

Three properties of that record:

- **The criteria and their cuts are read from the config file at screen time, never copied
  into code.** A cut written into the module would become a second opinion about the same
  bar the moment the config changed.
- One criterion borrows another's bounds by reference in the config; that reference is
  followed once, by name. Any other form of reference is reported as unresolved rather than
  parsed. Two bounds on one column, or a spec with no recognised cut, is recorded as
  unsupported and claims nothing. A reduction over several columns is read for its single cut
  against the reduction's label, which is not a metrics column, so it reports no measurement
  and claims nothing either.
- The reported value is the **worst** any design in the run reached, so the summary cannot
  read milder than the designs it summarises. A criterion fails if any design failed it: a
  run is not a passer because one of its designs was.

A criterion whose pass column is absent was not evaluated. On a run with no refolding oracle
that is all eleven, so the reason is given **once** rather than eleven times — but every
criterion still carries its bounds, so you can see what would have to hold before paying for
an oracle run. A config that cannot be read at all is a different silence and is named
differently.

### None of it moves the call

The interface flag has always been recorded without changing the verdict, and the other ten
are recorded on the same terms. They are the **evaluation's** verdict, not this screen's,
and a screen that silently adopted ten new ways to fail would change the meaning of every
stored call that had none of them.

## Reading a verdict correctly

A `screened_clear` means: every axis in the named bar was clear, against the inputs
supplied, under the cuts recorded, as applied by the code commit recorded. The record
carries all four so the sentence can be checked rather than trusted.

It does not mean the design binds, is safe, or is developable. It is a computational result,
and it is not experimental validation.
