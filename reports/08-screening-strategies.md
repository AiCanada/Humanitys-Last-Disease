# Production report 8 — Screening strategies

**Program:** `Humanitys-Last-Disease`, the disease-binder design flow built on RFD4-Proteina.
**Scope:** the named sets of axes a design can be judged against, and the repository's own
declared pass criteria.
**Status:** implemented and tested. Not medically verified.

## Where the code lives

The paths below are relative to this checkout.

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

The program exposes 22 axes and 12 selectable bars. Both nucleic bars include innate
immune response. A measured motif can flag that axis, while a clear requires a documented
curated reason. Consequently, neither nucleic bar can be cleared entirely by measurement.

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

### Reading a criterion

The report counts applicable designs and records how many have a determinate result.
When all applicable rows pass, the criterion passes. One measured failure establishes a
failure. Incomplete rows leave the remaining coverage unresolved. The pre-oracle filter
and the final screening bar have their own results.

Each bound lists the least favorable value over applicable designs. Multiple bounds
on one metric appear separately. The evaluator and reader share the implementation of
per-design minimum, maximum and mean reductions and require every operand to be finite.
The comparison uses the checkout's default thresholds. The stored flag records the
run's decision, including any overrides it used.

## Reading a verdict correctly

A `screened_clear` means: every axis in the named bar was clear, against the inputs
supplied, under the cuts recorded, as applied by the code commit recorded. The record
carries all four so the sentence can be checked rather than trusted.

It does not mean the design binds, is safe, or is developable. It is a computational result,
and it is not experimental validation.
