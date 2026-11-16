# Production report 6 — Screening: ligand binders

**Program:** `Humanitys-Last-Disease`, the disease-binder design flow built on RFD4-Proteina.
**Scope:** the ten-protein safety panel and the one measurement taken from the structure.
**Status:** implemented and tested. Not medically verified. The panel reads a table you
supply; this program runs no toxicology predictor.

## Where the code lives

Paths are relative to the repository root of the RFD4-Proteina fork carrying the disease
flow, which is not in this repository yet.

| Component | Path |
|---|---|
| Panel reader | `src/rfproteina/disease/toxicology.py` |
| Axis list, cuts and bars | `src/rfproteina/disease/combinations.py` |
| Formula weight | `src/rfproteina/disease/modality.py` |

## The panel

Ten proteins, each its own axis:

| Axis | Protein |
|---|---|
| `herg` | hERG |
| `nav1_5` | Nav1.5 |
| `cyp3a4` | CYP3A4 |
| `cyp2d6` | CYP2D6 |
| `cyp2c9` | CYP2C9 |
| `cyp1a2` | CYP1A2 |
| `cyp2c19` | CYP2C19 |
| `pgp` | P-gp |
| `bcrp` | BCRP |
| `bsep` | BSEP |

Each reading is a **concentration in nM**, taken from `toxicology.tsv` or `toxicology.csv`
in the run's output directory.

## Which way the cut runs

**A reading at or below the cut is a `hit`. A higher reading is `clear`.**

This is the direction that matters and the one a reader can invert by accident: inhibition
at a *low* concentration is the liability, so a small number is the bad number.

Every cut in the table is **10 000 nM (10 µM)**. That is a **program convention, not a
published threshold**, and the program says so: each record stores the cut it was judged
against together with a `threshold_source` field naming where that cut came from. A reader
can therefore tell a convention from a citation without reading the source.

A missing file, or a missing cell for one protein, leaves that axis `untested`. It does not
borrow a sibling protein's number.

### Bounded readings

A cell that bounds a value rather than stating one — a greater-than or less-than reading —
is resolved only when the bound falls clear of the cut. A bound that straddles the cut is
recorded as not decisive rather than rounded to a side. A cell that is not a non-negative
nM reading is refused as such.

## Every refusal is loud

This is the design principle of the module, in its own words: *"a wrong clear on a safety
panel is the one answer this screen must never produce."*

A table that cannot be read **in full** is refused **in full**. The shapes that break a
reader are precisely the ones that move one protein's number onto another protein's name:

- a misaligned row,
- a cell that swallowed the end of its row,
- a concentration attributed to no protein, or to two of them.

Where the table parses but a single cell does not, that axis stays `untested` rather than
taking a neighbouring value — unless another cell for that same axis already measured a hit,
which stands, because an unreadable sibling cell may not delete a liability the table
measured. The carve-out runs one way only: a `clear` beside a spoiled sibling is withheld.
Where the file itself is unreadable, nothing on the panel is judged.

## The bars

| Bar | Axes | Meaning |
|---|---|---|
| `coverage` *(default)* | hERG, P-gp, BCRP | cardiac plus the two efflux transporters |
| `strict` | hERG, the five drug–drug-interaction cytochromes, P-gp, BCRP | adds the CYPs |
| `plasma` | the three target plasma axes | a property of the target, not the ligand |
| `all` | every applicable axis | |

Neither `coverage` nor `strict` speaks for an axis left out of it. The bar a verdict was
taken against is recorded by name **and by its axis list**, so a later edit to the bar
definitions cannot silently change what a stored verdict claimed.

A ligand binder also carries the three target plasma axes, which describe the target rather
than the compound.

## Formula weight

Taken from the structure's chemical-component record. Where the record states a formula
weight, that value is used. Where it does not, the weight is **computed from the chemical
formula** when every element is known, and the record marks that it was calculated rather
than read.

Where the structure holds more than one non-polymer component, the composition is recorded
as ambiguous with the component count, rather than picking one. A quoted value keeps its
internal spaces, so a component name or formula cannot shift the weight.

## What this report does not claim

- This program **runs no toxicology predictor** and contacts no service for these numbers.
  Every panel value is one you supplied.
- The 10 µM cut is a convention of this program. It is not a regulatory or published
  threshold, and the record names it as such.
- A `screened_clear` on `coverage` says nothing about the five cytochromes, which that bar
  does not include.
- A clear panel is a computational result against a supplied table. It is not experimental
  validation.
