# Production report 7 — Evidence and provenance

**Program:** `Humanitys-Last-Disease`, the disease-binder design flow built on RFD4-Proteina.
**Scope:** what a stored verdict records about itself, and why each field is there.
**Status:** implemented and tested. Not medically verified.

## Where the code lives

Paths are relative to the repository root of the RFD4-Proteina fork carrying the disease
flow, which is not in this repository yet.

| Component | Path |
|---|---|
| Screen record | `src/rfproteina/disease/screen.py` |
| Declared criteria reader | `src/rfproteina/disease/refold.py` |
| Code identity | `src/rfproteina/disease/provenance.py` |
| Research traces | `src/rfproteina/disease/trace.py` |
| Local archive and pre-upload scrubbing | `src/rfproteina/disease/archive.py` |
| Scrubbing and relative paths | `src/rfproteina/disease/publish.py` |
| Shape guard | `tests/test_disease_screen_shape.py` |
| Reader reference | `docs/reference/screen-inputs.md` |

## The claim this report is about

A verdict that cannot be checked against its own inputs is not evidence. The screen record
is built so that, months later and on another machine, a reader can answer: *what was
measured, against which numbers, which axes had to clear, from which files, by which code —
and if the verdict changed, what changed with it.*

## The record declares its own shape

Every screen carries `screen_version`. It is **3**.

The point of the number is that a reader can tell which record they hold rather than
inferring it from which keys happen to be present. That only works if the number moves when
the shape does — and it once did not: the record gained roughly fourteen top-level keys
while the version stayed at 1, so records of two different shapes both claimed to be shape 1.

Raising it is therefore **no longer left to memory**. The code declares its own key sets
(19 always present, 12 present only when there is something to say), and a test pins both
sets as literals *beside* the version number. Adding a key fails that test, and the fix puts
the version in front of whoever is editing it.

Optional keys are **absent rather than null**, so a reader can tell "this run has no ligand
panel" from "this run has an empty one".

## What a verdict rests on

### `cuts` — every number it was taken against

GRAVY and instability maxima, the clash distance, the four MHC rank and IC50 cuts, the ADA
median-rank maximum, the ligand panel cut, and the three plasma cuts (null when none was
named).

A verdict is only reproducible against the cuts that produced it: edit one of those
constants later and every stored screen silently changes meaning. The recorded values are
**the constants themselves, not copies**, so what is written cannot drift from what was
applied.

### `bar_axes` — which axes had to clear

The bar's *name* was stored from the beginning, and that is not enough. Add an axis to a
bar's definition and every stored `screened_clear` starts claiming more than it was ever
checked for; remove one and it claims less. The axis list is read from the bar table at the
moment of the verdict.

### `metrics_file` and `metrics_blocker` — which table, or why none

The primary per-design metrics table wins; a re-evaluation table stands in when it is
absent. Two verdicts on one run can therefore rest on different measurements, and only this
field says which.

That table is the one input that moves a verdict, so it is read strictly. Seven named
outcomes cover an absent file, a header with no rows, an unreadable file, one past the size
bound, a repeated column, and a row wider or narrower than its own header. The last three
are **refusals, not failures to read**:

- A header that repeats a column this screen reads leaves the verdict resting on whichever
  copy the CSV reader kept — the last one — so a `False` in the first is invisible.
- A row wider or narrower than its header is misaligned, and a flag is then read under
  another column's name. A short row is also what an unterminated quote leaves behind,
  having swallowed the rows after it — so a failing design can disappear and leave the
  passes behind it looking like the whole run.

A refused table measures nothing: every flag reads absent and the call stays `untested`.

### `row_blocker` — which row measured this design

Taking the first matching row was safe while a run wrote one row per design, and wrong as
soon as two job tables were concatenated and a re-run's rows sat beside the ones they
replaced. Rows that name a design *and agree about it* are still that design's row — an
exact duplicate loses no verdict. Rows that **disagree** are refused, because nothing can
tell which one measured the file on disk.

### `chain_blocker` — whose chain was measured

The chain of interest used to be the first non-empty cell in the table, and a design whose
own row was silent inherited it. On a table naming two chains, every unnamed design was
measured against a chain it was never built for. The fallback now stands in only while the
table is of one mind.

### `covered_examples` — which designs a supplied table is a scan of

A supplied predictor table is **one scan**, and a run holds many designs. The table is
validated against the best-matching design — right for catching an inherited table — and
the verdict was then recorded for the whole run with nothing saying whose chain it came
from. The record now names which designs the table covers and which it does not.

**This records the scope of the claim; it does not narrow the claim.** Whether a clear taken
from one design should stand for designs that were never scored is a decision about what the
verdict means, and it is left to whoever runs the screen.

### `inputs` — every file the verdict rests on

Each predictor table, the ligand panel, the co-evolution notes, the solubility scores, the
curated evidence: name, size and a short digest, **whether or not an axis read it**. A table
that was present and refused is as much part of the record as one that decided an axis.

A folder holding more than 64 files is past what one run supplies, and the list is then
withheld whole rather than cut at the limit — a truncated list reads as a complete one, so
every file past the cut would look absent and the next re-screen would report them all as
removed: a change that never happened.

### `code` — which code applied all of it

The commit this checkout sat on, read out of the repository metadata **without running
git**, so a screen starts no subprocess. `tree_state` is always `not_checked` and says so:
telling a clean tree from a dirty one means asking git, and a commit recorded as though it
described the working tree would be worse than one that admits it is only the base that tree
sat on. A copy with no repository metadata records `not_recorded` rather than inventing a
commit.

It is the same helper the research traces use — shared, not copied, because two copies are
how two records come to disagree about one checkout.

### `replaced` and `changed_inputs` — what the verdict used to be

A re-screen overwrites the file outright, so the verdict it replaced left no trace: a table
dropped in afterwards could turn a hit into a clear and nothing on disk remembered the hit.
`replaced` records the previous call and a digest of it, **only when the verdict actually
changed** — which is also what keeps screening the same run twice byte-identical.

`changed_inputs` names each supplied file as added, removed or edited. Which input moved is
the part a reader can act on: a verdict that moved because a predictor table was swapped is
a different event from one that moved with nothing on disk touched — and if nothing moved,
the `code` field above is what it is pointing at.

### `curated` — what a declared state rests on

Some axes have no measured clear at all. For those, a curated clear is the *only* clear, and
a verdict resting on one that recorded no reason cannot be checked by anyone who was not in
the room. A curated entry may carry a `reason` and a `source`; one with neither records
`basis_not_stated`, because a curated call with nothing behind it is a fact about the
record, not an absence to pass over.

A field that is not one of the permitted three is refused by name rather than ignored — a
misspelled `reason` silently dropped leaves exactly the record the field was added to
prevent.

## Naming every silence

Across every axis, a check that could not decide records **why**, from a fixed vocabulary,
and the closed page prints each one together with what to do about it. The reference
documents every blocker in a table with its meaning and its remedy.

This is the property that makes the rest trustworthy: the screen does not fill a gap with a
guess, and an absence never reads as a pass.

## Research traces

Beside the screen, a per-binder trace records how a design was produced: inputs, the
effective configuration, code identity, seed and replicate, the structure file, and the
screening verdict. A re-screen refreshes the screening block of the traces beside it, so a
trace cannot cite a verdict it contradicts.

Supplied experimental results, when there are any, are recorded as supplied — never
computed. The traces repeat the distinction this whole program is built on: **a clear screen
is a computational result; it is not experimental validation.**

## Keeping records shareable

Text saved for a later upload is passed through a scrubber that removes home directories,
user-profile paths, the checkout path and a set of credential prefixes. Paths written into
tracked files are stored relative to the repository. The local archive re-writes its own copy
through the same scrubber and replaces the structure writer's account stamp with the
format's null value.

The scrubber's limits are documented rather than assumed: it does not remove hostnames,
email addresses or bare account names, so anything genuinely leaving the machine gets a
second, explicit check.
