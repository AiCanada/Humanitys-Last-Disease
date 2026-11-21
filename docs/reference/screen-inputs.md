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
- [How a verdict is reached](#how-a-verdict-is-reached)
- [Class I and class II predictor tables](#class-i-and-class-ii-predictor-tables)
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
| `solubility.tsv` | supplied solubility and aggregation scores, reported and not judged |
| `paralog_coevolution.tsv` | a co-evolution note stored on `paralog` |
| `axis_evidence.json` | any axis, as a curated judgement |

## How a verdict is reached

A design is screened against a **bar** — a named set of axes that must *all* be clear. For a
protein binder that is `immunogenicity`, `proteome_offtarget`, `mhc_i`, `mhc_ii`, `ada` and
`paralog`. For a ligand binder the default bar is coverage: `herg`, `pgp`, and `bcrp`.
One axis reading `hit` makes the call `axis_hit` regardless of the bar; every axis
clear makes it `screened_clear`; anything else stays `untested`.

Every screen records `screen_version`, the shape of the record. It is **3**. Version 3
adds `refold`, every declared pass criterion with its bounds and which were missed.
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

`metrics_file` names the per-design table the numbers came from, relative to the
run. The primary file wins and `per_example_metrics_reeval.csv` stands in when it
is absent, so two verdicts on one run can rest on different measurements, and
only this says which.

That table is the one input that moves the verdict, so it is read strictly and
`metrics_blocker` names why it yielded no rows: `metrics_not_configured`,
`metrics_not_reported` (a header with nothing under it), `metrics_unreadable`,
`metrics_too_large` (past 16 MiB), `metrics_columns_repeated`,
`metrics_row_too_long` or `metrics_row_too_short`. The last three are refusals
rather than failures to read. A header that repeats a column this screen reads
leaves the verdict resting on whichever copy `csv` kept, which is the last one,
so a `False` in the first is invisible. A row wider or narrower than its own
header is misaligned, and a flag is then read under another column's name — and
a row short of its header is also what an unterminated quote leaves behind,
having swallowed the rows after it, so a failing design can disappear and leave
the passes behind it looking like the whole run. A refused table measures
nothing: every flag stores `absent` and the call stays `untested`.

`row_blocker` sits on a design whose metrics row was not exactly one.
`metrics_row_absent` means no row named it. `metrics_row_ambiguous` means two
rows named it and disagreed about its chain, its identity to the input or its
Ca RMSD; the first used to win in silence, which is wrong as soon as two job
tables are concatenated and a re-run's rows sit beside the ones they replaced.
Rows that name a design and agree about it are still that design's row, so an
exact duplicate loses no verdict. A run whose table is missing or refused
records no `row_blocker` at all: the run-level reason says it once.

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
number in the cell.

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
resolves on the `panel` cell. Several nM cells for one axis use the lowest, and the record
stores that count as `n_readings`. A blank cell, `n.a.`, `true`, or `false` is no reading.
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

When the same shape is present, the note is one of these four:

| Same shape | Co-evolution | Other species | Note |
|---|---|---|---|
| yes | high | toxic | most likely toxic in people |
| yes | low | toxic | lowest signal of toxicity in people |
| yes | high | non-toxic | probably non-toxic in people |
| yes | low | non-toxic | lowest signal, non-toxic in people |

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
and Leu. `sequence_entropy_bits` is the Shannon entropy of the residue
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

This screen read two of them. `ppi.pass` became `ppi_pass`, `pre_oracle_pass` was read
directly, and the other nine were computed, written into the file this screen opens, and
discarded. `developability` is the pointed one: it bounds Rosetta SAP at 50 and
hydrophobic exposed SASA at 1500 alongside the same GRAVY and instability index this
screen recomputes from the sequence, so the screen was recomputing half of a criterion
while throwing away the whole of it.

A bare `ppi_pass: false` also said nothing about **which** bound was missed. `ptm_coi` at
80, `min_ipae_coi` at 1.5 and the aligned binder RMSD at 2.5 decide very different things
about a design, and a reader deciding whether to re-fold or redesign needs to know which.

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

The criteria, their columns and their cuts are read from that config at screen time, never
copied into the code: a cut written into this module would become a second opinion about
the same bar the moment someone edited the config. `internal_coords` borrows
`motif_scaffold`'s thresholds through a Hydra interpolation, which is followed once by
name; any other interpolation is reported as `thresholds_not_resolved` rather than parsed.
A bound this reader does not understand — a `min_of` reduction, or two bounds on one
column — is recorded as `unsupported` and claims nothing, because reading a reduction as a
plain cut would compare the wrong number.

`measured` is the **worst** value any design in the run reached against that bound, the
same rule `rollup_axis` applies to an axis, so the report cannot read milder than the
designs it summarizes. A criterion is `fail` if any design failed it: a run is not a passer
because one of its designs was.

A criterion whose `<name>.pass` column is absent was not evaluated. On a run with no
refolding oracle that is all eleven, so the reason is given once as
`refold_blocker: "not_evaluated"` rather than eleven times — but every criterion still
carries its bounds, so you can see what would have to hold before paying for an oracle.
`criteria_not_found` means the config itself could not be read, which is a different
silence: an installed copy with no `configs/` tree.

**None of this moves the call.** `ppi_pass` has always been recorded without changing the
verdict, and the other ten are recorded on the same terms. They are the evaluation's
verdict, not this screen's, and a screen that silently adopted nine new ways to fail would
change the meaning of every stored call that had none of them.

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

Aggregation is the developability failure that ends a binder quietly: it
expresses, it aggregates, and nothing before the bench said so. CamSol,
Aggrescan3D, and Protein-Sol are not run here. Their coefficient tables are
not in this program. A supplied score for any of them, including a supplied
AGGRESCAN row, comes from `solubility.tsv`, or `solubility.csv` when no `.tsv`
is present, in the run's own output directory. The sequence AGGRESCAN described
under drug-type reports is a separate calculation and does not fill this table.

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
record saying so would read as a measurement. A design whose chain could not be
read is left out of every count.

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
| `metrics_unreadable` | The metrics table could not be read. | Check the encoding and the delimiter. |
| `metrics_too_large` | The metrics file is larger than this screen will read. | Keep the file within 16 MiB. |
| `metrics_columns_repeated` | The metrics table names a column this screen reads more than once. | Keep one copy of each column and screen again. |
| `metrics_row_too_long` | A metrics row carries more fields than its own header. | Export the table again without the stray separator. |
| `metrics_row_too_short` | A metrics row stops short of its own header. | Export the table again with every row the width of its header. |
| `metrics_row_absent` | No row of the metrics table names this design. | Check the table names this design in `example_id`. |
| `metrics_row_ambiguous` | Two metrics rows name this design and disagree about it. | Screen one job's table at a time, or drop the row that is out of date. |
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
