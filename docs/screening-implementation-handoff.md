# Screening implementation handoff

Written 2026-11-16 for an adversarial review by another model.
Everything below was run on the machine described, not inferred. Where a number
was not measured, it says so. Where a shortcut was taken, it says that too.

Nothing in this work was committed or pushed. All changes sit uncommitted in the
working tree of this checkout.

---

## 1. What this adds, in one paragraph

Four screens that the repository did not have, written to use only packages
already pinned in `pyproject.toml` (biotite, numpy, RDKit) plus the standard
library: interface quality terms computed without PyRosetta, RDKit structure
alerts for small molecules, an MHCflurry-to-table adapter that finally lets the
class I axis read a prediction instead of refusing IEDB's assay dump, and a
construct-synthesis feasibility screen. Plus one CLI to run them and 29 tests
that run in about 10 seconds on CPU.

## 2. Files added

| File | Lines | What it is |
| --- | --- | --- |
| `src/rfproteina/disease/interface_quality.py` | ~285 | Buried surface, interface residues, polar contacts, buried unsatisfied polar atoms |
| `src/rfproteina/disease/ligand_alerts.py` | ~212 | RDKit PAINS/BRENK/NIH catalogues, QED, SA score, Lipinski properties |
| `src/rfproteina/disease/mhc_tables.py` | ~185 | Class I peptide windows, MHCflurry runner, MHCflurry CSV to `mhc_i.tsv` |
| `src/rfproteina/disease/synthesis.py` | ~275 | GC windows, homopolymers, direct repeats, hairpin stems, forbidden sites |
| `scripts/screen_extras.py` | ~130 | One CLI over all four, JSON out |
| `tests/test_disease_interface_quality.py` | 8 tests | geometry behaviour and refusals |
| `tests/test_disease_ligand_alerts.py` | 7 tests | alerts, refusals, the probability/concentration line |
| `tests/test_disease_mhc_tables.py` | 6 tests | the column rename and the axis reading it |
| `tests/test_disease_synthesis.py` | 8 tests | each term, plus that every bar admits it is a suggestion |

## 3. Files changed

**None.** Every existing file is untouched. The three files that were already
dirty in the working tree before this work started
(`src/rfproteina/disease/results_actions.py`,
`tests/test_disease_board_vocabulary.py`,
`tests/test_disease_results_actions.py`) are somebody else's edits and were not
read into or out of this work.

## 4. Each screen: method, inputs, outputs, units, thresholds

### 4.1 Interface quality (`interface_quality.py`)

* **Why it exists.** `configs/metrics/pass_criteria/all_pass_criteria.yaml`
  judges an interface on refolding RMSD, ipTM, ipAE and binder RMSD. It carries
  no shape complementarity, no buried surface area, no interface residue count
  and no unsatisfied hydrogen bond term. Every published binder filter stack
  carries those. This is the gap.
* **Input.** A CIF or PDB holding the complex, a binder chain id, optionally a
  list of target chain ids (default: every other chain).
* **Method.** biotite `sasa()` (Shrake-Rupley, single vdW radii) on the complex
  and on each side alone; per-atom differences summed per residue; heavy-atom
  distance matrix for contacts. Hydrogens and waters are dropped first. No
  PyRosetta, no FreeSASA, no network.
* **Outputs and units.** `dsasa_total_a2`, `dsasa_binder_a2`,
  `dsasa_target_a2` in square angstrom. `interface_residues_binder` and
  `interface_residues_target` as counts of residues burying more than 1.0 A^2.
  `interface_polar_contacts` and `interface_atom_contacts` as counts of atom
  pairs within 3.5 A and 5.0 A. `buried_unsatisfied_polar_atoms` as a count.
* **Thresholds.** All in `REFERENCE_THRESHOLDS`, each with a `source` string.
  Read from BindCraft's `settings_filters/default_filters.json` on 2026-11-16:
  interface residues >= 7, interface hydrogen bonds >= 3, unsatisfied hydrogen
  bonds <= 4. **These are that project's defaults, not published values, and
  they were fitted to Rosetta's definitions, not to the proxies here.** The
  dSASA entry deliberately has no cut, because BindCraft's `Average_dSASA`
  default of 1 is a fraction of a maximum, not an absolute area, and carrying
  it across would be inventing a number.
* **What is NOT computed.** Lawrence-Colman shape complementarity. It needs a
  molecular-surface dot algorithm the repository does not carry and that no
  verified free Python implementation was found for in this work. The field is
  emitted as `null` with a reason string rather than approximated, because any
  number under that name would be read as that statistic.

### 4.2 Ligand alerts (`ligand_alerts.py`)

* **Input.** One SMILES string.
* **Method.** RDKit `FilterCatalog` with PAINS_A, PAINS_B, PAINS_C, BRENK and
  NIH; RDKit descriptors; `QED.qed`; the contributed `sascorer` when importable.
* **Outputs and units.** Alert list with catalogue and name; `mol_weight` in Da;
  `clogp` unitless; `tpsa` in A^2; donor/acceptor/rotatable/heavy-atom counts;
  `qed` on 0-1; `sa_score` on the Ertl 1-10 scale, or `null` with
  `sa_score_reason` when the contributed scorer is not importable.
* **Thresholds.** Lipinski 1997 for MW, ClogP, donors and acceptors, cited as
  such. QED >= 0.5 and SA <= 6 are marked **suggested default pending
  calibration**: Bickerton 2012 and Ertl 2009 define the scales and publish no
  pass mark.
* **The line held on purpose.** No predicted hERG or other classifier
  probability is produced here, because `combinations.py` judges the ligand
  safety panel in nanomolar (`HIT_NM = 10_000.0`) and a probability is not a
  concentration. A test
  (`test_no_probability_is_dressed_up_as_a_concentration`) asserts no `*_nm`
  key exists in this module's record.

### 4.3 MHC class I adapter (`mhc_tables.py`)

* **The problem it solves.** `tables.py` can download IEDB
  `mhc_ligand_full.tsv` (9.56 GB, on disk already) and the registry marks it
  `handoff="none"`: it is assay records about other people's peptides, not a
  prediction over this design. Worse, dropping it beside a run under the
  `mhc_i` stem *displaces* the computation that would otherwise run. The class
  I axis therefore stays untested no matter how much IEDB is on disk. The
  missing piece was never code, it was a predictor.
* **What this does.** Cuts 8/9/10/11-mer windows (dropping any window with a
  non-standard residue); shells out to `mhcflurry-predict` **only if it is
  already on PATH**; renames MHCflurry's columns onto the ones `liability.py`
  reads.
* **The rename, explicitly.** `mhcflurry_affinity` -> `ic50_nm`,
  `mhcflurry_affinity_percentile` -> `percentile_rank`, and a literal `unit`
  column of `nM`. This matters: `liability.py` takes a concentration only from
  `ic50_nm`, `affinity_nm`, `aff_nm`, or a bare `ic50` whose row declares nM.
  **A reviewer should check this rename against both sides.** If MHCflurry ever
  changes its output column names or its percentile convention, this silently
  produces a wrong-scaled rank, which would read as a clear.
* **Thresholds.** None added. The class I cuts (rank <= 1 or IC50 <= 500 nM) are
  already in `liability.py` as IEDB's published convention and were not touched.
* **Install, not shipped.** `pip install mhcflurry` (Apache-2.0) then
  `mhcflurry-downloads fetch models_class1_presentation`. No weights are
  vendored and nothing is downloaded by this code. MHCflurry is **not**
  installed in the current venv; `predictor_available()` returns `False` there
  today, verified.

### 4.4 Construct synthesis feasibility (`synthesis.py`)

* **Not a duplicate of `oligo.py`.** That module judges a nucleic *design*:
  whole-sequence GC, CpG count, longest G run, off-target hybridisation on a
  supplied transcriptome. This asks the separate, vendor-facing question:
  sliding-window local GC, longest homopolymer of any base, longest exact
  direct repeat, longest reverse-complement hairpin stem, and caller-supplied
  forbidden sites.
* **Outputs and units.** Fractions for GC, base counts for everything else;
  1-based positions.
* **Thresholds.** Every single one is marked **suggested default pending
  calibration**, and a test enforces that wording. No vendor specification was
  read for this module, so nothing here is cited to one. **Do not gate on these
  until someone checks them against the vendor actually being used.**
* **Known behaviour worth reviewing.** The hairpin scan reports a stem in any
  self-complementary repeat, for example `ATGCATGCATGC...`. That is correct for
  what it measures and may still be noisy in practice. It is an exact
  reverse-complement match, **not** a folding free energy; a real hairpin call
  wants ViennaRNA or NUPACK.

## 5. Data: what is on disk and what each file feeds

The data directory (`$DATA_PATH`) holds 52,179 files, 17.231 GB. The download
registry's own report (`scripts/fetch_screening_tables.py --have`, run
2026-11-16 under WSL) says **all five registry sources are already present**.
Nothing needed downloading, so nothing was downloaded.

| File in `screening_tables\` | Size reported by the tool | Registry handoff | What consumes it |
| --- | --- | --- | --- |
| `mhc_ligand_full.tsv` | 9.56 GB | `none` | Nothing. Cannot feed the `mhc_i` axis and displaces it if renamed. Use section 4.3 instead. |
| `tcell_full_v3.tsv` | 1.26 GB | `none` | Nothing. Same hazard for `immunogenicity`. |
| `aggregated_ada_values.csv` | 9.09 MB | `none` | Nothing. Same hazard for `ada`. |
| `blood_pea_disease_de.tsv` | 63.29 MB | `convert` | `plasma_fold` axis, via `plasma_fold_from_hpa`. CC BY-SA 3.0. |
| `UP000005640_9606.fasta.gz` | 7.38 MB | `operator` | `proteome_offtarget` axis. Read where it lies; the tool confirms "off-target axis can read UP000005640_9606.fasta.gz where it lies (not staged)", so no `HUMAN_PROTEOME` edit was needed. CC BY 4.0. |

Checksums were not computed: the two large files are 9.56 GB and 1.26 GB and
hashing them would have taken longer than the rest of this work. A reviewer who
needs provenance should hash them and record it. **This is a real gap.**

### 5.1 A staging result a reviewer should know about

Run on a scratch directory in `/tmp`, never in the repo:

* `--stage /tmp/hld_stage --disease 'Breast cancer'` (no gene narrowing) ->
  `SHORT plasma_fold conversion_unusable:bytes_over_reader_limit:1605183>1048576`.
  The whole-disease fold table is 1.6 MB against a 1,048,576-byte reader limit,
  so it is written but would be refused at read time.
* `--stage /tmp/hld_stage2 --disease 'Breast cancer' --gene ERBB2` ->
  `staged plasma_fold plasma_fold.tsv`, 456 bytes, 5 rows, 1 gene, all with a
  UniProt accession.

So the behaviour is working as designed and the operating instruction is:
**always pass `--gene` when staging the fold table.** It is worth asking whether
the no-gene path should refuse earlier and say that, rather than writing a file
it knows the reader will reject.

## 6. Tests: what was actually run

All run under WSL Ubuntu against the project's Linux venv
(`.venv`, Python 3.12.10). These are real outputs, not
expectations.

```
./.venv/bin/python -m pytest tests/test_disease_interface_quality.py \
    tests/test_disease_ligand_alerts.py tests/test_disease_mhc_tables.py \
    tests/test_disease_synthesis.py -q
=> 29 passed in 10.34s
```

Regression check on the neighbouring suites that touch the same readers:

```
./.venv/bin/python -m pytest tests/test_disease_liability.py tests/test_disease_tables.py \
    tests/test_disease_table_staging.py tests/test_disease_toxicology.py \
    tests/test_disease_solubility.py -q
=> 210 passed in 215.52s
```

Lint:

```
./.venv/bin/python -m ruff check <the nine new files>
=> All checks passed!
```

CLI smoke tests, all four subcommands, real output:

* `interface /tmp/iface.pdb --binder A` on a hand-built two-chain PDB ->
  dSASA 192.998 A^2, 3 interface residues each side, 3 polar contacts, 46 atom
  contacts, 0 buried unsatisfied polar atoms.
* `ligand 'CC(=O)Oc1ccccc1C(=O)O'` (aspirin) -> one BRENK alert
  (`phenol_ester`), MW 180.159, ClogP 1.31, QED 0.5501, SA 1.58.
* `synthesis ATGGCCGGTCTCAAAAAAAAAAAAGGTCTCTTAGC --forbidden BsaI=GGTCTC` ->
  12-base A homopolymer, BsaI found at bases 7 and 25.
* `mhc-windows MKTIIALSYIFCLVFA --widths 9` -> 8 windows,
  `predictor_available: false`.

Three test failures were found and fixed during this work, all of them real:
RDKit's `FilterCatalogEntry` has no `HasProp` (fixed to use `GetPropList`); a
hairpin position assertion was wrong because two stems open in that test
sequence and the first is reported; and an `ATGC` repeat is a bad negative
control for hairpins because it is self-complementary.

## 7. What was deliberately NOT done, and why

1. **No existing file was modified.** The brief asked for upgrades to existing
   screening code. Every candidate upgrade turned out to need either a model
   weight, a licence, or a calibration decision that is not mine to make:
   * `configs/metrics/refolding_oracles/af3.yaml` carries a commented-out
     `oracle_off_target` block. Uncommenting it needs real
     `off_target_paths` pointing at paralog structures that do not exist on
     this disk. Wiring it to invented paths would break the config.
   * Swapping the refolding oracle to Chai-1, OpenFold3 or Boltz-2 changes what
     every existing pass criterion is calibrated against. That is a program
     decision with a validation cost, not a config edit.
   * ThermoMPNN for Rosetta ddG and ImmuneBuilder for IgFold are both the right
     direction and both need weights fetched and a numerical comparison against
     the current terms before anything is switched. Not done, not pretended.
2. **No encumbered tool was wired in.** IgFold, FoldX, SignalP, DiscoTope,
   BepiPred, NetMHCIIpan, MixMHC2pred and ColabDesign are all absent from this
   work.
3. **No PoseBusters wrapper**, even though `posebusters` 0.6.5 is installed.
   PoseBusters already has a working CLI and Python API; wrapping it would add
   a layer without adding a check. The honest recommendation is to call it
   directly in the pose pipeline.
4. **No Foldseek, MMseqs2, Commec or SecureDNA stage.** All four need external
   binaries and, in the biosecurity case, a database that is gated. A stub that
   cannot run is worse than an explicit absence.
5. **No BayeshERG.** It needs a repo clone and weights, and its output is a
   probability, which section 4.2 explains cannot be mixed into the nanomolar
   panel.
6. **No DnaChisel.** `synthesis.py` scores; DnaChisel (MIT) *fixes*. It is the
   natural next step and would slot in beside `screen_sequence`.
7. **No wiring into `screen.py`.** This is the biggest judgement call here and
   a reviewer should push on it. `screen.py` is 163 KB with a large test suite
   and its axis vocabulary (`untested` / `hit` / `clear`, supplied-table-wins
   precedence, bars per binder kind) is carefully specified. Adding four axes to
   it blind, with thresholds nobody has calibrated, would risk moving existing
   verdicts. The four screens are therefore standalone and reportable today,
   and the wiring is a deliberate, named follow-up.

## 8. Weaknesses, assumptions and shortcuts: attack these first

1. **The polar-contact and unsatisfied-polar proxies are not hydrogen bonds.**
   They are heavy-atom N/O distance tests, because designed CIFs carry no
   hydrogens. The BindCraft thresholds beside them were fitted to Rosetta's
   definitions. **Numbers from the two will not agree, and nothing calibrates
   the offset.** This is the single most likely place for a wrong call.
2. **Shape complementarity is missing**, which is the term BindCraft weighs
   most heavily among the interface filters.
3. **`buried_unsatisfied_polar_atoms` excludes same-residue partners only.** It
   does not know about covalent bonding, backbone geometry, or which polar
   atoms are donors versus acceptors. A backbone carbonyl satisfied by its own
   helix neighbour is handled; an atom satisfied by geometry that distance alone
   cannot see is not.
4. **`biotite.structure.sasa` with `vdw_radii="Single"`** is an element-level
   radius set. Results will differ from FreeSASA and from Rosetta.
5. **The interface distance matrix is dense**, `n_binder x n_target`. Fine for a
   binder against a domain; it will allocate badly on a very large complex. No
   neighbour list, no cell list.
6. **Direct-repeat detection is exact-match only** and grows a window one base
   at a time, so it is O(n * repeat_length) with a dict per width. No
   mismatches, no inverted repeats.
7. **Every synthesis bar is a guess.** Said three times in the code because it
   matters.
8. **The MHCflurry rename is a contract between two moving parts**, see 4.3.
9. **No checksums on the five data files**, see section 5.
10. **The MHCflurry runner passes `/dev/stdin`** to the CLI. That is POSIX-only.
    On Windows outside WSL it will fail. Given `.env.local` sets a WSL
    `DATA_PATH`, that was judged acceptable, but it is an assumption.
11. **`_catalog()` is annotated `-> object`** to satisfy ruff without importing
    RDKit at module scope. Cosmetic, but a reviewer may prefer a real type.

## 9. Questions for the reviewing model

1. Are the heavy-atom proxies in `interface_quality.py` honest enough to report
   beside BindCraft's thresholds at all, or should the thresholds be dropped
   until someone calibrates them on this program's own designs?
2. Is there a free, verifiable Lawrence-Colman shape complementarity
   implementation that could be added without PyRosetta or CCP4?
3. Should `screen.py` gain these as real axes, and if so, what should each one
   report when its input is absent: `untested`, or nothing at all?
4. Does the MHCflurry column rename in `COLUMN_MAP` match the current
   MHCflurry 2.x output exactly, including whether
   `mhcflurry_affinity_percentile` is 0-100 and not 0-1?
5. Should the fold-table staging path refuse earlier when the conversion will
   exceed the reader's byte limit, instead of writing a file the reader will
   reject?
6. Which of the deferred upgrades in section 7 is worth the validation cost
   first?

## 10. Exact commands to reproduce everything here

```bash
# from Windows PowerShell; the project runs under WSL with a Linux venv
wsl -e bash -lc "cd <checkout> && ./.venv/bin/python -m pytest \
  tests/test_disease_interface_quality.py tests/test_disease_ligand_alerts.py \
  tests/test_disease_mhc_tables.py tests/test_disease_synthesis.py -q"

wsl -e bash -lc "cd <checkout> && ./.venv/bin/python -m ruff check \
  src/rfproteina/disease/interface_quality.py src/rfproteina/disease/ligand_alerts.py \
  src/rfproteina/disease/mhc_tables.py src/rfproteina/disease/synthesis.py \
  scripts/screen_extras.py tests/test_disease_interface_quality.py \
  tests/test_disease_ligand_alerts.py tests/test_disease_mhc_tables.py \
  tests/test_disease_synthesis.py"

# data registry, read-only
wsl -e bash -lc "cd <checkout> && DATA_PATH='<data directory>' \
  ./.venv/bin/python scripts/fetch_screening_tables.py --have"

# staging, into a scratch directory, narrowed by gene
wsl -e bash -lc "cd <checkout> && DATA_PATH='<data directory>' \
  ./.venv/bin/python scripts/fetch_screening_tables.py --stage /tmp/stage --disease 'Breast cancer' --gene ERBB2"
```

---

## Worker B: novelty, off-target, biosecurity (appended 2026-11-16)

This section is append-only and does not change anything above it. Everything
here was run on a Windows PC under the WSL venv at
`.venv`. Nothing was committed or pushed; all files are
untracked/modified in the working tree.

### B.1 What this adds

Three screens the repository did not have, plus one shared helper, a CLI and
tests. Each wraps a free, open-source command-line tool the way `mhc_tables.py`
wraps MHCflurry: the tool is used **only when it is already on PATH**, this code
installs nothing and downloads nothing, and an absent tool or database yields an
`untested` record whose `blocker` names exactly what is missing. No new runtime
dependency is added to `pyproject.toml`.

### B.2 Files added (all new; no existing file edited)

| File | What it is |
| --- | --- |
| `src/rfproteina/disease/screening_tools.py` | Shared plumbing: `run_tool` (list-arg subprocess, never a shell; failure -> blocker, not a crash), FASTA read/write, `alphabet_of`, `axis_record`, state constants. |
| `src/rfproteina/disease/novelty.py` | Foldseek structural novelty: best-match TM-score per design, same-fold verdict. |
| `src/rfproteina/disease/offtarget.py` | MMseqs2 alignment off-target vs the human proteome, with on-target exclusion. Axis `proteome_offtarget_alignment`. |
| `src/rfproteina/disease/biosecurity.py` | Commec (IBBIS Common Mechanism) biosecurity stage with a conservative recommendation mapping. |
| `scripts/screen_safety.py` | One CLI over the three screens (`check`, `novelty`, `offtarget`, `biosecurity`), JSON out. |
| `configs/metrics/screening_extras/novelty.yaml` | Provenance/threshold reference. **Not** in any Hydra `defaults:`; not auto-loaded. |
| `configs/metrics/screening_extras/offtarget_alignment.yaml` | Provenance/threshold reference (not auto-loaded). |
| `configs/metrics/screening_extras/biosecurity.yaml` | Provenance/threshold reference (not auto-loaded). |
| `tests/test_disease_screening_tools.py` | 8 tests: runner, FASTA, alphabet, record shape. |
| `tests/test_disease_novelty.py` | 9 tests: parsing, verdicts, orchestration (mocked), roll-up. |
| `tests/test_disease_offtarget.py` | 10 tests: parsing, on-target exclusion, verdicts, roll-up. |
| `tests/test_disease_biosecurity.py` | 10 tests: flexible CSV parsing, mapping, orchestration (mocked). |

### B.3 Each screen: method, inputs, outputs, tools, data, licences

**Novelty (`novelty.py`).** Runs `foldseek easy-search <designs> <db> <m8> <tmp>
--alignment-type 1 --format-output query,target,fident,alntmscore,evalue,bits`.
Per design it keeps the highest-TM match. State `hit` means best TM >= 0.5 (a
known structure of this fold exists, i.e. **not novel**); `clear` means below it
(**novel**), including the no-match case. Measurement = best TM-score so a
roll-up reads the least-novel design.
- Tool: Foldseek, **GPL-3.0**. Install: bioconda or static binary from
  foldseek.com (free). Not installed in the venv today (`check` reports false).
- Databases (CC-BY-4.0, operator-fetched with `foldseek databases ...`,
  download sizes read from foldseek.steineggerlab.workers.dev on 2026-11-16, not
  extracted): PDB `pdb100.tar.gz` 2.2 GB; Alphafold/Swiss-Prot 1.5 GB;
  Alphafold/Proteome 1.7 GB; **Alphafold/UniProt50 `afdb50.tar.gz` 114.1 GB
  (large â€” do not fetch without saying so)**. The code downloads none of these.

**Off-target (`offtarget.py`).** Runs `mmseqs easy-search <designs.fasta>
<proteome.fasta.gz> <m8> <tmp> -s 7.5 -e 1e-3 --format-output
query,target,fident,alnlen,qcov,tcov,evalue,bits`. This is the **alignment-based
complement** to the existing exact-9-mer `proteome_offtarget` axis in
`liability.py`; it is a **distinct axis** (`proteome_offtarget_alignment`) and
does not touch or displace the 9-mer axis. Per design it excludes any hit whose
UniProt accession is in `ignore_targets` (the intended on-target, so a binder of
a human protein is not flagged for hitting its own target), then takes the
strongest remaining hit over identity/coverage/E-value thresholds. State `hit`
if a qualifying off-target exists; measurement = its identity.
- Tool: MMseqs2, **MIT** (relicensed 2024-11). Install: bioconda or static
  binary from mmseqs.com (free). Not installed today.
- Target data: `UP000005640_9606.fasta.gz` (**7.38 MB / 7,740,211 bytes**,
  CC-BY-4.0) is already present at
  `$DATA_PATH/screening_tables/UP000005640_9606.fasta.gz`.
  MMseqs2 reads it gzipped. Nothing downloaded.
- Structure-level off-target (Foldseek vs the AFDB human proteome) is
  **documented but unavailable**: that DB (`afdb_proteome.tar.gz` 1.7 GB) is not
  on disk. `structure_offtarget_requirements()` returns exactly what to fetch.

**Biosecurity (`biosecurity.py`).** Runs `commec screen -d <db> -o <dir>
[--skip-tx] <fasta>` then `commec flag -i <dir> -o <csv>`, and parses the CSV.
Recommendation mapping (constant `RECOMMENDATION_STATE`, deliberately
conservative): `Pass`->clear, `Flag`->hit, **`Warn`->hit** (a lower-confidence
concern still needs review; for a safety gate the conservative direction is to
flag), `Error`->untested, anything unrecognised->untested (a changed vocabulary
surfaces as not-screened, never a silent pass). A record Commec does not report
on is `untested` with `no_commec_verdict`.
- Tool + data: Commec, **MIT** (software and biorisk/benign/control-list
  databases). Install `pip install commec`; fetch DBs with `commec setup -d
  <dir>` (IBBIS states the full Commec download is **under 10 GB**). The
  regulated-taxonomy step additionally needs NCBI `nr` and `core_nt` (large,
  operator-provided) and external tools (hmmer, diamond/blast, cmscan); use
  `--skip-tx` (the default here) to run on biorisk + low-concern alone. Input is
  **nucleotide** FASTA; protein designs must be back-translated first.
- Alternative considered: **SecureDNA** (securedna.org) is free hosted screening
  but sends a hashed form off-machine and needs issued keys, so it is not
  local-and-free; Commec is implemented as the local, fully-open option.

### B.4 Thresholds and provenance

| Screen | Threshold | Value | Status | Source |
| --- | --- | --- | --- | --- |
| Novelty | same-fold TM-score | 0.5 | **cited** | Xu & Zhang, Bioinformatics 26(7):889-895 (2010): same-fold TM-score > 0.5 |
| Novelty | E-value max | 1e-3 | suggested | program default; Foldseek's own default E-value is 10 |
| Off-target | identity min | 0.5 | suggested | program default; no published cross-reactivity identity cut |
| Off-target | coverage min (min of q/t) | 0.5 | suggested | program default |
| Off-target | E-value max | 1e-3 | suggested | program default |
| Biosecurity | recommendation mapping | see B.3 | suggested | conservative choice; Warn->hit |

Only the TM-score 0.5 same-fold cut is a published value. Everything else is a
program default labelled `suggested` in both the code (`*_THRESHOLDS` dicts) and
the config YAMLs, pending calibration on this program's own designs.

### B.5 Integration request for worker A (screen.py)

I did **not** edit `screen.py`, `combinations.py`, `liability.py`, any
`pass_criteria` config, or any file another worker owns. Requested wiring:

1. **Record shape is already compatible.** Every per-design record is
   `{"state","measurement","blocker", ...extras}` and each module has a
   `rollup(records)` that calls `combinations.rollup_axis` with an `untested`
   empty record. Call `rollup()` to get one axis record per screen.
2. **Register three new axes, do not reuse existing names:**
   - `structural_novelty` (novelty) â€” suggest surfacing as an **informational**
     axis, not a hard pass/fail gate, because "matches a known fold" is only bad
     for an unconditional design, not for a scaffolded/antibody campaign. If you
     do gate, gate per campaign kind.
   - `proteome_offtarget_alignment` (off-target) â€” register **alongside**, never
     in place of, the exact-9-mer `proteome_offtarget`. Pass the run's intended
     target UniProt accession(s) into `offtarget_scan(..., ignore_targets=...)`.
   - `biosecurity` (biosecurity) â€” operates on the **nucleotide construct**, so
     wire it where `oligo.py`/`synthesis.py` have the orderable sequence, not on
     the protein axis.
3. **Unavailable path.** Each module raises a typed error
   (`NoveltyUnavailableError`, `OffTargetUnavailableError`,
   `BiosecurityUnavailableError`) carrying `.blocker`; catch it and record
   `module.unavailable_axis(err.blocker)` so the axis reads `untested` with the
   reason, matching how the MHC/proteome axes behave when their inputs are
   absent.
4. **Where inputs come from.** Novelty needs generated design structures (the
   `*_generated.cif` the screen already globs). Off-target needs design
   sequences (already derived for the sequence-liability axes) plus the proteome
   path (same file the `proteome_offtarget` axis reads, or `HUMAN_PROTEOME`).
   Biosecurity needs a nucleotide FASTA of the construct.
5. The config files under `configs/metrics/screening_extras/` are
   **provenance/reference only** and are intentionally not in any `defaults:`
   list; do not add them to Hydra defaults unless you convert the screens to
   `Metric` classes (they are not, because they shell out).

### B.6 Real test and ruff output (run 2026-11-16 in the WSL venv)

```
$ ./.venv/bin/python -m ruff check <9 new files>
All checks passed!

$ ./.venv/bin/python -m pytest tests/test_disease_screening_tools.py \
    tests/test_disease_novelty.py tests/test_disease_offtarget.py \
    tests/test_disease_biosecurity.py -q
.....................................
37 passed in 3.26s
```

The external tools are not installed, so the integration tests mock `run_tool`
(and the `*_available` checks) and feed canned tool output. `check` was run live
and reports `foldseek_available/mmseqs_available/commec_available` all false,
confirming the honest unavailable path. I did not run the full repo test suite;
I ran only these four files and ruff on the nine new files.

### B.7 Weaknesses and questions for the reviewer

1. **Commec CSV columns are parsed flexibly, not from a verified real run.** The
   id/recommendation columns are matched case-insensitively against candidate
   names (`_ID_COLUMNS`, `_RECOMMENDATION_COLUMNS`). The documented
   recommendation values are Pass/Flag/Warn. **Verify the exact header spelling
   and value set against a real `commec flag` output** and tighten if needed.
2. **`Warn` -> hit is a judgement call.** It over-flags toward safety. If the
   campaign would rather triage Warn separately, change `RECOMMENDATION_STATE`.
3. **Novelty "hit" semantics are inverted from the other axes.** Here `hit`
   means "not novel / known structure exists", which is only a liability for
   some campaign kinds. This is why B.5 asks for it to be informational. Is that
   the right default?
4. **Off-target thresholds are uncalibrated.** 50% identity / 50% coverage /
   1e-3 are placeholders. What identity/coverage actually predicts cross-
   reactivity for this program's binders?
5. **On-target exclusion depends on the caller passing the right accession(s).**
   If `ignore_targets` is empty and the design is derived from a human protein,
   the self-match is reported as an off-target hit (the record flags
   `skipped_on_target=False` so this is visible, but it is still a false hit).
6. **Structure-level off-target is not implemented** (AFDB proteome DB not on
   disk); only sequence-level is. Worth adding once the 1.7 GB DB is fetched?
7. **MMseqs2 `-s 7.5` and Foldseek `--alignment-type 1` are reasonable but not
   benchmarked here** for this proteome size; a reviewer may prefer different
   sensitivity/speed settings.
8. **No checksum is enforced on the proteome FASTA**; the off-target scan trusts
   whatever is at the path.
9. **`run_tool` default timeouts** (novelty/off-target 3600 s, biosecurity
   7200 s) are guesses for batch sizes; tune for real workloads.

---

# Worker D — ligand pose validation, interaction profiling, small-molecule safety

Appended 2026-11-16. Append-only; nothing above was altered.
No existing file was edited. Nothing was committed or pushed.

## D.1 Files added (all new, all mine)

| File | What it is |
| --- | --- |
| `src/rfproteina/disease/pose_validation.py` | PoseBusters + RDKit pose validity, MMFF strain, native protein clashes |
| `src/rfproteina/disease/interactions.py` | Native (or ProLIF) interaction fingerprint; native Fnat; optional DockQ |
| `src/rfproteina/disease/ligand_safety.py` | ChEMBL measured-nM panel; guarded BayeshERG/ADMET-AI probabilities; units guard |
| `configs/metrics/pose_safety_screens.yaml` | Reference config: axes, thresholds, provenance. NOT a loaded hydra metric |
| `scripts/screen_pose_safety.py` | CLI over all three screens, JSON out (separate from screen_extras.py) |
| `tests/test_disease_pose_validation.py` | 8 tests |
| `tests/test_disease_interactions.py` | 11 tests |
| `tests/test_disease_ligand_safety.py` | 11 tests |

Dependencies: uses only packages already present in the venv — RDKit, biotite,
numpy, and `posebusters` (already pinned in `pyproject.toml` optional extras).
ProLIF, DockQ, BayeshERG, ADMET-AI and MDAnalysis are **optional**: each is
imported inside its function and raises a named `*UnavailableError` when absent,
mirroring `mhc_tables.py`'s "only if on PATH" rule. None is a hard dependency.

## D.2 Each screen: method, inputs, outputs, units, thresholds

### D.2.1 Pose validation (`pose_validation.py`)
* **Input.** A ligand pose (SDF/MOL/MOL2/PDB, or an RDKit mol with a conformer);
  optionally a protein PDB/CIF.
* **PoseBusters** (Buttenschoen et al., Chem. Sci. 2024). `mol` mode (ligand
  alone) or `dock` mode (with protein). Each check pass/fail is PoseBusters'
  own; this module's only convention is **any failed check is a hit**
  (`posebusters_failures` max 0, labelled *program convention*). The exact `mol`
  check names are pinned in `POSEBUSTERS_MOL_CHECKS` so a rename surfaces.
* **Internal strain**, MMFF94: pose energy minus relaxed-copy energy, kcal/mol.
  **Reported, not gated** (`no_cut`): PoseCheck (Harris et al. 2023) publishes no
  pass mark. Returns `None` + reason when MMFF has no parameters.
* **Protein clashes**, native: heavy-atom pairs closer than vdW sum minus a
  0.4 A tolerance (Bondi 1964 radii). *Suggested* bar of 0, no published mark.
* **Units.** strain kcal/mol; distances angstrom; clashes a count.

### D.2.2 Interaction profiling (`interactions.py`)
* **Native fingerprint** of one RDKit ligand against one protein: hydrophobic,
  hbond, pi_stacking, salt_bridge, halogen. Cutoffs are **ProLIF's documented
  defaults** (Bouysset & Fiorucci 2021) applied to **heavy atoms** — every count
  is a geometric proxy, not a confirmed interaction (a designed model carries no
  hydrogens). Interaction counts are `no_cut`: no published pass mark.
* **ProLIF backend** (`prolif_fingerprint`) is the honest engine (it protonates
  and types atoms). It is behind an import guard and raises
  `ProLIFUnavailableError` when ProLIF/MDAnalysis are absent (they are, here).
* **Complex comparison** (`compare_to_reference`): native **Fnat** — fraction of
  the reference's interface residue-residue heavy-atom contacts (within 5.0 A,
  CAPRI convention) that the model reproduces, plus fnonnat. Labelled **Fnat
  only, NOT a DockQ score**. Full DockQ (`dockq`) is behind a guard and raises
  `DockQUnavailableError` when the `DockQ` package is absent.
* **Gating.** `comparison_verdicts` has NO default Fnat bar: a caller that gates
  must pass `fnat_min`, recorded as `caller_supplied`.

### D.2.3 Ligand safety (`ligand_safety.py`)
* **The honest route.** `fetch_measured_nm` reads **measured** IC50/Ki/Kd from
  the free ChEMBL REST API (no key), in nM, aggregated as the median, with the
  ChEMBL `activity_id`s recorded. Bounded readings (`>`/`<`) are excluded and
  counted in the note. Output rows (`to_toxicology_rows`) feed `toxicology.tsv`
  directly — the format the existing `toxicology.py` already reads.
* **ChEMBL target map** (`CHEMBL_TARGETS`) covers all ten
  `combinations.LIGAND_AXES`, each verified against the ChEMBL target API on
  2026-11-16 as the Homo sapiens SINGLE PROTEIN entry: hERG CHEMBL240, Nav1.5
  CHEMBL1980, CYP3A4 CHEMBL340, CYP2D6 CHEMBL289, CYP2C9 CHEMBL3397, CYP1A2
  CHEMBL3356, CYP2C19 CHEMBL3622, P-gp CHEMBL4302, BCRP CHEMBL5393, BSEP
  CHEMBL6020. None guessed.
* **The predicted route, fenced off.** `bayesherg_probability` (BayeshERG, MIT)
  and `admet_ai_predictions` (ADMET-AI, MIT) return **probabilities**, stored
  only under `*_prob` keys with `unit="probability"`. Both are behind import
  guards and raise `SafetyBackendUnavailableError` when absent (both are, here).
* **The units line, enforced.** `assert_no_probability_in_nm` walks a record and
  raises if a probability (0-1 under a `probability` unit) sits under an `*_nm`
  key, or if a `*_prob` value is out of range. `toxicology.py`'s `HIT_NM` cut is
  **not** restated as a threshold here — it stays a 10 uM *program convention*
  owned by `combinations.PANEL_NM_SOURCE`.

## D.3 Real test and lint output (run on the WSL venv, 2026-11-16)

```
$ ./.venv/bin/python -m pytest tests/test_disease_pose_validation.py \
    tests/test_disease_interactions.py tests/test_disease_ligand_safety.py -q
.............................                                            [100%]
29 passed in 14.23s

$ ./.venv/bin/python -m ruff check <the 3 modules, the CLI, the 3 test files>
All checks passed!
```

Tests are CPU-only. PoseBusters runs for real on a built ethanol conformer; the
ChEMBL fetch is driven by an injected fake opener (no network in tests); the
optional backends are asserted to raise when absent.

## D.4 Integration request for worker A (owns screen.py)

1. **New axes.** Wire these verdict producers into `screen.py`, each emitting the
   same `{value, state, threshold, direction, threshold_source}` shape the other
   screens use:
   - `pose_validation.verdicts(validate_pose(ligand, protein))` →
     `posebusters_failures`, `protein_clashes`, `internal_strain_kcal_mol`.
   - `interactions.native_verdicts(native_fingerprint(protein, mol))` → the five
     interaction counts (all `no_cut`, report-only).
   - `interactions.comparison_verdicts(compare_to_reference(...), fnat_min)` →
     `fnat`, only when a reference complex exists; pass `fnat_min` or leave it
     `no_cut`.
2. **Inputs.** These need a ligand pose file and a protein file from a run's
   outputs. I did not guess your run-dir layout — please map the pose/protein
   paths the same way the refolding oracles locate their inputs.
3. **Absent-input policy.** Match whatever `interface_quality`/`mhc_tables`
   settle on: I return `untested` for a missing value and `no_cut` for an
   un-gated metric; confirm that is the state screen.py should surface.
4. **Safety panel.** `ligand_safety.to_toxicology_rows(...)` is designed to be
   written as `toxicology.tsv` rows the existing screen reads — it is **not** a
   new axis. Decide whether screen.py should offer an opt-in step that fetches
   ChEMBL and stages that file, or whether that stays a manual/CLI step (it is
   network I/O, so it probably should not run inside a screen pass by default).
5. **Config.** `configs/metrics/pose_safety_screens.yaml` carries every axis,
   direction, threshold and source; read it rather than re-deriving them. It is
   intentionally not in `all_metrics.yaml` defaults.

## D.5 Weaknesses and questions for the reviewer

1. **Native interaction fingerprint is a heavy-atom proxy.** No hydrogens, no
   protonation, no atom typing. Salt bridges use ligand formal charges and
   protein residue identity, not a titration at pH 7.4. Pi-stacking is centroid
   distance only — no ring-normal angle test, so edge-to-face vs face-to-face is
   not distinguished, and a 5.5 A centroid cut will admit some non-stacked pairs.
   ProLIF is the right engine; it is wired but not installed here.
2. **`internal_steric_clash` / strain overlap.** PoseBusters already flags
   internal clashes; MMFF strain and PoseBusters' `internal_energy` are related
   but not identical. Is reporting both redundant, or useful cross-checks?
3. **Fnat requires matching residue labels** between model and reference
   (`chain:resid:resname`). A renumbered or re-chained model silently scores
   0 Fnat. Should there be an alignment step, or is label-match the contract?
4. **ChEMBL aggregation is a plain median** across assay types (IC50/Ki/Kd mixed)
   and across assays of varying quality. No confidence-score filter, no
   pChEMBL-only mode, no per-assay dedup. Is the median honest enough, or should
   it be min (most conservative) or restricted to a single standard_type?
5. **ChEMBL molecule lookup is by InChIKey.** A tautomer or salt form ChEMBL
   stores under a different parent will miss. No standardisation/parent lookup.
6. **No published pass mark exists for strain, clashes, interaction counts or
   Fnat.** Every one is `suggested` or `no_cut`. These are reports until someone
   calibrates them on this program's own designs — do not gate on them blind.
7. **Optional backends are untested beyond the "raises when absent" path**,
   because ProLIF/DockQ/BayeshERG/ADMET-AI are not installed in this venv. The
   column-mapping inside `prolif_fingerprint` and `admet_ai_predictions` has not
   been exercised against real output and is marked `# pragma: no cover`.
8. **`screen_pose_safety.py safety` makes live ChEMBL calls.** It is the only
   network path; keep it out of any automated pass that must be offline.

---

# Worker A — interface quality, shape complementarity, and wiring the four screens into `screen.py`

Appended 2026-11-16. Everything below was run on a
Windows PC under the project's WSL Linux venv (`.venv`, Python 3.12.10). Numbers
are real tool output, not estimates. **Nothing was committed or pushed**; all
changes sit in the working tree (`git log -1` unchanged at `8922de1`).

## A.1 What this worker did

1. Rewrote `interface_quality.py` with a real Lawrence–Colman shape
   complementarity (Sc), a KD-tree neighbour search replacing the dense distance
   matrix, and donor/acceptor-typed buried-unsatisfied logic.
2. Wired all four new screens into the real pipeline (`screen.py`): a first-class
   `designed_interface_quality` block (plus per-sample `interface_quality`), and
   optional `synthesis`, `ligand_alerts` and `mhc_class_i` blocks. All are
   **recorded, not judged** — none feeds `judge_combination`, so existing
   verdicts are byte-for-byte preserved.
3. Calibrated the interface thresholds on the program's own 23 generated designs
   and re-labelled them with measured numbers.
4. Declared the interface-quality thresholds in the pass_criteria config tree.

## A.2 Files changed (this worker's scope only)

| File | Status | What |
| --- | --- | --- |
| `src/rfproteina/disease/interface_quality.py` | rewritten | Sc, KD-tree, donor/acceptor buried-unsat, calibrated thresholds |
| `src/rfproteina/disease/screen.py` | modified | wired the four screens in; `SCREEN_VERSION` 3→4; new required/optional keys; helper functions |
| `configs/metrics/pass_criteria/interface_quality.yaml` | added | declared interface-quality thresholds (mirror of the module, pinned by a test) |
| `tests/test_disease_interface_quality.py` | rewritten | 16 tests incl. Sc, KD-tree equivalence, donor/acceptor typing |
| `tests/test_disease_interface_quality_config.py` | added | 4 tests: config==module, and refold untouched |
| `tests/test_disease_screen_shape.py` | modified | pinned the new keys + version 4 |

Not touched: the three pre-existing dirty files, `all_pass_criteria.yaml`, and
every other worker's module.

## A.3 Shape complementarity (Lawrence & Colman 1993)

Reference: Lawrence MC, Colman PM, *Shape complementarity at protein/protein
interfaces*, J Mol Biol 1993;234(4):946–950.

Implementation (`_shape_complementarity`):
* A dot surface is built on each molecule: dots on every atom's van der Waals
  sphere (Bondi radii), kept when a 1.7 Å probe resting tangent there clears all
  other atoms of the molecule — i.e. the **Connolly contact surface**. Normals
  point radially outward.
* Interface dots are found by **burial on complexation**: an isolated-surface dot
  is an interface dot when the full complex's atoms block its probe. This is the
  parameter-free equivalent of Lawrence–Colman's peripheral-band trim.
* For each interface dot on one side, the nearest surface dot on the other side
  is found with a KD-tree; `S = n·(−n′)·exp(−0.5·d²)` (w = 0.5 Å⁻², exactly LC).
  `Sc = mean of the two per-side medians`.
* Only atoms within 8 Å of the other molecule get dots; below 10 interface dots a
  side returns `None` with a reason rather than a noisy median.

**How it differs from Rosetta's `sc`:** Rosetta/CCP4 `sc` use the full molecular
(solvent-EXCLUDED) surface *including re-entrant (toroidal/concave) patches*;
this uses the **contact portion only** (no re-entrant patches) and is built from
dots rather than an analytic surface. Consequently the absolute values run lower
than Rosetta's and below the 0.64–0.76 LC quote for natural interfaces. Ordering
is preserved; absolute scale is not comparable — see the calibration below.

## A.4 KD-tree neighbour search (item 3) and benchmark

`_contacts` now uses `scipy.spatial.cKDTree` (`sparse_distance_matrix` for all
atom contacts, `query_ball_tree` over N/O subsets for polar contacts), replacing
the dense `n_binder × n_target` matrix. Benchmark on a real complex
`disease_runs/gene_9BSB_3/.../binder_0_generated.cif` (binder A = 614 atoms,
target = 3247 atoms, 1,993,658 dense pairs), 5 runs each, counts verified
identical:

```
dense : 63.5 ms  peak_mem 127.7 MB  -> polar=4 atom=495
kdtree:  1.9 ms  peak_mem   0.3 MB  -> polar=4 atom=495
counts identical: True
speedup: 33.7x   memory reduction: 424.1x
```

## A.5 Buried-unsatisfied polar atoms, donor/acceptor typed (item 4)

`_buried_unsatisfied` now (a) restricts to **interface** atoms (buried in the
complex AND buried on complexation, `delta_sasa > 0.1 Å²`) rather than the whole
complex, which is what made the first pass count protein cores; (b) types every
N/O as donor/acceptor/both by `(residue, atom_name)` (backbone N donor, O
acceptor; sidechains from a lookup); (c) requires a **complementary** partner
(donor↔acceptor) from another residue at 2.4–3.5 Å; (d) applies a coarse
heavy-atom angle (≥ 90° to the antecedent atom, e.g. backbone N→CA, O→C) where an
antecedent is known.

**What it is not:** a Rosetta unsatisfied-H-bond count. Without hydrogens the
true donor–H–acceptor angle cannot be measured; the angle here is the heavy-atom
D···A–antecedent angle, a stand-in. Reported as total plus a donor/acceptor
split.

## A.6 Calibration (item 5) — measured on the program's 23 generated designs

`measure_interface(path, "A")` over all 23 `inference_samples/*/binder_0_generated.cif`
(binder = chain A, target = the other chain), 19.5 s for 23 structures:

| metric | min | median | max | BindCraft bar | verdict on the bar |
| --- | --- | --- | --- | --- | --- |
| dsasa_total_a2 | 1353 | 2393 | 4356 | none (fraction, not area) | no cut adopted |
| interface_residues_binder | 16 | 27 | 47 | ≥ 7 | slack; kept as floor |
| interface_polar_contacts | 4 | 10 | 18 | ≥ 3 (Rosetta hbonds) | slack; kept as floor |
| buried_unsatisfied_polar_atoms | 1 | 4 | 9 | ≤ 4 | **lands on the bar**; adopted as calibrated-compatible |
| — donors | 1 | 3 | 7 | — | — |
| — acceptors | 0 | 1 | 3 | — | — |
| shape_complementarity | 0.069 | 0.166 | 0.254 | ≥ 0.55 (Rosetta scale) | **NOT comparable**; no cut adopted |

**Key calibration finding:** this Sc scale does not match BindCraft's Rosetta
`sc`. Its median is 0.166; adopting 0.55 would fail every accepted design. The
`shape_complementarity` threshold is therefore set to **no cut** (`min: null`)
with the measured distribution recorded, rather than carrying a bar from a
different engine. Likewise `interface_residues_binder` (≥7) and
`interface_polar_contacts` (≥3) are slack against this program's designs and kept
only as floors; `buried_unsatisfied_polar_atoms` (≤4) is the one BindCraft bar
that matches the measured distribution and is adopted.

Every threshold's `source` string in `interface_quality.py` /
`interface_quality.yaml` now carries its measured number and its provenance
label (BindCraft-default, cited, or Worker-A-calibrated).

## A.7 How the wiring preserves verdicts

The `call` comes from `judge_combination(evidence.states, …)`. None of the four
new blocks touch `evidence.states`, the liability axes, or `judge_combination` —
they are recorded beside the verdict exactly as `designed_interface`, `refold`,
`solubility` and `ppi_pass` already are. Test fixtures write sequence-only CIFs
(no `_atom_site.`), so `interface_quality` short-circuits to `coordinates_absent`
and the blocks are null on them. Verdict preservation is proven by the full
existing screen suite passing unchanged (A.8).

The record shape was extended through the repo's own governance: `SCREEN_VERSION`
3→4, `designed_interface_quality` added to `SCREEN_REQUIRED_KEYS`, and
`synthesis`/`ligand_alerts`/`mhc_class_i` to `SCREEN_OPTIONAL_KEYS`, with
`tests/test_disease_screen_shape.py` updated to pin all four.

Wiring specifics:
* **interface_quality** — computed per generated CIF (binder = chain of
  interest, target = the rest), compact per-sample block, top-level
  `designed_interface_quality` repeating the first design with the thresholds
  attached once.
* **synthesis** — nucleic binders only; runs `screen_sequence` on the designed
  oligo; optional `synthesis` block with `verdicts`.
* **ligand_alerts** — reads an optional `ligand_smiles.txt` beside the run (the
  screen carries no SMILES of its own); optional `ligand_alerts` block.
* **mhc_class_i** — protein/peptide only; reports window count and
  `predictor_available` (MHCflurry is not installed, so `false`). It deliberately
  does **not** run MHCflurry or write `mhc_i.tsv`, so installing a predictor can
  never silently move a stored call; to feed the axis, drop `mhc_i.tsv` beside
  the run (the existing supplied-table path).

Real-run smoke on `disease_runs/generations/asthma/IL33/gene_9WWH_3` (protein):
`call: untested` (unchanged), `screen_version: 4`, `designed_interface_quality`
populated — dSASA 1873.7 Å², 20 interface residues, 17 polar contacts, 5 buried
unsat (2 donors, 3 acceptors), Sc 0.1765, thresholds attached; `mhc_class_i`
present (226 windows, predictor false).

## A.8 Tests actually run (real output)

```
# interface_quality + its config pin
pytest tests/test_disease_interface_quality.py tests/test_disease_interface_quality_config.py -q
=> 20 passed in 11.68s

# record shape + version pin
pytest tests/test_disease_screen_shape.py -q
=> 11 passed in 198.64s

# verdict-preservation regression (the decisive run)
pytest tests/test_disease_screen_only.py tests/test_disease_screen_evidence.py \
       tests/test_disease_screen_board.py tests/test_disease_refold.py \
       tests/test_disease_interface_quality.py -q
=> 106 passed in 209.93s

# lint
ruff check interface_quality.py screen.py + the three test files
=> All checks passed!
```

(The screen suites are slow because each `screen_generation` imports atomworks
and the torch-backed contact-surface metric; that overhead is pre-existing.)

## A.9 Weaknesses — attack these first

1. **Sc is on a contact dot surface, not the analytic molecular surface.**
   Re-entrant patches are omitted and the surface is sampled by dots
   (`density=100`), so absolute Sc is lower than Rosetta/CCP4 `sc` and than the
   LC literature values. It is a faithful *implementation of the LC formula* on a
   *different surface*, not a drop-in for Rosetta `sc`. A reviewer should decide
   whether the correct next step is the true Connolly surface (re-entrant
   patches) or accepting the scale shift and gating on the measured distribution.
2. **Sc dot density and probe are fixed** (`SC_DOT_DENSITY=100`, `SC_PROBE_A=1.7`,
   `SC_WEIGHT=0.5`). The median shifts a little with density; 100 was chosen for
   speed. No convergence study was run.
3. **Buried-unsat still has no hydrogens.** The angle filter is heavy-atom only;
   a donor whose geometry is wrong but whose heavy atoms happen to sit >90° will
   read as satisfied, and vice versa. The donor/acceptor *typing* is a fixed
   table and does not know protonation state (HIS, carboxylates, termini).
4. **interface_quality re-reads and re-SASAs each CIF** separately from
   `_interface_block` (biotite vs atomworks), so a run pays for two structure
   loads and two SASA passes per design (~1 s extra per real design). Acceptable
   for a screen; a reviewer may want one shared load.
5. **The config is a mirror, not the driver.** `REFERENCE_THRESHOLDS` in the
   module is the runtime source; `interface_quality.yaml` is pinned equal to it
   by a test. This avoids import-time file IO but means the YAML does not *drive*
   behaviour. Flip it if you prefer config-as-source.
6. **ligand_alerts only runs from a supplied `ligand_smiles.txt`.** The pipeline
   carries no SMILES, so it is dormant until an operator drops one in. mhc never
   runs the predictor by design (see A.7).
7. **Concurrency:** the sibling modules (`synthesis.py`, `ligand_alerts.py`,
   `mhc_tables.py`) were owned by other workers and edited in parallel. The
   wiring imports them lazily and records a blocker on ImportError, and was
   verified against their state at the time of writing; re-run the wiring check
   if their public APIs changed.

## A.10 Questions for the reviewing model

1. Is the contact-dot-surface Sc acceptable as a reported signal, given it is not
   on Rosetta's scale, or should the Connolly re-entrant surface be implemented
   before Sc is shown at all?
2. Should any of these blocks ever feed the call? They are recorded-not-judged
   today; promoting even one (e.g. buried-unsat ≤4) to an axis would change
   stored verdicts and needs an owner's decision.
3. Is `ligand_smiles.txt` the right input contract for ligand_alerts, or should a
   SMILES be threaded through the design spec?
4. Is the donor/acceptor table complete enough (it omits protonation states and
   nucleic-acid polar atoms, which fall through to `both`)?

---

# Worker C: antibody developability, humanness and immunogenicity

Appended 2026-11-16. Nothing above this line was changed.
Nothing in this work was committed or pushed; every file below is untracked or
an in-place edit in the working tree of this checkout.

## C.1 What this adds

The class I axis now reaches a verdict on a real design instead of staying
untested, and the antibody layer the repository had none of exists: IMGT
numbering, CDRs, chemical liability motifs with their publications, a real
humanness measurement, and a class II path that is honest about being blocked
by licences rather than quietly absent.

## C.2 Files

| File | Lines | What it is |
| --- | --- | --- |
| `src/rfproteina/disease/developability.py` | 529 | ANARCII parse, IMGT regions, liability motif scan, DeepSP/CrossAbSense readers |
| `src/rfproteina/disease/humanness.py` | ~300 | OASis identity (promb peptide set) and AbLang-2 antibody-likeness |
| `src/rfproteina/disease/immunogenicity.py` | ~300 | Class II option registry with quoted licences, backend interface, converter |
| `src/rfproteina/disease/mhc_tables.py` | +133 | Extended: predictor lookup, temp-file runner, `predict_sequence`, reference table, `contract()` |
| `scripts/screen_antibody.py` | ~120 | One CLI: `liabilities`, `humanness`, `mhc1`, `mhc2-status` |
| `configs/metrics/antibody_developability/antibody_developability.yaml` | 10.7 kB | The motifs, licences and (absent) cuts as readable data |
| `tests/test_disease_developability.py` | 16 tests | parse, regions, every motif, cysteine parity, oracle readers |
| `tests/test_disease_humanness.py` | 9 tests | identity arithmetic, the humanness/likeness line, option licences |
| `tests/test_disease_immunogenicity.py` | 12 tests | unavailable path, backend interface, converter, two real `liability.py` integrations |
| `tests/test_disease_mhc_contract.py` | 10 tests | the rescaling guard, the scale, the `/dev/stdin` regression, reference columns |
| `tests/test_disease_screening_config.py` | 7 tests | the config may not drift from the code |

Files **not** touched: `screen.py`, `configs/metrics/pass_criteria/*`,
`interface_quality.py`, `ligand_alerts.py`, `synthesis.py`, `liability.py`,
`tables.py`, `pyproject.toml`, and the three pre-existing dirty files.

## C.3 MHC class I, end to end, really run

MHCflurry 2.3 requires `pandas>=3`; this project pins pandas 2.3.3. Installing
it into `.venv` would have moved a dependency under every other screen and
every other worker running right now. It went into its own environment
instead, and `mhc_tables.predictor_command()` finds it through a new
`MHCFLURRY_PREDICT` variable, falling back to `PATH`:

```bash
uv venv --python 3.12 ~/.venvs/mhcflurry
VIRTUAL_ENV=~/.venvs/mhcflurry uv pip install --python ~/.venvs/mhcflurry/bin/python 'mhcflurry>=2.3,<2.4'
~/.venvs/mhcflurry/bin/mhcflurry downloads fetch models_class1_presentation   # 160 MB, 48 files
export MHCFLURRY_PREDICT=~/.venvs/mhcflurry/bin/mhcflurry-predict
```

Installed: mhcflurry **2.3.29**, Apache-2.0, torch-backed (no TensorFlow).
Weights: `models_class1_presentation` release 20260928, extracted to
`~/.local/share/mhcflurry/4/2.3.0/models_class1_presentation/`.

**Measured, on this machine, 2026-11-16.** Sequence: the trastuzumab-like VH
used throughout this section (120 aa).

1. `predict_sequence(VH, ["HLA-A*02:01","HLA-B*07:02"], widths=[9], directory=...)`
   cut 112 windows, wrote 224 rows to `mhc_i.tsv` in **163.5 s**.
2. `liability.measure_sequence_liability` on that directory, `binder_kind="protein"`:
   `mhc_i` -> `untested`, blocker `reference_not_configured`, 6 binding
   peptides, density 5.3571 per 100 windows. **The immunogenicity axis moved
   at the same time**: it went from `predictor_not_configured` to a computed
   Calis verdict of `hit`, method `calis_2013_class_i_9mer`, because that axis
   restricts itself to the peptides a class I table calls presented. That is
   the dead axis the first pass described, now alive.
3. The `reference_not_configured` blocker is real and was closed too.
   `predict_reference_table` scans the three tolerated proteins
   `liability.py` accepts (P02768 albumin 609 aa, P01857 IgG1 constant 399 aa,
   P0CG47 ubiquitin 229 aa; sequences fetched from UniProt by the caller, not
   by the module) and writes `mhc_i_reference.tsv`: 2138 rows in **680.9 s**.
   With it in place the same call returns:

   ```
   mhc_i: state=clear, measurement=5.3571, reference_density=9.4629,
          reference_densities={P01857: 9.4629, P02768: 8.3195, P0CG47: 1.81},
          binding_peptides=6, unscored_peptides=0
   ```

   A design below the IgG1 bar. That is a complete class I path: sequence in,
   verdict out, with no number invented anywhere along it.

**The `/dev/stdin` defect** (section 8, item 10 above) is fixed: the runner
writes the request into a `tempfile.TemporaryDirectory` and passes a real
path, so it works off POSIX. `test_the_runner_no_longer_depends_on_dev_stdin`
reads the function own source and fails if the path comes back.

**The column contract, verified against the real output.** MHCflurry 2.3.29
writes `allele, peptide, mhcflurry_affinity, mhcflurry_affinity_percentile,
mhcflurry_processing_score, mhcflurry_presentation_score,
mhcflurry_presentation_percentile`. Every one is in `COLUMN_MAP`; a test
asserts that. Answering question 4 of section 9: **the percentile is 0-100,
not 0-1** -- `SIINFEKL`/`HLA-A*02:01` returns affinity 6249.061165259931 nM
with percentile 7.964161947308878, and a 6 uM non-binder cannot have a
percentile of 7.96 on a 0-1 scale. Those exact rows are stored in
`mhc_tables.OBSERVED_EXAMPLE` with the version and date.

**The rescaling test asked for.** `test_the_conversion_rescales_nothing`
asserts each converted cell is bit-for-bit `repr()`-identical to what
MHCflurry wrote. `test_the_percentile_is_a_percentage_and_a_rescale_would_flip_a_verdict`
asserts the live failure: 7.96 is above `liability._MHC_I_RANK` (1.0), and
7.96/100 is below it, so any silent division by 100 publishes a 6 uM
non-binder as a presented ligand. Both cuts are imported from `liability.py`
rather than restated, so a change on either side breaks the test.

## C.4 Antibody developability

`developability.py`. Numbering is ANARCII 2.0.8 (BSD-3-Clause, University of
Oxford -- licence read from the wheel own LICENCE file), already present in
`.venv`. It is imported lazily: it loads a torch model and takes about 28 s on
CPU. `parse_anarcii` takes ANARCII output shape, so the whole region cut and
motif scan is testable with no model.

**Regions.** IMGT only. CDR1 27-38, CDR2 56-65, CDR3 105-117, conserved
cysteines 23 and 104 -- Lefranc et al., Dev Comp Immunol 27:55-77 (2003).
Kabat and Chothia are deliberately absent: they are different spans on
different numbering, and reading one with IMGT spans cuts the wrong loops.
`parse_anarcii` refuses any scheme but IMGT rather than guessing.

**Motifs, each cited.**

| Liability | Motifs | Provenance |
| --- | --- | --- |
| Deamidation | NG, NS, NN, NT, NH | Lu et al., mAbs 11(1):45-57 (2019), doi:10.1080/19420862.2018.1548233 -- the "canonical motifs" list; hierarchy NG > NS > NT is the same paper |
| Isomerisation | DG, DS, DD, DT, DH | same paper; hierarchy DG > DS > DD or DT |
| Oxidation | M, W | Xu et al., mAbs 11(2):239-264 (2019), doi:10.1080/19420862.2018.1553476; and the in-silico PTM review, mAbs 14(1) (2022), PMC8791605, which states sequence-based approaches flag every Met and Trp |
| Unpaired cysteine | C outside IMGT 23/104 | Xu et al. 2019, which says candidates with unpaired CDR Cys should be eliminated |
| N-glycosylation sequon | N-X-S/T, X != P | Xu et al. 2019 |

**No pass cut is supplied, and that is a finding, not an omission.** Lu et al.
report that only 47 per cent of CDRH2 NG instances and 19 per cent of CDRH2 DG
instances modified under the matching stress. A motif is a place to look. The
record carries `pass_cut: None` and `pass_cut_reason:
no_published_motif_count_threshold`, and a test asserts both.

The even/odd cysteine reading is explicit: an even number of non-canonical
cysteines is reported as possibly an extra disulfide (which Xu et al. describe
in some CDRH loops), an odd number as at least one genuinely unpaired. The
record says which reading it took.

**Measured on a real paired sequence** (trastuzumab-like VH/VL, 2026-11-16):
H chain typed `H`, light chain typed `K`; CDR-H1 `GFNIKDTY`, CDR-H2
`IYPTNGYT`, CDR-H3 `SRWGGDGFYAMDY`; CDR-L1 `QDVNTA`, CDR-L2 `SAS`, CDR-L3
`QQHYTTPPT`. Findings in CDRs: deamidation NG at H62 (CDR2) and NT at K36
(CDR1), isomerisation DT at H36 and DG at H110 (CDR3), oxidation W at H107 and
M at H115. The CDR-H2 NG and CDR-H3 DG are the two liabilities that antibody
is known for, which is a useful sanity check on the scan.

**DeepSP and CrossAbSense were not installed, and the reason is concrete.**
DeepSP (MIT) ships Keras models and there is no TensorFlow in this
environment; adding one to a torch environment for three descriptors was not
worth the dependency risk while four other workers share it. CrossAbSense
v0.9 (Apache-2.0, weights on Hugging Face under the same licence -- model card
read 2026-11-16) runs a frozen ESM-Cambrian 6B encoder, which is not a CPU
screening step. Both are therefore supported as **readers**:
`read_deepsp_table` and `read_crossabsense_table` take the CSV those tools
write, attach the method and source, refuse a file that does not carry their
columns, and refuse an unparsable cell rather than dropping it. The five
CrossAbSense assays and their Spearman rho from the model card are recorded in
the module.

## C.5 Humanness

`humanness.py` reports two different quantities and refuses to conflate them.

* **OASis identity** -- the fraction of a domain 9-mers seen in human
  repertoires, at BioPhi default threshold of "more than 10 per cent of
  human subjects". The peptide set ships inside `promb` 1.0.2 (MIT, MSD),
  40 MB in the wheel, so the 6.5 GB / 21.5 GB OASis database is not needed.
  The database itself is CC-BY-4.0 (Zenodo record 5164685) and BioPhi is MIT,
  both read 2026-11-16, so this route is genuinely open including
  commercially. **Measured:** VH identity 0.70536 (112 peptides scored, 79
  matched), VL identity 0.82828 (99 scored, 82 matched).
* **AbLang-2 pseudo-log-likelihood** -- `ablang2` 0.2.1, BSD-3-Clause, weights
  fetched by the package. **Measured:** -0.93355 for the paired VH/VL, 18.5 s
  on CPU after a 48 s first load. It is labelled `antibody_likeness`, never
  humanness, because AbLang-2 is trained on OAS, which is about 88 per cent
  human and 11 per cent mouse (Olsen et al. 2022) -- a murine sequence can
  score well. A test asserts the record says so.

Neither carries a pass cut. BioPhi reports a percentile against clinical-stage
therapeutics rather than a pass mark, and no publication gives a
pseudo-log-likelihood cutoff.

IgFold was not used: academic-use only.

## C.6 MHC class II

There is no permissively licensed class II predictor. Each option was checked
and quoted in `immunogenicity.CLASS_II_OPTIONS` with the date:

| Option | Licence | Commercial |
| --- | --- | --- |
| NetMHCIIpan 4.3 (DTU) | service page offers a "download page for academic users; other users are requested to contact Health Tech Software Package Manager" (read 2026-11-16) | not granted |
| MixMHC2pred 2.0 (LICR) | LICENSE opens "FOR ACADEMIC NON-COMMERCIAL RESEARCH PURPOSES ONLY" | excluded |
| MHCnuggets 2.4.1 (JHU) | JHU Academic Software License: "restricted to non-commercial purposes" | separate agreement required |
| IEDB class II consensus tools | bundles NetMHCIIpan; components not all read | unverified |
| epytope 4.0.1 / mhctools 4.9.5 | framework is BSD-3 / Apache-2.0 | the wrapped predictor licence still binds |

**One trap worth attention:** MHCnuggets PyPI metadata says "Apache License"
while the repository LICENSE file is the JHU academic agreement. A licence
read off PyPI alone would have wired a non-commercial tool in. A test asserts
that contradiction stays recorded.

So nothing is wired in. What is implemented instead: `register_backend()` for
whoever holds a licence, `predict_class_ii()` which raises with the reason,
`class_ii_status()` as a documented unavailable path a screen can report, and
`write_class_ii_table()` for someone who ran a licensed predictor elsewhere.
The converter warns in advance about the two failures that would otherwise be
silent -- a rank on a scale the ADA threshold does not belong to, and an
incomplete DR panel -- and two tests prove the warnings are right by handing
the written table to `liability.measure_sequence_liability`: a consensus table
over the full panel at rank 2.0 produces `ada: hit`, and the identical table
declared as `netmhciipan_4.3_el` produces `ada: untested,
blocker=rank_scale_unknown`.

## C.7 Aggregation and solubility: what else is open

Not implemented, reported as asked. `solubility.py` already reads supplied
scores and judges nothing, so anything added here should feed that, not
duplicate it.

* **NetSolP** -- BSD-3 per the earlier pass; **I did not re-verify the licence
  file myself**, so treat it as unconfirmed until someone opens it. Sequence
  solubility from a transformer; would need weights.
* **DeepSP** (MIT) -- the SAP/SCM surrogate, see C.4. Needs TensorFlow.
* **CamSol** -- web service, not open source. Excluded.
* **Aggrescan3D** -- server and a standalone with its own terms; not read.
* **Protein-Sol** -- web service; the sequence method is simple enough to
  reimplement from the paper, but `modality.py` already carries an AGGRESCAN
  sequence score, so check for duplication first.
* **TAP (Therapeutic Antibody Profiler)** -- the five metrics are published
  with reference distributions from clinical-stage antibodies, but the
  thresholds are percentile bands against that set, so implementing it means
  carrying the reference distribution, not a number. Worth doing properly or
  not at all.

## C.8 Integration request for worker A

I did not touch `screen.py`. Precisely what I would ask for, in priority
order:

1. **Class I, the one that changes verdicts.** Where a run directory has no
   `mhc_i.tsv` and `MHCFLURRY_PREDICT` (or `mhcflurry-predict` on `PATH`)
   resolves, call
   `mhc_tables.predict_sequence(sequence, alleles, widths=[8,9,10,11], directory=run_dir)`
   before `measure_sequence_liability`. Guard it: it is ~160 s for one 120-mer
   at two alleles and one width, and it scales with alleles x widths. Make the
   allele list configurable; do **not** default to a large panel.
2. **The reference table.** For `binder_kind="protein"` the class I axis
   cannot clear or hit without `mhc_i_reference.tsv`. Call
   `mhc_tables.predict_reference_table({...}, alleles, widths=..., directory=run_dir)`
   with the same alleles and widths, once, and cache it: it took 680.9 s and
   does not depend on the design. The three sequences must be supplied by the
   caller (P02768, P01857, P0CG47); the module will not fetch them.
3. **Developability as a report, not an axis.** `developability_report(...)`
   has no `state` and no `measurement`, like `solubility.read_solubility`.
   Attach it beside the measured line; the screen call must not move on it.
   Only call it for an antibody modality -- ANARCII raises on a non-antibody
   sequence, and that raise should become a blocker string, not an exception.
4. **Humanness.** `humanness.humanness_report(sequences, pairs=...)`. The
   OASis half is fast once the 40 MB set is loaded (cached per process); the
   AbLang-2 half loads torch, so make it opt-in.
5. **Class II.** Report `immunogenicity.class_ii_status()` wherever `mhc_ii`
   or `ada` reads `predictor_not_configured`, so the record distinguishes "no
   file" from "no licensable predictor exists".
6. **Do not** add a pass criterion for any of this. Nothing here has a
   published cut.

## C.9 Test output, run on this machine

```
$ ./.venv/bin/python -m pytest tests/test_disease_developability.py \
    tests/test_disease_humanness.py tests/test_disease_immunogenicity.py \
    tests/test_disease_mhc_contract.py tests/test_disease_screening_config.py \
    tests/test_disease_mhc_tables.py -q
60 passed in 4.25s

$ ./.venv/bin/python -m pytest tests/test_disease_liability.py \
    tests/test_disease_liability_coverage.py tests/test_disease_liability_methods.py \
    tests/test_disease_liability_units.py tests/test_disease_screening_tools.py -q
155 passed in 198.87s (0:03:18)

$ ./.venv/bin/python -m ruff check src/rfproteina/disease/developability.py \
    src/rfproteina/disease/humanness.py src/rfproteina/disease/immunogenicity.py \
    src/rfproteina/disease/mhc_tables.py scripts/screen_antibody.py tests/test_disease_*.py
All checks passed!
```

54 of those 60 are new here; the other 6 are the first pass `mhc_tables`
tests, which still pass against the extended module.

## C.10 Weaknesses a reviewer should attack

1. **The reference scan is not pinned to the design widths in practice.**
   `predict_reference_table` takes `widths`, and `liability._reference_density`
   refuses a width mismatch -- but nothing stops a caller scanning the design
   at four widths and the reference at one, and then the refusal is a blocker
   the user has to diagnose. Worker A integration should pass the same
   `widths` object to both.
2. **The reference scan cost is real.** 680.9 s for three proteins at one
   width and two alleles. At four widths and seven alleles it is hours. There
   is no caching layer; I only noted that it should be cached.
3. **`_one_contiguous_scan` in `liability.py` will return False** for a
   reference table written with more than one allele, because each allele
   repeats the same start positions. It falls back to the safer pooling
   answer, so the verdict above is sound, but a reviewer should check whether
   that is costing anything.
4. **ANARCII confidence score is passed through unjudged.** It returned
   ~30.9 for both chains here and 0.0 with an error for a non-antibody. I set
   no minimum score, because I found no published one.
5. **The motif scan has no exposure term.** Said in the record, said in the
   module docstring, said here. Flagging every Met and Trp over-calls, and the
   cited review says so.
6. **The sequon scanner region attribution is a convention.** A sequon
   starting at H38 is reported in CDR1 even though its Thr sits in FR2. The
   finding names the first residue region.
7. **OASis identity here is only available at the 10 per cent threshold.**
   The packaged set is thresholded; loose/medium/strict need the 21.5 GB
   database. An identity without its threshold is meaningless, so the
   threshold is in every record, but a reader comparing against a published
   figure at a different threshold will be comparing different numbers.
8. **The `promb` peptide set version is not pinned by checksum.** It ships in
   the wheel; I recorded the wheel version (1.0.2) and nothing more.
9. **AbLang-2 pseudo-log-likelihood is a model score with no calibration**
   on this program designs. Reported, never judged.
10. **`CLASS_II_OPTIONS` quotes licences read on one day.** Licences change.
    Each entry carries its read date for exactly that reason, and the IEDB
    entry honestly says its terms were not read.
11. **The config YAML is generated from the modules, then checked against
    them.** That makes drift impossible but also makes the check circular: it
    proves consistency, not correctness.
12. **Nothing was added to `pyproject.toml`.** `ablang2`,
    `rotary-embedding-torch` and `promb` were installed into `.venv` with
    `--no-deps` so no shared pin moved; `mhcflurry` is in a separate
    environment entirely. That means a fresh clone will not have them, and the
    screens will report their blockers instead of running. Someone should
    decide whether these belong in an optional dependency group.

## C.11 Questions for the reviewing model

1. Is a separate virtual environment for MHCflurry the right call, or should
   the project move to pandas 3 and take it as a real dependency?
2. Should the class I reference table be cached in the repository (it is
   design-independent for a fixed allele/width set), and if so, where, given
   it is 2138 rows for the smallest useful configuration?
3. The developability report deliberately has no state. Is that right, or
   should an unpaired CDR cysteine -- the one liability the cited review says
   to eliminate outright -- be a hit?
4. Is reporting OASis identity at a single fixed threshold more useful than
   not reporting it at all, given a reader may compare it against a figure
   computed at another threshold?
5. Class II: is it better to leave the axis permanently untested, or to
   recalibrate the Paul et al. median rule onto a NetMHCIIpan rank for
   academic use only, with the commercial path left blocked?
6. The oxidation motif flags every Met and Trp. Would a crude exposure proxy
   from a predicted structure be better than no exposure term, or would it
   just launder the same over-call?

---

# Worker E — oracle upgrades, stability/ddG and free-energy rescoring

Appended 2026-11-16. Everything below was run on a
Windows PC under WSL against the project's Linux venv
(`.venv`), not inferred. Nothing was committed or pushed;
all changes sit uncommitted in the working tree. This section is append-only and
changed none of the sections above.

## E.1 Scope and one-paragraph summary

This pass adds the oracle and physics-rescoring work the first pass deferred:
genuine free, open-weights refolding-oracle adapters (Chai-1, Boltz-2) that drop
into the existing `OracleMetric` contract; the long-commented `oracle_off_target`
specificity oracle, wired to the paralog axis and fetching off-target structures
from AFDB on first use; a ThermoMPNN stability (ddG) screen as the free
replacement for Rosetta ddG; an ImmuneBuilder antibody-structure oracle in place
of academic-only IgFold; and a working, tested MM-GBSA rescoring stage plus a
documented FEP scaffold. No existing Python module was edited. No GPU or design
run was started; all heavy model calls are lazy-imported and mocked in tests. No
weights were downloaded.

## E.2 Files added

| File | Lines | What it is |
| --- | --- | --- |
| `src/rfproteina/disease/stability.py` | ~285 | ThermoMPNN site-saturation ddG screen: structured roll-up + verdicts |
| `src/rfproteina/disease/rescoring.py` | ~330 | Single-structure MM-GBSA via OpenMM + RBFE/FEP scaffold |
| `src/rfproteina/metrics/oracles/free_oracle_common.py` | ~200 | Shared base `SequenceRefoldOracle` + per-chain sequence extraction |
| `src/rfproteina/metrics/oracles/chai_oracle.py` | ~172 | Chai-1 refolding oracle adapter (Apache-2.0) |
| `src/rfproteina/metrics/oracles/boltz_oracle.py` | ~190 | Boltz-2 refolding oracle adapter (MIT) + affinity reader |
| `src/rfproteina/metrics/oracles/immunebuilder_oracle.py` | ~154 | ImmuneBuilder (ABodyBuilder2) antibody oracle (BSD-3) |
| `src/rfproteina/metrics/oracles/paralog_off_target.py` | ~290 | Paralog-fed off-target transform, AFDB fetch on first use |
| `configs/metrics/refolding_oracles/chai1.yaml` | ~35 | Chai-1 oracle config (oracle + oracle_ppi) |
| `configs/metrics/refolding_oracles/boltz2.yaml` | ~38 | Boltz-2 oracle config (oracle + oracle_ppi) |
| `tests/test_disease_stability.py` | 9 tests | summarise, normalise, injection, unavailable, verdicts |
| `tests/test_disease_rescoring.py` | 12 tests | energy subtraction, chain split, verdicts, FEP scaffold |
| `tests/test_oracle_free_common.py` | 6 tests | sequence extraction + fold/persist flow with a fake backend |
| `tests/test_oracle_chai.py` | 4 tests | FASTA, score coercion, `build_output` mapping |
| `tests/test_oracle_boltz.py` | 7 tests | YAML, `build_output` (both key conventions), affinity reader |
| `tests/test_oracle_immunebuilder.py` | 7 tests | chain picking, PDB save + error read, `.pdb` suffix |
| `tests/test_oracle_paralog_off_target.py` | 15 tests | identifier parsing, resolution, setup/cache, unavailable paths |

## E.3 File changed

**`configs/metrics/refolding_oracles/af3.yaml`** — the commented `oracle_off_target`
example was replaced with an active, wired block (see E.6). It is gated behind the
`off_target_refolding` tag, so it changes no existing verdict unless that tag is
set. `of3.yaml` inherits this block through its `defaults: [af3, _self_]`, so the
OpenFold3 oracle gets the same off-target assay for free.

The three files already dirty before any screening work
(`results_actions.py`, `test_disease_board_vocabulary.py`,
`test_disease_results_actions.py`) and the files owned by other parallel workers
(`screen.py`, `interface_quality.py`, etc.) were **not** touched by Worker E.

## E.4 Licences — verified from each project's own LICENSE on 2026-11-16

| Tool | Licence | How verified |
| --- | --- | --- |
| Chai-1 (chaidiscovery/chai-lab) | Apache-2.0, **code and weights** | repo `LICENSE` is Apache-2.0; README states "Apache 2.0 License (both code and model weights) ... commercial purposes, including for drug discovery" |
| Boltz (jwohlwend/boltz) | MIT | repo `LICENSE` |
| OpenFold (aqlaboratory/openfold) | Apache-2.0 | repo `LICENSE` (OF3 weights are OpenBind-0 per that project; of3.yaml already handled by the AF3 machinery) |
| ThermoMPNN (Kuhlman-Lab/ThermoMPNN) | MIT | repo `LICENSE` |
| ImmuneBuilder (oxpig/ImmuneBuilder) | BSD-3-Clause | repo `LICENSE` |
| OpenMM (openmm/openmm) | MIT (API, Reference, CPU platforms) | repo `docs-source/licenses/Licenses.txt` |
| OpenFE (OpenFreeEnergy/openfe) | MIT | repo `LICENSE` |
| pmx (deGrootLab/pmx) | LGPL-3.0 | repo `LICENSE` |
| alchemlyb (alchemistry/alchemlyb) | BSD-3-Clause | repo `LICENSE` |
| AFDB structures / UniProt data | CC-BY-4.0 | EMBL-EBI distribution terms |

None are paid; none needs a key or account. ESM3 was not used anywhere, so its
ambiguous licence is moot here.

## E.5 Each screen/oracle: method, inputs, outputs, thresholds

### E.5.1 ThermoMPNN stability screen (`stability.py`)
* **Method.** Site-saturation mutagenesis over a finished design. The design is
  the wild-type reference; ThermoMPNN predicts the ddG of every point mutation.
* **Field provenance (verified from `analysis/SSM.py`).** Output columns
  `position`, `wildtype`, `mutation`, `ddG_pred` (kcal/mol). Sign: `retrieve_best_mutants`
  selects the best mutant with `idxmin()` of `ddG_pred`, so **more negative = more
  stabilising**. That is the only interpretation relied on.
* **Outputs.** `most_stabilizing_ddg`, `most_destabilizing_ddg`, `mean_ddg`,
  `n_strongly_stabilizing` (ddG <= -1.0), `fraction_positions_improvable`
  (positions whose best mutation ddG <= -1.0), `mean_best_per_position`.
* **Thresholds.** `fraction_positions_improvable` max 0.25 is a **suggested
  default pending calibration** — ThermoMPNN publishes no design-level pass mark
  and none is invented. The other bars are informational (no gate).
* **Not done.** No verified public entry point into the installed ThermoMPNN's
  SSM runner was wired; `run_site_saturation` accepts an injected `predict_fn`
  and otherwise raises a clear, actionable error. ThermoMPNN is **not** installed
  in the venv; `predictor_available()` returns `False` today, verified.

### E.5.2 MM-GBSA rescoring + FEP scaffold (`rescoring.py`)
* **Method.** Single-structure implicit-solvent decomposition
  `dG_bind = E(complex) - E(receptor) - E(ligand)` via OpenMM (GBn2 implicit
  solvent, amber14 force field), each subsystem optionally minimised. Chain
  splitting uses biotite (a pinned dependency), so the split is exercised live.
* **Outputs/units.** `e_complex`, `e_receptor`, `e_ligand`, `dg_bind`, all
  kcal/mol.
* **Thresholds.** `dg_bind` favourable bar at <= -5.0 kcal/mol is a **suggested
  default pending calibration**; a single-structure MM-GBSA number has no
  absolute pass mark and is only a relative ranking signal.
* **Honest limits.** Not averaged over an MD ensemble; no explicit entropy term.
  `plan_rbfe()` is a documented scaffold only (OpenFE/pmx/alchemlyb named), and
  fabricates no free energy. OpenMM is **not** installed in the venv.

### E.5.3 Free refolding oracles (`chai_oracle.py`, `boltz_oracle.py`, base `free_oracle_common.py`)
* **Contract.** Both subclass `SequenceRefoldOracle(OracleMetric)`: extract
  per-chain protein sequences from the design AtomArray, run the backend, save a
  refolded structure to the canonical `refolded_cif_path`, and map the backend's
  AF3-shaped confidences through `extract_confidence_metrics` into the flat
  metric keys the val pipeline reads.
* **Chai-1 fields (verified from chai-lab `main`).** `run_inference(...)` ->
  `StructureCandidates` (`cif_paths`, `ranking_data`, per-token `plddt`).
  `SampleRanking.ptm_scores.{complex_ptm,interface_ptm,per_chain_ptm,per_chain_pair_iptm}`
  and `plddt_scores.{complex_plddt,per_chain_plddt}`, all 0..1. FASTA header
  `>protein|name=<id>`.
* **Boltz-2 fields (verified from `docs/prediction.md`).** `boltz predict <yaml>`;
  outputs `<stem>_model_0.cif` and `confidence_<stem>_model_0.json`
  (`confidence_score`, `ptm`, `iptm`, `complex_plddt`, `chains_ptm`,
  `pair_chains_iptm`). The affinity head (`affinity_pred_value` = log10 IC50 in
  micromolar) is **deliberately not** folded into refold confidences (units clash
  with the nanomolar panels); `parse_affinity()` reads it separately and labels
  the units.
* **Honest limits.** These save the predictor's own output structure; they do
  **not** reproduce AF3Oracle's ~300-line atom-index remap of predicted coords
  onto the input ordering. A downstream `StructureComparisonMetric` that assumes
  identical atom order will need an alignment step for these oracles. Only protein
  chains are forwarded (no ligands/nucleic acids yet). Neither backend is
  installed in the venv (`_backend_available()` returns `False`), so fold_batch
  raises a clear error rather than inventing a structure.

### E.5.4 ImmuneBuilder antibody oracle (`immunebuilder_oracle.py`)
* **Method.** ABodyBuilder2 predicts the Fv structure from heavy/light sequences.
  Verified API (README): `ABodyBuilder2().predict({'H':..., 'L':...})` -> object
  with `.save(path)` (PDB); needs ANARCI.
* **Honest fit.** ImmuneBuilder emits no pTM/ipTM, only a per-residue error
  estimate. This adapter saves the structure (as `.pdb`, via `output_cif_suffix`)
  and carries ImmuneBuilder's mean/max predicted error as clearly-named scalars
  (`immunebuilder_mean_predicted_error_a`) through the base's `_extra`
  passthrough. It never fabricates a pTM to look like the other oracles. Treat it
  as a structure source + error estimate, not an interface-confidence oracle.

### E.5.5 Off-target specificity oracle, fed by the paralog axis (`paralog_off_target.py`)
* **Problem solved.** The commented `oracle_off_target` block needed off-target
  structures on disk, which "don't exist". This derives them from the disease
  screen's paralog axis and fetches each from AFDB on first use.
* **Source → structure.** The paralog axis persists `paralogs` as gene symbols
  (falling back to accessions). Each is resolved to a reviewed human UniProt
  accession via UniProt REST (no key), then fetched as
  `https://alphafold.ebi.ac.uk/files/AF-<acc>-F1-model_v4.cif` and cached under
  `cache_dir`. AFDB files are small (~0.1-5 MB mmCIF each); `max_targets` caps the
  count (default 8). The fetch is lazy (in `setup()`), cached, and fully
  injectable for tests.
* **No invented paths.** If nothing resolves/fetches, `setup()` raises
  `OffTargetUnavailableError` with an actionable message rather than substituting
  a fabricated structure. `expected_save_ids()` reflects the successfully fetched
  set after setup, so the oracle's cache lookup and `forward()` stay in agreement.

## E.6 Integration request for Worker A (owner of `screen.py` and `pass_criteria`)

Worker E deliberately did not touch `screen.py` or any `pass_criteria`. Requested
wiring, all **opt-in** so no existing verdict moves:

1. **Stability axis.** Add a `stability` axis that calls
   `stability.screen_structure(cif, predict_fn=...)` and reports
   `stability.verdicts(report)`. It must stay `untested` whenever ThermoMPNN (or a
   `predict_fn`) is absent — never silently `clear`. Do **not** add its bars to
   any existing `pass_criteria` until someone calibrates
   `fraction_positions_improvable` on this program's own designs.
2. **Rescoring axis.** Add a `mmgbsa` axis calling
   `rescoring.rescore_complex(cif, receptor_chains, ligand_chains, energy_fn=...)`
   and `rescoring.verdicts(result)`. `untested` when OpenMM is absent. The
   `dg_bind <= -5` bar is a suggested default; treat it as a ranking signal, not a
   gate, until calibrated.
3. **Free oracle selection.** `chai1.yaml` / `boltz2.yaml` are drop-in oracle
   configs. Selecting one re-calibrates what every refolding pass criterion is
   measured against, so if Chai/Boltz is made the default oracle, the existing
   ipTM/ipAE thresholds must be re-validated before trusting the verdicts — that
   is a program decision, flagged here, not made by Worker E.
4. **Off-target oracle.** Enabling it needs the `off_target_refolding` tag plus a
   `paralog_record_path` (or explicit `paralog_accessions`) in `af3.yaml`. Its
   metrics land under `oracle_off_target.<accession>.*` for scoring against the
   on-target refold; a specificity pass criterion over those is a follow-up.

## E.7 Tests — what was actually run

Under WSL against `.venv`. Real outputs:

```
./.venv/bin/python -m pytest tests/test_disease_stability.py \
  tests/test_disease_rescoring.py tests/test_oracle_free_common.py \
  tests/test_oracle_chai.py tests/test_oracle_boltz.py \
  tests/test_oracle_immunebuilder.py tests/test_oracle_paralog_off_target.py -q
=> 58 passed in 176.35s
```

Ruff over all 7 source files and 7 test files:

```
./.venv/bin/python -m ruff check <the 14 files>
=> All checks passed!
```

Hydra end-to-end wiring smoke check (constructs the objects, no setup/network):

```
oracle_off_target -> AF3Oracle oracle_name= af3_off_target
transform -> ParalogOffTargetSubstitutionTransform
expected_save_ids (pre-setup) -> ['P04626', 'P00533']
chai1 oracle -> Chai1Oracle oracle_name= chai1 backend_available= False
boltz2 oracle -> Boltz2Oracle oracle_name= boltz2 backend_available= False
```

One real bug was found and fixed during that smoke check: Hydra passes
`paralog_accessions` as an OmegaConf `ListConfig` (not a `list`), which made
`expected_save_ids()` return `[]`; a `_as_plain()` normaliser fixes it. After the
fix, `tests/test_oracle_paralog_off_target.py` was re-run: `15 passed in 212.49s`.

The `af3.yaml`, `chai1.yaml` and `boltz2.yaml` configs each parse under
`yaml.safe_load`, and `af3.yaml` now carries keys
`['oracle', 'oracle_ppi', 'oracle_dna', 'oracle_apo', 'oracle_off_target']`.

## E.8 Weaknesses, assumptions and shortcuts — attack these first

1. **The free oracles save the predictor's structure, not a remap onto the input
   atom order.** AF3Oracle's `_normalize_af3_output`/`_build_atom_index_mapping`
   are not reproduced. Downstream structure-comparison metrics may misalign. This
   is the single biggest gap in the oracle adapters.
2. **ThermoMPNN's real SSM call is not wired** — only the injection seam and the
   output schema are. A reviewer should confirm the installed version's columns
   and sign before enabling it.
3. **The MM-GBSA is single-structure**, not ensemble-averaged, and has no entropy
   term. Good for ranking, not for absolute affinity.
4. **The off-target gene-symbol → accession resolution depends on UniProt REST
   `gene_exact` returning the intended reviewed human entry**; an ambiguous symbol
   could resolve to the wrong accession. `require_all=False` skips failures,
   which can silently shrink the off-target panel.
5. **Boltz per-chain dicts are keyed by either the YAML `id` or a positional
   index** depending on version; `_chain_key` tries both, but an unusual keying
   would mis-map chains.
6. **No checksums on fetched AFDB files**, and no retry/backoff on the fetch.
7. **`extract_protein_sequences` maps non-standard residues to `X`** and drops
   pure-ligand/nucleic chains, so a design whose binder is nucleic is not folded
   by these oracles.

## E.9 Questions for the reviewing model

1. Should the free oracles port AF3Oracle's coordinate remap so their refolds are
   directly comparable to the design, or is saving the predictor's own frame
   acceptable for the metrics actually consumed?
2. Is `fraction_positions_improvable <= 0.25` a defensible first stability bar, or
   should the stability axis report only raw ddG statistics until calibrated?
3. For the off-target oracle, is UniProt `gene_exact` reliable enough, or should
   symbol→accession resolution require a stricter match (and should `require_all`
   default to True to avoid a silently shrunk panel)?
4. Should Boltz-2's affinity head be surfaced anywhere, given its log10(IC50 in uM)
   units cannot enter the nanomolar panels — e.g. as its own clearly-labelled
   ranking axis?
5. Is wiring `of3.yaml` to inherit `oracle_off_target` (via its `defaults`) the
   intended behaviour, or should the off-target assay be AF3-only?
6. Which of these is worth the validation cost first — making Chai-1 the default
   oracle, or enabling the ThermoMPNN stability axis?
