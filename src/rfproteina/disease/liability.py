"""Measurements for five protein sequence-liability axes.

Immunogenicity is the Calis et al. 2013 class I 9-mer score used by the IEDB
immunogenicity tool. A sample is clear only when every scored 9-mer is at most
0. A positive window is a **hit**, not an untested record, and only when it is
one of the ligands the class I table says is presented: the Calis sign was fit
on predicted MHC ligands rather than on every window of a protein, so an
unrestricted scan of all windows is what cannot reach a verdict. This
paragraph said the opposite until 2026-11-21, while
``test_a_presented_ligand_above_zero_is_a_hit`` pinned the behaviour it denied.

ADA is class II allele breadth: a peptide that binds across the DR panel is the
accepted computational anti-drug-antibody risk signal, because a promiscuous
binder finds a restricting allele in most people. It reads the class II table,
not the sequence. The scheme and its threshold are Paul et al., J Immunol
Methods 422:28-34 (2015), blind-validated on fifteen antigens. This replaced the
Kolaskar and Tongaonkar 1990 antigenic-propensity scale, which is a B-cell
epitope scale with no anti-drug-antibody meaning and no pass cut, so it could
never clear or hit and only ever reported a number.

Proteome offtarget is exact distinct 9-mer identity against a FASTA at
``HUMAN_PROTEOME`` or the ``proteome`` argument. MHC class I and II stay on
predictor tables. A missing table stays untested. A present immunogenicity or
ADA table wins over the sequence formula.

Two things a predictor table cannot be made to say are held as limits rather
than guessed at. An affinity column named only for its analyte -- ``ic50``,
``ba_affinity`` -- states no unit, and the cutoffs are nanomolar, so such a
column is read only where the row declares nanomolar and is otherwise refused
with ``ic50_not_nm``; a micromolar figure is never rescaled onto the cut. And a
percentile rank written as a fraction of one cannot be told from a very small
percentage: 0.005 is 0.5 per cent in one export and 0.005 per cent here, and no
publication distinguishes them, so no floor is invented. A table whose every
rank is at or below 1 reads the same either way, because 1 is the largest a
fraction can be. On the reference side that is recorded as
``reference_rank_scale_ambiguous``, because the percentage reading calls more
rows binding and so sets a higher bar, and a bar that is too high clears
designs. On the design side the same reading calls more rows binding and so
errs toward a hit, which is the direction this module reports rather than
refuses -- it is a limit of every rank column read here, the ADA panel's
included, and not a per-axis guard.

GRAVY, instability, and the designed interface are not these axes. Paralog
and the plasma axes are not measured here. Nothing here calls a predictor,
downloads a proteome, or sends a sequence anywhere.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import os
import re
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from rfproteina.disease.combinations import rollup_axis
from rfproteina.disease.modality import membrane_axis

SEQUENCE_LIABILITY_AXES = ("immunogenicity", "proteome_offtarget", "mhc_i", "mhc_ii", "ada")
#: ``membrane`` is measured for a peptide only. ``PEPTIDE_ONLY_AXES`` has
#: carried it since the classes were defined and the peptide ``sequence``
#: bar requires it, so until it was measured no peptide could clear its own
#: bar; the judge refuses it for any other kind.
PEPTIDE_LIABILITY_AXES = ("membrane",)
_PROTEOME_ENV = "HUMAN_PROTEOME"
_KMER = 9
_AA = frozenset("ACDEFGHIKLMNPQRSTVWY")
#: Five bits hold one of the twenty residues, so a 9-mer packs into 45 bits and
#: fits one machine integer. Measured on 11M residues, the size the human
#: reference proteome is: 10.8M distinct 9-mers, 87 MB as a sorted array of
#: these against roughly 800 MB as a Python set of strings, built in 1.5s and
#: then searched in 5 microseconds per design.
_RESIDUE_BITS = 5
_RESIDUE_CODE = {letter: code for code, letter in enumerate(sorted(_AA))}
_NO_KMERS: np.ndarray = np.empty(0, dtype=np.int64)
#: Parsed proteomes by path, size and modification time. Reading a whole proteome
#: costs seconds, and one run screens every finished generation in turn.
#:
#: Bounded, because each entry is tens of megabytes and the panel is a long-lived
#: process: an unbounded cache keyed on modification time would grow by another
#: copy every time the file was rewritten. A run uses one proteome, so the bound
#: can be small and still hit every time.
_PROTEOME_CACHE: dict[tuple[str, int, int], tuple[str | None, np.ndarray]] = {}
_PROTEOME_CACHE_MAX = 2
#: Digests by the identity of the array they were taken over, bounded by the
#: same count as the proteome cache, which holds those arrays. See
#: ``_kmer_digest``: the digest is on every design's record and hashing a
#: proteome's whole k-mer array once per design dominated a run.
_DIGEST_CACHE: list[tuple[np.ndarray, str]] = []
#: The human reference proteome holds about 10.8 million distinct 9-mers. A
#: file an order of magnitude below that is not one, and every 9-mer it lacks
#: is an off-target the axis cannot see, so a clear against it is not a clear.
_PROTEOME_MIN_KMERS = 1_000_000
#: A coding FASTA read as protein. These five letters are amino acids too, so
#: the test is their share of the residues, sampled rather than counted whole.
_NUCLEOTIDE_LETTERS = frozenset("ACGTUN")
_NUCLEOTIDE_SHARE = 0.95
_ALPHABET_SAMPLE = 100_000
#: Below this many residues the alphabet says nothing: a short poly-alanine
#: record is 100% "nucleotide" letters and is still a protein. A coding FASTA
#: large enough to stand in for a proteome is far above this.
_ALPHABET_MIN = 1_000
#: IEDB recommended MHC class I selection: percentile rank <= 1, or IC50 <= 500 nM.
_MHC_I_RANK = 1.0
_MHC_I_IC50 = 500.0
#: IEDB MHC class II recommended percentile rank is 10. IC50 <= 1000 nM is the binding cutoff.
_MHC_II_RANK = 10.0
_MHC_II_IC50 = 1000.0
#: Rank columns, in the order a table is asked for one. The consensus spellings
#: come first because IEDB's consensus export names its rank column after the
#: method it combined -- ``consensus_percentile_rank`` -- and carries no bare
#: rank column at all, so a reader wanting only ``percentile_rank`` saw a table
#: full of ranks as a table with none. The DTU spellings are what NetMHCpan and
#: NetMHCIIpan write for their eluted-ligand and binding-affinity ranks.
_RANK_COLUMNS = (
    "consensus_percentile_rank",
    "consensus_percentile",
    "percentile_rank",
    "%rank",
    "rank_el",
    "el_rank",
    "%rank_el",
    "rank_ba",
    "ba_rank",
    "%rank_ba",
    "rank",
)
_IC50_COLUMNS = ("ic50_nm", "ic50", "affinity_nm", "ba_affinity", "aff_nm")
#: The affinity columns whose own name states the unit. The cutoffs are in
#: nanomolar, so these need no unit cell. ``ic50`` and ``ba_affinity`` name only
#: the analyte: nothing in such a column's name says what its numbers are in,
#: and read as nanomolar a micromolar table under a bare ``ic50`` header is a
#: thousandfold error -- 2.0, a 2 uM non-binder, was published as a hit at 2.0.
#: On the reference side the same cell counts as a binder and inflates the bar,
#: which errs toward clearing the design. ``rfproteina.disease.toxicology``
#: holds this rule for its own panel columns; this is that rule for these.
_IC50_NM_COLUMNS = frozenset({"ic50_nm", "affinity_nm", "aff_nm"})
#: Where an export states the unit its numbers are in, in the spellings
#: ``toxicology`` reads, ChEMBL's ``standard_unit`` among them.
_UNIT_COLUMNS = ("unit", "units", "uom", "standard_unit", "standard_units")
#: Spellings that mean the cell is already nanomolar. Anything else a unit
#: column names is refused rather than rescaled: a declared unit that is not
#: nanomolar is as likely to be a percent or a log value as a concentration
#: this could convert, and the module's rule is that a reading whose unit is
#: not nanomolar is never rescaled onto the cut.
_NM_UNITS = frozenset({"nm", "nanomolar", "nmol/l", "nmol/litre", "nmol/liter"})
_SCORE_COLUMNS = ("score", "score_el", "score_ba")
#: Cells that say "no value". Anything else that will not parse as a number is
#: a cell this cannot read, and is refused rather than silently skipped.
_NOT_A_VALUE = frozenset({"", "nan", "none", "null", "na", "n/a", "nd", "n.d.", ".", "-", "--", "true", "false"})
#: The columns a verdict is computed from. A cell this cannot read in any of
#: them is refused; position columns are not here because they only bound a
#: length and a missing one is already handled.
_NUMERIC_COLUMNS = _RANK_COLUMNS + _IC50_COLUMNS + _SCORE_COLUMNS
#: Columns whose number cannot be negative. A percentile rank is a percentage
#: in [0, 100] and an IC50 is a concentration. Several exporters write -1 for a
#: window the predictor declined, and -1 satisfies every cutoff there is, so
#: such a cell was the best binder in the table and was published as the
#: measurement. It is an unreadable cell like any other and is refused as one.
#: A score column is not here: the Calis score is signed by construction.
_NON_NEGATIVE_COLUMNS = _RANK_COLUMNS + _IC50_COLUMNS
#: The largest a percentile rank written as a fraction of one can be. Some
#: exports write the rank as a fraction rather than as a percentage, and 0.005
#: then means 0.5 per cent where this module reads 0.005 per cent -- a
#: hundredfold apart, and nothing published tells the two apart. No floor is
#: invented and no cell is rescaled. But a table holding any rank above this
#: cannot be fractions at all, so a table whose every rank is at or below it is
#: the ambiguous case, and that is recorded rather than passed off as measured.
_RANK_FRACTION_MAX = 1.0
_START_COLUMNS = ("start", "begin", "first", "pos")
#: Calis et al., PLoS Comput Biol 2013, doi:10.1371/journal.pcbi.1003266.
#: Positions 1, 2, and the C-terminus are masked. The other weights are the IEDB 9-mer mask.
_CALIS_SCALE = {
    "A": 0.127,
    "C": -0.175,
    "D": 0.072,
    "E": 0.325,
    "F": 0.380,
    "G": 0.110,
    "H": 0.105,
    "I": 0.432,
    "K": -0.700,
    "L": -0.036,
    "M": -0.570,
    "N": -0.021,
    "P": -0.036,
    "Q": -0.376,
    "R": 0.168,
    "S": -0.537,
    "T": 0.126,
    "V": 0.134,
    "W": 0.719,
    "Y": -0.012,
}
_CALIS_WEIGHT = (0.00, 0.00, 0.10, 0.31, 0.30, 0.29, 0.26, 0.18, 0.00)
_CALIS_METHOD = "calis_2013_class_i_9mer"
#: The anchor positions ``_CALIS_WEIGHT`` masks. Calis et al. make the excluded
#: set per molecule -- Methods: "The six positions with least impact on the
#: binding affinity were defined as non-anchor positions", which the Results
#: give as "P1, P2 and P9 for most HLA molecules, but e.g. P2, P5 and P9 for
#: HLA-B*0801". The paper's Table 2 publishes a position importance only for
#: that default masking, where P1, P2 and P9 read "NA (anchor)", so there is no
#: published weight for scoring P1 and no weight vector can be built for any
#: other mask without inventing one. So the default is what is applied, the
#: allele is recorded rather than used to choose a mask, and a ligand restricted
#: by the one molecule the paper names as an exception cannot clear this axis.
_CALIS_MASK = (1, 2, 9)
#: That exception, in ``_allele_key`` spelling. One entry, because one is what
#: the paper names; an anchor set for any other allele would be invented.
_CALIS_EXCEPTION_ALLELES = frozenset({"B0801"})
#: Paul et al., J Immunol Methods 422:28-34 (2015), doi:10.1016/j.jim.2015.03.022.
#: The three DRB1 alleles with population frequency at or above 12 per cent plus
#: the four DRB3/4/5 alleles. DP is deliberately absent: the paper reports that
#: including it made the scheme perform worse, and adding DRB3/4/5 is what made
#: it work. The panel is the method, so it is not a tuning knob.
_DR_PANEL = (
    "DRB1*03:01",
    "DRB1*07:01",
    "DRB1*15:01",
    "DRB3*01:01",
    "DRB3*02:02",
    "DRB4*01:01",
    "DRB5*01:01",
)
#: A peptide binds an allele at a consensus percentile rank of 20 or better, and
#: the blind-validated universal rule is a median of 20 or better across the
#: seven. Both numbers are the paper's, on the IEDB consensus rank scale.
_DR_RANK_BINDER = 20.0
_DR_MEDIAN_MAX = 20.0
_ADA_METHOD = "mhc_ii_dr_breadth_paul_2015"
_ALLELE_COLUMNS = ("allele", "mhc", "hla")
_PEPTIDE_COLUMNS = ("peptide", "sequence")
_METHOD_COLUMNS = ("method", "predictor")
#: The threshold above belongs to the IEDB consensus rank. A NetMHCIIpan EL rank
#: is calibrated differently -- its published bands are 1 per cent strong and 5
#: per cent weak -- so 20 on that scale means something else and is refused.
#:
#: This is a substring test, not an exact one, because the real tool writes the
#: components it combined into the name: "consensus (comblib/smm/nn)" and
#: "IEDB recommended (consensus)" are both the consensus scale, and an exact
#: match against a fixed set would have refused genuine output as an unknown
#: scale -- a false blocker on the one input the rule was calibrated for.
_CONSENSUS_MARK = "consensus"
#: Human proteins tolerated at high systemic exposure: serum albumin, the IgG1
#: constant region, and ubiquitin. The user scores these with the same tool,
#: alleles and version as the design, and the highest density among them is the
#: bar. Frame: De Groot and Scott, Trends Immunol 28:482-490 (2007) -- epitope
#: content relative to tolerated proteins, rather than an absolute cutoff.
_REFERENCE_PROTEINS = ("P02768", "P01857", "P0CG47")
#: The same three, for the membership test a reference table is held to. Only
#: these may set the bar: a density over some other protein is a bar with no
#: meaning, and one over a protein with high epitope content inflates it.
_TOLERATED = frozenset(_REFERENCE_PROTEINS)
#: The same three by entry name, because a UniProt or NCBI FASTA header carries
#: the name rather than only the accession.
_TOLERATED_NAMES = {
    "ALBU_HUMAN": "P02768",
    "IGHG1_HUMAN": "P01857",
    "UBB_HUMAN": "P0CG47",
}
#: Separators a protein label uses: FASTA pipes, list commas, brackets and
#: whitespace. NCBI writes ``sp|P02768.2| ALBU_HUMAN (Homo sapiens)``.
_LABEL_SPLIT = re.compile(r"[|,;()\[\]\s]+")
#: An isoform (``-1``) or version (``.2``) suffix names the same protein.
_ISOFORM = re.compile(r"[-.]\d+$")
#: UniProt accession shape. A part matching this and not tolerated means the
#: label names a different protein, so the label cannot set the bar.
_ACCESSION_SHAPE = re.compile(r"[OPQ]\d[A-Z0-9]{3}\d|[A-NR-Z]\d(?:[A-Z][A-Z0-9]{2}\d){1,2}")
_REFERENCE_SUFFIX = "_reference"
_PROTEIN_COLUMNS = ("protein", "accession", "entry", "id")
_END_COLUMNS = ("end", "stop", "last")
_LENGTH_COLUMNS = ("length", "protein_length")
#: A count of distinct binding peptides of width w is a count over the
#: ``L - w + 1`` windows of that width, not over the L residues, so windows are
#: what both sides are divided by. Per residue the two sides do not compare: a
#: 76-residue design holds 0.895 nine-mer windows per residue and a 609-residue
#: albumin reference holds 0.987, so the design's density came out 9.3 per cent
#: low against the bar, and at class II widths on a 60-residue design 21.5 per
#: cent low -- systematically toward clear, because a design is short and the
#: reference proteins are long. The name says which denominator was used, so a
#: stored record cannot be read on the wrong one.
_DENSITY_METHOD = "binding_peptides_per_100_windows"
#: Every column some axis reads. A header holding none of them was not parsed.
_KNOWN_COLUMNS = frozenset(
    _RANK_COLUMNS
    + _IC50_COLUMNS
    + _SCORE_COLUMNS
    + _START_COLUMNS
    + _ALLELE_COLUMNS
    + _PEPTIDE_COLUMNS
    + _METHOD_COLUMNS
    + _PROTEIN_COLUMNS
    + _END_COLUMNS
    + _LENGTH_COLUMNS
    + _UNIT_COLUMNS
)
#: Windows the density is quoted per, so a short chain and a long one compare.
_DENSITY_BASIS = 100.0
#: A complete scan starts at the protein's first residues. Two rather than one
#: because a class I scan of 9-mers can be reported from position 1 or 2
#: depending on the tool's indexing.
_SCAN_START_MAX = 2
#: The share of a table's peptides that must occur in a designed chain before
#: the table is accepted as that chain's scan. One incidental match is not
#: provenance: a stale table sharing only a linker cleared an axis on it.
_PROVENANCE_SHARE = 0.5
#: How much of a chain a scan must cover before a density of zero is a
#: measurement. Short of this the file does not say the binders are absent,
#: only that it did not look. A little below one, because a predictor drops a
#: window holding a non-standard residue.
_SCAN_COVER_SHARE = 0.9
#: Peptide widths a binding cutoff is defined for: class I 8-11, class II
#: 13-25. A width outside this cannot earn scan coverage, because no verdict is
#: calibrated for it -- a column of dipeptides is not a scan of anything.
_SCORED_WIDTHS = frozenset(range(8, 12)) | frozenset(range(13, 26))
#: The largest share of a reference protein's windows that may be called
#: binding before the table is the problem.
#:
#: **The 0.49-binders-per-100-residues figure this comment used to cite is not a
#: measurement.** It is ``3 * 100 / 609`` -- the three hand-written rows of the
#: fixture in ``test_a_filtered_reference_export_cannot_set_the_bar`` over
#: albumin's length, the same three that give its companion ``6.25`` as
#: ``3 * 100 / 48``. Measured instead on this machine (2026-11-20): albumin runs
#: at **10.28** binders per 100 windows under the installed class I predictor,
#: twenty-one times the figure quoted, and the old corollary that a quarter of
#: all windows is "two orders above anything a tolerated protein carries" is
#: false in both classes.
#:
#: **0.25 is below the floor any honest class II panel scan can reach**, so this
#: guard refuses every class II protein scan and the axis cannot report. The
#: cause is not unselective predictors: both installed scorers are calibrated on
#: their own scale, and the share compounds twice -- TEPITOPE's bins are
#: per-9-mer-core while ``_best_core`` takes the max over the seven cores in each
#: 15-mer, so one allele alone covers 31-42% of albumin's windows before the
#: any-of-seven union. Measured unions: TEPITOPE 0.716 albumin / 0.704 IgG1 /
#: 0.265 ubiquitin; a pure seven-allele PSSM table 0.338 / 0.296 / 0.065.
#:
#: The constant is deliberately left alone, because the right repair is contested
#: and raising the ceiling loses real protection: at rank 10 over seven alleles
#: albumin is 72% of its windows, so the reference stops representing depletion
#: and starts representing what any 600-residue protein looks like at that cut.
#: Tightening the class II cut, or scoring breadth per window as the ADA axis
#: does, are the alternatives. That is a calibration decision, not a repair.
_BAR_MAX_WINDOW_SHARE = 0.25
#: A number read from a table the user supplied, rather than one this module
#: computed. It is tagged separately because naming it after a formula the code
#: implements would claim a provenance the number does not have.
SUPPLIED_METHOD = "supplied_table"


def measure_sequence_liability(
    samples: list[tuple[str, str | None]],
    *,
    directory: Path,
    proteome: Path | None = None,
    binder_kind: str = "protein",
) -> tuple[dict[str, dict[str, object]], dict[str, dict[str, object]]]:
    """Five axis records, and one proteome record per example id.

    ``samples`` are ``(example_id, sequence)`` in filename order. A missing
    sequence is not a match. The proteome path argument wins over
    ``HUMAN_PROTEOME``. The returned records do not contain a sequence or a path.

    ``binder_kind`` decides how the MHC axes read their table: a peptide is
    judged by the window rule, and a protein by binding density against the
    reference proteins, because the cutoffs mark every protein otherwise.
    """
    loaded = _load_proteome(proteome if proteome is not None else _proteome_from_env())
    by_example: dict[str, dict[str, object]] = {}
    for example_id, sequence in samples:
        by_example[example_id] = _proteome_record(sequence, loaded)
    proteome_axis = _rollup_proteome(samples, by_example, loaded)
    axes = {
        "immunogenicity": _immunogenicity_record(directory, samples),
        "proteome_offtarget": proteome_axis,
        "mhc_i": _mhc_record(
            directory,
            "mhc_i",
            rank_max=_MHC_I_RANK,
            ic50_max=_MHC_I_IC50,
            samples=samples,
            binder_kind=binder_kind,
        ),
        "mhc_ii": _mhc_record(
            directory,
            "mhc_ii",
            rank_max=_MHC_II_RANK,
            ic50_max=_MHC_II_IC50,
            samples=samples,
            binder_kind=binder_kind,
        ),
        "ada": _ada_record(directory, samples),
    }
    found = {name: axes[name] for name in SEQUENCE_LIABILITY_AXES}
    if binder_kind == "peptide":
        membrane = {example_id: membrane_axis(sequence) for example_id, sequence in samples}
        found["membrane"] = rollup_axis(
            [example_id for example_id, _sequence in samples],
            membrane,
            membrane_axis(None),
        )
    return found, by_example


def _proteome_from_env() -> Path | None:
    raw = os.environ.get(_PROTEOME_ENV, "").strip()
    if not raw:
        return None
    return Path(raw)


def _load_proteome(path: Path | None) -> tuple[str | None, np.ndarray]:
    """A blocker and the packed 9-mers. A blocker means the array was not a measurement.

    A parsed proteome is cached on the file's identity, so screening several
    generations in one run reads it once. Only a success is cached: a file that
    was missing or unreadable may be there on the next call.
    """
    if path is None:
        return "proteome_not_configured", _NO_KMERS
    if not path.is_file():
        return "proteome_unreadable", _NO_KMERS
    try:
        stat = path.stat()
    except OSError:
        return "proteome_unreadable", _NO_KMERS
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    cached = _PROTEOME_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        # Read gzipped as well as plain, because that is how a reference
        # proteome ships. UniProt serves UP000005640_9606.fasta.gz and the
        # download registry fetches exactly that file, so a machine that had
        # run the fetch still reported this axis unreadable: the one file the
        # program itself downloads for this axis was the one shape it could
        # not open. Decided on the extension and not on the content, so a
        # plain .fasta is read exactly as before.
        if path.suffix == ".gz":
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                text = handle.read()
        else:
            text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError, EOFError, gzip.BadGzipFile):
        return "proteome_unreadable", _NO_KMERS
    except MemoryError:
        return "proteome_too_large", _NO_KMERS
    records = _fasta_records(text)
    if not records:
        return "proteome_empty", _NO_KMERS
    if _is_nucleotide(records):
        # cds.all.fa where pep.all.fa was meant. A coding FASTA packs into
        # millions of 9-mers that are not peptides, so it answers every query
        # with "no match" and the axis reads clear against nothing.
        return "proteome_not_protein", _NO_KMERS
    try:
        found = [_packed_kmers(record) for record in records]
        kmers = np.unique(np.concatenate(found)) if found else _NO_KMERS
    except MemoryError:
        # A FASTA far larger than a reference proteome, pointed at by mistake.
        # Saying so beats dying, and dying here would take the whole screen.
        return "proteome_too_large", _NO_KMERS
    if kmers.size == 0:
        # Records parsed but none held a 9-mer: a fragment or peptide FASTA,
        # not a proteome. Comparing against it finds nothing and would read as
        # a clear, so it is the same answer as an empty file.
        return "proteome_empty", _NO_KMERS
    if kmers.size < _PROTEOME_MIN_KMERS:
        # Too small to be a proteome, and the shortfall is what makes a clear
        # wrong: every 9-mer the file does not hold is an off-target this
        # cannot see. A UniProtKB export with the Sequence column unticked
        # parsed as one record of 104 distinct 9-mers and cleared; a single
        # albumin fragment gave 52. The human reference carries about 10.8
        # million, so this floor sits an order of magnitude below it.
        return "proteome_too_small", _NO_KMERS
    loaded: tuple[str | None, np.ndarray] = (None, kmers)
    while len(_PROTEOME_CACHE) >= _PROTEOME_CACHE_MAX:
        _PROTEOME_CACHE.pop(next(iter(_PROTEOME_CACHE)))
    _PROTEOME_CACHE[key] = loaded
    return loaded


def _is_nucleotide(records: list[str]) -> bool:
    """Whether these records read as DNA or RNA rather than protein.

    A, C, G, T and N are all amino acid letters too, so this is a share and not
    a membership test: in a protein they are about a fifth of the residues, and
    in a coding sequence they are all of them.
    """
    counted = 0
    nucleotide = 0
    for record in records:
        counted += len(record)
        nucleotide += sum(1 for letter in record if letter in _NUCLEOTIDE_LETTERS)
        if counted >= _ALPHABET_SAMPLE:
            break
    if counted < _ALPHABET_MIN:
        return False
    return nucleotide / counted >= _NUCLEOTIDE_SHARE


def _kmer_digest(kmers: np.ndarray) -> str:
    """A short digest of the packed 9-mer set.

    Of the parsed k-mers rather than the file, so the same proteome written
    with different line wrapping or record order digests the same, and a
    different proteome does not. Identity only; nothing is refused on it.

    Held against the array it was taken over, because every design's record
    carries the digest and the array is 90 MB for a human proteome: re-hashing
    it per design made the digest 8 of the 8.6 seconds a 200-design run took,
    against the 5 microseconds the search itself costs. Keyed on the array
    object rather than its contents, since hashing the contents is the cost
    being avoided; the reference kept here is what makes that identity sound,
    and the bound is the proteome cache's, which holds the same arrays.
    """
    for cached, digest in _DIGEST_CACHE:
        if cached is kmers:
            return digest
    digest = hashlib.sha256(kmers.tobytes()).hexdigest()[:16]
    _DIGEST_CACHE.append((kmers, digest))
    while len(_DIGEST_CACHE) > _PROTEOME_CACHE_MAX:
        _DIGEST_CACHE.pop(0)
    return digest


def _fasta_records(text: str) -> list[str]:
    """The residue runs of a FASTA, one string per record.

    A byte-order mark is stripped first. Excel and several export tools write
    one, and it sits in front of the first ``>``: that header then did not
    start a record and its own text -- accession, entry name, description --
    was read as residues, so letters out of a description joined the first
    record's 9-mers and a design matching one of them read as a proteome hit.

    A line opening with ``;`` is a comment. The original FASTA definition
    allows them, and reading one as residues does the same thing: its words
    become 9-mers of a proteome that does not contain them.
    """
    records: list[str] = []
    chunk: list[str] = []
    headed = False
    for line in text.lstrip("﻿").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(";"):
            continue
        if stripped.startswith(">"):
            headed = True
            if chunk:
                records.append("".join(chunk))
                chunk = []
            continue
        chunk.append("".join(stripped.split()).upper())
    if chunk:
        records.append("".join(chunk))
    if not headed and not records:
        return []
    return records


def _packed_kmers(sequence: str) -> np.ndarray:
    """The distinct 9-mers of a sequence, packed and sorted.

    A residue outside the twenty standard letters ends the run it is in, so no
    9-mer spans one. The result is what the proteome is compared against, and
    both sides use this function so the two cannot disagree.
    """
    found: list[int] = []
    token: list[str] = []
    for char in sequence.upper():
        if char in _AA:
            token.append(char)
            continue
        found.extend(_packed_windows("".join(token)))
        token = []
    found.extend(_packed_windows("".join(token)))
    if not found:
        return _NO_KMERS
    return np.unique(np.fromiter(found, dtype=np.int64, count=len(found)))


def _packed_windows(sequence: str) -> list[int]:
    """Every 9-mer of one run of standard residues, each as a packed integer.

    The window slides by shifting in the next residue and masking the oldest
    one off, so the whole run costs one pass rather than a slice per window.
    """
    if len(sequence) < _KMER:
        return []
    codes = [_RESIDUE_CODE[char] for char in sequence]
    packed = 0
    for code in codes[:_KMER]:
        packed = (packed << _RESIDUE_BITS) | code
    found = [packed]
    mask = (1 << (_RESIDUE_BITS * _KMER)) - 1
    for code in codes[_KMER:]:
        packed = ((packed << _RESIDUE_BITS) | code) & mask
        found.append(packed)
    return found


def _shared_count(design: np.ndarray, table: np.ndarray) -> int:
    """How many of the design's distinct 9-mers the proteome holds.

    Both sides are sorted and unique, so this is one binary search per design
    9-mer. Nothing builds a set of the whole proteome.
    """
    if design.size == 0 or table.size == 0:
        return 0
    index = np.searchsorted(table, design)
    index[index >= table.size] = table.size - 1
    return int(np.count_nonzero(table[index] == design))


def _proteome_record(sequence: str | None, loaded: tuple[str | None, np.ndarray]) -> dict[str, object]:
    blocker, kmers = loaded
    if blocker is not None:
        return _axis("untested", None, blocker)
    if not sequence:
        return _axis("untested", None, "sequence_absent")
    design = _packed_kmers(sequence)
    if design.size == 0:
        return _axis("untested", None, "sequence_absent")
    count = _shared_count(design, kmers)
    record = _axis("hit" if count else "clear", count, None)
    # A clear here means "no shared 9-mer in THIS file", so which file it was is
    # part of the claim. The size keeps a clear against a small proteome from
    # reading like one against the reference, and the digest makes a swapped or
    # rebuilt proteome visible between runs. Neither refuses anything: a file
    # nobody but its author would supply is their problem, not a guard's.
    record["proteome_kmers"] = int(kmers.size)
    record["proteome_digest"] = _kmer_digest(kmers)
    return record


def _rollup_proteome(
    samples: list[tuple[str, str | None]],
    by_example: dict[str, dict[str, object]],
    loaded: tuple[str | None, np.ndarray],
) -> dict[str, object]:
    """One proteome verdict for the run, by the rule every other axis uses.

    This axis applied that rule by hand, and ``rollup_axis`` says what the
    second copy costs: "a safety rule written twice is a safety rule that
    drifts." It had drifted. The copy kept a hit's measurement with
    ``isinstance(count, int)``, which admits ``True`` -- a bool is an int in
    Python -- and rejects a float, while the shared rule rejects the bool and
    keeps the float. Nothing in a stored screen was wrong, because
    ``_shared_count`` returns a real int and this axis has never measured
    anything else, and that is exactly how a divergence like this survives in
    a safety rule: the copy is only wrong about values its own axis cannot
    produce, until the day it can.

    Two other differences went with it, and both are gone by delegating rather
    than by patching. The hand-rolled hit built a fresh record and re-stamped
    the proteome provenance from ``loaded``; the shared rule carries the
    design's own record, which already holds the same two keys from the same
    array. And its undecided branch invented ``sequence_absent`` instead of
    carrying the first undecided design's blocker -- the same answer wherever
    this axis can reach it, since every design of an unloadable proteome
    carries that load blocker, but invented rather than read.

    ``empty`` is the no-design record, which is what a run with no samples at
    all answers with and where the load blocker comes from.
    """
    return rollup_axis(
        [example_id for example_id, _sequence in samples],
        by_example,
        _proteome_record(None, loaded),
    )


def _mhc_record(
    directory: Path,
    stem: str,
    *,
    rank_max: float,
    ic50_max: float,
    samples: list[tuple[str, str | None]],
    binder_kind: str,
) -> dict[str, object]:
    """Binding rows for one class, judged against what tolerated proteins carry.

    These cutoffs were calibrated for picking candidate epitopes out of a
    peptide library. Across every window of a 60-110 residue protein they are
    met by serum albumin, by IgG and by ubiquitin, so "any binding row is a hit"
    would mark every design and could never clear -- and a clear defined as "no
    binding window anywhere" would be a claim no human protein satisfies.

    So for a protein the measurement is binding peptides per 100 windows,
    compared with the same count over the reference proteins scored by the same
    tool. Without that reference there is no bar, and the density is recorded
    while the axis stays untested rather than guessing which side of it to fall.

    A peptide binder keeps the window rule, because for an 8-20mer "it has no
    binding window" is a statement that means something.
    """
    path = _table_path(directory, stem)
    if path is None:
        return _axis("untested", None, "predictor_not_configured")
    rows = _table_rows(path)
    if rows is None:
        return _axis("untested", None, "predictor_unreadable")
    unreadable = _unparsed_axis(rows)
    if unreadable is not None:
        return unreadable
    peptides, ranks, ic50s, named, unnamed_binding, scored = _binding_peptides(rows, rank_max, ic50_max)
    # An affinity column the table does not state in nanomolar is not read at
    # all -- see ``_float_cell`` -- so it is named on the record: a verdict has
    # to say that a column of numbers was left out of the binding call.
    not_nm = sorted(_not_nm_columns(rows))
    if not ranks and not ic50s:
        if not_nm:
            refused = _axis("untested", None, "ic50_not_nm")
            refused["not_nm_columns"] = not_nm
            return refused
        return _axis("untested", None, "score_absent")
    hit = any(rank <= rank_max for rank in ranks) or any(ic50 <= ic50_max for ic50 in ic50s)
    best_rank = min(ranks) if ranks else None
    best_ic50 = min(ic50s) if ic50s else None
    by_rank = best_rank is not None and best_rank <= rank_max
    by_ic50 = best_ic50 is not None and best_ic50 <= ic50_max
    # The reported number is the one that decided the verdict. Reporting the best
    # rank for a verdict an affinity triggered named a number that cleared.
    trigger = "rank" if by_rank else "ic50" if by_ic50 else None
    if trigger == "ic50":
        measurement = _round4(best_ic50)
    else:
        measurement = _round4(best_rank if best_rank is not None else best_ic50)
    # Peptides above the cutoff still say which chain was scanned. Ignoring
    # them makes an empty binding set a density of zero, and that zero clears
    # against a reference the table never measured for this design.
    provenance = _table_belongs(named, samples)[0] if named else None
    if provenance is not None:
        record = _axis("untested", measurement, provenance)
    elif binder_kind.strip().casefold() == "peptide":
        # A peptide design is one window, so a table that names a peptide
        # decides the axis outright. A table naming none cannot be tied to this
        # molecule by anything in this module -- but only the CLEAR needs that
        # tie. A rank under the cutoff is positive evidence whatever the file
        # calls the peptide it scored; an absence of binders is a claim about
        # this design, and an anonymous file cannot make it. Refusing both was
        # worse than the false clear it replaced: the strongest class I signal
        # there is, a rank of 0.4, came back as untested.
        if hit:
            record = _axis("hit", measurement, None)
        elif not named:
            record = _axis("untested", measurement, "peptide_column_absent")
        else:
            record = _axis("clear", measurement, None)
    elif not named:
        # For every other kind the table has to say which chain it scanned
        # before it can support a verdict of either sign: the count of binders
        # over a chain is only a count if the rows belong to that chain.
        record = _axis("untested", measurement, "peptide_column_absent")
    elif unnamed_binding:
        # A binding rank with no peptide is not a density of zero. Named
        # non-binders would otherwise clear the axis while that rank stays
        # under the cutoff and is never attributed to this chain.
        record = _axis("untested", measurement, "peptide_column_absent")
    else:
        record = _density_verdict(
            directory,
            stem,
            peptides=peptides,
            samples=samples,
            named=named,
            scanned=scored,
            rank_max=rank_max,
            ic50_max=ic50_max,
            design_methods=_table_methods(rows),
            design_alleles=_table_alleles(rows),
            design_widths=_table_widths(rows),
            fallback=measurement,
        )
    if not_nm:
        # A hit stands: a liability measured on the numbers that could be read
        # is not deleted by a sibling column whose unit the table never stated.
        # A clear is the other shape -- it says no row in the file binds, and
        # rows whose numbers were never read cannot support that.
        record["not_nm_columns"] = not_nm
        if record["state"] == "clear":
            record["state"] = "untested"
            record["blocker"] = "ic50_not_nm"
    record["rank"] = None if best_rank is None else _round4(best_rank)
    record["ic50"] = None if best_ic50 is None else _round4(best_ic50)
    record["trigger"] = trigger
    # Windows the table declined to score. They are not non-binders, and a
    # verdict that depended on them should say how many there were.
    record["unscored_peptides"] = len(named - scored)
    record.update(_coverage_fields(named, samples))
    return record


def _density_verdict(
    directory: Path,
    stem: str,
    *,
    peptides: set[str],
    samples: list[tuple[str, str | None]],
    named: set[str],
    scanned: set[str],
    rank_max: float,
    ic50_max: float,
    design_methods: set[str],
    design_alleles: frozenset[str],
    design_widths: frozenset[int],
    fallback: float,
) -> dict[str, object]:
    """Compare the design's binding density with the reference proteins'.

    ``fallback`` is the best rank or affinity, kept as the measurement when
    there is no density to report, so an unconfigured reference still shows the
    number the table held.
    """
    if not design_widths:
        # Every peptide the table names is of a width no binding cutoff is
        # defined for. ``_SCORED_WIDTHS`` holds the ranges these cuts come from
        # -- IEDB class I 8 to 11, class II 13 to 25 -- and 12 falls between
        # them, so a table of 12-mers carries no binding call this module can
        # make. Both sides' width sets are then empty, which made
        # ``reference_width_mismatch`` pass trivially and ``_scan_covers``
        # always false, and what the reader got was ``scan_incomplete``, which
        # sends them to re-export a file that is already complete, or
        # ``sequence_absent``, which names a chain that is present.
        record = _axis("untested", fallback, "width_without_cutoff")
        record["method"] = _DENSITY_METHOD
        return record
    mismatch, density, matched, coverage = _design_density(
        peptides,
        samples,
        named=named,
        scanned=scanned,
        widths=design_widths,
    )
    if mismatch is not None or density is None:
        record = _axis("untested", fallback, mismatch or "sequence_absent")
        record["method"] = _DENSITY_METHOD
        return record
    # The reference is read before the design's coverage is weighed, so a
    # reference the user can fix is named as the cause. Reporting the design's
    # coverage first sent someone with a filtered albumin export away to
    # re-scan a design that was scanned correctly.
    blocker, reference, detail = _reference_density(
        directory,
        stem,
        rank_max=rank_max,
        ic50_max=ic50_max,
        design_methods=design_methods,
        design_alleles=design_alleles,
        design_widths=design_widths,
    )
    rounded = _round4(density)
    if blocker is not None or reference is None:
        record = _axis("untested", rounded, blocker)
    elif density > reference:
        # Over the bar on the windows that were read. A wider scan can only add
        # binders, so a hit stands however much of the chain was covered, and
        # suppressing it would make the screen quieter about a real liability.
        # The coverage rides along so the reader knows what the number rests on.
        record = _axis("hit", rounded, None)
        record["reference_density"] = _round4(reference)
        if coverage is not None:
            record["scan_limit"] = coverage
    elif coverage is not None:
        # Under the bar, but only across the part that was read, and the
        # unread remainder is exactly where the missing binders would be. A
        # partial scan understates the numerator and keeps the whole chain as
        # the denominator, so the error always runs toward clear: a table
        # covering 20 of a chain's 68 windows cleared at 1.32 against a 1.64
        # bar where the complete scan of the same chain is a hit at 5.88.
        # The bar is left off the record because no comparison was made.
        record = _axis("untested", rounded, coverage)
    else:
        record = _axis("clear", rounded, None)
        record["reference_density"] = _round4(reference)
    record["method"] = _DENSITY_METHOD
    record["binding_peptides"] = matched
    record.update(detail)
    return record


def _score_record(directory: Path, stem: str, *, method: str) -> dict[str, object]:
    """A supplied score is stored. It does not become hit or clear.

    ``method`` travels with the number so the reported line can name where it
    came from. Without it the axis was dropped from that line entirely, which
    hid a supplied score rather than reporting it.
    """
    path = _table_path(directory, stem)
    if path is None:
        return _axis("untested", None, "predictor_not_configured")
    rows = _table_rows(path)
    if rows is None:
        return _axis("untested", None, "predictor_unreadable")
    unreadable = _unparsed_axis(rows)
    if unreadable is not None:
        return unreadable
    scores = [score for row in rows if (score := _float_cell(row, _SCORE_COLUMNS)) is not None]
    if not scores:
        return _axis("untested", None, "score_absent")
    record = _axis("untested", _round4(max(scores)), "no_pass_cut")
    record["method"] = method
    return record


def _immunogenicity_record(directory: Path, samples: list[tuple[str, str | None]]) -> dict[str, object]:
    """Calis scores over the 9-mers a class I table says are presented.

    The Calis sign was fit on predicted MHC ligands, not on every window of a
    protein, so scanning a whole chain asks the model a question it was not
    fitted to answer. A real binder had 359 windows above zero and could not
    clear whatever its sequence was. Restricting the scan to the 9-mers a class
    I predictor calls presented is the question the model does answer, and it
    makes the axis two-sided: a presented ligand above zero is a hit, and none
    above zero is a clear.

    Without a class I table that names peptides there is nothing to restrict
    to. Then the whole-chain maximum is reported as before and still only as
    information, and the record says the restriction was not configured.
    """
    if _table_path(directory, "immunogenicity") is not None:
        record = _score_record(directory, "immunogenicity", method=SUPPLIED_METHOD)
        # A supplied score has no pass cut, so this axis can no longer be hit
        # or clear -- and the same filename therefore means both "my scores"
        # and "turn this axis off". When the class I table that drives the
        # Calis verdict is sitting beside it, the record says which
        # computation the file displaced, so an axis that reads untested on a
        # directory holding the data for a verdict can be read as such.
        #
        # The precedence itself is left alone: a stored score does not become
        # hit or clear (see ``_score_record``), nothing published says a
        # supplied number should yield to a computed one, and the existing
        # suite pins that order. Naming the displacement is the part that was
        # missing.
        if _table_path(directory, "mhc_i") is not None:
            record["displaced_method"] = _CALIS_METHOD
            record["displaced_table"] = "mhc_i"
        return record
    if not samples:
        return _axis("untested", None, "predictor_not_configured")
    # The class I axis refuses a table holding a cell it cannot read, and the
    # restriction rests on the same numbers: a comma-decimal rank dropped the
    # four binding windows from the presented set, leaving one benign window,
    # and this axis cleared on it while ``mhc_i`` reported ``cell_unparsed``.
    restricting = _table_path(directory, "mhc_i")
    rows = _table_rows(restricting) if restricting is not None else None
    if rows:
        refused = _unparsed_axis(rows)
        if refused is not None:
            refused["method"] = _CALIS_METHOD
            refused["presentation"] = "restricted"
            return refused
    scanned = _presented_peptides(directory)
    if scanned is not None:
        presented, named, binders, scanned_whole, scored = scanned
        blocker, _sequence = _table_belongs(named, samples)
        if blocker is not None:
            record = _axis("untested", None, blocker)
            record["method"] = _CALIS_METHOD
            record["presentation"] = "restricted"
            return record
        found = _presented_immunogenicity(
            samples,
            presented,
            binders,
            scanned=scored,
            scanned_whole=scanned_whole,
        )
        # The presented set is only as wide as the numbers that could be read,
        # and an affinity column of unstated unit is not one of them. A hit on
        # the ligands that were read stands; a clear says every presented 9-mer
        # is at most zero, and a reduced presented set cannot say that.
        not_nm = sorted(_not_nm_columns(rows or []))
        if not_nm:
            found["not_nm_columns"] = not_nm
            if found["state"] == "clear":
                found["state"] = "untested"
                found["blocker"] = "ic50_not_nm"
        found.update(_coverage_fields(named, samples))
        # Which predictor called these 9-mers presented. The Calis verdict is
        # two-sided only because a class I table restricted the scan, so that
        # table's tool is part of the claim: without it a hit from one predictor
        # and a hit from another are the same record.
        found["restricting_method"] = sorted(_table_methods(rows or []))
        return found
    scored_max: float | None = None
    positive = 0
    positive_examples: list[str] = []
    clear_count = 0
    scored_samples = 0
    for example_id, sequence in samples:
        scores = _calis_scores(sequence)
        if not scores:
            continue
        scored_samples += 1
        local_max = max(scores)
        local_positive = sum(1 for score in scores if score > 0)
        positive += local_positive
        scored_max = local_max if scored_max is None else max(scored_max, local_max)
        if local_positive:
            positive_examples.append(example_id)
        else:
            clear_count += 1
    if scored_samples == 0 or scored_max is None:
        record = _axis("untested", None, "sequence_absent")
        record["positive_windows"] = 0
        record["positive_examples"] = []
        return record
    measurement = _round5(scored_max)
    if positive:
        record = _axis("untested", measurement, "no_pass_cut")
    elif clear_count == len(samples):
        record = _axis("clear", measurement, None)
    else:
        record = _axis("untested", measurement, "sequence_absent")
    record["positive_windows"] = positive
    record["positive_examples"] = positive_examples
    record["method"] = _CALIS_METHOD
    record["presentation"] = "not_configured"
    return record


def _presented_peptides(directory: Path) -> tuple[dict[str, set[str]], set[str], int, bool, set[str]] | None:
    """The presented 9-mers, every named peptide, how many bound, and whether the
    table also holds rows the cutoff rejected.

    Each presented 9-mer carries the alleles its binding rows named. Calis et
    al. make the masked anchor positions per molecule, so the restricting
    allele is what says whether the mask this module can compute is the one
    that molecule needs. It was read off every row and thrown away; see
    ``_CALIS_MASK``.

    The scored set is the windows that carried a readable rank or affinity. A
    rank cell reading ``-`` or ``n.d.`` is the predictor declining that window,
    and crediting it as scanned cleared this axis while 54 Calis-positive
    windows of the chain sat unscored, the highest of them at +0.41.

    None when the table cannot restrict anything: absent, unreadable, or
    holding ranks without naming the peptides they belong to. The second set is
    every named peptide whatever its length, which is what says whether the
    table was scanned from this design -- the presented subset is too small and
    too filtered to answer that.

    Only a 9-mer of standard residues can be scored, because the weights being
    restricted are the 9-mer weights.
    """
    path = _table_path(directory, "mhc_i")
    if path is None:
        return None
    rows = _table_rows(path)
    if not rows:
        return None
    presented: dict[str, set[str]] = {}
    named: set[str] = set()
    bound: set[str] = set()
    scored: set[str] = set()
    binders = 0
    for row in rows:
        peptide = _cell(row, _PEPTIDE_COLUMNS).upper()
        if not peptide:
            continue
        named.add(peptide)
        rank = _float_cell(row, _RANK_COLUMNS)
        ic50 = _float_cell(row, _IC50_COLUMNS)
        if rank is not None or ic50 is not None:
            scored.add(peptide)
        binds = (rank is not None and rank <= _MHC_I_RANK) or (ic50 is not None and ic50 <= _MHC_I_IC50)
        if not binds:
            continue
        bound.add(peptide)
        binders += 1
        if len(peptide) == _KMER and all(char in _AA for char in peptide):
            allele = _allele_key(_cell(row, _ALLELE_COLUMNS))
            restricting = presented.setdefault(peptide, set())
            if allele:
                restricting.add(allele)
    if not named:
        return None
    # Whether the table holds rows the cutoff rejected. One that does is a scan
    # and can be asked how much of the chain it covered; one holding only its
    # binders is a filtered export of the presented set, and its extent is not
    # knowable from the file.
    return presented, named, binders, bool(scored - bound), scored


def _presented_immunogenicity(
    samples: list[tuple[str, str | None]],
    presented: dict[str, set[str]],
    binders: int,
    *,
    scanned: set[str],
    scanned_whole: bool,
) -> dict[str, object]:
    """Calis over the presented 9-mers only. One above zero is a hit.

    A design whose presented ligands all score at or below zero is clear. A
    design with no presented ligand at all is neither: there was nothing for
    the model to score, which is not the same as nothing to worry about.

    A clear from a table that is a whole scan also needs that scan to have
    covered the chain. The presented set is only as complete as the table it
    came from: one covering 40 of a chain's 84 windows presented a single ligand
    scoring -0.107 and cleared, while 45 of that chain's windows are
    Calis-positive.

    ``scanned_whole`` is what separates that from a legitimate filtered export.
    A table holding rows the cutoff rejected is a scan, and a scan can be asked
    how much it covered. A table holding only its binders is the presented set
    as the user chose to export it, and its extent cannot be read off the file,
    so restriction is taken at its word -- which is the point of restricting.

    A hit needs no such check either way: a positive ligand found is positive
    evidence whatever else was skipped.

    Nor can a clear rest on a ligand restricted by a molecule whose anchors
    the paper puts elsewhere: the mask applied is the default one, and for
    HLA-B*08:01 the paper says that is the wrong mask and publishes no weight
    for the right one. See ``_CALIS_MASK``.
    """
    restricting = sorted({allele for alleles in presented.values() for allele in alleles})
    exceptions = sorted(_calis_exceptions(restricting))
    scored_max: float | None = None
    positive = 0
    positive_examples: list[str] = []
    ligands = 0
    covered: set[str] = set()
    for example_id, sequence in samples:
        if not sequence:
            continue
        windows = [window for window in _slices(sequence, _KMER) if window in presented]
        if not windows:
            continue
        covered.update(windows)
        ligands += len(windows)
        scores = [_calis_score(window) for window in windows]
        local_positive = sum(1 for score in scores if score > 0)
        positive += local_positive
        local_max = max(scores)
        scored_max = local_max if scored_max is None else max(scored_max, local_max)
        if local_positive:
            positive_examples.append(example_id)
    # A presented 9-mer that is not a window of any sample was not scored.
    # Clearing on the ones that were would say every presented 9-mer is at most 0.
    if set(presented) - covered:
        # Untested, because clearing on the windows that were scored would claim
        # every presented 9-mer is at most 0 when one of them was never scored.
        #
        # The counts stay TRUE, though. Overwriting them with 0 and [] used to
        # say none of the scored windows reached the cut while
        # ``presented_ligands`` said one had been scored and it had in fact
        # scored 0.272 -- one record contradicting itself, against the meaning
        # ``mcp_api`` publishes for these two keys. ``presented_complete`` is
        # what says the set is short; the counts say what was actually found,
        # and a reader needs them precisely because the scan is incomplete.
        record = _axis("untested", None, "presented_not_in_design")
        record["positive_windows"] = positive
        record["positive_examples"] = sorted(set(positive_examples))
        record["presented_complete"] = False
    elif scored_max is None:
        # Three different reasons there was nothing to score, and one blocker
        # for all three sent the user looking for data they already had. A
        # table presenting only 10- and 11-mers did present ligands; it just
        # presented none these 9-mer weights can score.
        if presented:
            blocker = "presented_not_in_design"
        elif binders:
            blocker = "no_nine_mer_ligand"
        else:
            blocker = "no_presented_ligand"
        record = _axis("untested", None, blocker)
        record["positive_windows"] = 0
        record["positive_examples"] = []
        record["presented_complete"] = False
    elif (
        scanned_whole
        and not positive
        and not any(_scan_covers(scanned, sequence) for _example_id, sequence in samples if sequence)
    ):
        record = _axis("untested", _round5(scored_max), "scan_incomplete")
        record["positive_windows"] = 0
        record["positive_examples"] = []
        record["presented_complete"] = False
    elif exceptions and not positive:
        # Calis et al. name HLA-B*08:01 as a molecule whose anchors are P2, P5
        # and P9 rather than the P1, P2 and P9 this mask excludes, and the
        # paper's position importances are published for the default masking
        # only, so no score for that molecule can be computed from anything
        # published. A clear is an absence claim about a number this module
        # cannot compute. A positive ligand found under the default mask is
        # still a positive ligand, so a hit is not suppressed by this.
        record = _axis("untested", _round5(scored_max), "calis_mask_unpublished_for_allele")
        record["positive_windows"] = 0
        record["positive_examples"] = []
        record["presented_complete"] = False
        record["mask_unpublished_alleles"] = exceptions
    else:
        record = _axis("hit" if positive else "clear", _round5(scored_max), None)
        record["positive_windows"] = positive
        record["positive_examples"] = positive_examples
        record["presented_complete"] = True
    record["method"] = _CALIS_METHOD
    record["presentation"] = "restricted"
    record["presented_ligands"] = ligands
    # Which mask the score applied, and the alleles the ligands were presented
    # by. The excluded anchors are per molecule in the paper and one vector is
    # applied to all of them here, so the record says that outright rather than
    # letting the method name imply an allele-specific treatment.
    record["calis_mask"] = list(_CALIS_MASK)
    record["calis_mask_allele_specific"] = False
    record["restricting_alleles"] = restricting
    return record


def _ada_record(directory: Path, samples: list[tuple[str, str | None]]) -> dict[str, object]:
    """Class II allele breadth over the DR panel, unless an ADA table is present.

    The stored number is the best median panel rank any peptide reached. A
    median at or below the threshold is a promiscuous binder, which is the risk
    signal: such a peptide finds a restricting allele in most people.

    The table has to cover every panel allele for at least one peptide. A rule
    whose threshold is a median over seven cannot be applied to four of them, so
    a partial panel stays untested and names what was missing. No peptide string
    is stored, only counts, because a peptide of a designed chain is sequence.

    The table also has to be this design's. Nothing tied the two together
    before, so a class II table left in the directory by an earlier run cleared
    this design on peptides belonging to another one.
    """
    if _table_path(directory, "ada") is not None:
        record = _score_record(directory, "ada", method=SUPPLIED_METHOD)
        # The same displacement as on the immunogenicity axis, and named the
        # same way: with mhc_ii.tsv beside it the DR-breadth verdict was
        # available, and a reader of the record could not tell.
        if _table_path(directory, "mhc_ii") is not None:
            record["displaced_method"] = _ADA_METHOD
            record["displaced_table"] = "mhc_ii"
        return record
    path = _table_path(directory, "mhc_ii")
    if path is None:
        return _axis("untested", None, "predictor_not_configured")
    rows = _table_rows(path)
    if rows is None:
        return _axis("untested", None, "predictor_unreadable")
    unreadable = _unparsed_axis(rows)
    if unreadable is not None:
        return unreadable
    if not any(_cell(row, _ALLELE_COLUMNS) for row in rows):
        return _axis("untested", None, "allele_column_absent")
    if not any(_cell(row, _PEPTIDE_COLUMNS) for row in rows):
        return _axis("untested", None, "peptide_column_absent")
    if not _declares_consensus(rows):
        return _axis("untested", None, "rank_scale_unknown")
    # ``_declares_consensus`` above is a table-level ``any()``, so in a file
    # merged from two exports one row saying consensus licensed the median over
    # rows another tool scored, whose 20 is not the paper's 20. Keep only the
    # rows that declare the scale themselves. A row naming no method at all is
    # left where the table-level answer put it, so a single-method table, a
    # table with blank method cells, and a table declaring the scale by its
    # rank column alone all read exactly as before; only a row naming a
    # different method is dropped, and a mixed table then reports
    # ``allele_panel_incomplete`` or ``rank_absent`` and names what is missing
    # instead of publishing a median across two scales.
    #
    # What makes that safe is the refusal immediately above: a table that
    # declares no scale is already untested, so this filter only ever sees a
    # table that declared one, and the rows it keeps for having no method cell
    # are rows that table already vouched for.
    rows = [row for row in rows if _declares_consensus([row]) or not _cell(row, _METHOD_COLUMNS)]
    wanted = {_allele_key(name) for name in _DR_PANEL}
    by_peptide: dict[str, dict[str, float]] = {}
    for row in rows:
        peptide = _cell(row, _PEPTIDE_COLUMNS).upper()
        # Read to the molecule, so a row spelling a panel allele with a third
        # or fourth field counts toward the panel it names. Skipped, it made
        # ``missing_alleles`` list the alleles the user had just scored.
        allele = _allele_molecule(_allele_key(_cell(row, _ALLELE_COLUMNS)), wanted)
        rank = _float_cell(row, _RANK_COLUMNS)
        if not peptide or allele is None or rank is None:
            continue
        best = by_peptide.setdefault(peptide, {})
        if allele not in best or rank < best[allele]:
            best[allele] = rank
    full = {peptide: ranks for peptide, ranks in by_peptide.items() if len(ranks) == len(wanted)}
    if not by_peptide and _panel_rows_present(rows, wanted):
        # The panel is covered but no row carried a rank. Reporting that as an
        # incomplete panel named the exact alleles the user had just scored, and
        # the remedy for that blocker is to go and score them -- a loop. The
        # median rule reads the rank column only, so an affinity-only table is
        # missing the number, not the alleles.
        return _axis("untested", None, "rank_absent")
    if not full:
        record = _axis("untested", None, "allele_panel_incomplete")
        seen = {allele for ranks in by_peptide.values() for allele in ranks}
        record["missing_alleles"] = sorted(wanted - seen)
        return record
    belongs, _sequence = _table_belongs(set(full), samples)
    if belongs is not None:
        record = _axis("untested", None, belongs)
        record["method"] = _ADA_METHOD
        record["peptides_scored"] = len(full)
        return record
    medians = [_median(sorted(ranks.values())) for ranks in full.values()]
    promiscuous = sum(1 for value in medians if value <= _DR_MEDIAN_MAX)
    # The median rule is applied per peptide, so "this design holds no
    # promiscuous binder" is a claim about every window of it and needs a
    # table that scanned them -- the rule ``_scan_is_complete`` applies to a
    # density. A table scoring 2 of a 76-residue chain's 62 fifteen-mers
    # cleared this axis on 3 per cent of the chain, where the complete scan of
    # the same chain is a hit at a median of 5.0.
    #
    # A hit needs no coverage, and the asymmetry is deliberate: one
    # promiscuous peptide is positive evidence however little else was
    # scanned, while an absence is only as wide as the scan that looked.
    if not promiscuous and not any(_scan_covers(set(full), sequence) for _example_id, sequence in samples if sequence):
        record = _axis("untested", _round4(min(medians)), "scan_incomplete")
        record["method"] = _ADA_METHOD
        record["alleles"] = len(wanted)
        record["peptides_scored"] = len(full)
        record.update(_coverage_fields(set(full), samples))
        return record
    record = _axis("hit" if promiscuous else "clear", _round4(min(medians)), None)
    record["method"] = _ADA_METHOD
    record["alleles"] = len(wanted)
    record["peptides_scored"] = len(full)
    record["promiscuous_peptides"] = promiscuous
    record["binding_alleles"] = max(
        sum(1 for rank in ranks.values() if rank <= _DR_RANK_BINDER) for ranks in full.values()
    )
    record.update(_coverage_fields(set(full), samples))
    return record


def _binding_peptides(
    rows: list[dict[str, str]],
    rank_max: float,
    ic50_max: float,
) -> tuple[set[str], list[float], list[float], set[str], bool, set[str]]:
    """Binding peptides, every number, every named peptide, whether a binding
    row named no peptide, and which peptides carried a readable number.

    The last of those is what scan coverage is counted over. A table whose rank
    cell reads ``-`` or ``n.d.`` is saying it declined to score that window, and
    counting such a window as scanned let four unscored 9-mers pass as
    non-binders: the record said zero binding peptides, measured, no blocker,
    while the same table with those four cells filled is a hit at 4.35.

    A table of ranks with no peptide column yields no binding peptides, which
    must not be read as a density of zero. A named peptide above the cutoff is
    still a window of the chain that was scanned. A binding rank on a row that
    names nothing is a third fact: the density never counts it.
    """
    peptides: set[str] = set()
    ranks: list[float] = []
    ic50s: list[float] = []
    named: set[str] = set()
    scored: set[str] = set()
    unnamed_binding = False
    for row in rows:
        rank = _float_cell(row, _RANK_COLUMNS)
        ic50 = _float_cell(row, _IC50_COLUMNS)
        if rank is not None:
            ranks.append(rank)
        if ic50 is not None:
            ic50s.append(ic50)
        binds = (rank is not None and rank <= rank_max) or (ic50 is not None and ic50 <= ic50_max)
        peptide = _cell(row, _PEPTIDE_COLUMNS).upper()
        if peptide:
            named.add(peptide)
            if rank is not None or ic50 is not None:
                scored.add(peptide)
        if binds and peptide:
            peptides.add(peptide)
        elif binds:
            unnamed_binding = True
    return peptides, ranks, ic50s, named, unnamed_binding, scored


def _coverage_fields(peptides: set[str], samples: list[tuple[str, str | None]]) -> dict[str, object]:
    """Which designs of the run this table is a scan of, and which it is not.

    ``_table_belongs`` accepts a table when it is a scan of the best-matching
    design, which is the right test for an inherited table. But a run holds
    many designs and one supplied table is one scan, and the verdict was then
    recorded for the whole run with nothing saying whose chain it had been
    taken from: a run of eight could read ``mhc_ii clear`` on a scan of one,
    and seven designs no predictor had seen were covered by that word.

    This does not move the state. It records what the state covers, so the
    claim can be read against the designs it was actually made about. The
    same share decides coverage here as decides provenance there, so a design
    is covered exactly when the table would have been accepted as its scan.
    """
    if not peptides or not samples:
        return {}
    need = max(1, int(len(peptides) * _PROVENANCE_SHARE))
    covered: list[str] = []
    uncovered: list[str] = []
    for example_id, sequence in samples:
        matched = sum(1 for peptide in peptides if peptide in sequence) if sequence else 0
        target = covered if matched >= need else uncovered
        target.append(example_id)
    found: dict[str, object] = {"covered_examples": covered}
    if uncovered:
        found["uncovered_examples"] = uncovered
    return found


def _table_belongs(
    peptides: set[str],
    samples: list[tuple[str, str | None]],
) -> tuple[str | None, str | None]:
    """The designed chain a table was scanned from, or a blocker saying it was not.

    A table's peptides are windows of the chain it was scanned from, so most of
    them occur in that chain and almost none occur in an unrelated one. A table
    left behind from an earlier design in the same directory shares only
    incidental windows -- a linker, a tag, a run of alanine -- and one shared
    window was enough to produce a confident verdict about a chain nothing had
    scanned.

    Requiring a share rather than a single match is what makes an inherited
    table fail here instead of clearing an axis. An empty peptide set is not a
    foreign table, only an absence of binders, so it returns no chain and no
    blocker and lets the caller decide.
    """
    if not peptides:
        return None, None
    best: tuple[int, str] | None = None
    for _example_id, sequence in samples:
        if not sequence:
            continue
        matched = sum(1 for peptide in peptides if peptide in sequence)
        if best is None or matched > best[0]:
            best = (matched, sequence)
    if best is None:
        return "sequence_absent", None
    matched, sequence = best
    if matched < max(1, int(len(peptides) * _PROVENANCE_SHARE)):
        return "table_not_from_this_design", None
    return None, sequence


def _scan_covers(named: set[str], sequence: str) -> bool:
    """Whether the table's peptides cover the chain they were scanned from.

    Asked only of a density of zero, because the two directions are different
    claims. A positive density says "these binders are in the chain", which a
    binders-only export supports. A zero says "there are none anywhere", and
    that is a statement about every window, so it needs a scan that looked at
    every window. This module already refuses the same thing on the reference
    side, where a filtered albumin export cannot set a bar.

    A table holding 40 of a chain's 84 9-mers, with the four binders at windows
    61 to 64 outside the file, reported zero binding peptides per 100 residues
    as a measurement and cleared the axis, and no field said how much had been
    scanned.

    Every width the table names has to be covered, not just one of them.
    Accepting any single width let a cheap width buy the credential: a file of
    85 8-mers plus 4 9-mers covered the 8-mers completely, covered 4 of the
    chain's 84 9-mers, and published a density of zero as measured. Coverage at
    one width says nothing about what another width would have found, and 8- to
    11-mers are all legitimate class I widths, so that file is one a real scan
    could produce.

    Only widths the cutoffs are calibrated for count. A dipeptide column cannot
    earn or deny the credential, because no binding call is defined for it.

    The denominator is the chain's *distinct* windows, not its positions. A
    hundred residues of alanine hold ninety-two 9-mer positions and one
    distinct 9-mer, so a table naming that one peptide has scanned the whole
    chain and its zero is a real zero.
    """
    counted: dict[int, int] = {}
    for peptide in named:
        width = len(peptide)
        if width in _SCORED_WIDTHS and peptide in sequence:
            counted[width] = counted.get(width, 0) + 1
    if not counted:
        return False
    for width, count in counted.items():
        windows = set(_slices(sequence, width))
        if not windows or count < len(windows) * _SCAN_COVER_SHARE:
            return False
    return True


def _window_count(length: int, widths: frozenset[int]) -> int:
    """How many windows a chain of this length holds at these widths.

    A count of distinct binding peptides is a count over windows, so this is
    the denominator that makes a short design and a long reference comparable.
    A chain shorter than a width holds no window of it and contributes none.
    """
    return sum(max(0, length - width + 1) for width in widths)


def _scan_is_complete(named: set[str], scanned: set[str], sequence: str) -> str | None:
    """A blocker when the file cannot speak for every window of the chain.

    Two conditions, and both are the same thought: a density is a count over
    the whole chain, so the whole chain has to have been looked at and scored.

    The scan has to cover the chain. And no window of the chain may have been
    left unscored -- a rank cell reading ``-`` or ``n.d.`` says the predictor
    declined that window, and four such cells in an 84-window scan still left
    80 covered, comfortably past the coverage share, while the record said zero
    binding peptides with no blocker. Coverage tolerates a gap because a
    predictor drops the odd window; a claim about every window cannot.

    One gate, called from both branches of ``_design_density``. Asking it only
    of a density of zero was the defect: a table covering 20 of a chain's 68
    windows and finding one binder there divided that 1 by the whole chain and
    cleared at 1.3158 against a 1.642 bar, where the complete scan of the same
    chain holds four binders and is a hit at 5.8824. A partial scan understates
    the numerator and keeps the full denominator, so it errs toward clear.
    """
    if not _scan_covers(scanned, sequence):
        return "scan_incomplete"
    if {peptide for peptide in named - scanned if peptide in sequence}:
        return "unscored_windows"
    return None


def _design_density(
    peptides: set[str],
    samples: list[tuple[str, str | None]],
    *,
    named: set[str],
    scanned: set[str],
    widths: frozenset[int],
) -> tuple[str | None, float | None, int, str | None]:
    """A blocker, binding peptides per 100 windows, the count, and a coverage blocker.

    The two blockers are not the same refusal. The first stops any verdict:
    without knowing which chain the rows belong to there is no density at all.
    The last stops only a CLEAR -- see :func:`_density_verdict`, which is where
    that distinction is applied.

    The table is one scan and a run holds several samples, so the chain the
    numerator belongs to is found rather than assumed, and only the peptides
    that chain contains are counted. Numerator and denominator are then always
    the same molecule.

    ``named`` is every peptide the table names, binding or not, which is what
    says how much of the chain was scanned. ``widths`` is the peptide widths
    the table scored, which is what the chain's windows are counted at: a
    count of 9-mers belongs over the chain's 9-mer windows, not its residues.
    """
    blocker, sequence = _table_belongs(peptides, samples)
    if blocker is not None:
        return blocker, None, 0, None
    if sequence is None:
        # The table named peptides and none of them bind, so the binding set is
        # empty and says nothing about which chain this is. Every named peptide
        # does, and it is the same question provenance already answers -- asking
        # by length instead picked the longest design in the run, so adding a
        # second, longer design to a folder made a complete and correct scan of
        # the first refuse with scan_incomplete.
        if not named:
            return "sequence_absent", None, 0, None
        blocker, chain = _table_belongs(named, samples)
        if blocker is not None:
            return blocker, None, 0, None
        if chain is None:
            return "sequence_absent", None, 0, None
        return _zero_density(named, scanned, chain)
    matched = sum(1 for peptide in peptides if peptide in sequence)
    if not matched:
        return _zero_density(named, scanned, sequence)
    windows = _window_count(len(sequence), widths)
    if not windows:
        # The table scored widths this chain is too short to hold, so there is
        # no window set the count belongs over and no density to report.
        return "sequence_absent", None, 0, None
    # The count stands whatever share of the chain was scanned: binders that
    # were found were found. Whether it may be compared down to a clear is the
    # caller's question, so the coverage answer travels beside the number
    # rather than replacing it. Suppressing the number here made a truncated
    # export that found twenty epitopes in twenty windows report nothing at
    # all, which is the one direction a liability screen must not move in.
    return None, matched * _DENSITY_BASIS / windows, matched, _scan_is_complete(named, scanned, sequence)


def _zero_density(
    named: set[str],
    scanned: set[str],
    sequence: str,
) -> tuple[str | None, float | None, int, str | None]:
    """Zero binding peptides per 100 windows, if the file can say that.

    A zero is the absence claim itself, so here the coverage gate is absolute
    rather than a condition on the comparison: a scan that did not read the
    whole chain cannot report that the chain holds nothing. Zero over any
    denominator is zero, so no window count is needed.
    """
    blocker = _scan_is_complete(named, scanned, sequence)
    if blocker is not None:
        return blocker, None, 0, None
    return None, 0.0, 0, None


def _table_methods(rows: list[dict[str, str]]) -> set[str]:
    """Every method a table declares, normalised. Empty when it declares none."""
    found = set()
    for row in rows:
        method = _cell(row, _METHOD_COLUMNS).casefold().replace("-", "_").replace(" ", "_")
        if method:
            found.add(method)
    return found


def _protein_lengths(rows: list[dict[str, str]]) -> dict[str, int]:
    """Each named protein's length, from a stated length or a complete scan.

    Two ways of getting this wrong are guarded here, and both inflate the bar,
    which always errs toward clearing the design.

    A stated length at or below the longest peptide in the table is a peptide
    length: some exports call that column ``length`` too. It is ignored rather
    than divided by.

    The furthest peptide end is the protein's length only if the scan covered
    the whole protein. A binders-only export ends at the last binder, far short
    of the protein: a filtered albumin table measured 6.25 binders per 100
    residues where the complete scan of the same protein measured 0.49. So the
    end is used only when some row starts at the protein's first residues,
    which is what a complete scan looks like. Without a start column there is
    no way to tell, and the caller is told the length is unknown.
    """
    stated: dict[str, int] = {}
    ends: dict[str, int] = {}
    starts: dict[str, int] = {}
    longest: dict[str, int] = {}
    for row in rows:
        protein = _cell(row, _PROTEIN_COLUMNS).upper()
        if not protein:
            continue
        peptide = _cell(row, _PEPTIDE_COLUMNS)
        if peptide:
            longest[protein] = max(longest.get(protein, 0), len(peptide))
        declared = _float_cell(row, _LENGTH_COLUMNS)
        if declared is not None and declared > 0:
            stated[protein] = max(stated.get(protein, 0), int(declared))
        end = _float_cell(row, _END_COLUMNS)
        if end is not None and end > 0:
            ends[protein] = max(ends.get(protein, 0), int(end))
        start = _float_cell(row, _START_COLUMNS)
        if start is not None and start > 0:
            starts[protein] = min(starts.get(protein, int(start)), int(start))
    lengths: dict[str, int] = {}
    for protein in {*stated, *ends}:
        declared = stated.get(protein, 0)
        if declared > longest.get(protein, 0):
            lengths[protein] = declared
            continue
        end = ends.get(protein, 0)
        first = starts.get(protein, 0)
        if end and first and first <= _SCAN_START_MAX:
            lengths[protein] = end
    return lengths


def _tolerated_accession(label: str) -> str | None:
    """The tolerated protein this label names, or None.

    Testing whether a label *contains* an accession let anything through:
    ``P02768-FRAGMENT-1-76`` and ``sp|Q9Y6K9|TNF_HUMAN,P02768`` both passed, and
    because the densities were keyed on the raw label, one protein under two
    spellings became two bar candidates -- so a 76-residue "fragment" at 85.5
    per 100 residues sat beside the honest albumin row at 0.49 and ``max`` took
    the fragment.

    So a part has to *be* an accession, or one of the three entry names, after
    an isoform (``-1``) or version (``.2``) suffix is dropped. A label naming
    some other protein as well is not one of these, whatever else it holds.
    Returning the canonical accession lets the caller merge spellings, so one
    protein is one candidate.
    """
    found: set[str] = set()
    for raw in _LABEL_SPLIT.split(label):
        part = raw.strip().strip(".,;:_").upper()
        if not part:
            continue
        base = _ISOFORM.sub("", part)
        if base in _TOLERATED:
            found.add(base)
            continue
        if base in _TOLERATED_NAMES:
            found.add(_TOLERATED_NAMES[base])
            continue
        # A part that is itself accession-shaped and not tolerated means the
        # label names another protein; "sp" and an entry name are not that.
        if _ACCESSION_SHAPE.fullmatch(base):
            return None
    if len(found) != 1:
        return None
    return next(iter(found))


def _pooled_proteins(
    rows: list[dict[str, str]],
    lengths: dict[str, int],
) -> set[str]:
    """Proteins whose rows cannot all be windows of one protein of that length.

    Two arithmetic facts, both decidable from the table alone -- this module
    holds no protein sequences and fetches none.

    A peptide starting at ``start`` ends at ``start + len(peptide) - 1``, which
    cannot lie beyond the protein. And a protein of length L has ``L - w + 1``
    windows of width w, so more distinct peptides of one width than that is more
    than one molecule's scan.

    Either one means rows from several proteins were pooled under one label. A
    real export that scored albumin, transferrin and C3 and then tagged every
    row with the first accession divides three molecules' binders by one
    molecule's residues: the measured bar rose 3.46x and turned a hit into a
    clear. An inflated bar always errs toward clearing the design, which is why
    this is refused rather than corrected.

    The bound is per width on purpose. A complete scan over widths 8 to 11 of a
    100-residue protein holds 366 distinct peptides, legitimately more than the
    protein has residues, so a bound on the total would refuse real scans. The
    cross-width inflation that bound was reaching for is a different fault --
    counting binders of every width over one length -- and it is answered by
    requiring the two tables to have scanned the same widths, below.
    """
    widths: dict[tuple[str, int], set[str]] = {}
    pooled: set[str] = set()
    for row in rows:
        protein = _cell(row, _PROTEIN_COLUMNS).upper()
        length = lengths.get(protein)
        if length is None:
            continue
        peptide = _cell(row, _PEPTIDE_COLUMNS).upper()
        if not peptide:
            continue
        widths.setdefault((protein, len(peptide)), set()).add(peptide)
        start = _float_cell(row, _START_COLUMNS)
        if start is not None and start > 0 and int(start) + len(peptide) - 1 > length:
            pooled.add(protein)
        end = _float_cell(row, _END_COLUMNS)
        if end is not None and end > length:
            pooled.add(protein)
    for (protein, width), peptides in widths.items():
        if len(peptides) > lengths[protein] - width + 1:
            pooled.add(protein)
    return pooled


def _scan_extent(rows: list[dict[str, str]], protein: str) -> int:
    """The last residue this protein's rows reach, or 0 when they do not say."""
    furthest = 0
    for row in rows:
        if _cell(row, _PROTEIN_COLUMNS).upper() != protein:
            continue
        peptide = _cell(row, _PEPTIDE_COLUMNS).upper()
        end = _float_cell(row, _END_COLUMNS)
        if end is not None and end > 0:
            furthest = max(furthest, int(end))
            continue
        start = _float_cell(row, _START_COLUMNS)
        if start is not None and start > 0 and peptide:
            furthest = max(furthest, int(start) + len(peptide) - 1)
    return furthest


def _one_contiguous_scan(rows: list[dict[str, str]], protein: str) -> bool:
    """Whether this protein's rows are one scan rather than several pooled.

    One scan visits each position once per width, so within a width the number
    of distinct start positions is the number of rows. Rows pooled from several
    proteins repeat positions, because each protein was scanned from residue
    one. Without a start column there is no way to tell, and the caller keeps
    the pooling answer, which is the safer of the two.
    """
    seen: dict[int, set[int]] = {}
    counts: dict[int, int] = {}
    for row in rows:
        if _cell(row, _PROTEIN_COLUMNS).upper() != protein:
            continue
        peptide = _cell(row, _PEPTIDE_COLUMNS).upper()
        start = _float_cell(row, _START_COLUMNS)
        if not peptide or start is None or start <= 0:
            return False
        width = len(peptide)
        seen.setdefault(width, set()).add(int(start))
        counts[width] = counts.get(width, 0) + 1
    if not counts:
        return False
    return all(len(seen[width]) == count for width, count in counts.items())


def _table_widths(rows: list[dict[str, str]]) -> frozenset[int]:
    """Every peptide width a table scored, for widths a cutoff is defined for.

    The density counts distinct binding peptides of every width and divides by
    one length, so comparing a one-width design against a four-width reference
    compares a count over one set of questions with a count over four. That is
    the same fault the allele guard already refuses, and it inflates the bar by
    about the number of extra widths: four widths over 76 residues reached 339
    per 100 against albumin's 0.49.
    """
    return frozenset(width for row in rows if (width := len(_cell(row, _PEPTIDE_COLUMNS))) in _SCORED_WIDTHS)


def _table_alleles(rows: list[dict[str, str]]) -> frozenset[str]:
    """Every allele a table names, normalised. Empty when it names none."""
    return frozenset(key for row in rows if (key := _allele_key(_cell(row, _ALLELE_COLUMNS))))


def _reference_density(
    directory: Path,
    stem: str,
    *,
    rank_max: float,
    ic50_max: float,
    design_methods: set[str],
    design_alleles: frozenset[str],
    design_widths: frozenset[int],
) -> tuple[str | None, float | None, dict[str, object]]:
    """The highest binding-peptide density among the tolerated reference proteins.

    A blocker means there was no bar to compare against. Four things have to
    hold for the comparison to mean anything, and none of them is checked by
    looking at the numbers.

    The tool: two predictors disagree about which peptides bind, so a density
    from one is not a bar for the other.

    The alleles: a design scanned on one HLA allele against a reference scanned
    on seven is a count over one set compared with a count over seven, and the
    wider scan carries more binders purely by having been asked more questions.
    That inflates the bar, which errs toward clearing the design.

    The protein: only the three tolerated proteins may set the bar. A density
    over anything else is a bar with no meaning, and one over a protein with
    high epitope content inflates it. A row for another protein is dropped
    before the count, so a stray entry cannot lift the bar and is not reported
    as having carried no binders either.

    The arithmetic: the rows attributed to a protein have to fit inside it.
    See ``_pooled_proteins``.
    """
    path = _table_path(directory, f"{stem}{_REFERENCE_SUFFIX}")
    if path is None:
        return "reference_not_configured", None, {}
    rows = _table_rows(path)
    if not rows:
        return "reference_unreadable", None, {}
    unparsed = _unparsed_columns(rows)
    if unparsed:
        return "reference_cell_unparsed", None, {"unparsed_columns": sorted(unparsed)}
    not_nm = sorted(_not_nm_columns(rows))
    if not_nm:
        # The bar is the number a clear is taken against, so an affinity column
        # whose unit the reference never states cannot be read past. Taken as
        # nanomolar it counts a 2 uM cell as a 2 nM binder and inflates the
        # bar, which clears the design; left unread it understates the bar.
        # Neither is a bar, so there is none.
        return "reference_ic50_not_nm", None, {"not_nm_columns": not_nm}
    reference_methods = _table_methods(rows)
    # An absent column is not agreement -- deleting one column turned this
    # refusal into a measured clear on unchanged numbers. But refusing whenever
    # it is absent was worse: neither NetMHCpan nor the IEDB standalone tools
    # write a method column by default, so that refused the normal output of
    # both documented predictors and left no way to reach a verdict.
    #
    # So: one side declaring and the other not is the asymmetry that hid a real
    # mismatch, and is refused. Neither side declaring is symmetric ignorance,
    # which is measured with the gap recorded on the axis, because the user
    # chose both files and tool identity is the thing being taken on trust.
    declared = bool(design_methods) + bool(reference_methods)
    if declared == 1:
        detail: dict[str, object] = {
            "design_method": sorted(design_methods),
            "reference_method": sorted(reference_methods),
        }
        return "reference_method_unknown", None, detail
    undeclared = declared == 0
    if not undeclared and not design_methods & reference_methods:
        detail = {
            "design_method": sorted(design_methods),
            "reference_method": sorted(reference_methods),
        }
        return "reference_method_mismatch", None, detail
    reference_alleles = _table_alleles(rows)
    if not design_alleles or not reference_alleles:
        detail = {
            "design_alleles": sorted(design_alleles),
            "reference_alleles": sorted(reference_alleles),
        }
        return "reference_allele_unknown", None, detail
    if design_alleles != reference_alleles:
        detail = {
            "design_alleles": sorted(design_alleles),
            "reference_alleles": sorted(reference_alleles),
        }
        return "reference_allele_mismatch", None, detail
    reference_widths = _table_widths(rows)
    if design_widths != reference_widths:
        detail = {
            "design_widths": sorted(design_widths),
            "reference_widths": sorted(reference_widths),
        }
        return "reference_width_mismatch", None, detail
    lengths = _protein_lengths(rows)
    if not lengths:
        return "reference_length_unknown", None, {}
    # The bar is what tolerated proteins carry, so only they may set it. A
    # reference scored over some other protein is a bar with no meaning, and a
    # protein with high epitope content inflates it, which errs toward clearing.
    # Proteins outside the set are dropped rather than ignored quietly, so a
    # stray row cannot raise the bar either.
    # Keyed on the CANONICAL accession, not the label. One protein written two
    # ways was two bar candidates, and a 76-residue "fragment" spelling at 85.5
    # per 100 residues outbid the honest albumin row at 0.49 under max().
    canonical: dict[str, str] = {}
    stated: dict[str, set[int]] = {}
    for name, length in lengths.items():
        accession = _tolerated_accession(name)
        if accession is None:
            continue
        canonical[name] = accession
        stated.setdefault(accession, set()).add(length)
    if not canonical:
        return "reference_not_tolerated_protein", None, {"reference_proteins": sorted(lengths)}
    # Spellings of one protein have to agree about how long it is, and the
    # disagreement is refused rather than resolved. Merging by taking either
    # length is wrong in both directions: the larger let a 76-residue fragment
    # spelling's rows pass a bound that should have used its own stated length,
    # and the smaller would divide the whole protein's binders by a fragment.
    disagreeing = sorted(acc for acc, found in stated.items() if len(found) > 1)
    if disagreeing:
        detail = {
            "reference_proteins": disagreeing,
            "stated_length": {acc: sorted(stated[acc]) for acc in disagreeing},
        }
        return "reference_length_inconsistent", None, detail
    tolerated = {acc: next(iter(found)) for acc, found in stated.items()}
    # The bound each label is held to is its OWN stated length, never the
    # merged one, which is what makes "the rows must fit inside it" mean
    # anything for a label that understates its protein.
    pooled = _pooled_proteins(rows, {name: lengths[name] for name in canonical})
    if pooled:
        # Two different mistakes reach here and need different remedies. One
        # contiguous scan whose stated length is simply wrong is a typo; rows
        # from several proteins under one accession is pooling.
        contiguous = {name for name in pooled if _one_contiguous_scan(rows, name)}
        if contiguous == pooled:
            detail = {
                "reference_proteins_pooled": sorted(pooled),
                "stated_length": {name: lengths[name] for name in sorted(pooled)},
                "observed_extent": {name: _scan_extent(rows, name) for name in sorted(pooled)},
            }
            return "reference_length_inconsistent", None, detail
        return "reference_rows_exceed_protein", None, {"reference_proteins_pooled": sorted(pooled)}
    counts: dict[str, set[str]] = {}
    by_width: dict[tuple[str, int], set[str]] = {}
    for row in rows:
        protein = _cell(row, _PROTEIN_COLUMNS).upper()
        peptide = _cell(row, _PEPTIDE_COLUMNS).upper()
        accession = canonical.get(protein)
        if not protein or not peptide or accession is None:
            continue
        rank = _float_cell(row, _RANK_COLUMNS)
        ic50 = _float_cell(row, _IC50_COLUMNS)
        if (rank is not None and rank <= rank_max) or (ic50 is not None and ic50 <= ic50_max):
            counts.setdefault(accession, set()).add(peptide)
            by_width.setdefault((accession, len(peptide)), set()).add(peptide)
    # A bar far above what the cutoff can produce is a bad table, not a high
    # bar. A file claiming every one of ubiquitin's 68 windows binds gives 89 per
    # 100 and clears almost any design. Judged as a share of the windows, so it
    # is scale-free.
    #
    # The share is taken per (protein, width). For class I that is stricter than
    # pooling: albumin pools to 10.3 binders per 100 windows but reaches 20.6%
    # at width 9 alone, because a per-allele rank budget is spent within one
    # width. See _BAR_MAX_WINDOW_SHARE for why the class II path cannot clear
    # this guard at all, and for the figure the old comment here misattributed.
    implausible = sorted(
        {
            accession
            for (accession, width), peptides in by_width.items()
            if (windows := tolerated[accession] - width + 1) > 0 and len(peptides) > windows * _BAR_MAX_WINDOW_SHARE
        }
    )
    if implausible:
        return "reference_density_implausible", None, {"reference_proteins": implausible}
    # Over windows, not residues, and over the same widths the design's count
    # was taken at -- ``reference_width_mismatch`` above has already forced the
    # two width sets equal, so both sides' denominators are built the same way.
    densities: dict[str, float] = {}
    too_short: dict[str, int] = {}
    for accession, length in tolerated.items():
        windows = _window_count(length, reference_widths)
        if not windows:
            # The length was read, and it is shorter than every width the table
            # was scored at, so this protein holds no window of that width and
            # has no density. Dropped silently and then reported as
            # ``reference_length_unknown``, which says the length could not be
            # read -- it was read and found too short, and the remedy for the
            # two is not the same one.
            too_short[accession] = length
            continue
        densities[accession] = len(counts.get(accession, set())) * _DENSITY_BASIS / windows
    if not densities:
        return (
            "reference_too_short_for_width",
            None,
            {
                "reference_proteins": sorted(too_short),
                "stated_length": {name: too_short[name] for name in sorted(too_short)},
                "reference_widths": sorted(reference_widths),
            },
        )
    highest = max(densities.values())
    detail = {
        "reference_proteins": sorted(densities),
        "reference_densities": {name: _round4(value) for name, value in sorted(densities.items())},
        # Which predictor produced each side. Every refusal above already names
        # them, and the standing verdict did not -- so the one record carrying a
        # measurement was the one that could not say what measured it. Two
        # predictors over the same windows wrote byte-identical records: two run
        # directories differing only in the method cell compared equal under
        # json.dumps(sort_keys=True). Both keys are in mcp_api._AXIS_FIELDS, so a
        # consumer sees them rather than their sitting unread in the file.
        "design_method": sorted(design_methods),
        "reference_method": sorted(reference_methods),
    }
    if too_short:
        # Some candidates set the bar and others were too short for the widths
        # scored. Which were which belongs on the record: the bar is a maximum,
        # and a reader cannot otherwise tell a protein that carried no binder
        # from one that was never in the running.
        detail["reference_proteins_too_short"] = sorted(too_short)
    if _rank_scale_ambiguous(rows):
        # Every rank in the reference at or below 1, so the column reads the
        # same whether it is a percentage or a fraction of one -- a hundredfold
        # apart, with nothing published to tell them apart. Read as a
        # percentage, which is what the cutoffs are defined for, and recorded,
        # because that reading calls more rows binding and so sets a higher
        # bar, and a bar that is too high clears designs. The disclosure is the
        # one ``reference_method_undeclared`` makes beside a standing verdict.
        detail["reference_rank_scale_ambiguous"] = True
    if undeclared:
        # The verdict stands, and says that tool identity was taken on trust.
        detail["reference_method_undeclared"] = True
    return None, highest, detail


def _panel_rows_present(rows: list[dict[str, str]], wanted: set[str]) -> bool:
    """Whether the table names a peptide against every allele of the panel.

    Used only to tell "the panel is not covered" apart from "the panel is
    covered but carries no rank", which produce the same empty result and need
    different answers from the user.
    """
    seen: dict[str, set[str]] = {}
    for row in rows:
        peptide = _cell(row, _PEPTIDE_COLUMNS).upper()
        # The same molecule reading its caller uses, or the two disagree about
        # whether the panel is covered and a three-field spelling lands on
        # ``allele_panel_incomplete`` instead of ``rank_absent``.
        allele = _allele_molecule(_allele_key(_cell(row, _ALLELE_COLUMNS)), wanted)
        if peptide and allele is not None:
            seen.setdefault(peptide, set()).add(allele)
    return any(len(alleles) == len(wanted) for alleles in seen.values())


def _cell(row: dict[str, str], names: tuple[str, ...]) -> str:
    """The first of these columns that holds anything."""
    for name in names:
        text = row.get(name, "").strip()
        if text:
            return text
    return ""


def _allele_molecule(key: str, wanted: Iterable[str]) -> str | None:
    """Which wanted molecule this key names, however many fields it carries.

    An HLA protein is its first two fields. ``_allele_key`` keeps every field a
    predictor wrote and drops the separators, so ``DRB1*03:01`` normalises to
    ``DRB10301`` and the three-field ``DRB1*03:01:02`` to ``DRB1030102``, and an
    exact comparison reads those as different molecules.

    Both callers were wrong on that, in opposite directions. The Calis exception
    missed the molecule and published a CLEAR on a score the module cannot
    compute. The DR panel missed it and skipped the row, so the panel read
    incomplete and ``missing_alleles`` named the very alleles the user had just
    scored -- the same loop the ``rank_absent`` branch above exists to avoid,
    and unresolvable, because re-scoring produces the same spelling again.

    A further field is a sub-type of the same protein, so a key that continues a
    wanted one with digits alone is that one. Two sub-types of one molecule
    therefore collapse to it, and the per-molecule best rank is kept, which is
    what the median rule wants.
    """
    for candidate in wanted:
        if key == candidate:
            return candidate
        if key.startswith(candidate) and key[len(candidate) :].isdigit():
            return candidate
    return None


def _calis_exceptions(restricting: list[str]) -> set[str]:
    """The exception molecules among these alleles, however many fields they name.

    ``_allele_key`` keeps every field a predictor wrote, so ``HLA-B*08:01``
    normalises to ``B0801`` and the three-field ``HLA-B*08:01:01`` to
    ``B080101``. An exact match against :data:`_CALIS_EXCEPTION_ALLELES` saw
    those as different molecules, and the second one published a ``clear`` on a
    score this module says it cannot compute -- the one place the gap errs toward
    an absence claim rather than toward withholding one.

    Matching a shade too broadly here costs a ``clear`` and never a hit, which
    is the direction to err in. The comparison itself is
    :func:`_allele_molecule`, shared with the DR panel's membership test, which
    had the same gap in the other direction.
    """
    found: set[str] = set()
    for allele in restricting:
        exception = _allele_molecule(allele, _CALIS_EXCEPTION_ALLELES)
        if exception is not None:
            found.add(exception)
    return found


def _allele_key(raw: str) -> str:
    """One form for an allele however the predictor spelled it.

    ``HLA-DRB1*03:01``, ``DRB1_0301`` and ``DRB1*03:01`` are one allele, and a
    panel membership test that missed that would read a full panel as partial.
    """
    text = raw.strip().upper()
    if text.startswith("HLA-"):
        text = text[4:]
    for drop in ("*", ":", "_", "-", " "):
        text = text.replace(drop, "")
    return text


def _declares_consensus(rows: list[dict[str, str]]) -> bool:
    """Whether the table says its ranks are on the scale the threshold belongs to.

    A method that names itself consensus is on the consensus scale however it
    spells the components it combined.

    Failing that, a rank column named after the consensus method says the same
    thing the method column would -- that is how the export writes it -- but
    only when no method column is present to say otherwise. A table declaring
    neither is not taken as consensus: the caller has to say, because the number
    means something else on the scale a different tool would have written.
    """
    if any(_CONSENSUS_MARK in _cell(row, _METHOD_COLUMNS).casefold() for row in rows):
        return True
    if any(_cell(row, _METHOD_COLUMNS) for row in rows):
        return False
    return any(key.startswith(_CONSENSUS_MARK) for row in rows for key in row)


def _median(values: list[float]) -> float:
    """The middle of a sorted list, or the mean of the middle two."""
    count = len(values)
    middle = count // 2
    if count % 2:
        return values[middle]
    return (values[middle - 1] + values[middle]) / 2


def _calis_scores(sequence: str | None) -> list[float]:
    if not sequence:
        return []
    return [_calis_score(window) for window in _slices(sequence, _KMER)]


def _calis_score(peptide: str) -> float:
    return sum(weight * _CALIS_SCALE[residue] for weight, residue in zip(_CALIS_WEIGHT, peptide, strict=True))


def _slices(sequence: str, width: int) -> list[str]:
    """Standard-residue windows. A nonstandard residue starts a new run."""
    found: list[str] = []
    token: list[str] = []
    for char in sequence.upper():
        if char in _AA:
            token.append(char)
            continue
        found.extend(_token_slices(token, width))
        token = []
    found.extend(_token_slices(token, width))
    return found


def _token_slices(token: list[str], width: int) -> list[str]:
    if len(token) < width:
        return []
    text = "".join(token)
    return [text[index : index + width] for index in range(len(text) - width + 1)]


def _table_path(directory: Path, stem: str) -> Path | None:
    for suffix in (".tsv", ".csv"):
        path = directory / f"{stem}{suffix}"
        if path.is_file():
            return path
    return None


def _table_rows(path: Path) -> list[dict[str, str]] | None:
    """The table's rows with normalised keys, or None when it did not parse.

    Read as ``utf-8-sig`` because Excel's "CSV UTF-8" writes a byte-order mark
    by default, and a mark is not whitespace: it stayed on the first column's
    name, so no lookup could reach that column and a table plainly containing
    an allele column reported having none.

    A leading banner or comment block is dropped before the header is read, and
    a file whose header holds none of the columns this understands returns None
    rather than an empty-looking table, so "unreadable" is not reported as
    "no numbers in it".
    """
    try:
        text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError, MemoryError):
        # MemoryError alongside the other two, which is what the fold reader
        # and the proteome reader in this file already do. Without it a
        # table too large to hold ended the screen instead of blocking one
        # axis of it -- and the registry points at a 10.26 GB download whose
        # stem is this very table, so the file that triggers it is the one an
        # operator is most likely to have. No file that parses today stops
        # parsing: this catches only the case that previously raised.
        return None
    lines = _table_lines(text)
    if not lines:
        return []
    dialect = csv.excel_tab if "\t" in lines[0] else csv.excel
    try:
        reader = csv.DictReader(io.StringIO("\n".join(lines)), dialect=dialect)
        # A repeated column name is ambiguous, and csv silently keeps the last
        # one. A file whose header carried the consensus rank twice -- the real
        # ranks in one column and another method's in the next -- lost the
        # column holding the binders, with nothing recording which was kept.
        # There is no way to tell which the user meant, so neither is used.
        if _repeated_columns(reader.fieldnames):
            return None
        rows: list[dict[str, str]] = []
        for row in reader:
            rows.append({_column_key(key): (value or "").strip() for key, value in row.items() if key is not None})
    except csv.Error:
        return None
    if rows and not any(_known_column(key) for key in rows[0]):
        return None
    return rows


def _table_lines(text: str) -> list[str]:
    """The table's lines, with a leading banner or comment block dropped.

    A predictor's export often opens with a generated-by line or a run header.
    Taking line one as the column names made every lookup miss, which read as a
    table holding no numbers.
    """
    lines = text.splitlines()
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped and not stripped.startswith(("#", "---", "//", ";;")):
            return [item for item in lines[index:] if item.strip()]
    return []


def _column_key(key: str | None) -> str:
    """One form for a column name, with any byte-order mark removed.

    Brackets, spaces and hyphens collapse to underscores, because NetMHCpan
    writes ``Aff(nM)`` and that did not reach ``aff_nm``: the affinity was
    never read, and a ``-BA`` table with no rank column became a density of
    zero and cleared four 12 nM binders. ``%`` is kept, so ``%Rank_EL`` still
    matches its own spelling.
    """
    text = (key or "").replace("﻿", "").strip().casefold()
    for character in "()[] -.":
        text = text.replace(character, "_")
    return "_".join(part for part in text.split("_") if part)


def _known_column(key: str) -> bool:
    """Whether this is a column some axis reads."""
    return key in _KNOWN_COLUMNS or key.startswith(_CONSENSUS_MARK)


def _repeated_columns(fieldnames: list[str] | None) -> set[str]:
    """Known column names a header uses more than once."""
    seen: set[str] = set()
    twice: set[str] = set()
    for raw in fieldnames or ():
        key = _column_key(raw)
        if not _known_column(key):
            continue
        if key in seen:
            twice.add(key)
        seen.add(key)
    return twice


def _unparsed_columns(rows: list[dict[str, str]]) -> set[str]:
    """Numeric columns holding a cell that is neither a number nor a blank.

    A cell that will not parse is skipped, which makes it indistinguishable
    from an empty one, and a skipped cell always loses a binder rather than
    inventing one. A German or French Excel export writes a rank of 0.5 as
    ``0,5``: every fractional rank was dropped while the integral non-binders
    parsed, so a table whose best rank was 0,5 reported 5.0, the design's
    binders all vanished, and the axis cleared at a density of zero. The
    reference collapsed the same way and printed albumin's bar as 0.0 per 100
    residues, which no real scan gives.

    A comma decimal is not read as a number here, because ``1,000`` is 1.0 in
    one locale and 1000 in another and guessing wrong moves a verdict.
    """
    bad: set[str] = set()
    for row in rows:
        for name in _NUMERIC_COLUMNS:
            text = row.get(name, "").strip()
            if not text or text.casefold() in _NOT_A_VALUE:
                continue
            try:
                number = float(text)
            except ValueError:
                bad.add(name)
                continue
            # A percentile rank is a percentage and an IC50 a concentration, so
            # neither can be negative. A cell that is one is a sentinel or a
            # corrupt export, not a number this can compare with a cutoff.
            if number < 0 and name in _NON_NEGATIVE_COLUMNS:
                bad.add(name)
    return bad


def _unparsed_axis(rows: list[dict[str, str]]) -> dict[str, object] | None:
    """An axis record refusing a table with an unreadable number, or None."""
    unparsed = _unparsed_columns(rows)
    if not unparsed:
        return None
    record = _axis("untested", None, "cell_unparsed")
    record["unparsed_columns"] = sorted(unparsed)
    return record


def _unit_kind(row: dict[str, str]) -> str:
    """``nm``, ``other`` or ``absent`` -- what this row says its numbers are in.

    A blank or ``n/a`` unit cell says nothing, which leaves a column whose own
    name states nanomolar as nanomolar and leaves a column named only for its
    analyte with no unit at all.
    """
    text = _cell(row, _UNIT_COLUMNS)
    if not text or text.casefold() in _NOT_A_VALUE:
        return "absent"
    token = text.casefold().replace("µ", "u").replace("μ", "u").replace(" ", "")
    return "nm" if token in _NM_UNITS else "other"


def _is_nm(row: dict[str, str], name: str) -> bool:
    """Whether this affinity cell is known to be in nanomolar.

    The cutoffs are nanomolar. A column whose own name says so is read as it
    stands; a column named only for its analyte is read only where the row
    declares a unit and declares it nanomolar. A row declaring some other unit
    overrides even the column name, because ``ic50_nm`` beside a ``uM`` unit
    cell is a table contradicting itself, not a nanomolar reading.
    """
    kind = _unit_kind(row)
    if kind == "other":
        return False
    return kind == "nm" or name in _IC50_NM_COLUMNS


def _not_nm_columns(rows: list[dict[str, str]]) -> set[str]:
    """Affinity columns holding a number that is not known to be nanomolar.

    ``_float_cell`` leaves such a cell unread, which on its own is a silent
    loss of a binder, so the callers put these column names on the record and
    refuse a clear that rests on the rows that were read.
    """
    found: set[str] = set()
    for row in rows:
        for name in _IC50_COLUMNS:
            if name not in row or _is_nm(row, name):
                continue
            if _cell_number(row[name].strip()) is not None:
                found.add(name)
    return found


def _rank_scale_ambiguous(rows: list[dict[str, str]]) -> bool:
    """Whether every rank in the table is low enough to be a fraction of one.

    A rank written as a fraction cannot exceed 1, so a table holding any rank
    above 1 is percentages and nothing else. Below that the two readings are
    indistinguishable, and this says so rather than guessing which one the file
    meant. See ``_RANK_FRACTION_MAX``.
    """
    ranks = [number for row in rows if (number := _float_cell(row, _RANK_COLUMNS)) is not None]
    return bool(ranks) and max(ranks) <= _RANK_FRACTION_MAX


def _cell_number(text: str) -> float | None:
    """The finite number this cell holds, or None when it holds no value."""
    if text.casefold() in _NOT_A_VALUE:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if number != number or number in {float("inf"), float("-inf")}:
        return None
    return number


def _float_cell(row: dict[str, str], names: tuple[str, ...]) -> float | None:
    for name in names:
        if name not in row:
            continue
        number = _cell_number(row[name].strip())
        if number is None:
            continue
        # The second line of defence for a negative rank or affinity, for the
        # readers that reach a cell without the table-wide refusal above: a
        # value no scale allows is no value, never the best binder found.
        if number < 0 and name in _NON_NEGATIVE_COLUMNS:
            continue
        # An affinity the table does not say is nanomolar is not a number that
        # can be compared with a nanomolar cutoff, and a micromolar figure is
        # never rescaled onto the cut. Read as nanomolar anyway, a 2 uM
        # non-binder under a bare ``ic50`` header was a hit at 2.0, and on the
        # reference side the same cell inflated the bar. ``_not_nm_columns``
        # is what keeps the omission from being a silent one.
        if name in _IC50_COLUMNS and not _is_nm(row, name):
            continue
        return number
    return None


def _round4(value: float) -> float:
    return float(f"{value:.4f}")


def _round5(value: float) -> float:
    return float(f"{value:.5f}")


def _axis(state: str, measurement: int | float | None, blocker: str | None) -> dict[str, object]:
    return {"state": state, "measurement": measurement, "blocker": blocker}
