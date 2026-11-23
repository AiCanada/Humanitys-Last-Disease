"""Where the screening reference tables come from, and fetching them safely.

The screen reads its reference tables as ``<stem>.tsv`` or ``<stem>.csv`` in a
run's output directory -- ``mhc_i``, ``mhc_ii``, ``ada``, ``plasma_fold`` and
the rest. Nothing told anyone where to get them. This is that list, with the
URL, the size and the licence of each, and a fetcher that honours one rule.

**The rule: no single download over 10 GB.** Every source below has a small,
human-only file *and* a whole-database counterpart that is enormous, and the
names do not warn you which is which:

===========================  ==============  ===================================
Whole database               Size            Human-only file to take instead
===========================  ==============  ===================================
UniProt reference proteomes  355.49 GB       7.74 MB (UP000005640_9606)
STRING all organisms         133.76 GB       62.81 MB (9606.protein.links)
HLA Ligand Atlas raw spectra 1.39 TB         a few MB (processed release)
PeptideAtlas whole human     ~44 GB          ~1.63 GB (plasma build)
Ensembl Compara all species  11.23 GB        109.48 MB (homo_sapiens)
===========================  ==============  ===================================

The Ensembl one is the reason the limit is enforced in code rather than left to
care: its own FTP listing rounds 11.23 GB to "10G", so it reads as though it
fits. :data:`REFUSED` names all five so the refusal is a recorded decision with
a substitute beside it, rather than a file nobody happened to list.

The guard runs twice, because one check is not enough. ``Content-Length`` is
checked before any body is read -- and IEDB sends none at all, so a size read
from headers alone would wave the largest source through unmeasured. The bytes
are therefore counted as they arrive and the transfer is abandoned the moment
it passes the limit.

Nothing here is downloaded by importing this module. :func:`fetch` is the only
function that touches the network, and it is called from
``scripts/fetch_screening_tables.py``.
"""

from __future__ import annotations

import csv
import gzip
import io
import math
import re
import shutil
import zipfile
from collections import Counter
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from rfproteina.disease.combinations import LIGAND_AXES
from rfproteina.disease.ligand_safety import BINDINGDB_PANEL_ACCESSIONS
from rfproteina.disease.limits import USER_AGENT
from rfproteina.disease.screening_tools import data_dir
from rfproteina.disease.target_liability import FOLD_FILE_LIMIT, FOLD_ROW_LIMIT

#: Anything urlopen-shaped. Injected so a test can drive a server whose
#: behaviour -- including lying about its size -- is written out in the test.
Opener = Callable[..., Any]

#: No single download may exceed this. The user's rule, kept as a number here
#: so the fetcher and its tests mean the same thing by it.
SIZE_LIMIT = 10 * 1024**3
#: Read in chunks so the running total is checked against the limit long before
#: a body that lies about its length can fill the disk.
_CHUNK = 1 << 20
#: A server that answers nothing in this long is not going to.
_TIMEOUT = 60


@dataclass(frozen=True)
class Source:
    """One downloadable table, and what the screen does with it."""

    name: str
    url: str
    #: The screen's own file stem for the axis this feeds, or ``None`` where the
    #: screen reads no table for it.
    #:
    #: It is NOT where the download goes. The download keeps its own name (see
    #: :attr:`file`); what reaches the screen is whatever :attr:`handoff` says.
    #: The distinction is the whole of this attribute: read as a destination it
    #: says "rename the download to this", and for three of the five sources
    #: here that produces a run that looks equipped and measures nothing --
    #: or worse, see :attr:`hazard`.
    stem: str | None
    #: The axis or axes it feeds, for the report.
    axis: str
    #: Bytes **over the wire**, where a HEAD request established it, which for
    #: a zipped source is the archive and not the table that comes out of it:
    #: the HPA download is 7.34 MB and unpacks to 63 MB. ``None`` where the
    #: server sends no ``Content-Length`` -- IEDB does not -- because the
    #: figure would then be the download page's claim and not a measurement.
    size: int | None
    licence: str
    #: What a reader has to know that the row above does not say.
    note: str = ""
    #: A member inside a zip to extract, rather than keeping the archive.
    member: str | None = None
    #: How this download reaches the screen, if it does.
    #:
    #: * ``"convert"`` -- a converter in this module rewrites it into
    #:   ``<stem>``. The download is not the screen's table and cannot stand in
    #:   for it; the conversion is named in :attr:`note`.
    #: * ``"operator"`` -- not a table the screen opens. It is passed on a flag
    #:   or an environment variable, so there is nothing to place in a run.
    #: * ``"none"`` -- no path from this file to the screen. Kept in the
    #:   registry anyway: the axis exists, the download is the one an operator
    #:   will reach for, and saying "this does not feed it" is the only thing
    #:   that stops the reach. :attr:`hazard` says what dropping it in does.
    handoff: str = "none"
    #: What happens if this download is renamed to ``<stem>`` and left in a run
    #: directory. Only on the sources where that is worse than useless.
    hazard: str = ""
    #: The file name left on disk once the download has been fetched and any
    #: archive unpacked, read off a completed download rather than guessed at.
    #: Defaults to the URL's own leaf, which is right for an unzipped file.
    file: str = ""


@dataclass(frozen=True)
class Refused:
    """A download this will not make, and the one to make instead.

    Recorded rather than left out. A list of good URLs says nothing about the
    much larger file sitting next to each of them under a similar name, and the
    whole point of the limit is that those are easy to reach for by accident.
    """

    name: str
    url: str
    size: int
    #: What to do instead. Either the exact name of a :class:`Source` or of a
    #: :class:`Query` -- so the way forward is a registry entry and not advice
    #: that can rot -- or prose beginning "the ", for the two cases where the
    #: substitute is a file at the same host that this does not itself fetch.
    instead: str


#: Verified by HEAD request against the official host. Sizes are exact
#: ``Content-Length`` values, not the figures printed on the download pages.
#: The panel axis each GtoPdb target answers, keyed by the HGNC symbol GtoPdb
#: writes in its ``Target Gene Symbol`` column. Verified against
#: ``interactions.csv`` at GtoPdb 2026.3.1 (published 2026-10-12): every symbol
#: below returns Human rows carrying a nanomolar affinity, and the ten cover
#: :data:`~rfproteina.disease.combinations.LIGAND_AXES` exactly.
GTOPDB_PANEL_GENES: dict[str, str] = {
    "herg": "KCNH2",
    "nav1_5": "SCN5A",
    "cyp3a4": "CYP3A4",
    "cyp2d6": "CYP2D6",
    "cyp2c9": "CYP2C9",
    "cyp1a2": "CYP1A2",
    "cyp2c19": "CYP2C19",
    "pgp": "ABCB1",
    "bcrp": "ABCG2",
    "bsep": "ABCB11",
}

#: GtoPdb's own column spellings, which are title-cased with spaces.
#:
#: Two of them matter more than the rest. GtoPdb carries every affinity twice:
#: once as a pX under ``Affinity Units``/``Affinity Median``, and once in real
#: units under ``Original Affinity Units`` and ``Original Affinity Median nm``.
#: The second pair is read here because it is **already nanomolar**, so no
#: logarithm is undone and no unit is inferred. The two agree: across the 14,097
#: rows carrying both, ``10 ** (9 - pX)`` lands within a factor of two of the
#: stated nM for all but 14 of them (0.099%), checked at 2026.3.1.
_GTOPDB_GENE = "Target Gene Symbol"
_GTOPDB_SPECIES = "Target Species"
_GTOPDB_ASSAY = "Original Affinity Units"
_GTOPDB_NM = "Original Affinity Median nm"
_GTOPDB_RELATION = "Original Affinity Relation"
_GTOPDB_LIGAND = "Ligand"
_GTOPDB_LIGAND_ID = "Ligand ID"
_GTOPDB_PUBMED = "PubMed ID"
_GTOPDB_ACTION = "Type"

#: Assay types taken as an affinity, in the preference order
#: :mod:`rfproteina.disease.ligand_safety` already uses for ChEMBL. ``EC50`` and
#: ``KB`` are deliberately left out: a half-maximal *effect* is not the same
#: quantity as a binding or inhibition constant, and the panel cut is written
#: against the latter.
_GTOPDB_ASSAYS = ("IC50", "Ki", "Kd")

#: How GtoPdb's relation reaches the toxicology reader. ``<`` and ``>`` are in
#: that reader's own ``_BOUNDS`` and decide an axis when everything they admit
#: falls on one side of the cut, so they are carried through for it to judge.
#:
#: ``~`` is not a bound it knows. GtoPdb writes it on 183 rows at 2026.3.1 for
#: an approximate value, and writing those through would hand the reader a
#: number it would treat as exact. They are dropped and counted instead, which
#: matches what :func:`~rfproteina.disease.ligand_safety.fetch_measured_nm` does
#: with a qualified ChEMBL relation.
_GTOPDB_RELATIONS = {"": "=", "=": "=", "<": "<", ">": ">"}

#: The screen's own stem for the ligand panel. Kept here rather than imported so
#: this module pulls in no reader; ``test_tables_gtopdb`` asserts it still equals
#: :data:`rfproteina.disease.toxicology.TOXICOLOGY_TABLE`.
_PANEL_STEM = "toxicology"


#: BindingDB's column spellings, at the 2026-11 release. 640 columns.
_BDB_INCHIKEY = "Ligand InChI Key"
_BDB_SMILES = "Ligand SMILES"
_BDB_TARGET_NAME = "Target Name"
_BDB_ORGANISM = "Target Source Organism According to Curator or DataSource"
_BDB_CURATION = "Curation/DataSource"
_BDB_DOI = "Article DOI"
_BDB_ENTRY_DOI = "BindingDB Entry DOI"

#: The affinity columns read, in the preference order
#: :mod:`rfproteina.disease.ligand_safety` already uses. Each is nanomolar in the
#: column name itself. ``EC50 (nM)`` is deliberately absent: a half-maximal
#: *effect* is not a binding or inhibition constant, which is what the panel cut
#: is written against, and the ChEMBL and GtoPdb routes exclude it too.
_BDB_ASSAYS: tuple[tuple[str, str], ...] = (
    ("IC50", "IC50 (nM)"),
    ("Ki", "Ki (nM)"),
    ("Kd", "Kd (nM)"),
)

#: A target's accession is **not** in one known column. BindingDB repeats the
#: UniProt block for each chain of the target, 50 chains at this release, on a
#: stride of 12 columns from index 44. In a 23,736-row sample, 5,983 rows carried
#: their accession somewhere other than chain 1, so reading chain 1 alone loses a
#: quarter of the file. Both the SwissProt and the TrEMBL primary-ID columns are
#: scanned, for every chain the header declares.
_BDB_CHAIN_PREFIXES = (
    "UniProt (SwissProt) Primary ID of Target Chain ",
    "UniProt (TrEMBL) Primary ID of Target Chain ",
)
_BDB_CHAIN_COUNT = "Number of Protein Chains in Target (>1 implies a multichain complex)"

#: BindingDB writes the relation **inside** the value cell -- ``>10000``,
#: ``<0.5``, and ``> 10000`` with a space -- so ``float`` on the cell raises and
#: stripping the operator would turn a bound into an exact reading. The operator
#: is split off and kept: the toxicology reader understands ``>`` and ``<`` as
#: bounds and decides an axis when everything a bound admits falls one side of
#: the cut.
_BDB_VALUE = re.compile(r"^\s*(>=|<=|>|<)?\s*([0-9]+(?:\.[0-9]+)?(?:[eE][-+]?[0-9]+)?)\s*$")

#: One format for every measured panel source, so BindingDB's volume and
#: GtoPdb's curation reach the same two readers in
#: :mod:`rfproteina.disease.ligand_safety` -- the per-ligand lookup and the
#: similarity read-across -- instead of each needing its own.
#: :func:`merge_panel_subsets` concatenates them, and ``curation`` is what still
#: says which source and which licence a row came from.
#:
#: Chosen so the file **cannot** be read as the screen's panel table even if it
#: is renamed to ``toxicology.tsv``: no column here is one of that reader's
#: axis, value or unit columns, so a misplaced copy reports
#: ``toxicology_not_reported`` rather than reporting other people's compounds as
#: this design's. Guarded by construction, not by the filename.
PANEL_SUBSET_COLUMNS = (
    "inchikey",
    #: The key the source published, kept beside the recomputed one. BindingDB's
    #: is stereo-blind and GtoPdb's is not, which is only visible if both are
    #: carried.
    "inchikey_upstream",
    # The structure, kept so a novel design can be compared against what was
    # actually measured. An exact-key lookup answers only for molecules already
    # measured, which a designed ligand never is; the neighbours are what make
    # these files useful, and they need something to fingerprint.
    "smiles",
    "panel_axis",
    "accession",
    "affinity_nanomolar",
    "relation",
    "assay",
    "chains",
    "curation",
    "source",
)

#: The subset's own name. Not ``toxicology.*``, for the reason above.
_BDB_SUBSET_FILE = "bindingdb_panel_subset.tsv"

SOURCES: tuple[Source, ...] = (
    Source(
        name="IEDB MHC ligand assays",
        url="https://www.iedb.org/downloader.php?file_name=doc/mhc_ligand_full_tsv.zip",
        file="mhc_ligand_full.tsv",
        stem="mhc_i",
        handoff="none",
        axis="mhc_i, mhc_ii",
        size=None,
        licence="CC BY 4.0",
        note=(
            "IEDB sends no Content-Length, so the size is unknown until it arrives; its own "
            "page says about 313 MB and 10.26 GB arrives. It is not the artifact the screen "
            "reads. That is a binding scan of THIS design's own chain; this is a census of "
            "measured epitopes from every organism IEDB holds, under a two-row header whose "
            "columns are group-and-subname pairs -- Epitope/Name for the peptide, a name "
            "under MHC Restriction for the allele, Assay/Quantitative measurement for the "
            "number. The reader looks for peptide, allele and consensus_percentile_rank, and "
            "none of the 112 columns is any of them. No split by allele turns one into the "
            "other; a predictor run over the designed sequence does, and none is wired up "
            "here."
        ),
        hazard=(
            "Renamed to mhc_i.tsv it cannot be read at all: 112 columns under a two-row "
            "header, none of them a peptide or allele column by the names the reader uses, "
            "so the axis reports predictor_unreadable. It is also 10.26 GB, and this reader "
            "-- unlike the fold reader beside it -- bounds no file size and catches no "
            "MemoryError before reading the whole file into memory."
        ),
    ),
    Source(
        name="IEDB T cell assays",
        url="https://www.iedb.org/downloader.php?file_name=doc/tcell_full_v3_tsv.zip",
        file="tcell_full_v3.tsv",
        stem="immunogenicity",
        handoff="none",
        axis="immunogenicity",
        size=None,
        licence="CC BY 4.0",
        note=(
            "About 43 MB by IEDB's own figure, 1.29 GB as it arrives; no Content-Length is "
            "sent. Same mismatch as the ligand export: T cell assay records across every "
            "organism, where the axis reads a scan of this design's own chain."
        ),
        hazard=(
            "Renamed to immunogenicity.tsv it suppresses a verdict rather than faking one. "
            "161 columns under a two-row header and no score column among them, so the axis "
            "reports predictor_unreadable -- and because a supplied table takes precedence, "
            "the Calis computation that mhc_i would have driven is displaced and never run."
        ),
    ),
    Source(
        name="IDC anti-drug antibody incidence",
        url=(
            "https://raw.githubusercontent.com/Immunogenicity-Database-Collaborative/"
            "IDC-DB/main/data/aggregated_ada_values.csv"
        ),
        file="aggregated_ada_values.csv",
        stem="ada",
        handoff="none",
        axis="ada",
        size=9_533_634,
        licence="CC BY 4.0",
        note=(
            "3,334 records, 57 columns, current to the end of 2024. Clinical anti-drug "
            "antibody rates for drugs that exist. The ada axis reads something else "
            "entirely: a score supplied for THIS design, or failing that it computes class "
            "II allele breadth from mhc_ii. There is no column here that answers either."
        ),
        hazard=(
            "Renamed to ada.csv it suppresses a verdict rather than faking one. None of its "
            "57 columns is a score the reader accepts, so the axis reports score_absent -- "
            "and because a supplied table takes precedence, the DR-breadth measurement that "
            "mhc_ii.tsv would have produced is displaced and never run. An axis that could "
            "have been a hit reads untested."
        ),
    ),
    Source(
        name="Human Protein Atlas disease blood differential expression",
        url="https://www.proteinatlas.org/download/tsv/blood_pea_disease_de.tsv.zip",
        file="blood_pea_disease_de.tsv",
        stem="plasma_fold",
        handoff="convert",
        axis="plasma_fold",
        size=7_700_305,
        licence="CC BY-SA 3.0 (proteinatlas.org; check before redistributing)",
        note=(
            "The aggregated, group-level public layer. Individual-level NPX values are "
            "restricted and are not in it. plasma_fold wants a fold change with a named "
            "study behind it, which this has. Convert with plasma_fold_from_hpa: the column "
            "spelling, the provenance column and the one-disease narrowing are all required, "
            "and the file is refused or misread without them."
        ),
    ),
    Source(
        name="UniProt human reference proteome (canonical)",
        url=(
            "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/"
            "reference_proteomes/Eukaryota/UP000005640/UP000005640_9606.fasta.gz"
        ),
        file="UP000005640_9606.fasta.gz",
        stem=None,
        handoff="operator",
        axis="proteome_offtarget",
        size=7_740_211,
        licence="CC BY 4.0",
        note=(
            "Two jobs. It is not a table the screen opens -- the off-target axis takes it on "
            "--proteome, or from HUMAN_PROTEOME -- and staging also reads it for the gene "
            "symbol to UniProt accession mapping that the fold table needs, 20,394 symbols "
            "of the 20,652 sequences. Without it the fold axis still reads a gene-named "
            "target and misses an accession-named one, which is what a real run asks."
        ),
    ),
    Source(
        name="IUPHAR/BPS Guide to PHARMACOLOGY interactions",
        url="https://www.guidetopharmacology.org/DATA/interactions.csv",
        file="interactions.csv",
        stem=_PANEL_STEM,
        handoff="none",
        axis=", ".join(LIGAND_AXES),
        size=7_560_316,
        licence="Open Database License (ODbL) 1.0 -- share-alike applies to any redistributed derivative",
        note=(
            "24,968 expert-curated ligand-target interactions, 19,987 of them Human, each with "
            "a PubMed id. Measured, and already in real units: the 'Original Affinity Median nm' "
            "column is nanomolar as it stands, so nothing here undoes a logarithm. "
            "Reduce it with gtopdb_offtarget_subset and read it with "
            "ligand_safety.offtarget_profile: that is what this download is for, and the "
            "ten-axis narrowing below was the mistake. Across every target the same filters keep "
            "9,365 interactions over 1,477 targets and 4,891 compounds -- 136 times the rows of "
            "the tox-panel narrowing -- spanning ten family classes and 95 channel and GPCR "
            "families a panel of ten cannot see. It carries what a bare affinity throws away: the "
            "action, the curator's selectivity call, Primary Target (false on 7,330 of them, so "
            "this is a curated OFF-target set), and whether the compound is an approved or a "
            "withdrawn drug. Driven on a novel analogue of terfenadine it returned H1 as the "
            "primary target, Kv11.1 at 56 nM marked off-target -- the hERG block terfenadine was "
            "withdrawn for -- and fexofenadine, its safe successor, as the next analogue. "
            "Also reduce it with "
            "gtopdb_panel_subset, which needs ligands.csv beside interactions.csv for the "
            "structures -- interactions.csv carries a Ligand ID and no SMILES -- and writes "
            "PANEL_SUBSET_COLUMNS, the same format bindingdb_panel_subset writes. That is the "
            "point of it: merge_panel_subsets concatenates the two, and GtoPdb's rows then reach "
            "ligand_safety's per-ligand lookup and similarity read-across instead of sitting in "
            "a reference table nobody opens. Take the volume literally -- this is curation, not "
            "scale: at 2026.3.1 the usable Human counts are hERG 17, CYP3A4 13, Nav1.5 10, P-gp "
            "9, BCRP 5, CYP2D6 and CYP2C9 4 each, CYP1A2 and CYP2C19 3 each, and BSEP 1. Ten "
            "compounds cannot calibrate a cut and one cannot anchor anything, so what it adds "
            "beside the BindingDB measurements is a hand-checked, PubMed-cited second opinion "
            "on the same axes, not coverage. It does NOT close the gap its catalogue row claims: "
            "ligand_safety.CHEMBL_TARGETS already covers all ten axes measured, nav1_5 included, "
            "and ADMET-AI's hole is in prediction, which measured per-compound data cannot fill "
            "for a novel design either. The REST web service at /services/ now requires a "
            "registered API key; this bulk CSV does not."
        ),
        hazard=(
            "Renamed to toxicology.csv the raw download suppresses every verdict rather than "
            "faking one: its first line is a version comment and not the header, and none of its "
            "44 columns is an axis, value or unit column by the names the reader uses, so all ten "
            "axes report untested with blocker toxicology_unreadable. Measured by running "
            "measure_toxicology over a directory holding it under that name. The reduced subset "
            "cannot be misused that way at all -- PANEL_SUBSET_COLUMNS names nothing that reader "
            "accepts -- which is why it is the format and the earlier long-form reference table, "
            "kept apart from the panel only by its filename, is gone. The live danger is reading "
            "the subset's own numbers as a verdict: they are curated actives, so elacridar's 1 nM "
            "hERG belongs to elacridar. Only the per-InChIKey lookup answers for the ligand in "
            "hand, and for a novel design the read-across answers by analogy or not at all."
        ),
    ),
    Source(
        name="IUPHAR/BPS Guide to PHARMACOLOGY ligands",
        url="https://www.guidetopharmacology.org/DATA/ligands.csv",
        file="ligands.csv",
        # No axis of its own and nothing to rename it to: it is an input to the
        # two reductions above, never a table the screen opens.
        stem=None,
        handoff="none",
        axis="companion to interactions.csv; feeds no axis by itself",
        size=6_880_425,
        licence="Open Database License (ODbL) 1.0 -- share-alike applies to any redistributed derivative",
        note=(
            "Not a table the screen reads, and there is nothing to rename it to: it is an input "
            "to the two reductions, never an axis. "
            "The structures, and the reason both GtoPdb reductions take two files: "
            "interactions.csv carries a Ligand ID and no SMILES, so without this one every row "
            "is counted under ligand_without_structure and nothing can be fingerprinted. "
            "gtopdb_panel_subset needs it for the structure; gtopdb_offtarget_subset needs it "
            "for the structure and for the approved/withdrawn marking that "
            "ligand_safety.offtarget_profile reports first, an analogue that is a withdrawn drug "
            "being the loudest thing this release can say. It also publishes its own InChIKey, "
            "which unlike BindingDB's is not stereo-blind; the subset keeps it beside the "
            "recomputed key as inchikey_upstream, and carrying both is the only way that "
            "difference stays visible. Measured on release 2026.3.1 (published 2026-10-12): "
            "6,880,425 bytes, and over the whole release the reductions resolve structures for "
            "58 panel ligands and 4,891 off-target compounds."
        ),
    ),
    Source(
        name="IUPHAR/BPS Guide to PHARMACOLOGY targets and families",
        url="https://www.guidetopharmacology.org/DATA/targets_and_families.csv",
        file="targets_and_families.csv",
        stem=None,
        handoff="none",
        axis="companion to interactions.csv; feeds no axis by itself",
        size=1_896_910,
        licence="Open Database License (ODbL) 1.0 -- share-alike applies to any redistributed derivative",
        note=(
            "Not a table the screen reads, and there is nothing to rename it to: it is an input "
            "to the off-target reduction, never an axis. "
            "The family and class each target belongs to, and the third file "
            "gtopdb_offtarget_subset requires. Without it that reduction returns "
            "targets_not_found and the secondary-pharmacology path does not run at all; "
            "gtopdb_panel_subset does not read it. The families are what make the off-target "
            "profile legible rather than a list of accessions -- on release 2026.3.1 the kept "
            "rows span 95 channel and GPCR families across 1,477 targets, which is the view a "
            "panel of ten proteins cannot give. Measured on that release: 1,896,910 bytes."
        ),
    ),
    Source(
        name="BindingDB measured affinities (full TSV)",
        url="https://www.bindingdb.org/rwd/bind/downloads/BindingDB_All_202611_tsv.zip",
        file="BindingDB_All.tsv",
        member="BindingDB_All.tsv",
        stem=_PANEL_STEM,
        # Not "convert", though there is a path. ``coverage`` counts a convert
        # source as a gap a download could close, and fetching this closes
        # nothing by itself: the path runs through a per-ligand lookup, so it
        # produces a panel reading only for a ligand BindingDB has already
        # measured, and for a novel design it correctly produces none. Calling
        # it "convert" told an operator the toxicology table was one download
        # away, which is the error that field exists to prevent.
        handoff="none",
        axis=", ".join(LIGAND_AXES),
        size=594_284_548,
        licence=(
            "Mixed and marked per row: CC BY 3.0 on BindingDB's own curation, CC BY-SA 3.0 on the "
            "ChEMBL-derived rows, so the share-alike rows govern any redistribution"
        ),
        note=(
            "The November 2026 ZIP is 594,284,548 bytes. Its TSV member is 9,004,582,822 bytes. "
            "bindingdb_panel_subset streams the member directly and retains human panel measurements of Ki, "
            "IC50 and Kd, including the reported relation, assay and curation source. Chain accessions identify "
            "the panel targets. BindingDB's declared number of protein chains describes the target preparation. "
            "measured_nm_from_bindingdb matches the full InChIKey derived from the ligand SMILES by default. "
            "Connectivity matching can include stereoisomers. A fingerprint neighbour's reported activity "
            "belongs to that neighbour."
        ),
        hazard=(
            "The subset is not the panel table and cannot be mistaken for it: its columns are "
            "named panel_axis and affinity_nanomolar precisely so that a copy renamed to "
            "toxicology.tsv reports toxicology_not_reported instead of reporting 1.44M other "
            "people's compounds as this design's. Renaming the raw download does the same. The "
            "live danger is subtler -- take the subset's own numbers as a verdict and you have "
            "read a reference set; only the per-InChIKey lookup answers for the ligand in hand, "
            "and for a novel design it correctly answers nothing."
        ),
    ),
)


@dataclass(frozen=True)
class Query:
    """A source reached by asking it a question instead of downloading it.

    Every entry here has a bulk file far past the limit -- STRING's is 133.76 GB
    and UniProt's 355.49 GB -- and none of it is needed, because the screen asks
    about one target at a time and the services answer that directly. These are
    the endpoints it already calls; the registry names them so that "where does
    this axis get its data" has one truthful answer rather than one per module.
    """

    name: str
    endpoint: str
    axis: str
    called_by: str
    #: How the question is narrowed, so a reader can see it never asks for the
    #: whole thing.
    narrowed: str
    note: str = ""


#: The live lookups. Read-only documentation of what the screen already does --
#: nothing here is called from this module.
QUERIES: tuple[Query, ...] = (
    Query(
        name="UniProt UniRef50 and UniProtKB",
        endpoint="https://rest.uniprot.org/uniref/search, https://rest.uniprot.org/uniprotkb/search",
        axis="paralog",
        called_by="target_liability._uniref_cluster, _cluster_members",
        narrowed="one accession, then that cluster filtered to organism_id:9606 and reviewed:true",
        note=(
            "Replaces the 355.49 GB all-species proteome archive. It does NOT replace "
            "Ensembl Compara, which an earlier version of this note claimed: a UniRef50 "
            "cluster is built around a representative and misses a protein's close "
            "relatives, so KRAS comes back a singleton while HRAS is 85.6 per cent "
            "identical to it. Positive membership here is a family; a singleton is handed "
            "to the Compara query below. The 7.74 MB human FASTA is still downloaded, for "
            "the k-mer scan that needs every sequence at once."
        ),
    ),
    Query(
        name="Ensembl Compara paralogues",
        endpoint="https://rest.ensembl.org/homology/symbol",
        axis="paralog",
        called_by="target_liability._compara_paralogs",
        narrowed=(
            "one gene symbol, type=paralogues, target_taxon=9606; only for a UniRef50 "
            "singleton, and only once a paralog identity cut is named"
        ),
        note=(
            "Asked for one gene instead of downloading the 11.23 GB all-species homology "
            "dump, which is the refusal below. Answers with per-pair identity, which is "
            "what this axis needs and what UniRef50 cannot give. The identity is kept and "
            "the COUNT is not: Compara holds every paralog in its gene trees at any "
            "identity -- 53 for ERBB2, median 17.4 per cent -- so a count judged against a "
            "cut would call a protein a liability on the strength of fifty others that "
            "share no surface with it."
        ),
    ),
    Query(
        name="Human Protein Atlas search API",
        endpoint="https://www.proteinatlas.org/api/search_download.php",
        axis="plasma_abundance",
        called_by="target_liability._abundance_record",
        narrowed="one accession, and only the concentration columns",
    ),
    Query(
        name="STRING interaction partners",
        endpoint="https://string-db.org/api/json/interaction_partners",
        axis="plasma_hub",
        called_by="target_liability._hub_record",
        narrowed=(
            "one identifier, species=9606, with a confidence floor and a partner limit; "
            "sends caller_identity, which STRING asks of automated callers"
        ),
        note="Replaces the 133.76 GB all-organisms link file.",
    ),
)


#: Downloads this refuses, each with the file to take instead. Every one of
#: these sits beside a source above under a near-identical name.
REFUSED: tuple[Refused, ...] = (
    Refused(
        name="UniProt all reference proteomes",
        url=(
            "https://ftp.ebi.ac.uk/pub/databases/uniprot/current_release/knowledgebase/"
            "reference_proteomes/Reference_Proteomes_2026_04.tar.gz"
        ),
        size=355_492_909_566,
        instead="UniProt human reference proteome (canonical)",
    ),
    Refused(
        name="STRING all organisms protein links",
        url="https://stringdb-downloads.org/download/protein.links.v12.5.txt.gz",
        size=133_758_389_005,
        instead="STRING interaction partners",
    ),
    Refused(
        name="HLA Ligand Atlas raw mass spectrometry (PRIDE PXD019643)",
        url="https://www.ebi.ac.uk/pride/archive/projects/PXD019643",
        size=1_393_178_404_881,
        instead="the processed release at hla-ligand-atlas.org, a few MB",
    ),
    Refused(
        name="Ensembl Compara all-species protein homologies",
        url=(
            "https://ftp.ensembl.org/pub/current/tsv/ensembl-compara/homologies/"
            "Compara.116.protein_default.homologies.tsv.gz"
        ),
        size=11_230_217_327,
        instead="Ensembl Compara paralogues",
    ),
    Refused(
        name="PeptideAtlas whole-human build coordinate mapping",
        url="https://peptideatlas.org/builds/human/202601/coordinate_mapping.txt",
        size=19_026_695_342,
        instead="the plasma build under /builds/human/plasma/202508/, about 1.63 GB in total",
    ),
)


class TooLargeError(Exception):
    """A download was refused, or abandoned, for exceeding :data:`SIZE_LIMIT`."""


def readable(size: int) -> str:
    """A byte count a person can compare against the limit at a glance."""
    value = float(size)
    # Up to TB, because the largest thing this refuses is 1.39 TB and reading
    # that as "1297.50 GB" buries the scale of what was nearly fetched.
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(value) < 1024 or unit == "TB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.2f} {unit}"
        value /= 1024
    return f"{value:.2f} TB"


def declared_size(url: str, *, opener: Opener = urlopen) -> int | None:
    """The size the server declares, or None when it declares none.

    None is not an assurance of smallness. IEDB sends no ``Content-Length`` for
    any of its exports, which is exactly why :func:`fetch` counts bytes as well
    as asking.
    """
    request = Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
    try:
        with opener(request, timeout=_TIMEOUT) as response:
            length = response.headers.get("Content-Length")
    except (HTTPError, URLError, OSError, ValueError):
        return None
    try:
        return int(length) if length is not None else None
    except ValueError:
        return None


def fetch(
    source: Source,
    destination: Path,
    *,
    limit: int = SIZE_LIMIT,
    opener: Opener = urlopen,
) -> Path:
    """Download one source into ``destination``, refusing anything over ``limit``.

    Raises :class:`TooLargeError` twice over: once if the server declares a size past
    the limit, and again if the body passes it while arriving. The second check
    is the one that counts -- a declared size is a claim, and the source most
    likely to be large is the one that declares nothing.

    Written to a ``.part`` file and renamed only on success, so an abandoned
    transfer never leaves something that looks like a complete table. A zip is
    unpacked afterwards and the archive removed.
    """
    announced = declared_size(source.url, opener=opener)
    if announced is not None and announced > limit:
        raise TooLargeError(
            f"{source.name} declares {readable(announced)}, over the {readable(limit)} limit. "
            f"Nothing was downloaded."
        )

    destination.mkdir(parents=True, exist_ok=True)
    target = destination / url_leaf(source)
    partial = target.with_suffix(target.suffix + ".part")

    request = Request(source.url, headers={"User-Agent": USER_AGENT})
    received = 0
    try:
        with opener(request, timeout=_TIMEOUT) as response, partial.open("wb") as handle:
            while True:
                chunk = response.read(_CHUNK)
                if not chunk:
                    break
                received += len(chunk)
                if received > limit:
                    raise TooLargeError(
                        f"{source.name} passed the {readable(limit)} limit while downloading "
                        f"and was abandoned. The server declared "
                        f"{'no size' if announced is None else readable(announced)}."
                    )
                handle.write(chunk)
    except TooLargeError:
        partial.unlink(missing_ok=True)
        raise
    except (HTTPError, URLError, OSError) as exc:
        partial.unlink(missing_ok=True)
        raise OSError(f"{source.name} could not be downloaded: {exc}") from exc

    partial.replace(target)
    return _unpack(target, destination, source)


def downloaded_name(source: Source) -> str:
    """The file name :func:`fetch` leaves on disk for this source.

    Recorded on the source rather than derived from the URL, because it is not
    derivable: ``mhc_ligand_full_tsv.zip`` unpacks to ``mhc_ligand_full.tsv``,
    and no rule turns one into the other. Each name here was read off a
    completed download.

    Shared with :func:`downloads` so the write and the presence check cannot
    disagree. They had: ``fetch`` writes the URL leaf, and the old check looked
    for ``<stem>.tsv``, which nothing writes -- so a directory holding every
    download reported every one of them missing.
    """
    return source.file


def url_leaf(source: Source) -> str:
    """What the transfer itself writes, before any unpacking."""
    leaf = source.url.split("/")[-1].split("?")[-1].replace("file_name=", "").split("/")[-1] or "download"
    return Path(leaf).name


def _unpack(archive: Path, destination: Path, source: Source) -> Path:
    """Unpack a zip, or hand back the file as it arrived.

    Only zips, and only members that stay inside ``destination``: an archive
    naming ``../`` is a file this did not ask for, written somewhere it was not
    asked to write.
    """
    if archive.suffix != ".zip":
        return archive
    try:
        with zipfile.ZipFile(archive) as bundle:
            members = [name for name in bundle.namelist() if not name.endswith("/")]
            if source.member is not None:
                members = [name for name in members if name == source.member]
            for name in members:
                leaf = Path(name).name
                if not leaf or ".." in Path(name).parts or Path(name).is_absolute():
                    continue
                with bundle.open(name) as inside, (destination / leaf).open("wb") as handle:
                    shutil.copyfileobj(inside, handle)
    except (zipfile.BadZipFile, OSError) as exc:
        raise OSError(f"{source.name} downloaded but could not be unpacked: {exc}") from exc
    archive.unlink(missing_ok=True)
    return destination


@dataclass(frozen=True)
class Coverage:
    """What one run directory can answer from disk, and what it must ask for."""

    present: tuple[tuple[str, Path], ...]
    missing: tuple[tuple[str, str], ...]
    online: tuple[Query, ...]

    @property
    def complete(self) -> bool:
        """Whether every downloadable table is on disk. The online lookups are
        not counted: they are not something a download could satisfy."""
        return not self.missing

    def as_dict(self) -> dict[str, object]:
        return {
            "present": {stem: str(path) for stem, path in self.present},
            "missing": [{"stem": stem, "source": name} for stem, name in self.missing],
            "online": [{"axis": query.axis, "endpoint": query.endpoint} for query in self.online],
            "complete": self.complete,
        }


def coverage(directory: Path) -> Coverage:
    """Which screen tables this RUN directory has, which it lacks, and what it
    will ask online.

    A run directory, not a download directory -- the two hold different files
    under different names, and :func:`downloads` answers the other question.
    Only stems with a ``handoff`` that can actually produce one are counted: an
    axis no download feeds is not a gap a download could close, and listing it
    as missing sent an operator off to fetch a file that would not have helped.

    A run with every table present is still not offline. Three axes are served
    by REST lookups that no download replaces, and saying so here is the point:
    the alternative is finding out from a network error on a machine that was
    believed stocked.
    """
    present: list[tuple[str, Path]] = []
    missing: list[tuple[str, str]] = []
    for source in SOURCES:
        if source.stem is None or source.handoff != "convert":
            continue
        found = next(
            (
                directory / f"{source.stem}{suffix}"
                for suffix in (".tsv", ".csv")
                if (directory / f"{source.stem}{suffix}").is_file()
            ),
            None,
        )
        if found is None:
            missing.append((source.stem, source.name))
        else:
            present.append((source.stem, found))
    return Coverage(tuple(present), tuple(missing), QUERIES)


@dataclass(frozen=True)
class Downloads:
    """What a download directory holds, by the names :func:`fetch` writes."""

    present: tuple[tuple[Source, Path], ...]
    absent: tuple[Source, ...]

    @property
    def complete(self) -> bool:
        return not self.absent

    def as_dict(self) -> dict[str, object]:
        return {
            "present": {source.name: str(path) for source, path in self.present},
            "absent": [source.name for source in self.absent],
            "complete": self.complete,
        }


def downloads(directory: Path) -> Downloads:
    """Which registry sources are on disk in a download directory.

    Looked up by :attr:`Source.file`, the name the transfer leaves behind, which
    is the only name any of them is ever stored under. The presence check used
    to look for ``<stem>.tsv`` instead, so a directory holding all four
    downloads reported all four missing -- and the obvious fix for that report,
    renaming each file to its stem, is the one action
    :attr:`Source.hazard` exists to prevent.
    """
    found: list[tuple[Source, Path]] = []
    gone: list[Source] = []
    for source in SOURCES:
        path = directory / source.file
        if path.is_file():
            found.append((source, path))
        else:
            gone.append(source)
    return Downloads(tuple(found), tuple(gone))


#: ``>sp|P04626|ERBB2_HUMAN Receptor ... OS=Homo sapiens OX=9606 GN=ERBB2 PE=1 SV=2``
#: -- the accession is the second bar-separated field and the symbol is ``GN=``.
_FASTA_HEAD = re.compile(r"^>(?:sp|tr)\|(?P<accession>[^|]+)\|")
_FASTA_GENE = re.compile(r"(?:^|\s)GN=(?P<gene>\S+)")


def gene_accessions(fasta: Path) -> dict[str, str]:
    """Gene symbol to UniProt accession, read from the human proteome FASTA.

    Needed because the two halves of this axis are keyed differently and the
    reader will not bridge them itself. HPA labels its rows by gene symbol; the
    screen resolves a target to an accession whenever a condition spec names
    one, which on a real run it does. ``_names_target`` compares the two
    literally and deliberately makes no symbol-to-accession lookup, so a
    gene-keyed fold table against an accession target reports
    ``fold_target_not_in_table`` -- the axis reads as "not in the table" when
    the protein is in it under its other name.

    The mapping is read from the file rather than requested, so a staged run
    needs no network for it. Only the reviewed ``sp`` and unreviewed ``tr``
    headers that carry a ``GN=`` are usable; a symbol appearing twice keeps its
    first accession, which in a reference proteome is the canonical one.

    Reads the compressed file directly: it ships as ``.fasta.gz``, and
    unpacking 20,000 sequences to disk to read two fields per header is work
    the caller did not ask for.
    """
    mapping: dict[str, str] = {}
    try:
        opener = gzip.open if fasta.suffix == ".gz" else open
        with opener(fasta, "rt", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if not line.startswith(">"):
                    continue
                head = _FASTA_HEAD.match(line)
                gene = _FASTA_GENE.search(line)
                if head is None or gene is None:
                    continue
                mapping.setdefault(gene.group("gene").strip().upper(), head.group("accession").strip())
    except (OSError, UnicodeError, EOFError) as exc:
        raise OSError(f"{fasta.name} could not be read as a proteome FASTA: {exc}") from exc
    return mapping


# ------------------------------------------------------------------ staging
@dataclass(frozen=True)
class Staging:
    """What a run directory was given, and what it was not.

    Both halves are the report. A screen reads its tables out of the run
    directory, so an axis nothing staged is an axis that comes back untested,
    and this is the cheap moment to learn it rather than after the run.
    """

    #: Stem and the file written, for each axis now readable from the run.
    staged: tuple[tuple[str, Path], ...]
    #: Stem and why, for every axis this could not equip -- both the ones a
    #: download was never going to serve and the ones that fell short.
    unstaged: tuple[tuple[str, str], ...]
    #: Axes that leave the machine when the screen runs, staged or not.
    online: tuple[Query, ...]
    #: The conversion that produced ``plasma_fold``, for its counts.
    fold: FoldConversion | None = None
    #: The human proteome FASTA found on disk, for the off-target axis. Not a
    #: staged table -- it is read where it lies and passed to the screen -- so
    #: it is reported here rather than in :attr:`staged`.
    proteome: Path | None = None
    #: The subset of :attr:`unstaged` a download or conversion could have
    #: closed and did not. Recorded at the point each one is found rather than
    #: recovered by matching on reason text, so a new reason cannot quietly
    #: classify itself as harmless.
    shortfalls: tuple[tuple[str, str], ...] = ()

    @property
    def equipped(self) -> bool:
        """Whether everything a download can serve is in place and answerable.

        Not "every axis is measurable". The axes no download feeds never were,
        and the online ones are not a download's business. What this does cover
        is the accession bridge: a fold table readable but keyed only by gene
        symbol is not equipped for a run that names its target by accession,
        which is what a run with a condition spec does.
        """
        return not self.shortfalls

    def as_dict(self) -> dict[str, object]:
        return {
            "staged": {stem: str(path) for stem, path in self.staged},
            "unstaged": [{"stem": stem, "reason": reason} for stem, reason in self.unstaged],
            "shortfalls": [{"stem": stem, "reason": reason} for stem, reason in self.shortfalls],
            "online": [{"axis": query.axis, "endpoint": query.endpoint} for query in self.online],
            "fold": self.fold.as_dict() if self.fold is not None else None,
            "proteome": str(self.proteome) if self.proteome is not None else None,
            "equipped": self.equipped,
        }


def default_directory() -> Path:
    """``DATA_PATH/screening_tables``, else ``data/screening_tables`` in the checkout.

    The checkout is found by searching upward for ``.project-root`` rather than
    by counting parent directories, so it stays correct if this module ever
    moves to a different depth instead of resolving silently to the wrong
    directory.
    """
    return data_dir("screening_tables")


def downloaded_proteome(source_directory: Path | None = None) -> Path | None:
    """The human reference proteome FASTA on disk, or None.

    The off-target axis takes its FASTA on ``--proteome`` or from
    ``HUMAN_PROTEOME``, and neither of those knows that this program downloads
    that exact file. So a machine that had run the fetch still reported the axis
    unconfigured, with the file it wanted sitting in the download directory.
    This is the lookup that closes that, kept here rather than in the reader so
    that the reader goes on knowing nothing about a download registry.

    Returns the path whether or not it is gzipped, because the reader takes
    either; what is not returned is a name that is absent.
    """
    where = default_directory() if source_directory is None else source_directory
    source = next((item for item in SOURCES if item.axis == "proteome_offtarget"), None)
    if source is None:
        return None
    path = where / source.file
    return path if path.is_file() else None


#: The three GtoPdb files the reductions read, in the order they are passed to
#: :func:`gtopdb_offtarget_subset`. Recorded once, here, for the reason
#: :func:`downloaded_name` gives: a name derived separately by the download, the
#: presence check and the reader is a name they can come to disagree about.
GTOPDB_FILES: tuple[str, ...] = ("interactions.csv", "ligands.csv", "targets_and_families.csv")

#: What :func:`bindingdb_panel_subset` writes, public so a caller that needs to
#: *find* that subset -- to merge it, rather than to write it -- does not have to
#: spell the name a second time.
BINDINGDB_PANEL_SUBSET_FILE = _BDB_SUBSET_FILE


@dataclass(frozen=True)
class GtoPdbRelease:
    """The three GtoPdb files, once every one of them is on disk."""

    interactions: Path
    ligands: Path
    targets: Path

    def as_arguments(self) -> tuple[Path, Path, Path]:
        """In the order :func:`gtopdb_offtarget_subset` takes them."""
        return (self.interactions, self.ligands, self.targets)


def missing_gtopdb_files(source_directory: Path | None = None) -> tuple[str, ...]:
    """Which of :data:`GTOPDB_FILES` are absent or empty, in registry order.

    Empty counts as absent. A zero-byte file is what a transfer that failed
    without raising leaves behind, and a reduction handed one reports a parse
    error rather than the plain truth that nothing was downloaded.
    """
    where = default_directory() if source_directory is None else source_directory
    return tuple(name for name in GTOPDB_FILES if not ((where / name).is_file() and (where / name).stat().st_size > 0))


def gtopdb_release(source_directory: Path | None = None) -> GtoPdbRelease | None:
    """The GtoPdb release on disk, or None while any of its three files is absent.

    The companion to :func:`downloaded_proteome`, and it exists for the same
    reason: the reductions take paths and know nothing about a download
    registry, so without this a machine that had fetched all three still had to
    be told where they were. ``None`` rather than a partial release, because
    two files out of three run neither reduction --
    :func:`gtopdb_panel_subset` needs the first two and
    :func:`gtopdb_offtarget_subset` all three.

    Use :func:`missing_gtopdb_files` to report *which* are absent; this answers
    only whether the release can be reduced.
    """
    where = default_directory() if source_directory is None else source_directory
    if missing_gtopdb_files(where):
        return None
    interactions, ligands, targets = (where / name for name in GTOPDB_FILES)
    return GtoPdbRelease(interactions, ligands, targets)


def stage_for_run(
    run: Path,
    *,
    disease: str,
    source_directory: Path | None = None,
    targets: Collection[str] | None = None,
    release: str = "",
) -> Staging:
    """Put every screen table the downloads can produce into ``run``.

    The bridge that was missing. Downloads land in one directory under their
    own names; the screen reads tables out of the run's own output directory
    under its own stems. Nothing joined the two, so a fully stocked machine
    screened exactly as a bare one did.

    Only ``plasma_fold`` crosses, and only by conversion -- see
    :attr:`Source.handoff`. The other three are named here with the reason they
    stay put, because an operator who has fetched 11 GB is owed that in a
    report rather than in an axis that quietly reads untested.

    ``disease`` is required and not defaulted. The fold table holds one disease
    because the axis asks how far the target moves *in this disease*; guessing
    it would answer a question the run did not ask.

    ``targets`` narrows the fold table, and takes a gene symbol or a UniProt
    accession in any mixture. Accepting both is not a convenience: the screen
    resolves a target to an accession whenever a condition spec names one, so
    the accession is the identifier a caller downstream of generation actually
    holds, while HPA labels its rows by symbol. Each accession is translated
    through the same proteome FASTA that supplies the accession column.

    Narrowing matters more than it looks. Unnarrowed, the conversion writes
    every gene HPA reports for the disease, which passes the reader's 1 MB file
    bound and is refused -- a file on disk that measures nothing.
    """
    where = default_directory() if source_directory is None else source_directory
    if targets is not None and not any(named.strip() for named in targets):
        # Asking for nothing is not asking for everything. Answered before any
        # file is read so the report is about the request and not the machine.
        reason = ("plasma_fold", "targets_named_but_empty")
        return Staging(
            staged=(),
            unstaged=(reason,),
            online=QUERIES,
            fold=None,
            proteome=None,
            shortfalls=(reason,),
        )

    staged: list[tuple[str, Path]] = []
    unstaged: list[tuple[str, str]] = []
    short: list[tuple[str, str]] = []
    fold: FoldConversion | None = None

    def fell_short(stem: str, reason: str) -> None:
        unstaged.append((stem, reason))
        short.append((stem, reason))

    # The gene-to-accession bridge, read once. Without it the conversion still
    # runs and still writes a table the reader accepts -- it just cannot answer
    # a lookup by accession, which is what a run with a condition spec makes.
    # A shortfall, then, not a failure: the screen is thinner, not broken.
    mapping: dict[str, str] = {}
    proteome = next((item for item in SOURCES if item.axis == "proteome_offtarget"), None)
    if proteome is None:
        fell_short("accession lookup", "no_proteome_source_in_registry")
    elif not (where / proteome.file).is_file():
        fell_short("accession lookup", f"proteome_absent:{proteome.file}")
    else:
        try:
            mapping = gene_accessions(where / proteome.file)
        except OSError as exc:
            fell_short("accession lookup", f"proteome_unreadable:{type(exc).__name__}")

    # A target named by accession, translated to the symbol HPA labels rows
    # with. An identifier that is neither is passed through as a symbol: it may
    # be one the proteome does not list, and the conversion reports a miss by
    # name rather than this guessing at it.
    wanted: set[str] | None = None
    if targets is not None:
        by_accession = {accession.upper(): gene for gene, accession in mapping.items()}
        wanted = {by_accession.get(text.upper().split("-", 1)[0], text) for named in targets if (text := named.strip())}

    for source in SOURCES:
        if source.stem is None:
            continue
        if source.handoff != "convert":
            # Not a retryable failure and not a shortfall: this download is a
            # different artifact from the one the axis reads. Reported so the
            # fetched file is accounted for, never placed.
            unstaged.append((source.stem, f"no_conversion_from:{source.file}"))
            continue
        table = where / source.file
        if not table.is_file():
            fell_short(source.stem, f"conversion_source_absent:{source.file}")
            continue
        made = plasma_fold_from_hpa(
            table,
            run,
            disease=disease,
            genes=wanted,
            release=release,
            # ``or None`` so a mapping that came back empty writes the plain
            # three-column table rather than one with an accession column of
            # blanks. The reader skips a blank target cell either way; this is
            # so the file does not advertise a column it cannot answer on.
            accessions=mapping or None,
        )
        fold = made
        if not made.usable or made.path is None:
            fell_short(source.stem, f"conversion_unusable:{made.blocker}")
            continue
        staged.append((source.stem, made.path))
        if mapping and not made.answers_accessions:
            # Written, readable, and matched by nothing a real run asks for.
            # Its own line because the axis otherwise reads as staged while
            # every accession lookup against it misses.
            fell_short(f"{source.stem} (accession lookup)", "no_symbol_mapped")

    # Named, not positional. These last three all default, and an earlier
    # version of this call passed them by position with the field order it
    # remembered rather than the one in the class: the shortfall list went into
    # ``proteome`` and the path into ``shortfalls``, which typechecks, returns
    # and reports a run as equipped while naming a Path as its shortfalls.
    return Staging(
        staged=tuple(staged),
        unstaged=tuple(unstaged),
        online=QUERIES,
        fold=fold,
        proteome=downloaded_proteome(where),
        shortfalls=tuple(short),
    )


def by_name(name: str) -> Source | None:
    """One source by its name, or None."""
    for source in SOURCES:
        if source.name == name:
            return source
    return None


# --------------------------------------------------------------- conversion
#: What the Human Protein Atlas calls its columns, and what the fold reader
#: calls the same things. The reader matches a target on ``gene`` and reads the
#: magnitude from ``log2_fold_change``; HPA writes ``Gene`` and ``logFC``, and
#: ``logFC`` normalises to ``logfc``, which is in neither list. Dropped in as
#: it stands the file is recognised as a table and matched to the target, and
#: then reports no fold at all -- untested rather than wrong, but not working.
_HPA_GENE = "Gene"
_HPA_DISEASE = "Disease"
_HPA_LOG2 = "logFC"
#: HPA's logFC is already a log2 ratio, so it is written out under the reader's
#: own log2 name and NOT rescaled here. ``_linear_fold`` raises 2 to it, and a
#: value converted twice would be a squared fold change compared against a
#: linear cut.
_FOLD_OUT = ("gene", "log2_fold_change", "source")
#: With a gene-to-accession mapping in hand the accession travels too, so a
#: target named either way matches. ``accession`` is in the reader's own
#: ``_FOLD_TARGET_COLUMNS`` and it tries every column in that tuple.
_FOLD_OUT_MAPPED = ("gene", "accession", "log2_fold_change", "source")
#: Diseases named in the report when the requested one matches nothing.
#: How many names a printed line shows before it says how many more there are.
#: A display bound, not a data one: the full list is carried on the record. It
#: was applied to the data, and the names are sorted, so a table holding 72
#: diseases reported the twelve beginning A to C -- an operator whose disease
#: began with S could not tell from that whether it was among the other sixty.
_DISEASE_SAMPLE = 12


def sample_line(names: Collection[str], limit: int = _DISEASE_SAMPLE) -> str:
    """``a, b, c`` or ``a, b, c (+9 more)``. Never a silent truncation."""
    ordered = sorted(names)
    if len(ordered) <= limit:
        return ", ".join(ordered)
    return ", ".join(ordered[:limit]) + f" (+{len(ordered) - limit} more)"


@dataclass(frozen=True)
class FoldConversion:
    """What a conversion wrote, or why it wrote nothing."""

    path: Path | None
    rows: int
    genes: int
    disease: str
    blocker: str | None = None
    #: Every disease name the file holds, when the requested one matched none.
    #: All of them, not a sample: a caller that wants a short line asks
    #: :func:`sample_line` for one, which says how many it left out.
    available: tuple[str, ...] = ()
    #: Every gene this disease reports, when the disease matched and the named
    #: targets did not. Separate from :attr:`available` because they answer
    #: different questions and one tuple holding either would leave a reader
    #: guessing which dimension it was looking at.
    available_genes: tuple[str, ...] = ()
    #: Rows that carry a UniProt accession beside the gene symbol. Reported
    #: because the screen looks the target up by accession on any run whose
    #: condition spec names one, and a table of symbols alone answers that
    #: lookup with "not in the table" for a protein that is in it. Zero here,
    #: with rows above it, is an axis that will read untested.
    mapped: int = 0

    @property
    def written(self) -> bool:
        """Whether a file was produced at all, usable or not."""
        return self.path is not None

    @property
    def usable(self) -> bool:
        """Whether the reader will accept the file at all.

        Accepting it is not the same as matching a target: see
        :attr:`mapped` and :meth:`answers_accessions`. A file the reader takes
        and matches nothing in is still a run that measures nothing.

        The refusals are the reader's own two bounds. Over 1 MB it is refused
        before a row is parsed; past 256 rows *for one target*, after the rows
        are narrowed to it. Narrow with ``genes``.
        """
        return self.path is not None and self.blocker is None

    @property
    def answers_accessions(self) -> bool:
        """Whether a target named by accession can match a row here.

        Separate from :attr:`usable` because it fails separately and silently.
        """
        return self.mapped > 0

    def as_dict(self) -> dict[str, object]:
        return {
            "path": str(self.path) if self.path is not None else None,
            "rows": self.rows,
            "genes": self.genes,
            "disease": self.disease,
            "blocker": self.blocker,
            "available": list(self.available),
            "available_genes": list(self.available_genes),
            "mapped": self.mapped,
        }


def plasma_fold_from_hpa(
    table: Path,
    destination: Path,
    *,
    disease: str,
    genes: Collection[str] | None = None,
    release: str = "",
    accessions: Mapping[str, str] | None = None,
) -> FoldConversion:
    """Write ``plasma_fold.tsv`` for one disease from the HPA blood table.

    The downloaded file is not something the screen can read. This is the step
    between them, and it does three things the reader requires:

    * renames ``logFC`` to ``log2_fold_change``, the spelling
      :func:`~rfproteina.disease.target_liability._linear_fold` recognises and
      raises 2 to. The number is copied across untouched -- HPA's column is
      already log2, and converting it here as well would compare a squared
      fold change against a linear cut;
    * carries ``Gene`` through as ``gene``, which is how the reader matches a
      row to this run's target;
    * adds a ``source`` column. The reader refuses a fold with no provenance --
      "a fold change without a provenance is a rumour" -- and the downloaded
      file names no study anywhere in it, so the provenance is the release this
      was taken from.

    One disease per file, because that is the question the axis asks: how far
    the target moves *in this disease* against healthy controls. The table
    holds every disease at once, and keeping them all would let the reader pick
    the loudest row from an unrelated condition.

    A disease that matches nothing writes **no file** and names the ones that
    are there. An empty ``plasma_fold.tsv`` would read as "no fold reported",
    which is the one outcome indistinguishable from a working file over a
    target that genuinely does not move.
    """
    if not table.is_file():
        return FoldConversion(None, 0, 0, disease, "table_not_found")

    wanted = disease.strip().casefold()
    if not wanted:
        return FoldConversion(None, 0, 0, disease, "disease_not_named")

    wanted_genes = {name.strip().casefold() for name in genes if name.strip()} if genes is not None else None
    if wanted_genes is not None and not wanted_genes:
        return FoldConversion(None, 0, 0, disease, "genes_named_but_empty")

    kept: list[tuple[str, str]] = []
    seen_diseases: dict[str, None] = {}
    #: Genes this disease does report, for the case where the gene filter is
    #: what emptied the result.
    in_disease: dict[str, None] = {}
    source_text = f"Human Protein Atlas blood_pea_disease_de{(' ' + release) if release else ''}; disease={disease}"
    try:
        with table.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            names = reader.fieldnames or []
            for column in (_HPA_GENE, _HPA_DISEASE, _HPA_LOG2):
                if column not in names:
                    return FoldConversion(None, 0, 0, disease, f"column_absent:{column}")
            for row in reader:
                named = (row.get(_HPA_DISEASE) or "").strip()
                if named:
                    seen_diseases.setdefault(named, None)
                if named.casefold() != wanted:
                    continue
                gene = (row.get(_HPA_GENE) or "").strip()
                # Counted before the gene filter, so an empty result can say
                # WHICH filter emptied it. Reported as one blocker, this sent
                # an operator to check a disease name that was never the
                # problem: a run whose target resolves to no accession narrows
                # by something like "gene_9GVN_1", nothing matches, and
                # "disease_not_in_table" named the one dimension that matched.
                if gene:
                    in_disease.setdefault(gene, None)
                if wanted_genes is not None and gene.casefold() not in wanted_genes:
                    continue
                value = (row.get(_HPA_LOG2) or "").strip()
                # A blank logFC is not a change of zero, and written out as
                # empty it is a cell the reader cannot parse -- one of which
                # refuses the whole table. Defensive rather than observed: the
                # v26.0 blood table has no blank and no non-numeric cell in
                # this column across its 676,059 rows.
                if not gene or not value:
                    continue
                try:
                    float(value)
                except ValueError:
                    continue
                kept.append((gene, value))
    except (OSError, UnicodeError, csv.Error) as exc:
        return FoldConversion(None, 0, 0, disease, f"table_unreadable:{type(exc).__name__}")

    if not kept and not in_disease:
        return FoldConversion(None, 0, 0, disease, "disease_not_in_table", tuple(sorted(seen_diseases)))
    if not kept:
        # The disease is there and the named targets are not. A different
        # blocker because it is a different fix: narrow to a gene this disease
        # reports, or accept that this target has no measured fold in it.
        return FoldConversion(
            None,
            0,
            0,
            disease,
            "targets_not_in_table",
            available_genes=tuple(sorted(in_disease)),
        )

    destination.mkdir(parents=True, exist_ok=True)
    out = destination / "plasma_fold.tsv"
    mapped = 0
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(_FOLD_OUT if accessions is None else _FOLD_OUT_MAPPED)
        for gene, value in kept:
            if accessions is None:
                writer.writerow((gene, value, source_text))
                continue
            # An unmapped symbol writes an empty cell rather than dropping
            # the row: its gene still matches a gene-named target, and the
            # reader skips a blank target cell rather than reading it.
            accession = accessions.get(gene.strip().upper(), "")
            mapped += 1 if accession else 0
            writer.writerow((gene, accession, value, source_text))
    # Checked in the reader's own order, and against the thing it actually
    # bounds. The byte limit is applied to the whole file before anything is
    # parsed. The row limit is applied AFTER the rows are narrowed to one
    # target, so it bounds a single protein's rows and not the file's: a
    # labelled campaign table of 19,000 rows is an ordinary campaign table, and
    # counting the file against that bound would call it unusable when the
    # reader would have taken it.
    oversized = None
    size = out.stat().st_size
    worst_gene = Counter(gene for gene, _ in kept).most_common(1)
    worst = worst_gene[0][1] if worst_gene else 0
    if size > FOLD_FILE_LIMIT:
        oversized = f"bytes_over_reader_limit:{size}>{FOLD_FILE_LIMIT}"
    elif worst > FOLD_ROW_LIMIT:
        oversized = f"rows_for_one_gene_over_reader_limit:{worst}>{FOLD_ROW_LIMIT}"
    # Named past ``blocker``. Passing these by position put ``mapped`` into
    # ``available_genes`` the moment a field was inserted between them, which
    # typechecks and reports a count as a list of gene names.
    return FoldConversion(
        out,
        len(kept),
        len({gene for gene, _ in kept}),
        disease,
        oversized,
        mapped=mapped,
    )


def _gtopdb_version(line: str) -> str:
    """``GtoPdb 2026.3.1 (published 2026-10-12)`` from the file's comment line."""
    text = line.strip().strip('"').lstrip("#").strip()
    match = re.search(r"Version:\s*(\S+?)\s*-\s*published:\s*([0-9-]+)", text)
    if match:
        return f"GtoPdb {match.group(1)} (published {match.group(2)})"
    return text


def _gtopdb_reader(handle: Any) -> tuple[str, Any]:
    """A reader positioned past GtoPdb's version comment, and that version.

    Line one is ``"# GtoPdb Version: 2026.3.1 - published: 2026-10-12"`` -- a
    quoted comment, not the header. Hand it to :class:`csv.DictReader` as the
    header and the whole file becomes one column with no usable field name in
    it. The line is consumed only when it really is a comment: if GtoPdb ever
    drops it, line one *is* the header, and seeking back is what keeps it.
    """
    first = handle.readline()
    if first.strip().strip('"').startswith("#"):
        return _gtopdb_version(first), csv.DictReader(handle)
    handle.seek(0)
    return "", csv.DictReader(handle)


def _gtopdb_source(row: Mapping[str, str], version: str) -> str:
    """Provenance for one reading, in the one cell the reader carries through."""
    parts = [version or "IUPHAR/BPS Guide to PHARMACOLOGY"]
    ligand = (row.get(_GTOPDB_LIGAND) or "").strip()
    ligand_id = (row.get(_GTOPDB_LIGAND_ID) or "").strip()
    if ligand:
        parts.append(f"ligand={ligand}" + (f" ({ligand_id})" if ligand_id else ""))
    assay = (row.get(_GTOPDB_ASSAY) or "").strip()
    if assay:
        parts.append(assay)
    action = (row.get(_GTOPDB_ACTION) or "").strip()
    if action and action.casefold() != "none":
        parts.append(action)
    pubmed = (row.get(_GTOPDB_PUBMED) or "").strip()
    if pubmed:
        parts.append(f"PMID {pubmed}")
    parts.append("ODbL")
    return "; ".join(parts)


@dataclass(frozen=True)
class SubsetResult:
    """What the BindingDB reduction wrote, or why it wrote nothing."""

    path: Path | None
    rows: int
    #: Distinct ligands, which is the number the per-ligand lookup can answer for.
    ligands: int
    axes: tuple[str, ...]
    #: Rows of the download read, so a caller can see the reduction ran over the
    #: whole file and not a truncated copy of it.
    scanned: int
    blocker: str | None = None
    missing: tuple[str, ...] = ()
    skipped: Mapping[str, int] = field(default_factory=dict)
    #: Distinct ``Curation/DataSource`` values kept, with counts. The licence is
    #: per row -- CC BY 3.0 for BindingDB's own curation, CC BY-SA 3.0 for the
    #: ChEMBL-derived rows -- so this says which obligations the written subset
    #: carries rather than leaving a redistributor to guess.
    curation: Mapping[str, int] = field(default_factory=dict)

    @property
    def written(self) -> bool:
        """Whether a file was produced at all."""
        return self.path is not None


def _bdb_open(table: Path) -> tuple[Any, Any]:
    """Stream the release TSV from the original archive or a plain export."""
    if table.suffix.lower() == ".zip":
        archive = zipfile.ZipFile(table)
        members = [i for i in archive.infolist() if not i.is_dir() and i.filename.lower().endswith(".tsv")]
        full = [i for i in members if Path(i.filename).name.casefold() == "bindingdb_all.tsv"]
        candidates = full or members
        if len(candidates) != 1:
            archive.close()
            raise KeyError("missing or ambiguous BindingDB release TSV")
        try:
            handle = io.TextIOWrapper(archive.open(candidates[0]), encoding="utf-8-sig", newline="")
        except (OSError, ValueError, RuntimeError, zipfile.BadZipFile):
            archive.close()
            raise
        return archive, handle
    return None, table.open("r", encoding="utf-8-sig", newline="")


def _bdb_source(row: Sequence[str], idx: Mapping[str, int], mark: str) -> str:
    """Provenance for one reading, including the per-row licence marking."""
    parts = ["BindingDB"]
    for column, label in ((_BDB_TARGET_NAME, "target"), (_BDB_DOI, "DOI")):
        position = idx.get(column)
        if position is not None and position < len(row):
            value = row[position].strip()
            if value:
                parts.append(f"{label}={value}" if label == "target" else f"{label} {value}")
    if mark:
        parts.append(mark)
    return "; ".join(parts)


def _inchikey_factory() -> Callable[[str], str]:
    """A memoised SMILES -> InChIKey function. Raises :class:`ImportError` without RDKit.

    BindingDB's own ``Ligand InChI Key`` **cannot** be the join key, and this is
    the measurement that settles it. Over 4,000 real rows at the 2026-11
    release, its first block -- the connectivity hash -- matched RDKit's on
    every single row, while its second block was ``UHFFFAOYSA`` on every row
    whose SMILES carried stereochemistry. That is the no-stereo, no-isotope
    hash: BindingDB stores a stereo-blind key beside a SMILES that is not.
    Only 64.60% of its keys agreed with RDKit's.

    So a design's RDKit key matched against BindingDB's would miss about a third
    of the file, and miss it silently -- an unmeasured-looking answer for a
    compound the file holds. Ketoconazole, with 10,763 CYP3A4 rows in the
    subset, came back untested. The join key is therefore recomputed here from
    BindingDB's SMILES with the same tool
    :func:`~rfproteina.disease.ligand_safety.measured_nm_from_bindingdb` uses on
    the design, which keeps the match exact *and* keeps stereochemistry: the
    14-character skeleton would match either enantiomer, and enantiomers do not
    share a hERG liability.

    A compound can occur in many assays, so its structure is parsed once.
    """
    from rdkit import Chem

    cache: dict[str, str] = {}

    def key(smiles: str) -> str:
        found = cache.get(smiles)
        if found is None:
            mol = Chem.MolFromSmiles(smiles)
            found = str(Chem.MolToInchiKey(mol)) if mol is not None else ""
            cache[smiles] = found
        return found

    return key


def bindingdb_panel_subset(
    table: Path,
    destination: Path,
    *,
    organism: str = "Homo sapiens",
    axes: Collection[str] | None = None,
) -> SubsetResult:
    """Reduce BindingDB's full TSV to the ligand-safety panel, once, offline.

    This is step one of two, and on its own it is **not** a screen. What it
    writes is every measured affinity BindingDB holds against the ten panel
    proteins, for every compound -- other people's compounds. Step two is
    :func:`~rfproteina.disease.ligand_safety.measured_nm_from_bindingdb`, which
    looks *this* design's ligand up in it by InChIKey and is the only thing here
    entitled to produce a panel reading.

    Four things about the file make a naive read wrong, and all four are handled
    rather than hoped about:

    * **Its own InChIKey is stereo-blind.** ``Ligand InChI Key`` agreed with
      RDKit on only 64.60% of 4,000 sampled rows: the connectivity block always
      matched, the second block was ``UHFFFAOYSA`` wherever the SMILES carried
      stereochemistry. The join key is recomputed from the SMILES instead, by
      :func:`_inchikey_factory`, and BindingDB's own key is kept beside it as
      ``inchikey_upstream`` for provenance.
    * **The accession is not in a fixed column.** BindingDB repeats its UniProt
      block per target chain, 50 chains at the 2026-11 release. Every chain's
      SwissProt and TrEMBL primary-ID column is scanned, because in a
      23,736-row sample 5,983 rows kept their accession outside chain 1.
    * **The relation is inside the value.** ``>10000`` and ``> 10000`` are
      bounds, not numbers; the operator is split off and kept in its own column
      for the reader to judge, never stripped to leave an exact-looking figure.
    * **The licence is per row.** ``Curation/DataSource`` marks BindingDB's own
      curation (CC BY 3.0) apart from ChEMBL-derived rows (CC BY-SA 3.0), so it
      is carried into every written row and summarised in
      :attr:`SubsetResult.curation`. A derived table that drops it cannot be
      redistributed honestly.

    ``chains`` records the release's declared number of protein chains. An
    invalid or conflicting declaration is encoded as zero (unknown). Older exports
    without the field use the highest populated chain slot. A measurement
    against a complex is an affinity for the complex, not for one chain of it,
    so the lookup can keep to single-chain rows; the count is kept here rather
    than the decision.

    A panel that matches nothing writes **no file**, and the blocker says which
    narrowing emptied it.
    """
    if not table.is_file():
        return SubsetResult(None, 0, 0, (), 0, "table_not_found")

    wanted = set(axes) if axes is not None else set(BINDINGDB_PANEL_ACCESSIONS)
    unknown = sorted(wanted - set(BINDINGDB_PANEL_ACCESSIONS))
    if unknown:
        return SubsetResult(None, 0, 0, (), 0, f"axes_not_in_panel:{','.join(unknown)}")
    if not wanted:
        return SubsetResult(None, 0, 0, (), 0, "axes_named_but_empty")
    target_organism = organism.strip().casefold()
    if not target_organism:
        return SubsetResult(None, 0, 0, (), 0, "organism_not_named")
    accession_to_axis = {BINDINGDB_PANEL_ACCESSIONS[axis]: axis for axis in wanted}

    # Bounded by the panel's own rows, not the file's: the ten targets are a
    # small slice of 11,509, so this holds tens of thousands of rows and not
    # millions.
    kept: list[tuple[str, ...]] = []
    skipped: Counter[str] = Counter()
    curation: Counter[str] = Counter()
    ligands: set[str] = set()
    scanned = 0
    archive = None
    try:
        archive, handle = _bdb_open(table)
        try:
            # QUOTE_NONE: a bare double quote inside a SMILES or a target name
            # would otherwise start a quoted field and swallow the rest of the
            # file into one cell.
            reader = csv.reader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)
            header = next(reader, None)
            if header is None:
                return SubsetResult(None, 0, 0, (), 0, "table_empty")
            idx = {name: position for position, name in enumerate(header)}
            for column in (_BDB_INCHIKEY, _BDB_SMILES, _BDB_CURATION, _BDB_ORGANISM):
                if column not in idx:
                    return SubsetResult(None, 0, 0, (), 0, f"column_absent:{column}")
            chain_slots = {}
            chains = []
            for name, position in idx.items():
                if name.startswith((*_BDB_CHAIN_PREFIXES, "BindingDB Target Chain Sequence ")):
                    slot = name.rsplit(" ", 1)[-1]
                    if re.fullmatch(r"[1-9][0-9]*", slot):
                        chain_slots[position] = int(slot)
                        if name.startswith(_BDB_CHAIN_PREFIXES):
                            chains.append(position)
            if not chains:
                return SubsetResult(None, 0, 0, (), 0, "column_absent:target chain accession")
            assays = [(label, idx[column]) for label, column in _BDB_ASSAYS if column in idx]
            if not assays:
                return SubsetResult(None, 0, 0, (), 0, "column_absent:affinity")
            inchikey_at = idx[_BDB_INCHIKEY]
            smiles_at = idx[_BDB_SMILES]
            organism_at = idx[_BDB_ORGANISM]
            curation_at = idx[_BDB_CURATION]
            chain_count_at = idx.get(_BDB_CHAIN_COUNT)
            counts = Counter(header)
            used = {_BDB_INCHIKEY, _BDB_SMILES, _BDB_CURATION, _BDB_ORGANISM, _BDB_CHAIN_COUNT,
                    _BDB_TARGET_NAME, _BDB_DOI, *(column for _, column in _BDB_ASSAYS)}
            if any(n > 1 and (key in used or key.startswith(_BDB_CHAIN_PREFIXES)) for key, n in counts.items()):
                return SubsetResult(None, 0, 0, (), 0, "ambiguous_header")
            try:
                inchikey_of = _inchikey_factory()
            except ImportError as error:
                # Without RDKit there is no join key the lookup could match, and
                # BindingDB's own key is stereo-blind. A subset keyed on it would
                # look usable and answer "unmeasured" for a third of the file.
                return SubsetResult(None, 0, 0, (), 0, f"rdkit_unavailable:{error}")
            for row in reader:
                scanned += 1
                matched = sorted(
                    {accession_to_axis[row[at]] for at in chains if at < len(row) and row[at] in accession_to_axis}
                )
                if not matched:
                    continue
                if organism_at >= len(row) or row[organism_at].strip().casefold() != target_organism:
                    skipped["other_organism"] += 1
                    continue
                stored_key = row[inchikey_at].strip() if inchikey_at < len(row) else ""
                smiles = row[smiles_at].strip() if smiles_at < len(row) else ""
                if not smiles:
                    skipped["no_smiles"] += 1
                    continue
                # The join key, recomputed from BindingDB's SMILES with the tool
                # the lookup uses. See _inchikey_factory for why its own key
                # cannot serve.
                inchikey = inchikey_of(smiles)
                if not inchikey:
                    skipped["smiles_unparsed"] += 1
                    continue
                last_chain = max(slot for at, slot in chain_slots.items() if at < len(row) and row[at].strip())
                chain_count = last_chain
                if chain_count_at is not None:
                    declared = row[chain_count_at].strip() if chain_count_at < len(row) else ""
                    try:
                        chain_count = int(declared) if declared.isascii() and declared.isdecimal() else 0
                    except ValueError:
                        chain_count = 0
                    if chain_count < last_chain:
                        chain_count = 0
                mark = row[curation_at].strip() if curation_at < len(row) else ""
                source_text = _bdb_source(row, idx, mark)
                wrote = False
                for label, at in assays:
                    cell = row[at].strip() if at < len(row) else ""
                    if not cell:
                        continue
                    found = _BDB_VALUE.match(cell)
                    if found is None:
                        skipped["value_unparsed"] += 1
                        continue
                    number = float(found.group(2))
                    if not math.isfinite(number) or number < 0:
                        skipped["value_not_a_concentration"] += 1
                        continue
                    relation = found.group(1) or "="
                    for axis in matched:
                        kept.append(
                            (
                                inchikey,
                                stored_key,
                                smiles,
                                axis,
                                BINDINGDB_PANEL_ACCESSIONS[axis],
                                found.group(2),
                                relation,
                                label,
                                str(chain_count),
                                mark,
                                source_text,
                            )
                        )
                    wrote = True
                if wrote:
                    curation[mark] += 1
                    ligands.add(inchikey)
        finally:
            handle.close()
    except (OSError, UnicodeError, csv.Error, KeyError, RuntimeError, zipfile.BadZipFile) as exc:
        return SubsetResult(None, 0, 0, (), scanned, f"table_unreadable:{type(exc).__name__}")
    finally:
        if archive is not None:
            archive.close()

    if not kept:
        return SubsetResult(None, 0, 0, (), scanned, "panel_not_in_table", skipped=dict(skipped))

    destination.mkdir(parents=True, exist_ok=True)
    out = destination / _BDB_SUBSET_FILE
    with TemporaryDirectory(dir=destination) as staging:
        pending_path = Path(staging) / out.name
        with pending_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(PANEL_SUBSET_COLUMNS)
            writer.writerows(kept)
        pending_path.replace(out)

    found = {row[3] for row in kept}
    return SubsetResult(
        out,
        len(kept),
        len(ligands),
        tuple(axis for axis in LIGAND_AXES if axis in found),
        scanned,
        None,
        missing=tuple(axis for axis in LIGAND_AXES if axis in wanted and axis not in found),
        skipped=dict(skipped),
        curation=dict(curation),
    )


#: What the GtoPdb reduction writes, in the shared format.
_GTOPDB_SUBSET_FILE = "gtopdb_panel_subset.tsv"
#: What a merge of several sources writes.
_MERGED_SUBSET_FILE = "panel_subset.tsv"
#: GtoPdb's licence marking. The whole database is ODbL, so unlike BindingDB
#: there is nothing per-row to distinguish -- but a merged file needs to say
#: which obligation a row carries, and share-alike by ODbL is not the same
#: obligation as share-alike by CC BY-SA.
GTOPDB_CURATION = "IUPHAR/BPS GtoPdb (ODbL)"
#: ``ligands.csv``'s own column spellings.
_GTOPDB_LIGAND_KEY = "Ligand ID"
_GTOPDB_LIGAND_SMILES = "SMILES"
_GTOPDB_LIGAND_INCHIKEY = "InChIKey"


def _gtopdb_structures(ligands: Path) -> dict[str, tuple[str, str]]:
    """``Ligand ID -> (SMILES, published InChIKey)`` from GtoPdb's ligands.csv.

    ``interactions.csv`` carries a ligand id and no structure, so the two files
    have to be joined before anything can be fingerprinted. 11,590 of GtoPdb's
    14,029 ligands have a SMILES at 2026.3.1; the rest are antibodies, peptides
    given by sequence, and inorganics, and a row whose ligand has no structure
    is counted rather than dropped silently.
    """
    out: dict[str, tuple[str, str]] = {}
    with ligands.open("r", encoding="utf-8-sig", newline="") as handle:
        _version, reader = _gtopdb_reader(handle)
        names = reader.fieldnames or []
        for column in (_GTOPDB_LIGAND_KEY, _GTOPDB_LIGAND_SMILES):
            if column not in names:
                raise KeyError(column)
        for row in reader:
            key = (row.get(_GTOPDB_LIGAND_KEY) or "").strip()
            smiles = (row.get(_GTOPDB_LIGAND_SMILES) or "").strip()
            if key and smiles:
                out[key] = (smiles, (row.get(_GTOPDB_LIGAND_INCHIKEY) or "").strip())
    return out


def gtopdb_panel_subset(
    interactions: Path,
    ligands: Path,
    destination: Path,
    *,
    species: str = "Human",
    axes: Collection[str] | None = None,
) -> SubsetResult:
    """Reduce GtoPdb to the ligand-safety panel, in the shared subset format.

    The same :data:`PANEL_SUBSET_COLUMNS` :func:`bindingdb_panel_subset` writes,
    which is the whole reason this exists in this shape: merged with BindingDB's
    by :func:`merge_panel_subsets`, GtoPdb's rows reach
    :func:`~rfproteina.disease.ligand_safety.measured_nm_from_bindingdb` and
    :func:`~rfproteina.disease.ligand_safety.nearest_measured_neighbours`
    instead of sitting in a reference table with no reader.

    The 69 usable GtoPdb rows carry independent curation. Each row is
    hand-checked against the primary literature with a PubMed id attached,
    which is worth having as a second opinion on the same compounds.

    Two files, because ``interactions.csv`` has a ligand id and no structure.
    A row whose ligand has no SMILES cannot be fingerprinted and is counted in
    ``skipped`` under ``ligand_without_structure``.

    Narrowing follows the BindingDB reduction: one species, the panel genes by
    HGNC symbol, IC50/Ki/Kd only, the nanomolar column copied across untouched,
    and GtoPdb's ``~`` dropped because it is not a bound the toxicology reader
    knows while ``<`` and ``>`` are.
    """
    if not interactions.is_file():
        return SubsetResult(None, 0, 0, (), 0, "interactions_not_found")
    if not ligands.is_file():
        return SubsetResult(None, 0, 0, (), 0, "ligands_not_found")

    wanted = set(axes) if axes is not None else set(GTOPDB_PANEL_GENES)
    unknown = sorted(wanted - set(GTOPDB_PANEL_GENES))
    if unknown:
        return SubsetResult(None, 0, 0, (), 0, f"axes_not_in_panel:{','.join(unknown)}")
    if not wanted:
        return SubsetResult(None, 0, 0, (), 0, "axes_named_but_empty")
    target_species = species.strip().casefold()
    if not target_species:
        return SubsetResult(None, 0, 0, (), 0, "species_not_named")
    gene_to_axis = {GTOPDB_PANEL_GENES[axis]: axis for axis in wanted}

    try:
        structures = _gtopdb_structures(ligands)
    except KeyError as error:
        return SubsetResult(None, 0, 0, (), 0, f"column_absent:{error.args[0]}")
    except (OSError, UnicodeError, csv.Error) as error:
        return SubsetResult(None, 0, 0, (), 0, f"ligands_unreadable:{type(error).__name__}")

    try:
        key_of = _inchikey_factory()
    except ImportError as error:
        return SubsetResult(None, 0, 0, (), 0, f"rdkit_unavailable:{error}")

    kept: list[tuple[str, ...]] = []
    skipped: Counter[str] = Counter()
    curation: Counter[str] = Counter()
    seen: set[str] = set()
    scanned = 0
    version = ""
    try:
        with interactions.open("r", encoding="utf-8-sig", newline="") as handle:
            version, reader = _gtopdb_reader(handle)
            names = reader.fieldnames or []
            for column in (_GTOPDB_GENE, _GTOPDB_SPECIES, _GTOPDB_ASSAY, _GTOPDB_NM, _GTOPDB_RELATION):
                if column not in names:
                    return SubsetResult(None, 0, 0, (), 0, f"column_absent:{column}")
            if _GTOPDB_LIGAND_ID not in names:
                return SubsetResult(None, 0, 0, (), 0, f"column_absent:{_GTOPDB_LIGAND_ID}")
            for row in reader:
                scanned += 1
                axis = gene_to_axis.get((row.get(_GTOPDB_GENE) or "").strip())
                if axis is None:
                    continue
                if (row.get(_GTOPDB_SPECIES) or "").strip().casefold() != target_species:
                    skipped["other_species"] += 1
                    continue
                if (row.get(_GTOPDB_ASSAY) or "").strip() not in _GTOPDB_ASSAYS:
                    skipped["assay_not_an_affinity"] += 1
                    continue
                raw = (row.get(_GTOPDB_NM) or "").strip()
                if not raw:
                    skipped["no_nanomolar_value"] += 1
                    continue
                try:
                    value = float(raw)
                except ValueError:
                    skipped["value_unparsed"] += 1
                    continue
                if not math.isfinite(value) or value < 0:
                    skipped["value_not_a_concentration"] += 1
                    continue
                relation = _GTOPDB_RELATIONS.get((row.get(_GTOPDB_RELATION) or "").strip())
                if relation is None:
                    skipped["approximate_relation"] += 1
                    continue
                found = structures.get((row.get(_GTOPDB_LIGAND_ID) or "").strip())
                if found is None:
                    skipped["ligand_without_structure"] += 1
                    continue
                smiles, published_key = found
                inchikey = key_of(smiles)
                if not inchikey:
                    skipped["smiles_unparsed"] += 1
                    continue
                kept.append(
                    (
                        inchikey,
                        published_key,
                        smiles,
                        axis,
                        BINDINGDB_PANEL_ACCESSIONS[axis],
                        raw,
                        relation,
                        (row.get(_GTOPDB_ASSAY) or "").strip(),
                        "1",
                        GTOPDB_CURATION,
                        _gtopdb_source(row, version),
                    )
                )
                curation[GTOPDB_CURATION] += 1
                seen.add(inchikey)
    except (OSError, UnicodeError, csv.Error) as error:
        return SubsetResult(None, 0, 0, (), scanned, f"interactions_unreadable:{type(error).__name__}")

    if not kept:
        return SubsetResult(None, 0, 0, (), scanned, "panel_not_in_table", skipped=dict(skipped))

    destination.mkdir(parents=True, exist_ok=True)
    out = destination / _GTOPDB_SUBSET_FILE
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(PANEL_SUBSET_COLUMNS)
        writer.writerows(kept)

    present = {row[3] for row in kept}
    return SubsetResult(
        out,
        len(kept),
        len(seen),
        tuple(axis for axis in LIGAND_AXES if axis in present),
        scanned,
        None,
        missing=tuple(axis for axis in LIGAND_AXES if axis in wanted and axis not in present),
        skipped=dict(skipped),
        curation=dict(curation),
    )


def merge_panel_subsets(sources: Sequence[Path], destination: Path) -> SubsetResult:
    """Concatenate panel subsets into one file the readers can take.

    BindingDB supplies the larger set of compounds. GtoPdb contributes
    measurements curated from the primary literature. Both sources reach the same lookup and the
    same read-across. ``curation`` keeps saying which source and which licence
    each row came from, which matters because the obligations differ -- ODbL
    share-alike on GtoPdb's rows, CC BY-SA on BindingDB's ChEMBL-derived ones.

    A row repeated between sources is written once. Identity is the whole row,
    so the same compound measured twice at different values is two
    measurements, which is what it is.
    """
    if not sources:
        return SubsetResult(None, 0, 0, (), 0, "no_sources_named")
    kept: list[tuple[str, ...]] = []
    seen_rows: set[tuple[str, ...]] = set()
    ligands: set[str] = set()
    curation: Counter[str] = Counter()
    skipped: Counter[str] = Counter()
    scanned = 0
    for source in sources:
        path = Path(source)
        if not path.is_file():
            return SubsetResult(None, 0, 0, (), scanned, f"subset_not_found:{path.name}")
        try:
            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle, delimiter="\t")
                names = frozenset(reader.fieldnames or ())
                absent = [column for column in PANEL_SUBSET_COLUMNS if column not in names]
                if absent:
                    return SubsetResult(None, 0, 0, (), scanned, f"column_absent:{absent[0]}")
                for row in reader:
                    scanned += 1
                    record = tuple((row.get(column) or "").strip() for column in PANEL_SUBSET_COLUMNS)
                    if record in seen_rows:
                        skipped["duplicate_row"] += 1
                        continue
                    seen_rows.add(record)
                    kept.append(record)
                    ligands.add(record[0])
                    if record[9]:
                        curation[record[9]] += 1
        except (OSError, UnicodeError, csv.Error) as error:
            return SubsetResult(None, 0, 0, (), scanned, f"subset_unreadable:{type(error).__name__}")

    if not kept:
        return SubsetResult(None, 0, 0, (), scanned, "sources_held_no_rows", skipped=dict(skipped))

    destination.mkdir(parents=True, exist_ok=True)
    out = destination / _MERGED_SUBSET_FILE
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(PANEL_SUBSET_COLUMNS)
        writer.writerows(kept)

    present = {row[3] for row in kept}
    return SubsetResult(
        out,
        len(kept),
        len(ligands),
        tuple(axis for axis in LIGAND_AXES if axis in present),
        scanned,
        None,
        missing=tuple(axis for axis in LIGAND_AXES if axis not in present),
        skipped=dict(skipped),
        curation=dict(curation),
    )


#: What the off-target reduction writes. A wider table than
#: :data:`PANEL_SUBSET_COLUMNS` because it answers a wider question: not "does
#: this hit one of ten tox targets" but "what else does this hit, and did it
#: get a drug withdrawn".
#:
#: Nothing here is one of the toxicology reader's axis, value or unit columns,
#: for the same reason as the panel subset.
GTOPDB_OFFTARGET_COLUMNS: tuple[str, ...] = (
    "inchikey",
    "inchikey_upstream",
    "smiles",
    "ligand_name",
    "ligand_id",
    "target_id",
    "target_name",
    "gene_symbol",
    "family_class",
    "family_name",
    "affinity_nanomolar",
    "relation",
    "assay",
    "action",
    "selectivity",
    "primary_target",
    "approved",
    "withdrawn",
    "source",
)

_GTOPDB_OFFTARGET_FILE = "gtopdb_offtarget_subset.tsv"

#: ``interactions.csv`` columns the wider reduction reads beyond the panel ones.
_GTOPDB_TARGET_ID = "Target ID"
_GTOPDB_TARGET = "Target"
_GTOPDB_SELECTIVITY = "Selectivity"
_GTOPDB_PRIMARY = "Primary Target"
#: ``ligands.csv`` marks an approved or withdrawn drug with the string ``yes``,
#: not ``true``. Read as a boolean the wrong way this reports every drug as
#: unapproved, which is the quiet direction.
_GTOPDB_YES = "yes"
#: ``targets_and_families.csv`` columns.
_GTOPDB_FAMILY_TARGET = "Target id"
_GTOPDB_FAMILY_CLASS = "Type"
_GTOPDB_FAMILY_NAME = "Family name"
_GTOPDB_FAMILY_SYMBOL = "HGNC symbol"


@dataclass(frozen=True)
class OffTargetSubset:
    """What the off-target reduction wrote, or why it wrote nothing."""

    path: Path | None
    rows: int
    ligands: int
    #: Distinct GtoPdb targets represented.
    targets: int
    #: Rows of the download read.
    scanned: int
    blocker: str | None = None
    #: Interactions per target family class: ``gpcr``, ``vgic``, ``enzyme`` and
    #: the rest. The shape of the coverage, which a single row count hides.
    families: Mapping[str, int] = field(default_factory=dict)
    #: Rows whose ligand is a withdrawn drug, and rows whose ligand is an
    #: approved one. The first is the loudest signal in the file.
    withdrawn_rows: int = 0
    approved_rows: int = 0
    skipped: Mapping[str, int] = field(default_factory=dict)

    @property
    def written(self) -> bool:
        return self.path is not None


def _gtopdb_families(targets: Path) -> dict[str, tuple[str, str, str]]:
    """``Target id -> (family class, family name, HGNC symbol)``."""
    out: dict[str, tuple[str, str, str]] = {}
    with targets.open("r", encoding="utf-8-sig", newline="") as handle:
        _version, reader = _gtopdb_reader(handle)
        names = reader.fieldnames or []
        if _GTOPDB_FAMILY_TARGET not in names:
            raise KeyError(_GTOPDB_FAMILY_TARGET)
        for row in reader:
            key = (row.get(_GTOPDB_FAMILY_TARGET) or "").strip()
            if key and key not in out:
                out[key] = (
                    (row.get(_GTOPDB_FAMILY_CLASS) or "").strip(),
                    (row.get(_GTOPDB_FAMILY_NAME) or "").strip(),
                    (row.get(_GTOPDB_FAMILY_SYMBOL) or "").strip(),
                )
    return out


def gtopdb_offtarget_subset(
    interactions: Path,
    ligands: Path,
    targets: Path,
    destination: Path,
    *,
    species: str = "Human",
) -> OffTargetSubset:
    """Reduce GtoPdb to a secondary-pharmacology panel: every target, not ten.

    :func:`gtopdb_panel_subset` narrows the same download to the ten
    ligand-safety axes and gets 69 rows, which is too few to do anything with.
    That narrowing was the mistake: GtoPdb is a secondary-pharmacology database
    and its own catalogue entry says so. Across **all** targets the same filters
    keep, at release 2026.3.1, 9,365 interactions over 1,477 targets and 4,891
    compounds -- 136 times the interactions and 148 times the targets --
    spanning 95 channel and GPCR families that a panel of ten cannot see: Na_v,
    TRP, opioid, 5-HT, dopamine, muscarinic, histamine, cannabinoid.

    Three columns a bare affinity throws away, all kept here:

    * **What it does.** ``Type`` and ``Action``: inhibitor, antagonist, agonist,
      channel blocker, allosteric modulator. An agonist and a blocker at the
      same potency are different liabilities.
    * **Whether it was the point.** ``Primary Target`` is ``false`` on 7,330 of
      the 9,365, so GtoPdb ships a curated *off*-target set by its own marking,
      and ``Selectivity`` says whether the curator judged the ligand selective.
    * **Whether it ended a drug.** ``ligands.csv`` marks 1,942 approved and 109
      withdrawn compounds that have structures. An analogue that is a withdrawn
      drug is the loudest thing this file can tell you, and
      :func:`~rfproteina.disease.ligand_safety.offtarget_profile` says so first.

    Three files, because the identifiers live apart: ``interactions.csv`` has
    affinities and a ligand id, ``ligands.csv`` the structures and drug status,
    ``targets_and_families.csv`` the family each target belongs to.
    """
    for path, blocker in (
        (interactions, "interactions_not_found"),
        (ligands, "ligands_not_found"),
        (targets, "targets_not_found"),
    ):
        if not path.is_file():
            return OffTargetSubset(None, 0, 0, 0, 0, blocker)
    target_species = species.strip().casefold()
    if not target_species:
        return OffTargetSubset(None, 0, 0, 0, 0, "species_not_named")

    try:
        structures = _gtopdb_structures(ligands)
        families = _gtopdb_families(targets)
    except KeyError as error:
        return OffTargetSubset(None, 0, 0, 0, 0, f"column_absent:{error.args[0]}")
    except (OSError, UnicodeError, csv.Error) as error:
        return OffTargetSubset(None, 0, 0, 0, 0, f"reference_unreadable:{type(error).__name__}")

    drug_status = _gtopdb_drug_status(ligands)
    try:
        key_of = _inchikey_factory()
    except ImportError as error:
        return OffTargetSubset(None, 0, 0, 0, 0, f"rdkit_unavailable:{error}")

    kept: list[tuple[str, ...]] = []
    skipped: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    seen_ligands: set[str] = set()
    seen_targets: set[str] = set()
    withdrawn_rows = approved_rows = 0
    scanned = 0
    version = ""
    try:
        with interactions.open("r", encoding="utf-8-sig", newline="") as handle:
            version, reader = _gtopdb_reader(handle)
            names = reader.fieldnames or []
            for column in (
                _GTOPDB_SPECIES,
                _GTOPDB_ASSAY,
                _GTOPDB_NM,
                _GTOPDB_RELATION,
                _GTOPDB_LIGAND_ID,
                _GTOPDB_TARGET_ID,
            ):
                if column not in names:
                    return OffTargetSubset(None, 0, 0, 0, 0, f"column_absent:{column}")
            for row in reader:
                scanned += 1
                if (row.get(_GTOPDB_SPECIES) or "").strip().casefold() != target_species:
                    skipped["other_species"] += 1
                    continue
                assay = (row.get(_GTOPDB_ASSAY) or "").strip()
                if assay not in _GTOPDB_ASSAYS:
                    skipped["assay_not_an_affinity"] += 1
                    continue
                raw = (row.get(_GTOPDB_NM) or "").strip()
                if not raw:
                    skipped["no_nanomolar_value"] += 1
                    continue
                try:
                    value = float(raw)
                except ValueError:
                    skipped["value_unparsed"] += 1
                    continue
                if not math.isfinite(value) or value < 0:
                    skipped["value_not_a_concentration"] += 1
                    continue
                relation = _GTOPDB_RELATIONS.get((row.get(_GTOPDB_RELATION) or "").strip())
                if relation is None:
                    skipped["approximate_relation"] += 1
                    continue
                ligand_id = (row.get(_GTOPDB_LIGAND_ID) or "").strip()
                found = structures.get(ligand_id)
                if found is None:
                    skipped["ligand_without_structure"] += 1
                    continue
                smiles, published_key = found
                inchikey = key_of(smiles)
                if not inchikey:
                    skipped["smiles_unparsed"] += 1
                    continue
                target_id = (row.get(_GTOPDB_TARGET_ID) or "").strip()
                family_class, family_name, symbol = families.get(target_id, ("", "", ""))
                if not family_class:
                    skipped["target_without_family"] += 1
                approved, withdrawn = drug_status.get(ligand_id, (False, False))
                withdrawn_rows += withdrawn
                approved_rows += approved
                kept.append(
                    (
                        inchikey,
                        published_key,
                        smiles,
                        (row.get(_GTOPDB_LIGAND) or "").strip(),
                        ligand_id,
                        target_id,
                        (row.get(_GTOPDB_TARGET) or "").strip(),
                        symbol,
                        family_class,
                        family_name,
                        raw,
                        relation,
                        assay,
                        (row.get(_GTOPDB_ACTION) or "").strip(),
                        (row.get(_GTOPDB_SELECTIVITY) or "").strip(),
                        (row.get(_GTOPDB_PRIMARY) or "").strip(),
                        "yes" if approved else "",
                        "yes" if withdrawn else "",
                        _gtopdb_source(row, version),
                    )
                )
                family_counts[family_class or "unclassified"] += 1
                seen_ligands.add(inchikey)
                if target_id:
                    seen_targets.add(target_id)
    except (OSError, UnicodeError, csv.Error) as error:
        return OffTargetSubset(None, 0, 0, 0, scanned, f"interactions_unreadable:{type(error).__name__}")

    if not kept:
        return OffTargetSubset(None, 0, 0, 0, scanned, "no_usable_interactions", skipped=dict(skipped))

    destination.mkdir(parents=True, exist_ok=True)
    out = destination / _GTOPDB_OFFTARGET_FILE
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(GTOPDB_OFFTARGET_COLUMNS)
        writer.writerows(kept)

    return OffTargetSubset(
        out,
        len(kept),
        len(seen_ligands),
        len(seen_targets),
        scanned,
        None,
        families=dict(family_counts),
        withdrawn_rows=withdrawn_rows,
        approved_rows=approved_rows,
        skipped=dict(skipped),
    )


def _gtopdb_drug_status(ligands: Path) -> dict[str, tuple[bool, bool]]:
    """``Ligand ID -> (approved, withdrawn)``, read as GtoPdb writes it.

    The cells hold ``yes`` or nothing. Tested against ``"true"`` -- the obvious
    guess -- every drug reads as unapproved and every withdrawal disappears,
    which is the failure that says nothing is wrong.
    """
    out: dict[str, tuple[bool, bool]] = {}
    with ligands.open("r", encoding="utf-8-sig", newline="") as handle:
        _version, reader = _gtopdb_reader(handle)
        for row in reader:
            key = (row.get(_GTOPDB_LIGAND_KEY) or "").strip()
            if not key:
                continue
            out[key] = (
                (row.get("Approved") or "").strip().casefold() == _GTOPDB_YES,
                (row.get("Withdrawn") or "").strip().casefold() == _GTOPDB_YES,
            )
    return out
