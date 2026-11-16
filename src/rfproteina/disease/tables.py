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
import os
import re
import shutil
import zipfile
from collections import Counter
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from rfproteina.disease.limits import USER_AGENT
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
    """``DATA_PATH/screening_tables``, else ``data/screening_tables`` in the checkout."""
    root = os.environ.get("DATA_PATH", "").strip()
    if root:
        return Path(root) / "screening_tables"
    return Path(__file__).resolve().parents[3] / "data" / "screening_tables"


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
