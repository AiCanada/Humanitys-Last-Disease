"""The two nucleic axes, which until now no measurement could reach.

``AXES_BY_KIND`` has given a nucleic binder ``off_target_hybridization`` and
``innate_immune`` since the classes were defined, and ``BARS`` makes both of
them the ``oligo`` bar. Nothing measured either one. A nucleic design could
therefore only ever come back ``untested``, with no blocker saying why and no
file a user could supply to change it: the strategy was declared and not
implemented.

Both axes are read from the designed chain, which the screen already has.

**Off-target hybridization** mirrors ``proteome_offtarget``: exact seed matches
against a reference the user supplies, counted, with the file's size and digest
stored beside the count so a clear says which file it is a clear against. The
seed is 15 bases, the >= 15-nt perfect-complementarity width used for antisense
off-target triage, which is why a seed-region match alone is not screened here;
4**7 is 16384 and every 7-mer occurs in a transcriptome thousands of times
over, which is why it is not shorter. Fifteen bases does **not** make a perfect
match rare: 4**15 is 1.07e9 against the roughly 3e8 bases of a human
transcriptome, so any one 15-mer is expected 0.28 times in it,
and a design presents 2*(len-14) seeds. Measured against a 200-Mbase random
reference, a 20-nt design matched 0 of its 12 seeds, a 60-nt design 19 of 92
and an 80-nt design 24 of 132, every one of them by chance. A hit here is a
pointer into the reference for someone to read, not a count of off-targets;
only a clear is a statement about the whole file. Both orientations are searched:
an antisense oligo hybridizes where a transcript carries its reverse complement,
and a sense or passenger strand hybridizes as itself. U is read as T throughout,
so an RNA design and a DNA reference compare.

The reference has to read as nucleotide before it is searched. The leading
letters of the stream are sampled, and a file that is not mostly bases is
refused as ``transcriptome_not_nucleic``: a protein FASTA holds effectively no
window of fifteen consecutive ACGT letters -- 0.25**15 is 9.3e-10 a position,
about one hundredth of a window across a whole proteome -- so searching one
answers every seed "no match" and would read as a clear against a file that
cannot hold a transcript. The file-size limit is set above a whole human
transcriptome rather than below it (see ``TRANSCRIPTOME_FILE_LIMIT``), because
the scan is streaming and the reference the seed width is argued against is the
one a user is most likely to supply. Measured on a 300,000,000-base random
FASTA wrapped at sixty columns, 305,000,010 bytes or 290.9 MiB: scanned in 26 s
at 288 MiB peak resident against a 118 MiB baseline, holding the file's letters
a block at a time and never whole. Nine of that run's seeds matched by chance,
which is the 0.28-per-seed arithmetic above showing up.

A **designed chain** may carry IUPAC ambiguity letters. They are kept: such a
letter holds a position and names more than one base, so no seed is packed
through one and no hexamer is read through one, and the rest of the chain is
still measured. Dropping the letter instead would join two bases that are not
neighbours and invent a seed the design does not carry, and refusing the whole
chain turned an ordinary degenerate oligo into ``sequence_not_nucleic``.

**Innate immune** reads two published TLR9 CpG motifs, and a design matching
either is a hit. The murine consensus is 5'-purine-purine-CG-pyrimidine-
pyrimidine-3' (Krieg et al., Nature 1995, 374:546; the optimal murine hexamer
GACGTT is one instance of it). The optimal **human** hexamer is 5'-GTCGTT-3'
(Hartmann et al., J Immunol 2000, 164:1617, from over 250 phosphorothioate ODN
on human B and NK cells). GTCGTT is **not** an instance of the murine
consensus -- its second base is a pyrimidine -- and Hartmann reports that ODN
built on the murine formula "show no or only weak activity in human immune
cells", so one rule cannot stand for the other and the two are not merged into
a third pattern neither paper states. Each hit names the motif that fired
beside its position. While only the murine rule was read, CpG 7909 / ODN 2006,
the prototype human CpG-B agonist, carried three GTCGTT hexamers and came back
``no_pass_cut``. A design with no CpG dinucleotide at all
is **not** a clear, and that is deliberate: it cannot present the CpG motif, but
TLR9 is not the only innate sensor an oligo meets -- RIG-I, TLR3, TLR7 and TLR8
and the backbone chemistry are all outside what a sequence consensus can
answer. It is recorded as ``cpg_free_not_evidence``, the same refusal
``paralog`` makes for a singleton cluster, so a clear here is curated into
``axis_evidence.json`` by someone who can speak for the rest.

Nothing here reaches the network, and no designed sequence is stored on a
record.
"""

from __future__ import annotations

import gzip
import hashlib
import os
import zlib
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO

import numpy as np

from rfproteina.disease.combinations import rollup_axis

#: The axes this module measures, in the order ``AXES`` declares them.
NUCLEIC_LIABILITY_AXES = ("off_target_hybridization", "innate_immune")
#: Bases in one seed. See the module docstring for why this is 15 and not 7.
SEED_WIDTH = 15
#: Where a reference may be named when none sits beside the run.
TRANSCRIPTOME_ENV = "HUMAN_TRANSCRIPTOME"
#: The stem a reference takes beside a run, the way ``solubility.tsv`` does.
TRANSCRIPTOME_FILE = "transcriptome"
#: Compressed spellings included, because that is the only form the references
#: this axis is argued for are distributed in: Ensembl and GENCODE ship cDNA as
#: ``.fa.gz`` and nothing ships it unpacked. Without them the file the
#: ``4**15`` seed arithmetic is stated for was never found beside a run, and
#: ``off_target_hybridization`` stayed untested on a machine that had it.
_TRANSCRIPTOME_SUFFIXES = (
    ".fa",
    ".fasta",
    ".fna",
    ".txt",
    ".fa.gz",
    ".fasta.gz",
    ".fna.gz",
    ".txt.gz",
)
#: A reference past this is not read. The scan is streaming and never holds
#: more than ``_BLOCK`` letters, so this is a guard against a mistyped path
#: rather than a memory bound, and it has to sit above the reference the seed
#: width is argued against. A human cDNA FASTA of roughly 3e8 bases costs
#: 3e8 * 61/60 bytes at sixty columns, i.e. 291 MiB, so the former 256 MiB
#: refused the one file the 4**15 arithmetic is stated for. A curated
#: off-target panel -- a family's transcripts, an expressed-gene subset, a CDS
#: selection -- is far below this; a whole multi-species archive is not.
TRANSCRIPTOME_FILE_LIMIT = 512 * 1024 * 1024
#: Below this a clear means too little to make. The same guard the proteome
#: reader applies: a two-record FASTA shares nothing with anything, and a clear
#: against it would read like a clear against a reference.
TRANSCRIPTOME_MIN_BASES = 1_000_000
#: Sequence letters held at once while scanning, plus the overlap that keeps a
#: seed straddling two blocks from being missed.
_BLOCK = 1 << 22

_CODES = {"A": 0, "C": 1, "G": 2, "T": 3, "U": 3}
#: The IUPAC nucleotide letters that name more than one base (NC-IUB
#: recommendations 1984, Nucleic Acids Res 1985, 13:3021). They are legal in an
#: ordered oligo and they hold a position, so they are kept rather than
#: dropped or taken as a reason to refuse the chain: ``_pack`` already returns
#: None for any window holding one, and ``_is_consensus`` tests membership in
#: ``_PURINE``/``_PYRIMIDINE``, which none of them is in, so neither a seed nor
#: a hexamer can be read through one.
_AMBIGUOUS = frozenset("NRYKMSWBDHV")
_PURINE = frozenset("AG")
_PYRIMIDINE = frozenset("CT")
#: Letters that are bases once U has been read as T. A nucleotide reference is
#: almost all of these; a protein FASTA is about a quarter of them, because A,
#: C, G and T are amino-acid letters too. This is the mirror of
#: ``liability._is_nucleotide``, which keeps a coding FASTA out of the protein
#: search.
_BASE_LETTERS = frozenset("ACGTN")
#: Where the two cases sit: a nucleotide reference is 1.00 of ``_BASE_LETTERS``
#: apart from its ambiguity codes and a uniform protein FASTA is 5/20 = 0.25,
#: so 0.90 leaves a real reference a tenth of its letters for masking and
#: ambiguity codes while a protein FASTA would have to be enriched 3.6-fold in
#: A, C, G, T and N over its own alphabet to reach it.
_BASE_SHARE = 0.90
#: Below this many sampled letters the share says nothing, and the sample is
#: capped so the guard costs the head of one block rather than a second pass.
_ALPHABET_MIN = 1_000
_ALPHABET_SAMPLE = 100_000
#: 2 bits a base, so one seed is 30 bits and fits a uint32 with room over.
_BITS = 2
_INVALID = np.uint8(255)


@dataclass(frozen=True)
class Reference:
    """What one scan of a supplied reference found, and what it was.

    ``matched`` holds the seeds that occurred in the file, out of the seeds the
    scan was asked about. Every design's seeds go into one scan, so a run of
    two hundred designs reads the reference once rather than two hundred times.
    """

    blocker: str | None = None
    bases: int = 0
    digest: str = ""
    matched: frozenset[int] = field(default_factory=frozenset)
    path_name: str = ""


def measure_nucleic_liability(
    samples: Sequence[tuple[str, str | None]],
    *,
    directory: Path,
    transcriptome: Path | None = None,
) -> tuple[dict[str, dict[str, object]], dict[str, dict[str, object]]]:
    """Both nucleic axes for one run, and the off-target record per design.

    ``samples`` are ``(example_id, sequence)`` in filename order, as the screen
    already read them. The returned axes are the run's rollup: any design that
    hits makes the axis a hit and is named in ``failed``; every design clear
    makes it clear; anything else stays untested.
    """
    ids = [example_id for example_id, _sequence in samples]
    reference = _scan(_reference_path(directory, transcriptome), _needles(samples))
    by_example: dict[str, dict[str, object]] = {}
    innate: dict[str, dict[str, object]] = {}
    for example_id, sequence in samples:
        by_example[example_id] = off_target_record(sequence, reference)
        innate[example_id] = innate_immune_record(sequence)
    axes = {
        "off_target_hybridization": rollup_axis(ids, by_example, _off_target_absent(reference)),
        "innate_immune": rollup_axis(ids, innate, innate_immune_record(None)),
    }
    return axes, by_example


def off_target_record(sequence: str | None, reference: Reference) -> dict[str, object]:
    """Seeds of one design that occur in the reference, or why none were counted.

    A clear is a statement about one file, so the file's base count and digest
    travel with it. Neither refuses anything on its own: a reference nobody but
    its author would supply is their problem, and the record says enough for a
    reader to see that it was theirs.
    """
    if reference.blocker is not None:
        return _axis("untested", None, reference.blocker)
    chain = _normalized(sequence)
    if chain is None:
        return _axis("untested", None, "sequence_not_nucleic" if sequence else "sequence_absent")
    if len(chain) < SEED_WIDTH:
        return _axis("untested", None, "chain_shorter_than_seed")
    seeds = _seeds(chain) | _seeds(_reverse_complement(chain))
    if not seeds:
        # Every window of the design spans an ambiguity code, so the search ran
        # over nothing. Reading that as zero matches published a clear against
        # a named reference file on the strength of no comparison at all --
        # the same empty claim the alphabet guard refuses on the other side.
        return _axis("untested", None, "design_windows_unscannable")
    count = len(seeds & reference.matched)
    record = _axis("hit" if count else "clear", count, None)
    record["seed_width"] = SEED_WIDTH
    record["seeds_searched"] = len(seeds)
    record["reference_bases"] = reference.bases
    record["reference_digest"] = reference.digest
    record["reference_file"] = reference.path_name
    return record


def innate_immune_record(sequence: str | None) -> dict[str, object]:
    """The two published TLR9 CpG motifs in one design.

    A hexamer matching the murine purine-purine-CG-pyrimidine-pyrimidine
    consensus or the human GTCGTT is a hit, and the count, the positions and
    the motif that fired at each one are stored so the hit can be found in the
    chain and read against its own paper. A chain with no CpG at all is refused
    rather than cleared, for the reason the module docstring gives.
    """
    chain = _normalized(sequence)
    if chain is None:
        return _axis("untested", None, "sequence_not_nucleic" if sequence else "sequence_absent")
    cpg = [index for index in range(len(chain) - 1) if chain[index : index + 2] == "CG"]
    motifs = [(index, _consensus_motif(chain, index)) for index in cpg]
    motifs = [(index, name) for index, name in motifs if name is not None]
    if motifs:
        record = _axis("hit", len(motifs), None)
        # The one-based start of the hexamer, which is two bases before the
        # CG it is named for. Not the same number as the drug-type report's
        # ``cpg_positions``, which is the one-based position of the C.
        record["motif_positions"] = [index - 1 for index, _name in motifs]
        # Which published rule each position was read by, since the two carry
        # different evidence: RRCGYY is Krieg's murine consensus and GTCGTT is
        # Hartmann's human-optimal hexamer.
        record["motif_sites"] = [{"position": index - 1, "motif": name} for index, name in motifs]
        record["cpg_count"] = len(cpg)
        record["method"] = _CONSENSUS_METHOD
        return record
    if not cpg:
        record = _axis("untested", 0, "cpg_free_not_evidence")
        record["cpg_count"] = 0
        record["method"] = _CONSENSUS_METHOD
        return record
    # CpG is present and none of it is one of the two consensus hexamers. This
    # was reported as ``no_pass_cut``, whose shared sentence says no pass cut
    # applies to the number -- and one does: a single consensus hexamer is a hit,
    # which this axis's own High guidance states. So the page denied the cut its
    # own guide asserts, and the board had to carry a per-axis override to stop
    # it stating the opposite of itself.
    #
    # The real reason the measured zero is refused is the one
    # ``cpg_free_not_evidence`` gives for its own zero: zero consensus hexamers
    # is not an absence of innate-immune signal, because the other sensors are
    # not scored here. That is a different fact from "no cut exists", so it gets
    # its own name.
    record = _axis("untested", 0, "cpg_not_consensus")
    record["cpg_count"] = len(cpg)
    record["method"] = _CONSENSUS_METHOD
    return record


#: The human-active hexamer. Hartmann et al., J Immunol 2000, 164:1617 tested
#: over 250 phosphorothioate ODN on human B and NK cells and found 5'-GTCGTT-3'
#: optimal for human, chimpanzee and rhesus leukocytes, while reporting that
#: ODN carrying the murine purine-purine-CG-pyrimidine-pyrimidine formula "show
#: no or only weak activity in human immune cells". GTCGTT is not an instance
#: of that formula -- its second base is a pyrimidine -- so the murine rule
#: alone cannot see CpG 7909 / ODN 2006, the prototype human CpG-B agonist and
#: the one a design is most likely to carry.
_HUMAN_HEXAMER = "GTCGTT"
#: What each rule is called in a record, so a hit says which one fired.
_MURINE_MOTIF = "RRCGYY"
#: Named once, so the record and the reference cannot drift apart.
_CONSENSUS_METHOD = "TLR9 CpG motifs: murine RRCGYY (Krieg 1995) and human GTCGTT (Hartmann 2000)"


def _consensus_motif(chain: str, index: int) -> str | None:
    """Which published TLR9 hexamer the CG at ``index`` sits in, if either.

    The human hexamer is tested first because it is literal; the murine rule is
    the Krieg consensus, purine-purine-CG-pyrimidine-pyrimidine. The two cannot
    both match one CG -- GTCGTT fails the second purine -- so a CG contributes
    at most one motif either way. A hexamer needs two bases each side of the
    CG, so a CG in the first two or last two positions cannot be one.
    """
    if index < 2 or index + 4 > len(chain):
        return None
    if chain[index - 2 : index + 4] == _HUMAN_HEXAMER:
        return _HUMAN_HEXAMER
    if (
        chain[index - 2] in _PURINE
        and chain[index - 1] in _PURINE
        and chain[index + 2] in _PYRIMIDINE
        and chain[index + 3] in _PYRIMIDINE
    ):
        return _MURINE_MOTIF
    return None


def _is_consensus(chain: str, index: int) -> bool:
    """Whether the CG at ``index`` sits in a hexamer either published rule reads."""
    return _consensus_motif(chain, index) is not None


def _off_target_absent(reference: Reference) -> dict[str, object]:
    return _axis("untested", None, reference.blocker or "sequence_absent")


def _axis(state: str, measurement: int | float | None, blocker: str | None) -> dict[str, object]:
    """The three fields every axis record carries, in the shape the judge reads."""
    return {"state": state, "measurement": measurement, "blocker": blocker}


def _needles(samples: Sequence[tuple[str, str | None]]) -> set[int]:
    """Every seed any design in the run could hybridize through, both orientations."""
    found: set[int] = set()
    for _example_id, sequence in samples:
        chain = _normalized(sequence)
        if chain is None or len(chain) < SEED_WIDTH:
            continue
        found |= _seeds(chain)
        found |= _seeds(_reverse_complement(chain))
    return found


def _seeds(chain: str) -> set[int]:
    """Packed seeds of one chain. A window holding an unknown base is left out."""
    found: set[int] = set()
    for start in range(len(chain) - SEED_WIDTH + 1):
        packed = _pack(chain[start : start + SEED_WIDTH])
        if packed is not None:
            found.add(packed)
    return found


def _pack(window: str) -> int | None:
    value = 0
    for letter in window:
        code = _CODES.get(letter)
        if code is None:
            return None
        value = (value << _BITS) | code
    return value


def _normalized(sequence: str | None) -> str | None:
    """Upper-case letters with U read as T, or None when this is not an oligo.

    An IUPAC ambiguity letter is kept rather than dropped or refused. It is a
    legal base in an ordered oligo and it occupies a position, so removing it
    would join two bases that are not neighbours, and refusing the chain for it
    threw away every window the letter does not touch. A window holding one
    cannot be claimed to match or not to match, so it is unscanned, not absent:
    ``_pack`` drops exactly those windows and ``_is_consensus`` cannot read a
    hexamer through one.
    """
    if not sequence:
        return None
    chain = "".join(sequence.split()).upper().replace("U", "T")
    if not chain:
        return None
    if any(letter not in _CODES and letter not in _AMBIGUOUS for letter in chain):
        return None
    return chain


def _reverse_complement(chain: str) -> str:
    return chain[::-1].translate(_COMPLEMENT)


#: The IUPAC complements, so an ambiguity letter reverse-complements to the
#: letter that names the complementary set rather than to itself (NC-IUB 1984).
#: No seed is packed through one either way; a chain that read back wrongly
#: would still be wrong to show.
_COMPLEMENT = str.maketrans("ACGTNRYKMSWBDHV", "TGCANYRMKSWVHDB")


def _size_limit(path: Path) -> int:
    """The byte bound for this reference, in the units the file is stored in.

    ``TRANSCRIPTOME_FILE_LIMIT`` is argued in uncompressed FASTA bytes -- 3e8
    bases at sixty columns is 291 MiB -- and it is a mistyped-path guard rather
    than a memory bound, because the scan streams. Applied unchanged to a
    gzipped reference it would be the wrong question by about fourfold: a human
    cDNA FASTA compresses to roughly 75 MiB, so the bound would admit a gzip
    holding four times the content it was chosen to admit.

    So a compressed reference is bounded on its compressed size, at the share
    of the limit that a FASTA of nucleotide text actually compresses to. The
    quarter is deliberately generous: the point is to refuse a path typed by
    mistake, not to second-guess a compressor.
    """
    if path.suffix == ".gz":
        return TRANSCRIPTOME_FILE_LIMIT // 4
    return TRANSCRIPTOME_FILE_LIMIT


def _open_reference(path: Path) -> IO[str]:
    """The reference as text, gzipped or not, tolerating a byte-order mark.

    ``utf-8-sig`` rather than ``utf-8``, which is what every sibling reader in
    this package uses and for the same reason: a mark is not whitespace, and a
    valid nucleotide FASTA carrying one was refused outright as
    ``transcriptome_unreadable``. Here it would also have been counted as a
    non-base letter by the alphabet guard.

    Decided on the extension, so an uncompressed reference opens exactly as it
    did before.
    """
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8-sig", errors="strict")
    return path.open(encoding="utf-8-sig", errors="strict")


def _reference_path(directory: Path, transcriptome: Path | None) -> Path | None:
    """The reference this run screens against. An explicit path wins over both.

    Beside the run first, because that is where every other supplied table
    lives and it travels with the run. The environment is the shared copy.
    """
    if transcriptome is not None:
        return transcriptome
    for suffix in _TRANSCRIPTOME_SUFFIXES:
        candidate = directory / f"{TRANSCRIPTOME_FILE}{suffix}"
        if candidate.is_file():
            return candidate
    named = os.environ.get(TRANSCRIPTOME_ENV, "").strip()
    return Path(named) if named else None


def _scan(path: Path | None, needles: set[int]) -> Reference:
    """One pass over the reference, answering every design's seeds at once.

    Blocked in sequence letters rather than in file bytes, with a seed's worth
    of overlap carried between blocks, so a match that straddles a block
    boundary is still found and the whole file is never held in memory. The
    digest is taken over the letters, not the bytes, so re-wrapping a FASTA
    does not read as a different reference.

    The leading letters are counted as they stream past, and a file whose
    alphabet is not nucleotide is refused rather than searched: every other
    check here passes for a protein FASTA, which would then clear the axis
    having matched nothing.
    """
    if path is None:
        return Reference(blocker="transcriptome_not_configured")
    try:
        if path.stat().st_size > _size_limit(path):
            return Reference(blocker="transcriptome_too_large")
    except OSError:
        return Reference(blocker="transcriptome_unreadable")
    wanted = np.fromiter(sorted(needles), dtype=np.uint64, count=len(needles))
    matched: set[int] = set()
    bases = 0
    sampled = 0
    base_letters = 0
    digest = hashlib.sha256()
    carry = ""
    try:
        with _open_reference(path) as handle:
            for block, starts_record in _letter_blocks(handle):
                if starts_record:
                    # A new entry, so the previous one's tail is not its
                    # neighbour. Dropping the carry here is what keeps a seed
                    # from being read across a record junction.
                    carry = ""
                if sampled < _ALPHABET_SAMPLE:
                    # Sampled off the stream rather than read a second time,
                    # so the guard costs the head of the first block and the
                    # scan stays one pass.
                    head = block[: _ALPHABET_SAMPLE - sampled]
                    sampled += len(head)
                    base_letters += sum(1 for letter in head if letter in _BASE_LETTERS)
                bases += len(block)
                digest.update(block.encode("ascii"))
                window = carry + block
                matched |= _found(window, wanted)
                carry = window[-(SEED_WIDTH - 1) :] if SEED_WIDTH > 1 else ""
    except (OSError, UnicodeError, MemoryError, ValueError, EOFError, zlib.error):
        return Reference(blocker="transcriptome_unreadable")
    if not bases:
        return Reference(blocker="transcriptome_empty")
    if sampled >= _ALPHABET_MIN and base_letters < _BASE_SHARE * sampled:
        # pep.all.fa where cds.all.fa was meant, or $HUMAN_PROTEOME copied into
        # $HUMAN_TRANSCRIPTOME. This is the mirror of the protein reader's
        # ``proteome_not_protein``: a protein FASTA holds effectively no window
        # of fifteen consecutive ACGT letters, so it answers every seed "no
        # match" and the axis read clear against a file that cannot hold a
        # transcript.
        return Reference(
            blocker="transcriptome_not_nucleic",
            bases=bases,
            digest=digest.hexdigest()[:16],
            path_name=path.name,
        )
    if bases < TRANSCRIPTOME_MIN_BASES:
        return Reference(
            blocker="transcriptome_too_small",
            bases=bases,
            digest=digest.hexdigest()[:16],
            path_name=path.name,
        )
    return Reference(
        bases=bases,
        digest=digest.hexdigest()[:16],
        matched=frozenset(matched),
        path_name=path.name,
    )


def _letter_blocks(handle: Iterable[str]) -> Iterator[tuple[str, bool]]:
    """Sequence letters of a FASTA or a bare sequence, in blocks.

    A generator, not a list: the point of blocking is that a reference the
    size of a transcriptome is never held whole, and collecting the blocks
    first would have held it twice over.

    Each block is paired with whether it **begins a record**, so the caller can
    refuse to stitch the end of one record onto the start of the next. Without
    that this function did to records exactly what its last paragraph forbids
    for letters: the last bases of one entry were joined to the first bases of
    the following one and every seed across that junction was reported as a
    match the reference does not hold. On the input shape
    ``TRANSCRIPTOME_FILE_LIMIT`` names as expected -- ten thousand entries of a
    hundred bases -- 139,869 of 999,512 searched windows, fourteen per cent,
    were invented by the join and every one of them reported matched.

    A ``>`` line starts a record and holds no sequence. A ``;`` line is a
    legacy FASTA comment: it holds no sequence either, but it does **not** start
    a record, and treating it as one would cut a record in two and lose a seed
    that genuinely spans it. Nothing else is dropped: a letter this module does
    not know still occupies a position, and silently removing it would join two
    bases that are not neighbours and invent a seed the reference does not
    contain.

    Flushing per record costs ``_scan`` one :func:`_found` call per entry where
    a four-mebibyte block had taken one. Measured on the shape above, ten
    thousand entries of a hundred bases: 0.789s against 0.187s, so about four
    times the work and still under a second for a million bases. That is the
    price of not reporting matches the reference does not hold, and it is
    recorded here so nobody pays it twice over by guessing at it.
    """
    buffer: list[str] = []
    size = 0
    starts_record = True
    for line in handle:
        text = line.strip()
        if not text:
            continue
        if text[0] == ">":
            if buffer:
                yield "".join(buffer), starts_record
                buffer = []
                size = 0
            starts_record = True
            continue
        if text[0] == ";":
            continue
        buffer.append(text.upper().replace("U", "T"))
        size += len(text)
        if size >= _BLOCK:
            yield "".join(buffer), starts_record
            buffer = []
            size = 0
            starts_record = False
    if buffer:
        yield "".join(buffer), starts_record


def _found(window: str, wanted: np.ndarray) -> set[int]:
    """Which of ``wanted`` occur in this block of letters.

    The block's own seeds are packed with fifteen vectorized shifts rather than
    a Python loop over every position, and a window holding a base outside
    ACGT is dropped by the same arithmetic instead of being skipped by hand.
    """
    if wanted.size == 0 or len(window) < SEED_WIDTH:
        return set()
    codes = _coded(window)
    width = len(window) - SEED_WIDTH + 1
    packed = np.zeros(width, dtype=np.uint64)
    bad = np.zeros(width, dtype=bool)
    for offset in range(SEED_WIDTH):
        column = codes[offset : offset + width]
        packed = (packed << np.uint64(_BITS)) | np.where(column == _INVALID, 0, column).astype(np.uint64)
        bad |= column == _INVALID
    usable = packed[~bad]
    if usable.size == 0:
        return set()
    hits = wanted[np.isin(wanted, usable)]
    return {int(value) for value in hits}


def _coded(window: str) -> np.ndarray:
    """Two-bit codes for a block, with every other letter marked invalid."""
    raw = np.frombuffer(window.encode("ascii", errors="replace"), dtype=np.uint8)
    codes = np.full(raw.shape, _INVALID, dtype=np.uint8)
    for letter, value in (("A", 0), ("C", 1), ("G", 2), ("T", 3)):
        codes[raw == ord(letter)] = value
    return codes
