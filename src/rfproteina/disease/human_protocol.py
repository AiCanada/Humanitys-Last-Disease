"""The H.U.M.A.N  Protocol: name what the design must not bind, then compare.

The order is the point, and it is the reverse of how the rest of this program
reads. Every other screen starts from the target and asks whether the design
reaches it; a design that passes can still reach a dozen other things. This
starts from the proteins a reader names as the ones it must not touch, reports
the design against those, and only then puts the on-target number beside them.
An off-target interaction found *after* the on-target number is good is a
finding somebody will want to argue with.

Two routes, and they answer different questions. Saying which is which is the
whole honesty of this module, because the cheap one -- the one that is
available on this machine today -- does **not** answer the question the tab's
title suggests.

``sequence_resemblance``
    MMseqs2 local alignment of each design against the human reference
    proteome already on disk, read out for the proteins that were chosen. It
    answers: does this design *look like* the protein you named. It does not
    answer whether it binds it. A de novo binder is not a homolog of what it
    binds -- nothing in an alignment score says what a surface will stick to --
    so a clear here is not selectivity. What a resemblance does say is real and
    is its own liability: a design carrying a human protein's own stretches is
    a design whose antibody response can read that protein too.

``interface_contacts``
    The interface terms in :py:mod:`rfproteina.disease.interface_quality`,
    measured on a model of the design *with* the chosen protein and on this
    run's own complex, by the same code in the same units. That is the
    comparison the title asks for: buried area, contacts and shape
    complementarity on-target beside the same numbers off-target. It answers
    the binding question and it needs a model of the off-target complex, which
    this program does not produce -- nothing here docks or folds anything.
    Supplied one, it measures. Without one it declines by name and says what to
    supply.

Two design decisions worth stating, because both could reasonably have gone
the other way:

*The search is run against the whole proteome and then read out for the chosen
proteins*, rather than against a small FASTA of those proteins alone. A tiny
database would be far quicker, and its E-values would be worthless: MMseqs2
computes an E-value against the size of the database it searched, so a number
from a ten-sequence file is not comparable with the one the
``proteome_offtarget_alignment`` axis already records, and the two would sit in
the same screen looking like the same measurement. One pass also serves both
steps -- the proposal below and the comparison -- so the honest arrangement is
also the cheaper one.

*No cut is invented.* A margin is reported and the call is held at
``margin_threshold_not_set`` until a caller names one, for the reason
:py:class:`rfproteina.disease.target_liability.TargetLimits` already gives for
its three axes: there is no published number that turns a selectivity margin
into a liability, and one chosen here would clear designs on this program's
authority.

Nothing is downloaded and nothing is sent anywhere. The proteome FASTA is the
file the download registry already fetched, MMseqs2 is the binary already in
the checkout, and the structure terms come from a module that loads biotite
inside the function that needs it, so a reader who never presses the compare
button pays nothing for it.

Everything here arrives from a browser form -- a name box, a pasted structure,
a path -- so every input is bounded, and bounded *before* the work it would
pay for: the name box is cut before it is split, a structure's size is checked
before it is parsed, and the two expensive halves do not run at all for a route
that is about to decline. The bounds are gathered under one heading below, and
each one is the size past which an input is not the thing it claims to be
rather than an opinion about biology.
"""

from __future__ import annotations

import gzip
import os
import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from rfproteina.disease.offtarget import (
    MMSEQS,
    OFFTARGET_THRESHOLDS,
    OffTargetHit,
    accession_of,
    build_search_command,
    mmseqs_command,
    parse_search_table,
)
from rfproteina.disease.provenance import code_version
from rfproteina.disease.refold_export import REQUEST_CHAIN_LIMIT, REQUEST_RESIDUE_LIMIT
from rfproteina.disease.results_compare import best_match, chain_labels, looks_like_pdb
from rfproteina.disease.screening_tools import CLEAR, HIT, UNTESTED, run_tool, write_fasta

#: The reader's own label for this screen, kept exactly as it was written. The
#: spacing is theirs; HTML collapses it, and a terminal shows it as typed.
PROTOCOL_LABEL = "The H.U.M.A.N  Protocol"

#: What the two routes are called in a record. Closed: a reading that belongs
#: to neither is a reading whose question nobody stated.
SEQUENCE_RESEMBLANCE = "sequence_resemblance"
INTERFACE_CONTACTS = "interface_contacts"
ROUTES = (SEQUENCE_RESEMBLANCE, INTERFACE_CONTACTS)

#: What each route answers, and -- the half that is usually left off -- what it
#: does not. Both ride on every record, so a number cannot be quoted without
#: the sentence saying which question it answered.
ROUTE_ANSWERS = {
    SEQUENCE_RESEMBLANCE: (
        "Whether the design's own sequence resembles the protein named, as a local "
        "alignment against the human reference proteome."
    ),
    INTERFACE_CONTACTS: (
        "How much interface the design makes with the protein named, in a supplied model "
        "of the two together, against the same terms measured on this run's own complex."
    ),
}
ROUTE_DOES_NOT_ANSWER = {
    SEQUENCE_RESEMBLANCE: (
        "Whether the design binds it. A de novo binder is not a homolog of what it binds, "
        "so no alignment score here is a selectivity result."
    ),
    INTERFACE_CONTACTS: (
        "Whether the complex in the supplied model is the one that would form. Nothing here "
        "docks or folds anything; the model is read as given."
    ),
}

#: Where a candidate anti-target came from. A candidate with no source is not
#: offered: the point of the list is that a reader can see why a protein is on
#: it before ticking it.
PROTEOME_ALIGNMENT = "proteome_alignment"
NAMED_BY_OPERATOR = "named_by_operator"
CANDIDATE_SOURCES = (PROTEOME_ALIGNMENT, NAMED_BY_OPERATOR)

#: How many proteins the proposal offers. The list is there to be read before
#: anything is ticked, and a hundred rows is not read.
CANDIDATE_LIMIT = 12
#: How many proteins one comparison will carry. Each is a row in the readout
#: and a lookup in the proteome, and a selection past this is a selection
#: nobody chose deliberately.
SELECTION_LIMIT = 24
#: How much of a FASTA description is kept. A description is a label, not a
#: document, and this one is stored in a record and shown on a page.
_TEXT_LIMIT = 90

# --- bounds on what a reader can hand this -----------------------------------
#
# Everything below is a bound on untrusted input. This screen is reached from a
# browser form, so every field is a reader's text: a name box, a pasted
# structure, a FASTA path. None of these is a policy about biology -- they are
# the sizes past which an input is not the thing it claims to be, and the
# protocol refuses by name rather than doing the work.

#: Characters one name may hold. A UniProt accession is 6 to 10, an entry name
#: and a gene symbol are shorter still.
_NAME_LIMIT = 40
#: Names one box is read for before the rest are left unread. Four times the
#: selection cap on purpose: a reader over the cap still gets the refusal that
#: names it rather than a silently shorter list, while a pasted spreadsheet
#: column cannot turn the box into a million-element list.
_NAME_FLOOD = SELECTION_LIMIT * 4
#: Characters of the whole name box that are read at all, which is the bound
#: that makes the two above cheap: it is applied before the box is split, so
#: nothing downstream ever allocates the rest.
_NAMES_TEXT_LIMIT = _NAME_FLOOD * (_NAME_LIMIT + 1)

#: Designs one pass will align. The results page shows one run's designs and a
#: run is tens; a FASTA with more than this handed to ``--human-protocol`` is a
#: proteome, and aligning one proteome against another is not this question.
DESIGN_LIMIT = 64
#: Residues one design may hold. The same bound ``refold_export`` puts on a
#: request, for the same reason: past it the chain is not one design.
DESIGN_RESIDUE_LIMIT = REQUEST_RESIDUE_LIMIT

#: Bytes of structure text the interface route will parse. ``results_retrieve``
#: accepts 64 MiB for a model it only reads chain labels out of; this one hands
#: the file to a SASA and a dot-surface calculation whose cost is per atom, so
#: the gate is tighter and it comes before the parse. 8 MiB is about 100,000
#: atoms, comfortably above the ``DESIGN_RESIDUE_LIMIT`` complex this measures.
MODEL_BYTE_LIMIT = 8 * 1024 * 1024
#: Chains one model may hold, the same bound ``refold_export`` puts on a
#: request: a design's complex is a handful, and a file with more than this is
#: not one.
MODEL_CHAIN_LIMIT = REQUEST_CHAIN_LIMIT

#: Seconds one alignment pass may take. ``offtarget`` allows an hour because it
#: screens a finished batch in the background; this one is behind a button, and
#: a whole-proteome pass of one design measured 4.6 s on the gzipped human
#: reference. Ten minutes is room for a wide batch on a slow disk, and short
#: enough that a wedged tool does not hold a panel request open for an hour.
SCAN_TIMEOUT_S = 600.0

#: The terms the interface route compares. Every one is measured by
#: ``interface_quality`` on both sides, so a margin is like for like.
INTERFACE_TERMS = (
    "dsasa_total_a2",
    "interface_atom_contacts",
    "interface_polar_contacts",
    "shape_complementarity",
)

#: A UniProt accession. The same pattern ``target_liability`` matches, kept
#: here rather than imported so this module does not pull a network reader in
#: for a regular expression.
_ACCESSION = re.compile(r"^[OPQ][0-9][A-Z0-9]{3}[0-9]$|^[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}$")
#: What separates two names a reader typed into one box.
_SEPARATORS = re.compile(r"[\s,;|]+")
#: Everything a file name may not hold, so a protein a reader named cannot
#: become a path. The same pattern ``ui._UNSAFE_NAME`` uses on an upload.
_UNSAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")
#: The ``KEY=value`` tokens UniProt writes after a FASTA description.
_HEADER_TAG = re.compile(r"\b([A-Z]{2})=(\S+)")

#: One sentence per blocker, in this surface's own voice. The protocol is read
#: on the results page and in a terminal, not on the screening board, so its
#: names are explained here and ``tests/test_disease_board_vocabulary.py``
#: routes them to this table rather than to ``screen_board._BLOCKERS`` -- the
#: board never renders them, and a second copy of a sentence there would be
#: free to drift from this one.
_BLOCKER_TEXT = {
    "anti_targets_not_named": (
        "Nothing was chosen as off-limits, so there is nothing to compare against. "
        "Name at least one protein, or take the proposal above."
    ),
    "too_many_anti_targets": (
        f"More than {SELECTION_LIMIT} proteins were named. Compare a shortlist you have "
        "read rather than a list nobody chose."
    ),
    "anti_target_not_in_proteome": (
        "The human reference proteome holds no entry under this name, so the search could "
        "not have found it and its absence from the hits means nothing. Check the accession."
    ),
    "design_sequence_absent": (
        "No design sequence could be read, so there was nothing to align. The binder chain "
        "is the one the run recorded as the chain of interest."
    ),
    "too_many_designs": (
        f"More than {DESIGN_LIMIT} designs were handed in, which is a library rather than a "
        "run's designs. Align a run's own output, or split the file."
    ),
    "design_over_residue_limit": (
        f"A design is longer than {DESIGN_RESIDUE_LIMIT} residues, so it is not one design's "
        "chain. Pass the designed chain rather than the whole complex."
    ),
    "proteome_not_configured": (
        "No human reference proteome was named, so the resemblance route did not run. Pass "
        "--proteome, set HUMAN_PROTEOME, or stage the tables this program downloads."
    ),
    "proteome_unreadable": (
        "The human reference proteome was named but could not be read, so the resemblance " "route did not run."
    ),
    "mmseqs_not_installed": (
        "MMseqs2 was not found, so no alignment was taken. It ships in this checkout under "
        "'screening subpackage/mmseqs'; MMSEQS names another copy."
    ),
    "search_failed": (
        "The alignment ran and did not finish cleanly, so no resemblance was measured. The "
        "tool's own reason is on the record."
    ),
    "resemblance_threshold_not_set": (
        "The resemblance was measured and not judged: no published identity says when a "
        "design resembles a human protein too closely, so this program will not pick one. "
        "Name a cut to get a verdict."
    ),
    "complex_not_supplied": (
        "No model of the design with this protein was supplied, so the binding question was "
        "not answered. Nothing here docks or folds; fold the pair elsewhere and paste it in."
    ),
    "complex_unreadable": "The supplied model could not be read as a structure, so no interface was measured.",
    "design_chain_not_found": (
        "No chain of the supplied model matches this design by sequence, so there is no "
        "interface of the design to measure. A model of something else looks like this."
    ),
    # Named for the fault and not for the side it was found on, because the same
    # measurement runs on the reader's model and on this run's own complex, and
    # a sentence saying "the supplied model" was wrong on one of the two.
    "no_partner_chain": (
        "This structure holds only the design, so there is no second chain to measure an " "interface against."
    ),
    "model_too_large": (
        f"The structure is larger than {MODEL_BYTE_LIMIT // (1024 * 1024)} MiB, which is not one "
        "design's complex. The interface terms cost per atom, so it was not parsed."
    ),
    "model_holds_too_many_chains": (
        f"The structure names more than {MODEL_CHAIN_LIMIT} chains, so it is not one design's "
        "complex. Pass the design with the one protein it is being measured against."
    ),
    "model_protein_not_chosen": (
        "The model names a protein that was not chosen as off-limits, so the interface would be "
        "measured against something this comparison is not about. Tick it, or name it in the box."
    ),
    "model_pairing_not_named": (
        "The model was supplied without saying which of the chosen proteins it is of, so the "
        "interface cannot be labelled. Name it beside the paste."
    ),
    "on_target_complex_absent": (
        "This run's own complex was not available, so an off-target interface has nothing to "
        "be compared with and only half the comparison could be reported."
    ),
    "interface_unreadable": (
        "The interface terms could not be measured on this structure. The reason the "
        "measuring code gave is on the record."
    ),
    "margin_threshold_not_set": (
        "Both interfaces were measured and the margin is not judged: no published number "
        "says how much more interface on-target than off-target is enough, so this program "
        "will not pick one. Name a cut to get a verdict."
    ),
}

#: What rides on every record that reports a resemblance, so the number cannot
#: be read as the binding result.
RESEMBLANCE_CAVEAT = ROUTE_DOES_NOT_ANSWER[SEQUENCE_RESEMBLANCE]

#: The one blocker a caller on another surface needs to name, exported so it is
#: named through this module and checked against this module's words. The
#: results page reaches it whenever the catalog holds no readable design.
NO_DESIGNS = "design_sequence_absent"

#: The sentence a reader must not be able to skip, shown above the buttons and
#: not under the result: somebody holding a number is past the point of reading
#: what it does not mean. Kept here rather than in the panel so the page and a
#: headless run state the limit in the same words.
PROTOCOL_LIMIT = (
    "Two different questions. The alignment route says whether the design looks like a protein "
    "you named, and it runs on what is already on this machine. Whether it binds that protein is "
    "answered only where you supply a model of the two together -- nothing here docks or folds a "
    "pair -- so a clear on resemblance alone is not selectivity."
)


def sentence(blocker: str) -> str:
    """The words for one blocker, or the name spaced out when it has none."""
    if not blocker:
        return ""
    return _BLOCKER_TEXT.get(blocker, blocker.replace("_", " "))


def blocked_record(blocker: str) -> dict[str, object]:
    """A record for a protocol that could not start at all.

    Here rather than at the caller, and that is not tidiness. A blocker name is
    checked against the vocabulary of the module that *writes* it, so the same
    literal written from the results page is checked against the results page's
    table -- which holds sentence fragments that read on from a colon, in a
    different voice, for a different surface. Emitting it here keeps one
    sentence for it, in the table that renders it.

    It carries the same keys a full record does, so a page that draws one draws
    this without a special case.
    """
    return {
        "protocol": PROTOCOL_LABEL,
        "order": "anti_targets_before_target",
        "target": {"named": False, "accession": ""},
        "anti_targets": [],
        "not_in_proteome": [],
        "routes": [],
        "call": UNTESTED,
        "blocker": blocker,
        "reason": sentence(blocker),
        "code": code_version(),
    }


@dataclass(frozen=True)
class ProteinRecord:
    """One entry of the reference proteome, as the FASTA header names it."""

    accession: str
    entry: str = ""
    gene: str = ""
    description: str = ""
    sequence: str = ""

    def as_record(self) -> dict[str, object]:
        """A plain dict. The sequence is deliberately left out: it is not a finding."""
        return {
            "accession": self.accession,
            "entry": self.entry,
            "gene": self.gene,
            "description": self.description,
            "residues": len(self.sequence),
        }


@dataclass(frozen=True)
class AntiTarget:
    """One protein the design must not bind, and why it is on the list."""

    accession: str
    entry: str = ""
    gene: str = ""
    description: str = ""
    source: str = NAMED_BY_OPERATOR
    #: Filled on a proposal: how much the design already resembles this protein,
    #: which is the reason it is being offered.
    identity: float | None = None
    bits: float | None = None

    @property
    def label(self) -> str:
        """What a reader is shown. The gene symbol when there is one, else the accession."""
        return self.gene or self.entry or self.accession

    def as_record(self) -> dict[str, object]:
        return {
            "accession": self.accession,
            "entry": self.entry,
            "gene": self.gene,
            "description": self.description,
            "source": self.source,
            "identity": self.identity,
            "bits": self.bits,
        }


@dataclass(frozen=True)
class ProtocolLimits:
    """The cuts a caller names. Each is None until then, and an unnamed cut withholds the call.

    Deliberate, and the same choice ``target_liability.TargetLimits`` makes:
    these are real measurements with no published number that turns one into a
    liability, and a cut invented here would clear designs on our authority.
    """

    #: Fractional sequence identity at or above which a design counts as
    #: resembling a protein it must not bind. None until a caller names it. The
    #: identity that would matter is over the residues that actually contact,
    #: and this is identity over the aligned stretch, so a cut chosen here is
    #: answering a slightly narrower question than it appears to.
    resemblance_identity_max: float | None = None
    #: The alignment coverage a resemblance has to reach before its identity is
    #: judged at all, so a nine-residue perfect match in a 200-residue design is
    #: not read as resembling the protein. Suggested, from
    #: ``offtarget.OFFTARGET_THRESHOLDS``, and not published.
    resemblance_coverage_min: float = float(OFFTARGET_THRESHOLDS["coverage_min"]["value"])  # type: ignore[arg-type]
    #: Square angstrom of buried interface the on-target complex has to lead the
    #: off-target one by. None until a caller names it.
    margin_dsasa_min: float | None = None


@dataclass(frozen=True)
class Scan:
    """One MMseqs2 pass of the designs against the reference proteome.

    ``blocker`` is empty on a pass that ran. Everything else on this object is
    only meaningful when it is: a pass that did not run holds no hits, and an
    empty hit list from a pass that never happened would read as "the design
    resembles nothing", which is the one answer it must not give.
    """

    hits: Mapping[str, tuple[OffTargetHit, ...]]
    designs: tuple[str, ...] = ()
    evalue_max: float | None = None
    blocker: str = ""
    #: What the tool itself said, when ``search_failed`` is the blocker.
    tool_blocker: str = ""

    @property
    def ran(self) -> bool:
        return not self.blocker


#: A scan, injected. Supplied by the caller, a test needs neither MMseqs2 nor a
#: proteome; the real one is :func:`scan_designs`.
Search = Callable[[Sequence[tuple[str, str]], Path | None], Scan]


@dataclass(frozen=True)
class InterfaceReading:
    """One interface, measured, or why it was not.

    ``where`` is ``on_target`` or the accession of the protein the design was
    measured against, so a reading cannot be read on the wrong side of the
    comparison.
    """

    where: str
    binder_chain: str = ""
    partner_chains: tuple[str, ...] = ()
    dsasa_total_a2: float | None = None
    interface_atom_contacts: int | None = None
    interface_polar_contacts: int | None = None
    shape_complementarity: float | None = None
    method: str = ""
    blocker: str = ""
    detail: str = ""

    @property
    def measured(self) -> bool:
        return not self.blocker and self.dsasa_total_a2 is not None

    def as_record(self) -> dict[str, object]:
        return {
            "where": self.where,
            "binder_chain": self.binder_chain,
            "partner_chains": list(self.partner_chains),
            "dsasa_total_a2": self.dsasa_total_a2,
            "interface_atom_contacts": self.interface_atom_contacts,
            "interface_polar_contacts": self.interface_polar_contacts,
            "shape_complementarity": self.shape_complementarity,
            "method": self.method,
            "blocker": self.blocker or None,
            "reason": sentence(self.blocker),
            "detail": self.detail,
        }


def parse_names(text: str) -> tuple[str, ...]:
    """The protein names a reader typed, de-duplicated with the order kept.

    Accessions, entry names and gene symbols all pass through: which of the
    three a name is decides how it is matched against the proteome, not whether
    it is accepted. Order is kept because it is the reader's own ranking of
    what matters, and the readout is drawn in it.

    This is the only door untrusted names come through, so it is where they are
    bounded. The whole box is cut to :data:`_NAMES_TEXT_LIMIT` *before* it is
    split, which is what keeps the two bounds below cheap: a pasted spreadsheet
    column is never allocated past that point, rather than being allocated and
    then discarded. An over-long piece keeps its first :data:`_NAME_LIMIT`
    characters instead of being dropped, so it still appears in the readout as a
    name the proteome does not hold -- which is what it is -- rather than
    vanishing between the box and the result.
    """
    found: list[str] = []
    for piece in _SEPARATORS.split((text or "")[:_NAMES_TEXT_LIMIT]):
        name = piece[:_NAME_LIMIT].strip().upper()
        if not name or name in found:
            continue
        found.append(name)
        if len(found) >= _NAME_FLOOD:
            # Past the selection cap several times over, so the caller's own
            # refusal is what the reader sees. Stopping here only bounds the
            # list; it does not decide anything.
            break
    return tuple(found)


def parse_many(names: Sequence[str]) -> tuple[str, ...]:
    """The names from several boxes at once, bounded before they are joined.

    The tab posts one value per tick box plus whatever was typed, so this
    arrives as a list whose length and whose elements are both a reader's --
    and a hand-made post's. Each is cut before the join, and the list itself is
    cut too: joining first and bounding after would allocate the whole of a
    hostile submission to throw it away, which is the bound not doing its job.
    """
    return parse_names(" ".join(str(name)[:_NAMES_TEXT_LIMIT] for name in list(names)[:_NAME_FLOOD]))


def is_accession(name: str) -> bool:
    """Whether a name is a UniProt accession rather than a symbol or entry name."""
    return _ACCESSION.fullmatch(name.strip().upper().split("-", 1)[0]) is not None


def _header_fields(header: str) -> tuple[str, str, str, str]:
    """``(accession, entry, gene, description)`` from one FASTA header line.

    UniProt writes ``sp|P69905|HBA_HUMAN Hemoglobin subunit alpha OS=Homo
    sapiens OX=9606 GN=HBA1 PE=1 SV=2``. The accession and entry name come out
    of the pipe-delimited id, the gene symbol out of ``GN=``, and the
    description is what sits between the two. A header in any other shape
    yields what it can and leaves the rest empty rather than guessing: an entry
    with no accession is simply not matchable by one.
    """
    body = header[1:].strip() if header.startswith(">") else header.strip()
    if not body:
        return "", "", "", ""
    identifier, _, rest = body.partition(" ")
    parts = identifier.split("|")
    if len(parts) >= 3:
        accession, entry = parts[1].upper(), parts[2]
    else:
        accession, entry = accession_of(identifier).upper(), identifier
    tags = dict(_HEADER_TAG.findall(rest))
    cut = len(rest)
    found = _HEADER_TAG.search(rest)
    if found is not None:
        cut = found.start()
    description = " ".join(rest[:cut].split())
    if len(description) > _TEXT_LIMIT:
        description = description[: _TEXT_LIMIT - 3].rstrip() + "..."
    return accession, entry, tags.get("GN", ""), description


def _proteome_entries(proteome: Path) -> Iterator[ProteinRecord]:
    """Every entry of the FASTA, one at a time, gzipped or plain.

    A generator and not a dict. The human reference is 20,652 records and about
    11 MB of residues unpacked, and this module reads it to answer a question
    about a handful of named proteins: holding the whole file so the next
    question is quicker would keep 11 MB alive in a panel process for the rest
    of the session. ``liability.py`` does cache its parse, and should -- it
    searches every design against all of it.

    The whole pass costs 0.3 s on the gzipped human reference, measured against
    that file, which is why the residues of an entry nobody asked about are
    joined and dropped rather than skipped on the header: making the fill
    conditional would leave ``ProteinRecord.sequence`` empty for some entries
    and filled for others, and the saving is a third of a second.
    """
    opener = gzip.open if proteome.suffix == ".gz" else open
    with opener(proteome, "rt", encoding="utf-8", errors="replace") as handle:  # type: ignore[operator]
        header = ""
        chunks: list[str] = []
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith(";"):
                continue
            if line.startswith(">"):
                if header:
                    accession, entry, gene, description = _header_fields(header)
                    yield ProteinRecord(accession, entry, gene, description, "".join(chunks))
                header = line
                chunks = []
            elif header:
                chunks.append("".join(line.split()).upper())
        if header:
            accession, entry, gene, description = _header_fields(header)
            yield ProteinRecord(accession, entry, gene, description, "".join(chunks))


def proteome_records(names: Sequence[str], proteome: Path | None) -> tuple[dict[str, ProteinRecord], tuple[str, ...]]:
    """The reference entries for the names chosen, and the names it does not hold.

    Keyed by the name the reader typed, not by accession, so the readout can say
    which of their own words found nothing. One pass over the file, matching on
    accession, entry name or gene symbol.

    A missing or unreadable file returns no records and every name as not found.
    That is handled by the caller as a blocker on the whole route: an empty
    result here must never be read as "none of these proteins exists".
    """
    wanted = {name.strip().upper() for name in names if name.strip()}
    if not wanted or proteome is None or not Path(proteome).is_file():
        return {}, tuple(names)
    found: dict[str, ProteinRecord] = {}
    try:
        for entry in _proteome_entries(Path(proteome)):
            for key in (entry.accession, entry.entry.upper(), entry.gene.upper()):
                if key and key in wanted and key not in found:
                    found[key] = entry
            if len(found) == len(wanted):
                break
    except (OSError, UnicodeError, EOFError, gzip.BadGzipFile, MemoryError):
        return {}, tuple(names)
    missing = tuple(name for name in names if name.strip().upper() not in found)
    return found, missing


def anti_targets_from(
    names: Sequence[str],
    records: Mapping[str, ProteinRecord],
) -> tuple[AntiTarget, ...]:
    """What the reader chose, filled in from the proteome where it was found.

    A name the proteome does not hold still becomes an anti-target, carrying
    only itself. Dropping it would leave the comparison silently shorter than
    the list the reader ticked.

    One protein named twice is one protein. ``parse_names`` de-duplicates what
    was typed, which cannot see that ``ALB`` and ``P02768`` are the same entry:
    ticking the box and typing the accession drew two identical rows, and the
    headline sentence counted "2 proteins chosen as off-limits" for one.
    """
    chosen: list[AntiTarget] = []
    seen: set[str] = set()
    for name in names:
        key = name.strip().upper()
        if not key:
            continue
        entry = records.get(key)
        accession = (entry.accession or key) if entry is not None else key
        if accession in seen:
            continue
        seen.add(accession)
        if entry is None:
            chosen.append(AntiTarget(accession=accession))
            continue
        chosen.append(
            AntiTarget(
                accession=accession,
                entry=entry.entry,
                gene=entry.gene,
                description=entry.description,
            )
        )
    return tuple(chosen)


def resolve(name: str, proteome: Path | None) -> str:
    """The accession the proteome holds for a typed name, or the name itself.

    The proposal keeps the intended target off the list, and a reader may have
    named that target by gene symbol. Matched as typed, ``TP53`` did not equal
    ``P04637`` and the protein the design was aimed at led the list of things
    to avoid -- which is noise, and noise at the top.
    """
    if not name.strip():
        return ""
    found, _missing = proteome_records((name,), proteome)
    for entry in found.values():
        return entry.accession or name.strip().upper()
    return name.strip().upper()


def default_proteome() -> Path | None:
    """The reference proteome this machine has, or None.

    ``HUMAN_PROTEOME`` wins, then the file the download registry fetches for
    the off-target axis. The results page has no ``args`` to read a
    ``--proteome`` flag off -- it runs after the generation call returned -- so
    without this the protocol would report the proteome unconfigured on a
    machine with the file sitting in its data directory, which is the same trap
    ``tables.downloaded_proteome`` was written to close for the 9-mer axis.
    """
    from rfproteina.disease.tables import downloaded_proteome

    named = os.environ.get("HUMAN_PROTEOME", "").strip()
    if named:
        return Path(named)
    return downloaded_proteome()


def proteome_blocker(proteome: Path | None) -> str | None:
    """Why the reference proteome cannot be read, or None when it can.

    Asked in two places, which is why it is a function. The resemblance route
    needs this file twice over: once for the alignment, and once to establish
    that an entry a reader named is actually in the searched set. A row saying
    a design resembles nothing is only a measurement if the thing it was
    searched against was there.

    Optional rather than an empty string for a blocker that is absent, which is
    the shape every blocker slot in this package has -- and, less obviously, the
    signal ``tests/test_disease_board_vocabulary.py`` follows to find the two
    names below and demand that this module have words for them.
    """
    if proteome is None:
        return "proteome_not_configured"
    if not Path(proteome).is_file():
        return "proteome_unreadable"
    return None


def scan_designs(
    designs: Sequence[tuple[str, str]],
    proteome: Path | None,
    work_dir: Path,
    *,
    mmseqs: str = MMSEQS,
    evalue_max: float | None = None,
    timeout_s: float = SCAN_TIMEOUT_S,
) -> Scan:
    """Align every design against the whole reference proteome, once.

    One pass serves both steps: the proposal reads its strongest hits and the
    comparison reads the rows for the proteins that were chosen. The E-values
    are therefore proteome-scale and comparable with the
    ``proteome_offtarget_alignment`` axis, which is the reason the search is not
    pointed at a small FASTA of the chosen proteins instead.

    The designs are bounded before anything is written: what reaches here from
    ``--human-protocol`` is a file a reader named, and a proteome handed in as
    "the designs" would start an all-against-all alignment behind a button.

    Never raises. Every way this can fail -- no sequence, too many, too long, no
    proteome, no MMseqs2, a tool that exited badly -- comes back as a
    :class:`Scan` carrying the blocker, because a failed search must not be able
    to read as a design that resembles nothing.
    """
    usable = [(name, sequence) for name, sequence in designs if sequence and sequence.strip()]
    if not usable:
        return Scan({}, blocker="design_sequence_absent")
    if len(usable) > DESIGN_LIMIT:
        return Scan({}, blocker="too_many_designs")
    if any(len("".join(sequence.split())) > DESIGN_RESIDUE_LIMIT for _name, sequence in usable):
        return Scan({}, blocker="design_over_residue_limit")
    absent = proteome_blocker(proteome)
    if absent is not None:
        return Scan({}, blocker=absent)
    resolved = mmseqs_command() if mmseqs == MMSEQS else mmseqs
    if resolved is None:
        return Scan({}, blocker="mmseqs_not_installed")
    cut = float(OFFTARGET_THRESHOLDS["evalue_max"]["value"]) if evalue_max is None else float(evalue_max)  # type: ignore[arg-type]
    names = tuple(name for name, _sequence in usable)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    query = work_dir / "protocol_designs.fasta"
    if not write_fasta(usable, query):
        return Scan({}, designs=names, blocker="design_sequence_absent")
    result_out = work_dir / "protocol_hits.m8"
    tmp_dir = work_dir / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    command = build_search_command(query, Path(proteome), result_out, tmp_dir, mmseqs=resolved, evalue_max=cut)
    outcome = run_tool(command, timeout_s=timeout_s)
    if not outcome.ok:
        return Scan(
            {},
            designs=names,
            evalue_max=cut,
            blocker="search_failed",
            tool_blocker=_tool_reason(outcome.blocker),
        )
    text = result_out.read_text(encoding="utf-8") if result_out.is_file() else outcome.stdout
    hits = {query_id: tuple(rows) for query_id, rows in parse_search_table(text).items()}
    return Scan(hits, designs=names, evalue_max=cut)


def _tool_reason(blocker: str | None) -> str:
    """What the tool said, with the executable reduced to its own name.

    ``run_tool`` builds its blocker out of ``command[0]``, which here is an
    absolute path to the binary inside this checkout:
    ``tool_error:/<checkout>/screening subpackage/mmseqs/mmseqs:1``. The record
    this ends up on is held in a panel session and can be saved, and
    ``AGENTS.md`` asks that a path identifying this machine not be among the
    things that leave it. The kind and the exit code are the whole of what a
    reader needs anyway.

    The shape is ``kind:exe`` or ``kind:exe:code``, so the kind comes off the
    front and a numeric code off the back before the middle is reduced to a
    name. Splitting on every colon instead would leave a Windows drive letter
    behind as a field of its own.
    """
    if not blocker:
        return ""
    kind, _, rest = blocker.partition(":")
    if not rest:
        return kind
    code = ""
    head, found, tail = rest.rpartition(":")
    if found and tail.isdigit():
        rest, code = head, tail
    name = rest.replace("\\", "/").rsplit("/", 1)[-1]
    return f"{kind}:{name}:{code}" if code else f"{kind}:{name}"


def _best_per_accession(scan: Scan) -> dict[str, tuple[str, OffTargetHit]]:
    """The strongest alignment to each human protein, over every design.

    Keyed by accession and not by the FASTA id, because that is what a reader
    types and what the proteome lookup matches. The design that produced it is
    kept beside it: a readout naming an identity without saying which design
    reached it is a readout a reader cannot act on.
    """
    best: dict[str, tuple[str, OffTargetHit]] = {}
    for design, rows in scan.hits.items():
        for hit in rows:
            accession = accession_of(hit.target).upper()
            held = best.get(accession)
            if held is None or hit.bits > held[1].bits:
                best[accession] = (design, hit)
    return best


def propose_anti_targets(
    scan: Scan,
    *,
    exclude: Sequence[str] = (),
    limit: int = CANDIDATE_LIMIT,
) -> tuple[AntiTarget, ...]:
    """The human proteins these designs already resemble most, strongest first.

    This is the list a reader picks from. It is derived rather than curated: no
    panel of proteins is asserted here, because a panel would be this program's
    guess at what matters for somebody else's therapeutic. What the alignment
    can say is which human proteins *this* design already looks like, and that
    is a reason to put a protein in front of a reader.

    ``exclude`` drops the design's own intended target, which would otherwise
    lead the list on any design aimed at a human protein.
    """
    skip = {accession_of(name).upper() for name in exclude if name.strip()}
    skip |= {name.strip().upper() for name in exclude if name.strip()}
    ranked = sorted(
        ((accession, pair) for accession, pair in _best_per_accession(scan).items() if accession not in skip),
        key=lambda item: item[1][1].bits,
        reverse=True,
    )
    # ``entry`` is deliberately left empty. MMseqs2 reports a UniProt-style
    # header as the bare accession, so there is no entry name in the hit to put
    # there, and copying the accession into a field named for something else
    # makes the record say it knows a name it never read. :func:`enrich` fills
    # it, and the gene symbol, from the proteome.
    return tuple(
        AntiTarget(
            accession=accession,
            source=PROTEOME_ALIGNMENT,
            identity=round(hit.fident, 4),
            bits=hit.bits,
        )
        for accession, (_design, hit) in ranked[: max(0, limit)]
    )


def enrich(candidates: Sequence[AntiTarget], proteome: Path | None) -> tuple[AntiTarget, ...]:
    """Fill in gene symbol, entry name and description from the proteome.

    Separate from :func:`propose_anti_targets` so that proposing stays a pure
    read of the alignment -- no file, no cost, trivially testable -- while the
    list a reader actually ticks still says ``HBA1`` rather than ``P69905``. A
    candidate the proteome does not name comes back unchanged rather than
    dropped: it was proposed because the alignment found it, and that stands
    whether or not a symbol could be attached to it.
    """
    if not candidates:
        return ()
    found, _missing = proteome_records([item.accession for item in candidates], proteome)
    filled: list[AntiTarget] = []
    for item in candidates:
        entry = found.get(item.accession)
        if entry is None:
            filled.append(item)
            continue
        filled.append(
            AntiTarget(
                accession=item.accession,
                entry=entry.entry,
                gene=entry.gene,
                description=entry.description,
                source=item.source,
                identity=item.identity,
                bits=item.bits,
            )
        )
    return tuple(filled)


def resemblance_report(
    scan: Scan,
    anti_targets: Sequence[AntiTarget],
    records: Mapping[str, ProteinRecord],
    *,
    limits: ProtocolLimits | None = None,
) -> dict[str, object]:
    """One row per chosen protein: how much the design looks like it.

    A protein the proteome does not hold gets ``anti_target_not_in_proteome``
    rather than a clear. No row could exist for an entry the searched file does
    not contain, so its absence from the hits is not evidence of anything --
    and that is exactly the case where a clear would be most confidently wrong.
    """
    bar = ProtocolLimits() if limits is None else limits
    report: dict[str, object] = {
        "route": SEQUENCE_RESEMBLANCE,
        "answers": ROUTE_ANSWERS[SEQUENCE_RESEMBLANCE],
        "does_not_answer": ROUTE_DOES_NOT_ANSWER[SEQUENCE_RESEMBLANCE],
        "designs": list(scan.designs),
        "thresholds": {
            "resemblance_identity_max": bar.resemblance_identity_max,
            "resemblance_coverage_min": bar.resemblance_coverage_min,
            "evalue_max": scan.evalue_max,
            "coverage_source": OFFTARGET_THRESHOLDS["coverage_min"]["source"],
        },
        "readings": [],
        "blocker": None,
        "reason": "",
    }
    if not scan.ran:
        report["blocker"] = scan.blocker
        report["reason"] = sentence(scan.blocker)
        if scan.tool_blocker:
            report["tool_blocker"] = scan.tool_blocker
        return report
    if not anti_targets:
        report["blocker"] = "anti_targets_not_named"
        report["reason"] = sentence("anti_targets_not_named")
        return report
    if len(anti_targets) > SELECTION_LIMIT:
        report["blocker"] = "too_many_anti_targets"
        report["reason"] = sentence("too_many_anti_targets")
        return report
    best = _best_per_accession(scan)
    held = {entry.accession for entry in records.values()}
    readings: list[dict[str, object]] = []
    for chosen in anti_targets:
        readings.append(_resemblance_reading(chosen, best, held, bar))
    report["readings"] = readings
    return report


def _resemblance_reading(
    chosen: AntiTarget,
    best: Mapping[str, tuple[str, OffTargetHit]],
    held: set[str],
    bar: ProtocolLimits,
) -> dict[str, object]:
    """One chosen protein's row, state included."""
    row: dict[str, object] = {
        "accession": chosen.accession,
        "label": chosen.label,
        "gene": chosen.gene,
        "description": chosen.description,
        "state": UNTESTED,
        "alignment": False,
        "design": "",
        "identity": None,
        "coverage": None,
        "evalue": None,
        "bits": None,
        "blocker": None,
        "reason": "",
    }
    if chosen.accession not in held:
        row["blocker"] = "anti_target_not_in_proteome"
        row["reason"] = sentence("anti_target_not_in_proteome")
        return row
    found = best.get(chosen.accession)
    if found is None:
        # The search covered this entry and aligned nothing to it over the
        # E-value cut. That is a measurement, and it is a clear for the question
        # this route asks -- not for the binding one, which the route's own
        # ``does_not_answer`` says in the same record.
        row["state"] = CLEAR
        return row
    design, hit = found
    row.update(
        {
            "alignment": True,
            "design": design,
            "identity": round(hit.fident, 4),
            "coverage": round(hit.coverage, 4),
            "evalue": hit.evalue,
            "bits": hit.bits,
        }
    )
    if bar.resemblance_identity_max is None:
        row["blocker"] = "resemblance_threshold_not_set"
        row["reason"] = sentence("resemblance_threshold_not_set")
        return row
    close = hit.fident >= bar.resemblance_identity_max and hit.coverage >= bar.resemblance_coverage_min
    row["state"] = HIT if close else CLEAR
    return row


def _structure_suffix(text: str) -> str:
    """``.pdb`` or ``.cif``, decided the way the results page already decides it."""
    return ".pdb" if looks_like_pdb(text) else ".cif"


def _detail(error: BaseException) -> str:
    """What went wrong, named by its type, with no path and no essay.

    Two things are kept out deliberately. An ``OSError``'s text ends in the
    file it was about, and this detail is stored in a record a reader can
    download -- a path belongs in a log, not in something that leaves the
    machine, which is what ``AGENTS.md`` asks. And an arbitrary exception's
    text has no length: a numpy failure can carry an array's whole repr, and
    this is rendered into a table cell.
    """
    text = (error.strerror or "") if isinstance(error, OSError) else " ".join(str(error).split())
    said = f"{type(error).__name__}: {text}".strip().rstrip(":")
    if len(said) > _TEXT_LIMIT:
        return said[: _TEXT_LIMIT - 3].rstrip() + "..."
    return said


def _work_file(work_dir: Path, where: str, text: str) -> Path:
    """Where one structure is written, under a name that cannot leave ``work_dir``.

    ``where`` reaches this from a browser field -- it is whichever protein the
    reader said their model is of -- and it was being interpolated straight into
    a path. Every separator is replaced the way ``ui._UNSAFE_NAME`` already
    replaces them for an uploaded file, and the fixed ``protocol_`` prefix means
    even a name of ``..`` cannot name the parent.
    """
    stem = _UNSAFE_NAME.sub("_", where)[:_NAME_LIMIT].strip("._") or "structure"
    return work_dir / f"protocol_{stem}{_structure_suffix(text)}"


def measure_one(text: str, binder_chain: str, work_dir: Path, *, where: str) -> InterfaceReading:
    """The interface terms for one structure's named chain against the rest.

    The measuring code takes a path and this surface holds bytes, so the text is
    written into the work directory first. ``interface_quality`` is imported
    here rather than at the top of the module: it is the only thing in this
    file that reaches for biotite, and a reader who never presses compare
    should not pay for loading it.

    Both size gates come before the parse, because the cost of these terms is
    per atom: a SASA and a dot surface over a pasted 60 MiB file would allocate
    gigabytes inside a panel request. This function also runs on this run's own
    complex, so the gates are not only about what a reader pastes.

    Never raises, which on this surface is a requirement rather than a courtesy:
    the run has already finished and a button must not be able to end it.
    """
    from rfproteina.disease.interface_quality import measure_interface

    # Size before emptiness, because ``strip`` copies the string and this one
    # may be megabytes: testing it first would allocate the whole of a file
    # that is about to be refused for being too big.
    if len(text) > MODEL_BYTE_LIMIT:
        return InterfaceReading(where, blocker="model_too_large")
    if not text.strip():
        return InterfaceReading(where, blocker="complex_not_supplied")
    chains = chain_labels(text)
    if not chains:
        return InterfaceReading(where, blocker="complex_unreadable")
    if len(chains) > MODEL_CHAIN_LIMIT:
        return InterfaceReading(where, blocker="model_holds_too_many_chains")
    if binder_chain not in chains:
        return InterfaceReading(where, binder_chain=binder_chain, blocker="design_chain_not_found")
    partners = tuple(name for name in chains if name != binder_chain)
    if not partners:
        return InterfaceReading(where, binder_chain=binder_chain, blocker="no_partner_chain")
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    path = _work_file(work_dir, where, text)
    try:
        path.write_text(text, encoding="utf-8")
    except OSError as error:
        return InterfaceReading(where, binder_chain=binder_chain, blocker="complex_unreadable", detail=_detail(error))
    try:
        measured = measure_interface(path, binder_chain, partners)
    except Exception as error:
        # Broad on purpose, and the same judgement ``interface_quality._structure``
        # makes one layer down: every way a structure can defeat the measurement
        # is the same answer to a reader, and the alternatives here are worse
        # than a wide catch. A MemoryError or a numpy failure from a file that
        # cleared both gates is not an OSError or a ValueError, so it would pass
        # ``_results_action``'s catch and take the panel down with it -- ending
        # a finished run from a button, which this surface promises it cannot do.
        return InterfaceReading(
            where,
            binder_chain=binder_chain,
            partner_chains=partners,
            blocker="interface_unreadable",
            detail=_detail(error),
        )
    return InterfaceReading(
        where=where,
        binder_chain=measured.binder_chain,
        partner_chains=tuple(measured.target_chains),
        dsasa_total_a2=round(measured.dsasa_total_a2, 1),
        interface_atom_contacts=measured.interface_atom_contacts,
        interface_polar_contacts=measured.interface_polar_contacts,
        shape_complementarity=measured.shape_complementarity,
        method=measured.method,
    )


def design_chain_in(model_text: str, complex_text: str, binder_chain: str) -> str:
    """Which chain of a supplied model is this design, decided by sequence.

    The reader's model of the design with an off-target names its chains
    however its own tool did, so the design cannot be identified by label. It is
    found the way the results page already identifies a retrieved model's
    chains: the longest shared run of residues against the chain this run
    recorded as the design. Returns an empty string when nothing matches.
    """
    if not model_text.strip() or not complex_text.strip() or not binder_chain:
        return ""
    candidates = chain_labels(model_text)
    if not candidates:
        return ""
    found = best_match(complex_text, binder_chain, model_text, candidates)
    return found.right_chain or ""


def interface_report(
    on_target: InterfaceReading,
    off_target: Sequence[InterfaceReading],
    *,
    limits: ProtocolLimits | None = None,
) -> dict[str, object]:
    """The on-target interface beside each off-target one, and the margins.

    A margin is only reported where both sides were measured, by the same code
    in the same units. Comparing a measured interface against a blocked one, or
    an identity against an area, is how a comparison starts meaning nothing.

    The two sides are checked in the order they fail, which is what keeps the
    reason right. Reading the on-target side first reported
    ``on_target_complex_absent`` whenever the reader's own model was the thing
    that could not be measured -- a true sentence about the wrong half of the
    comparison, which sends somebody to look at a file that is fine.
    """
    bar = ProtocolLimits() if limits is None else limits
    report: dict[str, object] = {
        "route": INTERFACE_CONTACTS,
        "answers": ROUTE_ANSWERS[INTERFACE_CONTACTS],
        "does_not_answer": ROUTE_DOES_NOT_ANSWER[INTERFACE_CONTACTS],
        "terms": list(INTERFACE_TERMS),
        "thresholds": {"margin_dsasa_min": bar.margin_dsasa_min},
        "on_target": on_target.as_record(),
        "readings": [reading.as_record() for reading in off_target],
        "margins": [],
        "blocker": None,
        "reason": "",
    }
    if not off_target:
        report["blocker"] = "complex_not_supplied"
        report["reason"] = sentence("complex_not_supplied")
        return report
    if not any(reading.measured for reading in off_target):
        # The reader's side is what failed, so its own reason is the route's.
        # The rows still go out: each one names which protein it was about and
        # why there is no number beside it.
        report["margins"] = [_margin(on_target, reading, bar) for reading in off_target]
        failed = next((reading.blocker for reading in off_target if reading.blocker), "")
        report["blocker"] = failed or None
        report["reason"] = sentence(failed)
        return report
    if not on_target.measured:
        # The off-target side measured and this one did not, so there is a
        # number with nothing to compare it against. No margin is reported at
        # all: half a comparison presented as a margin is the one shape of this
        # record that could be read as a result.
        report["blocker"] = on_target.blocker or "on_target_complex_absent"
        report["reason"] = sentence(str(report["blocker"]))
        return report
    margins = [_margin(on_target, reading, bar) for reading in off_target]
    report["margins"] = margins
    undecided = [row for row in margins if row["state"] == UNTESTED and row["blocker"]]
    if undecided:
        report["blocker"] = undecided[0]["blocker"]
        report["reason"] = sentence(str(undecided[0]["blocker"]))
    return report


def _margin(on_target: InterfaceReading, reading: InterfaceReading, bar: ProtocolLimits) -> dict[str, object]:
    """One off-target interface against the on-target one, term by term."""
    row: dict[str, object] = {
        "where": reading.where,
        "state": UNTESTED,
        "blocker": reading.blocker or None,
        "reason": sentence(reading.blocker),
    }
    for term in INTERFACE_TERMS:
        here = getattr(reading, term)
        there = getattr(on_target, term)
        if here is None or there is None:
            row[term] = None
        elif isinstance(here, int) and isinstance(there, int):
            # A lead in contacts is a count, and ``67.0`` reads as a number
            # somebody measured to one decimal place.
            row[term] = there - here
        else:
            row[term] = round(float(there) - float(here), 4)
    if not reading.measured:
        return row
    if bar.margin_dsasa_min is None:
        row["blocker"] = "margin_threshold_not_set"
        row["reason"] = sentence("margin_threshold_not_set")
        return row
    lead = row["dsasa_total_a2"]
    if lead is None:
        row["blocker"] = "interface_unreadable"
        row["reason"] = sentence("interface_unreadable")
        return row
    row["state"] = CLEAR if float(lead) >= bar.margin_dsasa_min else HIT
    row["blocker"] = None
    row["reason"] = ""
    return row


def _off_target_readings(
    model_text: str,
    model_of: str,
    anti_targets: Sequence[AntiTarget],
    *,
    complex_text: str,
    binder_chain: str,
    work_dir: Path,
) -> list[InterfaceReading]:
    """The off-target side of the comparison: nothing, or one reading.

    One, because the tab takes one pasted model. The list is a list so that a
    caller measuring several pairs does not have to reshape the record.

    Three refusals before anything is measured, and each is about labelling
    rather than about the structure. An unnamed pairing cannot be put on a row;
    a pairing naming a protein nobody chose would measure an interface this
    comparison is not about; and a model with no chain matching the design has
    no interface of the design in it.
    """
    if not model_text.strip():
        return []
    named = parse_names(model_of)
    if not named:
        return [InterfaceReading("supplied_model", blocker="model_pairing_not_named")]
    where = named[0]
    chosen = {item.accession for item in anti_targets} | {
        text.upper() for item in anti_targets for text in (item.gene, item.entry) if text
    }
    if where not in chosen:
        return [InterfaceReading(where, blocker="model_protein_not_chosen")]
    chain = design_chain_in(model_text, complex_text, binder_chain)
    if not chain:
        return [InterfaceReading(where, blocker="design_chain_not_found")]
    return [measure_one(model_text, chain, work_dir, where=where)]


def protocol_record(
    designs: Sequence[tuple[str, str]],
    *,
    names: Sequence[str],
    target: str = "",
    proteome: Path | None = None,
    work_dir: Path,
    model_text: str = "",
    model_of: str = "",
    complex_text: str = "",
    binder_chain: str = "",
    limits: ProtocolLimits | None = None,
    search: Search | None = None,
    mmseqs: str = MMSEQS,
) -> dict[str, object]:
    """The whole protocol, in the order it is meant to be read.

    Anti-targets first, then the target, then both routes. The call is
    assembled rather than chosen: a hit on either route is a hit, a clear needs
    every route that reported to have cleared, and anything else is undecided
    carrying the first blocker. In practice that leaves most runs undecided,
    which is the correct answer and the reason this screen exists -- selectivity
    cannot be cleared from a sequence alignment, and the route that could clear
    it needs a model of the off-target complex that nothing here produces.

    ``model_of`` names which chosen protein ``model_text`` is a model of. It is
    not inferred from the selection: taking the first ticked protein would
    label an interface with the name of a protein that is not in the file, and
    a mislabelled number is worse here than an unlabelled one. It also has to
    be one of the proteins that *were* chosen, or the interface is being
    measured against something this comparison is not about.
    """
    bar = ProtocolLimits() if limits is None else limits
    chosen_names = parse_many(names)
    if len(chosen_names) > SELECTION_LIMIT:
        # Checked before the proteome is opened. ``resemblance_report`` refuses
        # the same count, and reaching it the long way would walk a 7.7 MB file
        # for a list that is about to be refused anyway.
        records: dict[str, ProteinRecord] = {}
        missing: tuple[str, ...] = ()
    else:
        records, missing = proteome_records(chosen_names, proteome)
    anti_targets = anti_targets_from(chosen_names, records)
    absent = proteome_blocker(proteome)
    refusable = not anti_targets or len(chosen_names) > SELECTION_LIMIT
    if absent is not None:
        # Overrides even an injected scan, and deliberately. Without the FASTA
        # no row can be told from an entry that was never searched, so every
        # resemblance reading would be unfounded -- including the clears, which
        # are the ones that would be believed.
        scan = Scan({}, designs=tuple(name for name, _sequence in designs), blocker=absent)
    elif refusable:
        # Nothing was chosen, or too much was. ``resemblance_report`` refuses
        # either on its own, and running a whole-proteome alignment first would
        # spend seconds behind a button on a route that is about to decline.
        scan = Scan({}, designs=tuple(name for name, _sequence in designs))
    elif search is not None:
        scan = search(designs, proteome)
    else:
        scan = scan_designs(designs, proteome, Path(work_dir), mmseqs=mmseqs)
    resemblance = resemblance_report(scan, anti_targets, records, limits=bar)

    # The reader's side first. The on-target interface is only measured where
    # there is something to compare it with: it is a SASA and a dot surface over
    # a whole complex, and on every press of a button that cannot answer it
    # would be seconds spent on a number the route then declines to use.
    off_target = _off_target_readings(
        model_text,
        model_of,
        anti_targets,
        complex_text=complex_text,
        binder_chain=binder_chain,
        work_dir=Path(work_dir),
    )
    comparable = any(reading.measured for reading in off_target)
    if comparable and complex_text.strip() and binder_chain:
        on_target = measure_one(complex_text, binder_chain, Path(work_dir), where="on_target")
    else:
        on_target = InterfaceReading("on_target", blocker="on_target_complex_absent")
    interface = interface_report(on_target, off_target, limits=bar)

    routes = [resemblance, interface]
    states = [str(row.get("state")) for row in resemblance["readings"]]  # type: ignore[union-attr]
    states += [str(row.get("state")) for row in interface["margins"]]  # type: ignore[union-attr]
    call, blocker = _call(states, routes)
    return {
        "protocol": PROTOCOL_LABEL,
        "order": "anti_targets_before_target",
        "target": {"named": bool(target.strip()), "accession": target.strip().upper()},
        "anti_targets": [item.as_record() for item in anti_targets],
        "not_in_proteome": list(missing),
        "routes": routes,
        "call": call,
        "blocker": blocker or None,
        "reason": sentence(blocker),
        "code": code_version(),
    }


def _call(states: Sequence[str], routes: Sequence[Mapping[str, object]]) -> tuple[str, str]:
    """The protocol's own call, and the blocker behind it when there is one.

    One hit anywhere is a hit: the question is whether the design touches
    something it must not, and one protein it touches answers that. A clear
    needs every reading to have cleared, because a clear speaks for the whole
    list.
    """
    if HIT in states:
        return HIT, ""
    first = next((str(route["blocker"]) for route in routes if route.get("blocker")), "")
    if states and all(state == CLEAR for state in states) and not first:
        return CLEAR, ""
    return UNTESTED, first


def route_of(record: Mapping[str, object], route: str) -> dict[str, object]:
    """One route's block of a protocol record, or an empty one.

    By name. The order of that list is pinned by a test, so indexing it is not
    wrong today -- but the two entries mean opposite things, one answering the
    binding question and the other explicitly not, and code that reads the
    wrong one gets a plausible record rather than an error. A name costs
    nothing and cannot be off by one.
    """
    for found in record.get("routes") or []:
        if isinstance(found, Mapping) and found.get("route") == route:
            return dict(found)
    return {}


def on_target_measured(record: Mapping[str, object]) -> bool:
    """Whether the interface route measured this run's own complex.

    Asked by the page, which says which design the on-target interface came
    from. That measurement is taken only where there is an off-target one to
    compare it with, so the sentence naming the design has to be able to find
    out whether it happened -- it was being drawn on every comparison, naming a
    design for a number that was never taken.
    """
    reading = route_of(record, INTERFACE_CONTACTS).get("on_target")
    return isinstance(reading, Mapping) and reading.get("dsasa_total_a2") is not None


def summary_line(record: Mapping[str, object]) -> str:
    """One sentence for a page or a terminal: the call, the count, and the caveat."""
    anti = record.get("anti_targets")
    count = len(anti) if isinstance(anti, list) else 0
    call = str(record.get("call", UNTESTED))
    word = {HIT: "a hit", CLEAR: "clear", UNTESTED: "undecided"}.get(call, call)
    noun = "protein" if count == 1 else "proteins"
    said = f"{PROTOCOL_LABEL}: {word} against {count} {noun} chosen as off-limits."
    reason = str(record.get("reason") or "")
    return f"{said} {reason}".strip()


def readout_lines(record: Mapping[str, object]) -> list[str]:
    """The comparison as plain lines, for a terminal and for the saved record.

    The same numbers the page draws. Written here rather than in the panel so a
    headless run reports exactly what a reader sees in the browser.
    """
    lines = [summary_line(record)]
    for route in record.get("routes") or []:
        if not isinstance(route, Mapping):
            continue
        lines.append("")
        lines.append(f"{route.get('route')}: {route.get('answers')}")
        lines.append(f"  does not answer: {route.get('does_not_answer')}")
        blocker = route.get("blocker")
        if blocker:
            lines.append(f"  {blocker}: {sentence(str(blocker))}")
        if str(route.get("route")) == SEQUENCE_RESEMBLANCE:
            # Routed by route, and not for tidiness. Both routes carry a key
            # called ``readings`` and they hold different shapes: run through
            # the resemblance formatter, an interface reading printed
            # "None (complex_not_supplied)" -- a line assembled out of keys it
            # does not have.
            lines.extend("  " + _row_line(row) for row in route.get("readings") or [] if isinstance(row, Mapping))
        else:
            lines.extend(_interface_lines(route))
    return lines


def _interface_lines(route: Mapping[str, object]) -> list[str]:
    """The interface comparison as lines: every term on-target, off-target, lead.

    All three, for the reason the page draws all three: 213 square angstrom of
    lead is a different finding over a 300 square angstrom interface than over
    a 3000 one, and the lead alone cannot tell them apart.
    """
    held = route.get("on_target")
    on_target = held if isinstance(held, Mapping) else {}
    readings = {str(row.get("where")): row for row in (route.get("readings") or []) if isinstance(row, Mapping)}
    terms = [str(name) for name in (route.get("terms") or [])]
    found: list[str] = []
    for margin in route.get("margins") or []:
        if not isinstance(margin, Mapping):
            continue
        where = str(margin.get("where") or "")
        found.append(f"  against {where}: {margin.get('state')}")
        theirs = readings.get(where) or {}
        # Whether a margin was taken, which is not the same question as whether
        # the row carries a reason. ``margin_threshold_not_set`` is a row that
        # measured both sides perfectly well and is withholding only the
        # verdict: skipping its numbers because it has a sentence hid the
        # measurement behind "no cut was named", which is the one thing a
        # screen that measured something must not do.
        if any(margin.get(term) is not None for term in terms):
            for term in terms:
                found.append(
                    f"    {term}: on-target {on_target.get(term)}"
                    f" | off-target {theirs.get(term)} | lead {margin.get(term)}"
                )
        said = str(margin.get("reason") or "")
        if said:
            found.append(f"    {said}")
    return found


def _row_line(row: Mapping[str, object]) -> str:
    """One reading as a line. A blocked reading says so instead of showing numbers."""
    where = row.get("label") or row.get("accession") or row.get("where") or "?"
    if row.get("blocker"):
        return f"{where}: {row.get('state')} ({row.get('blocker')})"
    if row.get("alignment") is False:
        return f"{where}: {row.get('state')}, no alignment over the E-value cut"
    identity = row.get("identity")
    coverage = row.get("coverage")
    return f"{where}: {row.get('state')}, identity {identity}, coverage {coverage}, E {row.get('evalue')}"
