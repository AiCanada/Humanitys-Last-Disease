"""A research trace for every generated binder.

The trace is the reproducible record of one design: the disease that was
typed, the pathway and mechanism that were chosen, the databases that were
asked, the target structure, the model and design parameters, the screen, and
the confidence the metrics actually reported. A value this run does not have
is ``not_recorded``. A step this run did not take is ``not_performed``. Neither
is filled in with a guess. A clear screen is not a cure and is not an
experiment.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from rfproteina.disease.focus import BiologicalFocus, row_role
from rfproteina.disease.provenance import NOT_RECORDED, code_version
from rfproteina.disease.publish import (
    repository_relative,
    scrubbed_json,
    write_text_atomically,
)
from rfproteina.disease.research import SOURCE_LABELS, SOURCE_ORDER, SourceHit
from rfproteina.disease.screen import (
    _PPI_COLUMN,
    _PRE_ORACLE_COLUMN,
    _flag_cell,
    _metric_rows,
    _row_for_cif,
    base_name,
    sample_cifs,
)
from rfproteina.disease.shards import ORIGINAL_EXPERIMENT
from rfproteina.disease.structures import StructureOption

NOT_PERFORMED = "not_performed"
#: A live read that was attempted and errored. Distinct from ``NOT_PERFORMED``,
#: which says the run never asked, and from "consulted", which says an answer
#: came back: a caller that swallows the error leaves empty lists behind, and
#: empty lists are also what a successful read of a disease with no associated
#: target returns.
FETCH_FAILED = "fetch_failed"
#: The reads a caller can report as failed in ``BiologicalFocus.fetch_failures``.
#: One name per service note that would otherwise read "consulted" on a run
#: whose fetch raised.
OPEN_TARGETS_FOCUS = "open_targets_disease_targets"
REACTOME_PATHWAY_LIST = "reactome_pathway_list"
FETCH_KEYS = frozenset({OPEN_TARGETS_FOCUS, REACTOME_PATHWAY_LIST})
#: Shape of a stored trace. Raised when a key is added or changes meaning, for
#: the reason ``screen.SCREEN_VERSION`` gives: a reader can then tell which
#: record it is holding rather than inferring it from which keys happen to be
#: present. It matters more here than it looks, because the keys added at 2 are
#: blockers, and a blocker is absent when it does not apply -- so a trace with
#: no ``screening.sample.interface_blocker`` either came from a writer that
#: never knew the key or from one that did and had nothing to report, and only
#: this number tells those two apart.
#:
#: ``artifact_digest`` does not substitute. It answers whether the trace and
#: the screen still agree, not which fields the trace writer knew to copy: a
#: narrow writer beside a current screen produces a perfectly matching digest
#: with thirteen verdict fields and three reasons missing.
#:
#: 2 adds the four verdict fields that let a stored verdict be checked and
#: dated (``bar_axes``, ``states``, ``screen_version``, ``code``) and the three
#: per-design reasons a measurement is missing (``row_blocker``,
#: ``sequence_blocker``, ``interface_blocker``). Both sets are inside the
#: ``screening`` block and nowhere else, which is what lets
#: :func:`restamp_screening` stamp this number on a record it only partly
#: rewrites. ``tests/test_disease_trace_shape.py`` pins the key sets beside the
#: number, so the next addition cannot leave it behind.
TRACE_VERSION = 2
TRACE_DIR = "research_traces"
#: The screen a trace's screening block is taken from, and the file its digest
#: is over. One spelling, because the two have to name the same file.
SCREEN_ARTIFACT = "screen.json"
#: Where a bench result is written, beside the run, the way ``axis_evidence.json``
#: carries a curated screen call. Nothing in this program writes it.
VALIDATION_FILE = "experimental_validation.json"
CLEAR_NOTE = "A clear screen is a computational result. It is not a cure and it is not experimental validation."
VALIDATION_NOTE = (
    "No experimental assay is attached to this design. A computational screen is not experimental validation."
)
MISSING_NOTE = "A missing value is not_recorded or not_performed. It is not a measured number."
NO_BINDER_NOTE = "No generated binder CIF was written."
UNREADABLE_NOTE = "The generated binder list could not be read."

_SEQUENCE = re.compile(r"[ACDEFGHIKLMNPQRSTVWY]{30,}", re.IGNORECASE)
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
#: A git object name, so a ref file holding anything else is not stored as one.
_CONFIDENCE = re.compile(
    r"plddt|iptm|(?<![A-Za-z])ptm(?![A-Za-z])|ipae|(?<![A-Za-z])pae(?![A-Za-z])|confidence",
    re.IGNORECASE,
)
_SKIP_METRIC = re.compile(r"sequence|path|cif|structure", re.IGNORECASE)
_RELEASE_KEYS = ("hf_ckpt_path", "hf_ckpt_revision", "ckpt_name", "autoencoder_ckpt_path")
#: The verdict fields copied into a trace. The last four were added because
#: this block is a stored verdict and was not a readable one: a ``bar`` with no
#: ``bar_axes`` claims whatever that bar names today rather than what it named
#: when the verdict was taken, and a record with no ``screen_version`` and no
#: ``code`` cannot be dated or attributed at all. That is the same argument
#: ``mcp_api._VERDICT_FIELDS`` makes for returning them, and this record
#: outlives the view: it is copied into the generation archive with the designs.
#:
#: ``refold`` is deliberately not here. It is four kilobytes of the five a full
#: verdict costs, this file is written once per design, and
#: ``mcp_layers._SCREEN_OMITTED`` already holds it back from a tool that walks a
#: tree for exactly that reason. A summary belongs here if anything does.
_SCREEN_FIELDS = (
    "call",
    "bar",
    "bar_axes",
    "untested",
    "hits",
    "ppi_pass",
    "ppi_blocker",
    "pre_oracle_pass",
    "requested_class",
    "states",
    "screen_version",
    "code",
)
#: Verdict fields that are a list of names rather than a scalar. ``bar_axes``
#: is a list of axes or the text ``absent``, so it is read either way.
_SCREEN_LISTS = frozenset({"untested", "hits", "bar_axes"})
#: Verdict fields stored as a mapping of text to text.
_SCREEN_MAPS = frozenset({"states", "code"})
#: One design's measurements, and the three reasons a measurement is missing.
#: The reasons were not copied, so a stored trace carried ``sequence_pass``
#: false beside two nulls and said nothing about why -- next to a
#: ``screen.json`` that named all three. Kept equal to
#: ``mcp_layers._SAMPLE_FIELDS``, which is the same record read by the server,
#: and pinned against it in ``tests/test_disease_trace_shape.py``.
_SAMPLE_FIELDS = (
    "example_id",
    "contact_ms",
    "clash_fraction",
    "gravy",
    "instability_index",
    "sequence_pass",
    "sequence_identity",
    "ca_rmsd",
    "row_blocker",
    "sequence_blocker",
    "interface_blocker",
)
_EMPTY_METRIC = frozenset({"", "nan", "none", "null", "na", "."})
#: How many keys one bench record contributes, counting its status and source.
#: A hand-written file is not a place to put a dataset.
_VALIDATION_KEYS = 12
#: The two notes :func:`_service_notes` writes about where the structure list
#: came from. Whether this run searched live is recorded there and nowhere else,
#: and :func:`_alphafold_status` reads it back, so the names are written once
#: rather than spelled a second time in a comparison that could drift from them.
_LOCAL_STRUCTURES = ("Local structure file", "structure list")
_RCSB_STRUCTURES = ("RCSB PDB", "polymer entity search and deposit records")
#: ``StructureOption.source`` where the AlphaFold prediction record was fetched
#: and matched, in ``research._store_found``. An ``AF-`` entry id is a name.
_ALPHAFOLD_SOURCE = "alphafold"


@dataclass(frozen=True)
class ServiceNote:
    """One service this search did or did not call."""

    name: str
    role: str
    status: str


@dataclass(frozen=True)
class SearchTrace:
    """The lookups this run actually made.

    ``answers`` is ``None`` when the caller did not record the eight target
    databases. An empty tuple means those databases were not asked. A filled
    tuple is each asked identifier and the status each database returned.
    """

    services: tuple[ServiceNote, ...] = ()
    answers: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] | None = None


def search_trace(
    *,
    live_structures: bool,
    live_disease: bool,
    focus: BiologicalFocus | None,
    collected: Mapping[str, Mapping[str, SourceHit]] | None,
) -> SearchTrace:
    """Name the services this search called and the database statuses it holds.

    A local structure file does not consult Open Targets, Reactome, STRING, or
    RCSB. An empty ``collected`` map means the eight target databases were not
    asked. Statuses already on a hit are copied. A database the hit does not
    mention stays ``not_recorded``.
    """
    services = _service_notes(live_structures=live_structures, live_disease=live_disease, focus=focus)
    return SearchTrace(services=tuple(services), answers=_answers(collected))


def write_research_traces(
    output_dir: Path,
    *,
    disease_name: str,
    disease_id: str,
    option: StructureOption,
    binder_kind: str,
    argv: Sequence[str],
    screen: Mapping[str, object] | None,
    disease_query: str = "",
    focus: BiologicalFocus | None = None,
    search: SearchTrace | None = None,
    config: Path | None = None,
) -> list[Path]:
    """Write one scrubbed trace for each generated binder, inside ``output_dir``.

    A run that wrote no ``*_generated.cif`` still gets one trace that says so.
    The files sit in ``research_traces/`` so the generation archive copies them
    with the designs. Paths and secrets are removed. Sequences are left out.
    """
    try:
        destination = output_dir / TRACE_DIR
        destination.mkdir(parents=True, exist_ok=True)
        samples, listing_failed = _samples(output_dir)
    except OSError as exc:
        print(f"Research trace was not saved for {option.entry_id} chain {option.chain}: {exc}")
        return []
    rows = _metric_rows(output_dir)
    spec = _design_spec_text(output_dir)
    spec_digest = _spec_digest(output_dir)
    pins = _release_pins(config) if _uses_release(argv) else {}
    bench = _validation_file(output_dir)
    digest = screen_digest(output_dir)
    written: list[Path] = []
    if not samples:
        record = _record(
            disease_name=disease_name,
            disease_id=disease_id,
            disease_query=disease_query,
            focus=focus,
            option=option,
            binder_kind=binder_kind,
            argv=argv,
            screen=screen,
            search=search,
            pins=pins,
            spec=spec,
            spec_digest=spec_digest,
            bench=bench,
            digest=digest,
            example_id="",
            structure_file="",
            row={},
            kind="run",
            note=UNREADABLE_NOTE if listing_failed else NO_BINDER_NOTE,
        )
        path = _write(destination / "run.json", record, option)
        if path is not None:
            written.append(path)
            print(f"Research trace recorded. {UNREADABLE_NOTE if listing_failed else NO_BINDER_NOTE}")
        return written
    used: set[str] = set()
    for example_id, structure in samples.items():
        file_id = _file_id(example_id, used)
        record = _record(
            disease_name=disease_name,
            disease_id=disease_id,
            disease_query=disease_query,
            focus=focus,
            option=option,
            binder_kind=binder_kind,
            argv=argv,
            screen=screen,
            search=search,
            pins=pins,
            spec=spec,
            spec_digest=spec_digest,
            bench=bench,
            digest=digest,
            example_id=example_id,
            structure_file=_relative(output_dir, structure),
            row=_row_for_cif(rows, structure),
            kind="binder",
            note="",
        )
        path = _write(destination / f"{file_id}.json", record, option, example_id)
        if path is not None:
            written.append(path)
            print(f"Research trace recorded for {example_id}.")
    return written


def screen_digest(output_dir: Path) -> str:
    """A short digest of this run's screen file, or ``not_recorded``.

    A clock would say which file was written later. This says whether the two
    still agree, which is the question a reader holding both actually has, and
    it answers it without a clock: copying preserves it, a stopped or skewed
    machine cannot distort it, and anyone can check it with ``sha256sum``.
    """
    try:
        data = (output_dir / SCREEN_ARTIFACT).read_bytes()
    except OSError:
        return NOT_RECORDED
    return hashlib.sha256(data).hexdigest()[:16]


def restamp_screening(output_dir: Path, screen: Mapping[str, object] | None) -> list[Path]:
    """Refresh the screening block of every trace already written beside this run.

    ``--screen`` rewrites ``screen.json`` in place: a table dropped in
    afterwards, a re-run of the metrics, a newly configured proteome. The
    traces beside it are not rewritten with it, so each one kept the verdict of
    the screen it was born next to while naming that same file as its source --
    a record that cites its evidence and then contradicts it.

    Only the screening block is rebuilt, and the digest with it. Everything
    else in the record -- the disease that was typed, the pathway, the
    databases that were asked, the model, the command -- belongs to the run
    that generated the design and is not knowable from a re-screen, so
    rewriting the whole trace would blank it. A trace this cannot read is left
    exactly as it is: its stale digest is then the evidence that it is stale.
    """
    destination = output_dir / TRACE_DIR
    try:
        paths = sorted(destination.glob("*.json"))
    except OSError:
        return []
    if not paths:
        return []
    rows = _metric_rows(output_dir)
    digest = screen_digest(output_dir)
    written: list[Path] = []
    for path in paths:
        record = _stored_trace(path)
        if record is None:
            # Said out loud rather than passed over. The caller prints how many
            # traces were refreshed, so a trace missing from that count with
            # nothing said about it reads as one that needed no refreshing,
            # when it is in fact the one still carrying the replaced verdict.
            print(f"Research trace was not refreshed for {path.name}: it could not be read as a trace.")
            continue
        name = record.get("example_id")
        example_id = name if isinstance(name, str) and name != NOT_RECORDED else ""
        record["screening"] = _screening(screen, example_id, _traced_row(output_dir, record, rows), digest)
        # The rewritten block is the current shape, so the record says so. A
        # version-1 record carrying a version-2 screening block is the version
        # lying in the worst direction: a reader keying on the number skips
        # fields that are there, and cannot tell a trace this refreshed from
        # one it never touched.
        #
        # Stamping the whole record is honest only while every key this version
        # added lives inside ``screening``, which is true at 2. A later version
        # that changes anything outside this block must stop stamping here.
        record["version"] = TRACE_VERSION
        try:
            write_text_atomically(path, scrubbed_json(record))
        except OSError as exc:
            print(f"Research trace was not refreshed for {path.name}: {exc}")
            continue
        written.append(path)
    return written


def _stored_trace(path: Path) -> dict[str, object] | None:
    """A trace this can refresh, or None when it is not one."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError, RecursionError):
        return None
    if not isinstance(payload, dict) or "screening" not in payload:
        return None
    return payload


def _traced_row(output_dir: Path, record: Mapping[str, object], rows: list[dict[str, str]]) -> dict[str, str]:
    """This binder's metrics row, found through the file the trace already names."""
    stored = record.get("structure_file")
    if not isinstance(stored, str) or not stored or stored == NOT_RECORDED:
        return {}
    return _row_for_cif(rows, output_dir / stored)


def _service_notes(
    *,
    live_structures: bool,
    live_disease: bool,
    focus: BiologicalFocus | None,
) -> list[ServiceNote]:
    """Services the structure search calls, with the status this run earned."""
    services: list[ServiceNote] = []
    # A read the caller attempted and caught. ``cli._biological_focus`` logs a
    # warning and carries on with empty lists, which is the right thing for the
    # search to do and the wrong thing for the chain to call "consulted".
    failed = {name for name in (focus.fetch_failures if focus is not None else ()) if name in FETCH_KEYS}
    if live_disease:
        services.append(ServiceNote("Open Targets", "disease name search", "consulted"))
    else:
        services.append(ServiceNote("Local disease catalog", "disease name", "used"))
        services.append(ServiceNote("Open Targets", "disease name search", NOT_PERFORMED))
    if live_structures:
        focus_status = FETCH_FAILED if OPEN_TARGETS_FOCUS in failed else "consulted"
        services.extend(
            (
                ServiceNote("Open Targets", "disease-associated targets", focus_status),
                ServiceNote(*_RCSB_STRUCTURES, "consulted"),
                ServiceNote("Reactome", "pathway members of the associated proteins", "consulted"),
                ServiceNote("STRING", "interaction partners of the associated proteins", "consulted"),
                ServiceNote("Open Targets", "RNA-biotype genes for the disease", "consulted"),
            )
        )
    else:
        services.append(ServiceNote(*_LOCAL_STRUCTURES, "used"))
        services.extend(
            (
                ServiceNote("Open Targets", "disease-associated targets", NOT_PERFORMED),
                ServiceNote(*_RCSB_STRUCTURES, NOT_PERFORMED),
                ServiceNote("Reactome", "pathway members of the associated proteins", NOT_PERFORMED),
                ServiceNote("STRING", "interaction partners of the associated proteins", NOT_PERFORMED),
                ServiceNote("Open Targets", "RNA-biotype genes for the disease", NOT_PERFORMED),
            )
        )
    page = focus is not None and live_structures
    if page and REACTOME_PATHWAY_LIST in failed:
        page_status = FETCH_FAILED
    elif page and OPEN_TARGETS_FOCUS in failed:
        # This read takes the accessions the Open Targets read returns, so
        # ``cli._biological_focus`` makes it in that read's ``else`` branch and
        # a failed target list means Reactome was never called at all.
        # "consulted" states a consult that did not happen, and FETCH_FAILED
        # would blame Reactome for a call it never received. The run did not ask.
        page_status = NOT_PERFORMED
    else:
        page_status = "consulted" if page else NOT_PERFORMED
    services.append(
        ServiceNote(
            "Reactome",
            "pathway list on the biological target page",
            page_status,
        )
    )
    # ``sources.pathway_uniprot_members`` returns without a request unless the
    # id is a human Reactome stable id, which is the same test made here rather
    # than a looser "not blank and not any".
    selected = page and focus is not None and str(focus.pathway_id).startswith("R-HSA-")
    services.append(
        ServiceNote(
            "Reactome",
            "participants of the selected pathway",
            "consulted" if selected else NOT_PERFORMED,
        )
    )
    # The seeds decide whether this call happens at all.
    # ``sources.partner_symbols`` loops over them, so with none it sends no
    # request and returns an empty list -- and the seeds are the disease
    # accessions and symbols, which are empty whenever the Open Targets read
    # failed. The mechanism alone was taken as proof of a consult, so a run
    # whose target list never arrived still recorded STRING as consulted. True
    # of a successful lookup that simply found no target, too.
    seeded = focus is not None and bool(focus.disease_accessions or focus.disease_symbols)
    partners = page and focus is not None and focus.mechanism == "interaction" and seeded
    services.append(
        ServiceNote(
            "STRING",
            "partners for the interaction mechanism",
            "consulted" if partners else NOT_PERFORMED,
        )
    )
    return services


def _answers(
    collected: Mapping[str, Mapping[str, SourceHit]] | None,
) -> tuple[tuple[str, tuple[tuple[str, str], ...]], ...] | None:
    """Per-target database statuses. ``None`` means the caller did not record them."""
    if collected is None:
        return None
    if not collected:
        return ()
    rows: list[tuple[str, tuple[tuple[str, str], ...]]] = []
    for identifier, hits in collected.items():
        key = identifier.strip().casefold()
        if not key:
            continue
        statuses = tuple((name, _hit_status(hits.get(name))) for name in SOURCE_ORDER)
        rows.append((key, statuses))
    return tuple(rows)


def _hit_status(hit: object) -> str:
    """The status a database hit already carries. A missing hit stays unrecorded."""
    if hit is None:
        return NOT_RECORDED
    status = getattr(hit, "status", None)
    if isinstance(status, str) and status.strip():
        return status.strip()
    return NOT_RECORDED


def _samples(output_dir: Path) -> tuple[dict[str, Path], bool]:
    """Generated binders, and whether the sample directory could not be read."""
    try:
        return sample_cifs(output_dir), False
    except OSError:
        return {}, True


def _record(
    *,
    disease_name: str,
    disease_id: str,
    disease_query: str,
    focus: BiologicalFocus | None,
    option: StructureOption,
    binder_kind: str,
    argv: Sequence[str],
    screen: Mapping[str, object] | None,
    search: SearchTrace | None,
    pins: Mapping[str, str],
    spec: str | None,
    spec_digest: str,
    bench: Mapping[str, object] | None,
    digest: str,
    example_id: str,
    structure_file: str,
    row: Mapping[str, str],
    kind: str,
    note: str,
) -> dict[str, object]:
    """One binder's trail. Every string is plain data; scrubbing happens on write."""
    record: dict[str, object] = {
        "version": TRACE_VERSION,
        "kind": kind,
        "example_id": example_id or NOT_RECORDED,
        "structure_file": structure_file or NOT_RECORDED,
        "reproducibility": MISSING_NOTE,
        "disease_input": _disease_input(disease_query, disease_name, disease_id),
        "pathway": _pathway(focus),
        "source_databases": _databases(option, search),
        "selected_target": _selected_target(option, focus),
        "target_structure": _structure(option),
        "generative_model": _model(argv, row, pins),
        "design_parameters": _design(binder_kind, argv, spec, row, example_id, spec_digest),
        "screening": _screening(screen, example_id, row, digest),
        "model_confidence": _confidence(row),
        "experimental_validation": _validation(bench, example_id),
    }
    if note:
        record["note"] = note
    return record


def _disease_input(query: str, name: str, disease_id: str) -> dict[str, str]:
    """The text that was typed, then the disease that was confirmed."""
    typed = query.strip()
    withheld = bool(typed) and _SEQUENCE.search(typed) is not None
    if not typed or withheld:
        status = NOT_RECORDED
        stored = NOT_RECORDED
    else:
        status = "recorded"
        stored = typed
    recorded: dict[str, str] = {
        "status": status,
        "query": stored,
        "confirmed_name": _public_text(name),
        "disease_id": _public_text(disease_id),
    }
    if withheld:
        recorded["query_withheld"] = "sequence"
    return recorded


def _pathway(focus: BiologicalFocus | None) -> dict[str, object]:
    """The pathway and mechanism chosen on the biological target page.

    The three identifier lists are the proteins those choices actually added to
    the structure search. Without them the choice is a label: a reader can see
    that a pathway was picked and still cannot rebuild the candidate list the
    target was chosen from, which is the whole claim the choice makes. They are
    public UniProt accessions and gene symbols, nothing typed by the user.
    """
    if focus is None:
        return {
            "status": NOT_RECORDED,
            "pathway_id": NOT_RECORDED,
            "pathway_name": NOT_RECORDED,
            "mechanism": NOT_RECORDED,
            "pathway_members": NOT_RECORDED,
            "interaction_partners": NOT_RECORDED,
            "disease_proteins": NOT_RECORDED,
        }
    return {
        "status": "recorded",
        "pathway_id": focus.pathway_id or NOT_RECORDED,
        "pathway_name": focus.pathway_name or NOT_RECORDED,
        "mechanism": focus.mechanism or NOT_RECORDED,
        "pathway_members": _identifiers(focus.pathway_accessions),
        # Partners are fetched for one mechanism only, so an empty list here
        # would read as "STRING was asked and answered nothing".
        "interaction_partners": (
            _identifiers(focus.interaction_symbols) if focus.mechanism == "interaction" else NOT_PERFORMED
        ),
        "disease_proteins": _identifiers(focus.disease_accessions),
    }


def _identifiers(values: Sequence[str]) -> list[str] | str:
    """Public accessions or symbols the search used. An empty set is not recorded."""
    kept = [found for item in values if (found := _pin(item)) != NOT_RECORDED]
    return kept or NOT_RECORDED


def _databases(option: StructureOption, search: SearchTrace | None) -> dict[str, object]:
    """Structure-search services, then the eight target databases for this accession."""
    if search is None:
        services: list[dict[str, str]] = [
            {"name": "structure search", "role": "how the target list was built", "status": NOT_RECORDED}
        ]
        records = [_database_row(name, NOT_RECORDED) for name in SOURCE_ORDER]
    else:
        services = [{"name": note.name, "role": note.role, "status": note.status} for note in search.services]
        records = _target_records(option.accession, search.answers)
    if option.source.strip().casefold() == _ALPHAFOLD_SOURCE or option.entry_id.upper().startswith("AF-"):
        # The row stays, because an AlphaFold-named structure makes the database
        # a question a reader has. Its status is what this run earned.
        services.append(
            {
                "name": "AlphaFold Database",
                "role": "predicted model for this target",
                "status": _alphafold_status(option, search),
            }
        )
    return {"structure_search": services, "target_records": records}


def _alphafold_status(option: StructureOption, search: SearchTrace | None) -> str:
    """Whether this run called the AlphaFold Database for this structure.

    An ``AF-`` entry id is a name, not a call. A structure the user uploaded
    under that name had the chain reporting the database as ``consulted``, and
    so did a run whose whole structure list was read from a file with no
    service called at all -- a source recorded as consulted when nothing
    consulted it, which is the one thing this record exists not to do.

    ``source`` is set to ``alphafold`` only where the prediction record was
    fetched and matched, in ``research._store_found``, so it is evidence of the
    call -- but only in the run that made it, and a saved structure list
    carries it into later runs. So the services this run actually recorded
    decide first: a list read from a file did not perform the call, and a run
    that recorded no search at all cannot say either way.
    """
    if search is None:
        return NOT_RECORDED
    live = _searched_live(search.services)
    if live is None:
        return NOT_RECORDED
    if not live:
        return NOT_PERFORMED
    return "consulted" if option.source.strip().casefold() == _ALPHAFOLD_SOURCE else NOT_PERFORMED


def _searched_live(services: Sequence[ServiceNote]) -> bool | None:
    """Whether the structure list came from a live search, read back from the notes.

    ``None`` when the recorded services name neither, because then nothing
    says which it was.
    """
    for note in services:
        if (note.name, note.role) == _LOCAL_STRUCTURES:
            return False
        if (note.name, note.role) == _RCSB_STRUCTURES:
            return note.status == "consulted"
    return None


def _target_records(
    accession: str,
    answers: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] | None,
) -> list[dict[str, str]]:
    """The eight databases for this accession. An unasked target was not performed."""
    if answers is None:
        return [_database_row(name, NOT_RECORDED) for name in SOURCE_ORDER]
    if not answers:
        return [_database_row(name, NOT_PERFORMED) for name in SOURCE_ORDER]
    key = accession.strip().casefold()
    found = next((statuses for identifier, statuses in answers if identifier == key), None)
    if found is None:
        return [_database_row(name, NOT_PERFORMED) for name in SOURCE_ORDER]
    held = dict(found)
    return [_database_row(name, held.get(name, NOT_RECORDED)) for name in SOURCE_ORDER]


def _database_row(name: str, status: str) -> dict[str, str]:
    return {"name": SOURCE_LABELS[name], "key": name, "role": "target record", "status": status}


def _selected_target(option: StructureOption, focus: BiologicalFocus | None) -> dict[str, str]:
    """The structure that was generated against, and the name the user requested."""
    if focus is None:
        role = NOT_RECORDED
        requested = NOT_RECORDED
        accession = NOT_RECORDED
        symbol = NOT_RECORDED
    else:
        role = row_role(option, focus) or "none"
        requested = _public_text(focus.target)
        accession = _public_text(focus.target_accession) if focus.target_accession else NOT_RECORDED
        symbol = _public_text(focus.target_symbol) if focus.target_symbol else NOT_RECORDED
    return {
        "status": "recorded",
        "accession": _public_text(option.accession),
        "group": option.group,
        "role": role,
        "requested_name": requested,
        "requested_accession": accession,
        "requested_symbol": symbol,
    }


def _structure(option: StructureOption) -> dict[str, object]:
    """The deposit or model, and where it came from. The role is not initiation."""
    return {
        "group": option.group,
        "entry_id": option.entry_id,
        "chain": option.chain,
        "title": _public_text(option.title),
        "predicted": option.predicted,
        "methodology_known": option.methodology_known,
        "source": option.source or NOT_RECORDED,
        "provenance": _provenance(option),
    }


def _provenance(option: StructureOption) -> str:
    """Where this structure came from. A deposit is not labelled as the disease start."""
    source = option.source.strip().casefold()
    entry = option.entry_id
    chain = option.chain
    accession = option.accession
    if source == "rcsb":
        kind = "predicted model" if option.predicted else "deposit"
        return f"RCSB PDB {kind} {entry} chain {chain} for {accession}."
    if source == "alphafold":
        return f"AlphaFold Database model {entry} for {accession}."
    if source == "upload":
        return f"User-uploaded structure {entry} chain {chain}."
    if source == "llm":
        return f"Model-named structure {entry} chain {chain}, source {option.source}."
    named = option.source or NOT_RECORDED
    return f"Structure {entry} chain {chain} from {named}."


def _model(argv: Sequence[str], row: Mapping[str, str], pins: Mapping[str, str]) -> dict[str, object]:
    """The model the command selected, then the checkpoint the metrics named."""
    command = _command_map(argv)
    experiment = command.get("inference_experiment", NOT_RECORDED)
    recorded = {
        "experiment": experiment,
        "hub_repository": _hub(command.get("hf_ckpt_path", "")),
        "hub_revision": _pin(command.get("hf_ckpt_revision", "")),
        "checkpoint_name": _pin(command.get("ckpt_name", "")),
        "checkpoint_path": _stored_path(command.get("ckpt_path", "")),
        "autoencoder_checkpoint_name": NOT_RECORDED,
    }
    for key, field in (
        ("hf_ckpt_path", "hub_repository"),
        ("hf_ckpt_revision", "hub_revision"),
        ("ckpt_name", "checkpoint_name"),
        ("autoencoder_ckpt_path", "autoencoder_checkpoint_name"),
    ):
        if recorded[field] == NOT_RECORDED and pins.get(key):
            recorded[field] = pins[key]
    # Tested before ``code`` is added, and without it. ``code`` is this
    # repository's own commit, recorded for every run and saying nothing about
    # which model ran, and it is a dict: compared against a string it was never
    # equal, so the test was true for every run and ``status`` read ``recorded``
    # on a command that named no experiment, no hub revision, no checkpoint and
    # no autoencoder. A reader checking that field to see whether the record
    # identifies the model was told yes and then found nothing.
    identified = any(value != NOT_RECORDED for value in recorded.values())
    recorded["code"] = code_version()
    metrics = _model_metrics(row)
    known = experiment != NOT_RECORDED or metrics["status"] == "recorded" or identified
    return {"status": "recorded" if known else NOT_RECORDED, "command": recorded, "metrics": metrics}


def _model_metrics(row: Mapping[str, str]) -> dict[str, str]:
    """Checkpoint fields present on this sample's metrics row."""
    found: dict[str, str] = {}
    name = _pin(_metric_text(row.get("ckpt_name", "")))
    if name != NOT_RECORDED:
        found["ckpt_name"] = name
    path = row.get("ckpt_path", "")
    if isinstance(path, str):
        stored = _stored_path(path)
        if stored != NOT_RECORDED:
            found["ckpt_path"] = stored
    if not found:
        return {"status": NOT_RECORDED}
    found["status"] = "recorded"
    return found


def _design(
    binder_kind: str,
    argv: Sequence[str],
    spec: str | None,
    row: Mapping[str, str],
    example_id: str = "",
    spec_digest: str = NOT_RECORDED,
) -> dict[str, object]:
    """Binder length, contig, seed, and the command tokens. No sequence is stored.

    ``seed_scope`` says what the seed names. One job resolves one seed and
    stamps it on every replicate it writes, so an eight-design run carries the
    same number eight times: it reproduces the batch, not this design. Nothing
    records a per-design seed, and ``configs/inference.yaml`` leaves
    ``torch_deterministic`` false, so even rerunning the batch is not bit-exact.
    ``replicate`` is this design's own name within it.
    """
    command = _command_map(argv)
    metrics = _design_metrics(row)
    seed = metrics.get("seed", _metric_text(command.get("seed", "")))
    contig, hotspot = _spec_fields(spec)
    host = command.get("+disease.host", "")
    return {
        "binder_kind": binder_kind or NOT_RECORDED,
        "seed_scope": "run",
        "replicate": _public_text(example_id) if example_id else NOT_RECORDED,
        "contig": contig,
        "hotspot": hotspot,
        # Which spec those two were read out of, so a reader can tell whether
        # the file beside the run is still the one the design was asked for.
        "spec_digest": spec_digest or NOT_RECORDED,
        "seed": seed,
        "host": _public_text(host) if host else NOT_RECORDED,
        "inputs": _command_value(command.get("inputs", "")),
        "command": _command_items(argv),
        "metrics": {**metrics, "status": "recorded"} if metrics else {"status": NOT_RECORDED},
    }


def _spec_fields(spec: str | None) -> tuple[str, str]:
    """Contig text and whether a hotspot was set. A missing spec stays unrecorded."""
    if spec is None:
        return NOT_RECORDED, NOT_RECORDED
    contig = NOT_RECORDED
    hotspot = "not_set"
    for raw in spec.splitlines():
        stripped = raw.strip()
        if stripped.startswith("contig:"):
            contig = _public_text(stripped.split(":", 1)[1].strip().strip("'\""))
        elif stripped.startswith("hotspot:") or stripped.startswith("C_HOT"):
            value = stripped.split(":", 1)[-1].strip().strip("'\"")
            hotspot = _public_text(value) if value else "recorded"
    return contig, hotspot


def _screening(
    screen: Mapping[str, object] | None,
    example_id: str,
    row: Mapping[str, str],
    digest: str = NOT_RECORDED,
) -> dict[str, object]:
    """The screen for this binder. A clear call stays a computational result.

    The keys copied from the screen are the run's verdict. This binder's own
    numbers are under ``sample``, which is the only place a reader should take
    a per-design value from.

    ``artifact_digest`` is over the screen file this block was taken from, so a
    reader can tell whether the two still agree: ``sha256sum screen.json`` and
    compare the first sixteen characters. ``--screen`` rewrites that file in
    place, and a trace written beside an earlier verdict would otherwise cite
    the file while contradicting it, with nothing in either saying so.
    """
    if screen is None:
        return {
            "status": NOT_RECORDED,
            "method": "screen_generation",
            "artifact": NOT_RECORDED,
            "artifact_digest": NOT_RECORDED,
            "note": CLEAR_NOTE,
            "sample": _sample(None, example_id, row),
        }
    fields: dict[str, object] = {
        "status": "recorded",
        "method": "screen_generation",
        "artifact": SCREEN_ARTIFACT,
        "artifact_digest": digest or NOT_RECORDED,
        "note": CLEAR_NOTE,
    }
    for key in _SCREEN_FIELDS:
        if key not in screen:
            fields[key] = NOT_RECORDED
        elif key in _SCREEN_MAPS:
            fields[key] = _string_map(screen.get(key))
        elif key in _SCREEN_LISTS and isinstance(screen.get(key), list):
            fields[key] = _string_list(screen.get(key))
        else:
            fields[key] = _plain(screen.get(key))
    fields["sample"] = _sample(screen, example_id, row)
    return fields


def _sample(screen: Mapping[str, object] | None, example_id: str, row: Mapping[str, str]) -> dict[str, object]:
    """Measurements for this binder. A sequence key on the screen is not copied.

    The two pass flags are read from this binder's own metrics row, never from
    the screen. The screen's ``pre_oracle_pass`` and ``ppi_pass`` are one
    verdict over the whole run -- ``_flag_column`` reports false when any row is
    false -- so copying them here would stamp an eight-replicate run's single
    failure onto all eight traces, and would mark a binder whose own column was
    missing as having passed.
    """
    found: dict[str, object] = {
        "status": NOT_RECORDED,
        "pre_oracle_pass": _row_flag(row, _PRE_ORACLE_COLUMN),
        "ppi_pass": _row_flag(row, _PPI_COLUMN),
    }
    samples = screen.get("samples") if isinstance(screen, Mapping) else None
    if not isinstance(samples, list):
        return found
    for item in samples:
        if not isinstance(item, Mapping) or item.get("example_id") != example_id:
            continue
        found["status"] = "recorded"
        for key in _SAMPLE_FIELDS:
            found[key] = _plain(item.get(key)) if key in item else NOT_RECORDED
        return found
    return found


def _row_flag(row: Mapping[str, str], column: str) -> object:
    """One pass flag from this binder's own metrics row.

    An absent column, or a cell the metrics writer left blank, is not recorded
    rather than false: the difference between "this design failed" and "nothing
    measured this design" is the difference the whole record exists to keep.
    """
    if column not in row:
        return NOT_RECORDED
    flag = _flag_cell(row.get(column))
    return NOT_RECORDED if flag is None else flag


def _validation_file(output_dir: Path) -> Mapping[str, object] | None:
    """A bench record someone wrote beside the run, or None when there is none."""
    path = output_dir / VALIDATION_FILE
    try:
        if not path.is_file():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError, RecursionError):
        return None
    return payload if isinstance(payload, dict) else None


def _validation(bench: Mapping[str, object] | None, example_id: str) -> dict[str, object]:
    """What a person recorded about taking this design to the bench.

    This program runs no assay, so the only honest source is a file a person
    wrote: ``experimental_validation.json`` beside the run, the same shape
    ``axis_evidence.json`` already uses for a curated screen call. A key naming
    one example id is that binder's record; a ``status`` at the top level
    applies to every binder in the run.

    Every value is stored as the text it was written as. Nothing here is
    measured, so nothing here becomes a number, and a record without a status
    is no record: the default says the assay was not performed rather than
    leaving a reader to guess from an empty object.
    """
    default: dict[str, object] = {"status": NOT_PERFORMED, "note": VALIDATION_NOTE}
    if bench is None:
        return default
    entry = bench.get(example_id) if example_id else None
    if not isinstance(entry, Mapping):
        entry = bench if isinstance(bench.get("status"), str) else None
    if not isinstance(entry, Mapping):
        return default
    raw = entry.get("status")
    status = _public_text(raw) if isinstance(raw, str) else NOT_RECORDED
    if status == NOT_RECORDED:
        return default
    found: dict[str, object] = {"status": status, "source": VALIDATION_FILE}
    stored: dict[str, object] = {}
    dropped = 0
    for key in sorted(entry):
        if key == "status":
            continue
        value = entry[key]
        if not isinstance(value, str) or not value.strip():
            # A number, a nested per-binder record, or an empty string. The
            # first is deliberate -- nothing here is measured -- and the other
            # two are not this binder's field, so none of them is a field that
            # was left out of a record it belongs to.
            continue
        name = _public_text(key)
        if name in found or name in stored or len(stored) + len(found) >= _VALIDATION_KEYS:
            # A key whose stored name is already taken, or one past the cap.
            # Both used to vanish without a word. The first was the worse of
            # the two: `` status`` normalises to ``status``, so a second key in
            # the same file replaced the bench status -- the strongest claim
            # this record can carry -- with arbitrary text that was never
            # checked as a status. Two keys that normalise alike, or a name
            # that normalises to ``not_recorded``, lost a field the same way.
            dropped += 1
            continue
        stored[name] = _public_text(value)
    found.update(stored)
    if dropped:
        # Outside the cap, because it is not one of the record's own fields: a
        # record cut at the limit, or one with a colliding key, read as a
        # complete one, and a reader comparing the trace against the file
        # beside the run had nothing saying which it was.
        found["fields_not_stored"] = dropped
    return found


def _confidence(row: Mapping[str, str]) -> dict[str, object]:
    """Numeric confidence columns on this sample. An absent column is not a zero."""
    metrics: dict[str, float] = {}
    for key, raw in row.items():
        if not key or key.startswith("generation.") or _SKIP_METRIC.search(key):
            continue
        if _CONFIDENCE.search(key) is None:
            continue
        number = _finite(raw)
        if number is not None:
            metrics[key] = number
    if not metrics:
        return {"status": NOT_RECORDED, "metrics": {}}
    return {"status": "recorded", "metrics": metrics}


def _design_metrics(row: Mapping[str, str]) -> dict[str, str]:
    """Seed and generation settings present on this sample's metrics row."""
    found: dict[str, str] = {}
    for key, raw in row.items():
        if key != "seed" and not key.startswith("generation."):
            continue
        if _SKIP_METRIC.search(key):
            continue
        stored = _metric_text(raw)
        if stored != NOT_RECORDED:
            found[key] = stored
    return found


def _command_items(argv: Sequence[str]) -> list[dict[str, str]]:
    """Hydra tokens from the command, with paths and sequences removed."""
    items: list[dict[str, str]] = []
    for key, value in _command_pairs(argv):
        items.append({"key": key, "value": _command_value(value)})
    return items


def _command_value(value: str) -> str:
    if not value.strip():
        return NOT_RECORDED
    if _looks_like_path(value):
        return _stored_path(value)
    return _public_text(value)


def _command_map(argv: Sequence[str]) -> dict[str, str]:
    return dict(_command_pairs(argv))


def _command_pairs(argv: Sequence[str]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for part in argv:
        if part.startswith("+") and "=" in part[1:]:
            key, value = part[1:].split("=", 1)
            pairs.append((f"+{key}", value))
            continue
        if "=" not in part or part.startswith("rfd4-generate"):
            continue
        key, value = part.split("=", 1)
        pairs.append((key, value))
    return pairs


def _uses_release(argv: Sequence[str]) -> bool:
    """True when the command kept the original experiment and its config checkpoint."""
    command = _command_map(argv)
    if command.get("inference_experiment") != ORIGINAL_EXPERIMENT:
        return False
    if command.get("hf_ckpt_path") in {"null", "None"}:
        return False
    return "ckpt_name" not in command and "ckpt_path" not in command


def _release_pins(config: Path | None) -> dict[str, str]:
    """Checkpoint names from the inference config. The ``DATA_PATH`` template is not stored."""
    path = config if config is not None else _inference_config()
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return {}
    raw = _yaml_values(text, set(_RELEASE_KEYS))
    pins: dict[str, str] = {}
    hub = _hub(raw.get("hf_ckpt_path", ""))
    revision = _pin(raw.get("hf_ckpt_revision", ""))
    name = _pin(raw.get("ckpt_name", ""))
    autoencoder = _checkpoint_filename(raw.get("autoencoder_ckpt_path", ""))
    if hub != NOT_RECORDED:
        pins["hf_ckpt_path"] = hub
    if revision != NOT_RECORDED:
        pins["hf_ckpt_revision"] = revision
    if name != NOT_RECORDED:
        pins["ckpt_name"] = name
    if autoencoder != NOT_RECORDED:
        pins["autoencoder_ckpt_path"] = autoencoder
    return pins


def _inference_config() -> Path:
    return Path(__file__).resolve().parents[3] / "configs" / "inference.yaml"


def _yaml_values(text: str, keys: set[str]) -> dict[str, str]:
    """Top-level ``key: value`` lines. A later duplicate does not replace the first."""
    found: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        if key not in keys or key in found:
            continue
        cleaned = value.strip().strip("'\"")
        if cleaned:
            found[key] = cleaned
    return found


def _spec_path(output_dir: Path) -> Path | None:
    """The condition spec beside the run, or the copy the archive stored inside it."""
    names = (output_dir / f"{base_name(output_dir)}.yaml", output_dir.parent / f"{output_dir.name}.yaml")
    for path in names:
        if path.is_file():
            return path
    return None


def _design_spec_text(output_dir: Path) -> str | None:
    """That spec's text, or None when there is none and when it cannot be read."""
    path = _spec_path(output_dir)
    if path is None:
        return None
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return None


def _spec_digest(output_dir: Path) -> str:
    """A short digest of the condition spec this design was generated from.

    ``contig`` and ``hotspot`` are read out of that file, so the record says
    what was asked for -- and could not say whether the file on disk now is the
    file the run was given. The screen digests every file sitting beside the
    run, but a live run's spec sits one directory above it, so nothing digested
    the one input that decides the design. Over the file's bytes, like
    :func:`screen_digest`, so ``sha256sum`` and the first sixteen characters
    answer it.
    """
    path = _spec_path(output_dir)
    if path is None:
        return NOT_RECORDED
    try:
        data = path.read_bytes()
    except OSError:
        return NOT_RECORDED
    return hashlib.sha256(data).hexdigest()[:16]


def _write(path: Path, record: Mapping[str, object], option: StructureOption, example_id: str = "") -> Path | None:
    label = example_id or "the run"
    try:
        write_text_atomically(path, scrubbed_json(record))
    except OSError as exc:
        print(f"Research trace was not saved for {option.entry_id} chain {option.chain} {label}: {exc}")
        return None
    return path


def _file_id(example_id: str, used: set[str]) -> str:
    candidate = example_id if _SAFE_ID.fullmatch(example_id) else _digest(example_id)
    base = candidate
    suffix = 2
    while candidate in used:
        candidate = f"{base}_{suffix}"
        suffix += 1
        if suffix > 10000:
            candidate = _digest(f"{example_id}:{suffix}")
            break
    used.add(candidate)
    return candidate


def _digest(text: str) -> str:
    return f"binder_{hashlib.sha256(text.encode('utf-8')).hexdigest()[:8]}"


def _relative(output_dir: Path, path: Path) -> str:
    try:
        return path.relative_to(output_dir).as_posix()
    except ValueError:
        return path.name


def _string_list(value: object) -> list[str] | str:
    if not isinstance(value, list):
        return NOT_RECORDED
    return [_public_text(item) for item in value if isinstance(item, str)]


def _string_map(value: object) -> dict[str, str] | str:
    """A stored mapping of text to text, such as the per-axis states.

    ``_plain`` answers ``not_recorded`` for a mapping, so copying one through
    it stored the words "not recorded" where the axes were.
    """
    if not isinstance(value, Mapping):
        return NOT_RECORDED
    return {str(key): _public_text(item) for key, item in value.items() if isinstance(item, str)}


def _plain(value: object) -> object:
    if isinstance(value, str):
        return _public_text(value)
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int | float):
        if isinstance(value, float) and not math.isfinite(value):
            return NOT_RECORDED
        return value
    return NOT_RECORDED


def _public_text(text: str) -> str:
    """Short text with a pasted sequence removed."""
    cleaned = " ".join(text.split())
    if not cleaned or _SEQUENCE.search(cleaned) is not None:
        return NOT_RECORDED
    if len(cleaned) > 300:
        return cleaned[:297].rstrip() + "..."
    return cleaned


def _metric_text(value: object) -> str:
    if not isinstance(value, str):
        return NOT_RECORDED
    text = value.strip()
    if text.casefold() in _EMPTY_METRIC or len(text) > 80 or _SEQUENCE.search(text) is not None:
        return NOT_RECORDED
    return text


def _finite(value: object) -> float | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if text.casefold() in _EMPTY_METRIC:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if not math.isfinite(number):
        return None
    return number


def _looks_like_path(value: str) -> bool:
    text = value.strip()
    return text.startswith(("/", "\\")) or (len(text) > 2 and text[1] == ":")


def _stored_path(value: str) -> str:
    """A filesystem path with the checkout prefix and the ``DATA_PATH`` template removed."""
    text = value.strip()
    if not text or text == NOT_RECORDED or "${" in text or "DATA_PATH" in text or _SEQUENCE.search(text):
        return NOT_RECORDED
    if not _looks_like_path(text):
        return _public_text(text)
    return repository_relative(Path(text))


def _hub(value: str) -> str:
    text = value.strip()
    if not text or text in {"null", "None"}:
        return NOT_RECORDED
    if "$" in text or "\\" in text or " " in text or _SEQUENCE.search(text):
        return NOT_RECORDED
    return text


def _pin(value: str) -> str:
    text = value.strip()
    if not text or any(mark in text for mark in ("$", "/", "\\", " ")):
        return NOT_RECORDED
    if _SEQUENCE.search(text):
        return NOT_RECORDED
    return text


def _checkpoint_filename(value: str) -> str:
    name = value.replace("\\", "/").rsplit("/", 1)[-1].strip()
    if not name or "$" in name or _SEQUENCE.search(name):
        return NOT_RECORDED
    return name
