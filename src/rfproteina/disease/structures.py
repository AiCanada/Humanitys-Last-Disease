"""Structure options for one confirmed disease, in four groups."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, fields
from datetime import UTC, datetime
from pathlib import Path

from rfproteina.disease.publish import scrubbed_json

GROUPS = ("protein", "dna", "rna", "gene")
#: Design classes a returned 3D structure can start. The count of rows in each class
#: is the campaign score. One deposit can count in more than one class.
CAMPAIGNS = (
    "Proteins",
    "DNA & RNA",
    "Small Molecules",
    "Sugars (Glycans)",
    "Antibodies & Nanobodies",
)


@dataclass(frozen=True)
class RecordScore:
    """What the asked public databases hold for one target.

    ``instances`` counts every entry returned, so a database holding three entries
    contributes three and the total can exceed ``asked``. ``databases`` counts how
    many of the asked databases returned anything at all. Both read against
    ``asked`` so the two numbers share a denominator.
    """

    instances: int
    databases: int
    asked: int


def score_parts(score: RecordScore | tuple[int, int]) -> tuple[int, int, int]:
    """Instances, databases with a record, and how many databases were asked.

    A pair is the older ``(databases, asked)`` score. Its instance count is the
    database count, because that pair did not record entries separately.
    """
    if isinstance(score, RecordScore):
        return score.instances, score.databases, score.asked
    databases, asked = score
    return databases, databases, asked


@dataclass(frozen=True)
class StructureOption:
    """One atomic structure the user can send to generation."""

    group: str
    accession: str
    entry_id: str
    chain: str
    title: str
    predicted: bool
    source: str
    #: False when the deposit's determination method could not be read. Such a row is
    #: not treated as experimental, so one failed lookup cannot slip a computed model
    #: into a selection that asked for experimental structures only.
    methodology_known: bool = True
    #: Design classes this row can start. Empty means they are read from the group and title.
    campaigns: tuple[str, ...] = ()


def structure_selector(row: StructureOption) -> str:
    """Opaque, order-independent identity for selecting one freshly listed row."""
    identity = json.dumps(
        ["structure-selector-v1", row.group, row.entry_id, row.chain],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(identity).hexdigest()


def _write_source_record(dest: Path, record: StructureOption, payload: bytes) -> None:
    """Note where these coordinates came from, beside the file they went into.

    The archive held a copy of a structure with no way to show it was the file
    that was fetched. A later edit, a truncation, or a re-download of a revised
    deposit was invisible, and every downstream claim about the binder rests on
    these bytes being the ones that were used.

    The digest is what settles that; the rest is the identity the row was
    fetched under. The exact URL is not here: ``fetch`` is injected by the
    caller, so this function never sees one, and inventing the address from the
    entry id would record a guess as provenance. A failure to write is not
    allowed to lose the download that just succeeded.
    """
    found = {
        "version": 1,
        "file": dest.name,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        "fetched_at": datetime.now(UTC).isoformat(),
        "source": record.source or "not_recorded",
        "entry_id": record.entry_id,
        "chain": record.chain,
        "accession": record.accession,
        "predicted": record.predicted,
        "methodology_known": record.methodology_known,
    }
    try:
        dest.with_suffix(".source.json").write_text(scrubbed_json(found), encoding="utf-8")
    except OSError as exc:
        print(f"Source record was not written for {dest.name}: {exc}")


def options_from_records(payload: object) -> list[StructureOption]:
    """Read structure rows from a list or from a ``returned_structures.json`` file.

    The saved file is ``{"version": 1, "structures": [...]}``. Extra score fields
    are kept in that file for comparison and are not part of a structure row.
    """
    if isinstance(payload, dict):
        rows = payload.get("structures")
    elif isinstance(payload, list):
        rows = payload
    else:
        rows = None
    if not isinstance(rows, list):
        raise ValueError("Structure records must be a list, or a returned_structures file.")
    known = {item.name for item in fields(StructureOption)}
    loaded: list[StructureOption] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Each structure record must be an object.")
        values = {key: row[key] for key in known if key in row}
        campaigns = values.get("campaigns")
        if isinstance(campaigns, list):
            values["campaigns"] = tuple(str(name) for name in campaigns)
        try:
            loaded.append(StructureOption(**values))
        except TypeError as exc:
            raise ValueError(f"Structure record could not be read: {exc}") from exc
    return loaded


def campaign_counts(rows: list[StructureOption]) -> tuple[tuple[str, int], ...]:
    """How many listed structures can start a design of each class.

    A row with no stored classes is counted from its group and title. Proteins and
    genes count as protein interfaces. DNA and RNA count as nucleic-acid complexes.
    A title that names an antibody, a glycan, or a small-molecule ligand counts
    there too. Sequence motifs are not a class and are not counted.
    """
    counts = {name: 0 for name in CAMPAIGNS}
    for row in rows:
        for name in _campaigns(row):
            counts[name] += 1
    return tuple((name, counts[name]) for name in CAMPAIGNS)


def structure_campaigns(row: StructureOption) -> tuple[str, ...]:
    """Design classes this row can start, the same classes the campaign counts use."""
    return _campaigns(row)


def campaign_pick_numbers(rows: list[StructureOption], names: list[str]) -> list[str]:
    """1-based listed numbers of rows in any of the named design classes.

    A name that is not a design class is ignored. A row in more than one named
    class is returned once, in listed order. These numbers are the structure
    ticks a click on that class selects.
    """
    wanted = {name.strip() for name in names if name.strip() in CAMPAIGNS}
    if not wanted:
        return []
    found: list[str] = []
    for number, row in enumerate(rows, start=1):
        if wanted.intersection(_campaigns(row)):
            found.append(str(number))
    return found


def _campaigns(row: StructureOption) -> tuple[str, ...]:
    stored = tuple(name for name in row.campaigns if name in CAMPAIGNS)
    if stored:
        return stored
    found: list[str] = []
    if row.group in {"protein", "gene"}:
        found.append("Proteins")
    if row.group in {"dna", "rna"}:
        found.append("DNA & RNA")
    text = row.title.casefold()
    if any(word in text for word in ("antibody", "nanobody", "immunoglobulin", "vhh")):
        found.append("Antibodies & Nanobodies")
    if any(word in text for word in ("glycan", "carbohydrate", "oligosaccharide")):
        found.append("Sugars (Glycans)")
    if any(word in text for word in ("ligand", "inhibitor", "substrate", "compound")):
        found.append("Small Molecules")
    return tuple(found)


def group_options(records: list[StructureOption]) -> dict[str, list[StructureOption]]:
    """Place every record into protein, DNA, RNA, or gene. Predicted rows stay visible."""
    grouped: dict[str, list[StructureOption]] = {group: [] for group in GROUPS}
    for record in records:
        if record.group not in grouped:
            raise ValueError(f"Unknown structure group {record.group!r}. Expected one of {GROUPS}.")
        grouped[record.group].append(record)
    return grouped


def select_options(
    grouped: dict[str, list[StructureOption]],
    chosen: list[StructureOption] | None,
) -> list[StructureOption]:
    """Return the known-experimental options, or an explicit selection.

    ``chosen is None`` means take every structure known to be experimental.
    Predicted rows, and rows whose determination method could not be read, stay
    unselected unless the caller names them in ``chosen``.
    """
    flat = [record for group in GROUPS for record in grouped.get(group, [])]
    if chosen is None:
        return [record for record in flat if not record.predicted and record.methodology_known]
    chosen_ids = {(record.entry_id, record.chain, record.group) for record in chosen}
    return [record for record in flat if (record.entry_id, record.chain, record.group) in chosen_ids]


def structure_kind(row: StructureOption) -> str:
    """``predicted``, ``experimental``, or ``unknown`` when the method was not read."""
    if row.predicted:
        return "predicted"
    if row.methodology_known:
        return "experimental"
    return "unknown"


def apply_structure_view(
    rows: list[StructureOption],
    *,
    take_all: bool,
    picks: list[str],
    experimental: str = "include",
    predicted: str = "include",
) -> tuple[bool, list[str]]:
    """Drop ticks for kinds the page excluded. Numbers stay in listed order.

    ``take_all`` means every known experimental row. Excluding experimental
    structures clears it. A pick that does not name a listed row is kept so the
    caller can report it. Any value other than ``exclude`` includes that kind.
    A row whose method was not read follows the experimental choice.
    """
    show_experimental = experimental != "exclude"
    show_predicted = predicted != "exclude"
    if not show_experimental:
        take_all = False
    kept: list[str] = []
    for item in picks:
        text = item.strip()
        if not text.isdigit():
            if text:
                kept.append(text)
            continue
        index = int(text)
        if not 1 <= index <= len(rows):
            kept.append(text)
            continue
        kind = structure_kind(rows[index - 1])
        visible = show_predicted if kind == "predicted" else show_experimental
        if visible:
            kept.append(text)
    return take_all, kept


def returned_structure_records(
    rows: list[StructureOption],
    scores: dict[str, RecordScore | tuple[int, int]] | None = None,
) -> list[dict[str, object]]:
    """The returned structures, numbered in listed order, with program scores."""
    found: list[dict[str, object]] = []
    for number, row in enumerate(rows, start=1):
        score = (scores or {}).get(row.accession.strip().casefold())
        instances = databases = asked = None
        if score is not None:
            instances, databases, asked = score_parts(score)
        found.append(
            {
                "number": number,
                "selector": structure_selector(row),
                "group": row.group,
                "accession": row.accession,
                "entry_id": row.entry_id,
                "chain": row.chain,
                "title": row.title,
                "predicted": row.predicted,
                "methodology_known": row.methodology_known,
                "kind": structure_kind(row),
                "source": row.source,
                "campaigns": list(row.campaigns),
                "program_score": databases,
                "instances": instances,
                "databases": databases,
                "sources_asked": asked,
            }
        )
    return found


def save_returned_structures(
    directory: Path,
    rows: list[StructureOption],
    scores: dict[str, RecordScore | tuple[int, int]] | None = None,
) -> Path:
    """Write ``returned_structures.json`` for a later comparison with the LLM search."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "returned_structures.json"
    payload = {"version": 1, "structures": returned_structure_records(rows, scores)}
    path.write_text(scrubbed_json(payload), encoding="utf-8")
    return path


def download_selected(
    selected: list[StructureOption],
    directory: Path,
    fetch: Callable[[StructureOption], bytes],
    progress: Callable[[int, int, str], None] | None = None,
) -> list[Path]:
    """Write one CIF per selected structure. ``fetch`` is called only for those rows.

    ``progress`` receives the count already written, the selection size, and the
    structure being fetched. The last call reports the full count.
    """
    directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    total = len(selected)
    label = ""
    for index, record in enumerate(selected):
        label = f"{record.entry_id} chain {record.chain}"
        if progress is not None:
            progress(index, total, label)
        dest = directory / f"{record.group}_{record.entry_id}_{record.chain}.cif"
        payload = fetch(record)
        dest.write_bytes(payload)
        _write_source_record(dest, record, payload)
        written.append(dest)
    if progress is not None and total:
        progress(total, total, label)
    return written
