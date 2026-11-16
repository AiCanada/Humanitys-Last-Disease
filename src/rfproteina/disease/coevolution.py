"""Cross-species co-evolution note on the paralog screen.

A protein or peptide screen reads ``paralog_coevolution.tsv`` (or
``paralog_coevolution.csv`` when no ``.tsv`` is present) from the output
directory. The note is stored on the paralog record.

Four transfer notes apply when the table says the same shape exists in another
species. High and low are the words in the co-evolution column. A number is
left unclassified. Toxic and non-toxic are the words in the toxicity column.
The paralog state stays the UniRef50 result.
"""

from __future__ import annotations

import csv
import io
import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from rfproteina.disease.liability import _column_key, _table_lines

COEVOLUTION_TABLE = "paralog_coevolution"
#: Reader bounds, not biological cuts. A table past either bound is withheld
#: whole, so an earlier row cannot stand in for a later one.
COEVOLUTION_ROW_LIMIT = 256
COEVOLUTION_FILE_LIMIT = 1_048_576

#: Human sentences for the four rules. The board and the measured line share
#: this map. None of these sentences is a clinical assay.
COEVOLUTION_SENTENCES: dict[str, str] = {
    "most_likely_toxic": (
        "Same shape in another species, co-evolution high, and toxic there: most likely toxic in people."
    ),
    "lowest_signal_toxic": (
        "Same shape in another species, co-evolution low, and toxic there: lowest signal of toxicity in people."
    ),
    "probably_non_toxic": (
        "Same shape in another species, co-evolution high, and non-toxic there: probably non-toxic in people."
    ),
    "lowest_signal_non_toxic": (
        "Same shape in another species, co-evolution low, and non-toxic there: " "lowest signal, non-toxic in people."
    ),
}

_SHAPE_COLUMNS = ("same_shape", "same_shapes", "shape", "shapes")
_LEVEL_COLUMNS = ("coevolution", "co_evolution")
_TOXICITY_COLUMNS = ("toxicity", "other_species_toxicity")
_SPECIES_COLUMNS = ("species", "organism", "taxon", "other_species")
_USED_COLUMNS = _SHAPE_COLUMNS + _LEVEL_COLUMNS + _TOXICITY_COLUMNS + _SPECIES_COLUMNS
_ABSENT = frozenset(
    {
        "",
        "nan",
        "none",
        "null",
        "na",
        "n/a",
        "n.a.",
        "n.d.",
        "nd",
        ".",
        "-",
        "--",
        "inconclusive",
    }
)
_SHAPE_YES = frozenset({"yes", "y", "true", "same", "present"})
_SHAPE_NO = frozenset({"no", "n", "false", "different", "absent"})
_HIGH = frozenset({"high"})
_LOW = frozenset({"low"})
_TOXIC = frozenset({"toxic", "toxicity"})
_NON_TOXIC = frozenset({"nontoxic", "nottoxic"})
_SIGNAL = {
    ("high", "toxic"): "most_likely_toxic",
    ("low", "toxic"): "lowest_signal_toxic",
    ("high", "non_toxic"): "probably_non_toxic",
    ("low", "non_toxic"): "lowest_signal_non_toxic",
}
_SEQUENCE_UPPER = re.compile(r"[ACDEFGHIKLMNPQRSTVWY]{9,}")
_SEQUENCE_ANY = re.compile(r"[ACDEFGHIKLMNPQRSTVWY]{20,}", re.IGNORECASE)
#: A one-word label spelled entirely from the residue alphabet, in any case.
#: Between the two patterns above lies a band -- nine to nineteen residues in
#: lower or mixed case -- that neither catches, and a designed peptide pasted
#: into the organism column sits in it.
_RESIDUE_WORD = re.compile(r"[ACDEFGHIKLMNPQRSTVWY]+", re.IGNORECASE)
#: Below this, a residue-alphabet word is taken for an organism name rather
#: than a sequence. "salamander", "killifish", "planarian" and "hammerhead" are
#: each one word of nine or ten letters drawn only from that alphabet, so a
#: rule that refused every such word would drop organisms the table did name.
_SEQUENCE_WORD_MIN = 12
#: A sentinel for a field a row carries past the header's last column. ``csv``
#: files those under one key; a row that has one is misaligned from its first
#: extra separator onward, so every later cell is read under its neighbour's
#: column name.
_EXTRA = "\x00extra"


class _TableTooLargeError(Exception):
    """The file could not be held in memory."""


@dataclass
class _Parsed:
    """One data row, after the four columns have been classified."""

    kind: str
    signal: str | None = None
    level: str | None = None
    toxicity: str | None = None
    species: str | None = None
    columns: list[str] = field(default_factory=list)


def coevolution_phrase(record: Mapping[str, object]) -> str | None:
    """The measured-line sentence for a stored note, or None when there is none."""
    signal = record.get("coevolution_signal")
    if not isinstance(signal, str):
        return None
    sentence = COEVOLUTION_SENTENCES.get(signal)
    if sentence is None:
        return None
    return f"paralog coevolution {signal}. {sentence}"


def read_paralog_coevolution(directory: Path) -> dict[str, object]:
    """The co-evolution note for one output directory.

    A missing file is ``coevolution_not_configured``. Rows that share the same
    shape and agree supply one note and ``coevolution_rows``. A row that
    disagrees, a number written as co-evolution, an unparsed cell, or two
    alias columns that name different classes leaves the note unset. A file
    past :data:`COEVOLUTION_FILE_LIMIT` or more data rows than
    :data:`COEVOLUTION_ROW_LIMIT` is withheld whole. The returned keys never
    include ``state``, ``measurement``, or ``blocker``.
    """
    path = _table_path(directory)
    if path is None:
        return {"coevolution_blocker": "coevolution_not_configured"}
    size = _file_size(path)
    if size is None:
        return {"coevolution_blocker": "coevolution_unreadable"}
    if size > COEVOLUTION_FILE_LIMIT:
        return {"coevolution_blocker": "coevolution_too_large"}
    try:
        rows = _rows(path)
    except _TableTooLargeError:
        return {"coevolution_blocker": "coevolution_too_large"}
    if rows is None:
        return {"coevolution_blocker": "coevolution_unreadable"}
    parsed = [row for row in (_parse_row(row) for row in rows) if row.kind != "blank"]
    if len(parsed) > COEVOLUTION_ROW_LIMIT:
        return {"coevolution_blocker": "coevolution_truncated"}
    if not parsed:
        return {"coevolution_blocker": "coevolution_not_reported"}
    unparsed = [row for row in parsed if row.kind == "unparsed"]
    if unparsed:
        columns = sorted({column for row in unparsed for column in row.columns})
        return {"coevolution_blocker": "coevolution_unparsed", "unparsed_columns": columns}
    if any(row.kind == "level" for row in parsed):
        return {"coevolution_blocker": "coevolution_not_classified"}
    if any(row.kind == "toxicity" for row in parsed):
        return {"coevolution_blocker": "toxicity_not_classified"}
    if any(row.kind == "incomplete" for row in parsed):
        return {"coevolution_blocker": "coevolution_not_reported"}
    yes = [row for row in parsed if row.kind == "yes"]
    if not yes:
        return {"coevolution_blocker": "same_shape_absent", "same_shape": False}
    signals = {row.signal for row in yes if row.signal}
    species = _species_list(yes)
    if len(signals) != 1:
        # No row count here. The key means "rows that agreed on this note", and
        # these rows agreed on nothing; carrying it made the page open with
        # "The same note is on 2 rows." above the refusal that denies it.
        refused: dict[str, object] = {
            "coevolution_blocker": "coevolution_disagreement",
            "coevolution_signals": sorted(signals),
            "same_shape": True,
        }
        if species:
            refused["coevolution_species"] = species
        return refused
    chosen = yes[0]
    record: dict[str, object] = {
        "same_shape": True,
        "coevolution_level": chosen.level,
        "other_species_toxicity": chosen.toxicity,
        "coevolution_signal": chosen.signal,
        "coevolution_rows": len(yes),
    }
    if species:
        record["coevolution_species"] = species
    return record


def _file_size(path: Path) -> int | None:
    try:
        return path.stat().st_size
    except OSError:
        return None


def _table_path(directory: Path) -> Path | None:
    for suffix in (".tsv", ".csv"):
        path = directory / f"{COEVOLUTION_TABLE}{suffix}"
        if path.is_file():
            return path
    return None


def _rows(path: Path) -> list[dict[str, str]] | None:
    """Rows with normalised keys, or None when the file did not parse."""
    try:
        text = path.read_text(encoding="utf-8-sig")
    except MemoryError as exc:
        raise _TableTooLargeError from exc
    except (OSError, UnicodeError):
        return None
    text = text.replace("\u00a0", " ").replace("\u2007", " ").replace("\u202f", " ")
    lines = _table_lines(text)
    if not lines:
        return []
    dialect = csv.excel_tab if "\t" in lines[0] else csv.excel
    try:
        reader = csv.DictReader(io.StringIO("\n".join(lines)), dialect=dialect, restkey=_EXTRA)
        if _duplicate_keys(reader.fieldnames):
            return None
        if not any(_column_key(name) in _USED_COLUMNS for name in reader.fieldnames or ()):
            # The header holds none of the columns this reads, which is what a
            # semicolon export and an unmarked banner line both look like.
            # Read on, and the answer is "no usable row for this check" -- which
            # sends the reader to add three columns the file already has, and
            # rules out the delimiter, the one thing actually wrong.
            return None
        found: list[dict[str, str]] = []
        for row in reader:
            if _surplus(row.get(_EXTRA)):
                # Every field past the first unquoted separator is read under
                # the next column's name. On this table that silently moves the
                # co-evolution cell: a row stating high and toxic was read as
                # low and toxic, and the note flipped to the opposite one of
                # the four. A misaligned row is not partly usable.
                return None
            found.append(
                {
                    _column_key(key): (value or "").strip()
                    for key, value in row.items()
                    if isinstance(key, str) and key != _EXTRA
                }
            )
        if len(found) != len(lines) - 1:
            # An unterminated quote swallows every following line into one
            # cell. The rows it ate are not reported missing, so a table whose
            # second row disagreed came back as a single confident note. This
            # also refuses a quoted cell spanning real newlines, which the
            # banner strip above has already corrupted by then, so the reader
            # never supported one.
            return None
        return found
    except csv.Error:
        return None


def _surplus(extra: object) -> bool:
    """Whether a row carried a field the header has no column for."""
    if isinstance(extra, str):
        return bool(extra.strip())
    if isinstance(extra, list):
        return any(isinstance(item, str) and item.strip() for item in extra)
    return extra is not None


def _duplicate_keys(fieldnames: list[str] | None) -> bool:
    """Refuse a repeated column this reader uses. A repeated notes column stays."""
    seen: set[str] = set()
    for raw in fieldnames or ():
        key = _column_key(raw)
        if key not in _USED_COLUMNS:
            continue
        if key in seen:
            return True
        seen.add(key)
    return False


def _parse_row(row: dict[str, str]) -> _Parsed:
    shape, shape_columns = _classified(row, _SHAPE_COLUMNS, "shape")
    level, level_columns = _classified(row, _LEVEL_COLUMNS, "level")
    toxicity, toxicity_columns = _classified(row, _TOXICITY_COLUMNS, "toxicity")
    species = _species_from_row(row)
    if shape == "absent" and level == "absent" and toxicity == "absent" and species is None:
        return _Parsed("blank")
    conflicts = [
        column
        for kind, columns in ((shape, shape_columns), (level, level_columns), (toxicity, toxicity_columns))
        if kind == "conflict"
        for column in columns
    ]
    if conflicts:
        return _Parsed("unparsed", columns=sorted(set(conflicts)))
    if shape == "unknown":
        return _Parsed("unparsed", columns=shape_columns or ["same_shape"])
    if level == "unknown":
        return _Parsed("level", columns=level_columns or ["coevolution"])
    if toxicity == "unknown":
        return _Parsed("toxicity", columns=toxicity_columns or ["toxicity"])
    if shape == "no":
        return _Parsed("no", species=species)
    if shape != "yes" or level not in {"high", "low"} or toxicity not in {"toxic", "non_toxic"}:
        return _Parsed("incomplete")
    return _Parsed(
        "yes",
        signal=_SIGNAL[(level, toxicity)],
        level=level,
        toxicity=toxicity,
        species=species,
    )


def _classified(row: dict[str, str], names: tuple[str, ...], kind: str) -> tuple[str, list[str]]:
    """The shared class of the filled alias columns, or ``conflict``."""
    filled: list[tuple[str, str]] = []
    for name in names:
        if name not in row:
            continue
        klass = _cell_class(row[name], kind)
        if klass == "absent":
            continue
        filled.append((name, klass))
    if not filled:
        return "absent", []
    classes = {klass for _name, klass in filled}
    columns = [name for name, _klass in filled]
    if len(classes) != 1:
        return "conflict", columns
    return filled[0][1], columns


def _cell_class(raw: str, kind: str) -> str:
    """A known word for this column, ``absent``, or ``unknown``."""
    text = _normalize(raw)
    if text.casefold() in _ABSENT:
        return "absent"
    compact = "".join(character for character in text.casefold() if character.isalnum())
    if not compact:
        return "absent"
    if kind == "shape":
        if compact in _SHAPE_YES or text == "1":
            return "yes"
        if compact in _SHAPE_NO or text == "0":
            return "no"
        return "unknown"
    if kind == "level":
        if compact in _HIGH:
            return "high"
        if compact in _LOW:
            return "low"
        return "unknown"
    if compact in _TOXIC:
        return "toxic"
    if compact in _NON_TOXIC:
        return "non_toxic"
    return "unknown"


def _normalize(raw: str) -> str:
    """Strip Excel padding. A trailing period on a single word is not part of the word."""
    text = raw.replace("\u00a0", " ").replace("\u2007", " ").replace("\u202f", " ")
    text = "".join(character for character in text if unicodedata.category(character) != "Cf")
    text = text.strip()
    if text.endswith(".") and "." not in text[:-1]:
        text = text[:-1].strip()
    return text


def _species_from_row(row: dict[str, str]) -> str | None:
    """One species label. Two different labels on the same row are omitted."""
    names: list[str] = []
    for column in _SPECIES_COLUMNS:
        if column not in row:
            continue
        name = _species_name(row[column])
        if name and name not in names:
            names.append(name)
    if len(names) == 1:
        return names[0]
    return None


def _species_list(rows: list[_Parsed]) -> list[str]:
    names: list[str] = []
    for row in rows:
        if row.species and row.species not in names:
            names.append(row.species)
    return sorted(names)


def _species_name(raw: str) -> str | None:
    """A species label. A pasted sequence is dropped rather than stored."""
    text = _normalize(raw)
    if not text or text.casefold() in _ABSENT:
        return None
    compact = "".join(text.split())
    if _SEQUENCE_UPPER.search(text) or _SEQUENCE_UPPER.search(compact):
        return None
    if _SEQUENCE_ANY.search(text) or _SEQUENCE_ANY.search(compact):
        return None
    if any(unicodedata.category(character).startswith("C") for character in text):
        return None
    if _reads_as_sequence(text):
        return None
    if any(character.isspace() for character in text):
        return text if len(text) <= 80 else None
    if len(text) >= 20:
        return None
    return text


def _reads_as_sequence(text: str) -> bool:
    """Whether a one-word label is more likely a pasted sequence than an organism.

    Only a whitespace-free word spelled entirely from the residue alphabet can
    qualify, and then only when it is long enough that no organism name is
    likely, or carries a capital after its first letter the way a pasted
    sequence does and a name does not.
    """
    if any(character.isspace() for character in text) or not _RESIDUE_WORD.fullmatch(text):
        return False
    return len(text) >= _SEQUENCE_WORD_MIN or any(character.isupper() for character in text[1:])
