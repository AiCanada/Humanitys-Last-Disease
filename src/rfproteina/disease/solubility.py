"""Solubility and aggregation scores supplied for a finished design.

Aggregation is the developability failure that ends a binder quietly: it
expresses, it aggregates, and nothing before the bench said so. CamSol,
Aggrescan3D, and Protein-Sol are not run here. Their coefficient tables are
not in this program, and a designed sequence does not leave the machine. A
supplied score for any of them, including a supplied AGGRESCAN row, comes from
a table beside the run, the way ``plasma_fold.tsv`` carries a fold change. The
sequence AGGRESCAN is computed separately and stored on the modality report.

Every row has to name its method as well as its score. A solubility number is
meaningless without the tool that produced it: 0.4 is a poor CamSol score and a
low AGGRESCAN one, and the two mean opposite things. For the same reason this
module judges nothing. It reports what the table says, with the method and the
provenance attached, and no cut turns it into a hit or a clear -- the scales are
not comparable to each other and no published threshold spans them. It is a
report, like the chemical liabilities, not an axis: the screen's call does not
move.
"""

from __future__ import annotations

import csv
import io
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

from rfproteina.disease.liability import _column_key, _table_lines

SOLUBILITY_TABLE = "solubility"
#: Rows one table may carry. A predictor panel is a handful of scores, not a
#: dataset, and a file past this is withheld whole so an earlier row cannot
#: stand in for a later one.
SOLUBILITY_ROW_LIMIT = 32
SOLUBILITY_FILE_LIMIT = 1_048_576

_SCORE_COLUMNS = ("score", "solubility", "solubility_score", "aggregation", "aggregation_score", "value")
_METHOD_COLUMNS = ("method", "tool", "predictor")
_SOURCE_COLUMNS = ("source", "study", "reference", "doi", "version")
_UNIT_COLUMNS = ("unit", "units")
_NOTE_COLUMNS = ("note", "notes", "comment")
_USED_COLUMNS = _SCORE_COLUMNS + _METHOD_COLUMNS + _SOURCE_COLUMNS + _UNIT_COLUMNS
_ABSENT = frozenset({"", "nan", "none", "null", "na", "n/a", "n.a.", "n.d.", "nd", ".", "-", "--", "inconclusive"})
#: A field a row carries past the header's last column. A row with one is
#: misaligned from its first extra separator on, so a score would be read under
#: another column's name.
_EXTRA = "\x00extra"
_TEXT_LIMIT = 120


def read_solubility(directory: Path) -> dict[str, object]:
    """Solubility rows supplied beside one run, or a blocker saying why there are none.

    Each row needs a score, the method that produced it, and a provenance.
    A row missing any of the three is not stored: a bare number carries no
    scale, and a number with no study behind it cannot be checked. Rows are
    kept in file order, and the record never contains a state or a measurement
    the screen could mistake for an axis.
    """
    path = _table_path(directory)
    if path is None:
        return {"solubility_blocker": "solubility_not_configured"}
    size = _file_size(path)
    if size is None:
        return {"solubility_blocker": "solubility_unreadable"}
    if size > SOLUBILITY_FILE_LIMIT:
        return {"solubility_blocker": "solubility_too_large"}
    rows = _rows(path)
    if rows is None:
        return {"solubility_blocker": "solubility_unreadable"}
    if not rows:
        return {"solubility_blocker": "solubility_not_reported"}
    if len(rows) > SOLUBILITY_ROW_LIMIT:
        return {"solubility_blocker": "solubility_truncated"}
    scored: list[dict[str, object]] = []
    unparsed: list[str] = []
    incomplete = 0
    for row in rows:
        found, bad = _score_row(row)
        if bad:
            unparsed.extend(bad)
            continue
        if found is None:
            incomplete += 1
            continue
        scored.append(found)
    if unparsed:
        return {"solubility_blocker": "cell_unparsed", "unparsed_columns": sorted(set(unparsed))}
    if not scored:
        return {"solubility_blocker": "solubility_not_reported", "incomplete_rows": incomplete}
    record: dict[str, object] = {"scores": scored, "rows": len(scored)}
    if incomplete:
        # Said out loud: a row dropped for want of a method or a study is not
        # the same as a table that held nothing, and silence about it would
        # make a partial read look complete.
        record["incomplete_rows"] = incomplete
    return record


def solubility_phrase(record: Mapping[str, object]) -> str | None:
    """The measured-line clause, or None when no score was supplied."""
    scores = record.get("scores")
    if not isinstance(scores, list) or not scores:
        return None
    parts: list[str] = []
    for item in scores:
        if not isinstance(item, Mapping):
            continue
        method = item.get("method")
        score = item.get("score")
        if not isinstance(method, str) or not isinstance(score, int | float):
            continue
        piece = f"{method} {_number(score)}"
        unit = item.get("unit")
        if isinstance(unit, str) and unit:
            piece = f"{piece} {unit}"
        parts.append(piece)
    if not parts:
        return None
    return "solubility supplied: " + "; ".join(parts)


def _score_row(row: Mapping[str, str]) -> tuple[dict[str, object] | None, list[str]]:
    """One stored row, or None when it is incomplete, plus the columns to refuse."""
    filled = [name for name in _SCORE_COLUMNS if _filled(row, name)]
    if not filled:
        return None, []
    numbers = {name: _number_cell(row.get(name, "")) for name in filled}
    column = filled[0]
    number = numbers[column]
    if number is None:
        return None, [column]
    # A row carries one method, and the column the number came from is what
    # says which scale it is on: 0.4 is a poor CamSol score and a low
    # AGGRESCAN one. Two score columns holding different numbers leave nothing
    # here able to say which of them the method names, and the first in a
    # fixed alias order was being published while the other was dropped
    # without a word -- a solubility of 0.1 reported from a row that also said
    # aggregation 0.9. The same number written into two aliases is one
    # measurement and is kept.
    disagreeing = {name for name, value in numbers.items() if value is not None and value != number}
    if disagreeing:
        return None, sorted({column, *disagreeing})
    method = _text(_first_filled(row, _METHOD_COLUMNS)[1])
    source = _text(_first_filled(row, _SOURCE_COLUMNS)[1])
    if not method or not source:
        # A score with no tool has no scale, and one with no study cannot be
        # traced. Either way there is nothing a reader could do with the number.
        return None, []
    found: dict[str, object] = {"score": number, "method": method, "source": source}
    unit = _text(_first_filled(row, _UNIT_COLUMNS)[1])
    if unit:
        found["unit"] = unit
    note = _text(_first_filled(row, _NOTE_COLUMNS)[1])
    if note:
        found["note"] = note
    return found, []


def _first_filled(row: Mapping[str, str], names: Sequence[str]) -> tuple[str | None, str]:
    """The first of ``names`` this row fills, and its cell."""
    for name in names:
        if _filled(row, name):
            return name, row[name]
    return None, ""


def _filled(row: Mapping[str, str], name: str) -> bool:
    """Whether this row holds anything in that column. A sentinel holds nothing."""
    raw = row.get(name, "")
    return bool(raw) and raw.strip().casefold() not in _ABSENT


def _number_cell(raw: str) -> float | None:
    text = raw.strip()
    if text.casefold() in _ABSENT:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if not math.isfinite(number):
        return None
    return float(f"{number:.4f}")


def _text(raw: str) -> str:
    """Short public text. A long cell is cut rather than stored whole."""
    cleaned = " ".join(raw.split())
    if not cleaned or cleaned.casefold() in _ABSENT:
        return ""
    if len(cleaned) > _TEXT_LIMIT:
        return cleaned[: _TEXT_LIMIT - 3].rstrip() + "..."
    return cleaned


def _file_size(path: Path) -> int | None:
    try:
        return path.stat().st_size
    except OSError:
        return None


def _table_path(directory: Path) -> Path | None:
    for suffix in (".tsv", ".csv"):
        path = directory / f"{SOLUBILITY_TABLE}{suffix}"
        if path.is_file():
            return path
    return None


def _rows(path: Path) -> list[dict[str, str]] | None:
    """Rows with normalised keys, or None when the file did not parse.

    The three refusals here are the ones that change an answer rather than
    withhold it, and each was a real defect in a sibling reader before it was
    one here: a repeated column, a header holding nothing this reads, a row
    with a surplus field, and a quote left open.
    """
    try:
        text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError, MemoryError):
        return None
    lines = _table_lines(text)
    if not lines:
        return []
    dialect = csv.excel_tab if "\t" in lines[0] else csv.excel
    try:
        reader = csv.DictReader(io.StringIO("\n".join(lines)), dialect=dialect, restkey=_EXTRA)
        names = list(reader.fieldnames or ())
        if _duplicate_keys(names):
            return None
        if not any(_column_key(name) in _USED_COLUMNS for name in names):
            return None
        found: list[dict[str, str]] = []
        for row in reader:
            if _surplus(row.get(_EXTRA)):
                return None
            found.append(
                {
                    _column_key(key): (value or "").strip()
                    for key, value in row.items()
                    if isinstance(key, str) and key != _EXTRA
                }
            )
        if len(found) != len(lines) - 1:
            return None
        return found
    except csv.Error:
        return None


def _duplicate_keys(fieldnames: Sequence[str]) -> bool:
    """Refuse a repeated column this reader uses. A repeated notes column stays."""
    seen: set[str] = set()
    for raw in fieldnames:
        key = _column_key(raw)
        if key not in _USED_COLUMNS:
            continue
        if key in seen:
            return True
        seen.add(key)
    return False


def _surplus(extra: object) -> bool:
    if isinstance(extra, str):
        return bool(extra.strip())
    if isinstance(extra, list):
        return any(isinstance(item, str) and item.strip() for item in extra)
    return extra is not None


def _number(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.4f}".rstrip("0").rstrip(".")
