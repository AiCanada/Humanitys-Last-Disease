"""Solubility and aggregation scores supplied for a finished design.

A table beside a run supplies method-labelled solubility and aggregation scores.
Native Protein-Sol output also carries predictions for individual sequences.
The reader checks the sequence identity before reporting the four native fields.
Sequence AGGRESCAN is computed separately and stored on the modality report.

Each supplied score retains its method, source and reported scale. The reader
displays those observations without applying a cross-method threshold.
"""

from __future__ import annotations

import csv
import hashlib
import io
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

from rfproteina.disease.liability import _column_key, _table_lines

SOLUBILITY_TABLE = "solubility"
PROTEIN_SOL_METHOD = "Protein-Sol"
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

_PROTEIN_SOL_FIELDS = {"ID", "percent-sol", "scaled-sol", "population-sol", "pI"}
_PROTEIN_SOL_SOURCE = "Hebditch et al. 2017, doi:10.1093/bioinformatics/btx345"


def read_protein_sol(directory: Path, chains: Sequence[tuple[str, str | None]]) -> dict[str, object] | None:
    """Attach already-computed native Protein-Sol predictions to the matching sample sequences."""
    try:
        report = _protein_sol_text(directory / "seq_prediction.txt")
    except FileNotFoundError:
        return None
    except (OSError, ValueError, OverflowError):
        return {"blocker": "protein_sol_report_unreadable"}
    try:
        predictions = _protein_sol_predictions(report)
    except (ValueError, csv.Error):
        return {"blocker": "protein_sol_report_invalid"}
    if not chains:
        return {"blocker": "protein_sol_samples_missing"}
    try:
        fasta = _protein_sol_fasta(_protein_sol_text(directory / "protein_sol.fasta"))
    except FileNotFoundError:
        fasta = {}
    except (OSError, ValueError, OverflowError):
        return {"blocker": "protein_sol_fasta_invalid"}

    samples: list[dict[str, object]] = []
    for example_id, sequence in chains:
        item: dict[str, object] = {"example_id": example_id}
        samples.append(item)
        if not sequence or len(sequence) < 21 or set(sequence).difference("ACDEFGHIKLMNPQRSTVWY"):
            item["blocker"] = "protein_sol_sequence_unresolved"
            continue
        digest = hashlib.sha256(sequence.encode("ascii")).hexdigest()
        digest_id = f"sha256:{digest}"
        candidates = {digest_id} & predictions.keys()
        if example_id in predictions and example_id != digest_id:
            if fasta.get(example_id) != sequence:
                item["blocker"] = "protein_sol_input_unverified"
                continue
            candidates.add(example_id)
        if len(candidates) != 1:
            item["blocker"] = "protein_sol_ambiguous" if candidates else "protein_sol_sample_missing"
            continue
        identifier = candidates.pop()
        item.update(predictions[identifier])
        item.update(method=PROTEIN_SOL_METHOD, sequence_sha256=digest, input_id=identifier, unit="scaled-sol")
    return {"source": _PROTEIN_SOL_SOURCE, "samples": samples}


def _protein_sol_text(path: Path) -> str:
    with path.open("rb") as stream:
        data = stream.read(SOLUBILITY_FILE_LIMIT + 1)
    if len(data) > SOLUBILITY_FILE_LIMIT:
        raise OverflowError("Protein-Sol artifact exceeds the file size limit")
    return data.decode("utf-8-sig")


def _protein_sol_fasta(text: str) -> dict[str, str]:
    """Use the complete FASTA header and sequence from the predictor input."""
    entries: dict[str, list[str]] = {}
    name = ""
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            name = line[1:]
            if not name or name in entries:
                raise ValueError("Empty or repeated FASTA identifier")
            entries[name] = []
        elif name:
            entries[name].append(line.upper())
        else:
            raise ValueError("Missing FASTA header")
    return {key: "".join(parts) for key, parts in entries.items()}


def _protein_sol_predictions(report: str) -> dict[str, dict[str, float]]:
    fields: list[str] = []
    predictions: dict[str, dict[str, float]] = {}
    for line in report.splitlines():
        tag = line.partition(",")[0].strip()
        if tag not in {"HEADERS PREDICTIONS LINE", "SEQUENCE PREDICTIONS"}:
            continue
        cells = [s.strip() for s in next(csv.reader([line], strict=True))][1:]
        if tag == "HEADERS PREDICTIONS LINE":
            if fields or len(cells) != len(_PROTEIN_SOL_FIELDS) or set(cells) != _PROTEIN_SOL_FIELDS:
                raise ValueError("Ambiguous Protein-Sol prediction columns")
            fields = cells
            continue
        if not fields or len(cells) != len(fields):
            raise ValueError("Protein-Sol prediction row does not match its header")
        row = dict(zip(fields, cells, strict=True))
        identifier = row["ID"].removeprefix(">")
        if not row["ID"].startswith(">") or not identifier or identifier in predictions:
            raise ValueError("Missing or duplicate Protein-Sol sequence identifier")
        score, population, percent, pi = (
            float(row[key]) for key in ("scaled-sol", "population-sol", "percent-sol", "pI")
        )
        if not (0 <= score <= 1 and 0 <= population <= 1 and 0 <= percent <= 100 and 0 <= pi <= 14):
            raise ValueError("Invalid Protein-Sol measurement")
        predictions[identifier] = {
            "score": score,
            "population_scaled_sol": population,
            "predicted_percent_sol": percent,
            "predicted_pI": pi,
        }
    if not predictions:
        raise ValueError("No native Protein-Sol predictions")
    return predictions


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
    # The native Aggrescan3D CSV contains a separate score for each residue.
    # Adding a method and citation to those rows cannot turn them into whole-structure scores.
    if {"protein", "chain", "residue", "residue_name", "score"} <= rows[0].keys() and any(
        row.get("residue", "").strip() for row in rows
    ):
        return {"solubility_blocker": "solubility_residue_profile"}
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
        if (not isinstance(method, str) or not method.strip() or not isinstance(score, int | float)
                or isinstance(score, bool) or not math.isfinite(score)):
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
    invalid = [name for name, value in numbers.items() if value is None]
    if invalid:
        return None, invalid
    column = filled[0]
    number = numbers[column]
    # Distinct scores in one row cannot both be attributed to its single method.
    disagreeing = {name for name, value in numbers.items() if value is not None and value != number}
    if disagreeing:
        return None, sorted({column, *disagreeing})
    for aliases in (_METHOD_COLUMNS, _UNIT_COLUMNS):
        used = [key for key in aliases if _filled(row, key)]
        if len({row[key].strip() for key in used}) > 1:
            return None, used
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
    return number


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
        reader = csv.DictReader(io.StringIO("\n".join(lines)), dialect=dialect, restkey=_EXTRA, strict=True)
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
    return str(value)
