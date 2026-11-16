"""Ligand safety-panel readings for a finished generation.

A ligand binder is screened on the panel in :data:`rfproteina.disease.combinations.LIGAND_AXES`.
Each reading is a concentration in nM from ``toxicology.tsv`` or ``toxicology.csv``
in the output directory. :func:`rfproteina.disease.combinations.ligand_states`
applies that axis's own cut from :data:`rfproteina.disease.combinations.PANEL_NM`:
a reading at or below the cut is a hit, and a higher reading is clear. Every
entry of that table is 10_000 nM, which is a program convention and not a
published threshold, so each record stores the cut it was judged against and
the ``threshold_source`` naming where that cut came from. A missing file or a
missing cell stays untested.

Every refusal here is loud. A table this cannot read in full is refused in
full, because the shapes that break a reader -- a misaligned row, a cell that
swallowed the end of its row, a concentration attributed to no protein or to
two of them -- move one protein's number onto another protein's name, and a
wrong clear on a safety panel is the one answer this screen must never
produce. Where the table is readable but a cell is not, the axis stays
untested rather than borrowing a sibling's number.

This module does not launch a predictor, contact a service, or score a sequence.
A clear reading is that table's number against the existing cut.
"""

from __future__ import annotations

import csv
import io
import math
from dataclasses import dataclass, field
from pathlib import Path

from rfproteina.disease.combinations import HIT, LIGAND_AXES, ligand_states, panel_cut, panel_cut_source
from rfproteina.disease.liability import _column_key, _table_lines

TOXICOLOGY_TABLE = "toxicology"
TOXICOLOGY_METHOD = "ligand_panel_nm"

_AXES = frozenset(LIGAND_AXES)
_AXIS_COLUMNS = ("axis", "protein", "panel")
#: Where a long-form row writes its concentration. ``standard_value`` is
#: ChEMBL's, beside the ``standard_unit`` and ``standard_relation`` this
#: already reads: a panel exported from ChEMBL carried its readings under a
#: column this did not know, and every axis in it was reported
#: ``toxicology_not_reported`` -- a 50 nM hERG hit dropped, and the record
#: saying the table had not reported hERG when it had. ``ic50`` is the spelling
#: :mod:`rfproteina.disease.liability` reads for the same reason.
_VALUE_COLUMNS = (
    "nm",
    "ic50_nm",
    "value_nm",
    "reading_nm",
    "concentration_nm",
    "aff_nm",
    "standard_value",
    "ic50",
)
#: The value columns whose own name states the unit, which is the ones whose
#: name ends in nM. ``standard_value`` and ``ic50`` name only the analyte, so
#: nothing in such a column's name says what its numbers are in and the row has
#: to: this is the rule the bare panel column is read under, applied to the
#: long-form columns as well. Derived rather than listed, because a second list
#: of the same names is a list that gets left behind when the first one grows.
_VALUE_NM_COLUMNS = frozenset(name for name in _VALUE_COLUMNS if name == "nm" or name.endswith("_nm"))
_UNIT_COLUMNS = ("unit", "units", "uom", "standard_unit", "standard_units")
#: Where an export writes the relation its number is qualified by. ChEMBL calls
#: it ``standard_relation``. Read with the number, never apart from it: a 50000
#: under ``<`` is not a reading of 50000.
_RELATION_COLUMNS = ("relation", "standard_relation")
#: Spellings of a panel protein that normalise to something other than its axis
#: id. ``P-gp`` is the spelling the panel's own tab teaches, and ``_column_key``
#: turns its hyphen into an underscore.
_AXIS_ALIASES = {"p_gp": "pgp", "nav_1_5": "nav1_5"}
#: Bounds a cell or a relation column can carry, longest first so ``>=`` is not
#: read as ``>``.
_BOUNDS = (">=", "<=", "=>", "=<", "≥", "≤", ">", "<", "=")
_BOUND_FORM = {"=>": ">=", "=<": "<=", "≥": ">=", "≤": "<=", "=": ""}
#: A surplus field on a row with more values than the header has columns.
#: ``csv`` files them under this key, and a row that has one is misaligned from
#: its first extra separator onward, so every later cell belongs to its left
#: neighbour's column.
_EXTRA = "\x00extra"
#: Longer suffixes first, so ``herg_ic50_nm`` stays ``herg`` rather than ``herg_ic50``.
_VALUE_SUFFIXES = ("_concentration_nm", "_reading_nm", "_ic50_nm", "_value_nm", "_aff_nm", "_nm")
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
        "true",
        "false",
    }
)
#: Spellings that mean the cell is already nM. Anything else named in a unit
#: column is refused. A micromolar figure is not rescaled onto this cut.
_NM_UNITS = frozenset({"nm", "nanomolar", "nmol/l", "nmol/litre", "nmol/liter"})


@dataclass
class _AxisCells:
    """Every cell that named one panel axis.

    ``values`` holds one entry per usable cell: the number the judge is given,
    the number the table actually wrote, and the bound it was written with.
    The two numbers differ only where a bound sits exactly on the cut, and the
    stored record keeps the written one so a tab never shows a figure the
    table does not contain.
    """

    values: list[tuple[float, float, str]] = field(default_factory=list)
    unparsed: bool = False
    unparsed_columns: list[str] = field(default_factory=list)
    not_nm: bool = False
    undecided: bool = False
    bad: float | None = None


def measure_toxicology(directory: Path) -> dict[str, dict[str, object]]:
    """One record per panel axis, in panel order.

    ``toxicology.tsv`` is read before ``toxicology.csv``. Cells are nM. The
    lowest finite non-negative cell is the reading when several name one axis,
    and that count is ``n_readings``. A micromolar cell is never rescaled, and
    a column named only for the panel protein needs a unit cell saying nM
    before its numbers are read at all.

    A column named only for the analyte -- ``standard_value``, ``ic50`` --
    needs that unit cell for the same reason.

    A cell or a relation column may bound the number rather than state it, and
    a bound decides the axis when everything it admits falls on one side of the
    cut; ``qualifier`` then records the bound and ``measurement`` stays the
    figure the table wrote. An indecisive bound, an unparsed cell, a negative
    cell or a non-nM unit leaves the axis untested, **unless** another cell for
    the same axis already measured a hit: a reading this table made cannot be
    deleted by a sibling cell it could not read. The stored ``threshold`` is
    that axis's own cut, and ``threshold_source`` says where it came from.
    """
    path = _table_path(directory)
    if path is None:
        return _every("toxicology_not_configured")
    rows = _rows(path)
    if rows is None:
        return _every("toxicology_unreadable")
    if not rows:
        return _every("toxicology_not_reported")
    found, unattributed = _collect(rows)
    if unattributed:
        # A concentration on a row that names no protein, or that names two of
        # the panel's own proteins, cannot be filed. Read past it, the table
        # reports whatever its other rows say while silently holding a reading
        # nothing accounts for, and that reading is as likely to be the
        # liability as not.
        return _every("toxicology_unreadable")
    judged: dict[str, float] = {}
    best: dict[str, tuple[float, float, str, int]] = {}
    for axis in LIGAND_AXES:
        cells = found.get(axis)
        if cells is None or not cells.values:
            continue
        pick = min(cells.values, key=lambda item: item[0])
        best[axis] = (*pick, len(cells.values))
        judged[axis] = pick[0]
    states = ligand_states(judged) if judged else {}
    records: dict[str, dict[str, object]] = {}
    for axis in LIGAND_AXES:
        cells = found.get(axis)
        if cells is None or _untouched(cells):
            records[axis] = _record("untested", None, "toxicology_not_reported", axis=axis)
            continue
        state = states.get(axis)
        if state == HIT or (state is not None and not _spoiled(cells)):
            # A hit stands whatever else the row carried. A sibling cell that
            # could not be read cannot delete a liability this table measured:
            # dropping it turned a panel hit into a screened_clear headline,
            # which is the one direction this screen must never move in.
            _judged, shown, bound, count = best[axis]
            measured = _record(state, shown, None, axis=axis)
            measured["n_readings"] = count
            if bound:
                measured["qualifier"] = bound
            records[axis] = measured
            continue
        records[axis] = _blocked(cells, axis)
    return records


def _spoiled(cells: _AxisCells) -> bool:
    """Whether any cell for this axis could not be used."""
    return cells.unparsed or cells.not_nm or cells.undecided


def _blocked(cells: _AxisCells, axis: str) -> dict[str, object]:
    """Why this axis has no usable reading. The loudest cause is named first."""
    if cells.unparsed:
        refused = _record("untested", None, "cell_unparsed", axis=axis)
        refused["unparsed_columns"] = sorted(set(cells.unparsed_columns))
        return refused
    if cells.not_nm:
        return _record("untested", cells.bad, "reading_not_nm", axis=axis)
    if cells.undecided:
        return _record("untested", None, "reading_not_decisive", axis=axis)
    return _record("untested", None, "toxicology_not_reported", axis=axis)


def _every(blocker: str) -> dict[str, dict[str, object]]:
    return {axis: _record("untested", None, blocker, axis=axis) for axis in LIGAND_AXES}


def _record(state: str, measurement: float | None, blocker: str | None, *, axis: str) -> dict[str, object]:
    """One panel record, carrying the cut this axis was judged against.

    ``threshold`` used to be the one panel-wide constant for all ten proteins,
    so a reader of a stored record could not tell which number a clear was
    taken against, nor that no publication stood behind it. Both now travel
    with the record.
    """
    return {
        "state": state,
        "measurement": measurement,
        "blocker": blocker,
        "method": TOXICOLOGY_METHOD,
        "units": "nM",
        "threshold": panel_cut(axis),
        "threshold_source": panel_cut_source(axis),
    }


def _untouched(cells: _AxisCells) -> bool:
    return not (cells.values or cells.unparsed or cells.not_nm or cells.undecided)


def _read_column(key: str) -> bool:
    """Whether a duplicate of this column would make the table ambiguous."""
    return (
        key in _AXIS_COLUMNS
        or key in _VALUE_COLUMNS
        or key in _UNIT_COLUMNS
        or key in _RELATION_COLUMNS
        or _axis_from_column(key) is not None
    )


def _table_path(directory: Path) -> Path | None:
    for suffix in (".tsv", ".csv"):
        path = directory / f"{TOXICOLOGY_TABLE}{suffix}"
        if path.is_file():
            return path
    return None


def _rows(path: Path) -> list[dict[str, str]] | None:
    """Rows with normalised keys, or None when the file did not parse."""
    try:
        text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError, MemoryError):
        # MemoryError alongside the other two, as ``solubility._rows`` and the
        # fold reader in ``target_liability`` both already do. Reading the
        # whole panel table is fine for the file this is meant for, but a table
        # too large to hold used to raise out of ``measure_toxicology`` and end
        # the screen -- every other axis and every other design unmeasured
        # because one file in the directory was too big.
        #
        # No size bound is added. Unlike those two siblings this reader has
        # never had one, so a bound chosen now would start refusing panel
        # tables that parse today, and refusing a table that would have
        # measured is the failure this screen exists to avoid.
        return None
    lines = _table_lines(text)
    if not lines:
        return []
    dialect = csv.excel_tab if "\t" in lines[0] else csv.excel
    try:
        reader = csv.DictReader(io.StringIO("\n".join(lines)), dialect=dialect, restkey=_EXTRA)
        if _duplicate_keys(reader.fieldnames):
            return None
        if any("\n" in (name or "") for name in reader.fieldnames or ()):
            # A quote opened on the header line, so the column names ate the
            # rows below them: a panel whose hERG row read 10 nM came back with
            # no rows at all and every axis ``toxicology_not_reported``. The
            # header is not a row, so ``_spans_lines`` never sees it.
            return None
        if not any(_read_column(_column_key(name)) for name in reader.fieldnames or ()):
            # The header holds nothing this reads. Returning the rows instead
            # reported "no usable nM reading for this check" for a file whose
            # only fault was its delimiter, and sent the reader to add a column
            # that was already there.
            return None
        found: list[dict[str, str]] = []
        for row in reader:
            if _surplus(row.get(_EXTRA)):
                # Every field after the first unquoted separator is one column
                # left of where it belongs, so one protein is judged on
                # another's number. A misaligned row is not partly usable.
                return None
            if _spans_lines(row):
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


def _spans_lines(row: dict[str, object]) -> bool:
    """Whether a cell of this row swallowed the end of its row.

    ``_table_lines`` hands the reader one physical line per row, so a cell can
    hold a line break only where a quote opened and ``csv`` read on past the
    end of that row, taking the rows it crossed with it. ``csv`` reports none
    of this -- it returns what it parsed, and an open quote above
    ``cyp3a4,20`` left hERG unparsed, dropped a 20 nM cyp3a4 hit entirely and
    recorded cyp3a4 as a protein the table never reported. A quote that opens
    and later closes loses the rows in between the same way, which is why this
    is not a check on the last row alone.

    A cell that spans lines and closes cleanly is refused with them: the parsed
    rows do not say which of the two happened, and a panel of concentrations,
    units and protein names has no cell that spans lines either way. The
    refusal is loud and names the file, which is this module's stance on a
    table it cannot read in full.
    """
    for value in row.values():
        if isinstance(value, str):
            if "\n" in value:
                return True
            continue
        if isinstance(value, list) and any(isinstance(item, str) and "\n" in item for item in value):
            return True
    return False


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
        if not key or not _read_column(key):
            continue
        if key in seen:
            return True
        seen.add(key)
    return False


def _collect(rows: list[dict[str, str]]) -> tuple[dict[str, _AxisCells], bool]:
    """Cells per axis, and whether a concentration was left unattributed."""
    found: dict[str, _AxisCells] = {}
    unattributed = False
    for row in rows:
        unit = _unit_kind(row)
        relation = _relation_kind(row)
        named, labelled, several = _row_axis(row)
        for column in _VALUE_COLUMNS:
            if column not in row:
                continue
            if named is not None:
                _add(found, named, row[column], column, _unit_read(column in _VALUE_NM_COLUMNS, unit), relation)
            elif (several or not labelled) and _parse_cell(row[column])[0] != "absent":
                # A number under a value column that no one protein owns: a row
                # naming no protein at all, or one naming two of the panel's own
                # proteins. A row naming a protein outside the panel is a
                # different thing and is dropped on purpose.
                unattributed = True
        for key, value in row.items():
            if key in _AXIS_COLUMNS or key in _VALUE_COLUMNS or key in _UNIT_COLUMNS or key in _RELATION_COLUMNS:
                continue
            axis, declared_nm = _axis_and_unit(key)
            if axis is not None:
                _add(found, axis, value, key, _unit_read(declared_nm, unit), relation)
    return found, unattributed


def _unit_read(declares_nm: bool, unit: str) -> str:
    """The unit a cell is read under, given whether its column names nM itself.

    A column whose own name says nM is read as the row leaves it. A column
    named only for the panel protein, or only for the analyte, states no unit,
    so nothing in the table says its numbers are nM unless the row does: read
    as nM anyway, a micromolar export turned a clean 50 uM hERG reading into a
    50 nM hit. The module's rule is that a reading whose unit is not nM is
    never rescaled onto the cut, and ``other`` is how a cell is refused. One
    rule for both column shapes, because they are one rule.
    """
    if declares_nm or unit != "absent":
        return unit
    return "other"


def _row_axis(row: dict[str, str]) -> tuple[str | None, bool, bool]:
    """The panel axis this long-form row names, whether it named anything, and
    whether it named more than one of the panel's proteins.

    Every axis column is read, not just the first filled one. A table that
    writes the gene in ``protein`` and the panel id in ``panel`` stopped at the
    gene and was discarded whole, although the column holding the documented id
    was right beside it.

    The second value separates "this row names a protein outside the panel",
    which is dropped on purpose, from "this row names nothing", whose
    concentration cannot be filed at all.

    The third is a row naming two different panel proteins -- hERG in
    ``protein`` beside NAV1.5 in ``panel``. Returning the first one filed the
    concentration under a protein the same row contradicts, and a 50 per cent
    chance of a hit on the wrong name is the error this module refuses a
    misaligned row for. Two columns naming the same protein are one name.
    """
    labelled = False
    named: list[str] = []
    for column in _AXIS_COLUMNS:
        text = row.get(column, "").strip()
        if not text:
            continue
        labelled = True
        key = _axis_name(_column_key(text))
        if key in _AXES and key not in named:
            named.append(key)
    if len(named) > 1:
        return None, labelled, True
    return (named[0] if named else None), labelled, False


def _axis_name(key: str) -> str:
    """One spelling per panel protein, after ``_column_key`` has normalised it."""
    return _AXIS_ALIASES.get(key, key)


def _axis_from_column(key: str) -> str | None:
    axis, _declared = _axis_and_unit(key)
    return axis


def _axis_and_unit(key: str) -> tuple[str | None, bool]:
    """The panel axis this column names, and whether the name itself says nM.

    ``herg_ic50_nm`` names its unit; ``herg`` names only the protein and says
    nothing about what its numbers are in.
    """
    for suffix in _VALUE_SUFFIXES:
        if key.endswith(suffix):
            stem = _axis_name(key[: -len(suffix)])
            return (stem if stem in _AXES else None), True
    name = _axis_name(key)
    if name in _AXES:
        return name, False
    return None, False


def _unit_kind(row: dict[str, str]) -> str:
    """``absent`` when the row names no unit, ``nm`` when every unit it names
    says nM, else ``other``.

    A blank unit cell leaves a column whose own name says nM as nM. A column
    named only for the panel protein says nothing about its unit, and
    ``_collect`` refuses it when the row does not either. A declared micromolar
    or percent cell is ``other`` and is not converted onto the nM cut.

    Every unit column is read, not just the first one holding anything. A
    ChEMBL export carries ``units`` for the number it was given and
    ``standard_units`` for the number it normalised, and the two disagree by
    design -- 15 uM beside 15000 nM. Stopping at the first of them read a
    declared micromolar table as nanomolar whenever the nanomolar spelling sat
    in the earlier column, and an ``n/a`` in the earlier column hid the later
    column's uM altogether, which is the thousandfold error this rule exists to
    stop. Two unit cells that disagree are not a nanomolar reading in whichever
    one came first.
    """
    kinds = {_unit_token(row.get(column, "")) for column in _UNIT_COLUMNS}
    kinds.discard("absent")
    if not kinds:
        return "absent"
    return "nm" if kinds == {"nm"} else "other"


def _unit_token(raw: str) -> str:
    """What one unit cell says: ``absent``, ``nm``, or ``other``."""
    text = raw.strip()
    if not text or text.casefold() in _ABSENT:
        return "absent"
    token = text.casefold().replace("µ", "u").replace("μ", "u").replace(" ", "")
    return "nm" if token in _NM_UNITS else "other"


def _relation_kind(row: dict[str, str]) -> str:
    """The bound every relation column on this row agrees on, or ``""``.

    ``"other"`` is a relation this does not understand, such as ``~``: the
    number is then not a reading and not a bound either. Two relation columns
    naming different bounds are ``"other"`` as well. ChEMBL writes both
    ``relation`` and ``standard_relation``; taking the first of them put a
    ``>`` on a number the next column called ``<`` and cleared the axis on a
    bound that decides nothing, so a disagreement is refused like any other
    relation this cannot read.
    """
    bounds: set[str] = set()
    for column in _RELATION_COLUMNS:
        text = row.get(column, "").strip()
        if not text or text.casefold() in _ABSENT:
            continue
        bound = _bound_form(text)
        bounds.add(bound if bound is not None else "other")
    if not bounds:
        return ""
    if len(bounds) > 1:
        return "other"
    return bounds.pop()


def _bound_form(text: str) -> str | None:
    """One spelling per bound. ``=`` is no bound at all; an unknown token is None."""
    token = text.strip().replace(" ", "")
    for mark in _BOUNDS:
        if token == mark:
            return _BOUND_FORM.get(mark, mark)
    return None


def _bound_value(bound: str, number: float, cut: float) -> float | None:
    """A number standing for everything the bound admits, or None when it decides nothing.

    A bound names an interval, and ``cut`` is a point inside or outside it. It
    decides this axis only when every value the interval admits falls on one
    side: ``>10000`` is above a 10000 cut whatever the true figure is, and
    ``<100`` is below it. ``>=10000`` admits exactly 10000, which is a hit, and
    anything above it, which is a clear, so it decides nothing. ``cut`` is the
    axis's own cut, because the interval has to be compared with the number
    that will judge it and not with a panel-wide constant.
    """
    if bound in {">", ">="}:
        if number > cut:
            return number
        if bound == ">" and number == cut:
            return math.nextafter(number, math.inf)
        return None
    if bound in {"<", "<="} and number <= cut:
        return number
    return None


def _add(found: dict[str, _AxisCells], axis: str, raw: str, column: str, unit: str, relation: str = "") -> None:
    kind, number, bound = _parse_cell(raw)
    if kind == "absent":
        return
    if kind == "nm" and unit == "other":
        kind, number = "not_nm", None
    cells = found.setdefault(axis, _AxisCells())
    if kind == "nm" and relation == "other" and not bound:
        # A relation this cannot read -- ``~`` -- leaves a BARE number that is
        # in nM and is still not a reading, because nothing says how far from
        # it the truth may sit. Recorded as ``reading_not_nm`` it sent the
        # owner to the unit column, where nothing was wrong. It is the same
        # undecided case as a bound that straddles the cut, and that blocker's
        # remedy on the board already asks for a measured concentration or a
        # bound clear of the cut.
        #
        # ``not bound`` is the whole of the condition, and it was missing. A
        # cell that carries its own bound is settled by the rule below, which
        # says the in-cell bound outranks the relation column -- and it does,
        # for every relation this CAN read. Without the guard, a relation
        # carrying no directional claim at all destroyed a cell that a flatly
        # contradicting one did not: ``<100`` beside ``>`` is a hit at 100 nM,
        # and ``<100`` beside ``~`` was deleted outright. It cost readings in
        # both directions -- a bound 100-fold inside the cut lost its hit, and
        # a bound 5-fold clear of it lost its clear -- which is the one thing
        # this module's own docstring forbids: an unreadable cell may not
        # delete a reading the table made.
        cells.undecided = True
        return
    if kind == "nm" and number is not None:
        # A bound in the cell wins over a relation column, because it sits on
        # the number it qualifies; a row carrying both and disagreeing is the
        # cell's to settle.
        mark = bound or relation
        if not mark:
            cells.values.append((number, number, ""))
            return
        judged = _bound_value(mark, number, panel_cut(axis))
        if judged is None:
            cells.undecided = True
            return
        cells.values.append((judged, number, mark))
        return
    if kind == "unparsed":
        cells.unparsed = True
        if column not in cells.unparsed_columns:
            cells.unparsed_columns.append(column)
        return
    cells.not_nm = True
    if number is not None and cells.bad is None:
        cells.bad = number


def _parse_cell(raw: str) -> tuple[str, float | None, str]:
    """``nm``, ``absent``, ``unparsed`` or ``not_nm``, the number, and any bound in the cell.

    ``>10000`` and ``<100`` are what an assay export writes for a compound with
    no activity and for one that saturated the assay. Handing the whole cell to
    ``float`` refused both, which dropped a measured hit as readily as a clear.
    """
    text = raw.strip()
    if text.casefold() in _ABSENT:
        return "absent", None, ""
    bound = ""
    for mark in _BOUNDS:
        if text.startswith(mark):
            form = _BOUND_FORM.get(mark, mark)
            bound = form
            text = text[len(mark) :].strip()
            break
    if not text or text.casefold() in _ABSENT:
        return "unparsed", None, ""
    try:
        number = float(text)
    except ValueError:
        return "unparsed", None, ""
    if not math.isfinite(number) or number < 0:
        stored = number if math.isfinite(number) else None
        return "not_nm", stored, bound
    return "nm", number, bound
