"""What a design's mmCIF actually holds, however the file spells it.

An mmCIF category with one row is written as plain ``_name.item value`` lines;
the same category with two rows is written as a ``loop_``. Both forms are
correct and this repository's own writer emits both -- a two-chain binder
complex writes ``_entity_poly`` as a loop, and a monomer, an inverse-folding
output, an enzyme design, a motif scaffold or a small-molecule binder writes the
same category flat, because each has exactly one polymer entity.

A reader that handles only the loop form therefore sees **nothing** in every
single-polymer design, which is most of them. That is not an abstract risk:
``screen._polymer_sequences`` read loops only, and returned an empty mapping for
every monomer, inverse-folding output, enzyme design, motif scaffold and
small-molecule binder in ``runs/`` and ``disease_runs/`` -- so the screen
reported ``sequence_absent`` for a sequence the file was holding. It reads both
spellings now, and ``test_af3server_request`` keeps the two readers in step.

So this module reads both forms, and it reads the whole entity list rather than
the polymer table alone. The second part matters as much as the first. A
small-molecule design's ligand has no row in ``_entity_poly`` at all -- it is a
``non-polymer`` entity whose atoms are only in ``_atom_site`` -- and a request
built from the polymer table would have dropped the ligand and folded the
protein alone. That request does not fail. It answers a different question.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from rfproteina.disease.screen import _cif_tokens

#: Categories read here. Everything else in the file is skipped without being
#: tokenised, so an atom site is only walked when a non-polymer entity is found.
_ENTITY: Final = "_entity"
_ENTITY_POLY: Final = "_entity_poly"
_ATOM_SITE: Final = "_atom_site"

#: ``_entity.type`` values that carry no coordinates worth sending. Water is
#: never part of an AlphaFold Server request; the server adds none and takes none.
WATER_KINDS: Final = frozenset({"water"})

#: Hydrogen is not a token. AlphaFold 3 does not model hydrogens, and a file
#: that happens to carry them would otherwise inflate a ligand's token count.
_HYDROGEN: Final = frozenset({"H", "D"})


@dataclass(frozen=True)
class CifCategory:
    """One mmCIF category's rows, from either spelling.

    ``ragged`` marks a loop whose values did not fill its columns. That is a
    file that did not parse, which is a different fact from a category that is
    absent -- and a different thing to go and fix -- so it is carried rather
    than collapsed into "no rows".
    """

    name: str
    rows: tuple[dict[str, str], ...] = ()
    ragged: bool = False

    def column(self, item: str) -> list[str]:
        """One item's value from every row, in file order. Absent reads as empty."""
        return [_bare(row.get(f"{self.name}.{item}", "")) for row in self.rows]


@dataclass(frozen=True)
class Entity:
    """One entity of a structure, polymer or not."""

    entity_id: str
    #: ``_entity.type``, lower-cased: ``polymer``, ``non-polymer``, ``water``,
    #: ``branched``. Empty when the file names no ``_entity`` table.
    kind: str
    #: ``_entity_poly.type``, lower-cased. Empty for a non-polymer.
    polymer_type: str = ""
    #: Strand ids this entity is instantiated as.
    strands: tuple[str, ...] = ()
    #: The canonical one-letter sequence, for a polymer.
    sequence: str = ""
    #: Chemical component ids, for a non-polymer. Usually exactly one.
    components: tuple[str, ...] = ()
    #: Heavy atoms per copy, counted from ``_atom_site``. Zero when unknown.
    atoms: int = 0

    @property
    def is_polymer(self) -> bool:
        return bool(self.sequence)


def read_categories(text: str, wanted: tuple[str, ...]) -> dict[str, CifCategory]:
    """The named categories, parsed from whichever spelling the file used.

    ``wanted`` names categories without their trailing dot, as in ``_entity``.
    A name is matched exactly, so asking for ``_entity`` does not also collect
    ``_entity_poly`` or ``_entity_poly_seq``.

    Rows are keyed by the full item name (``_entity.type``), matching
    ``screen._cif_loops`` so the two readers can be compared row for row.
    """
    collected: dict[str, list[dict[str, str]]] = {name: [] for name in wanted}
    ragged: set[str] = set()
    flat: dict[str, dict[str, str]] = {name: {} for name in wanted}

    lines = text.splitlines()
    index, count = 0, len(lines)
    while index < count:
        stripped = lines[index].strip()
        if stripped == "loop_":
            index = _read_loop(lines, index + 1, wanted, collected, ragged)
            continue
        if stripped.startswith("_"):
            index = _read_flat(lines, index, wanted, flat)
            continue
        index += 1

    found: dict[str, CifCategory] = {}
    for name in wanted:
        rows = list(collected[name])
        # A category spelled flat contributes one row. A file would not sanely
        # spell the same category both ways, but if it did, both are kept rather
        # than one silently winning.
        if flat[name]:
            rows.append(flat[name])
        if rows or name in ragged:
            found[name] = CifCategory(name, tuple(rows), name in ragged)
    return found


def read_entities(text: str) -> list[Entity] | str:
    """Every entity of the structure, or a blocker name.

    The ``_entity`` table is **required**. Without it there is no way to know
    that the polymer rows are the whole structure, and the one error this must
    not make is reporting a complete request for a structure whose ligand it
    never saw. An absent entity table is therefore refused by name rather than
    treated as "the polymers are everything".
    """
    categories = read_categories(text, (_ENTITY, _ENTITY_POLY))

    entity = categories.get(_ENTITY)
    if entity is None or not entity.rows:
        return "entity_table_absent"
    if entity.ragged:
        return "entity_table_unreadable"
    kinds: dict[str, str] = {}
    for row in entity.rows:
        name = _bare(row.get(f"{_ENTITY}.id", ""))
        if name:
            kinds[name] = _bare(row.get(f"{_ENTITY}.type", "")).lower()
    if not kinds:
        return "entity_table_absent"

    poly = categories.get(_ENTITY_POLY)
    if poly is not None and poly.ragged:
        return "entity_poly_unreadable"
    polymers: dict[str, tuple[str, tuple[str, ...], str]] = {}
    for row in poly.rows if poly is not None else ():
        name = _bare(row.get(f"{_ENTITY_POLY}.entity_id", ""))
        sequence = _letters(row.get(f"{_ENTITY_POLY}.pdbx_seq_one_letter_code_can", "")) or _letters(
            row.get(f"{_ENTITY_POLY}.pdbx_seq_one_letter_code", "")
        )
        if not name or not sequence or sequence in {".", "?"}:
            continue
        polymers[name] = (
            _bare(row.get(f"{_ENTITY_POLY}.type", "")).lower(),
            tuple(_strands(row.get(f"{_ENTITY_POLY}.pdbx_strand_id", ""))),
            sequence,
        )

    # Only walked when something is not a polymer, which is the only case that
    # needs it. An atom site is most of the file's bytes.
    others = [name for name, kind in kinds.items() if name not in polymers and kind not in WATER_KINDS]
    atoms = _atoms_by_entity(text, frozenset(others)) if others else {}

    built: list[Entity] = []
    for name in sorted(kinds, key=_sort_key):
        kind = kinds[name]
        if name in polymers:
            polymer_type, strands, sequence = polymers[name]
            built.append(Entity(name, kind, polymer_type, strands, sequence))
            continue
        if kind in WATER_KINDS:
            continue
        components, atom_count, strands = atoms.get(name, ((), 0, ()))
        built.append(Entity(name, kind, components=components, atoms=atom_count, strands=strands))
    return built


def _atoms_by_entity(text: str, wanted: frozenset[str]) -> dict[str, tuple[tuple[str, ...], int, tuple[str, ...]]]:
    """Components, heavy-atom count and strands per entity, from ``_atom_site``.

    Read by column index rather than through a row dict: an annotated CIF's atom
    site carries upwards of a hundred condition columns, and building a dict per
    atom to read three of them is most of the cost of opening the file.

    The count is **per copy**: an entity instantiated as two strands has its
    atoms divided by the number of strands, because that is what the token
    budget charges for one copy.
    """
    lines = text.splitlines()
    index, count = 0, len(lines)
    components: dict[str, dict[str, None]] = {}
    strands: dict[str, dict[str, None]] = {}
    heavy: dict[str, int] = {}

    while index < count:
        if lines[index].strip() != "loop_":
            index += 1
            continue
        index += 1
        columns: list[str] = []
        while index < count and lines[index].lstrip().startswith("_"):
            columns.append(lines[index].strip())
            index += 1
        if not columns or not columns[0].startswith(f"{_ATOM_SITE}."):
            index = _skip_values(lines, index)
            continue
        places = {name: position for position, name in enumerate(columns)}
        entity_at = places.get(f"{_ATOM_SITE}.label_entity_id")
        comp_at = places.get(f"{_ATOM_SITE}.label_comp_id", places.get(f"{_ATOM_SITE}.auth_comp_id"))
        symbol_at = places.get(f"{_ATOM_SITE}.type_symbol")
        asym_at = places.get(f"{_ATOM_SITE}.label_asym_id", places.get(f"{_ATOM_SITE}.auth_asym_id"))
        if entity_at is None:
            index = _skip_values(lines, index)
            continue
        width = len(columns)
        values: list[str] = []
        while index < count:
            stripped = lines[index].strip()
            if stripped in {"loop_", "#"} or stripped.startswith(("data_", "_")):
                break
            if not stripped:
                index += 1
                continue
            values.extend(_cif_tokens(stripped))
            while len(values) >= width:
                row, values = values[:width], values[width:]
                name = _bare(row[entity_at])
                if name not in wanted:
                    continue
                if comp_at is not None:
                    components.setdefault(name, {}).setdefault(_bare(row[comp_at]), None)
                if asym_at is not None:
                    strands.setdefault(name, {}).setdefault(_bare(row[asym_at]), None)
                symbol = _bare(row[symbol_at]).upper() if symbol_at is not None else ""
                if symbol not in _HYDROGEN:
                    heavy[name] = heavy.get(name, 0) + 1
            index += 1

    found: dict[str, tuple[tuple[str, ...], int, tuple[str, ...]]] = {}
    for name in wanted:
        copies = max(len(strands.get(name, {})), 1)
        found[name] = (
            tuple(key for key in components.get(name, {}) if key),
            heavy.get(name, 0) // copies,
            tuple(key for key in strands.get(name, {}) if key),
        )
    return found


def _read_loop(
    lines: list[str],
    index: int,
    wanted: tuple[str, ...],
    collected: dict[str, list[dict[str, str]]],
    ragged: set[str],
) -> int:
    """Consume one loop, keeping it only when its category is wanted."""
    count = len(lines)
    columns: list[str] = []
    while index < count and lines[index].lstrip().startswith("_"):
        columns.append(lines[index].strip())
        index += 1
    name = _category_of(columns, wanted)
    if name is None:
        return _skip_values(lines, index)
    values, index = _read_values(lines, index)
    width = len(columns)
    if width and values and len(values) % width == 0:
        for start in range(0, len(values), width):
            collected[name].append(dict(zip(columns, values[start : start + width], strict=True)))
    elif width and values:
        ragged.add(name)
    return index


def _read_flat(lines: list[str], index: int, wanted: tuple[str, ...], flat: dict[str, dict[str, str]]) -> int:
    """Consume one ``_name.item value`` line, and its value's continuation.

    The value may sit on the same line, or -- when the item name is alone on its
    line -- on the following line, as a token or as a semicolon-delimited block.
    The monomer designs in this repository are written the third way, with the
    sequence split across lines inside such a block.
    """
    line = lines[index]
    stripped = line.strip()
    item, _, remainder = stripped.partition(" ")
    category = item.rpartition(".")[0]
    index += 1

    if remainder.strip():
        if category in wanted:
            tokens = _cif_tokens(remainder.strip())
            flat[category][item] = tokens[0] if tokens else ""
        return index

    # The value is on the following lines.
    count = len(lines)
    while index < count and not lines[index].strip():
        index += 1
    if index >= count:
        return index
    if lines[index].startswith(";"):
        parts: list[str] = []
        first = lines[index][1:]
        if first:
            parts.append(first)
        index += 1
        while index < count and lines[index].strip() != ";":
            parts.append(lines[index])
            index += 1
        if index < count and lines[index].strip() == ";":
            index += 1
        if category in wanted:
            flat[category][item] = "".join(parts)
        return index
    following = lines[index].strip()
    if following in {"loop_", "#"} or following.startswith(("data_", "_")):
        # The item had no value at all. Recorded as empty rather than consuming
        # the next category's first line as this item's value.
        return index
    if category in wanted:
        tokens = _cif_tokens(following)
        flat[category][item] = tokens[0] if tokens else ""
    return index + 1


def _read_values(lines: list[str], index: int) -> tuple[list[str], int]:
    """Every value of a loop body, with semicolon blocks as single cells."""
    count = len(lines)
    values: list[str] = []
    while index < count:
        stripped = lines[index].strip()
        if stripped in {"loop_", "#"} or stripped.startswith(("data_", "_")):
            break
        if not stripped:
            index += 1
            continue
        if lines[index].startswith(";"):
            parts: list[str] = []
            first = lines[index][1:]
            if first:
                parts.append(first)
            index += 1
            while index < count and lines[index].strip() != ";":
                parts.append(lines[index])
                index += 1
            if index < count and lines[index].strip() == ";":
                index += 1
            values.append("".join(parts))
            continue
        values.extend(_cif_tokens(stripped))
        index += 1
    return values, index


def _skip_values(lines: list[str], index: int) -> int:
    """Past a loop body, without tokenising it."""
    count = len(lines)
    while index < count:
        stripped = lines[index].strip()
        if stripped in {"loop_", "#"} or stripped.startswith(("data_", "_")):
            break
        if lines[index].startswith(";"):
            index += 1
            while index < count and lines[index].strip() != ";":
                index += 1
            if index < count and lines[index].strip() == ";":
                index += 1
            continue
        index += 1
    return index


def _category_of(columns: list[str], wanted: tuple[str, ...]) -> str | None:
    """The wanted category this loop's columns name, matched exactly.

    Exactly, so a request for ``_entity`` does not collect ``_entity_poly``:
    both start with ``_entity`` and only one of them is ``_entity``.
    """
    for column in columns:
        category = column.rpartition(".")[0]
        if category in wanted:
            return category
    return None


def _letters(sequence: str) -> str:
    return "".join(sequence.split()).upper()


def _strands(raw: str) -> list[str]:
    if raw.strip() in {"", ".", "?"}:
        return []
    return [part.strip() for part in raw.split(",") if part.strip() not in {"", ".", "?"}]


def _bare(value: str) -> str:
    text = value.strip()
    return "" if text in {".", "?"} else text


def _sort_key(name: str) -> tuple[int, int, str]:
    """Entity ids in numeric order where they are numbers, else by name."""
    return (0, int(name), "") if name.isdigit() else (1, 0, name)
