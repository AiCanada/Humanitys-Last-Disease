#!/usr/bin/env python3
"""Build a disease class-file directory from your own data.

A disease training run reads structure files whose *name* declares the binder kind, one liability
axis and that axis's outcome. This script takes a manifest of your data, judges every row with the
project's own rules, and writes a directory of correctly named symlinks or copies that
``disease_class_data.directory`` can point at.

The whole point is that a wrong label here is invisible later, so the script is deliberately
suspicious of its input. Nothing is written unless every row validates. A manifest that
contradicts itself is refused. A ragged CSV row is refused rather than silently truncated. Two
headers that mean the same field are refused rather than one quietly winning. Every emitted name
is parsed back before it is written, and no existing file is ever replaced or unlinked.

Run inside the project venv (``source .venv/bin/activate``).

Manifest formats: ``.csv``, ``.tsv``, ``.json`` (a list of objects) or ``.jsonl``. Recognised
columns, case-insensitive, first match wins:

===========  ====================================  =========================================
field        column names                          notes
===========  ====================================  =========================================
structure    path, file, filename, structure, cif  required; resolved against --root
kind         kind, binder, binder_kind             or pass --kind for the whole manifest
axis         axis, assay, target                   or pass --axis, or --bucket-axis
outcome      state, outcome, result, call          clean / clear / toxic / hit
             value_nm, nm, ic50_nm, ki_nm, value   a ligand reading in nM, judged at 10 uM
             label, bucket, class                  high_affinity_clean / high_affinity_toxic
===========  ====================================  =========================================

Quote any value containing the delimiter. ``48,000`` in a CSV is two fields, not a number, and
the script refuses the row rather than reading it as 48.

Usage::

    scripts/build_disease_class_files.py readings.csv --out-dir data/disease_classes
    scripts/build_disease_class_files.py buckets.csv --out-dir out --kind ligand --bucket-axis herg
    scripts/build_disease_class_files.py readings.csv --out-dir out --dry-run
"""

import argparse
import csv
import json
import os
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from rfproteina.disease.classes import load_class_files
from rfproteina.disease.combinations import (
    AXES_BY_KIND,
    CLEAR,
    HIT,
    axis_kinds,
    judge_combination,
    ligand_states,
    parse_class_filename,
)
from rfproteina.disease.tokens import BAD_CLASS, GOOD_CLASS
from rfproteina.utils.structure_files import is_structure_file, structure_file_stem

STRUCTURE_COLUMNS = ("path", "file", "filename", "structure", "cif")
KIND_COLUMNS = ("kind", "binder", "binder_kind")
AXIS_COLUMNS = ("axis", "assay", "target")
STATE_COLUMNS = ("state", "outcome", "result", "call")
VALUE_COLUMNS = ("value_nm", "nm", "ic50_nm", "ki_nm", "value")
LABEL_COLUMNS = ("label", "bucket", "class")

#: The words a state column may use, matching what the filename parser accepts.
STATE_WORDS = {"clean": CLEAR, "clear": CLEAR, "toxic": HIT, "hit": HIT}

#: The word a generated filename uses for each state.
STATE_SUFFIX = {CLEAR: "clean", HIT: "toxic"}

#: Where the emitted names are tied back to their sources, since the names do not carry it.
PROVENANCE_NAME = "class_files_provenance.json"

#: csv.DictReader parks fields beyond the header here, which means the row is ragged.
_OVERFLOW = "__fields_beyond_the_header__"

#: A spreadsheet export glues this to the first header name, and it is not whitespace.
_BOM = "﻿"


@dataclass(frozen=True)
class Entry:
    """One validated row: where the structure is and what its name will say."""

    source: Path
    kind: str
    axis: str
    state: str
    row: int


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=_axis_help(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("manifest", type=Path, help="CSV, TSV, JSON or JSONL describing your data.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory to build. Must be empty or new.")
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Base for relative structure paths. Defaults to the manifest's own directory.",
    )
    parser.add_argument("--kind", default=None, help="Binder kind for every row, when the manifest has no kind column.")
    parser.add_argument(
        "--axis",
        default=None,
        help="Liability axis for every row, when the manifest has no axis column.",
    )
    parser.add_argument("--bucket-axis", default=None, help="Axis a coarse clean/toxic label refers to.")
    parser.add_argument("--copy", action="store_true", help="Copy the structures instead of linking them.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and report without writing anything.")
    parser.add_argument("--skip-invalid", action="store_true", help="Build from the valid rows and list the rest.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Add to a directory that already has files in it. Existing files are never replaced.",
    )
    args = parser.parse_args()

    try:
        rows = _read_manifest(args.manifest)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Could not read {args.manifest}: {exc}", file=sys.stderr)
        return 2
    if not rows:
        print(f"{args.manifest} has no rows.", file=sys.stderr)
        return 2

    root = args.root or args.manifest.parent
    entries: list[Entry] = []
    problems: list[str] = []
    for number, row in enumerate(rows, start=1):
        try:
            entries.append(_entry(row, number, root, args))
        except ValueError as exc:
            problems.append(f"  row {number}: {exc}")

    if problems and not args.skip_invalid:
        print(f"{len(problems)} of {len(rows)} rows could not be used, so nothing was written:", file=sys.stderr)
        print("\n".join(problems), file=sys.stderr)
        print("\nFix the manifest, or pass --skip-invalid to build from the rest.", file=sys.stderr)
        return 1
    if problems:
        print(f"Skipping {len(problems)} of {len(rows)} rows:")
        print("\n".join(problems))
    if not entries:
        print("No usable rows.", file=sys.stderr)
        return 1

    # One structure cannot be both clear and a hit on the same axis, and guessing which row meant
    # it would be guessing at the training label.
    clashes = _contradictions(entries)
    if clashes:
        print("The manifest contradicts itself, so nothing was written:", file=sys.stderr)
        print("\n".join(f"  {clash}" for clash in clashes), file=sys.stderr)
        return 1

    try:
        _check_sources_outside(entries, args.out_dir)
    except ValueError as exc:
        print(f"{exc}", file=sys.stderr)
        return 2

    names = _names(entries, args.out_dir)
    _report(entries, names, args.out_dir)

    if args.dry_run:
        print("\nDry run: nothing was written.")
        return 0

    try:
        _prepare_out_dir(args.out_dir, args.force)
        created = _write(entries, names, args.out_dir, copy=args.copy)
    except ValueError as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 2
    _write_provenance(entries, names, args.out_dir, args.manifest)
    print(f"\nWrote {len(created)} class file(s) to {args.out_dir}")

    loaded = load_class_files(args.out_dir)
    usable = [item for item in loaded if not item.coarse and item.binder_kind and item.axis and item.state]
    print(f"The training loader reads {len(loaded)} class file(s), {len(usable)} of them usable as a contrast.")
    if len(usable) < len(created):
        print("WARNING: fewer than were written. Check the directory before training.", file=sys.stderr)
        return 1
    print(f"\nPoint the run at it:\n  disease_class_data.directory={args.out_dir}")
    return 0


def _entry(row: dict, number: int, root: Path, args: argparse.Namespace) -> Entry:
    """Validate one manifest row into an Entry, or explain why it cannot be used."""
    fields = _normalize(row)
    source = _resolve_structure(fields, root)
    kind = _value(fields, KIND_COLUMNS) or args.kind
    if not kind:
        raise ValueError("no binder kind; add a kind column or pass --kind")
    axis = _value(fields, AXIS_COLUMNS) or args.axis or args.bucket_axis
    if not axis:
        raise ValueError("no liability axis; add an axis column or pass --axis")
    axis = axis.strip().casefold().replace("-", "_").replace(" ", "_")
    state = _state(fields, axis)

    # The project's own judge decides whether this axis belongs to this kind, and gives back the
    # canonical kind, so an alias such as small_molecule becomes ligand in the filename.
    combination = judge_combination(kind, {axis: state})
    return Entry(source=source, kind=combination.binder_kind, axis=axis, state=state, row=number)


def _normalize(row: dict) -> dict[str, str]:
    """Lower-case the headers of one row, refusing any two that mean the same field.

    A spreadsheet export glues a byte-order mark to the first header, and U+FEFF is not
    whitespace, so it is stripped explicitly rather than left to ``strip``.
    """
    if _OVERFLOW in row:
        raise ValueError("more fields than the header; quote any value that contains the delimiter")
    fields: dict[str, str] = {}
    for key, value in row.items():
        if key is None:
            continue
        name = str(key).strip().lstrip(_BOM).strip().casefold()
        if name in fields:
            raise ValueError(f"two columns both read as {name!r}; rename one so the row is unambiguous")
        if value is None:
            raise ValueError(f"fewer fields than the header; column {name!r} onwards is missing")
        fields[name] = value if isinstance(value, str) else str(value)
    return fields


def _resolve_structure(fields: dict[str, str], root: Path) -> Path:
    raw = _value(fields, STRUCTURE_COLUMNS)
    if not raw:
        raise ValueError(f"no structure path; expected one of {', '.join(STRUCTURE_COLUMNS)}")
    source = Path(raw)
    if not source.is_absolute():
        source = root / source
    if not source.is_file():
        raise ValueError(f"{source} is not a file")
    if is_structure_file(source, include_pdb=True) and not is_structure_file(source):
        raise ValueError(f"{source.name} is a PDB encoding, which the training dataset cannot load; convert it to CIF")
    if not is_structure_file(source):
        raise ValueError(f"{source.name} is not a structure the training dataset can load")
    return source


def _state(fields: dict[str, str], axis: str) -> str:
    """Read the outcome from whichever column the manifest uses."""
    word = _value(fields, STATE_COLUMNS)
    if word:
        key = word.strip().casefold()
        if key not in STATE_WORDS:
            raise ValueError(f"state {word!r} is not one of clean, clear, toxic, hit")
        return STATE_WORDS[key]

    reading = _value(fields, VALUE_COLUMNS)
    if reading:
        try:
            value = float(reading)
        except ValueError:
            raise ValueError(f"measurement {reading!r} is not a number") from None
        if axis not in AXES_BY_KIND.get("ligand", ()):
            raise ValueError(f"a nM reading only judges a ligand panel axis, and {axis} is not one")
        return ligand_states({axis: value})[axis]

    label = _value(fields, LABEL_COLUMNS)
    if label:
        key = label.strip().casefold()
        if key == GOOD_CLASS:
            return CLEAR
        if key == BAD_CLASS:
            return HIT
        if key in STATE_WORDS:
            return STATE_WORDS[key]
        raise ValueError(f"label {label!r} is not {GOOD_CLASS!r} or {BAD_CLASS!r}")

    raise ValueError("no outcome; add a state, nM reading or label column")


def _contradictions(entries: list[Entry]) -> list[str]:
    """Rows that give one structure opposite outcomes on the same kind and axis."""
    first: dict[tuple[Path, str, str], Entry] = {}
    clashes: list[str] = []
    for entry in entries:
        key = (entry.source.resolve(), entry.kind, entry.axis)
        earlier = first.get(key)
        if earlier is None:
            first[key] = entry
        elif earlier.state != entry.state:
            clashes.append(
                f"rows {earlier.row} and {entry.row} give {entry.source.name} opposite outcomes "
                f"on {entry.kind}/{entry.axis}"
            )
    return clashes


def _check_sources_outside(entries: list[Entry], out_dir: Path) -> None:
    """Refuse to build inside the user's own data, where a write could reach an original."""
    if not out_dir.exists():
        return
    resolved = out_dir.resolve()
    inside = sorted({str(entry.source) for entry in entries if resolved in entry.source.resolve().parents})
    if inside:
        raise ValueError(
            f"{len(inside)} source structure(s) are inside {out_dir}, so building there would mix the "
            f"output into your data: {inside[0]}. Choose an output directory outside your structures."
        )


def _names(entries: list[Entry], out_dir: Path) -> list[str]:
    """A unique, parseable filename per entry, checked by parsing it back.

    Numbering steps over anything already in the directory, so a second run with --force adds to
    it instead of colliding, and no existing file ever has to be replaced.
    """
    taken = {path.name for path in out_dir.iterdir()} if out_dir.is_dir() else set()
    used: Counter = Counter()
    names: list[str] = []
    for entry in entries:
        suffix = entry.source.name[len(structure_file_stem(entry.source)) :]
        base = f"{entry.kind}_{entry.axis}_{STATE_SUFFIX[entry.state]}"
        name = ""
        while not name or name in taken:
            used[base] += 1
            name = f"{base}_{used[base]:04d}{suffix}"
        parsed = parse_class_filename(name)
        if parsed is None or (parsed.binder_kind, parsed.axis, parsed.state) != (entry.kind, entry.axis, entry.state):
            raise SystemExit(f"Generated name {name!r} does not read back as {entry.kind}/{entry.axis}/{entry.state}.")
        taken.add(name)
        names.append(name)
    return names


def _report(entries: list[Entry], names: list[str], out_dir: Path) -> None:
    """Show the contrast the run would see, and warn about any one-sided axis."""
    counts: dict[tuple[str, str], Counter] = {}
    for entry in entries:
        counts.setdefault((entry.kind, entry.axis), Counter())[entry.state] += 1
    print(f"{len(entries)} row(s) judged, to be written as:")
    print(f"  {'kind':<9} {'axis':<26} {'clear':>6} {'hit':>6}")
    one_sided = []
    for (kind, axis), tally in sorted(counts.items()):
        clear, hit = tally[CLEAR], tally[HIT]
        print(f"  {kind:<9} {axis:<26} {clear:>6} {hit:>6}")
        if not clear or not hit:
            one_sided.append(f"{kind}/{axis}")
    print(f"  example name: {names[0]}  ->  {out_dir / names[0]}")
    if one_sided:
        print(
            "\nWARNING: these axes have only one side, so they teach no contrast: "
            + ", ".join(one_sided)
            + "\n  The model learns from pairs that differ in the label, so add the missing side."
        )


def _prepare_out_dir(out_dir: Path, force: bool) -> None:
    if out_dir.exists() and any(out_dir.iterdir()) and not force:
        raise ValueError(f"{out_dir} already has files in it. Use a fresh directory, or pass --force.")
    out_dir.mkdir(parents=True, exist_ok=True)


def _write(entries: list[Entry], names: list[str], out_dir: Path, *, copy: bool) -> list[Path]:
    """Link or copy each structure under its class name, all or nothing."""
    created: list[Path] = []
    try:
        for entry, name in zip(entries, names, strict=True):
            destination = out_dir / name
            if destination.exists() or destination.is_symlink():
                raise ValueError(f"{destination} already exists; refusing to replace it")
            if copy:
                shutil.copy2(entry.source, destination)
            else:
                try:
                    os.symlink(entry.source.resolve(), destination)
                except (OSError, NotImplementedError):
                    # Windows needs a privilege for symlinks; a copy is always available.
                    shutil.copy2(entry.source, destination)
            created.append(destination)
    except (OSError, ValueError) as exc:
        for path in created:
            path.unlink(missing_ok=True)
        raise ValueError(
            f"Writing stopped after {len(created)} file(s), which were removed, so the directory is "
            f"as it was: {exc}"
        ) from exc
    return created


def _write_provenance(entries: list[Entry], names: list[str], out_dir: Path, manifest: Path) -> None:
    """Record which source produced which class file, keeping any earlier run's record."""
    path = out_dir / PROVENANCE_NAME
    record: dict = {"runs": []}
    if path.is_file():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = None
        if isinstance(existing, dict) and isinstance(existing.get("runs"), list):
            record = existing
    record["runs"].append(
        {
            "manifest": str(manifest.resolve()),
            "files": [
                {
                    "class_file": name,
                    "source": str(entry.source.resolve()),
                    "manifest_row": entry.row,
                    "binder_kind": entry.kind,
                    "axis": entry.axis,
                    "state": entry.state,
                }
                for entry, name in zip(entries, names, strict=True)
            ],
        }
    )
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")


def _read_manifest(path: Path) -> list[dict]:
    suffix = path.suffix.casefold()
    # utf-8-sig, because a spreadsheet export starts with a byte-order mark that would otherwise
    # become part of the first header name and make that whole column invisible.
    text = path.read_text(encoding="utf-8-sig")
    if suffix == ".json":
        payload = json.loads(text)
        if not isinstance(payload, list):
            raise ValueError("a JSON manifest must be a list of objects")
        return [_require_mapping(row, number) for number, row in enumerate(payload, start=1)]
    if suffix == ".jsonl":
        lines = [line for line in text.splitlines() if line.strip()]
        return [_require_mapping(json.loads(line), number) for number, line in enumerate(lines, start=1)]
    delimiter = "\t" if suffix == ".tsv" else ","
    return list(csv.DictReader(text.splitlines(), delimiter=delimiter, restkey=_OVERFLOW))


def _require_mapping(row: object, number: int) -> dict:
    if not isinstance(row, dict):
        raise ValueError(f"row {number} is {type(row).__name__}, not an object with named fields")
    return dict(row)


def _value(fields: dict[str, str], columns: tuple[str, ...]) -> str | None:
    """First non-empty value among the recognised column names."""
    for column in columns:
        text = fields.get(column, "").strip()
        if text:
            return text
    return None


def _axis_help() -> str:
    lines = ["Axes by binder kind:"]
    lines += [f"  {kind}: {', '.join(axes)}" for kind, axes in AXES_BY_KIND.items() if axes]
    shared = sorted({axis for axis in AXES_BY_KIND["protein"] if len(axis_kinds(axis)) > 1})
    lines.append(f"  shared, so the kind must be given: {', '.join(shared)}")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
