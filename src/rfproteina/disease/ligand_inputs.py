"""Read molecular nominations shared by post-generation small-molecule screens."""

from __future__ import annotations

import csv
import json
import os
import stat
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from io import StringIO
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rdkit import Chem


def read_regular(path: Path, limit: int, role: str, error: type[ValueError]) -> bytes:
    """Read a bounded snapshot of a regular file, preserving the caller's error vocabulary."""
    try:
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode):
            raise error(f"{role}_not_regular")
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0)
        with os.fdopen(os.open(path, flags), "rb") as stream:
            opened = os.fstat(stream.fileno())
            if not stat.S_ISREG(opened.st_mode) or not os.path.samestat(info, opened):
                raise error(f"{role}_changed")
            if opened.st_size > limit:
                raise error(f"{role}_size_limit")
            data = stream.read(limit + 1)
            after = os.fstat(stream.fileno())
        if len(data) != opened.st_size or (opened.st_mtime_ns, opened.st_ctime_ns) != (
            after.st_mtime_ns, after.st_ctime_ns
        ):
            raise error(f"{role}_changed")
    except FileNotFoundError as exc:
        raise error(f"{role}_missing") from exc
    except OSError as exc:
        raise error(f"{role}_unreadable") from exc
    return data


def nominated_rows(rows: Iterable[Sequence[str]], candidate: str | None) -> Iterator[Sequence[str]]:
    """Keep all rows for an explicit nomination so duplicate IDs still fail in the scientific reader."""
    return (row for row in rows if candidate is None or (row and row[0] == candidate))


def manifest_options(directory: Path, name: str) -> tuple[dict, bool]:
    """Read a molecular screen's settings and preserve whether the experiment requested it."""
    path = directory / name
    if not path.exists() and not path.is_symlink():
        sdf = directory / "ligand_candidates.sdf"
        return {"input": "sdf" if sdf.exists() or sdf.is_symlink() else "smiles"}, False
    if path.is_symlink() or not path.is_file():
        raise ValueError("invalid_molecular_manifest")
    limit = 4 * 1024 * 1024
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError("molecular_manifest_too_large")

    def unique(pairs: list) -> dict:
        result = dict(pairs)
        if len(result) != len(pairs):
            raise ValueError("duplicate_setting")
        return result

    options = json.loads(raw.decode("utf-8-sig"), object_pairs_hook=unique)
    if not isinstance(options, dict) or options.get("input", "smiles") not in ("sdf", "smiles"):
        raise ValueError("invalid_molecular_manifest")
    return options, True


def nominated_inputs(
    directory: Path, mode: str, candidate: str | None, *, max_bytes: int | None = None
) -> tuple[str, list[Sequence[str]], str | None]:
    """Read the selected molecular source, with candidate attribution and an optional planning budget."""
    if mode == "sdf":
        scope, path = "sdf", directory / "ligand_candidates.sdf"
    else:
        table = directory / "ligand_candidates.tsv"
        scope = "candidate_table" if table.exists() or table.is_symlink() else "run_ligand"
        path = table if scope == "candidate_table" else directory / "ligand_smiles.txt"
    if path.is_symlink() or not path.is_file():
        return scope, [], "sdf_input_unreadable" if mode == "sdf" else "ligand_input_missing"
    if candidate is not None and scope == "run_ligand":
        return scope, [], "candidate_requires_named_nomination"
    if max_bytes is not None and path.stat().st_size > max_bytes:
        return scope, [], "ligand_input_exceeds_interactive_size"
    if mode == "sdf":
        try:
            return scope, list(nominated_rows(read_sdf_candidates(path), candidate)), None
        except OSError:
            return scope, [], "sdf_input_unreadable"
    scope, rows, blocker = read_smiles_candidates(directory)
    if blocker or rows is None:
        return scope, [], blocker or "ligand_input_missing"
    if candidate is not None and scope != "candidate_table":
        return scope, [], "candidate_requires_named_nomination"
    return scope, list(nominated_rows(rows, candidate)), None


def read_run_smiles(path: Path) -> str:
    """Read exactly one run-level .smi record, optionally followed by its title."""
    with path.open(encoding="utf-8-sig") as stream:
        return _one_run_smiles(stream)


def _one_run_smiles(stream: Iterable[str]) -> str:
    lines = (s.strip() for s in stream if s.strip())
    first = next(lines, None)
    if first is None or next(lines, None) is not None:
        raise ValueError("expected_one_run_ligand")
    fields = first.split(maxsplit=1)
    if len(fields) > 1 and fields[1].startswith("|"):
        raise ValueError("unsupported_extended_smiles")
    return fields[0]


def valid_saved_smiles_scope(scope: object, rows: object) -> bool:
    """A plain SMILES file produces exactly one row with the reserved run-level label."""
    if not isinstance(rows, list):
        return False
    return scope == "candidate_table" or (
        scope == "run_ligand" and len(rows) == 1 and isinstance(rows[0], Mapping)
        and rows[0].get("candidate_id") == "run_ligand"
    )


def bounded_smiles_candidates(
    directory: Path, *, max_bytes: int, max_rows: int
) -> tuple[str, list[tuple[str, ...]], str | None]:
    """Read a stable, regular nomination file without trusting a stat-only byte check."""
    table = directory / "ligand_candidates.tsv"
    scope = "candidate_table" if table.exists() or table.is_symlink() else "run_ligand"
    path = table if scope == "candidate_table" else directory / "ligand_smiles.txt"
    try:
        file_info = path.lstat()
        if not stat.S_ISREG(file_info.st_mode):
            return scope, [], "ligand_input_unreadable"
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0)
        with os.fdopen(os.open(path, flags), "rb") as stream:
            opened = os.fstat(stream.fileno())
            if not stat.S_ISREG(opened.st_mode) or not os.path.samestat(file_info, opened):
                return scope, [], "ligand_input_unreadable"
            if opened.st_size > max_bytes:
                return scope, [], "ligand_input_exceeds_interactive_size"
            raw = stream.read(max_bytes + 1)
            after = os.fstat(stream.fileno())
        if len(raw) > max_bytes:
            return scope, [], "ligand_input_exceeds_interactive_size"
        if len(raw) != opened.st_size or (opened.st_mtime_ns, opened.st_ctime_ns) != (
            after.st_mtime_ns, after.st_ctime_ns
        ):
            return scope, [], "ligand_input_unreadable"
        with StringIO(raw.decode("utf-8-sig"), newline="") as decoded:
            if scope == "run_ligand":
                return scope, [("run_ligand", _one_run_smiles(decoded))], None
            reader = csv.reader(decoded, delimiter="\t", strict=True)
            if next(reader, None) != ["candidate_id", "smiles"]:
                return scope, [], "invalid_candidate_table_header"
            rows = []
            for row in reader:
                if not row:
                    continue
                if len(rows) == max_rows:
                    return scope, [], "ligand_input_exceeds_interactive_size"
                rows.append(tuple(cell.strip() for cell in row))
            return scope, rows, None
    except FileNotFoundError:
        return scope, [], "ligand_input_missing" if scope == "run_ligand" else "ligand_input_unreadable"
    except (OSError, UnicodeError, csv.Error):
        return scope, [], "ligand_input_unreadable"
    except ValueError as error:
        return scope, [], str(error)


def named_nominations(directory: Path) -> dict[str, str]:
    """Return unique named table entries that can identify a screening scope."""
    from rfproteina.disease import ligand_admet

    if not (directory / "ligand_candidates.tsv").is_file():
        return {}
    scope, rows, blocker = bounded_smiles_candidates(
        directory, max_bytes=ligand_admet.MAX_INPUT_BYTES, max_rows=ligand_admet.MAX_CANDIDATES
    )
    if scope != "candidate_table" or blocker:
        return {}
    counts = Counter(row[0] for row in rows if row)
    return {
        row[0]: row[1]
        for row in rows
        if len(row) == 2 and row[1] and counts[row[0]] == 1 and ligand_admet.valid_candidate_id(row[0])
    }


def matching_named_smiles(directory: Path, candidate: str) -> str | None:
    """Check a separately prepared structure against any matching authoritative table row."""
    from rfproteina.disease import ligand_admet

    table = directory / "ligand_candidates.tsv"
    if not (table.exists() or table.is_symlink()):
        return None
    scope, rows, blocker = bounded_smiles_candidates(
        directory, max_bytes=ligand_admet.MAX_INPUT_BYTES, max_rows=ligand_admet.MAX_CANDIDATES,
    )
    if blocker or scope != "candidate_table":
        raise ValueError("The named candidate table could not be validated.")
    found = list(nominated_rows(rows, candidate))
    if not found:
        return None
    if len(found) != 1 or len(found[0]) != 2 or not found[0][1]:
        raise ValueError("The candidate table does not contain one complete matching nomination.")
    return found[0][1]


def read_smiles_candidates(directory: Path) -> tuple[str, list[tuple[str, ...]] | None, str | None]:
    """Return the authoritative run input, retaining malformed rows for per-candidate attribution."""
    table = directory / "ligand_candidates.tsv"
    single = directory / "ligand_smiles.txt"
    scope = "candidate_table" if table.exists() or table.is_symlink() else "run_ligand"
    path = table if scope == "candidate_table" else single
    if not path.exists() and not path.is_symlink():
        return scope, None, None
    try:
        if scope == "candidate_table":
            with path.open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.reader(stream, delimiter="\t", strict=True)
                if next(reader, None) != ["candidate_id", "smiles"]:
                    raise ValueError("invalid_candidate_table_header")
                candidates = [tuple(cell.strip() for cell in row) for row in reader if row]
        else:
            candidates = [("run_ligand", read_run_smiles(path))]
    except (OSError, UnicodeError, csv.Error):
        return scope, [], "ligand_input_unreadable"
    except ValueError as error:
        return scope, [], str(error)
    return scope, candidates, None


def parse_nominated_smiles(value: str, chem: ModuleType, *, remove_hs: bool = False) -> Chem.Mol | None:
    if not value or not value.isascii() or not value.isprintable() or " " in value or "|" in value:
        return None
    options = chem.SmilesParserParams()
    options.removeHs = remove_hs
    options.parseName = options.allowCXSMILES = False
    return chem.MolFromSmiles(value, options)


def read_sdf_candidates(path: Path) -> Iterator[tuple[str, str, str]]:
    # Decode per record. Damage to one record must not erase its neighbours.
    block: list[bytes] = []
    with path.open("rb") as stream:
        for index, line in enumerate(stream):
            if index == 0:
                line = line.removeprefix(b"\xef\xbb\xbf")
            if line.rstrip(b"\r\n") == b"$$$$":
                name = block[0].decode("utf-8", errors="replace").strip() if block else ""
                try:
                    structure = b"".join(block).decode("utf-8")
                except UnicodeError:
                    yield name, "", "invalid_sdf_encoding"
                else:
                    yield name, structure, ""
                block = []
            else:
                block.append(line)
        if any(line.strip() for line in block):
            yield block[0].decode("utf-8", errors="replace").strip(), "", "truncated_sdf_record"
