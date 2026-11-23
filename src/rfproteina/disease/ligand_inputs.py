"""Read molecular nominations shared by post-generation small-molecule screens."""

from __future__ import annotations

import csv
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rdkit import Chem


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
            lines = [s.strip() for s in path.read_text(encoding="utf-8-sig").splitlines() if s.strip()]
            if len(lines) != 1:
                raise ValueError("expected_one_run_ligand")
            # Legacy .smi format permits a title following the structure.
            smiles = lines[0].split(maxsplit=1)
            if len(smiles) > 1 and smiles[1].startswith("|"):
                raise ValueError("unsupported_extended_smiles")
            candidates = [("run_ligand", smiles[0])]
    except (OSError, UnicodeError, csv.Error):
        return scope, [], "ligand_input_unreadable"
    except ValueError as error:
        return scope, [], str(error)
    return scope, candidates, None


def parse_nominated_smiles(value: str, chem: ModuleType) -> Chem.Mol | None:
    if not value or not value.isascii() or not value.isprintable() or " " in value or "|" in value:
        return None
    options = chem.SmilesParserParams()
    options.removeHs = options.parseName = options.allowCXSMILES = False
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
