"""Bounded Badapple 2.0 HierS scaffold keys for local database lookup.

Adapted from Badapple2-API at 1b56cb729d7cbaed46d1275fc52cd1f7d37d672e
(https://github.com/unmtransinfo/Badapple2-API), by Jack Ringer. The underlying
Murcko ring-system fragmenter is supplied by the separate ScaffoldGraph package.

BSD 3-Clause License

Copyright (c) 2025, UNM Translational Informatics Division

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

from __future__ import annotations

import json
import os
import sys
from contextlib import redirect_stderr, redirect_stdout
from importlib.metadata import version
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rdkit.Chem import Mol
    from scaffoldgraph.core.scaffold import Scaffold

RDKIT_VERSION = "2024.3.3"
SCAFFOLDGRAPH_VERSION = "1.1.1"
MAX_RING_SYSTEMS = 5
MAX_SCAFFOLDS = 128
_MAX_FRAGMENT_ATTEMPTS = 128
_MAX_HEAVY_ATOMS = 80
_MAX_SMILES = 1024
_MAX_REQUEST_BYTES = 64 * 1024
_MAX_BATCH = 32

_UNAVAILABLE = "badapple2_scaffold_engine_unavailable"
_UNSUPPORTED = "badapple2_scaffold_unsupported"
_LIMIT = "badapple2_scaffold_limit"
_BLOCKERS = {_UNAVAILABLE, _UNSUPPORTED, _LIMIT}


def _plain_smiles(value: object) -> bool:
    return (
        isinstance(value, str) and 0 < len(value) <= _MAX_SMILES
        and all(33 <= ord(char) <= 126 for char in value) and "." not in value and "|" not in value
    )


def _ring_system_count(mol: Mol) -> int:
    """Count ring-containing components after cutting acyclic single bonds."""
    from rdkit.Chem import BondType

    components = list(range(mol.GetNumAtoms()))

    def component(atom: int) -> int:
        while components[atom] != atom:
            atom = components[atom]
        return atom

    for bond in mol.GetBonds():
        if bond.IsInRing() or bond.GetBondType() != BondType.SINGLE:
            begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            components[component(begin)] = component(end)
    return len({component(atom.GetIdx()) for atom in mol.GetAtoms() if atom.IsInRing()})


def _canonical_key(mol: Mol) -> str:
    from rdkit import Chem

    # The database builder and API both reparse canonical aromatic SMILES to
    # converge on the same key when fragmentation changed aromaticity flags.
    parsed = Chem.MolFromSmiles(Chem.MolToSmiles(mol, canonical=True))
    if parsed is None:
        raise ValueError(_UNSUPPORTED)
    key = Chem.MolToSmiles(parsed, canonical=True)
    if not _plain_smiles(key):
        raise ValueError(_UNSUPPORTED)
    return key


def _enumerate_keys(mol: Mol) -> list[str]:
    from rdkit import Chem
    from scaffoldgraph.core.fragment import MurckoRingSystemFragmenter, get_murcko_scaffold
    from scaffoldgraph.core.scaffold import Scaffold

    if _ring_system_count(mol) > MAX_RING_SYSTEMS:
        raise ValueError(_LIMIT)
    murcko = get_murcko_scaffold(mol)
    if murcko.GetNumAtoms() == 0:
        return []
    Chem.RemoveStereochemistry(murcko)
    top = Scaffold(murcko, hash_func=_canonical_key)
    seen = {top.smiles}
    fragmenter = MurckoRingSystemFragmenter()
    attempts = 0

    def descend(child: Scaffold) -> None:
        nonlocal attempts
        rings = child.ring_systems.count
        if rings <= 1:
            return
        # The pinned fragmenter attempts one removal for each ring system.
        attempts += rings
        if attempts > _MAX_FRAGMENT_ATTEMPTS:
            raise ValueError(_LIMIT)
        for parent in fragmenter.fragment(child):
            if not parent:
                continue
            parent.hash_func = _canonical_key
            key = parent.smiles
            if key in seen:
                continue
            if len(seen) >= MAX_SCAFFOLDS:
                raise ValueError(_LIMIT)
            seen.add(key)
            descend(parent)

    descend(top)
    # Benzene is deliberately excluded by the database builder and API.
    return sorted(seen - {"c1ccccc1", "C1=CC=CC=C1"})


def scaffold_keys(canonical_smiles: str) -> list[str]:
    """Return exact lookup keys for an in-scope graph, or a named ValueError."""
    if not _plain_smiles(canonical_smiles):
        raise ValueError(_UNSUPPORTED)
    try:
        from rdkit import Chem, rdBase
    except (ImportError, OSError) as error:
        raise ValueError(_UNAVAILABLE) from error

    parser = Chem.SmilesParserParams()
    parser.allowCXSMILES = False
    parser.parseName = False
    try:
        with rdBase.BlockLogs():
            mol = Chem.MolFromSmiles(canonical_smiles, parser)
            if (
                mol is None or not 0 < mol.GetNumHeavyAtoms() <= _MAX_HEAVY_ATOMS
                or any(atom.GetAtomicNum() == 0 or atom.GetNumRadicalElectrons() for atom in mol.GetAtoms())
            ):
                raise ValueError(_UNSUPPORTED)
            return _enumerate_keys(mol)
    except (ImportError, OSError) as error:
        raise ValueError(_UNAVAILABLE) from error
    except (ValueError, RuntimeError) as error:
        if str(error) in _BLOCKERS:
            raise
        raise ValueError(_UNSUPPORTED) from error


def _request(raw: bytes) -> list[str]:
    if len(raw) > _MAX_REQUEST_BYTES:
        raise ValueError(_LIMIT)
    request = json.loads(raw.decode("utf-8"))
    if not (
        isinstance(request, list) and 0 < len(request) <= _MAX_BATCH
        and all(_plain_smiles(value) for value in request) and len(set(request)) == len(request)
    ):
        raise ValueError(_UNSUPPORTED)
    return request


def _worker(smiles: list[str]) -> dict[str, dict[str, list[str] | str]]:
    if version("rdkit") != RDKIT_VERSION or version("scaffoldgraph") != SCAFFOLDGRAPH_VERSION:
        raise ValueError(_UNAVAILABLE)
    result: dict[str, dict[str, list[str] | str]] = {}
    for key in smiles:
        try:
            result[key] = {"keys": scaffold_keys(key)}
        except ValueError as error:
            reason = str(error)
            result[key] = {"blocker": reason if reason in _BLOCKERS else _UNSUPPORTED}
    return result


def main() -> int:
    """Read one private request and emit only its bounded result."""
    try:
        with open(os.devnull, "w", encoding="utf-8") as sink, redirect_stdout(sink), redirect_stderr(sink):
            result = _worker(_request(sys.stdin.buffer.read(_MAX_REQUEST_BYTES + 1)))
        print(json.dumps(result, allow_nan=False, separators=(",", ":")))
    except Exception:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
