"""Attribute a saved ToxScan export to one explicitly nominated molecular graph."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from rfproteina.disease import ligand_admet, ligand_alerts, ligand_inputs

MANIFEST = "toxscan.json"
SOURCE_FILE = "toxscan.csv"
SOURCE = "https://funmg.dp.tech/Toxscan"
MAX_EXPORT_BYTES = 2 * 1024 * 1024
MAX_EXPORT_ROWS = 256
MAX_SMILES = 2048
SHA256 = re.compile(r"[a-f0-9]{64}")


class ToxScanError(ValueError):
    """The submitted export or molecular identity could not be attributed."""


def _unique(pairs: list[tuple]) -> dict:
    fields = dict(pairs)
    if len(fields) != len(pairs):
        raise ValueError("duplicate field")
    return fields


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _verify_bond_directions(smiles: str, chem: ModuleType) -> None:
    """Each directional bond in the input must refer to a double bond with recoverable geometry."""
    options = chem.SmilesParserParams()
    options.sanitize = False
    options.removeHs = False
    options.allowCXSMILES = False
    options.parseName = False
    graph = chem.MolFromSmiles(smiles, options)
    if graph is None:
        raise ValueError("unresolved directional bond")
    # Before sanitization the original / and \ marks are still available, including on explicit H.
    directions = (chem.BondDir.ENDUPRIGHT, chem.BondDir.ENDDOWNRIGHT)
    marked = [
        (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in graph.GetBonds() if bond.GetBondDir() in directions
    ]
    for atom in graph.GetAtoms():
        atom.SetAtomMapNum(0)
    chem.SanitizeMol(graph)
    chem.SetBondStereoFromDirections(graph)
    chem.AssignStereochemistry(graph, cleanIt=True, force=True)
    represented = {
        index for bond in graph.GetBonds()
        if bond.GetStereo() in (chem.BondStereo.STEREOE, chem.BondStereo.STEREOZ,
                                 chem.BondStereo.STEREOCIS, chem.BondStereo.STEREOTRANS)
        for index in (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
    }
    if any(left not in represented and right not in represented for left, right in marked):
        raise ValueError("unresolved directional bond")


def canonical(smiles: str) -> str:
    """Keep stereochemistry, isotopes, charges and every fragment of the submitted graph."""
    try:
        from rdkit import Chem
    except (ImportError, OSError) as error:
        raise ToxScanError("toxscan_rdkit_unavailable") from error
    if not isinstance(smiles, str) or not 0 < len(smiles) <= MAX_SMILES:
        raise ToxScanError("toxscan_invalid_structure")
    try:
        mol = ligand_alerts.parse_molecule(smiles, nominated=True)
        if mol.HasQuery():
            raise ValueError("an exact molecular structure is required")
        # Atom-map annotations identify atoms in reactions, not chemical structures.
        for atom in mol.GetAtoms():
            if atom.GetAtomMapNum():
                atom.SetAtomMapNum(0)
        if "@" in smiles or "/" in smiles or "\\" in smiles:
            Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
        identity = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        # RDKit can accept @AL1 and @AL2, then silently erase the axial stereocenter from both.
        if len(re.findall(r"@+", smiles)) != len(re.findall(r"@+", identity)):
            raise ValueError("unresolved stereocenter")
        if "/" in smiles or "\\" in smiles:
            _verify_bond_directions(smiles, Chem)
        return identity
    except (ligand_alerts.LigandUnreadableError, ValueError, RuntimeError, OverflowError) as error:
        raise ToxScanError("toxscan_invalid_structure") from error


@dataclass(frozen=True)
class Nomination:
    candidate_id: str
    canonical_smiles: str
    export_sha256: str


def read_manifest(directory: Path) -> dict:
    try:
        raw = ligand_inputs.read_regular(directory / MANIFEST, 4096, "toxscan_manifest", ToxScanError)
        doc = json.loads(raw.decode("utf-8-sig"), object_pairs_hook=_unique)
        if not isinstance(doc, dict) or type(doc.get("version")) is not int or doc["version"] not in (1, 2):
            raise ValueError("invalid manifest version")
        keys = {"version", "candidate_id", "source", "export_sha256"}
        if doc["version"] == 2:
            keys.add("structure_sha256")
            if not isinstance(doc.get("structure_sha256"), str) or not SHA256.fullmatch(doc["structure_sha256"]):
                raise ValueError("invalid structure digest")
        if (
            doc.keys() != keys or doc["source"] != SOURCE
            or not isinstance(doc["export_sha256"], str) or not SHA256.fullmatch(doc["export_sha256"])
            or not isinstance(doc["candidate_id"], str) or not ligand_admet.valid_candidate_id(doc["candidate_id"])
        ):
            raise ValueError("invalid nomination")
    except (ValueError, RecursionError) as error:
        if str(error) == "toxscan_manifest_missing":
            raise
        raise ToxScanError("toxscan_invalid_manifest") from error
    return doc


def candidate_structure(directory: Path, candidate_id: str) -> str:
    """Read a single explicit, unambiguous nomination independently of an existing export."""
    if not isinstance(candidate_id, str) or not ligand_admet.valid_candidate_id(candidate_id):
        raise ToxScanError("toxscan_candidate_not_nominated")
    scope, rows, blocker = ligand_inputs.bounded_smiles_candidates(
        directory, max_bytes=ligand_admet.MAX_INPUT_BYTES, max_rows=ligand_admet.MAX_CANDIDATES,
    )
    if scope != "candidate_table" or blocker:
        raise ToxScanError("toxscan_candidate_table_required")
    selected = [row for row in rows if row and row[0] == candidate_id]
    if not selected:
        raise ToxScanError("toxscan_candidate_not_nominated")
    if len(selected) != 1 or len(selected[0]) != 2:
        raise ToxScanError("toxscan_ambiguous_nomination")
    return canonical(selected[0][1])


def nomination(directory: Path, candidate: str | None) -> Nomination:
    doc = read_manifest(directory)
    candidate_id = doc["candidate_id"]
    if candidate is not None and candidate != candidate_id:
        raise ToxScanError("toxscan_candidate_not_nominated")
    structure = candidate_structure(directory, candidate_id)
    if doc["version"] == 2 and doc["structure_sha256"] != sha256(structure.encode("ascii")):
        raise ToxScanError("toxscan_nomination_changed")
    return Nomination(candidate_id, structure, doc["export_sha256"])


def export_bytes(directory: Path, nomination: Nomination) -> bytes:
    """Bind the file to the operator's source declaration. The service issues no publisher signature here."""
    raw = read_export_file(directory / SOURCE_FILE)
    if sha256(raw) != nomination.export_sha256:
        raise ToxScanError("toxscan_export_digest_mismatch")
    return raw


def read_export_file(path: Path) -> bytes:
    try:
        return ligand_inputs.read_regular(path, MAX_EXPORT_BYTES, "toxscan_export", ToxScanError)
    except ToxScanError as error:
        if str(error) == "toxscan_export_missing":
            raise
        raise ToxScanError("toxscan_invalid_export") from error
