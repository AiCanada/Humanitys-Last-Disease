"""Compare the shape and electrostatic potentials of two prealigned ligands."""

from __future__ import annotations

import io
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING

from rfproteina.disease import corpus_provenance, espsim_runtime, ligand_admet, ligand_inputs

if TYPE_CHECKING:
    from rdkit.Chem.rdchem import Mol

METHOD_ID = "espsim"
METHOD = "espsim_gaussian_gasteiger_0_0_1"
ENDPOINT = "aligned_ligand_shape_and_esp_similarity"
MANIFEST = "espsim.json"
_FILE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,119}\.sdf", re.IGNORECASE)
_DEVICE = re.compile(r"(?:CON|PRN|AUX|NUL|COM[0-9]|LPT[0-9])(?:\.|$)", re.IGNORECASE)
_HASH = re.compile(r"[a-f0-9]{64}")
_ELEMENTS = {1, 6, 7, 8, 9, 15, 16, 17, 35, 53}
_MAX_FILE = 256 * 1024
_MAX_AXIS_SPAN_ANGSTROM = 48
BLOCKERS = frozenset(
    {
        "espsim_manifest_missing",
        "invalid_espsim_manifest",
        "espsim_candidate_mismatch",
        "espsim_reference_mismatch",
        "espsim_structure_unreadable",
        "espsim_invalid_structure",
        "espsim_unsupported_structure",
        "espsim_frame_out_of_scope",
        "espsim_rdkit_unavailable",
        "espsim_invalid_report",
        "espsim_charge_unavailable",
        *espsim_runtime.BLOCKERS,
    }
)


class EspsimError(ValueError):
    """The nomination or comparison could not be validated."""


@dataclass(frozen=True)
class Plan:
    candidate_data: bytes
    reference_data: bytes
    candidate_id: str
    reference_id: str


def _unique(pairs: list[tuple[str, object]]) -> dict:
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("Duplicate field.")
    return result


def _id(value: object) -> bool:
    return isinstance(value, str) and ligand_admet.valid_candidate_id(value)


def _filename(value: object) -> bool:
    return isinstance(value, str) and bool(_FILE.fullmatch(value)) and not _DEVICE.match(value)


def _settings(directory: Path, candidate: str | None) -> dict:
    keys = {"version", "candidate_id", "reference_id", "candidate_structure", "reference_structure", "prealigned"}
    try:
        data = ligand_inputs.read_regular(directory / MANIFEST, 4096, "espsim_manifest", EspsimError)
        settings = json.loads(data.decode("utf-8-sig"), object_pairs_hook=_unique)
        if not (
            isinstance(settings, dict)
            and settings.keys() == keys
            and type(settings["version"]) is int
            and settings["version"] == 1
            and settings["prealigned"] is True
            and _id(settings["candidate_id"])
            and _id(settings["reference_id"])
            and _filename(settings["candidate_structure"])
            and _filename(settings["reference_structure"])
            and settings["candidate_id"] != settings["reference_id"]
            and settings["candidate_structure"].casefold() != settings["reference_structure"].casefold()
        ):
            raise ValueError("Invalid aligned pair nomination.")
    except (ValueError, RecursionError) as error:
        if isinstance(error, EspsimError) and str(error) == "espsim_manifest_missing":
            raise
        raise EspsimError("invalid_espsim_manifest") from error
    if candidate is not None and candidate != settings["candidate_id"]:
        raise EspsimError("espsim_candidate_mismatch")
    return settings


def _bounded(value: object, minimum: float, maximum: float) -> bool:
    return type(value) in (int, float) and minimum <= value <= maximum


def _positions(directory: Path, filename: str, name: str, role: str) -> tuple[bytes, list[tuple], str]:
    try:
        data = ligand_inputs.read_regular(directory / filename, _MAX_FILE, "espsim_structure", EspsimError)
    except EspsimError as error:
        raise EspsimError("espsim_structure_unreadable") from error
    try:
        from rdkit import Chem, rdBase
        from rdkit.Chem import rdDetermineBonds
    except (ImportError, OSError) as error:
        raise EspsimError("espsim_rdkit_unavailable") from error
    lines = data.splitlines()
    if not lines or lines[0] != name.encode("ascii"):
        raise EspsimError(f"espsim_{role}_mismatch")
    if lines[-1] != b"$$$$" or lines.count(b"$$$$") != 1:
        raise EspsimError("espsim_invalid_structure")
    with rdBase.BlockLogs():
        try:
            molecule = next(Chem.ForwardSDMolSupplier(io.BytesIO(data), removeHs=False, strictParsing=True), None)
        except (RuntimeError, ValueError) as error:
            raise EspsimError("espsim_invalid_structure") from error
    if molecule is None or molecule.GetNumConformers() != 1 or not molecule.GetConformer().Is3D():
        raise EspsimError("espsim_invalid_structure")
    atoms = tuple(molecule.GetAtoms())
    if not (
        3 <= molecule.GetNumHeavyAtoms() <= 48
        and len(atoms) <= 128
        and len(Chem.GetMolFrags(molecule)) == 1
        and any(atom.GetAtomicNum() == 6 for atom in atoms)
        and all(
            atom.GetAtomicNum() in _ELEMENTS
            and not (
                atom.GetFormalCharge()
                or atom.GetIsotope()
                or atom.GetNumRadicalElectrons()
                or atom.GetAtomMapNum()
                or atom.GetNumImplicitHs()
                or atom.GetNumExplicitHs()
            )
            for atom in atoms
        )
        and all(stereo.specified == Chem.StereoSpecified.Specified for stereo in Chem.FindPotentialStereo(molecule))
    ):
        raise EspsimError("espsim_unsupported_structure")
    conformer = molecule.GetConformer()
    positions = [tuple(float(value) for value in point) for point in conformer.GetPositions()]
    if not all(_bounded(value, -10000, 10000) for point in positions for value in point):
        raise EspsimError("espsim_invalid_structure")
    # Check that the coordinates describe the declared bonded graph and stereoisomer.
    connectivity = Chem.RWMol()
    for atom in atoms:
        connectivity.AddAtom(Chem.Atom(atom.GetAtomicNum()))
    connectivity.AddConformer(conformer)
    try:
        rdDetermineBonds.DetermineConnectivity(connectivity)
        if _bonds(connectivity) != _bonds(molecule):
            raise EspsimError("espsim_invalid_structure")
        identity = Chem.MolToSmiles(Chem.RemoveHs(molecule), isomericSmiles=True)
        geometry = Chem.Mol(molecule)
        Chem.AssignStereochemistryFrom3D(geometry, replaceExistingTags=True)
        if identity != Chem.MolToSmiles(Chem.RemoveHs(geometry), isomericSmiles=True):
            raise EspsimError("espsim_invalid_structure")
    except (RuntimeError, ValueError) as error:
        raise EspsimError("espsim_invalid_structure") from error
    return data, positions, identity


def _bonds(molecule: Mol) -> set[tuple[int, int]]:
    return {tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))) for bond in molecule.GetBonds()}


def prepare(directory: Path, candidate: str | None = None) -> Plan:
    settings = _settings(directory, candidate)
    candidate_id, reference_id = settings["candidate_id"], settings["reference_id"]
    candidate_data, coordinates, identity = _positions(
        directory, settings["candidate_structure"], candidate_id, "candidate"
    )
    reference_data, reference_coordinates, _ = _positions(
        directory, settings["reference_structure"], reference_id, "reference"
    )
    if any(
        max(axis) - min(axis) > _MAX_AXIS_SPAN_ANGSTROM
        for axis in zip(*coordinates, *reference_coordinates, strict=True)
    ):
        raise EspsimError("espsim_frame_out_of_scope")
    try:
        named = ligand_inputs.matching_named_smiles(directory, candidate_id)
    except ValueError as error:
        raise EspsimError("espsim_candidate_mismatch") from error
    if named is not None:
        from rdkit import Chem, rdBase

        with rdBase.BlockLogs():
            molecule = ligand_inputs.parse_nominated_smiles(named, Chem, remove_hs=True)
        if molecule is None or identity != Chem.MolToSmiles(molecule, isomericSmiles=True):
            raise EspsimError("espsim_candidate_mismatch")
    return Plan(candidate_data, reference_data, candidate_id, reference_id)


def installed() -> bool:
    return espsim_runtime.installed()


def _base() -> dict:
    return {
        "method": METHOD,
        "endpoint": ENDPOINT,
        "release": espsim_runtime.RELEASE,
        "source_sha256": espsim_runtime.SOURCE_SHA256,
        "scope": "prealigned_angstrom_pair",
        "charge_model": "gasteiger_12_iterations",
        "shape_method": "rdkit_heavy_atom_shape_tanimoto",
        "electrostatic_method": "analytic_gaussian_carbo",
        "status": "untested",
        "corpus_provenance": corpus_provenance.declare(METHOD, against=ENDPOINT),
    }


def _result(report: dict, raw: Mapping) -> dict:
    shape, esp, blocker = (raw.get(key) for key in ("shape_tanimoto", "electrostatic_carbo", "blocker"))
    if not _bounded(shape, 0, 1):
        raise EspsimError("espsim_invalid_report")
    if blocker == "espsim_charge_unavailable" and "electrostatic_carbo" not in raw:
        return {**report, "status": "partial", "shape_tanimoto": shape, "blocker": blocker}
    if not _bounded(esp, -1, 1) or "blocker" in raw:
        raise EspsimError("espsim_invalid_report")
    return {**report, "status": "complete", "shape_tanimoto": shape, "electrostatic_carbo": esp}


def screen_run(directory: Path, candidate: str | None = None) -> dict:
    report = _base()
    try:
        plan = prepare(directory, candidate)
        report.update(
            candidate_id=plan.candidate_id,
            reference_id=plan.reference_id,
            candidate_sha256=sha256(plan.candidate_data).hexdigest(),
            reference_sha256=sha256(plan.reference_data).hexdigest(),
        )
        raw = espsim_runtime.predict(plan.candidate_data, plan.reference_data)
        if not isinstance(raw, dict) or raw.keys() not in (
            {"shape_tanimoto", "electrostatic_carbo"},
            {"shape_tanimoto", "blocker"},
        ):
            raise EspsimError("espsim_invalid_report")
        return _result(report, raw)
    except (EspsimError, espsim_runtime.EspsimRuntimeError) as error:
        return {**report, "blocker": str(error)}


def report_view(raw: object) -> dict:
    """Validate the two independent score scales and the nominated pair identity."""
    report = _base()
    invalid = {**report, "blocker": "espsim_invalid_report"}
    attestation = (
        "method",
        "endpoint",
        "release",
        "source_sha256",
        "scope",
        "charge_model",
        "shape_method",
        "electrostatic_method",
    )
    if not isinstance(raw, Mapping) or any(raw.get(key) != report[key] for key in attestation):
        return invalid
    status, blocker = raw.get("status"), raw.get("blocker")
    identities = ("candidate_id", "reference_id", "candidate_sha256", "reference_sha256")
    if any(key in raw for key in identities):
        if not (
            _id(raw.get("candidate_id"))
            and _id(raw.get("reference_id"))
            and raw["candidate_id"] != raw["reference_id"]
            and all(isinstance(raw.get(key), str) and _HASH.fullmatch(raw[key]) for key in identities[2:])
        ):
            return invalid
        report.update({key: raw[key] for key in identities})
    if (
        status == "untested"
        and isinstance(blocker, str)
        and blocker in BLOCKERS
        and not ({"shape_tanimoto", "electrostatic_carbo"} & raw.keys())
    ):
        return {**report, "blocker": blocker}
    if "candidate_id" not in report:
        return invalid
    try:
        result = _result(report, raw)
    except EspsimError:
        return invalid
    return result if result["status"] == status else invalid
