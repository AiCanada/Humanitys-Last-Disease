"""Sample conformers of one explicitly nominated organic ligand with CREST."""

from __future__ import annotations

import io
import json
import re
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from hashlib import sha256
from math import isfinite
from pathlib import Path
from typing import TYPE_CHECKING

from rfproteina.disease import corpus_provenance, crest_runtime, ligand_admet, ligand_inputs

if TYPE_CHECKING:
    from rdkit import Chem

METHOD_ID = "crest"
METHOD = "crest_gfn2_xtb_imtd_gc_quick"
MANIFEST = "crest.json"
ENDPOINT = "relative_conformer_energy"
_SDF_LIMIT = 256 * 1024
_MAX_ATOMS = 96
_MAX_FRAMES = 32
_MAX_PROVIDER_FRAMES = 2048
_HARTREE_TO_KCAL_MOL = 627.509474
_FILENAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,119}\.sdf", re.IGNORECASE)
_DEVICE = re.compile(r"(?:CON|PRN|AUX|NUL|COM[0-9]|LPT[0-9])(?:\.|$)", re.IGNORECASE)
_DIGEST = re.compile(r"[a-f0-9]{64}")
_ELEMENTS = frozenset({"H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"})
BLOCKERS = frozenset({
    "crest_manifest_missing", "invalid_crest_manifest", "crest_candidate_mismatch", "crest_structure_unreadable",
    "crest_invalid_structure", "crest_unsupported_ligand", "crest_rdkit_unavailable", "crest_invalid_ensemble",
    "crest_no_valid_conformers", *crest_runtime.BLOCKERS,
})


class CrestError(ValueError):
    """The nomination or conformer ensemble is outside the qualified contract."""


@dataclass(frozen=True)
class Plan:
    candidate_id: str
    structure_sha256: str
    molecule: Chem.Mol
    elements: tuple[str, ...]
    bonds: frozenset[tuple[int, int]]
    identity: str
    xyz: bytes


def _unique(pairs: list[tuple[str, object]]) -> dict:
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("Duplicate field.")
    return result


def _settings(directory: Path, candidate: str | None) -> tuple[str, str]:
    try:
        data = ligand_inputs.read_regular(directory / MANIFEST, 4096, "crest_manifest", CrestError)
        settings = json.loads(data.decode("utf-8-sig"), object_pairs_hook=_unique)
        if not isinstance(settings, dict) or settings.keys() != {"version", "structure", "candidate_id"}:
            raise ValueError("Invalid nomination fields.")
        name, filename = settings["candidate_id"], settings["structure"]
        if not (
            type(settings["version"]) is int and settings["version"] == 1
            and isinstance(name, str) and ligand_admet.valid_candidate_id(name)
            and isinstance(filename, str) and _FILENAME.fullmatch(filename) and not _DEVICE.match(filename)
        ):
            raise ValueError("Invalid nomination values.")
    except (ValueError, RecursionError) as error:
        if isinstance(error, CrestError) and str(error) == "crest_manifest_missing":
            raise
        raise CrestError("invalid_crest_manifest") from error
    if candidate is not None and name != candidate:
        raise CrestError("crest_candidate_mismatch")
    return name, filename


def _number(value: object, lower: float, upper: float) -> bool:
    return type(value) in (float, int) and lower <= value <= upper


def _bonds(molecule: Chem.Mol) -> frozenset[tuple[int, int]]:
    return frozenset(tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))) for b in molecule.GetBonds())


def _identity(molecule: Chem.Mol) -> str:
    from rdkit import Chem

    return Chem.MolToSmiles(Chem.RemoveHs(molecule), isomericSmiles=True)


def _same_geometry(plan: Plan, coordinates: tuple[tuple[float, float, float], ...]) -> bool:
    from rdkit import Chem
    from rdkit.Chem import rdDetermineBonds

    conformer = Chem.Conformer(len(plan.elements))
    conformer.Set3D(True)
    for index, point in enumerate(coordinates):
        conformer.SetAtomPosition(index, point)

    atoms = Chem.RWMol()
    for element in plan.elements:
        atoms.AddAtom(Chem.Atom(element))
    atoms.AddConformer(conformer)
    try:
        rdDetermineBonds.DetermineConnectivity(atoms)
        if _bonds(atoms) != plan.bonds:
            return False
        ligand = Chem.Mol(plan.molecule)
        ligand.RemoveAllConformers()
        ligand.AddConformer(conformer)
        Chem.AssignStereochemistryFrom3D(ligand, replaceExistingTags=True)
        return _identity(ligand) == plan.identity
    except (RuntimeError, ValueError):
        return False


def installed() -> bool:
    try:
        from rdkit.Chem import rdDetermineBonds
    except (ImportError, OSError):
        return False
    return callable(rdDetermineBonds.DetermineConnectivity) and crest_runtime.installed()


def prepare(directory: Path, candidate: str | None = None) -> Plan:
    name, filename = _settings(directory, candidate)
    try:
        data = ligand_inputs.read_regular(directory / filename, _SDF_LIMIT, "crest_structure", CrestError)
    except CrestError as error:
        raise CrestError("crest_structure_unreadable") from error
    try:
        from rdkit import Chem, rdBase
    except (ImportError, OSError) as error:
        raise CrestError("crest_rdkit_unavailable") from error

    lines = data.splitlines()
    if not lines or lines[0] != name.encode("ascii"):
        raise CrestError("crest_candidate_mismatch")
    if lines[-1] != b"$$$$" or lines.count(b"$$$$") != 1:
        raise CrestError("crest_invalid_structure")
    molecule = None
    with rdBase.BlockLogs(), suppress(RuntimeError, ValueError):
        molecule = next(Chem.ForwardSDMolSupplier(io.BytesIO(data), removeHs=False, strictParsing=True), None)
    if molecule is None or molecule.GetNumConformers() != 1 or not molecule.GetConformer().Is3D():
        raise CrestError("crest_invalid_structure")
    atoms = tuple(molecule.GetAtoms())
    elements = tuple(atom.GetSymbol() for atom in atoms)
    if not (
        len(atoms) <= _MAX_ATOMS and 4 <= molecule.GetNumHeavyAtoms() <= 32
        and set(elements) <= _ELEMENTS and len(Chem.GetMolFrags(molecule)) == 1
        and Chem.GetFormalCharge(molecule) == 0
        and all(not (atom.GetNumImplicitHs() or atom.GetNumExplicitHs() or atom.GetIsotope()
                    or atom.GetNumRadicalElectrons() or atom.GetAtomMapNum()) for atom in atoms)
        and all(stereo.specified == Chem.StereoSpecified.Specified and stereo.type in (
            Chem.StereoType.Atom_Tetrahedral, Chem.StereoType.Bond_Double,
        ) for stereo in Chem.FindPotentialStereo(molecule))
    ):
        raise CrestError("crest_unsupported_ligand")
    coordinates = tuple(tuple(float(value) for value in point) for point in molecule.GetConformer().GetPositions())
    if not all(_number(value, -1000, 1000) for point in coordinates for value in point):
        raise CrestError("crest_invalid_structure")
    xyz = [str(len(atoms)), "nominated structure"]
    xyz.extend(
        f"{element:2s} {x:.8f} {y:.8f} {z:.8f}"
        for element, (x, y, z) in zip(elements, coordinates, strict=True)
    )
    identity = _identity(molecule)
    try:
        named_smiles = ligand_inputs.matching_named_smiles(directory, name)
    except ValueError as error:
        raise CrestError("crest_candidate_mismatch") from error
    if named_smiles is not None:
        with rdBase.BlockLogs():
            nominated = ligand_inputs.parse_nominated_smiles(named_smiles, Chem, remove_hs=True)
        if nominated is None or _identity(nominated) != identity:
            raise CrestError("crest_candidate_mismatch")
    plan = Plan(
        name, sha256(data).hexdigest(), molecule, elements, _bonds(molecule), identity,
        ("\n".join(xyz) + "\n").encode("ascii"),
    )
    if not _same_geometry(plan, coordinates):
        raise CrestError("crest_invalid_structure")
    return plan


def _frames(raw: bytes, elements: tuple[str, ...]) -> list[tuple[float, tuple]]:
    try:
        lines = raw.decode("ascii").splitlines()
        stride = len(elements) + 2
        count, remainder = divmod(len(lines), stride)
        if remainder or not 0 < count <= _MAX_PROVIDER_FRAMES:
            raise ValueError("Invalid frame count.")
        frames = []
        for start in range(0, len(lines), stride):
            if lines[start].strip() != str(len(elements)):
                raise ValueError("Invalid frame atom count.")
            energy = float(lines[start + 1].strip())
            if not isfinite(energy) or abs(energy) > 1e7:
                raise ValueError("Invalid frame energy.")
            positions = []
            for element, line in zip(elements, lines[start + 2:start + stride], strict=True):
                symbol, x, y, z = line.split()
                point = tuple(float(value) for value in (x, y, z))
                if element != symbol or not all(_number(value, -1000, 1000) for value in point):
                    raise ValueError("Invalid frame atom.")
                positions.append(point)
            frames.append((energy, tuple(positions)))
        return frames
    except (ValueError, AttributeError) as error:
        raise CrestError("crest_invalid_ensemble") from error


def _base() -> dict:
    return {
        "method": METHOD, "endpoint": ENDPOINT, "release": crest_runtime.RELEASE,
        "executable_sha256": crest_runtime.EXECUTABLE_SHA256,
        "environment": "gas_phase", "status": "untested", "conformers": [],
        "corpus_provenance": corpus_provenance.declare(METHOD, against=ENDPOINT),
    }


def screen_run(directory: Path, candidate: str | None = None) -> dict:
    report = _base()
    try:
        plan = prepare(directory, candidate)
        report.update(candidate_id=plan.candidate_id, structure_sha256=plan.structure_sha256, elements=plan.elements)
        frames = _frames(crest_runtime.predict(plan.xyz), plan.elements)
        accepted = [frame for frame in frames if _same_geometry(plan, frame[1])]
        if not accepted:
            raise CrestError("crest_no_valid_conformers")
        accepted.sort(key=lambda frame: frame[0])
        minimum = accepted[0][0]
        rows = [
            {
                "rank": rank, "energy_hartree": energy, "coordinates_angstrom": points,
                "relative_energy_kcal_mol": (energy - minimum) * _HARTREE_TO_KCAL_MOL,
            }
            for rank, (energy, points) in enumerate(accepted[:_MAX_FRAMES], 1)
        ]
        rejected = len(frames) - len(accepted)
        omitted = len(accepted) - len(rows)
        return {**report, "status": "partial" if rejected or omitted else "complete", "conformers": rows,
                "rejected_conformers": rejected, "omitted_conformers": omitted}
    except (CrestError, crest_runtime.CrestRuntimeError) as error:
        return {**report, "blocker": str(error)}


def report_view(raw: object) -> dict:
    """Project validated energy evidence without exporting candidate coordinates."""
    view = _base()
    invalid = {**view, "blocker": "crest_invalid_ensemble"}
    attestation = ("method", "endpoint", "release", "executable_sha256", "environment")
    if not isinstance(raw, Mapping) or any(raw.get(field) != view[field] for field in attestation):
        return invalid
    digest, name, rows = raw.get("structure_sha256"), raw.get("candidate_id"), raw.get("conformers")
    if (isinstance(name, str) and ligand_admet.valid_candidate_id(name)
            and isinstance(digest, str) and _DIGEST.fullmatch(digest)):
        view.update(candidate_id=name, structure_sha256=digest)
    blocker = raw.get("blocker")
    if raw.get("status") == "untested" and rows == [] and isinstance(blocker, str) and blocker in BLOCKERS:
        return {**view, "blocker": blocker}
    rejected, omitted = (raw.get(field) for field in ("rejected_conformers", "omitted_conformers"))
    if not (
        "structure_sha256" in view and isinstance(rows, list) and 0 < len(rows) <= _MAX_FRAMES
        and all(type(count) is int and 0 <= count <= _MAX_PROVIDER_FRAMES for count in (rejected, omitted))
        and len(rows) + rejected + omitted <= _MAX_PROVIDER_FRAMES and (not omitted or len(rows) == _MAX_FRAMES)
        and raw.get("status") == ("partial" if rejected or omitted else "complete") and "blocker" not in raw
    ):
        return invalid
    minimum = None
    previous = None
    conformers = []
    for rank, row in enumerate(rows, 1):
        if not isinstance(row, Mapping) or type(row.get("rank")) is not int or row["rank"] != rank:
            return invalid
        energy, relative = row.get("energy_hartree"), row.get("relative_energy_kcal_mol")
        if not _number(energy, -1e7, 1e7) or not _number(relative, 0, 1.3e10):
            return invalid
        if rank == 1:
            minimum = energy
        expected = (energy - minimum) * _HARTREE_TO_KCAL_MOL
        if (previous is not None and energy < previous) or abs(relative - expected) > 1e-7:
            return invalid
        conformers.append({"rank": rank, "energy_hartree": energy, "relative_energy_kcal_mol": relative})
        previous = energy
    return {**view, "status": raw["status"], "conformers": conformers,
            "rejected_conformers": rejected, "omitted_conformers": omitted}
