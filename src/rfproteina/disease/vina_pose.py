"""Score a supplied, coordinate-verified small-molecule pose with rigid-receptor Vina."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path, PureWindowsPath
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

from rfproteina.disease.corpus_provenance import declare
from rfproteina.disease.screening_tools import tool_code_path

if TYPE_CHECKING:
    import gemmi

METHOD = "autodock_vina_fixed_pose"
MANIFEST = "vina_poses.json"
_ENDPOINT = "protein_ligand_binding_affinity"
_SPACING = 0.375
# Bound native map allocation before constructing it. The limit is on map points, not on a score.
_MAX_GRID_POINTS = 4_000_000
_MAX_INPUT_BYTES = 32 * 1024 * 1024
# PDBQT stores Cartesian coordinates to three decimal places, independently on each axis.
_XYZ_TOL = 0.000501
_ELEMENTS = {
    "A": "C",
    "C": "C",
    "N": "N",
    "NA": "N",
    "O": "O",
    "OA": "O",
    "P": "P",
    "S": "S",
    "SA": "S",
    "F": "F",
    "Cl": "Cl",
    "Br": "Br",
    "I": "I",
    "Si": "Si",
    "H": "H",
    "HD": "H",
}
_AMINO_ACIDS = frozenset("ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split())


class PoseError(ValueError):
    """A named reason why no score can be attributed to this pose."""


def corpus_provenance() -> dict:
    return declare(METHOD, against=_ENDPOINT)


def _untested(reason: str) -> dict:
    return {"method": METHOD, "status": "untested", "blocker": reason}


def _unique(pairs: list) -> dict:
    record = {}
    for key, value in pairs:
        if key in record:
            raise PoseError("duplicate_manifest_key")
        record[key] = value
    return record


def _read(path: Path, limit: int, reason: str) -> bytes:
    try:
        if not path.is_file():
            raise PoseError(reason)
        with path.open("rb") as stream:
            data = stream.read(limit + 1)
    except OSError as error:
        raise PoseError(reason) from error
    if len(data) > limit:
        raise PoseError("input_size_limit")
    return data


def _local_file(root: Path, name: object) -> Path:
    if not isinstance(name, str) or not name or "\x00" in name or PureWindowsPath(name).anchor:
        raise PoseError("invalid_prepared_path")
    try:
        path = (root / name).resolve()
        if Path(name).is_absolute() or ".." in PureWindowsPath(name).parts or not path.is_relative_to(root.resolve()):
            raise PoseError("invalid_prepared_path")
    except (OSError, RuntimeError) as error:
        raise PoseError("invalid_prepared_path") from error
    return path


def _name(value: object) -> bool:
    return isinstance(value, str) and value.strip() == value and value not in ("", ".", "?") and value.isprintable()


def _triple(value: object) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != 3 or any(type(v) not in (int, float) for v in value):
        raise PoseError("invalid_scoring_box")
    try:
        result = tuple(float(v) for v in value)
    except OverflowError as error:
        raise PoseError("invalid_scoring_box") from error
    if not all(math.isfinite(v) for v in result):
        raise PoseError("invalid_scoring_box")
    return result


@dataclass(frozen=True)
class Pose:
    receptor: Path
    ligand: Path
    receptor_chains: tuple[str, ...]
    ligand_chain: str
    ligand_component: str
    center: tuple[float, ...]
    size: tuple[float, ...]

    @classmethod
    def parse(cls, root: Path, entry: object) -> Pose:
        if not isinstance(entry, dict) or entry.keys() != cls.__dataclass_fields__.keys():
            raise PoseError("invalid_pose_settings")
        chains = entry["receptor_chains"]
        chain, comp = entry["ligand_chain"], entry["ligand_component"]
        if (
            not isinstance(chains, list)
            or not chains
            or not all(map(_name, chains))
            or not _name(chain)
            or not _name(comp)
            or len(set(chains)) != len(chains)
            or chain in chains
        ):
            raise PoseError("invalid_partner_selection")
        center, size = _triple(entry["center"]), _triple(entry["size"])
        if min(size) <= 0:
            raise PoseError("invalid_scoring_box")
        if max(size) > _SPACING * _MAX_GRID_POINTS:
            raise PoseError("scoring_grid_size_limit")
        if math.prod(math.ceil(v / _SPACING) + 1 for v in size) > _MAX_GRID_POINTS:
            raise PoseError("scoring_grid_size_limit")
        return cls(
            _local_file(root, entry["receptor"]),
            _local_file(root, entry["ligand"]),
            tuple(chains),
            chain,
            comp,
            center,
            size,
        )

    def selection(self) -> dict:
        return {
            "receptor_label_asym_ids": list(self.receptor_chains),
            "ligand_label_asym_id": self.ligand_chain,
            "ligand_component": self.ligand_component,
            "center_angstrom": list(self.center),
            "size_angstrom": list(self.size),
            "grid_spacing_angstrom": _SPACING,
        }


@dataclass(frozen=True)
class Atom:
    element: str
    xyz: tuple[float, ...]
    name: str
    residue: str
    serial: int = 0
    pdbqt_type: str = ""


@dataclass
class Preparation:
    atoms: list[Atom]
    branches: list[tuple[int, int]]


def _prepared(data: bytes, *, ligand: bool) -> Preparation:
    """Read heavy atoms from the exact bytes passed to Vina. The engine checks the torsion tree."""
    try:
        lines = data.decode("ascii").splitlines()
    except UnicodeDecodeError as error:
        raise PoseError("invalid_pdbqt") from error
    atoms = []
    ligand_residues = set()
    serials = set()
    branches = []
    tags = []
    for line in lines:
        tag = line[:6].strip()
        if tag in {"MODEL", "ENDMDL", "BEGIN_", "END_RE"}:
            raise PoseError("single_rigid_receptor_and_one_pose_required")
        if tag not in {"ATOM", "HETATM"}:
            if line.strip():
                tags.append(line.split()[0])
            if tag == "BRANCH":
                try:
                    _, parent, child = line.split()
                    branches.append((int(parent), int(child)))
                except ValueError as error:
                    raise PoseError("invalid_ligand_torsion_tree") from error
            continue
        pdbqt_type = line[77:].strip()
        elem = _ELEMENTS.get(pdbqt_type)
        if elem is None:
            raise PoseError("unsupported_pdbqt_atom_type")
        try:
            xyz = tuple(float(line[i : i + 8]) for i in (30, 38, 46))
            charge = float(line[68:76])
            serial = int(line[6:11])
        except ValueError as error:
            raise PoseError("invalid_pdbqt_atom") from error
        if not all(math.isfinite(v) for v in (*xyz, charge)) or line[16:17] != " ":
            raise PoseError("invalid_pdbqt_atom")
        if ligand:
            ligand_residues.add(line[17:27])
            if serial in serials:
                raise PoseError("duplicate_ligand_atom_serial")
            serials.add(serial)
        if elem != "H":
            atoms.append(Atom(elem, xyz, line[12:16].strip(), line[17:20].strip(), serial, pdbqt_type))
    if ligand:
        if len(ligand_residues) != 1 or any(tags.count(tag) != 1 for tag in ("ROOT", "ENDROOT", "TORSDOF")):
            raise PoseError("single_prepared_ligand_required")
        if any(a == b or a not in serials or b not in serials for a, b in branches):
            raise PoseError("invalid_ligand_torsion_tree")
    elif set(tags) - {"REMARK", "TER", "END", "WARNING"}:
        raise PoseError("rigid_prepared_receptor_required")
    if not atoms:
        raise PoseError("prepared_heavy_atoms_absent")
    return Preparation(atoms, branches)


def _ligand_inventory(block: gemmi.cif.Block, component: str, ligand: list[Atom]) -> str:
    """Check source completeness when the CIF declares a component atom list or elemental formula."""
    import gemmi

    expected = []
    for row in block.find("_chem_comp_atom.", ["comp_id", "atom_id", "type_symbol"]):
        comp, name, elem = map(gemmi.cif.as_string, row)
        if comp == component and elem.upper() not in {"H", "D"}:
            expected.append((name, elem.capitalize()))
    if expected:
        if Counter(expected) != Counter((atom.name, atom.element) for atom in ligand):
            raise PoseError("ligand_component_inventory_mismatch")
        return "component_atom_names_and_elements"
    for row in block.find("_chem_comp.", ["id", "formula"]):
        comp, formula = map(gemmi.cif.as_string, row)
        if comp != component or not formula:
            continue
        parts = formula.split()
        if parts and re.fullmatch(r"[+-]?\d+[+-]?", parts[-1]):
            parts.pop()  # The PDBx sum formula may end in a separate formal charge.
        atoms = "".join(parts)
        terms = re.findall(r"([A-Z][a-z]?)(\d*)", atoms)
        if not terms or "".join(e + n for e, n in terms) != atoms:
            break
        counts = Counter()
        for elem, n in terms:
            if elem not in {"H", "D"}:
                counts[elem] += int(n or 1)
        if counts != Counter(atom.element for atom in ligand):
            raise PoseError("ligand_component_inventory_mismatch")
        return "component_formula_heavy_elements"
    return "observed_heavy_atoms_only"


@dataclass
class LigandBonds:
    pairs: set[frozenset[str]] = field(default_factory=set)
    aromatic_flags: dict[str, set[str]] = field(default_factory=dict)

    @classmethod
    def from_cif(cls, block: gemmi.cif.Block, component: str) -> LigandBonds:
        import gemmi

        result = cls()
        tags = ["comp_id", "atom_id_1", "atom_id_2", "?pdbx_aromatic_flag"]
        for row in block.find("_chem_comp_bond.", tags):
            comp, a, b = (gemmi.cif.as_string(row[i]) for i in range(3))
            if comp == component:
                result.pairs.add(frozenset((a, b)))
                flag = gemmi.cif.as_string(row[3]).upper() if row.has(3) else ""
                for atom in (a, b):
                    result.aromatic_flags.setdefault(atom, set()).add(flag)
        return result

    def verify(self, prepared: Preparation, matched: dict[int, str]) -> dict:
        checked_carbons = checked_torsions = 0
        for atom in prepared.atoms:
            flags = self.aromatic_flags.get(matched[atom.serial], set())
            if atom.element == "C" and ("Y" in flags or flags == {"N"}):
                if (atom.pdbqt_type == "A") != ("Y" in flags):
                    raise PoseError("ligand_aromatic_carbon_type_mismatch")
                checked_carbons += 1
        for a, b in prepared.branches:
            if self.pairs and a in matched and b in matched:
                if frozenset((matched[a], matched[b])) not in self.pairs:
                    raise PoseError("ligand_torsion_bond_mismatch")
                checked_torsions += 1
        return {"carbon_types_checked": checked_carbons, "heavy_torsion_bonds_checked": checked_torsions}


def _source(text: str, pose: Pose) -> tuple[list[Atom], list[Atom], str, LigandBonds]:
    import gemmi

    doc = gemmi.cif.read_string(text)
    if len(doc) != 1:
        raise PoseError("single_structure_required")
    block = doc.sole_block()
    for row in block.find("_struct_conn.", ["conn_type_id", "ptnr1_label_asym_id", "ptnr2_label_asym_id"]):
        kind, left, right = map(gemmi.cif.as_string, row)
        linked = kind.startswith("covale") or kind in {"metalc", "disulf"}
        if linked and left != right and pose.ligand_chain in (left, right):
            raise PoseError("covalent_or_metal_linked_ligand")
    entity = dict(tuple(map(gemmi.cif.as_string, row)) for row in block.find("_entity.", ["id", "type"]))
    polymer = dict(tuple(map(gemmi.cif.as_string, row)) for row in block.find("_entity_poly.", ["entity_id", "type"]))
    fields = [
        "label_asym_id",
        "label_entity_id",
        "label_comp_id",
        "label_atom_id",
        "type_symbol",
        "label_seq_id",
        "Cartn_x",
        "Cartn_y",
        "Cartn_z",
        "occupancy",
        "?label_alt_id",
        "?pdbx_PDB_model_num",
        "?auth_seq_id",
        "?pdbx_PDB_ins_code",
    ]
    table = block.find("_atom_site.", fields)
    if not table:
        raise PoseError("required_atom_site_fields_absent")
    receptor, ligand = [], []
    seen_chains, identities, models, ligand_residues = set(), set(), set(), set()
    for row in table:
        # str() on a gemmi CIF value keeps quotes. as_string decodes quotes and CIF nulls.
        chain, ent, comp, name, elem, seq, x, y, z, occ, alt, model, auth, ins = (
            gemmi.cif.as_string(row[i]) if row.has(i) else "" for i in range(len(fields))
        )
        if chain != pose.ligand_chain and chain not in pose.receptor_chains:
            continue
        models.add(model or "1")
        if len(models) > 1:
            raise PoseError("multiple_structure_models")
        if elem.upper() in {"H", "D"}:
            continue
        if alt:
            raise PoseError("alternate_conformer_present")
        xyz = tuple(map(float, (x, y, z)))
        if not all(map(math.isfinite, xyz)) or not 0 < float(occ) <= 1:
            raise PoseError("invalid_source_atom")
        identity = (chain, seq, auth, ins, name)
        if not name or identity in identities:
            raise PoseError("duplicate_source_atom")
        identities.add(identity)
        atom = Atom(elem.capitalize(), xyz, name, comp)
        if chain == pose.ligand_chain:
            if comp != pose.ligand_component or entity.get(ent) != "non-polymer" or comp in {"HOH", "DOD"}:
                raise PoseError("selected_ligand_identity_mismatch")
            ligand_residues.add((ent, seq, auth, ins))
            ligand.append(atom)
        else:
            if entity.get(ent) != "polymer" or polymer.get(ent) != "polypeptide(L)" or comp not in _AMINO_ACIDS:
                raise PoseError("unsupported_receptor_residue")
            if not seq:
                raise PoseError("receptor_residue_identifier_absent")
            seen_chains.add(chain)
            receptor.append(atom)
    if seen_chains != set(pose.receptor_chains):
        raise PoseError("selected_protein_chain_absent")
    if len(ligand_residues) != 1:
        raise PoseError("single_selected_ligand_required")
    if not any(atom.element == "C" for atom in ligand):
        raise PoseError("organic_ligand_required")
    return receptor, ligand, _ligand_inventory(block, pose.ligand_component, ligand), LigandBonds.from_cif(
        block, pose.ligand_component
    )


def _match(source: list[Atom], prepared: Preparation, role: str, bonds: LigandBonds | None = None) -> dict:
    from scipy.spatial import cKDTree

    mismatch = role + "_heavy_atom_mismatch"
    if len(source) != len(prepared.atoms):
        raise PoseError(mismatch)
    tree = cKDTree([atom.xyz for atom in source])
    # Max-norm gives each printed coordinate its own rounding allowance.
    # Two neighbours suffice to distinguish a unique match from an ambiguous one,
    # even when many source atoms share exactly the same position.
    distances, neighbours = tree.query([atom.xyz for atom in prepared.atoms], k=2, p=math.inf)
    seen = set()
    matched = {}
    max_delta = 0.0
    for atom, (delta, second), (index, _) in zip(prepared.atoms, distances, neighbours, strict=True):
        if delta > _XYZ_TOL or second <= _XYZ_TOL or index in seen:
            raise PoseError(mismatch)
        original = source[index]
        if atom.element != original.element:
            raise PoseError(mismatch)
        if role == "receptor" and (original.name, original.residue) != (atom.name, atom.residue):
            raise PoseError("receptor_atom_identity_mismatch")
        if bonds is not None:
            matched[atom.serial] = original.name
        seen.add(index)
        max_delta = max(max_delta, float(delta))
    record = {"heavy_atoms": len(seen), "max_axis_delta_angstrom": max_delta}
    if bonds is not None:
        record.update(bonds.verify(prepared, matched))
    return record


def _execute(pose: Pose, receptor: bytes, ligand: bytes) -> dict:
    name = "vina.exe" if os.name == "nt" else "vina"
    configured = os.getenv("AUTODOCK_VINA_EXEC")
    exe = shutil.which(configured) if configured else tool_code_path("autodock-vina", name) or shutil.which(name)
    if not exe:
        raise PoseError("vina_unavailable")
    try:
        binary = Path(exe).resolve(strict=True)
        with binary.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        with TemporaryDirectory(prefix="vina-pose-") as directory:
            root = Path(directory)
            (root / "receptor.pdbqt").write_bytes(receptor)
            (root / "ligand.pdbqt").write_bytes(ligand)
            args = [
                str(binary),
                "--score_only",
                "--scoring",
                "vina",
                "--cpu",
                "1",
                "--seed",
                "1",
                "--spacing",
                str(_SPACING),
                "--receptor",
                "receptor.pdbqt",
                "--ligand",
                "ligand.pdbqt",
            ]
            for axis, center, size in zip("xyz", pose.center, pose.size, strict=True):
                args.extend(["--center_" + axis, str(center), "--size_" + axis, str(size)])
            proc = subprocess.run(
                args,
                cwd=root,
                capture_output=True,
                timeout=120,
                stdin=subprocess.DEVNULL,
                env={**os.environ, "LC_ALL": "C"},
                check=False,
            )
    except subprocess.TimeoutExpired as error:
        raise PoseError("vina_timeout") from error
    except OSError as error:
        raise PoseError("vina_unavailable") from error
    if proc.returncode:
        raise PoseError("vina_rejected_preparation")
    score, version = _reported_score(proc.stdout.decode("ascii", errors="replace"))
    return {
        "status": "scored",
        "score_kcal_mol": score,
        "provider_version": version,
        "provider_sha256": digest,
        "scoring_function": "vina",
        "mode": "score_only",
        "cpu_threads": 1,
        "seed": 1,
    }


def _reported_score(output: str) -> tuple[float, str]:
    versions = re.findall(r"^AutoDock Vina v(\d+\.\d+\.\d+)[ \t]*\r?$", output, re.M)
    lines = [s for s in output.splitlines() if s.strip().startswith("Estimated Free Energy of Binding")]
    if versions != ["1.2.7"]:
        raise PoseError("unsupported_vina_version")
    match = (
        re.fullmatch(
            r"Estimated Free Energy of Binding\s*:\s*([-+]?\d+(?:\.\d+)?)\s+\(kcal/mol\)(?:\s+\[.*\])?",
            lines[0].strip(),
        )
        if len(lines) == 1
        else None
    )
    if not match or not math.isfinite(score := float(match[1])):
        raise PoseError("vina_score_unreadable")
    return score, versions[0]


@dataclass
class RunPoses:
    directory: Path
    settings: dict = field(default_factory=dict)
    manifest_sha256: str | None = None
    blocker: str | None = None
    results: list[dict] = field(default_factory=list)

    @classmethod
    def read(cls, path: Path) -> RunPoses:
        run = cls(path.parent)
        try:
            data = _read(path, 4 * 1024 * 1024, "pose_manifest_unreadable")
            run.manifest_sha256 = hashlib.sha256(data).hexdigest()
            obj = json.loads(data, object_pairs_hook=_unique)
            if (
                not isinstance(obj, dict)
                or obj.keys() != {"samples"}
                or not isinstance(obj["samples"], dict)
                or not all(map(_name, obj["samples"]))
            ):
                raise PoseError("invalid_pose_manifest")
            run.settings = obj["samples"]
        except PoseError as error:
            run.blocker = str(error)
        except (ValueError, RecursionError):
            run.blocker = "invalid_pose_manifest"
        return run

    @classmethod
    def configured(cls, directory: Path) -> RunPoses | None:
        path = directory / MANIFEST
        return cls.read(path) if path.exists() or path.is_symlink() else None

    def score(self, example: str, text: str | None) -> dict:
        identity = {"method": METHOD, "example_id": example, "manifest_sha256": self.manifest_sha256}
        try:
            if self.blocker:
                raise PoseError(self.blocker)
            if text is None:
                raise PoseError("structure_unreadable")
            if example not in self.settings:
                raise PoseError("sample_settings_absent")
            pose = Pose.parse(self.directory, self.settings[example])
            identity.update(pose.selection(), structure_text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest())
            receptor = _read(pose.receptor, _MAX_INPUT_BYTES, "prepared_receptor_unreadable")
            ligand = _read(pose.ligand, _MAX_INPUT_BYTES, "prepared_ligand_unreadable")
            identity.update(
                receptor_pdbqt_sha256=hashlib.sha256(receptor).hexdigest(),
                ligand_pdbqt_sha256=hashlib.sha256(ligand).hexdigest(),
            )
            protein_input = _prepared(receptor, ligand=False)
            ligand_input = _prepared(ligand, ligand=True)
            source_protein, source_ligand, inventory, bonds = _source(text, pose)
            identity["ligand_inventory_basis"] = inventory
            identity["receptor_match"] = _match(source_protein, protein_input, "receptor")
            identity["ligand_match"] = _match(source_ligand, ligand_input, "ligand", bonds)
            for atom in ligand_input.atoms:
                if any(abs(v - c) > s / 2 for v, c, s in zip(atom.xyz, pose.center, pose.size, strict=True)):
                    raise PoseError("ligand_outside_scoring_box")
            result = _execute(pose, receptor, ligand)
        except PoseError as error:
            result = _untested(str(error))
        except ImportError:
            result = _untested("coordinate_reader_unavailable")
        except (ValueError, RuntimeError, OverflowError, IndexError):
            result = _untested("structure_unreadable")
        return {**identity, **result}

    def observe(self, example: str, text: str | None) -> None:
        self.results.append(self.score(example, text))

    def as_record(self) -> dict:
        scored = sum(row["status"] == "scored" for row in self.results)
        unknown = self.settings.keys() - {row["example_id"] for row in self.results}
        return {
            "method": METHOD,
            "corpus_provenance": corpus_provenance(),
            "manifest_sha256": self.manifest_sha256,
            "attempted": len(self.results),
            "scored": scored,
            "untested": len(self.results) - scored,
            "unmatched_manifest_ids": sorted(unknown),
            "blocker": self.blocker or (None if self.results else "designs_absent"),
            "samples": self.results,
        }
