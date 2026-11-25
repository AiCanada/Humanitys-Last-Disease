"""Run pinned Jazzy calculations using a single checked RDKit conformer."""

from __future__ import annotations

import hashlib
import json
import math
import stat
import sys
import sysconfig
from contextlib import redirect_stdout
from importlib.metadata import distribution, version
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rdkit.Chem import Mol

VERSIONS = {
    "jazzy": "0.1.4",
    "kallisto": "1.0.10",
    "rdkit": "2024.3.1",
    "numpy": "1.26.4",
    "scipy": "1.13.1",
    "click": "8.5.0",
    "pillow": "13.0.0",
}
MAX_ROWS = 32
MAX_INPUT_BYTES = 64 * 1024
MAX_SMILES = 512
EMBEDDING_SEED = 11
EMBEDDING_ATTEMPTS = 64
MMFF_ITERATIONS = 200
READY = "jazzy-ready"
GEOMETRY_UNAVAILABLE = "jazzy_geometry_unavailable"
CALCULATION_FAILED = "jazzy_calculation_failed"
RESULT_FIELDS = frozenset(
    {
        "hydration_kj_mol",
        "apolar_kj_mol",
        "polar_kj_mol",
        "interaction_kj_mol",
        "ch_donor_strength",
        "xh_donor_strength",
        "acceptor_strength",
    }
)
_SOURCE_FINGERPRINTS = {
    "jazzy": (10, "24b1c158f1ad44bb9f0e89f5b3716f0b853afe69a5173bcb07b2fddff09ec5c5"),
    "kallisto": (19, "959fa062b820952cbdb03f313bd69ccdee2084b9cd65ce97e58c3924caa06b45"),
}
_ELEMENTS = frozenset({1, 6, 7, 8, 9, 15, 16, 17, 35, 53})


def _verify_provider() -> None:
    if sys.version_info[:2] != (3, 12) or any(version(name) != wanted for name, wanted in VERSIONS.items()):
        raise ValueError("unsupported provider dependencies")
    packages = Path(sysconfig.get_path("purelib")).resolve()
    for name, (count, expected) in _SOURCE_FINGERPRINTS.items():
        metadata = distribution(name)
        files = sorted(
            path.as_posix()
            for path in (metadata.files or ())
            if path.parts and path.parts[0] == name and path.suffix == ".py"
        )
        if len(files) != count:
            raise ValueError("unsupported provider source layout")
        digests = []
        for filename in files:
            path = metadata.locate_file(filename)
            info = path.lstat()
            if (
                not stat.S_ISREG(info.st_mode)
                or not 0 <= info.st_size <= 512 * 1024
                or not path.resolve().is_relative_to(packages)
            ):
                raise ValueError("invalid provider source")
            with path.open("rb") as source:
                blob = source.read(512 * 1024 + 1)
            if len(blob) != info.st_size:
                raise ValueError("provider source changed")
            digests.append(f"{filename}:{hashlib.sha256(blob).hexdigest()}")
        if hashlib.sha256("\n".join(digests).encode("ascii")).hexdigest() != expected:
            raise ValueError("unexpected provider source")


def valid_smiles(value: object) -> bool:
    """Return whether a string fits the worker's text boundary."""
    return (
        isinstance(value, str)
        and 0 < len(value) <= MAX_SMILES
        and value.isascii()
        and all(char.isprintable() and not char.isspace() for char in value)
    )


def _requests(raw: bytes) -> list[str]:
    if not 0 < len(raw) <= MAX_INPUT_BYTES:
        raise ValueError("invalid request size")
    smiles = json.loads(raw.decode("ascii"))
    if not isinstance(smiles, list) or not 0 < len(smiles) <= MAX_ROWS or not all(map(valid_smiles, smiles)):
        raise ValueError("invalid molecular request")
    return smiles


def _geometry(smiles: str) -> Mol | None:
    import numpy as np
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdDistGeom

    parser = Chem.SmilesParserParams()
    parser.allowCXSMILES = False
    parser.parseName = False
    molecule = Chem.MolFromSmiles(smiles, parser)
    if (
        molecule is None
        or not 2 <= molecule.GetNumHeavyAtoms() <= 40
        or len(Chem.GetMolFrags(molecule)) != 1
        or not any(atom.GetAtomicNum() == 6 for atom in molecule.GetAtoms())
        or any(
            atom.GetAtomicNum() not in _ELEMENTS
            or atom.GetFormalCharge()
            or atom.GetIsotope()
            or atom.GetNumRadicalElectrons()
            or atom.GetAtomMapNum()
            for atom in molecule.GetAtoms()
        )
        or any(stereo.specified != Chem.StereoSpecified.Specified for stereo in Chem.FindPotentialStereo(molecule))
    ):
        return None
    molecule = Chem.AddHs(molecule)
    if molecule.GetNumAtoms() > 160:
        return None
    parameters = rdDistGeom.ETDG()
    parameters.randomSeed = EMBEDDING_SEED
    parameters.maxIterations = EMBEDDING_ATTEMPTS
    parameters.numThreads = 1
    if rdDistGeom.EmbedMolecule(molecule, parameters) != 0:
        return None
    if AllChem.MMFFOptimizeMolecule(molecule, mmffVariant="MMFF94", maxIters=MMFF_ITERATIONS) != 0:
        return None
    conformer = molecule.GetConformer()
    if not conformer.Is3D() or not np.isfinite(conformer.GetPositions()).all():
        return None
    return molecule


def _calculate(molecule: Mol) -> dict[str, float]:
    from jazzy import core
    from jazzy.config import ROUNDING_DIGITS, Config
    from jazzy.helpers import sum_atomic_map

    parameters = Config()
    kallisto = core.kallisto_molecule_from_rdkit_molecule(molecule)
    charges = core.get_charges_from_kallisto_molecule(kallisto, 0)
    if len(charges) != molecule.GetNumAtoms() or any(not math.isfinite(charge) for charge in charges):
        raise ValueError("invalid partial charges")
    neighbours = core.get_covalent_atom_idxs(molecule)
    atomic = core.calculate_polar_strength_map(molecule, kallisto, neighbours, charges)
    strengths = sum_atomic_map(atomic)
    apolar = core.calculate_delta_apolar(
        molecule,
        atomic,
        parameters.g0,
        parameters.gs,
        parameters.gr,
        parameters.gpi1,
        parameters.gpi2,
    )
    polar = core.calculate_delta_polar(
        atomic,
        neighbours,
        parameters.gd,
        parameters.ga,
        parameters.expd,
        parameters.expa,
    )
    interaction = core.calculate_delta_interaction(
        molecule,
        atomic,
        neighbours,
        parameters.gi,
        parameters.expa,
        parameters.f,
    )
    raw = {
        "hydration_kj_mol": apolar + polar + interaction,
        "apolar_kj_mol": apolar,
        "polar_kj_mol": polar,
        "interaction_kj_mol": interaction,
        "ch_donor_strength": strengths["sdc"],
        "xh_donor_strength": strengths["sdx"],
        "acceptor_strength": strengths["sa"],
    }
    if any(not math.isfinite(value) for value in raw.values()):
        raise ValueError("invalid calculation")
    return {key: round(float(value), ROUNDING_DIGITS) for key, value in raw.items()}


def _row(smiles: str) -> dict[str, str | float]:
    from jazzy.exception import KallistoError, NegativeLonePairsError

    try:
        molecule = _geometry(smiles)
    except (ArithmeticError, RuntimeError, ValueError):
        molecule = None
    if molecule is None:
        return {"blocker": GEOMETRY_UNAVAILABLE}
    try:
        return _calculate(molecule)
    except (ArithmeticError, LookupError, RuntimeError, ValueError, KallistoError, NegativeLonePairsError):
        return {"blocker": CALCULATION_FAILED}


def main() -> int:
    """Check the pinned worker or process one bounded request."""
    from importlib import import_module

    try:
        with redirect_stdout(sys.stderr):
            _verify_provider()
            import_module("jazzy.core")
            if sys.argv[1:] == ["--check-installation"]:
                print(READY, file=sys.__stdout__)
                return 0
            if len(sys.argv) != 1:
                return 1
            from rdkit import rdBase

            with rdBase.BlockLogs():
                requests = _requests(sys.stdin.buffer.read(MAX_INPUT_BYTES + 1))
                calculations = {smiles: _row(smiles) for smiles in dict.fromkeys(requests)}
                rows = [calculations[smiles] for smiles in requests]
    except (ImportError, OSError, RuntimeError, ValueError):
        return 1
    json.dump(rows, sys.stdout, allow_nan=False, separators=(",", ":"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
