"""Score two supplied conformers with a verified, offline ESP-Sim release."""

from __future__ import annotations

import ast
import hashlib
import io
import json
import math
import os
import stat
import struct
import sys
from contextlib import redirect_stdout
from itertools import islice
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from rdkit.Chem import Mol

RELEASE = "0.0.1"
ARCHIVE_URL = (
    "https://files.pythonhosted.org/packages/3d/ab/"
    "608c2a3cc1e409ed720529e67d62a9628d6068a95f4e964da583e3d9f8ea/espsim-0.0.1.tar.gz"
)
ARCHIVE_SHA256 = "0b4438b4a3c8fe57c69dff9cb5e6e4044e9d376078133a79a01c1bbb3d689443"
ARCHIVE_BYTES = 19_765_098
FILES = {
    "LICENSE.txt": (1068, "6af08440bec8f5cf7e3fa67b898a035b55eeca7f2a917d7be6ba84007a8ab592"),
    "espsim/__init__.py": (187, "47356864bac8cff7e3a9a7cb02a1980fe94170ace52b98e3b970b8c342edb94c"),
    "espsim/electrostatics.py": (25195, "04f50f56700f64c7fb564ba1a10e64e6812eeb42c4d71aeeeb425f839cd0b3cf"),
    "espsim/helpers.py": (7653, "45e0a28426aea8b84adfddd821b43304a1912bf643ca3bae6fc7beaa3e046aa1"),
}
MAX_SDF_BYTES = 512 * 1024
MAX_ATOMS = 160
MAX_AXIS_SPAN = 48.0
HEADER = struct.Struct("!II")
READY = "espsim-ready"
CHARGE_UNAVAILABLE = "espsim_charge_unavailable"


def release_directory() -> Path:
    """Name the slim provider installation under the current Python prefix."""
    return Path(sys.prefix) / "share" / "espsim" / RELEASE


def _check_directory(directory: Path, names: set[str]) -> None:
    if (
        not stat.S_ISDIR(directory.lstat().st_mode)
        or {entry.name for entry in islice(directory.iterdir(), len(names) + 1)} != names
    ):
        raise ValueError("unexpected ESP-Sim installation layout")


def verify_assets() -> dict[str, bytes]:
    """Return authenticated provider source and license bytes."""
    release = release_directory()
    for directory in (release.parent.parent, release.parent):
        if not stat.S_ISDIR(directory.lstat().st_mode):
            raise ValueError("invalid provider directory")
    _check_directory(release, {"LICENSE.txt", "espsim"})
    _check_directory(release / "espsim", {"__init__.py", "electrostatics.py", "helpers.py"})
    assets = {}
    for filename, (size, expected) in FILES.items():
        path = release / filename
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_size != size:
            raise ValueError("invalid provider file")
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0)
        with os.fdopen(os.open(path, flags), "rb") as handle:
            if not os.path.samestat(info, os.fstat(handle.fileno())):
                raise ValueError("provider file changed")
            data = handle.read(size + 1)
        if len(data) != size or hashlib.sha256(data).hexdigest() != expected:
            raise ValueError("unexpected provider file")
        assets[filename] = data
    return assets


def _provider() -> SimpleNamespace:
    import numpy as np
    from rdkit.Chem import rdShapeHelpers

    assets = verify_assets()
    # These unchanged functions need only NumPy and the RDKit shape kernel.
    # Never run the upstream module imports, alignment functions, or ML paths.
    context = {"np": np, "AllChem": rdShapeHelpers}
    functions = {"espsim/electrostatics.py": {"GetShapeSim", "GaussInt"}, "espsim/helpers.py": {"SimilarityMetric"}}
    for filename, requested in functions.items():
        tree = ast.parse(assets[filename], filename=filename)
        definitions = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in requested and not node.decorator_list
        ]
        if len(definitions) != len(requested) or {item.name for item in definitions} != requested:
            raise ValueError("unexpected pinned scoring functions")
        code = compile(ast.Module(body=definitions, type_ignores=[]), filename, "exec")
        exec(code, context)
    return SimpleNamespace(
        GetShapeSim=context["GetShapeSim"], GaussInt=context["GaussInt"], SimilarityMetric=context["SimilarityMetric"]
    )


def _split(raw: bytes) -> tuple[bytes, bytes]:
    if len(raw) < HEADER.size:
        raise ValueError("invalid request header")
    candidate_bytes, reference_bytes = HEADER.unpack_from(raw)
    split = HEADER.size + candidate_bytes
    if not (0 < candidate_bytes <= MAX_SDF_BYTES and 0 < reference_bytes <= MAX_SDF_BYTES):
        raise ValueError("invalid background structure size")
    if len(raw) != split + reference_bytes:
        raise ValueError("incomplete molecular request")
    return raw[HEADER.size : split], raw[split:]


def _structure(data: bytes) -> tuple[Mol, np.ndarray]:
    import numpy as np
    from rdkit import Chem

    missing = object()
    with Chem.ForwardSDMolSupplier(io.BytesIO(data), sanitize=True, removeHs=False, strictParsing=True) as supplier:
        molecule = next(supplier, None)
        if molecule is None or next(supplier, missing) is not missing:
            raise ValueError("expected exactly one structure")
    if (
        not 1 <= molecule.GetNumAtoms() <= MAX_ATOMS
        or not molecule.GetNumHeavyAtoms()
        or molecule.GetNumConformers() != 1
        or len(Chem.GetMolFrags(molecule)) != 1
    ):
        raise ValueError("unsupported molecular structure")
    conformer = molecule.GetConformer()
    coordinates = conformer.GetPositions()
    if not conformer.Is3D() or not np.isfinite(coordinates).all() or np.max(np.abs(coordinates)) > 1e6:
        raise ValueError("invalid molecular coordinates")
    return molecule, coordinates


def _charges(molecule: Mol) -> np.ndarray:
    import numpy as np
    from rdkit import Chem
    from rdkit.Chem import rdPartialCharges

    if any(atom.GetTotalNumHs(includeNeighbors=False) for atom in molecule.GetAtoms()):
        raise ValueError("explicit hydrogen coordinates are required")
    for atom in molecule.GetAtoms():
        for prop in ("_GasteigerCharge", "_GasteigerHCharge"):
            if atom.HasProp(prop):
                atom.ClearProp(prop)
    rdPartialCharges.ComputeGasteigerCharges(molecule, nIter=12, throwOnParamFailure=True)
    charges = np.asarray([atom.GetDoubleProp("_GasteigerCharge") for atom in molecule.GetAtoms()], dtype=float)
    if not np.isfinite(charges).all() or abs(float(charges.sum()) - Chem.GetFormalCharge(molecule)) > 0.01:
        raise ValueError("invalid Gasteiger charges")
    return charges


def _distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    import numpy as np

    return np.sqrt(np.square(a[:, None, :] - b[None, :, :]).sum(axis=2))


def _esp(provider: SimpleNamespace, candidate: Mol, reference: Mol, a: np.ndarray, b: np.ndarray) -> float:
    candidate_charge, reference_charge = _charges(candidate), _charges(reference)
    aa = float(provider.GaussInt(_distances(a, a), candidate_charge, candidate_charge))
    bb = float(provider.GaussInt(_distances(b, b), reference_charge, reference_charge))
    if not math.isfinite(aa) or not math.isfinite(bb) or min(aa, bb) <= 1e-12:
        raise ValueError("electrostatic norm is undefined")
    ab = float(provider.GaussInt(_distances(a, b), candidate_charge, reference_charge))
    value = float(provider.SimilarityMetric(aa, bb, ab, "carbo"))
    if not math.isfinite(value) or not -1.000000001 <= value <= 1.000000001:
        raise ValueError("invalid electrostatic overlap")
    return min(1.0, max(-1.0, value))


def _predict(provider: SimpleNamespace, raw: bytes) -> dict[str, float | str]:
    import numpy as np
    from rdkit import rdBase

    candidate_data, reference_data = _split(raw)
    with rdBase.BlockLogs():
        candidate, a = _structure(candidate_data)
        reference, b = _structure(reference_data)
        if np.any(np.ptp(np.concatenate((a, b)), axis=0) > MAX_AXIS_SPAN):
            raise ValueError("comparison frame exceeded its bounds")
        shape = float(
            provider.GetShapeSim(
                candidate,
                reference,
                prbCid=candidate.GetConformer().GetId(),
                refCid=reference.GetConformer().GetId(),
            )
        )
        if not math.isfinite(shape) or not 0 <= shape <= 1:
            raise ValueError("invalid shape overlap")
        try:
            electrostatic = _esp(provider, candidate, reference, a, b)
        except (ArithmeticError, LookupError, RuntimeError, ValueError):
            return {"shape_tanimoto": shape, "blocker": CHARGE_UNAVAILABLE}
    return {"shape_tanimoto": shape, "electrostatic_carbo": electrostatic}


def main() -> int:
    """Verify the provider or return one bounded numeric comparison."""
    try:
        with redirect_stdout(sys.stderr):
            provider = _provider()
            if sys.argv[1:] == ["--check-installation"]:
                print(READY, file=sys.__stdout__)
                return 0
            if len(sys.argv) != 1:
                return 1
            data = sys.stdin.buffer.read(HEADER.size + 2 * MAX_SDF_BYTES + 1)
            result = _predict(provider, data)
    except (ArithmeticError, ImportError, OSError, RuntimeError, TypeError, ValueError):
        return 1
    json.dump(result, sys.stdout, allow_nan=False, separators=(",", ":"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
