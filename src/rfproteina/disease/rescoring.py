"""Implicit-solvent potential energies for a supplied complex and its components.

:func:`rescore_complex` reports three single-structure potential energies in
kcal/mol. The historical field ``dg_bind`` holds their arithmetic difference,
``E(complex) - E(receptor) - E(ligand)``. The built-in OpenMM backend uses
PDBFixer to place missing atoms and hydrogens independently in each structure.
Repeats on the same coordinates changed the sign of this difference. The
reader preserves the energies as diagnostics and withholds an affinity verdict.

:func:`plan_rbfe` describes the inputs for a separate relative free-energy
calculation. Its return value records a plan and backend availability.
"""

from __future__ import annotations

import importlib.util
import math
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from importlib import import_module
from pathlib import Path

METHOD = "mmgbsa_single_structure_v1"

#: Physical units of every energy this module reports.
ENERGY_UNITS = "kcal/mol"

#: Default implicit-solvent model name passed through to OpenMM. GBn2 (Nguyen et
#: al. 2013) is OpenMM's recommended Generalized Born model. Named here, not
#: silently chosen, so a reviewer can see and change it.
DEFAULT_IMPLICIT_SOLVENT = "GBn2"

#: Default protein force field passed through to OpenMM.
DEFAULT_FORCEFIELD = "amber14-all.xml"

#: Historical project cutoff for an externally supplied energy calculation.
#: The built-in independently prepared energies have no applicable cutoff.
FAVOURABLE_DG_BIND = -5.0

RESCORING_BARS: dict[str, dict[str, object]] = {
    "dg_bind": {
        "max": FAVOURABLE_DG_BIND,
        "source": (
            "suggested default pending calibration for an external energy calculation"
        ),
    },
}


class OpenMMUnavailableError(RuntimeError):
    """Raised when OpenMM is not installed, so no energy can be computed."""


@dataclass(frozen=True, slots=True)
class MMGBSAResult:
    """Three single-structure potential energies and their arithmetic difference."""

    e_complex: float
    e_receptor: float
    e_ligand: float
    dg_bind: float
    units: str
    implicit_solvent: str
    forcefield: str
    minimized: bool
    notes: tuple[str, ...]
    preparation: str | None = None

    def as_record(self) -> dict[str, object]:
        return asdict(self)


def openmm_available() -> bool:
    """True iff OpenMM is importable in this environment."""
    return importlib.util.find_spec("openmm") is not None


_FEP_APIS = {
    "openfe": ("openfe", "Transformation"),
    "pmx": ("pmx.model", "Model"),
    "alchemlyb": ("alchemlyb.estimators", "MBAR"),
}


def _fep_importable(name: str) -> bool:
    module, symbol = _FEP_APIS[name]
    try:
        return callable(getattr(import_module(module), symbol, None))
    except (ImportError, OSError):
        return False


def fep_backends_available() -> dict[str, bool]:
    """Check the scientific Python API of each optional free-energy package."""
    return {name: _fep_importable(name) for name in _FEP_APIS}


# Energy function signature: (pdb_path, implicit_solvent, forcefield, minimize) -> energy in kcal/mol.
EnergyFn = Callable[[Path, str, str, bool], float]


def rescore_complex(
    structure: str | Path,
    receptor_chains: Sequence[str],
    ligand_chains: Sequence[str],
    *,
    minimize: bool = True,
    implicit_solvent: str = DEFAULT_IMPLICIT_SOLVENT,
    forcefield: str = DEFAULT_FORCEFIELD,
    energy_fn: EnergyFn | None = None,
) -> MMGBSAResult:
    """Report potential energies for a complex split into receptor and ligand chains.

    Args:
        structure: A PDB/mmCIF holding the full complex.
        receptor_chains: Chain ids forming the receptor side.
        ligand_chains: Chain ids forming the ligand (binder) side.
        minimize: Whether each subsystem is energy-minimised before the
            single-point energy. Passed through to the energy back-end.
        implicit_solvent: Generalized Born model name (default :data:`DEFAULT_IMPLICIT_SOLVENT`).
        forcefield: Force field xml name (default :data:`DEFAULT_FORCEFIELD`).
        energy_fn: Injection seam. If given, called as
            ``energy_fn(pdb_path, implicit_solvent, forcefield, minimize)`` for
            each of the three subsystems and must return an energy in kcal/mol.
            If ``None``, OpenMM is used.

    Raises:
        FileNotFoundError: If ``structure`` does not exist.
        ValueError: If a chain set is empty or the two sets overlap.
        OpenMMUnavailableError: If ``energy_fn`` is ``None`` and OpenMM is absent.
    """
    path = Path(structure)
    if not path.exists():
        raise FileNotFoundError(f"structure not found: {path}")
    receptor = list(receptor_chains)
    ligand = list(ligand_chains)
    if not receptor or not ligand:
        raise ValueError("both receptor_chains and ligand_chains must be non-empty")
    if set(receptor) & set(ligand):
        raise ValueError(f"receptor and ligand chains overlap: {sorted(set(receptor) & set(ligand))}")

    compute = energy_fn
    if compute is None:
        if not openmm_available():
            raise OpenMMUnavailableError(
                "OpenMM is not installed, so no energy can be computed. Install openmm "
                "(MIT, https://openmm.org) or pass energy_fn=... No energy was invented."
            )
        compute = _openmm_implicit_energy

    import tempfile

    with tempfile.TemporaryDirectory(prefix="mmgbsa_") as tmp:
        tmp_dir = Path(tmp)
        complex_pdb = _write_subset(path, receptor + ligand, tmp_dir / "complex.pdb")
        receptor_pdb = _write_subset(path, receptor, tmp_dir / "receptor.pdb")
        ligand_pdb = _write_subset(path, ligand, tmp_dir / "ligand.pdb")
        e_complex = float(compute(complex_pdb, implicit_solvent, forcefield, minimize))
        e_receptor = float(compute(receptor_pdb, implicit_solvent, forcefield, minimize))
        e_ligand = float(compute(ligand_pdb, implicit_solvent, forcefield, minimize))

    dg_bind = e_complex - e_receptor - e_ligand
    if not all(math.isfinite(value) for value in (e_complex, e_receptor, e_ligand, dg_bind)):
        raise ValueError("nonfinite_subsystem_energy")
    return MMGBSAResult(
        e_complex=e_complex,
        e_receptor=e_receptor,
        e_ligand=e_ligand,
        dg_bind=dg_bind,
        units=ENERGY_UNITS,
        implicit_solvent=implicit_solvent,
        forcefield=forcefield,
        minimized=minimize,
        preparation="independent_subsystems" if energy_fn is None else "external_energy_function",
        notes=(
            "Single-structure potential energy difference in kcal/mol.",
            "The built-in calculation prepares each subsystem independently with PDBFixer. "
            "Repeated preparations changed the sign of the difference."
            if energy_fn is None else "The caller supplied the energy calculation.",
        ),
    )


def _write_subset(source: Path, chains: Sequence[str], destination: Path) -> Path:
    """Write ``source`` restricted to ``chains`` as a PDB, using biotite.

    biotite is already a pinned dependency, so the structure splitting runs
    without OpenMM and is exercised directly in tests. Hydrogens and the chain
    selection are left exactly as the input carries them; solvent/ligand atom
    handling is the energy back-end's concern.
    """
    import numpy as np
    from biotite.structure.io.pdb import PDBFile
    from biotite.structure.io.pdbx import CIFFile, get_structure

    wanted = set(chains)
    suffix = source.suffix.lower()
    if suffix in (".cif", ".mmcif", ".bcif"):
        atoms = get_structure(CIFFile.read(str(source)), model=1)
    else:
        atoms = PDBFile.read(str(source)).get_structure(model=1)
    mask = np.isin(atoms.chain_id, list(wanted))
    if not mask.any():
        raise ValueError(f"no atoms for chains {sorted(wanted)!r} in {source.name}")
    out = PDBFile()
    out.set_structure(atoms[mask])
    out.write(str(destination))
    return destination


def _openmm_implicit_energy(pdb_path: Path, implicit_solvent: str, forcefield: str, minimize: bool) -> float:
    """Single-point implicit-solvent potential energy (kcal/mol) via OpenMM.

    Imported lazily so the module loads without OpenMM, and so tests never need a
    physics engine. Builds an implicit-solvent System from the named force field,
    optionally minimises, and returns the potential energy. PDBFixer (if present)
    is used to add missing hydrogens; without it OpenMM requires the input to
    already carry them, and that requirement is surfaced as a clear error rather
    than guessed around.
    """
    import openmm
    from openmm import app, unit

    # The model is chosen by the XML loaded below and by nothing else. The
    # ``implicitSolvent=`` keyword belongs to ``AmberPrmtopFile.createSystem``,
    # which takes a prmtop and has no force field to read it from;
    # ``ForceField.createSystem`` has never accepted it, and OpenMM 8 refuses
    # the call outright rather than ignoring it ("the argument 'implicitSolvent'
    # was specified ... but was never used"). Passing it here made every real
    # run raise, which the injected ``energy_fn`` in the tests stepped around.
    implicit_models = ("GBn2", "GBn", "OBC1", "OBC2", "HCT")
    if implicit_solvent not in implicit_models:
        raise ValueError(f"unknown implicit solvent {implicit_solvent!r}; expected one of {sorted(implicit_models)}")

    pdb = app.PDBFile(str(pdb_path))
    topology, positions = pdb.topology, pdb.positions
    if importlib.util.find_spec("pdbfixer") is not None:
        from pdbfixer import PDBFixer

        fixer = PDBFixer(filename=str(pdb_path))
        fixer.findMissingResidues()
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(7.0)
        topology, positions = fixer.topology, fixer.positions

    ff = app.ForceField(forcefield, f"implicit/{implicit_solvent.lower()}.xml")
    system = ff.createSystem(
        topology,
        nonbondedMethod=app.CutoffNonPeriodic,
        constraints=app.HBonds,
        soluteDielectric=1.0,
        solventDielectric=78.5,
    )
    integrator = openmm.LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 0.002 * unit.picoseconds)
    context = openmm.Context(system, integrator, openmm.Platform.getPlatformByName("CPU"))
    context.setPositions(positions)
    if minimize:
        openmm.LocalEnergyMinimizer.minimize(context)
    energy = context.getState(getEnergy=True).getPotentialEnergy()
    return energy.value_in_unit(unit.kilocalorie_per_mole)


def plan_rbfe(
    ligand_a: str,
    ligand_b: str,
    *,
    backend: str = "openfe",
) -> dict[str, object]:
    """Describe a backend's role and report whether its scientific Python API imports."""
    known = {
        "openfe": ("OpenFreeEnergy relative hybrid topology", [
            "equilibrium free-energy estimates for the complex and solvent legs",
            "combine completed estimates and report repeat uncertainty",
        ]),
        "pmx": ("pmx nonequilibrium free-energy workflow", [
            "forward and reverse work distributions for each leg",
            "estimate both leg free energies from work and compare them",
        ]),
        "alchemlyb": ("alchemlyb analysis of previously collected samples", [
            "match the recorded data to the estimator",
            "analyse reduced potentials with MBAR or BAR, or energy derivatives with thermodynamic integration",
        ]),
    }
    if backend not in known:
        raise ValueError(f"unknown FEP backend {backend!r}; expected one of {sorted(known)}")
    if not ligand_a.strip() or not ligand_b.strip() or ligand_a == ligand_b:
        raise ValueError("two distinct ligand names are required")
    description, stages = known[backend]
    analysis_commands = {"openfe": "rbfe", "pmx": "pmx-bar", "alchemlyb": "ti"}
    return {
        "method": "rbfe_scaffold_v1",
        "status": "scaffold_only",
        "edge": {"ligand_a": ligand_a, "ligand_b": ligand_b},
        "backend": backend,
        "backend_plan": description,
        "backend_importable": _fep_importable(backend),
        "stages": stages,
        "completed_analysis_command": f"scripts/screen_pose_safety.py {analysis_commands[backend]}",
        "note": "This entry point describes the backend. Its analysis command reads completed simulation data.",
    }


def verdicts(result: MMGBSAResult) -> dict[str, dict[str, object]]:
    """Withhold a verdict for independently prepared subsystem energies."""
    record = result.as_record()
    out: dict[str, dict[str, object]] = {}
    for name, rule in RESCORING_BARS.items():
        value = record.get(name)
        minimum = rule.get("min")
        maximum = rule.get("max")
        independent = result.preparation == "independent_subsystems" or (
            result.preparation is None and any("NOT REPRODUCIBLE" in note for note in result.notes)
        )
        if value is None or not math.isfinite(value) or independent:
            state = "untested"
        else:
            low_ok = minimum is None or value >= minimum
            high_ok = maximum is None or value <= maximum
            state = "clear" if (low_ok and high_ok) else "hit"
        out[name] = {
            "value": value,
            "state": state,
            "min": minimum,
            "max": maximum,
            "threshold_source": rule["source"],
        }
        if independent:
            out[name].update(
                min=None, max=None, threshold_source=None, blocker="independently_prepared_subsystems"
            )
    return out
