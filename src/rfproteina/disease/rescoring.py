"""Late-stage physics rescoring of a binder-target complex.

Two tiers, both free and open:

1. :func:`rescore_complex` -- a single-structure MM-GBSA-style binding free
   energy estimate using OpenMM (MIT for the API, Reference and CPU platforms,
   verified from OpenMM's own ``docs-source/licenses/Licenses.txt`` on
   2026-11-16) with an implicit (Generalized Born) solvent. This is the working,
   tested tier.

2. :func:`plan_rbfe` -- a documented scaffold for a relative binding free energy
   (RBFE / FEP) calculation built on OpenFE (MIT), pmx (LGPL-3.0) or alchemlyb
   (BSD-3), all verified from their own LICENSE files on 2026-11-16. FEP needs an
   explicit-solvent MD campaign that is out of this pass's compute budget, so the
   scaffold names the pieces and refuses to fabricate a free energy.

What MM-GBSA here is, and is not
--------------------------------
The estimate is the classic single-trajectory decomposition::

    dG_bind ~= E(complex) - E(receptor) - E(ligand)

with each energy evaluated in implicit solvent on one structure (optionally
energy-minimised first). It is **not** a converged MM-GBSA: it averages over no
MD ensemble and includes no explicit configurational entropy term. It is a fast,
reproducible rescoring signal to rank designs, not a calibrated affinity. Real
MM-GBSA averages the decomposition over a trajectory; that is a documented
follow-up, not this function. No absolute-affinity threshold is invented: more
negative is stronger binding, and that is all the verdict claims.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
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

#: A single-structure dG_bind at or below this (kcal/mol) is flagged as a
#: favourable interaction. Suggested default pending calibration: a
#: single-structure MM-GBSA number has no published pass mark and is only
#: meaningful as a relative ranking across this program's own designs.
FAVOURABLE_DG_BIND = -5.0

RESCORING_BARS: dict[str, dict[str, object]] = {
    "dg_bind": {
        "max": FAVOURABLE_DG_BIND,
        "source": (
            "suggested default pending calibration; single-structure MM-GBSA has no absolute "
            "pass mark and is only a relative ranking signal"
        ),
    },
}


class OpenMMUnavailableError(RuntimeError):
    """Raised when OpenMM is not installed, so no energy can be computed."""


@dataclass(frozen=True, slots=True)
class MMGBSAResult:
    """One single-structure MM-GBSA-style binding estimate."""

    e_complex: float
    e_receptor: float
    e_ligand: float
    dg_bind: float
    units: str
    implicit_solvent: str
    forcefield: str
    minimized: bool
    notes: tuple[str, ...]

    def as_record(self) -> dict[str, object]:
        return asdict(self)


def openmm_available() -> bool:
    """True iff OpenMM is importable in this environment."""
    return importlib.util.find_spec("openmm") is not None


def fep_backends_available() -> dict[str, bool]:
    """Which free-energy back-ends are importable: ``openfe``, ``pmx``, ``alchemlyb``."""
    return {name: importlib.util.find_spec(name) is not None for name in ("openfe", "pmx", "alchemlyb")}


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
    """Estimate dG_bind for a complex split into receptor and ligand chains.

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
    return MMGBSAResult(
        e_complex=round(e_complex, 4),
        e_receptor=round(e_receptor, 4),
        e_ligand=round(e_ligand, 4),
        dg_bind=round(dg_bind, 4),
        units=ENERGY_UNITS,
        implicit_solvent=implicit_solvent,
        forcefield=forcefield,
        minimized=minimize,
        notes=(
            "single-structure MM-GBSA-style estimate; not averaged over an MD ensemble",
            "no explicit configurational entropy term is included",
            "use as a relative ranking across designs, not as a calibrated affinity",
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

    implicit_map = {
        "GBn2": app.GBn2,
        "GBn": app.GBn,
        "OBC1": app.OBC1,
        "OBC2": app.OBC2,
        "HCT": app.HCT,
    }
    if implicit_solvent not in implicit_map:
        raise ValueError(f"unknown implicit solvent {implicit_solvent!r}; expected one of {sorted(implicit_map)}")

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
        implicitSolvent=implicit_map[implicit_solvent],
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
    """Describe (do not run) a relative binding free energy calculation.

    FEP/RBFE needs an explicit-solvent, lambda-windowed MD campaign and MBAR
    analysis, which is out of this pass's compute budget. Rather than fabricate a
    dG, this returns the concrete plan and the free, licence-cleared back-end to
    run it with, and reports whether that back-end is importable.

    Args:
        ligand_a: Identifier of the reference ligand in the perturbation.
        ligand_b: Identifier of the ligand being compared to it.
        backend: One of ``openfe`` (MIT), ``pmx`` (LGPL-3.0) or ``alchemlyb`` (BSD-3).

    Raises:
        ValueError: If ``backend`` is not one of the three known back-ends.
    """
    known = {
        "openfe": "OpenFE (MIT): RelativeHybridTopologyProtocol over an OpenMM MD campaign",
        "pmx": "pmx (LGPL-3.0): hybrid-topology non-equilibrium RBFE with GROMACS",
        "alchemlyb": "alchemlyb (BSD-3): MBAR/BAR estimation over dHdl/u_nk from an existing MD campaign",
    }
    if backend not in known:
        raise ValueError(f"unknown FEP backend {backend!r}; expected one of {sorted(known)}")
    return {
        "method": "rbfe_scaffold_v1",
        "status": "scaffold_only",
        "edge": {"ligand_a": ligand_a, "ligand_b": ligand_b},
        "backend": backend,
        "backend_plan": known[backend],
        "backend_importable": fep_backends_available()[backend],
        "stages": [
            "atom-mapping between ligand_a and ligand_b",
            "solvate and parameterise both legs (complex, solvent)",
            "run lambda-windowed MD (out of this pass's compute budget)",
            "estimate dG with MBAR via alchemlyb",
        ],
        "note": "no free energy is computed here; this is a plan, not a result",
    }


def verdicts(result: MMGBSAResult) -> dict[str, dict[str, object]]:
    """Apply the suggested dg_bind bar, carrying that it is only suggested."""
    record = result.as_record()
    out: dict[str, dict[str, object]] = {}
    for name, rule in RESCORING_BARS.items():
        value = record.get(name)
        minimum = rule.get("min")
        maximum = rule.get("max")
        if value is None:
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
    return out
