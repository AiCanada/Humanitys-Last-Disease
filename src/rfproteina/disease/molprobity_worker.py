"""Private, process-isolated CCTBX 2025.11 worker. No network or project dependencies."""

from __future__ import annotations

import importlib.util
import json
import math
import os
import re
import sys
from collections.abc import Callable
from contextlib import redirect_stdout
from importlib import metadata
from pathlib import Path
from typing import Any

VERSION = "2025.11"
MAX_BYTES = 16 * 1024 * 1024
MAX_ATOMS = 16_000
PROVIDER = {"cctbx_base": VERSION, "device": "cpu", "clash_preparation": "reduce2_no_flips", "clash_method": "probe2"}
_BACKBONE = frozenset({"N", "CA", "C", "O"})
_SIDECHAINS = {
    "ALA": "CB", "ARG": "CB CG CD NE CZ NH1 NH2", "ASN": "CB CG OD1 ND2", "ASP": "CB CG OD1 OD2",
    "CYS": "CB SG", "GLN": "CB CG CD OE1 NE2", "GLU": "CB CG CD OE1 OE2", "GLY": "", "HIS": "CB CG ND1 CD2 CE1 NE2",
    "ILE": "CB CG1 CG2 CD1", "LEU": "CB CG CD1 CD2", "LYS": "CB CG CD CE NZ", "MET": "CB CG SD CE",
    "PHE": "CB CG CD1 CD2 CE1 CE2 CZ", "PRO": "CB CG CD", "SER": "CB OG", "THR": "CB OG1 CG2",
    "TRP": "CB CG CD1 CD2 NE1 CE2 CE3 CZ2 CZ3 CH2", "TYR": "CB CG CD1 CD2 CE1 CE2 CZ OH", "VAL": "CB CG1 CG2",
}
_WATERS = {"HOH", "WAT", "DOD"}


class MolProbityWorkerError(ValueError):
    """A stable reason the submitted coordinates cannot be validated."""


def _unavailable(blocker: str) -> dict:
    return {"status": "untested", "blocker": blocker}


def _inspect(hierarchy: Any) -> str | None:
    """Reject unsafe models. Distinguish torsion eligibility from full-atom coverage."""
    models = hierarchy.models()
    if len(models) != 1:
        raise MolProbityWorkerError("molprobity_unsupported_model")
    atoms = hierarchy.atoms()
    if not len(atoms) or len(atoms) > MAX_ATOMS:
        raise MolProbityWorkerError("molprobity_input_over_limit")
    lo, hi = [math.inf] * 3, [-math.inf] * 3
    can_report_contacts = True
    for atom in atoms:
        if (not all(math.isfinite(float(d)) for d in (*atom.xyz, atom.occ, atom.b))
                or not 0 <= float(atom.occ) <= 1):
            raise MolProbityWorkerError("molprobity_structure_invalid")
        for i, coord in enumerate(atom.xyz):
            if abs(coord) > 1_000_000:
                raise MolProbityWorkerError("molprobity_input_over_limit")
            lo[i], hi[i] = min(lo[i], coord), max(hi[i], coord)
            can_report_contacts &= -999.999 <= coord <= 9999.999
    if any(b - a > 1000 for a, b in zip(lo, hi, strict=True)):
        raise MolProbityWorkerError("molprobity_input_over_limit")
    residues = 0
    complete = True
    chain_ids = set()
    for chain in models[0].chains():
        if chain.id in chain_ids:
            raise MolProbityWorkerError("molprobity_unsupported_model")
        chain_ids.add(chain.id)
        can_report_contacts &= bool(re.fullmatch(r"[A-Za-z0-9 ]?", chain.id))
        residue_ids = set()
        for residue in chain.residue_groups():
            residue_id = (residue.resseq, residue.icode)
            if residue_id in residue_ids:
                raise MolProbityWorkerError("molprobity_structure_invalid")
            residue_ids.add(residue_id)
            can_report_contacts &= bool(re.fullmatch(r" *-?[0-9]+", residue.resseq)) and len(residue.resseq) <= 4
            can_report_contacts &= bool(re.fullmatch(r"[A-Za-z0-9 ]?", residue.icode))
            groups = residue.atom_groups()
            names = {g.resname.upper().strip() for g in groups}
            if len(names) != 1:
                raise MolProbityWorkerError("molprobity_unsupported_model")
            name = next(iter(names))
            if name not in _SIDECHAINS and name not in _WATERS:
                raise MolProbityWorkerError("molprobity_unsupported_model")
            atoms_by_alt = {}
            for group in groups:
                alt = group.altloc.strip()
                if alt in atoms_by_alt:
                    raise MolProbityWorkerError("molprobity_structure_invalid")
                can_report_contacts &= bool(re.fullmatch(r"[A-Za-z0-9]?", alt))
                heavy_atoms = [atom for atom in group.atoms() if atom.element.strip().upper() not in {"H", "D"}]
                heavy = [atom.name.strip().upper() for atom in heavy_atoms]
                if any(not label or atom.element.strip().upper() != label[0] for atom, label in zip(
                    heavy_atoms, heavy, strict=True,
                )):
                    raise MolProbityWorkerError("molprobity_structure_invalid")
                if len(heavy) != len(set(heavy)):
                    raise MolProbityWorkerError("molprobity_structure_invalid")
                if name not in _WATERS and any(float(atom.occ) == 0 for atom in heavy_atoms):
                    complete = False
                atoms_by_alt[alt] = set(heavy)
            if name in _WATERS:
                if any(group - {"O"} for group in atoms_by_alt.values()):
                    raise MolProbityWorkerError("molprobity_unsupported_model")
                continue
            residues += 1
            expected = _BACKBONE | set(_SIDECHAINS[name].split())
            shared = atoms_by_alt.get("", set())
            alternatives = [group for key, group in atoms_by_alt.items() if key] or [set()]
            for alternative in alternatives:
                present = shared | alternative
                if shared & alternative or present - (expected | {"OXT"}):
                    raise MolProbityWorkerError("molprobity_unsupported_model")
                complete &= expected <= present
    if not residues:
        raise MolProbityWorkerError("molprobity_unsupported_model")
    if not can_report_contacts:
        return "molprobity_unsupported_model"
    return None if complete else "molprobity_incomplete_heavy_atoms"


def _torsion(analysis: Any) -> dict:
    if not analysis.n_total:
        return _unavailable("molprobity_no_eligible_residues")
    return {
        "status": "reported", "evaluated": analysis.n_total, "outliers": analysis.n_outliers,
        "favored": analysis.n_favored, "allowed": analysis.n_allowed,
    }


def _monomer_library_available() -> bool:
    from mmtbx.monomer_library import server

    return all(server.find_mon_lib_file(relative_path_components=parts) for parts in (
        ["list", "mon_lib_list.cif"], ["ener_lib.cif"],
    ))


def _ramachandran(hierarchy: Any) -> dict:
    from mmtbx.validation.ramalyze import ramalyze

    return _torsion(ramalyze(pdb_hierarchy=hierarchy, outliers_only=False, quiet=True))


def _rotamers(hierarchy: Any) -> dict:
    from mmtbx.rotamer.rotamer_eval import find_rotarama_data_dir
    from mmtbx.validation.rotalyze import rotalyze

    reference = find_rotarama_data_dir(optional=True)
    if not reference or not (Path(reference) / "rota8000-arg.pickle").is_file() or not _monomer_library_available():
        return _unavailable("molprobity_reference_unavailable")
    return _torsion(rotalyze(pdb_hierarchy=hierarchy, data_version="8000", outliers_only=False, quiet=True))


def _clashscore(data_manager: Any) -> dict:
    from libtbx.phil import parse
    from mmtbx.chemical_components import is_code
    from mmtbx.probe import Helpers
    from mmtbx.validation.clashscore2 import clashscore2

    residues = {group.resname.strip().upper() for group in data_manager.get_model().get_hierarchy().atom_groups()}
    if not _monomer_library_available() or not all(is_code(name) for name in residues - _WATERS):
        return _unavailable("molprobity_reference_unavailable")
    result = clashscore2(
        parse(Helpers.probe_phil_parameters).extract().probe, data_manager=data_manager, fast=True,
        condensed_probe=True, keep_hydrogens=False, nuclear=False, do_flips=False, out=sys.stderr,
    )
    manager = result.probe_clashscore_manager
    atoms, clashes = manager.n_atoms, manager.n_clashes
    if not atoms:
        return _unavailable("molprobity_no_eligible_contacts")
    if not math.isclose(result.get_clashscore(), 1000 * clashes / atoms, rel_tol=1e-10, abs_tol=1e-10):
        return _unavailable("molprobity_metric_failed")
    return {"status": "reported", "atoms": atoms, "serious_clashes": clashes}


def _try_metric(metric: Callable[[Any], dict], source: Any) -> dict:
    try:
        return metric(source)
    except (ImportError, FileNotFoundError):
        return _unavailable("molprobity_reference_unavailable")
    except Exception:
        # Different C++ entry points surface different exception classes. Do
        # not discard successful independent metrics or expose native logs.
        return _unavailable("molprobity_metric_failed")


def measure(raw: bytes) -> dict:
    from iotbx.data_manager import DataManager

    if not raw or len(raw) > MAX_BYTES:
        return {"blocker": "molprobity_input_over_limit"}
    path = Path("input.cif")
    with path.open("xb") as stream:
        stream.write(raw)
    manager = DataManager(datatypes=["model"], custom_options=["model_skip_expand_with_mtrix"])
    try:
        manager.process_model_file(str(path))
        model = manager.get_model()
        cif_model = getattr(model.get_model_input(), "cif_model", None)
        if cif_model is None or len(cif_model) != 1:
            return {"blocker": "molprobity_unsupported_model"}
        polymer_types = next(iter(cif_model.values())).get("_entity_poly.type", ())
        polymer_types = (polymer_types,) if isinstance(polymer_types, str) else polymer_types
        if any(str(polymer).casefold() == "polypeptide(d)" for polymer in (polymer_types or ())):
            return {"blocker": "molprobity_unsupported_model"}
        hierarchy = model.get_hierarchy()
        clash_blocker = _inspect(hierarchy)
    except MolProbityWorkerError as error:
        return {"blocker": str(error)}
    except Exception:
        return {"blocker": "molprobity_structure_invalid"}
    return {"provider": PROVIDER, "metrics": {
        "ramachandran": _try_metric(_ramachandran, hierarchy),
        "rotamers": _try_metric(_rotamers, hierarchy),
        "clashscore": _unavailable(clash_blocker) if clash_blocker else _try_metric(_clashscore, manager),
    }}


def main() -> None:
    os.environ.update(dict.fromkeys(("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"), ""))
    with redirect_stdout(sys.stderr):
        try:
            if metadata.version("cctbx-base") != VERSION:
                raise ValueError("incompatible CCTBX release")
            action = sys.argv[1:] or ["probe"]
            if action == ["probe"]:
                required = ("iotbx", "mmtbx", "mmtbx_validation_ramachandran_ext")
                if not all(importlib.util.find_spec(name) for name in required):
                    raise ImportError("missing CCTBX components")
                response = {"cctbx_base": VERSION}
            elif action == ["measure"]:
                response = measure(sys.stdin.buffer.read(MAX_BYTES + 1))
            else:
                response = {"blocker": "molprobity_provider_failure"}
        except (ValueError, ImportError, metadata.PackageNotFoundError):
            response = {"blocker": "molprobity_runtime_unavailable"}
    print(json.dumps(response, allow_nan=False))


if __name__ == "__main__":
    main()
