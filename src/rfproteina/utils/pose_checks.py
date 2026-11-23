"""Read the boolean check table returned by PoseBusters for one ligand pose."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from rdkit.Chem import Mol


POSEBUSTERS_MOL_CHECKS = (
    "mol_pred_loaded", "sanitization", "inchi_convertible", "all_atoms_connected", "no_radicals",
    "bond_lengths", "bond_angles", "internal_steric_clash", "aromatic_ring_flatness",
    "non-aromatic_ring_non-flatness", "double_bond_flatness", "internal_energy",
)
POSEBUSTERS_DOCK_CHECKS = (
    *POSEBUSTERS_MOL_CHECKS, "mol_cond_loaded", "protein-ligand_maximum_distance",
    "minimum_distance_to_protein", "minimum_distance_to_organic_cofactors",
    "minimum_distance_to_inorganic_cofactors", "minimum_distance_to_waters",
    "volume_overlap_with_protein", "volume_overlap_with_organic_cofactors",
    "volume_overlap_with_inorganic_cofactors", "volume_overlap_with_waters",
)


class PoseBustersUnavailableError(Exception):
    """The PoseBusters provider is unavailable in this environment."""


class PoseBustersResultError(Exception):
    """The backend could not return one attributable set of check results."""


def molecule_checks_available() -> bool:
    """Inspect the installed ligand-only checklist without running any geometry tests."""
    from posebusters import PoseBusters
    from rdkit.Chem import inchi
    from yaml import YAMLError

    try:
        config = PoseBusters(config="mol", max_workers=0).config
        outputs = [
            module["rename_outputs"].get(key, key).lower().replace(" ", "_")
            for module in config["modules"]
            for key in module.get("chosen_binary_test_output", ())
        ]
    except (AttributeError, KeyError, OSError, TypeError, ValueError, YAMLError):
        return False
    return inchi.INCHI_AVAILABLE and all(outputs.count(check) == 1 for check in POSEBUSTERS_MOL_CHECKS)


def pose_checks(mol: Mol, receptor: Mol | None = None) -> tuple[str, dict[str, bool | None]]:
    from pandas import DataFrame
    from rdkit import Chem

    try:
        from posebusters import PoseBusters
    except ImportError as error:
        raise PoseBustersUnavailableError("posebusters_unavailable") from error
    mode = "mol" if receptor is None else "dock"
    try:
        frame = PoseBusters(config=mode, max_workers=0).bust([Chem.Mol(mol)], None, receptor)
    except TimeoutError:
        raise
    except Exception as error:
        raise PoseBustersResultError(f"posebusters_run_failed:{type(error).__name__}") from error
    if not isinstance(frame, DataFrame):
        raise PoseBustersResultError("posebusters_invalid_check_table")
    if len(frame) != 1:
        raise PoseBustersResultError("posebusters_no_rows" if len(frame) == 0 else "posebusters_multiple_rows")
    if not frame.columns.is_unique:
        raise PoseBustersResultError("posebusters_duplicate_checks")
    required = POSEBUSTERS_DOCK_CHECKS if mode == "dock" else POSEBUSTERS_MOL_CHECKS
    checks: dict[str, bool | None] = dict.fromkeys(required)
    for name, value in frame.iloc[0].items():
        checks[name] = bool(value) if isinstance(value, bool | np.bool_) else None
    return mode, checks
