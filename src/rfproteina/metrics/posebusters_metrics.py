"""PoseBusters validation metric for ligand quality assessment."""

from typing import Any, Literal

import numpy as np
from atomworks.constants import METAL_ELEMENTS
from atomworks.io.tools.posebusters import DEFAULT_EXCLUDED_CHECKS, validate_organic_fragments
from atomworks.io.tools.rdkit import atom_array_to_rdkit
from atomworks.io.utils.annotator import ensure_annotations
from atomworks.ml.utils.timer import timeout
from beartype.typing import Sequence
from biotite.structure import AtomArray, AtomArrayStack

from rfproteina.metrics.metric import Metric
from rfproteina.metrics.utils.metrics_utils import normalize_to_atom_array_list
from rfproteina.utils.ddp import RankedLogger
from rfproteina.utils.pose_checks import PoseBustersResultError, PoseBustersUnavailableError, pose_checks

ranked_logger = RankedLogger(__name__, rank_zero_only=True)


def _ligand_has_metal(ligand_atoms: AtomArray) -> bool:
    """Check if a ligand contains any metal atoms."""
    elements = np.char.upper(np.char.strip(ligand_atoms.element.astype(str)))
    return any(e in METAL_ELEMENTS for e in elements)


def validate_single_ligand(
    ligand_atoms: AtomArray,
    ligand_id: str = "ligand",
    excluded_checks: frozenset[str] | None = None,
    timeout_seconds: float | None = 5.0,
) -> dict[str, Any]:
    """Validate a single ligand molecule with PoseBusters.

    Report a whole-ligand verdict. AtomWorks' organic-fragment diagnostics for
    metal-containing ligands are recorded separately when a whole pose fails.

    Args:
      ligand_atoms: AtomArray containing a single ligand molecule.
      ligand_id: Identifier for logging. Defaults to ``"ligand"``.
      excluded_checks: Check names to exclude from pass/fail calculation.
        If ``None``, uses the atomworks default excluded checks.
      timeout_seconds: Timeout in seconds. Defaults to ``5.0``.
        Set to ``None`` to disable.

    Returns:
      Dictionary with check results and metadata.
    """
    if excluded_checks is None:
        excluded_checks = DEFAULT_EXCLUDED_CHECKS

    base_result: dict[str, Any] = {
        "ligand_id": ligand_id,
        "n_atoms": len(ligand_atoms),
    }

    def _run() -> dict[str, Any]:
        coordinates = ligand_atoms.coord
        observed = np.isfinite(coordinates).all(axis=1)
        # AtomWorks can expand the CCD template with unobserved hydrogens.
        # Their absent coordinates do not invalidate measured heavy-atom geometry.
        template_h = (ligand_atoms.element == "H") & np.isnan(coordinates).all(axis=1)
        if "occupancy" in ligand_atoms.get_annotation_categories():
            template_h &= ligand_atoms.occupancy == 0
        if not observed.any() or not (observed | template_h).all():
            raise PoseBustersResultError("ligand_coordinates_absent_or_nonfinite")
        base_result["n_unresolved_hydrogens"] = int(template_h.sum())
        observed_atoms = ligand_atoms[observed] if template_h.any() else ligand_atoms
        mol = atom_array_to_rdkit(
            observed_atoms, set_coord=True, hydrogen_policy="keep", infer_bonds=False,
            sanitize=True, assume_metal_bonds_are_dative=True,
        )
        _, checks = pose_checks(mol)
        reported = {f"pb_{name}": value for name, value in checks.items()}
        relevant = [value for key, value in reported.items() if key not in excluded_checks]
        if checks.get("mol_pred_loaded") is False:
            raise PoseBustersResultError("posebusters_input_not_loaded")
        passed = None
        if False in relevant:
            passed = False
        elif relevant and all(value is True for value in relevant):
            passed = True
        result = {**base_result, **reported, "passes_validation": passed}
        if passed is None:
            result.update(skipped=True, blocker="posebusters_checks_incomplete")
        if passed is not True and _ligand_has_metal(ligand_atoms):
            try:
                _, fragments = validate_organic_fragments(mol, excluded_checks=excluded_checks)
                result["organic_fragment_checks"] = fragments
            except (TimeoutError, ValueError, RuntimeError) as error:
                result["organic_fragment_blocker"] = type(error).__name__
        return result

    try:
        if timeout_seconds is not None:
            return timeout(timeout_seconds, strategy="signal")(_run)()
        return _run()
    except TimeoutError:
        ranked_logger.warning(f"Validation timed out for ligand {ligand_id} after {timeout_seconds}s")
        return {**base_result, "timed_out": True, "passes_validation": None, "blocker": "posebusters_timed_out"}
    except (PoseBustersResultError, PoseBustersUnavailableError) as error:
        return {**base_result, "skipped": True, "passes_validation": None, "blocker": str(error)}
    except (ValueError, RuntimeError, KeyError, IndexError) as error:
        return {**base_result, "skipped": True, "passes_validation": None,
                "blocker": f"ligand_conversion_failed:{type(error).__name__}"}


def validate_ligands(
    atom_array: AtomArray,
    excluded_checks: Sequence[str] | None = None,
    timeout_seconds: float | None = 5.0,
) -> list[dict[str, Any]]:
    """Validate all ligands in a structure with PoseBusters.

    Identifies ligand molecules by the ``is_ligand`` annotation and validates
    each distinct chain, residue number, insertion code and residue name separately.

    Args:
      atom_array: Structure containing ligands to validate.
      excluded_checks: Check names to exclude from pass/fail calculation.
      timeout_seconds: Timeout in seconds for each ligand validation.
        Defaults to ``5.0``. Set to ``None`` to disable.

    Returns:
      List of per-ligand validation results. Returns empty list if no
      ligands found.
    """
    ensure_annotations(atom_array, "is_ligand")

    ligand_mask = atom_array.is_ligand
    if not np.any(ligand_mask):
        return []

    ligand_atoms = atom_array[ligand_mask]
    excluded_frozen = frozenset(excluded_checks) if excluded_checks is not None else None

    residue_keys = zip(ligand_atoms.chain_id, ligand_atoms.res_id, ligand_atoms.ins_code, ligand_atoms.res_name,
                       strict=True)
    groups: dict[tuple, list[int]] = {}
    for index, key in enumerate(residue_keys):
        groups.setdefault(key, []).append(index)

    results = []
    for (chain_id, res_id, ins_code, res_name), indices in groups.items():
        single_ligand = ligand_atoms[indices]
        ligand_id = f"{chain_id}/{res_name}/{res_id}{ins_code}"
        result = validate_single_ligand(
            single_ligand,
            ligand_id=ligand_id,
            excluded_checks=excluded_frozen,
            timeout_seconds=timeout_seconds,
        )
        result["has_metal"] = _ligand_has_metal(single_ligand)
        results.append(result)

    return results


def aggregate_ligand_results(
    ligand_results: list[dict[str, Any]],
    output_level: Literal["aggregate", "per_check", "per_ligand"] = "per_check",
) -> dict[str, Any]:
    """Aggregate per-ligand validation results into summary statistics.

    Args:
      ligand_results: List of per-ligand validation results from
        :py:func:`validate_ligands` or :py:func:`validate_single_ligand`.
      output_level: Controls output verbosity:
        - ``"aggregate"``: Summary statistics only
        - ``"per_check"``: Adds per-check pass rates
        - ``"per_ligand"``: Adds detailed per-ligand results

    Returns:
      Aggregated results dictionary.
    """
    n = len(ligand_results)
    attempted = [r for r in ligand_results if not r.get("timed_out")]
    measured = [r for r in attempted if not r.get("skipped") and isinstance(r.get("passes_validation"), bool)
                and any(k.startswith("pb_") and isinstance(v, bool) for k, v in r.items())]
    n_passed = sum(r["passes_validation"] for r in measured)
    output: dict[str, Any] = {
        "pass_rate": n_passed / n if measured else float("nan"),
        "measured_pass_rate": n_passed / len(measured) if measured else float("nan"),
        "n_ligands": n,
        "n_measured": len(measured),
        "n_unmeasured": n - len(measured),
        "n_passed": n_passed,
        "n_timed_out": sum(r.get("timed_out", False) for r in ligand_results),
        "n_skipped": sum(r.get("skipped", False) for r in ligand_results),
    }

    # Add per-check pass rates if requested
    if output_level in ("per_check", "per_ligand"):
        for check_name in sorted({k for r in ligand_results for k in r if k.startswith("pb_")}):
            values = [r[check_name] for r in attempted if isinstance(r.get(check_name), bool)]
            output[f"{check_name}_n_measured"] = len(values)
            output[f"{check_name}_pass_rate"] = sum(values) / n if values else float("nan")

    # Add per-ligand results if requested
    if output_level == "per_ligand":
        for i, r in enumerate(ligand_results):
            for k, v in r.items():
                if k == "ligand_id":
                    output[f"ligand_{i}_id"] = v
                elif k in {"blocker", "organic_fragment_blocker"} or (k == "passes_validation" and v is None):
                    output[f"ligand_{i}_{k}"] = v
                elif k == "organic_fragment_checks":
                    for name, flag in v.items():
                        output[f"ligand_{i}_organic_fragment_{name}"] = float(flag)
                elif isinstance(v, bool):
                    output[f"ligand_{i}_{k}"] = float(v)
                elif isinstance(v, int | float):
                    output[f"ligand_{i}_{k}"] = v

    return output


class PoseBustersMetric(Metric):
    """Validate ligand structures using PoseBusters stereochemical checks.

    Args:
      excluded_checks: Check names to exclude from pass/fail calculation.
      output_level: Controls output verbosity.
      timeout_seconds: Timeout per ligand in seconds. Defaults to ``5.0``.
      input_kwarg_names: Mapping from compute() parameter names to input kwargs.
    """

    def __init__(
        self,
        excluded_checks: Sequence[str] | None = None,
        output_level: Literal["aggregate", "per_check", "per_ligand"] = "per_check",
        timeout_seconds: float | None = 5.0,
        input_kwarg_names: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize PoseBusters metric."""
        self._input_kwarg_names = input_kwarg_names or {
            "atom_array_stack": "predicted_atom_array",
        }
        self.excluded_checks = list(excluded_checks) if excluded_checks is not None else None
        self.output_level = output_level
        self.timeout_seconds = timeout_seconds
        super().__init__(**kwargs)

    @property
    def kwargs_to_compute_args(self) -> dict[str, Any]:
        """Map validation inputs to compute() parameters."""
        return self._input_kwarg_names

    def compute(self, atom_array_stack: "AtomArray | AtomArrayStack | list[AtomArray]") -> dict[str, Any]:
        """Compute PoseBusters validation for ligands in the structure(s).

        Args:
          atom_array_stack: Structure(s) containing ligands to validate.

        Returns:
          Dictionary with validation results based on ``output_level``.
        """
        atom_arrays = normalize_to_atom_array_list(atom_array_stack)

        all_ligand_results: list[dict[str, Any]] = []
        for atom_array in atom_arrays:
            ligand_results = validate_ligands(
                atom_array,
                excluded_checks=self.excluded_checks,
                timeout_seconds=self.timeout_seconds,
            )
            all_ligand_results.extend(ligand_results)

        output = aggregate_ligand_results(all_ligand_results, output_level=self.output_level)

        if all_ligand_results:
            metal_results = [r for r in all_ligand_results if r.get("has_metal", False)]
            organic_results = [r for r in all_ligand_results if not r.get("has_metal", False)]
            for prefix, subset in [("metal", metal_results), ("organic", organic_results)]:
                sub = aggregate_ligand_results(subset, output_level=self.output_level)
                for k, v in sub.items():
                    output[f"{prefix}_{k}"] = v

        return output
