"""Audit a single, completed OpenFE relative binding free-energy comparison."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import Counter
from collections.abc import Iterator, Sequence
from pathlib import Path


def _fingerprint(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _unique_fields(pairs: list) -> dict:
    obj = dict(pairs)
    if len(obj) != len(pairs):
        raise ValueError("duplicate field in result JSON")
    return obj


def _energy(quantity: dict) -> float:
    value = quantity["magnitude"]
    if quantity["unit"] != "kilocalorie / mole":
        raise ValueError("expected an OpenFE energy in kilocalories per mole")
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("invalid energy magnitude")
    return float(value)


def _phase_context(inputs: dict, edge: tuple[str, str]) -> tuple[str, str, str]:
    """Use serialized components to establish direction, phase and compatible conditions."""
    mapping = inputs["ligandmapping"]
    if mapping["__qualname__"] != "LigandAtomMapping" or not mapping["componentA_to_componentB"]:
        raise ValueError("missing ligand atom mapping")
    components = []
    for side, name in zip(("A", "B"), edge, strict=True):
        mapped = mapping["component" + side]
        state = inputs["state" + side]["components"]
        if mapped["molprops"]["ofe-name"] != name or mapped["__qualname__"] != "SmallMoleculeComponent":
            raise ValueError("requested ligand identities or direction differ from the recorded mapping")
        ligand_keys = [key for key, comp in state.items() if comp == mapped]
        if len(ligand_keys) != 1:
            raise ValueError("mapped ligand differs from the simulated chemical system")
        components.append({key: comp for key, comp in state.items() if key != ligand_keys[0]})
    if components[0] != components[1]:
        raise ValueError("background components differ between the two end states")
    environment = {comp["__qualname__"]: comp for comp in components[0].values()}
    if len(environment) != len(components[0]) or set(environment) not in (
        {"SolventComponent"}, {"SolventComponent", "ProteinComponent"},
    ):
        raise ValueError("expected a standard solvent or protein-in-solvent leg")
    settings = inputs["settings"]
    shared = _fingerprint([
        mapping, environment["SolventComponent"], settings["thermo_settings"],
        settings["forcefield_settings"]["small_molecule_forcefield"],
    ])
    phase = "complex" if "ProteinComponent" in environment else "solvent"
    return phase, shared, _fingerprint([components[0], settings])


def _read_repeats(path: Path, edge: tuple[str, str]) -> Iterator[tuple[str, str, str, dict]]:
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    doc = json.loads(raw, object_pairs_hook=_unique_fields)
    if not isinstance(doc, dict) or doc.get("unit_failures"):
        raise ValueError("invalid or failed protocol result")
    protocol = doc["protocol_result"]
    if (protocol["__qualname__"] != "RelativeHybridTopologyProtocolResult"
            or not protocol["__module__"].startswith("openfe.protocols.")):
        raise ValueError("expected an OpenFE relative hybrid topology result")
    units = doc["unit_results"]
    completed_sources = {unit["source_key"] for unit in units.values() if unit["__qualname__"] == "ProtocolUnitResult"}
    failures = Counter(unit["source_key"] for unit in units.values() if unit["__qualname__"] == "ProtocolUnitFailure")
    recognized = {"ProtocolUnitResult", "ProtocolUnitFailure"}
    if (not completed_sources or failures.keys() - completed_sources
            or any(unit["__qualname__"] not in recognized for unit in units.values())):
        raise ValueError("missing or unresolved protocol unit")
    if not protocol["data"]:
        raise ValueError("no completed repeats")
    seen_units = set()
    for repeat_id, generations in protocol["data"].items():
        if len(generations) != 1:
            raise ValueError("expected one completed generation per repeat")
        row = generations[0]
        outputs = row["outputs"]
        unit_id = row["_key"]
        mirrored = units[unit_id]
        if (unit_id in seen_units or mirrored["__qualname__"] != "ProtocolUnitResult" or mirrored["_key"] != unit_id
                or mirrored["source_key"] != row["source_key"] or mirrored["outputs"] != outputs):
            raise ValueError("repeat disagrees with the recorded protocol unit")
        seen_units.add(unit_id)
        if str(outputs["repeat_id"]) != repeat_id or outputs["generation"] != 0:
            raise ValueError("inconsistent repeat identity or unsupported continuation")
        inputs = row["inputs"]
        if (str(inputs.get("repeat_id", repeat_id)) != repeat_id or inputs.get("generation", 0) != 0):
            raise ValueError("repeat identity disagrees with the recorded inputs")
        if "setup_results" in inputs:
            inputs = inputs["setup_results"]["inputs"]
        phase, shared, context = _phase_context(inputs, edge)
        estimate = _energy(outputs["unit_estimate"])
        mbar_error = _energy(outputs["unit_estimate_error"])
        if mbar_error < 0:
            raise ValueError("negative MBAR uncertainty")
        yield phase, shared, context, {
            "repeat_id": repeat_id, "unit_id": unit_id, "source_unit": row["source_key"], "result_sha256": digest,
            "estimate": estimate, "mbar_uncertainty": mbar_error,
            "failed_attempts_for_unit": failures[row["source_key"]],
        }
    if any("unit_estimate" in unit["outputs"] and key not in seen_units for key, unit in units.items()):
        raise ValueError("unaccounted estimate in the unit results")


def read_openfe_rbfe(ligand_a: str, ligand_b: str, result_files: Sequence[str | Path]) -> dict:
    """Read both phases of one directed edge from completed, self-describing OpenFE JSONs.

    Each phase carries its own repeat estimates and MBAR errors in kcal/mol.
    The combined uncertainty uses the sample standard deviation across independent
    repeats in each phase, as in OpenFE 1.12 and later. A singleton has no such estimate.
    """
    if not all(isinstance(name, str) and name.strip() for name in (ligand_a, ligand_b)) or ligand_a == ligand_b:
        raise ValueError("two distinct ligand names are required")
    repeats = {"complex": [], "solvent": []}
    shared_contexts: set[str] = set()
    phase_contexts: dict[str, str] = {}
    seen_ids: set[tuple[str, str]] = set()
    seen_units: set[str] = set()
    for source in result_files:
        path = Path(source)
        try:
            for phase, shared, context, row in _read_repeats(path, (ligand_a, ligand_b)):
                if (phase, row["repeat_id"]) in seen_ids or row["source_unit"] in seen_units:
                    raise ValueError("duplicate simulation repeat")
                seen_ids.add((phase, row["repeat_id"]))
                seen_units.add(row["source_unit"])
                shared_contexts.add(shared)
                if len(shared_contexts) != 1 or phase_contexts.setdefault(phase, context) != context:
                    raise ValueError("inconsistent ligand mapping, chemical system or protocol settings")
                repeats[phase].append(row)
        except (OSError, KeyError, TypeError, IndexError, AttributeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{path.name}: {exc}") from exc
    legs = {}
    for phase, rows in repeats.items():
        if not rows:
            raise ValueError(f"missing {phase} leg for the requested edge")
        rows.sort(key=lambda row: row["repeat_id"])
        values = [row["estimate"] for row in rows]
        legs[phase] = {
            "mean": statistics.mean(values), "sample_sd": statistics.stdev(values) if len(rows) > 1 else None,
            "context_sha256": phase_contexts[phase], "repeats": rows,
        }
    complex_sd, solvent_sd = (legs[leg]["sample_sd"] for leg in ("complex", "solvent"))
    ddg = legs["complex"]["mean"] - legs["solvent"]["mean"]
    uncertainty = None if None in (complex_sd, solvent_sd) else math.hypot(complex_sd, solvent_sd)
    if not math.isfinite(ddg) or (uncertainty is not None and not math.isfinite(uncertainty)):
        raise ValueError("nonfinite relative energy or uncertainty")
    return {
        "method": "openfe_completed_rbfe", "edge": {"ligand_a": ligand_a, "ligand_b": ligand_b},
        "units": "kcal/mol", "ddg_b_minus_a": ddg, "uncertainty": uncertainty,
        "uncertainty_basis": "quadrature of per-leg sample SDs across independent repeats",
        "legs": legs,
    }
