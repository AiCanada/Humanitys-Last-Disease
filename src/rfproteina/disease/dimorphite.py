"""Enumerate protonation states for a declared pH window on nominated ligands."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, distribution
from importlib.util import find_spec
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

from rfproteina.disease import ligand_admet, ligand_inputs

if TYPE_CHECKING:
    from rdkit import Chem

METHOD_ID = "dimorphite_dl"
METHOD = "dimorphite_dl_empirical_protonation"
ENDPOINT = "pH_window_protonation_states"
MANIFEST = "dimorphite.json"
RELEASE = "2.1.0"
MAX_CANDIDATES = 32
MAX_VARIANTS = 32
MAX_ATOMS = 80
MAX_SMILES = 512
PRECISION = 1.0
RULES_SHA256 = "9611d13d0381e97ef4fec5784ca47f9cba0b02a4f4c38918f35dadb6ee747137"
_RULES_FILE = "dimorphite_dl/smarts/site_substructures.smarts"
_MAX_RESPONSE_BYTES = 2 * 1024 * 1024
_ELEMENTS = frozenset({1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53})
_PROVIDER_BLOCKERS = frozenset({
    "variant_budget_exceeded", "no_recognized_sites", "dimorphite_provider_failure", "dimorphite_variant_rejected",
})
_INPUT_BLOCKERS = frozenset({
    "dimorphite_manifest_missing", "invalid_dimorphite_manifest", "invalid_candidate_id",
    "candidate_requires_named_nomination", "candidate_missing_from_table", "invalid_candidate_table_header",
    "ligand_input_missing", "ligand_input_unreadable", "ligand_input_exceeds_interactive_size",
    "expected_one_run_ligand", "unsupported_extended_smiles", "dimorphite_candidates_over_limit",
    "duplicate_candidate_nomination", "invalid_nominated_smiles", "dimorphite_unsupported_molecule",
    "rdkit_unavailable", "dimorphite_no_candidates",
})
BLOCKERS = _INPUT_BLOCKERS | _PROVIDER_BLOCKERS | {
    "dimorphite_runtime_unavailable", "dimorphite_worker_timeout", "dimorphite_invalid_report",
    "dimorphite_no_states",
}


class DimorphiteError(ValueError):
    """Carry a stable blocker code without including molecular data or paths."""


@dataclass(frozen=True)
class Nomination:
    candidate_id: str
    smiles: str = ""
    blocker: str | None = None


@dataclass(frozen=True)
class Plan:
    ph_min: float
    ph_max: float
    scope: str
    nominations: tuple[Nomination, ...]


def _unique(pairs: list[tuple[str, object]]) -> dict:
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("Duplicate JSON field.")
    return result


def valid_window(ph_min: object, ph_max: object) -> bool:
    return (
        type(ph_min) in (int, float) and type(ph_max) in (int, float)
        and 0 <= ph_min <= ph_max <= 14
    )


def _settings(directory: Path) -> tuple[float, float]:
    try:
        raw = ligand_inputs.read_regular(directory / MANIFEST, 4096, "dimorphite_manifest", DimorphiteError)
        settings = json.loads(raw.decode("utf-8-sig"), object_pairs_hook=_unique)
        if not (
            isinstance(settings, dict) and settings.keys() == {"version", "ph_min", "ph_max"}
            and type(settings["version"]) is int and settings["version"] == 1
            and valid_window(settings["ph_min"], settings["ph_max"])
        ):
            raise ValueError("Invalid settings.")
    except (ValueError, RecursionError) as error:
        if isinstance(error, DimorphiteError) and str(error) == "dimorphite_manifest_missing":
            raise
        raise DimorphiteError("invalid_dimorphite_manifest") from error
    return float(settings["ph_min"]), float(settings["ph_max"])


def rules_verified() -> bool:
    """Check the released chemistry table without importing the provider."""
    try:
        package = distribution("dimorphite-dl")
        if package.version != RELEASE:
            return False
        path = Path(package.locate_file(_RULES_FILE))
        rules = ligand_inputs.read_regular(path, 64 * 1024, "dimorphite_rules", DimorphiteError)
        return sha256(rules).hexdigest() == RULES_SHA256
    except (PackageNotFoundError, DimorphiteError):
        return False


def installed() -> bool:
    try:
        return rules_verified() and all(find_spec(name) is not None for name in ("dimorphite_dl", "loguru", "rdkit"))
    except (ImportError, ValueError):
        return False


def _molecule(smiles: str, chem: ModuleType) -> tuple[Chem.Mol, str]:
    if len(smiles) > MAX_SMILES:
        raise DimorphiteError("dimorphite_unsupported_molecule")
    molecule, canonical = ligand_admet.molecular_graph(smiles, chem, unsupported="dimorphite_unsupported_molecule")
    if (
        molecule.GetNumHeavyAtoms() > MAX_ATOMS or len(canonical) > MAX_SMILES
        or any(atom.GetAtomicNum() not in _ELEMENTS or atom.GetIsotope() for atom in molecule.GetAtoms())
    ):
        raise DimorphiteError("dimorphite_unsupported_molecule")
    return molecule, canonical


def prepare(directory: Path, candidate: str | None = None) -> Plan:
    ph_min, ph_max = _settings(directory)
    if candidate is not None and not ligand_admet.valid_candidate_id(candidate):
        raise DimorphiteError("invalid_candidate_id")
    scope, rows, blocker = ligand_inputs.bounded_smiles_candidates(
        directory, max_bytes=ligand_admet.MAX_INPUT_BYTES, max_rows=ligand_admet.MAX_CANDIDATES,
    )
    if blocker:
        raise DimorphiteError(blocker if blocker in _INPUT_BLOCKERS else "ligand_input_unreadable")
    if candidate is not None and scope != "candidate_table":
        raise DimorphiteError("candidate_requires_named_nomination")
    selected = list(ligand_inputs.nominated_rows(rows, candidate))
    if not selected:
        raise DimorphiteError("candidate_missing_from_table" if candidate is not None else "dimorphite_no_candidates")
    if len(selected) > MAX_CANDIDATES:
        raise DimorphiteError("dimorphite_candidates_over_limit")
    try:
        from rdkit import Chem, rdBase
    except (ImportError, OSError) as error:
        raise DimorphiteError("rdkit_unavailable") from error

    counts = Counter(row[0] for row in selected)
    nominations = []
    with rdBase.BlockLogs():
        for row in selected:
            name = row[0]
            if not ligand_admet.valid_candidate_id(name):
                nominations.append(Nomination("", blocker="invalid_candidate_id"))
            elif counts[name] != 1:
                nominations.append(Nomination(name, blocker="duplicate_candidate_nomination"))
            elif len(row) != 2:
                nominations.append(Nomination(name, blocker="invalid_nominated_smiles"))
            else:
                try:
                    _, smiles = _molecule(row[1], Chem)
                    nominations.append(Nomination(name, smiles))
                except (DimorphiteError, ligand_admet.AdmetError) as error:
                    nominations.append(Nomination(name, blocker=str(error)))
    return Plan(ph_min, ph_max, scope, tuple(nominations))


def _enumerate(ph_min: float, ph_max: float, smiles: list[str]) -> dict[str, dict]:
    if not installed():
        raise DimorphiteError("dimorphite_runtime_unavailable")
    request = json.dumps({"ph_min": ph_min, "ph_max": ph_max, "smiles": smiles}).encode("ascii")
    # The upstream package reads its logging settings during import, including a
    # file sink. Keep this run local to stdin/stdout regardless of that config.
    env = {key: value for key, value in os.environ.items() if not key.startswith("DIMORPHITE_DL_")}
    try:
        process = subprocess.run(
            [sys.executable, "-B", "-m", "rfproteina.disease.dimorphite_worker"],
            input=request, capture_output=True, check=True, timeout=45, env=env,
        )
    except subprocess.TimeoutExpired as error:
        raise DimorphiteError("dimorphite_worker_timeout") from error
    except (OSError, subprocess.CalledProcessError) as error:
        raise DimorphiteError("dimorphite_provider_failure") from error
    try:
        if len(process.stdout) > _MAX_RESPONSE_BYTES:
            raise ValueError("Worker response over limit.")
        result = json.loads(process.stdout, object_pairs_hook=_unique)
        if not isinstance(result, dict) or result.keys() != set(smiles):
            raise ValueError("Mismatched worker identities.")
    except (ValueError, RecursionError) as error:
        raise DimorphiteError("dimorphite_invalid_report") from error
    return result


def _parent_key(mol: Chem.Mol, chem: ModuleType) -> str:
    """Remove charge and hydrogen counts while keeping bonds, aromaticity and defined stereo."""
    stripped = chem.Mol(mol)
    for atom in stripped.GetAtoms():
        atom.SetFormalCharge(0)
        atom.SetNumExplicitHs(0)
        atom.SetNoImplicit(True)
    return chem.MolToSmiles(stripped, isomericSmiles=True)


def _states(parent: str, result: object) -> list[dict]:
    from rdkit import Chem, rdBase

    if not isinstance(result, dict):
        raise DimorphiteError("dimorphite_invalid_report")
    blocker = result.get("blocker")
    if result.keys() == {"blocker"} and isinstance(blocker, str) and blocker in _PROVIDER_BLOCKERS:
        raise DimorphiteError(blocker)
    values = result.get("states")
    if result.keys() != {"states"} or not isinstance(values, list) or not 0 < len(values) <= MAX_VARIANTS:
        raise DimorphiteError("dimorphite_invalid_report")
    states = {}
    with rdBase.BlockLogs():
        try:
            parent_mol, _ = _molecule(parent, Chem)
            key = _parent_key(parent_mol, Chem)
            for value in values:
                if not isinstance(value, str):
                    raise DimorphiteError("dimorphite_invalid_report")
                variant, canonical = _molecule(value, Chem)
                if _parent_key(variant, Chem) != key:
                    raise DimorphiteError("dimorphite_invalid_report")
                states[canonical] = Chem.GetFormalCharge(variant)
        except (DimorphiteError, ligand_admet.AdmetError, RuntimeError, OverflowError) as error:
            raise DimorphiteError("dimorphite_invalid_report") from error
    return [{"smiles": value, "formal_charge": states[value]} for value in sorted(states)]


def _record() -> dict:
    return {
        "method": METHOD, "endpoint": ENDPOINT, "release": RELEASE, "precision": PRECISION,
        "max_variants": MAX_VARIANTS, "rules_sha256": RULES_SHA256, "status": "untested", "molecules": [],
    }


def _finish(report: dict, rows: list[dict]) -> dict:
    count = sum(row["status"] == "complete" for row in rows)
    status = ("complete" if count == len(rows) else "partial") if count else "untested"
    return {**report, "status": status, "molecules": rows, **({} if count else {"blocker": "dimorphite_no_states"})}


def screen_run(directory: Path, candidate: str | None = None) -> dict:
    report = _record()
    try:
        plan = prepare(directory, candidate)
    except DimorphiteError as error:
        return {**report, "blocker": str(error)}

    report.update(ph_min=plan.ph_min, ph_max=plan.ph_max, input_scope=plan.scope)
    parents = sorted({row.smiles for row in plan.nominations if not row.blocker})
    states = {}
    if parents:
        try:
            provided = _enumerate(plan.ph_min, plan.ph_max, parents)
        except DimorphiteError as error:
            states = {key: ([], str(error)) for key in parents}
        else:
            for key in parents:
                try:
                    states[key] = (_states(key, provided.get(key)), None)
                except DimorphiteError as error:
                    states[key] = ([], str(error))

    rows = []
    for row in plan.nominations:
        values, blocker = ([], row.blocker) if row.blocker else states[row.smiles]
        result = {
            "candidate_id": row.candidate_id, "input_smiles": row.smiles, "states": values,
            "status": "untested" if blocker else "complete",
        }
        if blocker:
            result["blocker"] = blocker
        rows.append(result)
    return _finish(report, rows)


def report_view(raw: object) -> dict:
    """Project saved or live results without trusting arbitrary text or claimed status."""
    report = _record()
    if not isinstance(raw, Mapping) or (
        (raw.get("method"), raw.get("endpoint"), raw.get("release")) != (METHOD, ENDPOINT, RELEASE)
        or type(raw.get("precision")) is not float or raw["precision"] != PRECISION
        or type(raw.get("max_variants")) is not int or raw["max_variants"] != MAX_VARIANTS
        or raw.get("rules_sha256") != RULES_SHA256
    ):
        return {**report, "blocker": "dimorphite_invalid_report"}
    rows = raw.get("molecules")
    if not isinstance(rows, list) or len(rows) > MAX_CANDIDATES:
        return {**report, "blocker": "dimorphite_invalid_report"}
    if not rows:
        blocker = raw.get("blocker")
        return {**report, "blocker": blocker if isinstance(blocker, str) and blocker in BLOCKERS
                else "dimorphite_invalid_report"}
    low, high, scope = raw.get("ph_min"), raw.get("ph_max"), raw.get("input_scope")
    if not valid_window(low, high) or scope not in ("candidate_table", "run_ligand"):
        return {**report, "blocker": "dimorphite_invalid_report"}
    report.update(ph_min=low, ph_max=high, input_scope=scope)
    sanitized = []
    for row in rows:
        if not isinstance(row, Mapping):
            return {**report, "blocker": "dimorphite_invalid_report"}
        name, parent, blocker = row.get("candidate_id"), row.get("input_smiles"), row.get("blocker")
        entry = {"candidate_id": "", "input_smiles": "", "states": [], "status": "untested"}
        if isinstance(name, str) and ligand_admet.valid_candidate_id(name):
            entry["candidate_id"] = name
        if isinstance(parent, str) and parent:
            try:
                from rdkit import Chem, rdBase

                with rdBase.BlockLogs():
                    _, entry["input_smiles"] = _molecule(parent, Chem)
            except (ImportError, OSError, ValueError, RuntimeError):
                return {**report, "blocker": "dimorphite_invalid_report"}
        if row.get("status") == "complete" and entry["candidate_id"] and entry["input_smiles"] and blocker is None:
            states = row.get("states")
            if not isinstance(states, list) or not 0 < len(states) <= MAX_VARIANTS or not all(
                isinstance(state, dict) and state.keys() == {"smiles", "formal_charge"}
                and type(state["formal_charge"]) is int for state in states
            ):
                return {**report, "blocker": "dimorphite_invalid_report"}
            try:
                checked = _states(entry["input_smiles"], {"states": [state["smiles"] for state in states]})
            except (DimorphiteError, ImportError, OSError):
                return {**report, "blocker": "dimorphite_invalid_report"}
            if checked != states:
                return {**report, "blocker": "dimorphite_invalid_report"}
            entry.update(states=checked, status="complete")
        elif row.get("status") == "untested" and isinstance(blocker, str) and blocker in BLOCKERS:
            entry["blocker"] = blocker
        else:
            return {**report, "blocker": "dimorphite_invalid_report"}
        sanitized.append(entry)
    names = Counter(row["candidate_id"] for row in sanitized if row["candidate_id"])
    if any(row["status"] == "complete" and names[row["candidate_id"]] != 1 for row in sanitized):
        return {**report, "blocker": "dimorphite_invalid_report"}
    return _finish(report, sanitized)
