"""Predict solid solubility in nominated pure organic solvents at declared temperatures."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from rfproteina.disease import corpus_provenance, fastsolv_runtime, ligand_admet, ligand_inputs

METHOD_ID = "fastsolv"
METHOD = "fastsolv_bigsol_ensemble_1_0_1"
ENDPOINT = "organic_solid_solubility_log10_mol_l"
MANIFEST = "fastsolv.json"
_SOLVENTS = {
    "acetone": "CC(C)=O",
    "acetonitrile": "CC#N",
    "dimethyl_sulfoxide": "CS(C)=O",
    "dimethylformamide": "CN(C)C=O",
    "ethanol": "CCO",
    "ethyl_acetate": "CCOC(C)=O",
    "isopropanol": "CC(C)O",
    "methanol": "CO",
    "tetrahydrofuran": "C1CCOC1",
    "toluene": "Cc1ccccc1",
}
_TEMPERATURE_K = (295.0, 323.0)
_MAX_CONDITIONS = 8
_MAX_CANDIDATES = 16
_HASH = re.compile(r"[a-f0-9]{64}")
_ELEMENTS = {"H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"}
BLOCKERS = frozenset(
    {
        "fastsolv_manifest_missing",
        "invalid_fastsolv_manifest",
        "fastsolv_candidates_unreadable",
        "fastsolv_no_candidates",
        "fastsolv_candidates_over_limit",
        "fastsolv_duplicate_candidate",
        "fastsolv_candidate_scope_missing",
        "fastsolv_rdkit_unavailable",
        "fastsolv_unsupported_candidate",
        "fastsolv_no_supported_candidates",
        "fastsolv_invalid_report",
        *fastsolv_runtime.BLOCKERS,
    }
)


class FastsolvError(ValueError):
    """The requested conditions, structures or results cannot be attributed."""


@dataclass(frozen=True)
class Condition:
    solvent: str
    temperature_k: float

    def record(self) -> dict:
        return {"solvent": self.solvent, "temperature_k": self.temperature_k}


@dataclass(frozen=True)
class Plan:
    scope: str
    conditions: tuple[Condition, ...]
    candidates: tuple[tuple[str, str | None], ...]


def _unique(pairs: list[tuple[str, object]]) -> dict:
    settings = dict(pairs)
    if len(settings) != len(pairs):
        raise ValueError("Duplicate field.")
    return settings


def _number(value: object, lower: float, upper: float) -> bool:
    return type(value) in (int, float) and lower <= value <= upper


def _conditions(raw: object) -> tuple[Condition, ...]:
    if not isinstance(raw, list) or not 1 <= len(raw) <= _MAX_CONDITIONS:
        raise FastsolvError("invalid_fastsolv_manifest")
    conditions = []
    for row in raw:
        if (
            not isinstance(row, dict)
            or row.keys() != {"solvent", "temperature_k"}
            or not (
                isinstance(row["solvent"], str)
                and row["solvent"] in _SOLVENTS
                and _number(row["temperature_k"], *_TEMPERATURE_K)
            )
        ):
            raise FastsolvError("invalid_fastsolv_manifest")
        conditions.append(Condition(row["solvent"], float(row["temperature_k"])))
    if len(conditions) != len(set(conditions)):
        raise FastsolvError("invalid_fastsolv_manifest")
    return tuple(conditions)


def _manifest(directory: Path) -> tuple[Condition, ...]:
    try:
        raw = ligand_inputs.read_regular(directory / MANIFEST, 4096, "fastsolv_manifest", FastsolvError)
        manifest = json.loads(raw.decode("utf-8-sig"), object_pairs_hook=_unique)
        if not (
            isinstance(manifest, dict)
            and manifest.keys() == {"version", "conditions"}
            and type(manifest["version"]) is int
            and manifest["version"] == 1
        ):
            raise ValueError("Invalid solvent condition fields.")
        return _conditions(manifest["conditions"])
    except (ValueError, RecursionError) as error:
        if isinstance(error, FastsolvError) and str(error) == "fastsolv_manifest_missing":
            raise
        raise FastsolvError("invalid_fastsolv_manifest") from error


def _solute(smiles: str) -> str | None:
    from rdkit import Chem

    try:
        molecule = ligand_inputs.parse_nominated_smiles(smiles, Chem, remove_hs=True) if len(smiles) <= 512 else None
        if molecule is None or not (
            4 <= molecule.GetNumHeavyAtoms() <= 50
            and len(Chem.GetMolFrags(molecule)) == 1
            and any(atom.GetAtomicNum() == 6 for atom in molecule.GetAtoms())
            and Chem.GetFormalCharge(molecule) == 0
            and all(
                atom.GetSymbol() in _ELEMENTS
                and not (atom.GetIsotope() or atom.GetNumRadicalElectrons() or atom.GetAtomMapNum())
                for atom in molecule.GetAtoms()
            )
        ):
            return None
        return Chem.MolToSmiles(molecule, isomericSmiles=True)
    except (ValueError, RuntimeError):
        return None


def prepare(directory: Path, candidate: str | None = None) -> Plan:
    conditions = _manifest(directory)
    scope, rows, blocker = ligand_inputs.bounded_smiles_candidates(
        directory,
        max_bytes=ligand_admet.MAX_INPUT_BYTES,
        max_rows=ligand_admet.MAX_CANDIDATES,
    )
    if blocker:
        raise FastsolvError("fastsolv_candidates_unreadable")
    if candidate is not None and (scope != "candidate_table" or not ligand_admet.valid_candidate_id(candidate)):
        raise FastsolvError("fastsolv_candidate_scope_missing")
    selected = list(ligand_inputs.nominated_rows(rows, candidate))
    if not selected:
        raise FastsolvError("fastsolv_no_candidates")
    if len(selected) > _MAX_CANDIDATES:
        raise FastsolvError("fastsolv_candidates_over_limit")
    if any(len(row) != 2 or not ligand_admet.valid_candidate_id(row[0]) for row in selected):
        raise FastsolvError("fastsolv_candidates_unreadable")
    if len({row[0] for row in selected}) != len(selected):
        raise FastsolvError("fastsolv_duplicate_candidate")
    try:
        from rdkit import rdBase
    except (ImportError, OSError) as error:
        raise FastsolvError("fastsolv_rdkit_unavailable") from error
    with rdBase.BlockLogs():
        nominations = tuple((name, _solute(smiles)) for name, smiles in selected)
    return Plan(scope, conditions, nominations)


def installed() -> bool:
    return fastsolv_runtime.installed()


def _base() -> dict:
    return {
        "method": METHOD,
        "endpoint": ENDPOINT,
        "release": fastsolv_runtime.RELEASE,
        "checkpoint_sha256": fastsolv_runtime.checkpoint_digests(),
        "ensemble_members": 4,
        "status": "untested",
        "candidates": [],
        "corpus_provenance": corpus_provenance.declare(METHOD, against=ENDPOINT),
    }


def _prediction(row: object) -> dict:
    if not isinstance(row, Mapping) or not (
        _number(row.get("log_s"), -20, 3) and _number(row.get("ensemble_sd"), 0, 10)
    ):
        raise FastsolvError("fastsolv_invalid_report")
    return {"log_s": row["log_s"], "ensemble_sd": row["ensemble_sd"]}


def screen_run(directory: Path, candidate: str | None = None) -> dict:
    report = _base()
    try:
        plan = prepare(directory, candidate)
        report.update(scope=plan.scope, conditions=[condition.record() for condition in plan.conditions])
        batch = [
            {
                "solute_smiles": smiles,
                "solvent_smiles": _SOLVENTS[condition.solvent],
                "temperature_k": condition.temperature_k,
            }
            for _, smiles in plan.candidates
            if smiles is not None
            for condition in plan.conditions
        ]
        if not batch:
            return {
                **report,
                "blocker": "fastsolv_no_supported_candidates",
                "candidates": [
                    {"candidate_id": name, "status": "untested", "blocker": "fastsolv_unsupported_candidate"}
                    for name, _ in plan.candidates
                ],
            }
        answers = fastsolv_runtime.predict(batch)
        if not isinstance(answers, list) or len(answers) != len(batch):
            raise FastsolvError("fastsolv_invalid_report")
        answers = iter(answers)
        rows = []
        skipped = 0
        for name, smiles in plan.candidates:
            if smiles is None:
                rows.append({"candidate_id": name, "status": "untested", "blocker": "fastsolv_unsupported_candidate"})
                skipped += 1
                continue
            predictions = []
            for condition in plan.conditions:
                found = next(answers)
                if not isinstance(found, dict) or found.keys() != {"log_s", "ensemble_sd"}:
                    raise FastsolvError("fastsolv_invalid_report")
                predictions.append({**condition.record(), **_prediction(found)})
            rows.append(
                {
                    "candidate_id": name,
                    "status": "predicted",
                    "predictions": predictions,
                    "candidate_sha256": sha256(smiles.encode("ascii")).hexdigest(),
                }
            )
        return {**report, "status": "partial" if skipped else "complete", "candidates": rows}
    except (FastsolvError, fastsolv_runtime.FastsolvRuntimeError) as error:
        return {**report, "blocker": str(error)}


def report_view(raw: object) -> dict:
    """Validate typed results and solvent attribution before sharing predictions."""
    report = _base()
    invalid = {**report, "blocker": "fastsolv_invalid_report"}
    attestation = ("method", "endpoint", "release", "checkpoint_sha256", "ensemble_members")
    if not isinstance(raw, Mapping) or any(raw.get(key) != report[key] for key in attestation):
        return invalid
    status, blocker, source = (raw.get(key) for key in ("status", "blocker", "candidates"))
    if status == "untested" and source == [] and isinstance(blocker, str) and blocker in BLOCKERS:
        return {**report, "blocker": blocker}
    scope = raw.get("scope")
    if (
        not ligand_inputs.valid_saved_smiles_scope(scope, source)
        or not 0 < len(source) <= _MAX_CANDIDATES
    ):
        return invalid
    names = set()
    candidates = []
    tested = 0
    try:
        conditions = _conditions(raw.get("conditions"))
        for row in source:
            if (
                not isinstance(row, Mapping)
                or not isinstance((name := row.get("candidate_id")), str)
                or (not ligand_admet.valid_candidate_id(name) or name in names)
            ):
                return invalid
            names.add(name)
            if row.get("status") == "untested" and row.get("blocker") == "fastsolv_unsupported_candidate":
                candidates.append(
                    {"candidate_id": name, "status": "untested", "blocker": "fastsolv_unsupported_candidate"}
                )
                continue
            digest, source_predictions = row.get("candidate_sha256"), row.get("predictions")
            if not (
                row.get("status") == "predicted"
                and "blocker" not in row
                and isinstance(digest, str)
                and _HASH.fullmatch(digest)
                and isinstance(source_predictions, list)
                and len(source_predictions) == len(conditions)
            ):
                return invalid
            predictions = []
            for condition, prediction in zip(conditions, source_predictions, strict=True):
                if not isinstance(prediction, Mapping) or any(
                    prediction.get(key) != value for key, value in condition.record().items()
                ):
                    return invalid
                predictions.append({**condition.record(), **_prediction(prediction)})
            candidates.append(
                {"candidate_id": name, "status": "predicted", "predictions": predictions, "candidate_sha256": digest}
            )
            tested += 1
    except FastsolvError:
        return invalid
    expected = "complete" if tested == len(source) else "partial" if tested else "untested"
    if status != expected or blocker != (None if tested else "fastsolv_no_supported_candidates"):
        return invalid
    report.update(status=status, scope=scope, conditions=[item.record() for item in conditions], candidates=candidates)
    if blocker:
        report["blocker"] = blocker
    return report
