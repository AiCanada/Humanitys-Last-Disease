"""Isoform-specific substrate classifications for nominated systemic small molecules."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from rfproteina.disease import cypstrate_runtime, ligand_admet, ligand_alerts, ligand_inputs
from rfproteina.disease.corpus_provenance import CYPSTRATE_ISOFORMS, declare
from rfproteina.disease.cypstrate_runtime import CypstrateError

METHOD_ID = "cypstrate"
MODES = {"best_performance": "cypstrate_best_performance", "full_coverage": "cypstrate_full_coverage"}
ENDPOINT = "human_cyp_substrate"
MANIFEST = "cypstrate.json"
_HEX = re.compile(r"[0-9a-f]{64}")
_CLASSES = {"Substrate": "substrate", "Non-substrate": "non_substrate"}


@dataclass(frozen=True)
class Nomination:
    name: str
    isoforms: tuple[str, ...]
    smiles: str = ""
    blocker: str | None = None


@dataclass(frozen=True)
class Plan:
    mode: str
    members: tuple[Nomination, ...]
    blocker: str | None = None


def _unique(pairs: list) -> dict:
    result = dict(pairs)
    if len(result) != len(pairs):
        raise CypstrateError("cypstrate_invalid_manifest")
    return result


def _manifest(directory: Path) -> tuple[str, bool, dict[str, tuple[str, ...]]]:
    raw = ligand_inputs.read_regular(directory / MANIFEST, 64 * 1024, "cypstrate_manifest", CypstrateError)
    try:
        config = json.loads(raw.decode("utf-8-sig"), object_pairs_hook=_unique)
    except (ValueError, RecursionError) as error:
        raise CypstrateError("cypstrate_invalid_manifest") from error
    if not isinstance(config, dict) or config.keys() - {"systemic", "candidates", "prediction_mode"}:
        raise CypstrateError("cypstrate_invalid_manifest")
    mode = config.get("prediction_mode", "best_performance")
    if not isinstance(mode, str) or mode not in MODES:
        raise CypstrateError("cypstrate_invalid_mode")
    if type(config.get("systemic")) is not bool:
        raise CypstrateError("cypstrate_exposure_unspecified")
    candidates = config.get("candidates")
    if not isinstance(candidates, dict) or not 1 <= len(candidates) <= 256:
        raise CypstrateError("cypstrate_invalid_manifest")
    hypotheses = {}
    for name, requested in candidates.items():
        if not ligand_admet.valid_candidate_id(name) or not isinstance(requested, list) or not 1 <= len(requested) <= 9:
            raise CypstrateError("cypstrate_invalid_manifest")
        isoforms = tuple("CYP" + item.upper().removeprefix("CYP") for item in requested if isinstance(item, str))
        if (
            len(isoforms) != len(requested)
            or len(set(isoforms)) != len(isoforms)
            or set(isoforms) - set(CYPSTRATE_ISOFORMS)
        ):
            raise CypstrateError("cypstrate_invalid_isoforms")
        hypotheses[name] = isoforms
    return mode, config["systemic"], hypotheses


def declared_exposure(directory: Path) -> dict[str, bool]:
    """Return the declared systemic exposure only for explicitly nominated candidates."""
    _, systemic, hypotheses = _manifest(directory)
    return dict.fromkeys(hypotheses, systemic)


def _molecule(name: str, isoforms: tuple[str, ...], rows: list) -> Nomination:
    from rdkit import Chem

    if not rows:
        return Nomination(name, isoforms, blocker="candidate_missing_from_table")
    if len(rows) != 1:
        return Nomination(name, isoforms, blocker="duplicate_candidate_nomination")
    row = rows[0]
    if len(row) != 2 or len(row[1]) > 4096:
        return Nomination(name, isoforms, blocker="invalid_molecular_nomination")
    try:
        mol = ligand_alerts.parse_molecule(row[1], nominated=True)
        if mol.GetNumAtoms() > 256 or not any(atom.GetAtomicNum() == 6 for atom in mol.GetAtoms()):
            return Nomination(name, isoforms, blocker="cypstrate_unsupported_input")
        # The pinned model standardizes its own input, including salts and stereochemistry.
        # Preserve the nominated form here so both identities can be attributed.
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
        return Nomination(name, isoforms, smiles=Chem.MolToSmiles(mol, isomericSmiles=True))
    except (ligand_alerts.LigandUnreadableError, ValueError, RuntimeError, OverflowError):
        return Nomination(name, isoforms, blocker="invalid_nominated_smiles")


def prepare(directory: Path, candidate: str | None) -> Plan:
    mode, systemic, hypotheses = _manifest(directory)
    if candidate is not None:
        if candidate not in hypotheses:
            return Plan(mode, (), blocker="cypstrate_candidate_not_nominated")
        hypotheses = {candidate: hypotheses[candidate]}
    if not systemic:
        return Plan(mode, (), blocker="cypstrate_nonsystemic")
    limit = 1024 * 1024
    scope, rows, blocker = ligand_inputs.bounded_smiles_candidates(directory, max_bytes=limit, max_rows=limit)
    if scope == "run_ligand" and (candidate is not None or set(hypotheses) != {"run_ligand"}):
        blocker = "candidate_requires_named_nomination"
    if blocker:
        return Plan(mode, tuple(Nomination(n, iso, blocker=blocker) for n, iso in hypotheses.items()), blocker=blocker)
    by_name = {name: [] for name in hypotheses}
    for row in rows:
        if row and row[0] in by_name:
            by_name[row[0]].append(row)
    try:
        from rdkit import rdBase

        with rdBase.BlockLogs():
            members = tuple(_molecule(n, iso, by_name[n]) for n, iso in hypotheses.items())
    except (ImportError, OSError):
        return Plan(
            mode,
            tuple(Nomination(n, iso, blocker="rdkit_unavailable") for n, iso in hypotheses.items()),
            blocker="rdkit_unavailable",
        )
    return Plan(mode, members)


def _attributed(record: dict, ids: set[str]) -> dict:
    rows = record.get("results")
    if not isinstance(rows, list):
        raise CypstrateError("cypstrate_invalid_output")
    result = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or row["id"] not in ids:
            raise CypstrateError("cypstrate_invalid_output")
        key = row["id"]
        result[key] = None if key in result else row
    return result


def _observation(result: dict | None, isoform: str, mode: str) -> dict:
    if result is None:
        return {"blocker": "cypstrate_unattributed_result"}
    predictions = result.get("predictions")
    standardized = result.get("preprocessed_id")
    if "preprocessed_id" in result and standardized is None and predictions == {}:
        return {"blocker": "cypstrate_preprocessing_failed"}
    if not isinstance(standardized, str) or not _HEX.fullmatch(standardized) or not isinstance(predictions, dict):
        return {"blocker": "cypstrate_invalid_output"}
    changed = result.get("upstream_smiles_changed")
    pair = predictions.get(isoform)
    if type(changed) is not bool or not isinstance(pair, dict):
        return {"blocker": "cypstrate_invalid_output"}
    label, similarity = pair.get("label"), pair.get("similarity")
    if not isinstance(label, str) or type(similarity) not in (int, float) or not 0 <= similarity <= 1:
        return {"blocker": "cypstrate_invalid_output"}
    metrics = {
        "preprocessed_compound_id": "smiles_sha256_" + standardized,
        "upstream_smiles_changed": changed,
        "nearest_training_maccs_tanimoto": similarity,
    }
    if label == "No prediction" and mode == "best_performance":
        return {**metrics, "blocker": "cypstrate_no_consensus", "status": "abstained"}
    if label not in _CLASSES:
        return {"blocker": "cypstrate_invalid_output"}
    return {**metrics, "classification": _CLASSES[label], "status": "classified"}


def measure(directory: Path, candidate: str | None = None) -> dict:
    samples = []
    report = {
        "method": METHOD_ID,
        "endpoint": ENDPOINT,
        "species": "human",
        "scope": "candidate" if candidate is not None else "run",
        "status": "untested",
        "attempted": 0,
        "classified": 0,
        "abstained": 0,
        "samples": samples,
        "interpretation": "Categorical CYP substrate predictions for planning metabolism assays. "
        "Clearance rates and clinical interaction risk require separate evidence.",
    }
    try:
        plan = prepare(directory, candidate)
    except CypstrateError as error:
        return {**report, "blocker": str(error)}
    method = MODES[plan.mode]
    report.update(method=method, prediction_mode=plan.mode, corpus_provenance=declare(method, against=ENDPOINT))
    if plan.blocker == "cypstrate_nonsystemic":
        return {**report, "status": "skipped", "reason": plan.blocker}
    inputs = {}
    for member in plan.members:
        key = hashlib.sha256(member.smiles.encode()).hexdigest() if member.smiles else None
        for isoform in member.isoforms:
            row = {
                "candidate_id": member.name,
                "isoform": isoform,
                "endpoint": isoform.lower() + "_substrate",
                "method": method,
                "status": "untested",
            }
            if member.blocker:
                row["blocker"] = member.blocker
            else:
                row["compound_id"] = "smiles_sha256_" + key
                entry = inputs.setdefault(key, {"id": key, "smiles": member.smiles, "isoforms": []})
                if isoform not in entry["isoforms"]:
                    entry["isoforms"].append(isoform)
            samples.append(row)
    report["attempted"] = len(samples)
    if not inputs:
        return {**report, "blocker": plan.blocker or "no_eligible_nominations"}
    try:
        response = cypstrate_runtime.predict(list(inputs.values()), plan.mode)
        by_id = _attributed(response, set(inputs))
    except CypstrateError as error:
        for row in samples:
            row.setdefault("blocker", str(error))
        return {**report, "blocker": str(error)}
    report["provider"] = {
        "version": cypstrate_runtime.VERSION,
        "image": cypstrate_runtime.IMAGE,
        "revision": cypstrate_runtime.REVISION,
        "device": "cpu",
    }
    for row in samples:
        if "blocker" not in row:
            key = row["compound_id"].removeprefix("smiles_sha256_")
            row.update(_observation(by_id.get(key), row["isoform"], plan.mode))
    report["classified"] = sum(row["status"] == "classified" for row in samples)
    report["abstained"] = sum(row["status"] == "abstained" for row in samples)
    if report["classified"] == len(samples):
        report["status"] = "classified"
    elif report["abstained"] == len(samples):
        report["status"] = "abstained"
    elif report["classified"] or report["abstained"]:
        report["status"] = "partial"
    else:
        report["blocker"] = "cypstrate_no_usable_results"
    return report
