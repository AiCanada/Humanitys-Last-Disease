"""A local fingerprint-only Ames model fitted to the ToxScan study labels."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path

from rfproteina.disease import corpus_provenance, ligand_admet, ligand_inputs, toxscan_inputs

METHOD_ID = "toxscan_fingerprint_ames"
ENDPOINT = "AMES"
ARCHIVE = "e199ef53ee13ed2fe598ac96dd1ca4340b2b9aa602677c61e489cbc2dc4dd71f"
TABLE = "ligand_candidates.tsv"
BLOCKERS = frozenset({
    "toxscan_ames_model_missing", "toxscan_ames_model_integrity", "toxscan_ames_runtime_unavailable",
    "toxscan_ames_invalid_structure", "toxscan_ames_invalid_metrics", "toxscan_ames_invalid_report",
    "toxscan_ames_select_one", "toxscan_candidate_table_required", "toxscan_candidate_not_nominated",
    "toxscan_ambiguous_nomination", "toxscan_invalid_structure", "toxscan_rdkit_unavailable",
})


def installed() -> bool:
    from rfproteina.disease import toxscan_ames_model

    return toxscan_ames_model.installed()


def _nomination(directory: Path, candidate: str | None) -> tuple[str, str]:
    if candidate is None:
        scope, rows, blocker = ligand_inputs.bounded_smiles_candidates(
            directory, max_bytes=ligand_admet.MAX_INPUT_BYTES, max_rows=ligand_admet.MAX_CANDIDATES,
        )
        if scope != "candidate_table" or blocker or not rows:
            raise toxscan_inputs.ToxScanError("toxscan_candidate_table_required")
        if len(rows) != 1 or len(rows[0]) != 2:
            raise toxscan_inputs.ToxScanError("toxscan_ames_select_one")
        candidate = rows[0][0]
    smiles = toxscan_inputs.candidate_structure(directory, candidate)
    from rdkit import Chem

    try:
        ligand_admet.molecular_graph(smiles, Chem, unsupported="toxscan_ames_invalid_structure")
    except ligand_admet.AdmetError as error:
        raise toxscan_inputs.ToxScanError("toxscan_ames_invalid_structure") from error
    return candidate, smiles


def _identity(candidate: str, smiles: str) -> tuple[str, str]:
    return hashlib.sha256(candidate.encode("utf-8")).hexdigest(), hashlib.sha256(smiles.encode("ascii")).hexdigest()


def nomination_identity(directory: Path, candidate: str | None = None) -> tuple[str, str]:
    """Candidate ID and graph digests for comparing a saved result to the current selection."""
    return _identity(*_nomination(directory, candidate))


def readiness(directory: Path, candidate: str | None) -> tuple[bool, bool, str]:
    from rfproteina.disease import toxscan, toxscan_ames_model
    from rfproteina.disease.screen_board import _blocker_lines

    try:
        candidate, smiles = _nomination(directory, candidate)
    except toxscan_inputs.ToxScanError as error:
        return False, False, " ".join(_blocker_lines(str(error)))
    if (directory / toxscan.MANIFEST).is_file():
        author = toxscan.screen_run(directory, candidate)
        if author["results"].get("classifications", {}).get("Ames Mutagenicity") is not None:
            return True, False, "A matched author-platform Ames call already uses the same study corpus."
    try:
        if toxscan_ames_model.fingerprint_seen_in_training(smiles):
            return True, False, "Training included this fingerprint. Select the model to review a training control."
    except toxscan_ames_model.AmesModelError as error:
        return True, False, " ".join(_blocker_lines(str(error)))
    return True, True, "Estimate an Ames assay response offline for the nominated small molecule."


def _base() -> dict:
    return {
        "method": METHOD_ID, "endpoint": ENDPOINT, "scope": "candidate", "status": "untested",
        "corpus_provenance": corpus_provenance.declare(METHOD_ID, against=ENDPOINT),
    }


def _digest(fields: dict) -> str:
    return hashlib.sha256(json.dumps(fields, sort_keys=True, allow_nan=False,
                                     separators=(",", ":")).encode("ascii")).hexdigest()


def _result(data: Mapping) -> dict:
    from rfproteina.disease import toxscan_ames_model as model

    fields = {k: data[k] for k in ("score", "class_call", "model_sha256", "archive_sha256", "graph_membership",
                                   "fingerprint_seen_in_training", "candidate_id_sha256", "structure_sha256")}
    score = fields["score"]
    if (type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1
            or fields["class_call"] != ("YES" if score >= 0.5 else "NO")
            or fields["archive_sha256"] != ARCHIVE
            or fields["graph_membership"] not in ("train", "validation", "test", "not_in_retained_splits")
            or type(fields["fingerprint_seen_in_training"]) is not bool
            or (fields["graph_membership"] == "train" and not fields["fingerprint_seen_in_training"])
            or any(not isinstance(fields[k], str) or not toxscan_inputs.SHA256.fullmatch(fields[k])
                   for k in ("structure_sha256", "candidate_id_sha256", "model_sha256"))):
        raise ValueError("toxscan_ames_invalid_report")
    fields["validation"] = model.validate_validation(data["validation"])
    return fields


def report_view(data: object) -> dict:
    base = _base()
    invalid = {**base, "blocker": "toxscan_ames_invalid_report"}
    if not isinstance(data, Mapping) or any(data.get(k) != base[k] for k in ("method", "endpoint", "scope")):
        return invalid
    if data.get("status") == "untested":
        blocker = data.get("blocker")
        return {**base, "blocker": blocker} if isinstance(blocker, str) and blocker in BLOCKERS else invalid
    if data.get("status") != "complete" or "blocker" in data:
        return invalid
    try:
        fields = _result(data)
        digest = _digest(fields)
    except (KeyError, ValueError, TypeError, OverflowError):
        return invalid
    if data.get("result_sha256") != digest:
        return invalid
    return {**base, **fields, "status": "complete", "result_sha256": digest}


def screen_run(directory: Path, candidate: str | None = None) -> dict:
    from rfproteina.disease import toxscan_ames_model as model

    base = _base()
    try:
        candidate, smiles = _nomination(directory, candidate)
        candidate_sha, graph_sha = _identity(candidate, smiles)
        fields = _result({
            **model.predict(smiles),
            "candidate_id_sha256": candidate_sha, "structure_sha256": graph_sha,
        })
    except (toxscan_inputs.ToxScanError, model.AmesModelError) as error:
        return {**base, "blocker": str(error) if str(error) in BLOCKERS else "toxscan_ames_invalid_report"}
    except (ValueError, TypeError, KeyError, OverflowError):
        return {**base, "blocker": "toxscan_ames_invalid_report"}
    return {**base, **fields, "result_sha256": _digest(fields), "status": "complete"}
