"""Attribute a locally saved ToxScan author-service export to a nominated small molecule."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from rfproteina.disease import corpus_provenance, ligand_admet, toxscan_csv, toxscan_inputs, toxscan_validation
from rfproteina.disease.toxscan_csv import CLASSIFICATIONS as CLASSIFICATIONS
from rfproteina.disease.toxscan_csv import REGRESSIONS as REGRESSIONS
from rfproteina.disease.toxscan_inputs import MANIFEST as MANIFEST
from rfproteina.disease.toxscan_inputs import SOURCE as SOURCE
from rfproteina.disease.toxscan_inputs import SOURCE_FILE as SOURCE_FILE
from rfproteina.disease.toxscan_inputs import ToxScanError as ToxScanError

METHOD_ID = "toxscan"
METHOD = "toxscan_author_export"
ENDPOINT = "toxscan_toxicity_profile"
IDENTITY_CHECK_VERSION = 2
BLOCKERS = frozenset({
    "toxscan_manifest_missing", "toxscan_invalid_manifest", "toxscan_candidate_table_required",
    "toxscan_candidate_not_nominated", "toxscan_ambiguous_nomination", "toxscan_invalid_structure",
    "toxscan_rdkit_unavailable", "toxscan_export_missing", "toxscan_invalid_export", "toxscan_export_digest_mismatch",
    "toxscan_invalid_export_schema", "toxscan_invalid_export_value", "toxscan_structure_not_found",
    "toxscan_ambiguous_export", "toxscan_invalid_report", "toxscan_invalid_domain", "toxscan_nomination_changed",
})


def installed() -> bool:
    """This offline export reader uses RDKit for chemical identity."""
    try:
        from rdkit import Chem
    except (ImportError, OSError):
        return False
    return callable(getattr(Chem, "MolToSmiles", None))


def _digest(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, allow_nan=False, separators=(",", ":")).encode("ascii")
    return toxscan_inputs.sha256(raw)


def prepare(directory: Path, candidate: str | None = None) -> dict:
    sample = toxscan_inputs.nomination(directory, candidate)
    export = toxscan_inputs.export_bytes(directory, sample)
    fields = {
        "candidate_id": sample.candidate_id, "source_sha256": sample.export_sha256,
        "identity_check_version": IDENTITY_CHECK_VERSION,
        "structure_sha256": toxscan_inputs.sha256(sample.canonical_smiles.encode("ascii")),
        **toxscan_csv.read_export(export, sample.canonical_smiles),
    }
    return {**fields, "result_sha256": _digest(fields)}


def readiness(directory: Path, candidate: str | None = None) -> tuple[bool, bool, str]:
    try:
        prepared = prepare(directory, candidate)
    except ToxScanError as error:
        from rfproteina.disease.screen_board import _blocker_lines

        return False, False, " ".join(_blocker_lines(str(error)))
    discrepancy = toxscan_validation.domain_disagreement(prepared["applicability_domain"], prepared["similarity_score"])
    if discrepancy:
        return True, False, discrepancy
    if prepared["applicability_domain"] == "in_domain":
        if any(call is not None for call in prepared["results"]["classifications"].values()):
            return True, True, "The matched ToxScan export has in-domain assay calls for confirmatory testing."
        return True, False, "The matched export contains only numeric displays whose scales need verification."
    if prepared["applicability_domain"] == "out_of_domain":
        return True, False, "ToxScan marks this matched molecule outside its applicability domain."
    return True, False, "The matched ToxScan export leaves its applicability domain unreported."


def _base() -> dict:
    return {
        "method": METHOD, "endpoint": ENDPOINT, "scope": "candidate", "source": SOURCE,
        "status": "untested", "candidate_id": None, "source_sha256": "", "structure_sha256": "", "result_sha256": "",
        "identity_check_version": None, "applicability_domain": "unreported", "similarity_score": None, "results": {},
        "classification_validation": {}, "classification_validation_source": toxscan_validation.SOURCE,
        "corpus_provenance": corpus_provenance.declare(METHOD, against=ENDPOINT),
    }


def screen_run(directory: Path, candidate: str | None = None) -> dict:
    report = _base()
    try:
        fields = prepare(directory, candidate)
        benchmarks = toxscan_validation.for_reported_calls(fields["results"]["classifications"])
        return {**report, **fields, "classification_validation": benchmarks, "status": "complete"}
    except ToxScanError as error:
        return {**report, "blocker": str(error)}


def report_view(raw: object) -> dict:
    """Verify the export receipt, exact assay names, calls, identity and applicability on readback."""
    base = _base()
    invalid = {**base, "blocker": "toxscan_invalid_report"}
    if not isinstance(raw, Mapping) or any(raw.get(k) != base[k] for k in ("method", "endpoint", "scope", "source")):
        return invalid
    if raw.get("classification_validation_source", toxscan_validation.SOURCE) != toxscan_validation.SOURCE:
        return invalid
    if raw.get("status") == "untested":
        blocker = raw.get("blocker")
        if (not isinstance(blocker, str) or blocker not in BLOCKERS or raw.get("classification_validation", {}) != {}
                or raw.get("identity_check_version") is not None):
            return invalid
        if any(raw.get(k, object()) != base[k] for k in (
            "candidate_id", "source_sha256", "structure_sha256", "result_sha256", "applicability_domain",
            "similarity_score", "results",
        )):
            return invalid
        return {**base, "blocker": blocker}
    if (
        raw.get("status") != "complete" or "blocker" in raw or "similarity_score" not in raw
        or not isinstance(raw.get("candidate_id"), str) or not ligand_admet.valid_candidate_id(raw["candidate_id"])
        or any(not isinstance(raw.get(k), str) or not toxscan_inputs.SHA256.fullmatch(raw[k]) for k in (
            "source_sha256", "structure_sha256", "result_sha256",
        ))
    ):
        return invalid
    try:
        fields = {
            **{k: raw[k] for k in ("candidate_id", "source_sha256", "structure_sha256")},
            **toxscan_csv.validate_fields(dict(raw)),
        }
    except ToxScanError:
        return invalid
    # Original receipts predate stereo-loss checks and hashed only the normalized result fields.
    version = raw.get("identity_check_version", 1)
    if type(version) is not int or version not in (1, IDENTITY_CHECK_VERSION):
        return invalid
    if version == IDENTITY_CHECK_VERSION:
        fields["identity_check_version"] = version
    if _digest(fields) != raw["result_sha256"]:
        return invalid
    benchmarks = toxscan_validation.for_reported_calls(fields["results"]["classifications"])
    if "classification_validation" in raw and raw["classification_validation"] != benchmarks:
        return invalid
    return {
        **base, **fields, "result_sha256": raw["result_sha256"], "identity_check_version": version,
        "classification_validation": benchmarks, "status": "complete",
    }
