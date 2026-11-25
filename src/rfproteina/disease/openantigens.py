"""Read public OpenAntigens evidence for an explicitly identified human run target."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path

from rfproteina.disease import corpus_provenance, ligand_inputs
from rfproteina.disease import openantigens_snapshot as snapshot

METHOD_ID = "openantigens"
METHOD = "openantigens_human_target_index"
ENDPOINT = "human_surface_target_evidence"
MANIFEST = "openantigens.json"
_ENTRY = re.compile(r"[A-Z0-9]{1,20}_HUMAN")
_TOPOLOGIES = ("GPI", "Multipass", "Secreted", "Single-pass")
_FLAGS = ("has_alphafold_structure", "has_pdb", "has_interpro", "has_family_context", "has_assembly_requirements")
BLOCKERS = frozenset({
    "openantigens_target_missing", "invalid_openantigens_manifest", "openantigens_invalid_job",
    "openantigens_target_mismatch", "openantigens_target_unresolved", "openantigens_no_target_evidence",
    "openantigens_requires_run_scope", "openantigens_invalid_report", *snapshot.BLOCKERS,
})


class OpenAntigensError(ValueError):
    """The requested target could not be attributed to this run and snapshot."""


def _unique(pairs: list[tuple[str, object]]) -> dict:
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("Duplicate field.")
    return result


def _manifest(directory: Path) -> str | None:
    try:
        data = ligand_inputs.read_regular(directory / MANIFEST, 4096, "openantigens_manifest", OpenAntigensError)
        settings = json.loads(data.decode("utf-8-sig"), object_pairs_hook=_unique)
        if not (
            isinstance(settings, dict) and settings.keys() == {"version", "target_accession"}
            and type(settings["version"]) is int and settings["version"] == 1
            and snapshot.valid_accession(settings["target_accession"])
        ):
            raise ValueError("Invalid target nomination.")
        return settings["target_accession"]
    except (ValueError, RecursionError) as error:
        if isinstance(error, OpenAntigensError) and str(error) == "openantigens_manifest_missing":
            return None
        raise OpenAntigensError("invalid_openantigens_manifest") from error


def _record_target(path: Path) -> str | None:
    try:
        data = ligand_inputs.read_regular(path, 512 * 1024, "openantigens_job", OpenAntigensError)
        job = json.loads(data.decode("utf-8-sig"), object_pairs_hook=_unique)
        if not isinstance(job, dict) or ("target_id" in job and not isinstance(job["target_id"], str)):
            raise ValueError("Invalid run target metadata.")
        target = job.get("target_id")
    except (ValueError, RecursionError) as error:
        if isinstance(error, OpenAntigensError) and str(error) == "openantigens_job_missing":
            return None
        raise OpenAntigensError("openantigens_invalid_job") from error
    if not target:
        return None
    if not snapshot.valid_accession(target):
        raise OpenAntigensError("openantigens_target_unresolved")
    return target


def _job_target(directory: Path) -> str | None:
    from rfproteina.disease.screen import base_name, job_file_for_directory

    canonical = directory / f"{base_name(directory)}.job.json"
    if canonical.exists() or canonical.is_symlink():
        path = canonical
    else:
        paths = iter(directory.glob("*.job.json"))
        if next(paths, None) is not None and next(paths, None) is not None:
            raise OpenAntigensError("openantigens_invalid_job")
        path = job_file_for_directory(directory)
    job = _record_target(path) if path is not None else None
    generation = _record_target(directory / "generation.json")
    if job and generation and job != generation:
        raise OpenAntigensError("openantigens_target_mismatch")
    return generation or job


def _target_view(raw: object) -> dict:
    if not isinstance(raw, Mapping) or not (
        isinstance(raw.get("entry_name"), str) and _ENTRY.fullmatch(raw["entry_name"])
        and raw.get("topology") in _TOPOLOGIES and all(type(raw.get(flag)) is bool for flag in _FLAGS)
    ):
        raise OpenAntigensError("openantigens_invalid_report")
    return {name: raw[name] for name in ("entry_name", "topology", *_FLAGS)}


def installed() -> bool:
    return snapshot.installed()


def prepare(directory: Path, candidate: str | None = None, *, expected_target: str | None = None) -> tuple[str, dict]:
    if candidate is not None:
        raise OpenAntigensError("openantigens_requires_run_scope")
    nominated, recorded = _manifest(directory), _job_target(directory)
    if nominated and recorded and nominated != recorded:
        raise OpenAntigensError("openantigens_target_mismatch")
    accession = recorded or nominated
    if accession is None:
        raise OpenAntigensError("openantigens_target_missing")
    if expected_target and expected_target != accession:
        raise OpenAntigensError("openantigens_target_mismatch")
    evidence = snapshot.target(accession)
    if evidence is None:
        raise OpenAntigensError("openantigens_no_target_evidence")
    return accession, _target_view(evidence)


def _base() -> dict:
    return {
        "method": METHOD, "endpoint": ENDPOINT, "release": snapshot.RELEASE, "scope": "run_target",
        "index_sha256": snapshot.INDEX_SHA256, "accession_map_sha256": snapshot.ACCESSION_MAP_SHA256,
        "status": "untested", "corpus_provenance": corpus_provenance.declare(METHOD, against=ENDPOINT),
    }


def screen_run(directory: Path, candidate: str | None = None, *, expected_target: str | None = None) -> dict:
    report = _base()
    try:
        accession, target = prepare(directory, candidate, expected_target=expected_target)
    except (OpenAntigensError, snapshot.SnapshotError) as error:
        return {**report, "blocker": str(error)}
    return {**report, "status": "reported", "target_accession": accession, "target": target}


def report_view(raw: object) -> dict:
    """Keep only typed target metadata from this published release."""
    report = _base()
    invalid = {**report, "blocker": "openantigens_invalid_report"}
    attestation = ("method", "endpoint", "release", "scope", "index_sha256", "accession_map_sha256")
    if not isinstance(raw, Mapping) or any(raw.get(name) != report[name] for name in attestation):
        return invalid
    blocker = raw.get("blocker")
    if raw.get("status") == "untested" and isinstance(blocker, str) and blocker in BLOCKERS:
        return {**report, "blocker": blocker}
    accession = raw.get("target_accession")
    if raw.get("status") != "reported" or not snapshot.valid_accession(accession) or "blocker" in raw:
        return invalid
    try:
        target = _target_view(raw.get("target"))
    except OpenAntigensError:
        return invalid
    return {**report, "status": "reported", "target_accession": accession, "target": target}
