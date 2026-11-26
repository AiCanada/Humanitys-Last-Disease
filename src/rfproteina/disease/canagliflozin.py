"""Run the published canagliflozin multi-organ PBPK reference scenario."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from rfproteina.disease import canagliflozin_runtime as runtime
from rfproteina.disease import canagliflozin_worker as worker
from rfproteina.disease import corpus_provenance, ligand_inputs

METHOD_ID = "canagliflozin_pbpk"
METHOD = "canagliflozin_reference_pbpk"
ENDPOINT = "canagliflozin_multiorgan_reference_pk"
MANIFEST = "canagliflozin_pbpk.json"
SCENARIO = "oral_reference"
RECOVERY_STUDY_SOURCE = "https://www.accessdata.fda.gov/drugsatfda_docs/nda/2013/204042Orig1s000ClinPharmR.pdf#page=39"
RECOVERY_STUDY_N = 6
_RECOVERY_MEANS = {"urine": 32.5, "feces": 60.4}
BLOCKERS = frozenset({
    "canagliflozin_manifest_missing", "canagliflozin_invalid_manifest", "canagliflozin_requires_run_scope",
    "canagliflozin_invalid_report", *runtime.BLOCKERS,
})


class CanagliflozinError(ValueError):
    """The reference run lacks an attributable nomination or result."""


def prepare(directory: Path, candidate: str | None = None) -> str:
    """Require a run-level nomination for the fixed oral reference scenario."""
    if candidate is not None:
        raise CanagliflozinError("canagliflozin_requires_run_scope")
    try:
        raw = ligand_inputs.read_regular(directory / MANIFEST, 4096, "canagliflozin_manifest", CanagliflozinError)
        record = json.loads(raw.decode("utf-8-sig"), object_pairs_hook=runtime.unique_fields)
        if (
            not isinstance(record, dict) or record.keys() != {"version", "scenario"}
            or type(record["version"]) is not int or record["version"] != 1 or record["scenario"] != SCENARIO
        ):
            raise ValueError("invalid scenario")
    except (ValueError, RecursionError) as error:
        if str(error) == "canagliflozin_manifest_missing":
            raise
        raise CanagliflozinError("canagliflozin_invalid_manifest") from error
    return SCENARIO


def installed() -> bool:
    return runtime.installed()


def _recovery_comparison(measurements: dict) -> dict:
    """Compare seven-day modeled parent equivalents with the measured fraction of administered radiolabel."""
    dose_umol = worker.DOSE_MG * 1000 / worker.PARENT_MOLAR_MASS
    comparison = {"study_n": RECOVERY_STUDY_N, "study_source": RECOVERY_STUDY_SOURCE}
    for route, study_mean in _RECOVERY_MEANS.items():
        simulated = 100 * measurements[f"{route}_total_168h"]["value"] / dose_umol
        comparison[route] = {
            "simulated_percent_dose": simulated,
            "study_mean_percent_dose": study_mean,
            "difference_percentage_points": simulated - study_mean,
        }
    return comparison


def _base() -> dict:
    return {
        "method": METHOD, "endpoint": ENDPOINT, "scope": "reference_control", "scenario": SCENARIO,
        "release": runtime.MODEL_RELEASE, "model_revision": runtime.MODEL_REVISION,
        "model_sha256": runtime.MODEL_SHA256, "solver": "libroadrunner", "solver_version": worker.SOLVER_VERSION,
        "dose_mg": worker.DOSE_MG, "duration_h": worker.DURATION_H, "status": "untested",
        "measurements": {}, "profile": {}, "recovery_168h": {},
        "corpus_provenance": corpus_provenance.declare(METHOD, against=ENDPOINT),
    }


def screen_run(directory: Path, candidate: str | None = None) -> dict:
    """Run the fixed control after validating the nomination and pinned assets."""
    report = _base()
    try:
        prepare(directory, candidate)
        fields = worker.validated_result(runtime.simulate())
        return {**report, "status": "complete", **fields, "recovery_168h": _recovery_comparison(fields["measurements"])}
    except (CanagliflozinError, runtime.CanagliflozinRuntimeError) as error:
        return {**report, "blocker": str(error)}
    except ValueError:
        return {**report, "blocker": "canagliflozin_invalid_report"}


def report_view(raw: object) -> dict:
    """Project the exact reference release, units and scenario from a saved run."""
    report = _base()
    invalid = {**report, "blocker": "canagliflozin_invalid_report"}
    if not isinstance(raw, Mapping):
        return invalid
    for key in (
        "method", "endpoint", "scope", "scenario", "release", "model_revision", "model_sha256",
        "solver", "solver_version", "dose_mg", "duration_h",
    ):
        if type(raw.get(key)) is bool or raw.get(key) != report[key]:
            return invalid
    blocker = raw.get("blocker")
    if raw.get("status") == "untested":
        if not isinstance(blocker, str) or blocker not in BLOCKERS:
            return invalid
        if raw.get("measurements") != {} or raw.get("profile") != {} or raw.get("recovery_168h", {}) != {}:
            return invalid
        return {**report, "blocker": blocker}
    if raw.get("status") != "complete" or "blocker" in raw:
        return invalid
    try:
        fields = worker.validated_result({"measurements": raw.get("measurements"), "profile": raw.get("profile")})
    except ValueError:
        return invalid
    comparison = _recovery_comparison(fields["measurements"])
    # Older records contain the molar amounts. Derive the same comparison for them on readback.
    if "recovery_168h" in raw and raw["recovery_168h"] != comparison:
        return invalid
    return {**report, "status": "complete", **fields, "recovery_168h": comparison}
