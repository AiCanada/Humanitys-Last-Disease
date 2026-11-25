"""Bounded, local CCTBX/MolProbity geometry for finished protein coordinate models."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from functools import lru_cache
from importlib import metadata
from pathlib import Path

from rfproteina.disease import molprobity_inputs
from rfproteina.disease.corpus_provenance import declare

METHOD_ID = "molprobity_geometry"
METHOD = "cctbx_molprobity_geometry_v1"
MODEL_SCOPE = "whole_model"
CCTBX_VERSION = "2025.11"
MAX_MODELS = molprobity_inputs.MAX_MODELS
METRICS = ("ramachandran", "rotamers", "clashscore")
PROVIDER = {
    "cctbx_base": CCTBX_VERSION, "device": "cpu", "clash_preparation": "reduce2_no_flips", "clash_method": "probe2",
}
BLOCKERS = frozenset({
    "molprobity_models_missing", "molprobity_too_many_models", "molprobity_candidate_not_found",
    "molprobity_candidate_ambiguous", "molprobity_structure_unreadable", "molprobity_input_over_limit",
    "molprobity_zstd_unavailable", "molprobity_structure_invalid", "molprobity_runtime_unavailable",
    "molprobity_provider_failure", "molprobity_timeout", "molprobity_no_measurements", "molprobity_unsupported_model",
    "molprobity_reference_unavailable", "molprobity_incomplete_heavy_atoms", "molprobity_no_eligible_residues",
    "molprobity_no_eligible_contacts", "molprobity_metric_failed",
})
_CPU_ENV = {
    "CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": "", "ROCR_VISIBLE_DEVICES": "",
    "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
}
_WORKER = Path(__file__).with_name("molprobity_worker.py").resolve()
_MAX_RESPONSE_BYTES = 32 * 1024
_MODEL_TIMEOUT = 120.0
_RUN_TIMEOUT = 300.0


def runtime_path() -> Path | None:
    """Accept only administrator configuration or the known local install locations."""
    if configured := os.environ.get("RFD4_MOLPROBITY_PYTHON"):
        candidate = Path(configured).absolute()
        return candidate if candidate.is_file() else None
    if data := os.environ.get("DATA_PATH"):
        prefix = Path(data) / "molprobity" / "venv"
        candidate = (prefix / ("Scripts/python.exe" if os.name == "nt" else "bin/python")).absolute()
        if candidate.is_file():
            return candidate
    try:
        return Path(sys.executable) if metadata.version("cctbx-base") == CCTBX_VERSION else None
    except metadata.PackageNotFoundError:
        return None


def _worker_call(python: Path, action: str, data: bytes, timeout: float) -> dict:
    try:
        with tempfile.TemporaryDirectory(prefix="local-molprobity-") as directory:
            run = subprocess.run(
                [str(python), "-I", str(_WORKER), action], input=data, cwd=directory, timeout=timeout,
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True,
                env={**os.environ, **_CPU_ENV, **dict.fromkeys(("TMPDIR", "TMP", "TEMP", "MPLCONFIGDIR"), directory)},
            )
        if len(run.stdout) <= _MAX_RESPONSE_BYTES:
            response = json.loads(run.stdout)
            if isinstance(response, dict):
                return response
    except subprocess.TimeoutExpired:
        return {"blocker": "molprobity_timeout"}
    except (OSError, subprocess.SubprocessError, ValueError, RecursionError):
        pass
    return {"blocker": "molprobity_provider_failure"}


@lru_cache(maxsize=8)
def _probe(python: Path, executable_mtime: int, environment_mtime: int, refresh_epoch: int) -> bool:
    del executable_mtime, environment_mtime, refresh_epoch  # Renew the probe periodically and after install.
    return _worker_call(python, "probe", b"", 30) == {"cctbx_base": CCTBX_VERSION}


def asset_available() -> bool:
    """Check the configured CPU interpreter without parsing any candidate models."""
    try:
        if (python := runtime_path()) is None:
            return False
        environment = python.parent.parent
        return _probe(python, python.stat().st_mtime_ns, environment.stat().st_mtime_ns, int(time.monotonic() // 60))
    except OSError:
        return False


def validate_metric(name: str, value: object) -> dict:
    """Accept native denominators only; derive the displayed ratios from counts."""
    invalid = {"status": "untested", "blocker": "molprobity_metric_failed"}
    if not isinstance(value, dict) or name not in METRICS:
        return invalid
    if value.get("status") == "untested":
        reason = value.get("blocker")
        return {"status": "untested", "blocker": reason} if isinstance(reason, str) and reason in BLOCKERS else invalid
    if value.get("status") != "reported" or "blocker" in value:
        return invalid
    if name == "clashscore":
        atoms, clashes = value.get("atoms"), value.get("serious_clashes")
        if (type(atoms) is not int or not 0 < atoms <= 100_000 or type(clashes) is not int
                or not 0 <= clashes <= atoms * (atoms - 1) // 2):
            return invalid
        return {"status": "reported", "atoms": atoms, "serious_clashes": clashes,
                "clashes_per_1000_atoms": 1000 * clashes / atoms}
    fields = ("evaluated", "outliers", "favored", "allowed")
    count, outliers, favored, allowed = (value.get(key) for key in fields)
    if (any(type(number) is not int or not 0 <= number <= 16_000 for number in (count, outliers, favored, allowed))
            or count == 0 or outliers + favored + allowed != count):
        return invalid
    return {"status": "reported", "evaluated": count, "outliers": outliers, "favored": favored, "allowed": allowed,
            "outlier_percent": 100 * outliers / count}


def screen_run(directory: Path, *, candidate: str | None = None) -> dict:
    """Validate the original coordinate payloads. Never modify a model or gate the verdict."""
    run = {
        "method": METHOD, "model_scope": MODEL_SCOPE, "scope": "run" if candidate is None else "candidate",
        "status": "untested", "attempted": 0, "reported": 0, "complete": 0, "samples": [],
        "corpus_provenance": declare(METHOD),
    }
    inputs, blocker = molprobity_inputs.resolve_inputs(directory, candidate)
    if blocker:
        return {**run, "blocker": blocker}
    python = runtime_path()
    end = time.monotonic() + _RUN_TIMEOUT
    samples = []
    reported = complete = 0
    for model in inputs:
        row = {"candidate_id": model.candidate_id, "status": "untested"}
        samples.append(row)
        if model.blocker or model.path is None:
            row["blocker"] = model.blocker or "molprobity_structure_unreadable"
            continue
        if python is None:
            row["blocker"] = "molprobity_runtime_unavailable"
            continue
        if end <= time.monotonic():
            row["blocker"] = "molprobity_timeout"
            continue
        try:
            raw = molprobity_inputs.read_structure(model.path)
        except molprobity_inputs.MolProbityInputError as error:
            row["blocker"] = str(error)
            continue
        row["input_sha256"] = hashlib.sha256(raw).hexdigest()
        timeout = min(_MODEL_TIMEOUT, end - time.monotonic())
        if timeout <= 0:
            row["blocker"] = "molprobity_timeout"
            continue
        reply = _worker_call(python, "measure", raw, timeout)
        reason = reply.get("blocker")
        if isinstance(reason, str) and reason in BLOCKERS:
            row["blocker"] = reason
            continue
        findings = reply.get("metrics")
        if reply.get("provider") != PROVIDER or not isinstance(findings, dict):
            row["blocker"] = "molprobity_provider_failure"
            continue
        metrics = {name: validate_metric(name, findings.get(name)) for name in METRICS}
        count = sum(item["status"] == "reported" for item in metrics.values())
        row["metrics"] = metrics
        if count:
            reported += 1
            complete += count == len(METRICS)
            row["status"] = "reported" if count == len(METRICS) else "partial"
        else:
            row["blocker"] = "molprobity_no_measurements"
    run.update(attempted=len(samples), reported=reported, complete=complete, samples=samples)
    if reported:
        run.update(status="reported" if complete == len(samples) else "partial", provider=PROVIDER.copy())
    else:
        run["blocker"] = "molprobity_no_measurements"
    return run
