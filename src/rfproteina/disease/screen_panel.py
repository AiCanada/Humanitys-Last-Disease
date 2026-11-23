"""Run selected supplemental measurements on the finished run or one identified candidate."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

from rfproteina.disease import (
    ligand_metabolites,
    ligand_solubility,
    qupkake,
    screen,
    screen_assays,
    screen_inventory,
    screen_physical,
    screen_selection,
)
from rfproteina.disease.prodigy import RunAffinity
from rfproteina.disease.solubility import read_protein_sol, read_solubility
from rfproteina.disease.tnp import RunProfiles
from rfproteina.disease.validation_checks import ValidationCandidate, _candidates, _inventory, _select_candidate
from rfproteina.disease.vina_pose import RunPoses


def context(directory: Path, candidate_key: str | None) -> tuple[str, ValidationCandidate | None]:
    from rfproteina.disease.cli import _screened_identity

    kind, _ = _screened_identity(directory, [], regular_files_only=True)
    candidate = (
        _select_candidate(_candidates(_inventory(directory)), candidate_key) if candidate_key is not None else None
    )
    return screen._canonical_kind(kind or ""), candidate


def choices_for(directory: Path, candidate_key: str | None) -> tuple[screen_selection.Choice, ...]:
    kind, candidate = context(directory, candidate_key)
    return screen_selection.plan(directory, kind, example=(candidate.example_id or "") if candidate else None)


def _measure(
    directory: Path, kind: str, key: str, candidate: ValidationCandidate | None, structures: screen_physical.Structures
) -> dict | None:
    example = candidate.example_id if candidate else None
    if key == "ligand_alerts":
        if (directory / "ligand_candidates.tsv").is_file():
            return screen_assays.safe_measure(directory, key, example)
        return screen._ligand_alerts_block(directory)
    if key == "ligand_solubility":
        return ligand_solubility.screen_run(directory, candidate=example)
    if key in {"qupkake", "sygma"}:
        provider = qupkake if key == "qupkake" else ligand_metabolites
        return provider.screen_run(directory, kind=kind, candidate=example, use_defaults=True)
    if key == "solubility":
        return read_solubility(directory, candidate=example)
    if key == "protein_sol":
        return read_protein_sol(directory, structures.chains)
    providers = {"prodigy_affinity": RunAffinity, "tnp_profile": RunProfiles, "vina_pose": RunPoses}
    provider = providers[key].configured(directory)
    if provider is None:
        return None
    for name, path in structures.paths.items():
        provider.observe(name, screen._design_text(path))
    result = provider.as_record()
    if candidate and "unmatched_manifest_ids" in result:
        result.pop("unmatched_manifest_ids")
    return result


def _status(key: str, report: dict) -> str:
    if isinstance(report.get("status"), str):
        return report["status"]
    if any(value for name, value in report.items() if name.endswith("blocker")):
        return "untested"
    if key == "ligand_alerts" and report.get("missing_catalogues"):
        return "partial"
    if key == "solubility" and report.get("incomplete_rows"):
        return "partial"
    if key == "protein_sol":
        samples = report.get("samples", [])
        completed = sum(isinstance(sample.get("score"), int | float) for sample in samples)
        attempted = len(samples)
    elif key in {"tnp_profile", "vina_pose", "prodigy_affinity"}:
        count = {"tnp_profile": "profiled", "vina_pose": "scored", "prodigy_affinity": "predicted"}[key]
        completed = report.get(count, 0)
        attempted = report.get("attempted", 0)
        if report.get("partial"):
            return "partial"
    else:
        return "reported"
    if not completed:
        return "untested"
    return "reported" if completed == attempted else "partial"


def run(
    directory: Path,
    candidate_key: str | None,
    mode: str,
    requested: Sequence[str] = (),
    *,
    on_line: Callable[[str], None] | None = None,
) -> list[dict]:
    """Evaluate the selected methods serially. Every skipped input or failed provider retains its own result."""
    from rfproteina.disease.mcp_api import _redact, _verdict

    kind, candidate = context(directory, candidate_key)
    example = (candidate.example_id or "") if candidate else None
    evidence = screen_inventory.Evidence(directory, example)
    choices, selected = screen_selection.execution_plan(
        directory, kind, mode, requested, example=example, evidence=evidence
    )
    results = []
    structures = screen_physical.Structures(directory, candidate)
    for choice in choices:
        key = choice.method.key
        if key not in selected:
            continue
        if not choice.applicable:
            results.append({"id": key, "status": "inapplicable", "reason": choice.reason})
            if on_line is not None:
                on_line(f"Skipped {choice.method.label}. {choice.reason}")
            continue
        if on_line is not None:
            on_line(f"Running {choice.method.label}.")
        if choice.method.operation == "recorded_evidence":
            report = evidence.coverage(screen_inventory.BY_KEY[key], include_values=True)
            measurement = _redact(report)
        elif key in screen_assays.FILES:
            report = screen_assays.safe_measure(directory, key, example)
            measurement = _redact(report)
        elif key in screen_physical.METHODS:
            report = structures.measure(key)
            measurement = _redact(report)
        else:
            try:
                report = _measure(directory, kind, key, candidate, structures)
            except (ImportError, OSError, RuntimeError, ValueError, OverflowError):
                report = {"blocker": "provider_or_input_unavailable"}
            report = report or {"blocker": "input_unavailable"}
            measurement = _verdict({key: report}, include_record=False).get(key, {})
        status = _status(key, report)
        results.append({"id": key, "status": status, "measurement": measurement})
        if on_line is not None:
            on_line(f"Finished {choice.method.label} ({status}).")
    return results
