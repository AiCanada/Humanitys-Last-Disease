"""Run selected supplemental measurements on the finished run or one identified candidate."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from rfproteina.disease import (
    canagliflozin,
    delphi_psr,
    httk,
    ligand_ames,
    ligand_caco2,
    ligand_dili,
    ligand_inputs,
    ligand_metabolites,
    ligand_microsomal_clearance,
    ligand_solubility,
    molprobity_geometry,
    peptide_rbc,
    qupkake,
    screen,
    screen_assays,
    screen_criteria,
    screen_inventory,
    screen_physical,
    screen_selection,
    toxscan,
    toxscan_ames,
)
from rfproteina.disease.prodigy import RunAffinity
from rfproteina.disease.solubility import read_protein_sol, read_solubility
from rfproteina.disease.tnp import RunProfiles
from rfproteina.disease.validation_checks import ValidationCandidate, _candidates, _inventory
from rfproteina.disease.vina_pose import RunPoses


@dataclass(frozen=True)
class ScreenCandidate:
    key: str
    label: str
    example_id: str | None
    structure: ValidationCandidate | None


def _kind(directory: Path) -> str:
    from rfproteina.disease.cli import _screened_identity

    kind, _ = _screened_identity(directory, [], regular_files_only=True)
    return screen._canonical_kind(kind or "")


NOMINATION_FILES = frozenset({"ligand_candidates.tsv", canagliflozin.MANIFEST})


def nominated_ligand_run(directory: Path, *, binder_kind: str | None = None) -> bool:
    """Recognize a prepared ligand run without requiring generated coordinate files."""
    has_table = (directory / "ligand_candidates.tsv").is_file()
    if not (has_table or (directory / canagliflozin.MANIFEST).is_file()):
        return False
    kind = screen._canonical_kind(binder_kind) if binder_kind is not None else _kind(directory)
    if kind != "ligand":
        return False
    if has_table and ligand_inputs.named_nominations(directory):
        return True
    try:
        canagliflozin.prepare(directory)
    except canagliflozin.CanagliflozinError:
        return False
    return True


def matches_toxscan_nomination(directory: Path, report: Mapping) -> bool:
    """Compare an earlier import receipt with the nomination already checked by the current plan."""
    from rfproteina.disease import toxscan_inputs

    if report.get("identity_check_version") != toxscan.IDENTITY_CHECK_VERSION:
        return False
    try:
        nomination = toxscan_inputs.nomination(directory, None)
    except toxscan_inputs.ToxScanError:
        return False
    return (
        report.get("source_sha256") == nomination.export_sha256
        and report.get("structure_sha256") == toxscan_inputs.sha256(nomination.canonical_smiles.encode("ascii"))
        and report.get("candidate_id_sha256") == toxscan_inputs.sha256(nomination.candidate_id.encode("utf-8"))
    )


def matches_toxscan_ames_nomination(directory: Path, candidate: str | None, report: Mapping) -> bool:
    """Check the current nomination and verified checkpoint without rerunning inference."""
    from rfproteina.disease import toxscan_ames_model, toxscan_inputs

    try:
        candidate_hash, graph_hash = toxscan_ames.nomination_identity(directory, candidate)
    except toxscan_inputs.ToxScanError:
        return False
    return (
        report.get("candidate_id_sha256") == candidate_hash
        and report.get("structure_sha256") == graph_hash
        and report.get("model_sha256") is not None
        and report["model_sha256"] == toxscan_ames_model.installed_sha256()
    )


def candidates_for(
    directory: Path, *, structures: Sequence[ValidationCandidate] | None = None
) -> tuple[ScreenCandidate, ...]:
    """Retain structural selectors and add uniquely identified, named small molecules."""
    from rfproteina.disease.mcp_api import _redact

    structures = _candidates(_inventory(directory)) if structures is None else structures
    candidates = [ScreenCandidate(c.key, c.label, c.example_id, c) for c in structures]
    if _kind(directory) == "ligand":
        structured = {c.example_id for c in candidates}
        for index, (name, smiles) in enumerate(ligand_inputs.named_nominations(directory).items(), 1):
            if name in structured:
                continue
            # A selector for a replaced molecule requires a fresh plan even when the name is reused.
            identity = "\0".join((directory.resolve().as_posix(), "ligand_candidates.tsv", name, smiles))
            key = hashlib.sha256(identity.encode()).hexdigest()[:24]
            label = name if _redact(name) == name else f"Ligand {index}"
            candidates.append(ScreenCandidate(key, label, name, None))
    return tuple(candidates)


def context(
    directory: Path, candidate_key: str | None, candidates: Sequence[ScreenCandidate] | None = None,
) -> tuple[str, ScreenCandidate | None]:
    kind = _kind(directory)
    if candidate_key is None:
        return kind, None
    for candidate in candidates if candidates is not None else candidates_for(directory):
        if candidate.key == candidate_key:
            return kind, candidate
    raise ValueError("Choose an available candidate from the current screening plan.")


def choices_for(
    directory: Path, candidate_key: str | None, candidates: Sequence[ScreenCandidate] | None = None,
) -> tuple[screen_selection.Choice, ...]:
    kind, candidate = context(directory, candidate_key, candidates)
    example = (candidate.example_id or "") if candidate else None
    evidence = screen_inventory.Evidence(directory, example, kind=kind)
    return screen_selection.plan(directory, kind, example=example, evidence=evidence)


def _measure(
    directory: Path, kind: str, key: str, candidate: ScreenCandidate | None, structures: screen_physical.Structures
) -> dict | None:
    example = (candidate.example_id or "") if candidate else None
    if key == delphi_psr.METHOD_ID:
        return delphi_psr.screen_run(directory, candidate=(example or "") if candidate else None)
    if key == molprobity_geometry.METHOD_ID:
        return molprobity_geometry.screen_run(directory, candidate=(example or "") if candidate else None)
    if key == peptide_rbc.METHOD_ID:
        return peptide_rbc.screen_run(
            directory,
            candidate=(example or "") if candidate else None,
            known_candidates=screen_criteria.peptide_rbc_known_candidates(directory),
        )
    if key == "ligand_alerts":
        if (directory / "ligand_candidates.tsv").is_file():
            return screen_assays.safe_measure(directory, key, example)
        return screen._ligand_alerts_block(directory)
    if key == "ligand_solubility":
        return ligand_solubility.screen_run(directory, candidate=example)
    if key == ligand_ames.METHOD_ID:
        return ligand_ames.screen_run(directory, candidate=example)
    if key == ligand_caco2.METHOD_ID:
        return ligand_caco2.screen_run(directory, candidate=example)
    if key == ligand_dili.METHOD_ID:
        return ligand_dili.screen_run(directory, candidate=example)
    if key == ligand_microsomal_clearance.METHOD_ID:
        return ligand_microsomal_clearance.screen_run(directory, candidate=example)
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


def _redact_fresh_assay(key: str, report: dict) -> dict:
    from rfproteina.disease.mcp_api import _AttestationDigest, _Prose, _redact

    if key == "badapple2":
        from rfproteina.disease.mcp_api import _badapple2_view

        return _redact(_badapple2_view(report))
    if key == "dimorphite_dl":
        from rfproteina.disease.mcp_api import _dimorphite_view

        return _redact(_dimorphite_view(report))
    if key == "p2rank":
        from rfproteina.disease.mcp_api import _p2rank_view

        return _redact(_p2rank_view(report))
    if key == "propka":
        from rfproteina.disease.mcp_api import _propka_view

        return _redact(_propka_view(report))
    if key == "crest":
        from rfproteina.disease.mcp_api import _crest_view

        return _redact(_crest_view(report))
    if key == canagliflozin.METHOD_ID:
        from rfproteina.disease.mcp_api import _canagliflozin_view

        return _redact(_canagliflozin_view(report))
    if key == toxscan.METHOD_ID:
        from rfproteina.disease.mcp_api import _toxscan_view

        return _redact(_toxscan_view(report))
    if key == toxscan_ames.METHOD_ID:
        from rfproteina.disease.mcp_api import _toxscan_ames_view

        return _redact(_toxscan_ames_view(report))
    if key == "openantigens":
        from rfproteina.disease.mcp_api import _openantigens_view

        return _redact(_openantigens_view(report))
    if key == "metfrag":
        from rfproteina.disease.mcp_api import _metfrag_view

        return _redact(_metfrag_view(report))
    if key == "fastsolv":
        from rfproteina.disease.mcp_api import _fastsolv_view

        return _redact(_fastsolv_view(report))
    if key == "jazzy":
        from rfproteina.disease.mcp_api import _jazzy_view

        return _redact(_jazzy_view(report))
    if key == "espsim":
        from rfproteina.disease.mcp_api import _espsim_view

        return _redact(_espsim_view(report))
    if key == "bayesherg":
        from rfproteina.disease.ligand_safety import _BAYESHERG_WEIGHTS_SHA256

        result = report.copy()
        for field in ("inputs", "compounds"):
            rows = []
            for original in report.get(field, []):
                row = original.copy()
                identity = row.get("compound_id")
                if isinstance(identity, str) and re.fullmatch(r"smiles_sha256_[a-f0-9]{64}", identity):
                    row["compound_id"] = _AttestationDigest(identity)
                provenance = row.get("provenance")
                if isinstance(provenance, dict) and provenance.get("weights_sha256") == _BAYESHERG_WEIGHTS_SHA256:
                    row["provenance"] = {**provenance, "weights_sha256": _AttestationDigest(_BAYESHERG_WEIGHTS_SHA256)}
                rows.append(row)
            result[field] = rows
        return _redact(result)
    if key == httk.METHOD_ID:
        from rfproteina.disease.mcp_api import _httk_view

        return _redact(_httk_view(report))
    if key == "cellprofiler":
        from rfproteina.disease import cellprofiler_runtime

        # These digests attest to bytes read during this run. Other labels in
        # the image metadata still pass through the usual redaction.
        def digest_fields(record: dict, keys: tuple[str, ...]) -> dict:
            result = record.copy()
            for name in keys:
                value = result.get(name)
                if isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value):
                    result[name] = _AttestationDigest(value)
            return result

        result = digest_fields(report, ("manifest_sha256",))
        if result.get("provider_image_digest") != cellprofiler_runtime.DIGEST:
            result.pop("provider_image_digest", None)
        if "comparisons" in result:
            comparisons = []
            for comparison in result["comparisons"]:
                row = comparison.copy()
                for role in ("test", "negative", "positive"):
                    if isinstance(row.get(role), dict):
                        row[role] = digest_fields(row[role], ("image_sha256", "mask_sha256"))
                comparisons.append(row)
            result["comparisons"] = comparisons
        return _redact(result)
    if key != "pksim" or not report.get("samples"):
        return _redact(report)
    # Fresh PK-Sim rows derive identity from the nominated structure and hash the
    # actual model and output bytes. Saved records have no such trust boundary.
    samples = []
    for sample in report["samples"]:
        row = dict(sample)
        inchikey = row.get("compound_inchikey")
        if isinstance(inchikey, str) and re.fullmatch(r"[A-Z]{14}-[A-Z]{10}-[A-Z]", inchikey):
            row["compound_inchikey"] = _Prose(inchikey)
        for field in ("model_sha256", "export_sha256"):
            digest = row.get(field)
            if isinstance(digest, str) and re.fullmatch(r"[a-f0-9]{64}", digest):
                row[field] = _AttestationDigest(digest)
        samples.append(row)
    return _redact({**report, "samples": samples})


def run(
    directory: Path,
    candidate_key: str | None,
    mode: str,
    requested: Sequence[str] = (),
    *,
    on_line: Callable[[str], None] | None = None,
    include_performance: bool = False,
) -> list[dict]:
    """Evaluate the selected methods serially. Every skipped input or failed provider retains its own result."""
    from rfproteina.disease.mcp_api import _redact, _verdict

    kind, candidate = context(directory, candidate_key)
    example = (candidate.example_id or "") if candidate else None
    evidence = screen_inventory.Evidence(directory, example, kind=kind)
    choices, selected = screen_selection.execution_plan(
        directory, kind, mode, requested, example=example, evidence=evidence
    )
    # A provider can replace its local model manifest during this run. Capture the
    # evidence for the entire selection before starting the first calculation.
    selected_choices = [choice for choice in choices if choice.method.key in selected]
    results = [
        {"id": choice.method.key, **({"performance": choice.performance} if include_performance else {})}
        for choice in selected_choices
    ]
    structures = screen_physical.Structures(directory, candidate.structure if candidate else None)
    for choice, result in zip(selected_choices, results, strict=True):
        key = choice.method.key
        if not choice.applicable:
            result.update(status="inapplicable", reason=choice.reason)
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
            measurement = _redact_fresh_assay(key, report)
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
        projected = {
            canagliflozin.METHOD_ID, httk.METHOD_ID, molprobity_geometry.METHOD_ID, peptide_rbc.METHOD_ID,
            toxscan.METHOD_ID, toxscan_ames.METHOD_ID,
        }
        status = _status(key, measurement if key in projected else report)
        result.update(status=status, measurement=measurement)
        if on_line is not None:
            on_line(f"Finished {choice.method.label} ({status}).")
    return results
