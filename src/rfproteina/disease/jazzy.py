"""Compute native Jazzy hydration energetics for one nominated molecular form."""

from __future__ import annotations

import re
from collections.abc import Mapping
from hashlib import sha256
from pathlib import Path

from rfproteina.disease import corpus_provenance, jazzy_runtime, jazzy_worker, ligand_admet, ligand_inputs

METHOD_ID = "jazzy"
METHOD = "jazzy_neutral_hydration_0_1_4"
ENDPOINT = "gas_to_water_hydration_free_energy_kj_mol"
SOURCES = ("ligand_candidates.tsv", "ligand_smiles.txt")
_MAX_CANDIDATES = 32
_HASH = re.compile(r"[a-f0-9]{64}")
_ENERGIES = ("hydration_kj_mol", "apolar_kj_mol", "polar_kj_mol", "interaction_kj_mol")
_STRENGTHS = ("ch_donor_strength", "xh_donor_strength", "acceptor_strength")
_ROW_BLOCKERS = frozenset({"jazzy_unsupported_candidate", "jazzy_geometry_unavailable", "jazzy_calculation_failed"})
BLOCKERS = frozenset(
    {
        "jazzy_candidates_unreadable",
        "jazzy_no_candidates",
        "jazzy_candidates_over_limit",
        "jazzy_duplicate_candidate",
        "jazzy_candidate_scope_missing",
        "jazzy_rdkit_unavailable",
        "jazzy_invalid_report",
        "jazzy_no_usable_results",
        *_ROW_BLOCKERS,
        *jazzy_runtime.BLOCKERS,
    }
)


class JazzyError(ValueError):
    """A nomination or result did not meet the qualified Jazzy contract."""


def _molecule(smiles: str) -> str | None:
    from rdkit import Chem

    try:
        molecule = ligand_inputs.parse_nominated_smiles(smiles, Chem, remove_hs=True) if len(smiles) <= 512 else None
        if molecule is None or not 2 <= molecule.GetNumHeavyAtoms() <= 40 or len(Chem.GetMolFrags(molecule)) != 1:
            return None
        allowed = {1, 6, 7, 8, 9, 15, 16, 17, 35, 53}
        atoms = tuple(molecule.GetAtoms())
        if not any(atom.GetAtomicNum() == 6 for atom in atoms) or any(
            atom.GetAtomicNum() not in allowed
            or atom.GetFormalCharge()
            or atom.GetIsotope()
            or atom.GetNumRadicalElectrons()
            or atom.GetAtomMapNum()
            for atom in atoms
        ):
            return None
        if any(stereo.specified != Chem.StereoSpecified.Specified for stereo in Chem.FindPotentialStereo(molecule)):
            return None
        return Chem.MolToSmiles(molecule, isomericSmiles=True)
    except (ValueError, RuntimeError):
        return None


def prepare(directory: Path, candidate: str | None = None) -> tuple[str, list[tuple[str, str | None]]]:
    scope, rows, blocker = ligand_inputs.bounded_smiles_candidates(
        directory,
        max_bytes=ligand_admet.MAX_INPUT_BYTES,
        max_rows=ligand_admet.MAX_CANDIDATES,
    )
    if blocker:
        raise JazzyError("jazzy_candidates_unreadable")
    if candidate is not None and (scope != "candidate_table" or not ligand_admet.valid_candidate_id(candidate)):
        raise JazzyError("jazzy_candidate_scope_missing")
    selected = list(ligand_inputs.nominated_rows(rows, candidate))
    if not selected:
        raise JazzyError("jazzy_no_candidates")
    if len(selected) > _MAX_CANDIDATES:
        raise JazzyError("jazzy_candidates_over_limit")
    if any(len(row) != 2 or not ligand_admet.valid_candidate_id(row[0]) for row in selected):
        raise JazzyError("jazzy_candidates_unreadable")
    if len({row[0] for row in selected}) != len(selected):
        raise JazzyError("jazzy_duplicate_candidate")
    try:
        from rdkit import rdBase
    except (ImportError, OSError) as error:
        raise JazzyError("jazzy_rdkit_unavailable") from error
    with rdBase.BlockLogs():
        return scope, [(name, _molecule(smiles)) for name, smiles in selected]


def installed() -> bool:
    return jazzy_runtime.installed()


def _finite(value: object, minimum: float) -> bool:
    return type(value) in (int, float) and minimum <= value <= 1e4


def _measures(row: Mapping) -> dict:
    if not (
        all(_finite(row.get(name), -1e4) for name in _ENERGIES)
        and all(_finite(row.get(name), 0) for name in _STRENGTHS)
    ):
        raise JazzyError("jazzy_invalid_report")
    total, apolar, polar, interaction = (row[key] for key in _ENERGIES)
    if abs(total - (apolar + polar + interaction)) > 0.00021:
        raise JazzyError("jazzy_invalid_report")
    return {key: row[key] for key in (*_ENERGIES, *_STRENGTHS)}


def _base() -> dict:
    return {
        "method": METHOD,
        "endpoint": ENDPOINT,
        "release": jazzy_runtime.RELEASE,
        "runtime_versions": dict(jazzy_runtime.VERSIONS),
        "geometry": "ETDG-3D-MMFF94",
        "embedding_seed": jazzy_worker.EMBEDDING_SEED,
        "embedding_attempts": jazzy_worker.EMBEDDING_ATTEMPTS,
        "mmff_iterations": jazzy_worker.MMFF_ITERATIONS,
        "status": "untested",
        "candidates": [],
        "corpus_provenance": corpus_provenance.declare(METHOD, against=ENDPOINT),
    }


def _with_rows(report: dict, rows: list[dict]) -> dict:
    count = sum(row["status"] == "predicted" for row in rows)
    if not count:
        return {**report, "candidates": rows, "blocker": "jazzy_no_usable_results"}
    return {**report, "candidates": rows, "status": "complete" if count == len(rows) else "partial"}


def screen_run(directory: Path, candidate: str | None = None) -> dict:
    report = _base()
    try:
        scope, nominations = prepare(directory, candidate)
        report["scope"] = scope
        batch = [smiles for _, smiles in nominations if smiles is not None]
        raw = jazzy_runtime.predict(batch) if batch else []
        if not isinstance(raw, list) or len(raw) != len(batch):
            raise JazzyError("jazzy_invalid_report")
        results = iter(raw)
        rows = []
        for name, smiles in nominations:
            if smiles is None:
                rows.append({"candidate_id": name, "status": "untested", "blocker": "jazzy_unsupported_candidate"})
                continue
            row = next(results)
            digest = sha256(smiles.encode("ascii")).hexdigest()
            if not isinstance(row, dict):
                raise JazzyError("jazzy_invalid_report")
            base = {"candidate_id": name, "candidate_sha256": digest}
            if row.keys() == {"blocker"} and row["blocker"] in (
                "jazzy_geometry_unavailable",
                "jazzy_calculation_failed",
            ):
                rows.append({**base, "status": "untested", "blocker": row["blocker"]})
            elif row.keys() == {*_ENERGIES, *_STRENGTHS}:
                rows.append({**base, "status": "predicted", **_measures(row)})
            else:
                raise JazzyError("jazzy_invalid_report")
        return _with_rows(report, rows)
    except (JazzyError, jazzy_runtime.JazzyRuntimeError) as error:
        return {**report, "blocker": str(error)}


def report_view(raw: object) -> dict:
    """Project bounded molecule identifiers and hydration measurements only."""
    report = _base()
    invalid = {**report, "blocker": "jazzy_invalid_report"}
    attestation = (
        "method",
        "endpoint",
        "release",
        "runtime_versions",
        "geometry",
        "embedding_seed",
        "embedding_attempts",
        "mmff_iterations",
    )
    if not isinstance(raw, Mapping) or any(raw.get(key) != report[key] for key in attestation):
        return invalid
    status, blocker, source, scope = (raw.get(key) for key in ("status", "blocker", "candidates", "scope"))
    if status == "untested" and source == [] and isinstance(blocker, str) and blocker in BLOCKERS:
        return {**report, "blocker": blocker}
    if (
        not ligand_inputs.valid_saved_smiles_scope(scope, source)
        or not 0 < len(source) <= _MAX_CANDIDATES
    ):
        return invalid
    rows = []
    names = set()
    try:
        for row in source:
            if (
                not isinstance(row, Mapping)
                or not isinstance((name := row.get("candidate_id")), str)
                or (not ligand_admet.valid_candidate_id(name) or name in names)
            ):
                return invalid
            names.add(name)
            row_status, row_blocker, digest = (row.get(field) for field in ("status", "blocker", "candidate_sha256"))
            result = {"candidate_id": name, "status": row_status}
            if row_blocker != "jazzy_unsupported_candidate":
                if not isinstance(digest, str) or not _HASH.fullmatch(digest):
                    return invalid
                result["candidate_sha256"] = digest
            if row_status == "untested" and isinstance(row_blocker, str) and row_blocker in _ROW_BLOCKERS:
                result["blocker"] = row_blocker
            elif row_status == "predicted" and "blocker" not in row:
                result.update(_measures(row))
            else:
                return invalid
            rows.append(result)
    except JazzyError:
        return invalid
    result = _with_rows({**report, "scope": scope}, rows)
    if result["status"] != status or result.get("blocker") != blocker:
        return invalid
    return result
