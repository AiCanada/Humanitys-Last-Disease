"""Attribute historical Badapple 2 assay evidence to nominated small molecules."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path

from rfproteina.disease import badapple2_runtime, ligand_admet, ligand_inputs
from rfproteina.disease.badapple2_runtime import BadappleRuntimeError
from rfproteina.disease.corpus_provenance import declare

METHOD_ID = "badapple2"
METHOD = "badapple2_hiers_pubchem_pscore"
ENDPOINT = "historical_scaffold_promiscuity"
SOURCES = ("ligand_candidates.tsv", "ligand_smiles.txt")
_MAX_HEAVY_ATOMS = 80
_FIELDS = {"id", "pscore", "in_drug", *badapple2_runtime.COUNT_FIELDS}
_INPUT_BLOCKERS = frozenset({
    "invalid_candidate_id", "invalid_nominated_smiles", "candidate_requires_named_nomination",
    "candidate_missing_from_table", "duplicate_candidate_nomination", "invalid_candidate_table_header",
    "ligand_input_missing", "ligand_input_unreadable", "ligand_input_exceeds_interactive_size",
    "expected_one_run_ligand", "unsupported_extended_smiles", "rdkit_unavailable",
    "badapple2_candidates_over_limit", "badapple2_unsupported_molecule", "badapple2_no_nominations",
    "badapple2_no_scaffolds", "badapple2_no_scored_scaffolds",
})
BLOCKERS = _INPUT_BLOCKERS | badapple2_runtime.BLOCKERS


class BadappleError(ValueError):
    """A known reason that preserves the scope of an untested result."""


@dataclass(frozen=True)
class Nomination:
    candidate_id: str
    smiles: str = ""
    blocker: str | None = None


def installed() -> bool:
    return find_spec("rdkit") is not None and badapple2_runtime.installed()


def prepare(directory: Path, candidate: str | None = None) -> tuple[str, list[Nomination]]:
    if candidate is not None and not ligand_admet.valid_candidate_id(candidate):
        raise BadappleError("invalid_candidate_id")
    scope, rows, blocker = ligand_inputs.bounded_smiles_candidates(
        directory, max_bytes=ligand_admet.MAX_INPUT_BYTES, max_rows=ligand_admet.MAX_CANDIDATES,
    )
    if blocker:
        raise BadappleError(blocker if blocker in _INPUT_BLOCKERS else "ligand_input_unreadable")
    if candidate is not None and scope != "candidate_table":
        raise BadappleError("candidate_requires_named_nomination")
    selected = list(ligand_inputs.nominated_rows(rows, candidate))
    if not selected:
        raise BadappleError("candidate_missing_from_table" if candidate is not None else "badapple2_no_nominations")
    if len(selected) > badapple2_runtime.MAX_CANDIDATES:
        raise BadappleError("badapple2_candidates_over_limit")
    try:
        from rdkit import Chem, rdBase
    except (ImportError, OSError) as error:
        raise BadappleError("rdkit_unavailable") from error

    counts = Counter(row[0] for row in selected)
    nominations = []
    with rdBase.BlockLogs():
        for row in selected:
            name = row[0]
            if not ligand_admet.valid_candidate_id(name):
                nominations.append(Nomination("", blocker="invalid_candidate_id"))
            elif counts[name] != 1:
                nominations.append(Nomination(name, blocker="duplicate_candidate_nomination"))
            elif len(row) != 2 or len(row[1]) > badapple2_runtime.MAX_SMILES:
                nominations.append(Nomination(name, blocker="invalid_nominated_smiles"))
            else:
                try:
                    molecule, key = ligand_admet.molecular_graph(
                        row[1], Chem, unsupported="badapple2_unsupported_molecule",
                    )
                    if molecule.GetNumHeavyAtoms() > _MAX_HEAVY_ATOMS or len(key) > badapple2_runtime.MAX_SMILES:
                        nominations.append(Nomination(name, blocker="badapple2_unsupported_molecule"))
                    else:
                        code = None if molecule.GetRingInfo().NumRings() else "badapple2_no_scaffolds"
                        nominations.append(Nomination(name, key, code))
                except ligand_admet.AdmetError as error:
                    nominations.append(Nomination(name, blocker=str(error)))
    return scope, nominations


def _valid_counts(row: dict) -> bool:
    if not all(
        row[key] is None or (type(row[key]) is int and 0 <= row[key] < 2**31)
        for key in badapple2_runtime.COUNT_FIELDS
    ):
        return False
    for prefix in ("nass", "ncpd", "nsub", "nsam"):
        active, tested = row[f"{prefix}_active"], row[f"{prefix}_tested"]
        if active is not None and tested is not None and active > tested:
            return False
    return True


def _valid_measurement(row: object) -> bool:
    if not isinstance(row, dict) or row.keys() != _FIELDS:
        return False
    if (
        type(row["id"]) is not int or not 0 < row["id"] < 2**31
        or (row["in_drug"] is not None and type(row["in_drug"]) is not bool)
        or not _valid_counts(row)
    ):
        return False
    pscore = row["pscore"]
    return pscore is None or (
        type(pscore) in (int, float) and 0 <= pscore <= 100_000
        and all(type(row[key]) is int and row[key] > 0 for key in ("nass_tested", "nsub_tested", "nsam_tested"))
    )


def _scaffold_rows(provided: object) -> list[dict]:
    if not isinstance(provided, dict):
        raise BadappleError("badapple2_invalid_report")
    blocker = provided.get("blocker")
    if provided.keys() == {"blocker"} and isinstance(blocker, str) and blocker in badapple2_runtime.BLOCKERS:
        raise BadappleError(blocker)
    data = provided.get("scaffolds")
    if provided.keys() != {"scaffolds"} or not isinstance(data, list) or len(data) > badapple2_runtime.MAX_SCAFFOLDS:
        raise BadappleError("badapple2_invalid_report")
    rows = {}
    for entry in data:
        if not isinstance(entry, dict) or entry.keys() != {"scafsmi", "matches"}:
            raise BadappleError("badapple2_invalid_report")
        key, matches = entry["scafsmi"], entry["matches"]
        if (
            not badapple2_runtime.valid_scaffold(key) or key in rows or not isinstance(matches, list)
            or len(matches) > 2 or not all(_valid_measurement(row) for row in matches)
            or len({row["id"] for row in matches}) != len(matches)
        ):
            raise BadappleError("badapple2_invalid_report")
        if len(matches) != 1:
            rows[key] = {"scafsmi": key, "status": "ambiguous" if matches else "not_in_corpus"}
        else:
            values = matches[0]
            rows[key] = {"scafsmi": key, "status": "unscored" if values["pscore"] is None else "reported", **values}
    return [rows[key] for key in sorted(rows)]


def _lookup(inputs: list[str]) -> dict:
    return badapple2_runtime.lookup(inputs)


def _record() -> dict:
    return {
        "method": METHOD, "endpoint": ENDPOINT, "snapshot": badapple2_runtime.SNAPSHOT,
        "dump_sha256": badapple2_runtime.DUMP_SHA256, "api_revision": badapple2_runtime.API_REVISION,
        "corpus_provenance": declare(METHOD, against=ENDPOINT), "status": "untested", "molecules": [],
    }


def _nomination_result(nomination: Nomination, rows: list[dict], blocker: str | None) -> dict:
    scored = sum(row["status"] == "reported" for row in rows)
    result = {"candidate_id": nomination.candidate_id, "input_smiles": nomination.smiles, "scaffolds": rows}
    if scored:
        return {**result, "status": "complete" if scored == len(rows) else "partial"}
    return {
        **result, "status": "untested",
        "blocker": blocker or ("badapple2_no_scored_scaffolds" if rows else "badapple2_no_scaffolds"),
    }


def _finish(report: dict, rows: list[dict]) -> dict:
    statuses = {row["status"] for row in rows}
    status = "complete" if statuses == {"complete"} else (
        "partial" if statuses.intersection({"complete", "partial"}) else "untested"
    )
    return {**report, "molecules": rows, "status": status}


def screen_run(directory: Path, candidate: str | None = None) -> dict:
    report = _record()
    try:
        scope, nominations = prepare(directory, candidate)
    except BadappleError as error:
        return {**report, "blocker": str(error)}
    report["input_scope"] = scope
    inputs = sorted({item.smiles for item in nominations if not item.blocker})
    evidence: dict[str, tuple[list[dict], str | None]] = {}
    if inputs:
        try:
            provided = _lookup(inputs)
            if not isinstance(provided, dict) or provided.keys() != set(inputs):
                raise BadappleRuntimeError("badapple2_invalid_report")
        except BadappleRuntimeError as error:
            evidence = dict.fromkeys(inputs, ([], str(error)))
        else:
            for key, raw in provided.items():
                try:
                    evidence[key] = (_scaffold_rows(raw), None)
                except BadappleError as error:
                    evidence[key] = ([], str(error))
    rows = [
        _nomination_result(item, *([[], item.blocker] if item.blocker else evidence[item.smiles]))
        for item in nominations
    ]
    return _finish(report, rows)


def _saved_rows(value: object) -> list[dict]:
    if not isinstance(value, list) or len(value) > badapple2_runtime.MAX_SCAFFOLDS:
        raise BadappleError("badapple2_invalid_report")
    rows = {}
    for row in value:
        if not isinstance(row, Mapping):
            raise BadappleError("badapple2_invalid_report")
        key, status = row.get("scafsmi"), row.get("status")
        if not badapple2_runtime.valid_scaffold(key) or key in rows:
            raise BadappleError("badapple2_invalid_report")
        entry = {"scafsmi": key, "status": status}
        if status in ("unscored", "reported"):
            fields = {name: row.get(name) for name in _FIELDS}
            if not _valid_measurement(fields) or (status == "unscored") != (fields["pscore"] is None):
                raise BadappleError("badapple2_invalid_report")
            entry.update(fields)
        elif status not in ("not_in_corpus", "ambiguous"):
            raise BadappleError("badapple2_invalid_report")
        rows[key] = entry
    return [rows[key] for key in sorted(rows)]


def report_view(raw: object) -> dict:
    """Project only the pinned endpoint, covered counts and attributed scaffold rows."""
    report = _record()
    invalid = {**report, "blocker": "badapple2_invalid_report"}
    if not isinstance(raw, Mapping) or any(raw.get(key) != report[key] for key in (
        "method", "endpoint", "snapshot", "dump_sha256", "api_revision",
    )):
        return invalid
    samples, blocker = raw.get("molecules"), raw.get("blocker")
    if samples == [] and raw.get("status") == "untested" and isinstance(blocker, str) and blocker in BLOCKERS:
        return {**report, "blocker": blocker}
    if not isinstance(samples, list) or not 0 < len(samples) <= badapple2_runtime.MAX_CANDIDATES:
        return invalid
    scope = raw.get("input_scope")
    if not ligand_inputs.valid_saved_smiles_scope(scope, samples):
        return invalid

    result = []
    try:
        for row in samples:
            if not isinstance(row, Mapping):
                return invalid
            name, smiles, row_blocker = row.get("candidate_id"), row.get("input_smiles"), row.get("blocker")
            if (
                not isinstance(name, str) or (name and not ligand_admet.valid_candidate_id(name))
                or not isinstance(smiles, str) or (smiles and not badapple2_runtime.valid_scaffold(smiles))
                or (row_blocker is not None and (not isinstance(row_blocker, str) or row_blocker not in BLOCKERS))
            ):
                return invalid
            rows = _saved_rows(row.get("scaffolds"))
            if rows and (not name or not smiles):
                return invalid
            evidence = _nomination_result(Nomination(name, smiles), rows, row_blocker)
            if evidence["status"] != row.get("status") or evidence.get("blocker") != row_blocker:
                return invalid
            result.append(evidence)
    except BadappleError:
        return invalid
    counts = Counter(row["candidate_id"] for row in result)
    if any(row["scaffolds"] and counts[row["candidate_id"]] != 1 for row in result):
        return invalid
    return _finish({**report, "input_scope": scope}, result)
