"""SCScore tie-breaking within one explicitly nominated small-molecule comparison."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import TYPE_CHECKING

from rfproteina.disease import ligand_alerts, ligand_inputs, scscore_model
from rfproteina.disease.corpus_provenance import declare
from rfproteina.disease.scscore_model import SCScoreError

if TYPE_CHECKING:
    from rdkit import Chem

METHOD_ID = "scscore"
METHOD = "scscore_reaxys_1024bool"
ENDPOINT = "synthetic_complexity"
MANIFEST = "scscore_comparison.json"
RESOLVED = "verified_route_and_procurement_costs"
_MAX_CANDIDATES = 256
_TABLE_BYTES = 1024 * 1024
_CANDIDATE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
_REFERENCE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:+-]{0,127}")
_EVIDENCE = {"verified_route_ref", "procurement_costs_ref"}


@dataclass(frozen=True)
class Member:
    name: str
    identity: str = ""
    molecule: Chem.Mol | None = None
    blocker: str | None = None


@dataclass(frozen=True)
class Comparison:
    names: tuple[str, ...]
    evidence: dict[str, str | bool]
    members: tuple[Member, ...]
    blocker: str | None

    def selected(self, candidate: str | None) -> tuple[Member, ...]:
        return self.members if candidate is None else tuple(m for m in self.members if m.name == candidate)


def _unique(pairs: list[tuple[str, object]]) -> dict:
    record = dict(pairs)
    if len(record) != len(pairs):
        raise SCScoreError("duplicate_comparison_field")
    return record


def _manifest(directory: Path) -> tuple[tuple[str, ...], dict[str, str | bool]]:
    raw = scscore_model.read_regular(directory / MANIFEST, 64 * 1024, "scscore_comparison")
    try:
        record = json.loads(raw.decode("utf-8-sig"), object_pairs_hook=_unique)
    except SCScoreError:
        raise
    except (ValueError, RecursionError) as exc:
        raise SCScoreError("invalid_scscore_comparison") from exc
    if not isinstance(record, dict) or record.keys() - {"candidates", "decision_evidence"}:
        raise SCScoreError("invalid_scscore_comparison")
    names = record.get("candidates")
    if not isinstance(names, list) or not 2 <= len(names) <= _MAX_CANDIDATES:
        raise SCScoreError("comparison_requires_two_to_256_candidates")
    if not all(isinstance(n, str) and _CANDIDATE.fullmatch(n) for n in names) or len(set(names)) != len(names):
        raise SCScoreError("invalid_or_duplicate_comparison_candidate")
    evidence = record.get("decision_evidence", {})
    if not isinstance(evidence, dict) or evidence.keys() - (_EVIDENCE | {"resolves_comparison"}):
        raise SCScoreError("invalid_decision_evidence")
    if not all(
        isinstance(v, str)
        and _REFERENCE.fullmatch(v)
        and v.lower()
        not in {"na", "none", "null", "unknown", "pending", "true", "false", "yes", "no", "tbd", "verified"}
        for k, v in evidence.items()
        if k in _EVIDENCE
    ):
        raise SCScoreError("invalid_decision_evidence_reference")
    resolved = evidence.get("resolves_comparison", False)
    if not isinstance(resolved, bool):
        raise SCScoreError("invalid_resolution_declaration")
    if resolved and not _EVIDENCE.issubset(evidence):
        raise SCScoreError("incomplete_route_and_cost_resolution")
    return tuple(names), evidence


def _member(name: str, rows: list[tuple[str, ...]]) -> Member:
    from rdkit import Chem

    if not rows:
        return Member(name, blocker="candidate_missing_from_table")
    if len(rows) != 1:
        return Member(name, blocker="duplicate_candidate_nomination")
    if len(rows[0]) != 2 or len(rows[0][1]) > 4096:
        return Member(name, blocker="invalid_molecular_nomination")
    try:
        mol = ligand_alerts.parse_molecule(rows[0][1], nominated=True)
        if (
            mol.GetNumAtoms() > 256
            or len(Chem.GetMolFrags(mol)) != 1
            or not any(a.GetAtomicNum() == 6 for a in mol.GetAtoms())
            or any(a.GetNumRadicalElectrons() for a in mol.GetAtoms())
        ):
            return Member(name, blocker="requires_one_supported_organic_molecule")
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
        return Member(name, Chem.MolToSmiles(mol, isomericSmiles=True), mol)
    except (ligand_alerts.LigandUnreadableError, ValueError, RuntimeError, OverflowError):
        return Member(name, blocker="invalid_molecular_nomination")


def prepare(directory: Path, candidate: str | None) -> Comparison:
    """Check the comparison and evidence without loading the network or generating fingerprints."""
    names, evidence = _manifest(directory)
    if candidate is not None and candidate not in names:
        return Comparison((), {}, (), blocker="candidate_outside_comparison")
    if evidence.get("resolves_comparison"):
        return Comparison(names, evidence, (), RESOLVED)
    table = directory / "ligand_candidates.tsv"
    if not table.exists() and not table.is_symlink():
        return Comparison(names, evidence, tuple(Member(n) for n in names), blocker="named_candidate_table_required")
    # Each nonempty row costs at least one byte. Only the nominated subset is subject to the model's batch limit.
    scope, rows, problem = ligand_inputs.bounded_smiles_candidates(
        directory, max_bytes=_TABLE_BYTES, max_rows=_TABLE_BYTES
    )
    if scope != "candidate_table" or problem:
        return Comparison(
            names,
            evidence,
            tuple(Member(n) for n in names),
            blocker=problem if scope == "candidate_table" else "named_candidate_table_required",
        )
    by_name = {name: [] for name in names}
    for row in rows:
        if row and row[0] in by_name:
            by_name[row[0]].append(row)
    try:
        from rdkit import rdBase

        with rdBase.BlockLogs():
            members = tuple(_member(name, by_name[name]) for name in names)
    except (ImportError, OSError):
        return Comparison(names, evidence, tuple(Member(n) for n in names), blocker="rdkit_unavailable")
    identities = {member.identity for member in members if member.identity}
    return Comparison(
        names, evidence, members, blocker=None if len(identities) >= 2 else "two_distinct_compounds_required"
    )


def measure(directory: Path, candidate: str | None = None) -> dict:
    """Report attributed scores. Only a complete comparison receives within-group ranks."""
    samples = []
    report = {
        "method": METHOD,
        "endpoint": ENDPOINT,
        "unit": "SCScore",
        "scale": [1, 5],
        "interpretation": (
            "Higher scores indicate greater predicted complexity. Route feasibility requires chemistry review."
        ),
        "scope": "comparison" if candidate is None else "candidate",
        "status": "untested",
        "attempted": 0,
        "scored": 0,
        "samples": samples,
        "corpus_provenance": declare(METHOD, against=ENDPOINT),
    }
    try:
        comparison = prepare(directory, candidate)
    except SCScoreError as exc:
        return {**report, "blocker": str(exc)}
    if comparison.names:
        report.update(comparison_size=len(comparison.names), decision_evidence=comparison.evidence)
    if comparison.blocker == RESOLVED:
        return {**report, "status": "skipped", "reason": RESOLVED}
    selected = comparison.selected(candidate)
    report["attempted"] = len(selected)
    molecules = {}
    for member in selected:
        row = {"candidate_id": member.name, "method": METHOD, "status": "untested"}
        samples.append(row)
        if member.identity:
            row["compound_id"] = "smiles_sha256_" + hashlib.sha256(member.identity.encode()).hexdigest()
            molecules[member.identity] = member.molecule
        if member.blocker or comparison.blocker:
            row["blocker"] = member.blocker or comparison.blocker
    if comparison.blocker or not molecules:
        return {**report, "blocker": comparison.blocker or "no_eligible_nominations"}
    try:
        values, report["provider"] = scscore_model.score(molecules)
    except SCScoreError as exc:
        report["blocker"] = str(exc)
    except (ImportError, AttributeError, OSError, RuntimeError, ValueError, OverflowError):
        report["blocker"] = "scscore_provider_unavailable"
    if "blocker" in report:
        for row in samples:
            row.setdefault("blocker", report["blocker"])
        return report
    for member, row in zip(selected, samples, strict=True):
        if member.blocker:
            continue
        value = values.get(member.identity)
        if isinstance(value, Real) and not isinstance(value, bool) and 1 <= value <= 5:
            row.update(status="scored", scscore=float(value))
            report["scored"] += 1
        else:
            row["blocker"] = (
                "scscore_fingerprint_unavailable"
                if isinstance(value, str) and value == "scscore_fingerprint_unavailable"
                else "invalid_scscore_result"
            )
    if report["scored"] == len(samples):
        report["status"] = "scored"
        if candidate is None:
            scores = sorted({row["scscore"] for row in samples})
            ranks = {value: rank for rank, value in enumerate(scores, 1)}
            for row in samples:
                row["comparison_rank"] = ranks[row["scscore"]]
    elif report["scored"]:
        report["status"] = "partial"
    else:
        report["blocker"] = "no_valid_scscore_results"
    return report
