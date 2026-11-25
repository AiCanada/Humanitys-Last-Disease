"""Read intended therapeutic exposure for named compounds without running a screen."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from rfproteina.disease import cypstrate, ligand_admet, ligand_inputs

MANIFEST = "screening_context.json"
ROUTES = frozenset({"oral", "intravenous", "subcutaneous", "intramuscular", "inhaled", "topical", "other"})


@dataclass(frozen=True)
class Exposure:
    route: str | None = None
    systemic: bool | None = None


def _unique(pairs: list[tuple[str, object]]) -> dict:
    fields = dict(pairs)
    if len(fields) != len(pairs):
        raise ValueError("Duplicate field in screening context")
    return fields


def _read(directory: Path) -> dict[str, Exposure]:
    try:
        raw = ligand_inputs.read_regular(directory / MANIFEST, 64 * 1024, "screening_context", ValueError)
    except ValueError as exc:
        if str(exc) == "screening_context_missing":
            return {}
        raise
    doc = json.loads(raw.decode("utf-8-sig"), object_pairs_hook=_unique)
    if not isinstance(doc, dict) or doc.keys() != {"candidates"}:
        raise ValueError("Expected a candidates object")
    entries = doc["candidates"]
    if not isinstance(entries, dict) or not 1 <= len(entries) <= ligand_admet.MAX_CANDIDATES:
        raise ValueError("Invalid candidate list")
    result = {}
    for name, entry in entries.items():
        if not ligand_admet.valid_candidate_id(name) or not isinstance(entry, dict) or not entry:
            raise ValueError("Invalid candidate context")
        if entry.keys() - {"route", "systemic"}:
            raise ValueError("Unknown exposure field")
        if "route" in entry and (not isinstance(entry["route"], str) or entry["route"] not in ROUTES):
            raise ValueError("Unknown administration route")
        if "systemic" in entry and type(entry["systemic"]) is not bool:
            raise ValueError("Expected a boolean systemic exposure declaration")
        result[name] = Exposure(entry.get("route"), entry.get("systemic"))
    return result


def for_candidates(directory: Path, names: Iterable[str]) -> tuple[list[Exposure], str | None]:
    """Match intent by candidate ID. Keep incomplete or conflicting scopes manual."""
    try:
        context = _read(directory)
    except (ValueError, RecursionError):
        return [], f"Review the unreadable or invalid {MANIFEST} before preselecting an exposure-dependent screen."
    try:
        systemic = cypstrate.declared_exposure(directory)
    except cypstrate.CypstrateError:
        # An absent or unusable metabolism hypothesis supplies no exposure declaration.
        systemic = {}
    facts = []
    for name in names:
        item = context.get(name, Exposure())
        declared = systemic.get(name)
        if declared is not None and item.systemic is not None and declared != item.systemic:
            return [], f"Resolve conflicting systemic exposure declarations in {MANIFEST} and {cypstrate.MANIFEST}."
        facts.append(Exposure(item.route, declared if item.systemic is None else item.systemic))
    return facts, None
