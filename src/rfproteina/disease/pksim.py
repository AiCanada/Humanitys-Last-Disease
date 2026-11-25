"""Run an attributed individual human PK-Sim scenario and report concentration-time profiles."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from rfproteina.disease import ligand_admet, ligand_alerts, ligand_inputs, pksim_curves, pksim_runtime
from rfproteina.disease.corpus_provenance import declare

METHOD_ID = "pksim"
ENDPOINT = "whole_body_pbpk_exposure"
MANIFEST = "pksim.json"
_SHA256 = re.compile(r"[a-f0-9]{64}")
_MAX_CASES = 8
_MAX_OUTPUTS = 8
_MAX_VALUES_PER_CASE = 8192
_TIME_MINUTES = {"s": 1 / 60, "min": 1.0, "h": 60.0, "d": 1440.0, "day": 1440.0}


class PKSimError(ValueError):
    """A documented issue with a model nomination or its attribution."""


def _unique(items: list[tuple[str, object]]) -> dict:
    fields = dict(items)
    if len(fields) != len(items):
        raise ValueError("Duplicate key")
    return fields


def _label(value: object, limit: int = 160) -> bool:
    return isinstance(value, str) and 0 < len(value) <= limit and value == value.strip() and value.isprintable()


def _nominations(directory: Path, candidate: str | None) -> dict:
    try:
        raw = ligand_inputs.read_regular(directory / MANIFEST, 64 * 1024, "pksim_manifest", ValueError)
        doc = json.loads(raw.decode("utf-8-sig"), object_pairs_hook=_unique)
    except (ValueError, RecursionError) as error:
        if str(error) == "pksim_manifest_missing":
            raise PKSimError("pksim_manifest_missing") from error
        raise PKSimError("pksim_invalid_manifest") from error
    if not isinstance(doc, dict) or doc.keys() != {"samples"} or not isinstance(doc["samples"], dict):
        raise PKSimError("pksim_invalid_manifest")
    samples = doc["samples"]
    if not samples or any(not ligand_admet.valid_candidate_id(name) for name in samples):
        raise PKSimError("pksim_invalid_manifest")
    if len(samples) > _MAX_CASES:
        raise PKSimError("pksim_budget_exceeded")
    if candidate is not None:
        if candidate not in samples:
            raise PKSimError("pksim_candidate_not_nominated")
        return {candidate: samples[candidate]}
    return samples


@dataclass(frozen=True)
class Scenario:
    model_sha256: str
    simulation: str
    compound: str
    paths: tuple[str, ...]

    @classmethod
    def from_record(cls, entry: object) -> Scenario:
        if not isinstance(entry, dict) or entry.keys() != {"model_sha256", "simulation", "compound", "outputs"}:
            raise PKSimError("pksim_invalid_manifest")
        digest, name, compound, outputs = (entry[k] for k in ("model_sha256", "simulation", "compound", "outputs"))
        if (
            not isinstance(digest, str)
            or not _SHA256.fullmatch(digest)
            or not pksim_runtime.valid_simulation_name(name)
            or not _label(compound)
        ):
            raise PKSimError("pksim_invalid_manifest")
        if not isinstance(outputs, list) or not outputs or not all(_label(p, 512) for p in outputs):
            raise PKSimError("pksim_invalid_manifest")
        if len(outputs) > _MAX_OUTPUTS:
            raise PKSimError("pksim_budget_exceeded")
        if len(set(outputs)) != len(outputs) or any(
            not p.startswith("Organism|") or any(not _label(part) for part in p.split("|")) for p in outputs
        ):
            raise PKSimError("pksim_invalid_manifest")
        return cls(digest, name, compound, tuple(outputs))


def _named(items: object, name: str) -> dict:
    if not isinstance(items, list):
        raise PKSimError("pksim_invalid_model")
    matches = [item for item in items if isinstance(item, dict) and item.get("Name") == name]
    if len(matches) != 1:
        raise PKSimError("pksim_invalid_model")
    return matches[0]


def _output_window(intervals: object) -> tuple[float, float] | None:
    """Read explicit interval boundaries. Older snapshots may use model defaults."""
    if not intervals:
        return None
    if not isinstance(intervals, list):
        raise PKSimError("pksim_invalid_model")
    bounds = []
    for interval in intervals:
        params = interval.get("Parameters") if isinstance(interval, dict) else None
        if not isinstance(params, list):
            return None
        times = {}
        for p in params:
            if not isinstance(p, dict) or p.get("Name") not in ("Start time", "End time"):
                continue
            value, unit = p.get("Value"), p.get("Unit")
            if type(value) not in (int, float) or not isinstance(unit, str) or unit not in _TIME_MINUTES:
                return None
            if p["Name"] in times or not 0 <= value < 1e15:
                raise PKSimError("pksim_invalid_model")
            times[p["Name"]] = value * _TIME_MINUTES[unit]
        if len(times) != 2:
            return None
        if times["End time"] <= times["Start time"]:
            raise PKSimError("pksim_invalid_model")
        bounds.append((times["Start time"], times["End time"]))
    return min(start for start, _ in bounds), max(end for _, end in bounds)


def _context(raw: bytes, case: Scenario) -> tuple[dict, dict[str, str]]:
    """Recover provenance from the same snapshot bytes that the simulator will consume."""
    if hashlib.sha256(raw).hexdigest() != case.model_sha256:
        raise PKSimError("pksim_model_mismatch")
    try:
        model = pksim_runtime.parse_snapshot(raw)
    except pksim_runtime.PKSimError as exc:
        raise PKSimError("pksim_invalid_model") from exc
    sim = _named(model.get("Simulations"), case.simulation)
    person = sim.get("Individual")
    if not _label(person) or sim.get("Population"):
        raise PKSimError("pksim_invalid_model")
    origin = _named(model.get("Individuals"), person).get("OriginData")
    if not isinstance(origin, dict) or origin.get("Species") != "Human":
        raise PKSimError("pksim_invalid_model")
    primary = _named(model.get("Compounds"), case.compound)
    if primary.get("IsSmallMolecule") is not True:
        raise PKSimError("pksim_invalid_model")
    participants = sim.get("Compounds")
    selected = _named(participants, case.compound)
    reg = selected.get("Protocol")
    if not isinstance(reg, dict) or not _label(reg.get("Name")):
        raise PKSimError("pksim_invalid_model")
    protocol = _named(model.get("Protocols"), reg["Name"])
    names = []
    for item in participants:
        if not isinstance(item, dict) or not _label(item.get("Name")) or item["Name"] in names:
            raise PKSimError("pksim_invalid_model")
        names.append(item["Name"])
        if item is not selected and item.get("Protocol"):
            raise PKSimError("pksim_invalid_model")
    outputs = sim.get("OutputSelections")
    if not isinstance(outputs, list):
        raise PKSimError("pksim_invalid_model")
    analytes = {}
    for path in case.paths:
        parts = path.split("|")
        hits = [name for name in names if name in parts[2:-1]]
        if outputs.count(path) != 1 or len(hits) != 1:
            raise PKSimError("pksim_invalid_model")
        analytes[path] = hits[0]
    context = {"species": "Human", "individual": person, "protocol": reg["Name"]}
    if window := _output_window(sim.get("OutputSchema")):
        context["configured_window_min"] = list(window)
    if _label(protocol.get("ApplicationType")):
        context["protocol_application_type"] = protocol["ApplicationType"]
    pksim_runtime.prepare_individual_project(model, case.simulation, case.compound)
    return context, analytes


def _candidate_key(rows: list[tuple[str, ...]]) -> str:
    from rdkit import Chem, rdBase

    if len(rows) != 1 or len(rows[0]) != 2 or len(rows[0][1]) > 4096:
        raise PKSimError("pksim_molecule_mismatch")
    try:
        with rdBase.BlockLogs():
            mol = ligand_alerts.parse_molecule(rows[0][1], nominated=True)
            key = Chem.MolToInchiKey(mol)
    except (ligand_alerts.LigandUnreadableError, ValueError, RuntimeError, OverflowError) as exc:
        raise PKSimError("pksim_molecule_mismatch") from exc
    if not pksim_runtime.STANDARD_INCHIKEY.fullmatch(key):
        raise PKSimError("pksim_molecule_mismatch")
    return key


def _cases(directory: Path, candidate: str | None, *, simulate: bool) -> Iterator[dict]:
    samples = _nominations(directory, candidate)
    source, rows, problem = ligand_inputs.bounded_smiles_candidates(directory, max_bytes=1024 * 1024, max_rows=16384)
    by_name = {name: [] for name in samples}
    for row in rows:
        if row and row[0] in by_name:
            by_name[row[0]].append(row)
    for name, entry in samples.items():
        record = {"candidate_id": name, "method": METHOD_ID, "status": "untested"}
        try:
            case = Scenario.from_record(entry)
            if problem or (source == "run_ligand" and (candidate is not None or set(samples) != {"run_ligand"})):
                raise PKSimError("pksim_molecule_mismatch")
            key = _candidate_key(by_name[name])
            snapshot, registered_key = pksim_runtime.load_model(case.model_sha256, case.compound)
            if key != registered_key:
                raise PKSimError("pksim_molecule_mismatch")
            context, analytes = _context(snapshot, case)
            record.update(
                model_sha256=case.model_sha256,
                simulation=case.simulation,
                compound=case.compound,
                compound_inchikey=key,
                **context,
            )
            if simulate:
                csv = pksim_runtime.run(snapshot, case.simulation)
                curves = pksim_curves.extract(csv, case.paths)
                if sum(curve["point_count"] for curve in curves) > _MAX_VALUES_PER_CASE:
                    raise PKSimError("pksim_budget_exceeded")
                for curve in curves:
                    path = curve["output_path"]
                    if window := context.get("configured_window_min"):
                        factor = _TIME_MINUTES[curve["time_unit"]]
                        actual = (curve["auc_start"] * factor, curve["auc_end"] * factor)
                        if not all(
                            math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-8) for a, b in zip(actual, window, strict=True)
                        ):
                            raise PKSimError("pksim_invalid_export")
                    curve.update(method=METHOD_ID, analyte=analytes[path], organ_or_blood_pool=path.split("|")[1])
                record.update(status="simulated", curves=curves, export_sha256=hashlib.sha256(csv).hexdigest())
            else:
                record["status"] = "ready"
        except (PKSimError, pksim_runtime.PKSimError, pksim_curves.CurveError) as exc:
            record["blocker"] = str(exc)
        except (ImportError, OSError):
            record["blocker"] = "pksim_runtime_unavailable"
        yield record


def readiness(directory: Path, candidate: str | None = None) -> tuple[bool, bool, str]:
    """Check nomination, registered chemical identity and model scope without starting the solver."""
    try:
        cases = list(_cases(directory, candidate, simulate=False))
    except PKSimError as exc:
        return False, False, f"No eligible PK-Sim nomination for this selection ({exc})."
    ready = sum(case["status"] == "ready" for case in cases)
    if ready == len(cases):
        if ready == 1:
            return True, True, "One registered human PK-Sim scenario matches the nominated compound."
        return True, False, f"{ready} registered human PK-Sim scenarios require separate solver runs."
    reasons = ", ".join(sorted({case["blocker"] for case in cases if "blocker" in case}))
    return bool(ready), False, f"{ready} of {len(cases)} scenarios ready ({reasons})."


def measure(directory: Path, candidate: str | None = None) -> dict:
    report = {
        "method": METHOD_ID,
        "endpoint": ENDPOINT,
        "scope": "candidate" if candidate is not None else "run",
        "status": "untested",
        "attempted": 0,
        "simulated": 0,
        "samples": [],
        "provider_version": pksim_runtime.VERSION,
        "corpus_provenance": declare(METHOD_ID, against=ENDPOINT),
        "interpretation": "Simulated exposure for the specified model and regimen. "
        "Organ injury and agreement with measured concentrations require separate evidence.",
    }
    try:
        samples = list(_cases(directory, candidate, simulate=True))
    except PKSimError as exc:
        return {**report, "blocker": str(exc)}
    done = sum(row["status"] == "simulated" for row in samples)
    report.update(samples=samples, attempted=len(samples), simulated=done)
    if done:
        report["status"] = "simulated" if done == len(samples) else "partial"
    else:
        blockers = {row["blocker"] for row in samples}
        if len(blockers) == 1:
            report["blocker"] = blockers.pop()
    return report
