"""Run structure-conditioned PROPKA calculations on nominated PDB coordinates."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, distribution
from importlib.util import find_spec
from math import isfinite
from pathlib import Path

from rfproteina.disease import corpus_provenance, ligand_admet, ligand_inputs

METHOD_ID = "propka"
METHOD = "propka_empirical_structure_pka_3_5_1"
ENDPOINT = "structure_conditioned_group_pka"
MANIFEST = "propka.json"
RELEASE = "3.5.1"
PARAMETER_SHA256 = "354298a0fb5d1e0d0c7cf3ea9dea117875cd6acbcf72e2dda955578a0417d2e0"
BONDS_SHA256 = "f0b03022a3840fc6135b2e366904eb0b1518aee976d6da581fcdd4bb9202338d"
_MAX_BYTES = 8 * 1024 * 1024
_MAX_ATOMS = 40_000
_MAX_GROUPS = 2048
_FILENAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,119}\.pdb", re.IGNORECASE)
_DEVICE = re.compile(r"(?:CON|PRN|AUX|NUL|COM[0-9]|LPT[0-9])(?:\.|$)", re.IGNORECASE)
_HASH = re.compile(r"[a-f0-9]{64}")
_CHAIN = re.compile(r"[A-Za-z0-9]?")
_LABEL = re.compile(r"[A-Za-z0-9_+'*-]{1,16}")
_IDENTITY = ("kind", "chain", "residue_number", "insertion_code", "residue_name", "atom_name")
_GROUP_FIELDS = {*_IDENTITY, "group_type", "pka", "model_pka", "coupled"}
BLOCKERS = frozenset({
    "propka_manifest_missing", "invalid_propka_manifest", "propka_candidate_mismatch",
    "propka_structure_unreadable", "propka_structure_invalid", "propka_structure_out_of_scope",
    "propka_gemmi_unavailable", "propka_runtime_unavailable", "propka_timeout",
    "propka_provider_failure", "propka_invalid_report", "propka_no_usable_groups",
})


class PropkaError(ValueError):
    """A named scope, input, runtime, or result problem."""


@dataclass(frozen=True)
class Plan:
    data: bytes
    sha256: str
    atoms: frozenset[tuple]
    candidate_id: str | None
    assay_ph: float | None
    incomplete_termini: bool


def _unique(pairs: list[tuple[str, object]]) -> dict:
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("Duplicate JSON field.")
    return result


def _ph(value: object) -> bool:
    return type(value) in (int, float) and 0 <= value <= 14


def _settings(directory: Path) -> dict:
    try:
        raw = ligand_inputs.read_regular(directory / MANIFEST, 4096, "propka_manifest", PropkaError)
        settings = json.loads(raw.decode("utf-8-sig"), object_pairs_hook=_unique)
        if not isinstance(settings, dict) or not (
            {"version", "structure"} <= settings.keys() <= {"version", "structure", "candidate_id", "assay_ph"}
            and type(settings["version"]) is int and settings["version"] == 1
            and isinstance(settings["structure"], str) and _FILENAME.fullmatch(settings["structure"])
            and not _DEVICE.match(settings["structure"])
        ):
            raise ValueError("Invalid structure nomination.")
        if "candidate_id" in settings and not (
            isinstance(settings["candidate_id"], str) and ligand_admet.valid_candidate_id(settings["candidate_id"])
        ):
            raise ValueError("Invalid candidate.")
        if "assay_ph" in settings and not _ph(settings["assay_ph"]):
            raise ValueError("Invalid assay pH.")
    except (ValueError, RecursionError) as error:
        if isinstance(error, PropkaError) and str(error) == "propka_manifest_missing":
            raise
        raise PropkaError("invalid_propka_manifest") from error
    return settings


def installed() -> bool:
    try:
        package = distribution("propka")
        if package.version != RELEASE or find_spec("gemmi") is None or find_spec("propka") is None:
            return False
        for name, expected in (("propka.cfg", PARAMETER_SHA256), ("protein_bonds.json", BONDS_SHA256)):
            path = Path(package.locate_file(f"propka/{name}"))
            data = ligand_inputs.read_regular(path, 256 * 1024, "propka_parameters", PropkaError)
            if sha256(data).hexdigest() != expected:
                return False
    except (PackageNotFoundError, OSError, ImportError, ValueError):
        return False
    return True


def _validate_termini(text: str, atom_count: int) -> bool:
    """Check chain transitions against the fixed columns PROPKA actually reads."""
    positions = {}
    seen_chains = set()
    terminal_oxygens = set()
    first_positions = {}
    terminal_nitrogens = set()
    terminal_atom_names = ("OXT", "O''")
    previous_chain = None
    previous_number = None
    native_boundary = True
    coordinates = 0
    ended = False
    for line in text.splitlines():
        tag = line[:6]
        if tag.rstrip() == "END":
            ended = True
        if ended and tag.rstrip() in {"ATOM", "HETATM", "TER", "MODEL", "ENDMDL"}:
            raise PropkaError("propka_structure_invalid")
        if tag == "TER   ":
            native_boundary = True
        if tag not in ("ATOM  ", "HETATM"):
            continue
        if len(line) < 60 or any(not line[start:stop].strip() for start, stop in (
            (30, 38), (38, 46), (46, 54), (54, 60),
        )):
            raise PropkaError("propka_structure_invalid")
        coordinates += 1
        if tag != "ATOM  ":
            continue
        chain, number, atom = line[21], line[22:26], line[12:16].strip()
        if chain != previous_chain:
            if chain in seen_chains or not native_boundary or number == previous_number:
                raise PropkaError("propka_structure_out_of_scope")
            seen_chains.add(chain)
            first_positions[chain] = number
        elif native_boundary and number != previous_number:
            raise PropkaError("propka_structure_out_of_scope")
        positions[chain] = number
        if atom == "N" and number == first_positions[chain]:
            terminal_nitrogens.add(chain)
        if atom in terminal_atom_names:
            terminal_oxygens.add((chain, number))
            native_boundary = True
        elif chain != previous_chain:
            native_boundary = False
        previous_chain, previous_number = chain, number
    if coordinates != atom_count:
        raise PropkaError("propka_structure_invalid")
    if not terminal_oxygens <= set(positions.items()):
        raise PropkaError("propka_structure_out_of_scope")
    return set(positions.items()) != terminal_oxygens or seen_chains != terminal_nitrogens


def _inspect(data: bytes) -> tuple[frozenset[tuple], bool]:
    try:
        import gemmi
    except (ImportError, OSError) as error:
        raise PropkaError("propka_gemmi_unavailable") from error
    try:
        text = data.decode("ascii")
        structure = gemmi.read_pdb_string(text)
    except (UnicodeError, ValueError, RuntimeError) as error:
        raise PropkaError("propka_structure_invalid") from error
    if len(structure) != 1:
        raise PropkaError("propka_structure_out_of_scope")
    identities = set()
    positions = set()
    amino_acids = set()
    for chain in structure[0]:
        for residue in chain:
            number, insertion = residue.seqid.num, residue.seqid.icode.strip()
            chain_id = chain.name.strip()
            if (
                not _CHAIN.fullmatch(chain_id) or type(number) is not int or not -999 <= number <= 9999
                or insertion or not _LABEL.fullmatch(residue.name)
            ):
                raise PropkaError("propka_structure_out_of_scope")
            kind = "protein" if residue.het_flag == "A" else "ligand"
            if (chain_id, number) in positions:
                raise PropkaError("propka_structure_out_of_scope")
            positions.add((chain_id, number))
            position = (kind, chain_id, number, insertion, residue.name)
            if kind == "protein" and gemmi.find_tabulated_residue(residue.name).is_amino_acid():
                amino_acids.add(position)
            for atom in residue:
                if atom.altloc not in ("", " ", "\0"):
                    raise PropkaError("propka_structure_out_of_scope")
                if not all(isfinite(value) for value in (atom.pos.x, atom.pos.y, atom.pos.z, atom.occ)):
                    raise PropkaError("propka_structure_invalid")
                if atom.element.name != "H" and not 0 < atom.occ <= 1:
                    raise PropkaError("propka_structure_invalid")
                key = (*position, atom.name)
                if len(identities) >= _MAX_ATOMS or key in identities or not _LABEL.fullmatch(atom.name):
                    raise PropkaError("propka_structure_out_of_scope")
                identities.add(key)
    if len(amino_acids) < 10:
        raise PropkaError("propka_structure_out_of_scope")
    return frozenset(identities), _validate_termini(text, len(identities))


def prepare(directory: Path, candidate: str | None = None) -> Plan:
    settings = _settings(directory)
    nominated = settings.get("candidate_id")
    if candidate is not None and candidate != nominated:
        raise PropkaError("propka_candidate_mismatch")
    try:
        data = ligand_inputs.read_regular(
            directory / settings["structure"], _MAX_BYTES, "propka_structure", PropkaError,
        )
    except PropkaError as error:
        raise PropkaError("propka_structure_unreadable") from error
    ph = float(settings["assay_ph"]) if "assay_ph" in settings else None
    atoms, incomplete_termini = _inspect(data)
    return Plan(data, sha256(data).hexdigest(), atoms, nominated, ph, incomplete_termini)


def _predict(plan: Plan) -> dict:
    if not installed():
        raise PropkaError("propka_runtime_unavailable")
    worker = Path(__file__).with_name("propka_worker.py").resolve()
    try:
        process = subprocess.run(
            [sys.executable, "-I", "-B", str(worker)], input=plan.data, capture_output=True, check=True, timeout=90,
        )
    except subprocess.TimeoutExpired as error:
        raise PropkaError("propka_timeout") from error
    except (OSError, subprocess.CalledProcessError) as error:
        raise PropkaError("propka_provider_failure") from error
    try:
        if len(process.stdout) > 1024 * 1024:
            raise ValueError("Overlong result.")
        return json.loads(process.stdout, object_pairs_hook=_unique)
    except (ValueError, RecursionError) as error:
        raise PropkaError("propka_invalid_report") from error


def _value(value: object) -> bool:
    return type(value) in (int, float) and -100 < value < 100 and value != 99.99


def _groups(value: object, atoms: frozenset[tuple] | None = None) -> list[dict]:
    if not isinstance(value, list) or len(value) > _MAX_GROUPS:
        raise PropkaError("propka_invalid_report")
    groups = {}
    for group in value:
        if not isinstance(group, dict) or group.keys() != _GROUP_FIELDS or not (
            group["kind"] in ("protein", "ligand") and isinstance(group["chain"], str)
            and _CHAIN.fullmatch(group["chain"]) and type(group["residue_number"]) is int
            and -999 <= group["residue_number"] <= 9999 and group["insertion_code"] == ""
            and all(isinstance(group[key], str) and _LABEL.fullmatch(group[key]) for key in (
                "residue_name", "atom_name", "group_type",
            ))
            and type(group["coupled"]) is bool
            and (
                (group["pka"] is None and group["model_pka"] is None)
                or (_value(group["pka"]) and _value(group["model_pka"]))
            )
        ):
            raise PropkaError("propka_invalid_report")
        key = tuple(group[name] for name in _IDENTITY)
        typed = (*key, group["group_type"])
        if (atoms is not None and key not in atoms) or typed in groups:
            raise PropkaError("propka_invalid_report")
        groups[typed] = {**group}
    return [groups[key] for key in sorted(groups)]


def _base() -> dict:
    return {
        "method": METHOD, "endpoint": ENDPOINT, "release": RELEASE, "parameter_sha256": PARAMETER_SHA256,
        "protein_bonds_sha256": BONDS_SHA256, "scope": "structure", "status": "untested", "groups": [],
        "corpus_provenance": corpus_provenance.declare(METHOD, against=ENDPOINT),
    }


def _with_groups(report: dict, groups: list[dict], incomplete: bool) -> dict:
    measured = sum(row["pka"] is not None for row in groups)
    if not measured:
        return {**report, "groups": groups, "incomplete_input": incomplete, "blocker": "propka_no_usable_groups"}
    status = "partial" if incomplete or measured != len(groups) else "complete"
    return {**report, "groups": groups, "status": status, "incomplete_input": incomplete}


def screen_run(directory: Path, candidate: str | None = None) -> dict:
    report = _base()
    try:
        plan = prepare(directory, candidate)
        report.update(structure_sha256=plan.sha256)
        if plan.candidate_id is not None:
            report["candidate_id"] = plan.candidate_id
        if plan.assay_ph is not None:
            report["assay_ph"] = plan.assay_ph
        output = _predict(plan)
        if not isinstance(output, dict) or output.keys() != {"groups", "incomplete_input"} or (
            type(output["incomplete_input"]) is not bool
        ):
            raise PropkaError("propka_invalid_report")
        return _with_groups(
            report, _groups(output["groups"], plan.atoms), output["incomplete_input"] or plan.incomplete_termini,
        )
    except PropkaError as error:
        return {**report, "blocker": str(error)}


def report_view(raw: object) -> dict:
    """Validate saved PROPKA evidence before displaying it in shared views."""
    report = _base()
    invalid = {**report, "blocker": "propka_invalid_report"}
    if not isinstance(raw, Mapping) or any(raw.get(key) != report[key] for key in (
        "method", "endpoint", "release", "parameter_sha256", "protein_bonds_sha256", "scope",
    )):
        return invalid
    digest, candidate, ph, blocker = (raw.get(key) for key in (
        "structure_sha256", "candidate_id", "assay_ph", "blocker",
    ))
    if digest is not None:
        if not isinstance(digest, str) or not _HASH.fullmatch(digest):
            return invalid
        report["structure_sha256"] = digest
    if candidate is not None:
        if digest is None or not isinstance(candidate, str) or not ligand_admet.valid_candidate_id(candidate):
            return invalid
        report["candidate_id"] = candidate
    if ph is not None:
        if digest is None or not _ph(ph):
            return invalid
        report["assay_ph"] = ph
    if raw.get("status") == "untested" and raw.get("groups") == [] and isinstance(blocker, str) and blocker in BLOCKERS:
        return {**report, "blocker": blocker}
    if digest is None or type(raw.get("incomplete_input")) is not bool:
        return invalid
    try:
        view = _with_groups(report, _groups(raw.get("groups")), raw["incomplete_input"])
    except PropkaError:
        return invalid
    if view.get("blocker") != blocker or view["status"] != raw.get("status"):
        return invalid
    return view
