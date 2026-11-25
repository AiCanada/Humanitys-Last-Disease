"""Local, structure-attributed pocket predictions for a nominated receptor."""

from __future__ import annotations

import csv
import io
import json
import re
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from hashlib import sha256
from importlib.util import find_spec
from math import isfinite
from pathlib import Path

from rfproteina.disease import ligand_inputs, p2rank_runtime
from rfproteina.disease.corpus_provenance import declare

METHOD_ID = "p2rank"
METHOD = "p2rank_pocket_rank"
ENDPOINT = "receptor_ligand_binding_pockets"
MANIFEST = "p2rank.json"
MODELS = {profile: f"p2rank_{profile}_2_5_1" for profile in ("default", "alphafold")}
_MAX_RECEPTOR_BYTES = 8 * 1024 * 1024
_MAX_OUTPUT_BYTES = 4 * 1024 * 1024
_MAX_ATOMS = 40_000
_MAX_POCKETS = 256
_FILENAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,119}\.(?:pdb|cif|mmcif)", re.IGNORECASE)
_RESIDUE_ID = re.compile(r"[A-Za-z0-9]{1,12}_-?[0-9]{1,8}[A-Za-z]?")
_HASH = re.compile(r"[a-f0-9]{64}")
_CSV_FIELDS = (
    "name", "rank", "score", "probability", "sas_points", "surf_atoms",
    "center_x", "center_y", "center_z", "residue_ids", "surf_atom_ids",
)
_SOURCES = {"xray": "xray", "alphafold": "predicted", "predicted": "predicted", "cryoem": "cryoem", "nmr": "nmr"}
BLOCKERS = frozenset({
    "p2rank_manifest_missing", "invalid_p2rank_manifest", "p2rank_receptor_unreadable",
    "p2rank_receptor_invalid", "p2rank_receptor_out_of_scope", "p2rank_structure_source_mismatch",
    "p2rank_xray_b_factors_missing", "p2rank_gemmi_unavailable", "p2rank_invalid_output",
    "p2rank_runtime_unavailable", "p2rank_runtime_integrity", "p2rank_timeout",
    "p2rank_provider_failure", "p2rank_output_unreadable",
})


class P2RankError(ValueError):
    """A stable reason for withholding a pocket result."""


@dataclass(frozen=True)
class Receptor:
    data: bytes
    suffix: str
    structure_source: str
    profile: str
    sha256: str
    residue_ids: frozenset[str]


def _unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("Duplicate JSON field.")
    return result


def _manifest(directory: Path) -> tuple[str, str]:
    try:
        raw = ligand_inputs.read_regular(directory / MANIFEST, 4096, "p2rank_manifest", P2RankError)
        settings = json.loads(raw.decode("utf-8-sig"), object_pairs_hook=_unique)
        if not (
            isinstance(settings, dict) and settings.keys() == {"version", "receptor", "structure_source"}
            and type(settings["version"]) is int and settings["version"] == 1
            and isinstance(settings["receptor"], str) and _FILENAME.fullmatch(settings["receptor"])
            and isinstance(settings["structure_source"], str) and settings["structure_source"] in _SOURCES
        ):
            raise ValueError("Invalid nomination.")
    except (ValueError, RecursionError) as error:
        if isinstance(error, P2RankError) and str(error) == "p2rank_manifest_missing":
            raise
        raise P2RankError("invalid_p2rank_manifest") from error
    return settings["receptor"], settings["structure_source"]


def _profile(source: str) -> str:
    return "default" if source == "xray" else "alphafold"


def _model_evidence(profile: str) -> dict:
    model = MODELS[profile]
    return {"model": model, "corpus_provenance": declare(model, against=ENDPOINT)}


def _method(text: str) -> str:
    value = text.upper().replace("-", " ")
    if "X RAY" in value:
        return "xray"
    if "ELECTRON MICROSCOPY" in value or "CRYO EM" in value:
        return "cryoem"
    if "NMR" in value:
        return "nmr"
    if any(term in value for term in ("COMPUTATIONAL", "THEORETICAL", "PREDICTED")):
        return "predicted"
    return "unknown"


def _inspect(data: bytes, suffix: str, source: str) -> frozenset[str]:
    try:
        import gemmi
    except (ImportError, OSError) as error:
        raise P2RankError("p2rank_gemmi_unavailable") from error
    try:
        if suffix == ".pdb":
            text = data.decode("ascii")
            methods = [item.strip() for line in text.splitlines() if line.startswith("EXPDTA")
                       for item in line[10:].split(";") if item.strip()]
            structure = gemmi.read_pdb_string(text)
        else:
            block = gemmi.cif.read_string(data.decode("utf-8-sig")).sole_block()
            methods = [gemmi.cif.as_string(item) for item in block.find_values("_exptl.method")
                       if item not in ("?", ".")]
            if len(block.find_values("_ma_model_list.model_id")):
                methods.append("COMPUTATIONAL MODEL")
            structure = gemmi.make_structure_from_block(block)
    except (UnicodeError, ValueError, RuntimeError) as error:
        raise P2RankError("p2rank_receptor_invalid") from error
    if methods and {_method(value) for value in methods} != {_SOURCES[source]}:
        raise P2RankError("p2rank_structure_source_mismatch")
    if len(structure) != 1:
        raise P2RankError("p2rank_receptor_out_of_scope")

    residue_ids = set()
    has_b_factors = False
    count = 0
    for chain in structure[0]:
        for residue in chain:
            count += len(residue)
            if count > _MAX_ATOMS:
                raise P2RankError("p2rank_receptor_out_of_scope")
            if not all(
                isfinite(value) for atom in residue for value in (atom.pos.x, atom.pos.y, atom.pos.z, atom.b_iso)
            ):
                raise P2RankError("p2rank_receptor_invalid")
            if not gemmi.find_tabulated_residue(residue.name).is_amino_acid() or not len(residue):
                continue
            identifier = f"{chain.name}_{residue.seqid.num}{residue.seqid.icode.strip()}"
            if not _RESIDUE_ID.fullmatch(identifier) or identifier in residue_ids:
                raise P2RankError("p2rank_receptor_out_of_scope")
            residue_ids.add(identifier)
            has_b_factors |= any(atom.b_iso > 0 and not atom.element.is_hydrogen for atom in residue)
    if len(residue_ids) < 20:
        raise P2RankError("p2rank_receptor_out_of_scope")
    if source == "xray" and not has_b_factors:
        raise P2RankError("p2rank_xray_b_factors_missing")
    return frozenset(residue_ids)


def prepare(directory: Path) -> Receptor:
    name, source = _manifest(directory)
    suffix = ".pdb" if name.lower().endswith(".pdb") else ".cif"
    try:
        data = ligand_inputs.read_regular(directory / name, _MAX_RECEPTOR_BYTES, "p2rank_receptor", P2RankError)
    except P2RankError as error:
        raise P2RankError("p2rank_receptor_unreadable") from error
    return Receptor(data, suffix, source, _profile(source), sha256(data).hexdigest(), _inspect(data, suffix, source))


def installed() -> bool:
    return find_spec("gemmi") is not None and p2rank_runtime.installed()


def _number(value: object, low: float, high: float) -> bool:
    return type(value) in (float, int) and low <= value <= high


def _pocket_valid(pocket: object, expected_rank: int) -> bool:
    if not isinstance(pocket, dict) or pocket.keys() != {
        "rank", "score", "probability", "center_angstrom", "sas_points", "surface_atoms", "residue_ids",
    }:
        return False
    center, residues = pocket["center_angstrom"], pocket["residue_ids"]
    return (
        type(pocket["rank"]) is int and pocket["rank"] == expected_rank
        and _number(pocket["score"], 0, 1e9) and _number(pocket["probability"], 0, 1)
        and type(pocket["sas_points"]) is int and 0 <= pocket["sas_points"] <= 10_000_000
        and type(pocket["surface_atoms"]) is int and 0 <= pocket["surface_atoms"] <= _MAX_ATOMS
        and isinstance(center, list) and len(center) == 3 and all(_number(value, -1e6, 1e6) for value in center)
        and isinstance(residues, list) and 0 < len(residues) <= _MAX_ATOMS
        and all(isinstance(token, str) and _RESIDUE_ID.fullmatch(token) for token in residues)
        and len(set(residues)) == len(residues)
    )


def parse_predictions(data: bytes, residue_ids: Collection[str]) -> list[dict]:
    """Parse the pinned 2.5.1 CSV and require residue IDs from the nominated receptor."""
    if len(data) > _MAX_OUTPUT_BYTES:
        raise P2RankError("p2rank_invalid_output")
    pockets = []
    try:
        rows = csv.reader(io.StringIO(data.decode("utf-8-sig"), newline=""), strict=True)
        header = next(rows, ())
        if tuple(value.strip() for value in header) != _CSV_FIELDS:
            raise ValueError("Unexpected P2Rank header.")
        allowed = set(residue_ids)
        for row in rows:
            if not row:
                continue
            rank = len(pockets) + 1
            if len(row) != len(_CSV_FIELDS) or rank > _MAX_POCKETS:
                raise ValueError("Malformed P2Rank row.")
            name, number, score, probability, points, atoms, x, y, z, ids, atom_ids = map(str.strip, row)
            surface_count = int(atoms)
            surface_ids = atom_ids.split()
            ids = ids.split()
            pocket = {
                "rank": int(number), "score": float(score), "probability": float(probability),
                "sas_points": int(points), "surface_atoms": surface_count,
                "center_angstrom": [float(x), float(y), float(z)], "residue_ids": ids,
            }
            if (
                name != f"pocket{rank}" or not _pocket_valid(pocket, rank)
                or (pockets and pocket["score"] > pockets[-1]["score"])
                or not allowed.issuperset(ids) or len(surface_ids) != surface_count
                or len(set(surface_ids)) != surface_count
                or any(not token.isascii() or not token.isdecimal() for token in surface_ids)
            ):
                raise ValueError("Inconsistent P2Rank pocket.")
            pockets.append(pocket)
    except (ValueError, UnicodeError, csv.Error, OverflowError) as error:
        raise P2RankError("p2rank_invalid_output") from error
    return pockets


def _predict(receptor: Receptor) -> bytes:
    return p2rank_runtime.predict(receptor.data, receptor.suffix, receptor.profile)


def _record() -> dict:
    return {
        "method": METHOD, "endpoint": ENDPOINT, "release": p2rank_runtime.RELEASE,
        "archive_sha256": p2rank_runtime.ARCHIVE_SHA256, "scope": "receptor", "status": "untested", "pockets": [],
    }


def screen_run(directory: Path, candidate: str | None = None) -> dict:
    """Report the same nominated receptor for each ligand in this run."""
    report = _record()
    try:
        receptor = prepare(directory)
    except P2RankError as error:
        return {**report, "blocker": str(error)}
    report.update(
        receptor_sha256=receptor.sha256, structure_source=receptor.structure_source, profile=receptor.profile,
        receptor_format=receptor.suffix[1:], residue_count=len(receptor.residue_ids),
        **_model_evidence(receptor.profile),
    )
    try:
        pockets = parse_predictions(_predict(receptor), receptor.residue_ids)
    except (P2RankError, RuntimeError) as error:
        code = str(error)
        return {**report, "blocker": code if code in BLOCKERS else "p2rank_provider_failure"}
    return {**report, "status": "complete", "pockets": pockets}


def report_view(raw: object) -> dict:
    """Keep only attributed, typed fields from a saved or fresh pocket report."""
    view = _record()
    invalid = {**view, "blocker": "p2rank_invalid_output"}
    if not isinstance(raw, Mapping) or any(raw.get(key) != view[key] for key in (
        "method", "endpoint", "release", "archive_sha256", "scope",
    )):
        return invalid
    blocker = raw.get("blocker")
    before_prediction = raw.get("status") == "untested" and "receptor_sha256" not in raw
    if before_prediction:
        return {**view, "blocker": blocker} if isinstance(blocker, str) and blocker in BLOCKERS else invalid
    digest, source, profile = raw.get("receptor_sha256"), raw.get("structure_source"), raw.get("profile")
    receptor_format, residues = raw.get("receptor_format"), raw.get("residue_count")
    if not (
        isinstance(digest, str) and _HASH.fullmatch(digest) and isinstance(source, str) and source in _SOURCES
        and profile == _profile(source) and receptor_format in ("pdb", "cif")
        and type(residues) is int and 20 <= residues <= _MAX_ATOMS
    ):
        return invalid
    view.update(
        receptor_sha256=digest, structure_source=source, profile=profile, receptor_format=receptor_format,
        residue_count=residues,
        **_model_evidence(profile),
    )
    pockets = raw.get("pockets")
    if raw.get("status") == "untested" and pockets == [] and isinstance(blocker, str) and blocker in BLOCKERS:
        return {**view, "blocker": blocker}
    if (
        raw.get("status") != "complete" or blocker is not None or not isinstance(pockets, list)
        or len(pockets) > _MAX_POCKETS
        or any(not _pocket_valid(pocket, rank) or len(pocket["residue_ids"]) > residues
               or (rank > 1 and pocket["score"] > pockets[rank - 2]["score"])
               for rank, pocket in enumerate(pockets, 1))
    ):
        return invalid
    return {**view, "status": "complete", "pockets": [dict(pocket) for pocket in pockets]}
