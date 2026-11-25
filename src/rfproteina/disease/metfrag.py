"""Score a supplied positive-ion MS/MS spectrum against nominated local ligands."""

from __future__ import annotations

import csv
import io
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from math import isfinite
from pathlib import Path

from rfproteina.disease import corpus_provenance, ligand_admet, ligand_inputs, metfrag_runtime

METHOD_ID = "metfrag"
METHOD = "metfrag_local_fragmenter_2_6_11"
ENDPOINT = "msms_candidate_fragment_explanation"
MANIFEST = "metfrag.json"
ADDUCT = "[M+H]+"
_PROTON_MASS = 1.007276466621
_MAX_CANDIDATES = 32
_MAX_PEAKS = 500
_FILE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,119}\.tsv", re.IGNORECASE)
_DEVICE = re.compile(r"(?:CON|PRN|AUX|NUL|COM[0-9]|LPT[0-9])(?:\.|$)", re.IGNORECASE)
_HASH = re.compile(r"[a-f0-9]{64}")
_ELEMENTS = {"H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"}
_SETTINGS = {
    "precursor_mz": (20.0, 1500.0),
    "precursor_ppm": (0.5, 50.0),
    "fragment_ppm": (0.0, 100.0),
    "fragment_da": (0.0, 0.05),
}
_UNTESTED_ROWS = frozenset({"metfrag_unsupported_candidate", "metfrag_precursor_mismatch"})
BLOCKERS = frozenset(
    {
        "metfrag_manifest_missing",
        "invalid_metfrag_manifest",
        "metfrag_peaks_unreadable",
        "metfrag_invalid_peaks",
        "metfrag_candidates_unreadable",
        "metfrag_no_candidates",
        "metfrag_candidates_over_limit",
        "metfrag_duplicate_candidate",
        "metfrag_candidate_scope_missing",
        "metfrag_rdkit_unavailable",
        "metfrag_no_precursor_matches",
        "metfrag_unsupported_candidate",
        "metfrag_precursor_mismatch",
        "metfrag_invalid_report",
        *metfrag_runtime.BLOCKERS,
    }
)


class MetfragError(ValueError):
    """A named input, attribution or output failure."""


@dataclass(frozen=True)
class Candidate:
    name: str
    smiles: str = ""
    formula: str = ""
    inchi: str = ""
    inchi_key: str = ""
    neutral_mass: float | None = None
    mass_error_ppm: float | None = None
    blocker: str = ""

    def record(self) -> dict:
        row = {"candidate_id": self.name}
        if self.smiles:
            row.update(
                candidate_sha256=sha256(self.smiles.encode("ascii")).hexdigest(),
                neutral_monoisotopic_mass_da=self.neutral_mass,
                precursor_mass_error_ppm=self.mass_error_ppm,
            )
        if self.blocker:
            row.update(status="untested", blocker=self.blocker)
        return row


@dataclass(frozen=True)
class Plan:
    settings: dict[str, float]
    scope: str
    candidates: tuple[Candidate, ...]
    peaks: bytes
    input_peak_count: int
    spectrum_sha256: str

    def representatives(self) -> dict[str, Candidate]:
        """Keep the first native identifier for each eligible canonical isomeric SMILES."""
        seen = set()
        representatives = {}
        for index, candidate in enumerate(self.candidates, 1):
            if not candidate.blocker and candidate.smiles not in seen:
                representatives[f"m{index:03d}"] = candidate
                seen.add(candidate.smiles)
        return representatives


def _unique(pairs: list[tuple[str, object]]) -> dict:
    fields = dict(pairs)
    if len(fields) != len(pairs):
        raise ValueError("Duplicate field.")
    return fields


def _number(value: object, lower: float, upper: float) -> bool:
    return type(value) in (int, float) and lower <= value <= upper


def _settings(directory: Path) -> tuple[str, dict[str, float]]:
    try:
        raw = ligand_inputs.read_regular(directory / MANIFEST, 4096, "metfrag_manifest", MetfragError)
        data = json.loads(raw.decode("utf-8-sig"), object_pairs_hook=_unique)
        if not (
            isinstance(data, dict)
            and data.keys() == {"version", "peaks", "adduct", *_SETTINGS}
            and type(data["version"]) is int
            and data["version"] == 1
            and data["adduct"] == ADDUCT
        ):
            raise ValueError("Invalid measured spectrum settings.")
        peaks = data["peaks"]
        if not isinstance(peaks, str) or not _FILE.fullmatch(peaks) or _DEVICE.match(peaks):
            raise ValueError("Invalid peak filename.")
        if not all(_number(data[key], *bounds) for key, bounds in _SETTINGS.items()):
            raise ValueError("Invalid instrument settings.")
        if not (data["fragment_ppm"] or data["fragment_da"]):
            raise ValueError("Missing fragment mass tolerance.")
        return peaks, {key: float(data[key]) for key in _SETTINGS}
    except (ValueError, RecursionError) as error:
        if isinstance(error, MetfragError) and str(error) == "metfrag_manifest_missing":
            raise
        raise MetfragError("invalid_metfrag_manifest") from error


def _peaks(directory: Path, filename: str, precursor_mz: float) -> tuple[bytes, int, str]:
    try:
        raw = ligand_inputs.read_regular(directory / filename, 64 * 1024, "metfrag_peaks", MetfragError)
    except MetfragError as error:
        raise MetfragError("metfrag_peaks_unreadable") from error
    try:
        rows = csv.reader(io.StringIO(raw.decode("utf-8-sig"), newline=""), delimiter="\t", strict=True)
        if next(rows, None) != ["mz", "intensity"]:
            raise ValueError("Invalid peak header.")
        values = {}
        for mz_text, intensity_text in rows:
            mz, intensity = float(mz_text), float(intensity_text)
            if not (1 <= mz < precursor_mz and 0 < intensity <= 1e15 and mz not in values and len(values) < _MAX_PEAKS):
                raise ValueError("Invalid, duplicate or excess measured peak.")
            values[mz] = intensity
        if len(values) < 3:
            raise ValueError("Insufficient measured peaks.")
    except (ValueError, csv.Error) as error:
        raise MetfragError("metfrag_invalid_peaks") from error
    text = "".join(f"{mz:.17g} {values[mz]:.17g}\n" for mz in sorted(values))
    return text.encode("ascii"), len(values), sha256(raw).hexdigest()


def _candidate(name: str, smiles: str, settings: dict[str, float]) -> Candidate:
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors

    try:
        molecule = ligand_inputs.parse_nominated_smiles(smiles, Chem, remove_hs=True) if len(smiles) <= 512 else None
        if molecule is None or not (
            4 <= molecule.GetNumHeavyAtoms() <= 40
            and Chem.GetFormalCharge(molecule) == 0
            and len(Chem.GetMolFrags(molecule)) == 1
            and all(
                atom.GetSymbol() in _ELEMENTS
                and not (atom.GetIsotope() or atom.GetNumRadicalElectrons() or atom.GetAtomMapNum())
                for atom in molecule.GetAtoms()
            )
        ):
            raise ValueError("Unsupported chemistry.")
        inchi = Chem.MolToInchi(molecule)
        inchi_key = Chem.InchiToInchiKey(inchi)
        if not inchi.startswith("InChI=1S/") or not inchi_key:
            raise ValueError("No reproducible structure identifier.")
        neutral = rdMolDescriptors.CalcExactMolWt(molecule)
        measured = settings["precursor_mz"]
        predicted = neutral + _PROTON_MASS
        error_ppm = (predicted - measured) / measured * 1e6
        blocker = "metfrag_precursor_mismatch" if abs(error_ppm) > settings["precursor_ppm"] else ""
        return Candidate(
            name,
            Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True),
            rdMolDescriptors.CalcMolFormula(molecule),
            inchi,
            inchi_key,
            neutral,
            error_ppm,
            blocker,
        )
    except (ValueError, RuntimeError):
        return Candidate(name, blocker="metfrag_unsupported_candidate")


def prepare(directory: Path, candidate: str | None = None) -> Plan:
    peaks_file, settings = _settings(directory)
    peaks, peak_count, digest = _peaks(directory, peaks_file, settings["precursor_mz"])
    scope, rows, blocker = ligand_inputs.bounded_smiles_candidates(
        directory,
        max_bytes=ligand_admet.MAX_INPUT_BYTES,
        max_rows=ligand_admet.MAX_CANDIDATES,
    )
    if blocker:
        raise MetfragError("metfrag_candidates_unreadable")
    if candidate is not None and (scope != "candidate_table" or not ligand_admet.valid_candidate_id(candidate)):
        raise MetfragError("metfrag_candidate_scope_missing")
    selected = list(ligand_inputs.nominated_rows(rows, candidate))
    if not selected:
        raise MetfragError("metfrag_no_candidates")
    if len(selected) > _MAX_CANDIDATES:
        raise MetfragError("metfrag_candidates_over_limit")
    if any(len(row) != 2 or not ligand_admet.valid_candidate_id(row[0]) for row in selected):
        raise MetfragError("metfrag_candidates_unreadable")
    if len({row[0] for row in selected}) != len(selected):
        raise MetfragError("metfrag_duplicate_candidate")
    try:
        from rdkit import rdBase
    except (ImportError, OSError) as error:
        raise MetfragError("metfrag_rdkit_unavailable") from error
    with rdBase.BlockLogs():
        candidates = tuple(_candidate(name, smiles, settings) for name, smiles in selected)
    return Plan(settings, scope, candidates, peaks, peak_count, digest)


def installed() -> bool:
    try:
        from rdkit import Chem
    except (ImportError, OSError):
        return False
    return callable(Chem.MolToInchi) and metfrag_runtime.installed()


def _database(candidates: dict[str, Candidate]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["Identifier", "SMILES", "InChI", "InChIKey", "MolecularFormula", "MonoisotopicMass"])
    for key, row in candidates.items():
        writer.writerow([key, row.smiles, row.inchi, row.inchi_key, row.formula, f"{row.neutral_mass:.10f}"])
    return buffer.getvalue().encode("ascii")


def _result_rows(raw: bytes, candidates: dict[str, Candidate], peak_count: int) -> dict[str, dict]:
    try:
        rows = csv.DictReader(io.StringIO(raw.decode("utf-8-sig"), newline=""), strict=True)
        required = {"Identifier", "InChIKey", "FragmenterScore", "NoExplPeaks", "NumberPeaksUsed"}
        if (
            not rows.fieldnames
            or len(rows.fieldnames) != len(set(rows.fieldnames))
            or not required <= set(rows.fieldnames)
        ):
            raise ValueError("Unexpected MetFrag header.")
        results = {}
        for row in rows:
            key = row["Identifier"]
            if key not in candidates or None in row:
                raise ValueError("Unattributed result.")
            candidate = candidates[key]
            if candidate.smiles in results or row["InChIKey"] != candidate.inchi_key:
                raise ValueError("Duplicate or mismatched structure.")
            used, explained = int(row["NumberPeaksUsed"]), int(row["NoExplPeaks"])
            score = float(row["FragmenterScore"])
            if not (
                used == peak_count and 0 <= explained <= used and 0 <= score < 1e25 and bool(score) == bool(explained)
            ):
                raise ValueError("Invalid matched peak count or score.")
            results[candidate.smiles] = {
                "status": "scored" if explained else "no_matches",
                "fragmenter_score": score,
                "explained_peaks": explained,
                "used_peaks": used,
            }
        if len(results) != len(candidates):
            raise ValueError("Incomplete candidate coverage.")
        return results
    except (ValueError, TypeError, csv.Error) as error:
        raise MetfragError("metfrag_invalid_report") from error


def _base() -> dict:
    return {
        "method": METHOD,
        "endpoint": ENDPOINT,
        "release": metfrag_runtime.RELEASE,
        "jar_sha256": metfrag_runtime.JAR_SHA256,
        "adduct": ADDUCT,
        "status": "untested",
        "candidates": [],
        "corpus_provenance": corpus_provenance.declare(METHOD, against=ENDPOINT),
    }


def _comparison(candidates: list[dict]) -> dict:
    """Count experimental alternatives once per evaluated graph in this spectrum."""
    by_structure = {}
    tested = 0
    for row in candidates:
        if row["status"] == "untested":
            continue
        tested += 1
        evidence = (row["fragmenter_score"], row["explained_peaks"])
        if by_structure.setdefault(row["candidate_sha256"], evidence) != evidence:
            raise MetfragError("metfrag_invalid_report")
    scores = [score for score, _ in by_structure.values()]
    top = max(scores, default=0)
    return {
        "tested_nominations": tested,
        "distinct_structures": len(scores),
        "top_scoring_structures": scores.count(top) if top > 0 else 0,
    }


def screen_run(directory: Path, candidate: str | None = None) -> dict:
    report = _base()
    try:
        plan = prepare(directory, candidate)
        report.update(
            spectrum_sha256=plan.spectrum_sha256,
            input_peak_count=plan.input_peak_count,
            scope=plan.scope,
            **plan.settings,
        )
        eligible = plan.representatives()
        skipped = [item.record() for item in plan.candidates if item.blocker]
        if not eligible:
            return {**report, "blocker": "metfrag_no_precursor_matches", "candidates": skipped}
        raw = metfrag_runtime.predict(plan.peaks, _database(eligible), **plan.settings)
        found = _result_rows(raw, eligible, plan.input_peak_count)
        rows = [{**item.record(), **found[item.smiles]} for item in plan.candidates if not item.blocker]
        rows.sort(key=lambda result: (-result["fragmenter_score"], result["candidate_id"]))
        return {
            **report, "status": "partial" if skipped else "complete", "candidates": [*rows, *skipped],
            "comparison": _comparison(rows),
        }
    except (MetfragError, metfrag_runtime.MetfragRuntimeError) as error:
        return {**report, "blocker": str(error)}


def report_view(raw: object) -> dict:
    """Validate prepared scientific fields and keep private measurement inputs out of shared views."""
    report = _base()
    invalid = {**report, "blocker": "metfrag_invalid_report"}
    attestation = ("method", "endpoint", "release", "jar_sha256", "adduct")
    if not isinstance(raw, Mapping) or any(raw.get(key) != report[key] for key in attestation):
        return invalid
    status, blocker, source = (raw.get(key) for key in ("status", "blocker", "candidates"))
    if status == "untested" and source == [] and isinstance(blocker, str) and blocker in BLOCKERS:
        return {**report, "blocker": blocker}
    digest, peaks, scope = (raw.get(key) for key in ("spectrum_sha256", "input_peak_count", "scope"))
    if not (
        isinstance(digest, str)
        and _HASH.fullmatch(digest)
        and type(peaks) is int
        and 3 <= peaks <= _MAX_PEAKS
        and ligand_inputs.valid_saved_smiles_scope(scope, source)
        and 0 < len(source) <= _MAX_CANDIDATES
        and all(_number(raw.get(key), *bounds) for key, bounds in _SETTINGS.items())
        and (raw["fragment_da"] or raw["fragment_ppm"])
    ):
        return invalid
    names = set()
    complete = 0
    candidates = []
    for row in source:
        if (
            not isinstance(row, Mapping)
            or not isinstance((name := row.get("candidate_id")), str)
            or (not ligand_admet.valid_candidate_id(name) or name in names)
        ):
            return invalid
        names.add(name)
        result = {"candidate_id": name}
        row_status, row_blocker = row.get("status"), row.get("blocker")
        if row_status == "untested" and isinstance(row_blocker, str) and row_blocker in _UNTESTED_ROWS:
            result.update(status="untested", blocker=row_blocker)
        elif row_status in ("scored", "no_matches") and row_blocker is None:
            score, matches, used = (row.get(key) for key in ("fragmenter_score", "explained_peaks", "used_peaks"))
            if not (
                type(score) in (int, float) and 0 <= score < 1e25
                and type(matches) is int
                and type(used) is int
                and used == peaks
                and 0 <= matches <= used
                and bool(matches) == bool(score) == (row_status == "scored")
            ):
                return invalid
            result.update(status=row_status, fragmenter_score=score, explained_peaks=matches, used_peaks=used)
            complete += 1
        else:
            return invalid
        if row_blocker != "metfrag_unsupported_candidate":
            structure, mass, ppm = (
                row.get(key)
                for key in (
                    "candidate_sha256",
                    "neutral_monoisotopic_mass_da",
                    "precursor_mass_error_ppm",
                )
            )
            if not (
                isinstance(structure, str)
                and _HASH.fullmatch(structure)
                and _number(mass, 10, 1e4)
                and _number(ppm, -1e6, 5e8)
                and isfinite(ppm)
            ):
                return invalid
            expected_ppm = (mass + _PROTON_MASS - raw["precursor_mz"]) / raw["precursor_mz"] * 1e6
            if abs(expected_ppm - ppm) > 1e-6 or ((abs(ppm) > raw["precursor_ppm"]) != (row_status == "untested")):
                return invalid
            result.update(candidate_sha256=structure, neutral_monoisotopic_mass_da=mass, precursor_mass_error_ppm=ppm)
        candidates.append(result)
    expected = "complete" if complete == len(source) else "partial" if complete else "untested"
    if status != expected or blocker != ("metfrag_no_precursor_matches" if not complete else None):
        return invalid
    report.update(
        status=status,
        spectrum_sha256=digest,
        input_peak_count=peaks,
        scope=scope,
        candidates=candidates,
        **{key: raw[key] for key in _SETTINGS},
    )
    if blocker:
        report["blocker"] = blocker
    if complete:
        try:
            report["comparison"] = _comparison(candidates)
        except MetfragError:
            return invalid
    return report
