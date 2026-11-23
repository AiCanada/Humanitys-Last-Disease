"""Native TNP worker for a single finished VHH. Runs inside the separate Bioconda environment."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
from collections.abc import Callable
from importlib.metadata import version
from pathlib import Path

# Digests published with the four NanoBodyBuilder2 checkpoints in Zenodo record 7258553.
# Checking before invoking the provider prevents its first-use downloader from running on a design.
_WEIGHT_MD5 = (
    "4591075f467ca9f76a37a5d1d3cfe591",
    "620fc916720bc7068cd18c7afa1aea8d",
    "f1ef0a66a54efb9d14ddbb97fdc30785",
    "a7e14a33f4c00e96df896120e9cf8522",
)
_FIELDS = {
    "L": ("Total CDR Length", "residues"),
    "L3": ("CDR3 Length", "residues"),
    "C": ("CDR3 Compactness", "residues/angstrom"),
    "PSH": ("PSH", "TNP patch score"),
    "PPC": ("PPC", "TNP patch score"),
    "PNC": ("PNC", "TNP patch score"),
}


class ProfileError(ValueError):
    """A report cannot be attributed to the selected complete domain."""


def unique_fields(pairs: list) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ProfileError("duplicate_json_key")
        result[key] = value
    return result


def _provider() -> dict:
    versions = {
        name: version(name) for name in ("tnp", "ImmuneBuilder", "torch", "openmm", "biopython", "anarci", "freesasa")
    }
    if versions["tnp"] != "0.1.0" or versions["ImmuneBuilder"] != "1.2":
        raise ProfileError("unsupported_tnp_provider")
    spec = importlib.util.find_spec("ImmuneBuilder")
    if spec is None or not spec.origin:
        raise ProfileError("tnp_provider_unavailable")
    dssp = subprocess.run(
        [str(Path(sys.executable).with_name("mkdssp")), "--version"],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    if dssp.returncode or not dssp.stdout.startswith("mkdssp version "):
        raise ProfileError("dssp_version_unavailable")
    versions["DSSP"] = dssp.stdout.removeprefix("mkdssp version ").strip()
    weights_dir = Path(spec.origin).parent / "trained_model"
    fingerprints = []
    for i, expected in enumerate(_WEIGHT_MD5, 1):
        path = weights_dir / f"nanobody_model_{i}"
        if not path.is_file():
            raise ProfileError("model_weights_missing")
        md5, sha = hashlib.md5(usedforsecurity=False), hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                md5.update(chunk)
                sha.update(chunk)
        if md5.hexdigest() != expected:
            raise ProfileError("model_weight_digest_mismatch")
        fingerprints.append(sha.hexdigest())
    return {"versions": versions, "model_weights_sha256": fingerprints}


def _number_domain(sequence: str) -> list[tuple[int, str]]:
    from anarci import anarci

    numbering, details, _ = anarci([("domain", sequence)], scheme="imgt", allowed_species=None, ncpu=1)
    domains = numbering[0]
    if not domains or len(domains) != 1 or not details[0] or details[0][0]["chain_type"] != "H":
        raise ProfileError("single_heavy_variable_domain_required")
    positions, start, end = domains[0]
    residues = [(pos, aa) for pos, aa in positions if aa != "-"]
    if start != 0 or end != len(sequence) - 1 or "".join(aa for _, aa in residues) != sequence:
        raise ProfileError("domain_covers_only_part_of_chain")
    return [pos for pos, _ in residues]


def _check_model(data: bytes, sequence: str, numbering: list[tuple[int, str]]) -> None:
    from Bio.Data.PDBData import protein_letters_3to1

    residues = {}
    backbone = {}
    if not data or len(data) > 8 * 1024 * 1024:
        raise ProfileError("native_model_unreadable")
    for line in data.decode("ascii").splitlines():
        if line.startswith(("MODEL ", "ENDMDL", "HETATM")):
            raise ProfileError("unexpected_native_model")
        if not line.startswith("ATOM  "):
            continue
        aa = protein_letters_3to1.get(line[17:20])
        if aa is None or line[21:22] != "H" or line[16:17] != " ":
            raise ProfileError("unexpected_native_model")
        key = (int(line[22:26]), line[26:27])
        if residues.setdefault(key, aa) != aa:
            raise ProfileError("unexpected_native_model")
        xyz = [float(line[i : i + 8]) for i in (30, 38, 46)]
        atom = line[12:16].strip()
        atoms = backbone.setdefault(key, set())
        if not all(map(math.isfinite, xyz)) or atom in atoms:
            raise ProfileError("unexpected_native_model")
        atoms.add(atom)
    if "".join(residues.values()) != sequence:
        raise ProfileError("model_sequence_mismatch")
    if list(residues) != numbering or any(not {"N", "CA", "C", "O"} <= atoms for atoms in backbone.values()):
        raise ProfileError("model_numbering_or_backbone_mismatch")


def _measurements(row: object, lengths: tuple[int, int], assign_flag: Callable) -> tuple[dict, list[str]]:
    if not isinstance(row, dict) or row.get("name") != "domain":
        raise ProfileError("native_profile_absent")
    flags = row.get("Flags")
    if not isinstance(flags, dict) or flags.keys() != _FIELDS.keys():
        raise ProfileError("invalid_native_flags")
    if not {name for name, _ in _FIELDS.values()}.issubset(row):
        raise ProfileError("incomplete_native_report")
    values, missing = {}, []
    for key, (field, unit) in _FIELDS.items():
        value = row[field]
        # TNP 0.1.0 emits a JSON NaN with a grey flag for a missing compactness anchor.
        if key == "C" and (value is None or (type(value) is float and math.isnan(value))):
            if flags[key] != "grey":
                raise ProfileError("native_flag_mismatch")
            missing.append(key)
            value = None
        else:
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ProfileError("invalid_native_measurement")
            if key in ("L", "L3") and type(value) is not int:
                raise ProfileError("invalid_native_length")
            if key == "C" and value == 0:
                raise ProfileError("invalid_native_measurement")
            if assign_flag(key, value) != flags[key]:
                raise ProfileError("native_flag_mismatch")
        values[key] = {"value": value, "unit": unit, "flag": flags[key]}
    if (row["Total CDR Length"], row["CDR3 Length"]) != lengths:
        raise ProfileError("native_cdr_length_mismatch")
    return values, missing


def _check_details(path: Path, sequence: str, numbering: list[tuple[int, str]]) -> None:
    from Bio.Data.PDBData import protein_letters_3to1

    # The native CLI can return six numbers and exit zero after DSSP or model processing fails.
    if not path.is_file() or path.stat().st_size > 8 * 1024 * 1024:
        raise ProfileError("native_model_annotation_incomplete")
    details = json.loads(path.read_bytes(), object_pairs_hook=unique_fields)
    if not isinstance(details, dict):
        raise ProfileError("native_model_annotation_incomplete")
    dssp = details.get("dssp_properties")
    if (
        details.get("scheme") != "imgt"
        or details.get("sequences") != {"H": sequence}
        or details.get("heavy_numbering") != [f"{pos}{ins}" for pos, ins in numbering]
        or details.get("dssp_message") != ""
        or not isinstance(dssp, dict)
        or not isinstance(dssp.get("H"), list)
        or len(dssp["H"]) != len(sequence)
    ):
        raise ProfileError("native_model_annotation_incomplete")
    for row, residue, position in zip(dssp["H"], sequence, numbering, strict=True):
        if (
            not isinstance(row, list)
            or len(row) != 3
            or row[0] != list(position)
            or protein_letters_3to1.get(row[1]) != residue
            or not isinstance(row[2], dict)
            or "SS_DSSP" not in row[2]
        ):
            raise ProfileError("native_model_annotation_incomplete")


def profile(sequence: str, directory: Path) -> dict:
    """Run in a private, initially empty directory with a constant provider input name."""
    provider = _provider()
    from theraprofnano.thresholds import assign_flag

    numbering = _number_domain(sequence)
    cdr3 = sum(105 <= pos <= 117 for pos, _ in numbering)
    lengths = (cdr3 + sum(27 <= pos <= 38 or 56 <= pos <= 65 for pos, _ in numbering), cdr3)
    executable = Path(sys.executable).with_name("TNP")
    result = subprocess.run(
        [str(executable), "--seq", sequence, "--name", "domain", "--hscale", "0", "--output", "."],
        cwd=directory,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if result.returncode:
        raise ProfileError("native_tnp_failed")
    report = directory / "TNP_Results_SingleSeqEntry_domain.json"
    if not report.is_file() or report.stat().st_size > 65536:
        raise ProfileError("native_profile_absent")
    raw = report.read_bytes()
    payload = json.loads(raw, object_pairs_hook=unique_fields)
    if not isinstance(payload, dict) or payload.keys() != {"domain"}:
        raise ProfileError("native_sample_mismatch")
    values, missing = _measurements(payload["domain"], lengths, assign_flag)
    path = directory / "Final_Models/domain_NanoBodyBuilder2_Model.pdb"
    if not path.is_file() or path.stat().st_size > 8 * 1024 * 1024:
        raise ProfileError("native_model_unreadable")
    model = path.read_bytes()
    _check_model(model, sequence, numbering)
    _check_details(directory / "Final_Models/domain_NanoBodyBuilder2_Model_Details.json", sequence, numbering)
    return {
        "status": "partial" if missing else "profiled",
        "missing_metrics": missing,
        "measurements": values,
        "provider": provider,
        "structure_source": "NanoBodyBuilder2",
        "model_sha256": hashlib.sha256(model).hexdigest(),
        "native_report_sha256": hashlib.sha256(raw).hexdigest(),
        "numbering": "IMGT",
        "hydrophobicity_scale": "Kyte-Doolittle",
        "pH": 7.4,
    }


def main() -> None:
    # Keep third-party prints, including input sequences, out of the result transport.
    with os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8") as output, open(os.devnull, "w") as quiet:
        os.dup2(quiet.fileno(), sys.stdout.fileno())
        try:
            result = profile(sys.stdin.read().strip(), Path.cwd())
        except ProfileError as error:
            result = {"status": "untested", "blocker": str(error)}
        except (ImportError, OSError):
            result = {"status": "untested", "blocker": "tnp_provider_unavailable"}
        except Exception:
            # The optional native libraries do not share one exception vocabulary.
            result = {"status": "untested", "blocker": "native_tnp_failed"}
        print(json.dumps(result, allow_nan=False), file=output)


if __name__ == "__main__":
    main()
