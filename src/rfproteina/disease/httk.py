"""Offline, fixed-scenario EPA httk summaries for nominated reference substances."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

from rfproteina.disease import httk_identity, ligand_admet, ligand_inputs
from rfproteina.disease.corpus_provenance import declare

METHOD_ID = "epa_httk"
METHOD = "epa_httk_human_css_v2"
MANIFEST = "httk.json"
ENDPOINT = "total_plasma_steady_state_concentration"
UNIT = "mg/L"
SCENARIO = "human_oral_unit_intake"
VERSION = "2.7.4"
DRAWS = 1000
SEED = 2181
MAX_ENTRIES = 32
MAX_MANIFEST_BYTES = 64 * 1024
PROVIDER = {
    "package": "httk", "version": VERSION, "model": "3compartmentss", "population": "httkpop_adults_18_79",
    "draws": DRAWS, "seed": SEED, "in_vitro_uncertainty": False,
    "identity_source": httk_identity.SOURCE,
}
_CAS = re.compile(r"[1-9][0-9]{1,6}-[0-9]{2}-[0-9]")
_DTX = re.compile(r"DTXSID[0-9]{7,12}")
_WORKER = Path(__file__).with_name("httk_worker.R").resolve()
_WORKER_BLOCKERS = frozenset({
    "httk_reference_identity_unavailable", "httk_human_inputs_unavailable", "httk_model_inapplicable",
    "httk_invalid_draws", "httk_simulation_failed",
})
BLOCKERS = _WORKER_BLOCKERS | frozenset({
    "httk_manifest_missing", "httk_manifest_unreadable", "httk_manifest_over_limit", "httk_invalid_manifest",
    "httk_scenario_unsupported", "httk_candidate_not_nominated", "httk_named_nominations_required",
    "httk_input_over_limit", "httk_missing_ligand", "httk_ambiguous_ligand", "httk_invalid_ligand",
    "httk_rdkit_unavailable", "httk_runtime_unavailable", "httk_worker_timeout", "httk_worker_failed",
    "httk_no_usable_results",
    "httk_identity_reference_unavailable", "httk_structure_unmapped", "httk_structure_mismatch",
})
_CPU_ENV = dict.fromkeys(("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"), "") | dict.fromkeys(
    ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "R_DATATABLE_NUM_THREADS"), "1",
)


class HttkError(ValueError):
    """A named, non-sensitive reason EPA httk cannot report an estimate."""


@dataclass(frozen=True)
class Nomination:
    name: str
    cas: str
    dtxsid: str
    compound_id: str = ""
    blocker: str | None = None
    reference_inchikey: str = ""

    @property
    def reference(self) -> tuple[str, str]:
        return self.cas, self.dtxsid


@dataclass(frozen=True)
class Plan:
    members: tuple[Nomination, ...] = ()
    manifest_sha256: str | None = None
    blocker: str | None = None


def valid_cas(value: object) -> bool:
    if not isinstance(value, str) or not _CAS.fullmatch(value):
        return False
    digits = value.replace("-", "")
    return sum(i * int(n) for i, n in enumerate(reversed(digits[:-1]), 1)) % 10 == int(digits[-1])


def valid_dtxsid(value: object) -> bool:
    return isinstance(value, str) and bool(_DTX.fullmatch(value))


def read_manifest(directory: Path) -> tuple[bytes | None, str | None]:
    try:
        return ligand_inputs.read_regular(directory / MANIFEST, MAX_MANIFEST_BYTES, "httk_manifest", HttkError), None
    except HttkError as error:
        if str(error) == "httk_manifest_size_limit":
            return None, "httk_manifest_over_limit"
        return None, str(error) if str(error) in BLOCKERS else "httk_manifest_unreadable"


def _unique(pairs: list[tuple]) -> dict:
    values = dict(pairs)
    if len(values) != len(pairs):
        raise HttkError("httk_invalid_manifest")
    return values


def _associations(raw: bytes) -> tuple[Nomination, ...]:
    try:
        obj = json.loads(raw.decode("utf-8-sig"), object_pairs_hook=_unique)
    except (ValueError, RecursionError) as error:
        raise HttkError("httk_invalid_manifest") from error
    if not isinstance(obj, dict) or obj.keys() != {"scenario", "candidates"}:
        raise HttkError("httk_invalid_manifest")
    if obj["scenario"] != SCENARIO:
        raise HttkError("httk_scenario_unsupported")
    entries = obj["candidates"]
    if not isinstance(entries, dict) or not entries:
        raise HttkError("httk_invalid_manifest")
    if len(entries) > MAX_ENTRIES:
        raise HttkError("httk_input_over_limit")
    result = []
    for name, ids in entries.items():
        if (
            not ligand_admet.valid_candidate_id(name) or not isinstance(ids, dict) or ids.keys() != {"cas", "dtxsid"}
            or not valid_cas(ids["cas"]) or not valid_dtxsid(ids["dtxsid"])
        ):
            raise HttkError("httk_invalid_manifest")
        result.append(Nomination(name, ids["cas"], ids["dtxsid"]))
    return tuple(result)


def _identify(members: tuple[Nomination, ...], rows: list) -> tuple[Nomination, ...]:
    from rdkit import Chem, rdBase

    selected = {item.name for item in members}
    nominations: dict[str, list] = defaultdict(list)
    for row in rows:
        if row and row[0] in selected:
            nominations[row[0]].append(row)
    bound = []
    keys = {}
    for member in members:
        found = nominations[member.name]
        if len(found) != 1:
            reason = "httk_ambiguous_ligand" if found else "httk_missing_ligand"
            bound.append(replace(member, blocker=reason))
            continue
        if len(found[0]) != 2:
            bound.append(replace(member, blocker="httk_invalid_ligand"))
            continue
        try:
            with rdBase.BlockLogs():
                molecule, graph = ligand_admet.molecular_graph(found[0][1], Chem, unsupported="httk_invalid_ligand")
                key = Chem.MolToInchiKey(molecule)
                if not httk_identity.STANDARD_KEY.fullmatch(key):
                    raise HttkError("httk_invalid_ligand")
            digest = hashlib.sha256(graph.encode("ascii")).hexdigest()
            keys[member.name] = key
            bound.append(replace(member, compound_id="smiles_sha256_" + digest))
        except (ValueError, RuntimeError, OverflowError):
            bound.append(replace(member, blocker="httk_invalid_ligand"))
    by_reference: dict[tuple[str, str], set[str]] = defaultdict(set)
    by_graph: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for member in bound:
        if member.compound_id:
            by_reference[member.reference].add(member.compound_id)
            by_graph[member.compound_id].add(member.reference)
    try:
        index = httk_identity.load_index() if keys else {}
    except httk_identity.IdentityReferenceError:
        index = None
    verified = []
    for member in bound:
        if member.blocker:
            verified.append(member)
        elif len(by_reference[member.reference]) != 1 or len(by_graph[member.compound_id]) != 1:
            verified.append(replace(member, blocker="httk_ambiguous_ligand"))
        elif index is None:
            verified.append(replace(member, blocker="httk_identity_reference_unavailable"))
        elif member.reference not in index:
            verified.append(replace(member, blocker="httk_structure_unmapped"))
        elif keys[member.name] != index[member.reference]:
            verified.append(replace(member, blocker="httk_structure_mismatch"))
        else:
            verified.append(replace(member, reference_inchikey=keys[member.name]))
    return tuple(verified)


def prepare(directory: Path, candidate: str | None = None) -> Plan:
    """Validate the explicit association and frozen ligand nomination; do not call R."""
    raw, error = read_manifest(directory)
    if raw is None:
        return Plan(blocker=error or "httk_manifest_unreadable")
    digest = hashlib.sha256(raw).hexdigest()
    try:
        members = _associations(raw)
    except HttkError as error:
        return Plan(manifest_sha256=digest, blocker=str(error))
    if candidate is not None and not any(item.name == candidate for item in members):
        return Plan(manifest_sha256=digest, blocker="httk_candidate_not_nominated")
    scope, rows, blocker = ligand_inputs.bounded_smiles_candidates(
        directory, max_bytes=ligand_admet.MAX_INPUT_BYTES, max_rows=ligand_admet.MAX_CANDIDATES,
    )
    if scope != "candidate_table" or blocker:
        reason = "httk_input_over_limit" if blocker == "ligand_input_exceeds_interactive_size" else (
            "httk_named_nominations_required" if scope != "candidate_table" else "httk_invalid_ligand"
        )
        return Plan(manifest_sha256=digest, blocker=reason)
    try:
        bound = _identify(members, rows)
    except (ImportError, OSError):
        return Plan(manifest_sha256=digest, blocker="httk_rdkit_unavailable")
    return Plan(tuple(item for item in bound if candidate is None or candidate == item.name), digest)


def runtime_path() -> Path | None:
    """Resolve only trusted runtime configuration, never a file nominated by a client."""
    if override := os.environ.get("RFD4_HTTK_RSCRIPT"):
        chosen = Path(override).absolute()
        return chosen if chosen.is_file() else None
    if data := os.environ.get("DATA_PATH"):
        leaf = "Rscript.exe" if os.name == "nt" else "Rscript"
        chosen = (Path(data) / "httk" / "bin" / leaf).absolute()
        if chosen.is_file():
            return chosen
    return Path(path).absolute() if (path := shutil.which("Rscript")) else None


def _run(executable: Path, action: str, data: bytes, *, timeout: int) -> str:
    try:
        with tempfile.TemporaryDirectory(prefix="rfd4-httk-") as directory:
            # R rejects spaces in its startup temp path. The subprocess already
            # starts inside this private directory, so "." denotes the same place.
            scratch = "." if " " in directory else directory
            environment = {**os.environ, **_CPU_ENV, **dict.fromkeys(("TMPDIR", "TMP", "TEMP"), scratch)}
            result = subprocess.run(
                [str(executable), "--vanilla", str(_WORKER), action], cwd=directory, env=environment,
                input=data, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True, timeout=timeout,
            )
        if len(result.stdout) > 32 * 1024:
            raise HttkError("httk_worker_failed")
        return result.stdout.decode("ascii")
    except subprocess.TimeoutExpired as error:
        raise HttkError("httk_worker_timeout") from error
    except (OSError, subprocess.SubprocessError, UnicodeError) as error:
        raise HttkError("httk_worker_failed") from error


@lru_cache(maxsize=8)
def _available(executable: Path, fingerprint: int, epoch: int) -> bool:
    del fingerprint, epoch  # Reload if the installation changes, or after the refresh interval.
    try:
        return _run(executable, "probe", b"", timeout=12).strip() == f"httk\t{VERSION}"
    except HttkError:
        return False


# Exact nonmissing labels from chem.physical_and_invitro.data in httk 2.7.4.
# A saved evidence record may expose only a reference present for that assay.
_COMMON_ASSAY_REFERENCES = frozenset({
    "Aylward et al. 2010", "Beaudouin et al. 2010", "Collins et al. 1999", "Dodd et al. 1993",
    "Horton et al. 1992", "Ito (2004) Pharm Res 21(5):785-92", "Ito 2004", "Jarnberg and Johanson 1999",
    "Jongeneelen and Berge 2011", "Kreutz 2023", "Naritomi 2003", "Ng et al. 2007", "Paini 2020",
    "Paixao 2012", "Price and Krishnan 2011", "Shibata 2002", "Sternbeck 2012", "Thrall et al. 2002",
    "Thrall et al. 2004", "Tonnelier 2012", "Tornero-Velez et al. 2004", "Wambaugh 2019", "Wetmore 2012",
    "Wetmore 2015", "Williamson 2020", "Wood 2017", "Yamagata 2017",
})
_CLINT_REFERENCES = _COMMON_ASSAY_REFERENCES | {
    "Black 2021", "Brown 2007", "Crizer 2024", "Dawson 2021", "Jones 2017", "Lau2002", "McGinnity 2004",
    "Pirovano 2016", "Zanelli 2012",
}
_FUP_REFERENCES = _COMMON_ASSAY_REFERENCES | {
    "Dixit 1986", "ECVAM", "Gulden 2002", "Hagelberg 1989 (N.A.)", "Lombardo 2018", "Lynn 2025",
    "Nicol 2024", "Obach 1999", "Smeltz 2023", "Sultatos 1986", "Uchimura 2010",
}


def valid_human_in_vitro(value: object) -> bool:
    """Validate raw human assay central values and source labels for the pinned release."""
    if not isinstance(value, dict) or value.keys() != {
        "clint_ul_min_per_million_hepatocytes", "clint_reference", "fraction_unbound_plasma",
        "fraction_unbound_plasma_reference",
    }:
        return False
    clint = value["clint_ul_min_per_million_hepatocytes"]
    fup = value["fraction_unbound_plasma"]
    clint_ref = value["clint_reference"]
    fup_ref = value["fraction_unbound_plasma_reference"]
    return (
        type(clint) in (int, float) and 0 <= clint <= sys.float_info.max
        and type(fup) in (int, float) and 1e-4 <= fup <= 1
        and isinstance(clint_ref, str) and clint_ref in _CLINT_REFERENCES
        and isinstance(fup_ref, str) and fup_ref in _FUP_REFERENCES
    )


def asset_available() -> bool:
    try:
        if not importlib.util.find_spec("rdkit") or not _WORKER.is_file() or (executable := runtime_path()) is None:
            return False
        httk_identity.load_index()
        return _available(executable, executable.stat().st_mtime_ns, int(time.monotonic() // 60))
    except (OSError, ValueError, ImportError):
        return False


def _read_results(text: str, references: tuple[tuple[str, str], ...]) -> dict[tuple[str, str], dict]:
    result = {}
    lines = text.splitlines()
    if len(lines) != len(references):
        raise HttkError("httk_worker_failed")
    for line, expected in zip(lines, references, strict=True):
        fields = line.split("\t")
        if len(fields) != 11 or tuple(fields[:2]) != expected:
            raise HttkError("httk_worker_failed")
        _, _, status, low, center, high, blocker, clint, clint_ref, fup, fup_ref = fields
        if status == "untested":
            if any((low, center, high, clint, clint_ref, fup, fup_ref)) or blocker not in _WORKER_BLOCKERS:
                raise HttkError("httk_worker_failed")
            result[expected] = {"status": "untested", "blocker": blocker}
            continue
        if status != "reported" or blocker:
            raise HttkError("httk_worker_failed")
        try:
            p05, median, p95 = map(float, (low, center, high))
        except ValueError:
            p05 = median = p95 = math.nan
        if not (0 <= p05 <= median <= p95 < math.inf):
            result[expected] = {"status": "untested", "blocker": "httk_invalid_draws"}
            continue
        try:
            inputs = {
                "clint_ul_min_per_million_hepatocytes": float(clint), "clint_reference": clint_ref,
                "fraction_unbound_plasma": float(fup), "fraction_unbound_plasma_reference": fup_ref,
            }
        except ValueError:
            inputs = {}
        if valid_human_in_vitro(inputs):
            result[expected] = {"status": status, "p05": p05, "median": median, "p95": p95, "human_in_vitro": inputs}
        else:
            result[expected] = {"status": "untested", "blocker": "httk_human_inputs_unavailable"}
    return result


def measure(directory: Path, candidate: str | None = None) -> dict:
    """Return population summary results without clinical or screening decisions."""
    record = {
        "method": METHOD, "endpoint": ENDPOINT, "unit": UNIT, "scenario": SCENARIO, "reference_intake_mg_kg_day": 1,
        "scope": "run" if candidate is None else "candidate", "status": "untested", "attempted": 0,
        "reported": 0, "members": [], "corpus_provenance": declare(METHOD),
    }
    plan = prepare(directory, candidate)
    if plan.manifest_sha256:
        record["manifest_sha256"] = plan.manifest_sha256
    if plan.blocker:
        return {**record, "blocker": plan.blocker}
    references = tuple(dict.fromkeys(member.reference for member in plan.members if not member.blocker))
    results = {}
    problem = None
    if references:
        if (executable := runtime_path()) is None or not asset_available():
            problem = "httk_runtime_unavailable"
        else:
            data = "".join(f"{cas}\t{dtxsid}\n" for cas, dtxsid in references).encode("ascii")
            try:
                results = _read_results(_run(executable, "run", data, timeout=240), references)
            except HttkError as error:
                problem = str(error)
    rows = []
    for member in plan.members:
        row = {"candidate_id": member.name, "cas": member.cas, "dtxsid": member.dtxsid}
        if member.compound_id:
            row["compound_id"] = member.compound_id
        if member.reference_inchikey:
            row["reference_inchikey"] = member.reference_inchikey
        if reason := member.blocker or problem:
            row.update(status="untested", blocker=reason)
        else:
            row.update(results.get(member.reference, {"status": "untested", "blocker": "httk_worker_failed"}))
        rows.append(row)
    reported = sum(row["status"] == "reported" for row in rows)
    record.update(attempted=len(rows), reported=reported, members=rows)
    if reported:
        record.update(status="reported" if reported == len(rows) else "partial", provider=PROVIDER.copy())
    else:
        record["blocker"] = problem or "httk_no_usable_results"
    return record
