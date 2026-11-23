"""Site-resolved aqueous microscopic pKa for nominated finished-run small molecules."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import math
import os
import signal
import subprocess
import sys
from collections import Counter
from collections.abc import Iterable, Sequence
from contextlib import suppress
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rdkit import Chem

from rfproteina.disease.corpus_provenance import declare
from rfproteina.disease.ligand_inputs import parse_nominated_smiles, read_sdf_candidates, read_smiles_candidates
from rfproteina.disease.qupkake_worker import METHOD, REVISION, SEED

ENDPOINT = "aqueous_microscopic_pKa"
MANIFEST = "qupkake.json"
# Elements represented in the organic ChEMBL pretraining corpus (QupKake, Methods).
_TRAINING_ELEMENTS = frozenset({1, 6, 7, 8, 9, 15, 16, 17, 35, 53})


def _base(tautomerize: bool, source: str) -> dict:
    return {
        "method": METHOD,
        "revision": REVISION,
        "endpoint": ENDPOINT,
        "unit": "pKa",
        "status": "untested",
        "source": source,
        "tautomerize": tautomerize,
        "embedding_seed": SEED,
        "applicability": (
            "The model was fitted to the dominant acidic or basic transition. Additional sites in polyprotic "
            "molecules have weaker experimental support. Aliphatic carbon acids were absent from training."
        ),
        "attempted": 0,
        "unique_compounds": 0,
        "compounds_with_predictions": 0,
        "compounds_without_enumerated_sites": 0,
        "predicted_sites": 0,
        "inputs": [],
        "compounds": [],
        "structures": {},
        "corpus_provenance": declare(METHOD, against=ENDPOINT),
    }


def _runtime() -> Path | None:
    explicit = os.environ.get("QUPKAKE_PYTHON")
    if explicit:
        path = Path(explicit).expanduser().absolute()
        return path if path.is_file() else None
    local = Path.home() / ".local/share/rfproteina/qupkake-f7a294e-cpu/env/bin/python"
    if local.is_file():
        return local
    return Path(sys.executable) if importlib.util.find_spec("qupkake") else None


def _valid_report(result: dict) -> bool:
    """Accept only attributable, finite values across the isolated interpreter boundary."""
    status, sites, states = result.get("status"), result.get("sites"), result.get("states", {})
    if not isinstance(sites, list) or not isinstance(states, dict):
        return False
    if status == "untested" and not sites and "provider" not in result:
        return isinstance(result.get("blocker"), str) and bool(result["blocker"])
    if status not in ("predicted", "partial", "untested", "no_sites"):
        return False
    provider = result.get("provider")
    if not isinstance(provider, dict) or provider.get("revision") != REVISION or provider.get("device") != "cpu":
        return False
    if any(
        not isinstance(state, dict)
        or not isinstance(state.get("smiles"), str)
        or not isinstance(state.get("atom_mapped_smiles"), str)
        or type(state.get("formal_charge")) is not int
        for state in states.values()
    ):
        return False
    if any(
        not isinstance(result.get(k), str) or result[k] not in states for k in ("input_state_id", "selected_state_id")
    ):
        return False
    rejected = result.get("rejected_nominations")
    if not isinstance(rejected, list) or any(not isinstance(r, dict) or not r.get("blocker") for r in rejected):
        return False
    failed_heads = result.get("failed_site_heads")
    if not isinstance(failed_heads, list) or any(
        not isinstance(r, dict) or r.get("kind") not in ("acidic", "basic") or not r.get("blocker")
        for r in failed_heads
    ):
        return False
    predicted = 0
    for row in sites:
        if not isinstance(row, dict) or row.get("method") != METHOD or row.get("kind") not in ("acidic", "basic"):
            return False
        if type(row.get("site_atom_map")) is not int or row["site_atom_map"] < 1:
            return False
        nominees = row.get("nominated_atom_maps")
        if (
            row.get("endpoint") != ENDPOINT
            or row.get("unit") != "pKa"
            or type(row.get("atomic_number")) is not int
            or row.get("atomic_number") not in _TRAINING_ELEMENTS
        ):
            return False
        if not isinstance(nominees, list) or not nominees or any(type(n) is not int or n < 1 for n in nominees):
            return False
        acid, base = row.get("protonated_state_id"), row.get("deprotonated_state_id")
        if not isinstance(acid, str) or not isinstance(base, str) or acid not in states or base not in states:
            return False
        if states[acid]["formal_charge"] != states[base]["formal_charge"] + 1:
            return False
        if result["selected_state_id"] != (base if row["kind"] == "basic" else acid):
            return False
        if row.get("status") == "untested" and row.get("blocker") and "micro_pka" not in row:
            continue
        value = row.get("micro_pka")
        if row.get("status") != "predicted" or type(value) not in (float, int) or not math.isfinite(value):
            return False
        predicted += 1
    if status == "no_sites":
        return not sites and not rejected and not failed_heads
    if status == "untested":
        return predicted == 0 and bool(result.get("blocker"))
    return predicted > 0 and (status == "partial") == bool(rejected or failed_heads or predicted != len(sites))


def _run(smiles: str, tautomerize: bool) -> dict:
    failure = {"status": "untested", "sites": []}
    runtime = _runtime()
    if runtime is None:
        return {**failure, "blocker": "qupkake_runtime_unavailable"}
    worker = Path(__file__).with_name("qupkake_worker.py").resolve()
    try:
        with TemporaryDirectory(prefix="qupkake-") as tmp:
            env = {
                **os.environ,
                "CUDA_VISIBLE_DEVICES": "",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "TMPDIR": tmp,
                "TMP": tmp,
                "TEMP": tmp,
            }
            with subprocess.Popen(
                [str(runtime), "-I", str(worker)],
                cwd=tmp,
                env=env,
                text=True,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                start_new_session=os.name == "posix",
            ) as child:
                try:
                    output, _ = child.communicate(
                        json.dumps({"smiles": smiles, "tautomerize": tautomerize}), timeout=600
                    )
                except subprocess.TimeoutExpired:
                    if os.name == "posix":
                        with suppress(ProcessLookupError):
                            os.killpg(child.pid, signal.SIGKILL)
                    else:
                        child.kill()
                    child.communicate()
                    return {**failure, "blocker": "qupkake_timeout"}
                if child.returncode:
                    return {**failure, "blocker": "qupkake_worker_failed"}
            result = json.loads(output)
            if not isinstance(result, dict) or result.pop("input", None) != smiles:
                return {**failure, "blocker": "qupkake_response_identity_mismatch"}
            if not _valid_report(result):
                return {**failure, "blocker": "qupkake_worker_invalid_report"}
            return result
    except (OSError, ValueError, OverflowError):
        return {**failure, "blocker": "qupkake_worker_failed"}


def _identify(mol: Chem.Mol | None, chem: ModuleType) -> tuple[str, list[int]]:
    if mol is None or not mol.GetNumAtoms():
        raise ValueError("invalid_molecular_structure")
    if len(chem.GetMolFrags(mol)) != 1:
        raise ValueError("disconnected_structure_requires_nominated_microstate")
    if any(a.GetAtomicNum() == 0 or a.HasQuery() for a in mol.GetAtoms()) or any(b.HasQuery() for b in mol.GetBonds()):
        raise ValueError("query_or_dummy_atoms_unsupported")
    if any(a.GetAtomicNum() not in _TRAINING_ELEMENTS for a in mol.GetAtoms()):
        raise ValueError("element_outside_qupkake_training_domain")
    if any(a.GetNumRadicalElectrons() for a in mol.GetAtoms()):
        raise ValueError("open_shell_structure_unsupported")
    if mol.GetStereoGroups():
        raise ValueError("enhanced_stereochemistry_unsupported")
    for atom in mol.GetAtoms():
        atom.SetIntProp("_q_source", atom.GetIdx())
        atom.SetAtomMapNum(0)
    core = chem.RemoveHs(mol)
    if any(a.GetAtomicNum() == 1 for a in core.GetAtoms()):
        raise ValueError("nonremovable_hydrogen_unsupported")
    smiles = chem.MolToSmiles(core, isomericSmiles=True)
    ordering = ast.literal_eval(core.GetProp("_smilesAtomOutputOrder"))
    if sorted(ordering) != list(range(core.GetNumAtoms())):
        raise ValueError("canonical_atom_order_unavailable")
    return smiles, [core.GetAtomWithIdx(i).GetIntProp("_q_source") for i in ordering]


def profile(
    candidates: Iterable[Sequence[str]], *, tautomerize: bool = False, source: str = "smiles", sdf: bool = False
) -> dict:
    record = _base(tautomerize, source)
    try:
        from rdkit import Chem
    except (OSError, ImportError):
        Chem = None
    rows = list(candidates)
    record["attempted"] = len(rows)
    if not rows:
        return {**record, "blocker": "no_candidates"}
    names = [row[0] if row and row[0] else (f"sdf_{i}" if sdf else "") for i, row in enumerate(rows, 1)]
    counts = Counter(names)
    graphs = {}
    for row, name in zip(rows, names, strict=True):
        item = {"candidate_id": name}
        record["inputs"].append(item)
        if len(row) != (3 if sdf else 2):
            item["blocker"] = "invalid_candidate_table_width"
            continue
        if not name or counts[name] != 1:
            item["blocker"] = "missing_candidate_id" if not name else "duplicate_candidate_id"
            continue
        if sdf and row[2]:
            item["blocker"] = row[2]
            continue
        if Chem is None:
            item["blocker"] = "rdkit_unavailable"
            continue
        try:
            mol = (
                Chem.MolFromMolBlock(row[1], removeHs=False, strictParsing=True)
                if sdf
                else parse_nominated_smiles(row[1], Chem)
            )
            smiles, order = _identify(mol, Chem)
        except (ValueError, RuntimeError, OverflowError) as error:
            reason = str(error) if type(error) is ValueError else ""
            item["blocker"] = reason if reason.isidentifier() else "invalid_molecular_structure"
            continue
        key = "smiles_sha256_" + hashlib.sha256(smiles.encode("ascii")).hexdigest()
        item.update(compound_id=key, source_atom_indices_by_map=order)
        graphs[key] = smiles
    record["unique_compounds"] = len(graphs)
    if not graphs:
        return {**record, "blocker": "no_usable_candidates"}
    for key, smiles in graphs.items():
        result = _run(smiles, tautomerize)
        record["structures"][key] = {"canonical_input_smiles": smiles, "states": result.pop("states", {})}
        provider = result.pop("provider", None)
        if provider:
            record["provider"] = provider
        result.update(compound_id=key, method=METHOD)
        record["compounds"].append(result)
        n = sum(row.get("status") == "predicted" for row in result["sites"])
        record["predicted_sites"] += n
        record["compounds_with_predictions"] += bool(n)
        record["compounds_without_enumerated_sites"] += result["status"] == "no_sites"
    if all(x["status"] in {"predicted", "no_sites"} for x in record["compounds"]) and all(
        "compound_id" in x for x in record["inputs"]
    ):
        record["status"] = "complete"
    elif record["compounds_with_predictions"] or record["compounds_without_enumerated_sites"]:
        record["status"] = "partial"
    else:
        reasons = {x.get("blocker") for x in record["compounds"]}
        record["blocker"] = reasons.pop() if len(reasons) == 1 else "no_completed_compounds"
    return record


def profile_sdf(path: Path, *, tautomerize: bool = False) -> dict:
    try:
        return profile(read_sdf_candidates(path), tautomerize=tautomerize, source="sdf", sdf=True)
    except (OSError, UnicodeError):
        return {**_base(tautomerize, "sdf"), "blocker": "sdf_input_unreadable"}


def _unique(pairs: list) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate setting")
        result[key] = value
    return result


def screen_run(directory: Path, *, kind: str) -> dict | None:
    manifest = directory / MANIFEST
    if not manifest.exists() and not manifest.is_symlink():
        return None
    try:
        settings = json.loads(manifest.read_text(encoding="utf-8-sig"), object_pairs_hook=_unique)
        if not isinstance(settings, dict) or settings.keys() - {"tautomerize", "input"}:
            raise ValueError("invalid settings")
        mode = settings.get("tautomerize", False)
        selection = settings.get("input", "smiles")
        if type(mode) is not bool or selection not in ("smiles", "sdf"):
            raise ValueError("invalid mode")
    except (ValueError, OSError, RecursionError):
        result = _base(False, "unreadable_manifest")
        result["tautomerize"] = None
        return {**result, "blocker": "invalid_qupkake_manifest"}
    if kind != "ligand":
        return {**_base(mode, selection), "blocker": "ligand_binder_kind_required"}
    if selection == "sdf":
        return profile_sdf(directory / "ligand_candidates.sdf", tautomerize=mode)
    scope, rows, blocker = read_smiles_candidates(directory)
    if blocker or rows is None:
        return {**_base(mode, scope), "blocker": blocker or "ligand_input_missing"}
    return profile(rows, tautomerize=mode, source=scope)
