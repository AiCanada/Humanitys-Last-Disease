"""Rank possible human metabolites of explicitly nominated small-molecule parents."""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from rfproteina.disease.corpus_provenance import declare
from rfproteina.disease.ligand_inputs import (
    manifest_options,
    nominated_inputs,
    parse_nominated_smiles,
    read_sdf_candidates,
)
from rfproteina.disease.sygma_engine import ENDPOINT, METHOD, RULE_HASHES, ScreenError, scenario_valid, structure_id

if TYPE_CHECKING:
    from rdkit import Chem

MANIFEST = "sygma.json"
_INCHIKEY = re.compile(r"[A-Z]{14}-[A-Z]{10}-[A-Z]")


def _base(source: str, phase1: int | None, phase2: int | None) -> dict:
    return {
        "method": METHOD,
        "endpoint": ENDPOINT,
        "status": "untested",
        "source": source,
        "scenario": {"phase1_cycles": phase1, "phase2_cycles": phase2},
        "score_definition": "Product of empirical rule scores along the best permitted pathway. Ranks are per parent.",
        "fragment_filter": "More than 15% of the parent heavy atoms retained by the reported pathway.",
        "attempted": 0,
        "unique_parents": 0,
        "screened_parents": 0,
        "parents_without_products": 0,
        "ranked_products": 0,
        "inputs": [],
        "parents": [],
        "structures": {},
        "corpus_provenance": declare(METHOD, against=ENDPOINT),
    }


def _parent_smiles(mol: Chem.Mol | None) -> str:
    from rdkit import Chem

    if mol is None or not mol.GetNumAtoms():
        raise ScreenError("invalid_molecular_structure")
    if len(Chem.GetMolFrags(mol)) != 1:
        raise ScreenError("disconnected_structure_requires_nominated_parent")
    if any(a.GetAtomicNum() == 0 or a.HasQuery() for a in mol.GetAtoms()) or any(b.HasQuery() for b in mol.GetBonds()):
        raise ScreenError("query_or_dummy_atoms_unsupported")
    if any(a.GetNumRadicalElectrons() for a in mol.GetAtoms()):
        raise ScreenError("open_shell_parent_unsupported")
    if mol.GetStereoGroups():
        raise ScreenError("enhanced_stereochemistry_unsupported")
    if not any(a.GetAtomicNum() == 6 for a in mol.GetAtoms()):
        raise ScreenError("organic_parent_required")
    if mol.GetNumHeavyAtoms() > 256:
        raise ScreenError("sygma_parent_atom_budget")
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(Chem.RemoveHs(mol), isomericSmiles=True)


def _valid_report(report: dict, parent_id: str, phase1: int, phase2: int) -> bool:
    if report.get("compound_id") != parent_id or not isinstance(report.get("products"), list):
        return False
    rows = report["products"]
    if report.get("status") == "untested":
        return not rows and isinstance(report.get("blocker"), str) and bool(report["blocker"])
    provider, structures = report.get("provider"), report.get("structures")
    if not isinstance(provider, dict) or provider.get("rule_sha256") != RULE_HASHES or not isinstance(structures, dict):
        return False
    if (
        provider.get("rule_release") != "1.1.0"
        or provider.get("device") != "cpu"
        or provider.get("rules_per_phase") != {"phase1": 148, "phase2": 27}
    ):
        return False
    if parent_id not in structures or any(
        not isinstance(s, str) or k != structure_id(s) for k, s in structures.items()
    ):
        return False
    if report.get("status") != ("ranked" if rows else "no_products"):
        return False
    atom_count = report.get("parent_heavy_atoms")
    if type(atom_count) is not int or atom_count < 1:
        return False
    seen = {parent_id}
    previous = 1.0
    rank = 0
    for idx, row in enumerate(rows, 1):
        if not isinstance(row, dict) or row.get("method") != METHOD:
            return False
        key = row.get("product_id")
        score = row.get("path_score")
        if not isinstance(key, str) or key not in structures or key in seen:
            return False
        retained, mass = row.get("parent_heavy_atoms_retained"), row.get("exact_mass_da")
        if type(retained) is not int or retained > atom_count or 100 * retained <= 15 * atom_count:
            return False
        if type(mass) not in (int, float) or not math.isfinite(mass) or mass <= 0:
            return False
        if not isinstance(row.get("inchikey"), str) or not _INCHIKEY.fullmatch(row["inchikey"]):
            return False
        if type(row.get("unspecified_stereoelements")) is not int or row["unspecified_stereoelements"] < 0:
            return False
        if type(row.get("formal_charge")) is not int or not isinstance(row.get("molecular_formula"), str):
            return False
        if type(score) not in (int, float) or not math.isfinite(score) or not 0 < score <= previous:
            return False
        if idx == 1 or score != previous:
            rank = idx
        if type(row.get("rank")) is not int or row["rank"] != rank:
            return False
        seen.add(key)
        path = row.get("pathway")
        if not isinstance(path, list) or not 1 <= len(path) <= phase1 + phase2:
            return False
        phases = []
        value = 1.0
        for step in path:
            if (
                not isinstance(step, dict)
                or not isinstance(step.get("product_id"), str)
                or step["product_id"] not in structures
            ):
                return False
            weight = step.get("empirical_rule_score")
            if type(step.get("rule_line")) is not int or step["rule_line"] < 1 or not isinstance(step.get("rule"), str):
                return False
            if type(weight) not in (int, float) or not 0 < weight <= 1 or step.get("phase") not in ("phase1", "phase2"):
                return False
            phases.append(step["phase"])
            value *= weight
        if phases != sorted(phases) or phases.count("phase1") > phase1 or phases.count("phase2") > phase2:
            return False
        if path[-1]["product_id"] != key or not math.isclose(score, value, rel_tol=1e-12):
            return False
        previous = score
    return True


def _run(smiles: str, phase1: int, phase2: int) -> dict:
    key = structure_id(smiles)
    failure = {"compound_id": key, "status": "untested", "products": []}
    try:
        process = subprocess.run(
            [sys.executable, "-I", str(Path(__file__).with_name("sygma_engine.py").resolve())],
            input=json.dumps({"smiles": smiles, "phase1": phase1, "phase2": phase2}),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=120,
            check=True,
            env={
                **os.environ,
                "CUDA_VISIBLE_DEVICES": "",
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
            },
        )
        report = json.loads(process.stdout)
        if not isinstance(report, dict) or not _valid_report(report, key, phase1, phase2):
            return {**failure, "blocker": "sygma_invalid_worker_report"}
        return report
    except subprocess.TimeoutExpired:
        return {**failure, "blocker": "sygma_parent_timeout"}
    except (OSError, subprocess.SubprocessError, ValueError, OverflowError):
        return {**failure, "blocker": "sygma_worker_failed"}


def prepare_parents(rows: Sequence[Sequence[str]], *, sdf: bool) -> tuple[list[dict], dict[str, str]]:
    """Validate nominations and identify distinct parent graphs without enumerating metabolites."""
    try:
        from rdkit import Chem
    except (ImportError, OSError):
        Chem = None
    names = [r[0] if r and r[0] else (f"sdf_{i}" if sdf else "") for i, r in enumerate(rows, 1)]
    counts = Counter(names)
    inputs = []
    parents = {}
    for name, row in zip(names, rows, strict=True):
        item = {"candidate_id": name}
        inputs.append(item)
        try:
            if len(row) != (3 if sdf else 2):
                raise ScreenError("invalid_candidate_table_width")
            if not name or counts[name] != 1:
                raise ScreenError("duplicate_candidate_id" if name else "missing_candidate_id")
            if sdf and row[2]:
                raise ScreenError(row[2])
            if Chem is None:
                raise ScreenError("rdkit_unavailable")
            mol = (
                Chem.MolFromMolBlock(row[1], removeHs=False, strictParsing=True)
                if sdf
                else parse_nominated_smiles(row[1], Chem)
            )
            smiles = _parent_smiles(mol)
        except ScreenError as error:
            item["blocker"] = str(error)
            continue
        except (ValueError, RuntimeError, OverflowError):
            item["blocker"] = "invalid_molecular_structure"
            continue
        key = structure_id(smiles)
        item["compound_id"] = key
        parents[key] = smiles
    return inputs, parents


def profile(
    candidates: Iterable[Sequence[str]], *, phase1: int = 1, phase2: int = 1, source: str = "smiles", sdf: bool = False
) -> dict:
    report = _base(source, phase1, phase2)
    rows = list(candidates)
    report["attempted"] = len(rows)
    if not scenario_valid(phase1, phase2):
        return {**report, "blocker": "invalid_sygma_scenario"}
    if not rows:
        return {**report, "blocker": "no_candidates"}
    inputs, parents = prepare_parents(rows, sdf=sdf)
    report.update(inputs=inputs, unique_parents=len(parents))
    if not parents:
        return {**report, "blocker": "no_usable_candidates"}
    for key, smiles in parents.items():
        row = _run(smiles, phase1, phase2)
        report["structures"][key] = smiles
        report["structures"].update(row.pop("structures", {}))
        provider = row.pop("provider", None)
        if provider:
            report["provider"] = provider
        row["method"] = METHOD
        report["parents"].append(row)
        report["screened_parents"] += row["status"] in ("ranked", "no_products")
        report["parents_without_products"] += row["status"] == "no_products"
        report["ranked_products"] += len(row["products"])
    if all("compound_id" in v for v in report["inputs"]) and report["screened_parents"] == len(parents):
        report["status"] = "complete"
    elif report["screened_parents"]:
        report["status"] = "partial"
    else:
        reasons = {row["blocker"] for row in report["parents"]}
        report["blocker"] = reasons.pop() if len(reasons) == 1 else "no_completed_parents"
    return report


def profile_sdf(path: Path, *, phase1: int = 1, phase2: int = 1) -> dict:
    try:
        return profile(read_sdf_candidates(path), source="sdf", sdf=True, phase1=phase1, phase2=phase2)
    except OSError:
        return {**_base("sdf", phase1, phase2), "blocker": "sdf_input_unreadable"}


def settings_for(directory: Path) -> tuple[int, int, str, bool]:
    options, configured = manifest_options(directory, MANIFEST)
    phase1, phase2 = options.get("phase1_cycles", 1), options.get("phase2_cycles", 1)
    if options.keys() - {"input", "phase1_cycles", "phase2_cycles"} or not scenario_valid(phase1, phase2):
        raise ValueError("invalid_sygma_manifest")
    return phase1, phase2, options.get("input", "smiles"), configured


def screen_run(directory: Path, *, kind: str, candidate: str | None = None, use_defaults: bool = False) -> dict | None:
    try:
        p1, p2, source, configured = settings_for(directory)
    except (OSError, ValueError, RecursionError):
        return {**_base("unreadable_manifest", None, None), "blocker": "invalid_sygma_manifest"}
    if not configured and not use_defaults:
        return None
    if kind != "ligand":
        return {**_base(source, p1, p2), "blocker": "ligand_binder_kind_required"}
    scope, rows, reason = nominated_inputs(directory, source, candidate)
    if reason:
        return {**_base(scope, p1, p2), "blocker": reason}
    return profile(rows, source=scope, sdf=source == "sdf", phase1=p1, phase2=p2)
