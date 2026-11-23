"""Predict aqueous log10 molar solubility for explicitly supplied small molecules."""

from __future__ import annotations

import base64
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from collections import Counter
from collections.abc import Iterable, Sequence
from contextlib import redirect_stdout
from importlib import metadata
from numbers import Real
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from admet_ai import ADMETModel
    from pandas import DataFrame

from rfproteina.disease.corpus_provenance import declare

METHOD = "admet_ai_aqsoldb_v2"
ENDPOINT = "Solubility_AqSolDB"
UNIT = "log10(mol/L)"
_BATCH = 64


class SolubilityError(ValueError):
    """A named reason the declared measurement could not be made."""


def _model() -> tuple[ADMETModel, dict]:
    """Check the installed release and its five regression members before inference."""
    dist = metadata.distribution("admet-ai")
    if dist.version.split(".")[0] != "2":
        raise SolubilityError("unsupported_admet_ai_version")
    root = Path(dist.locate_file("admet_ai/resources"))
    reg = root / "models" / "admet_regression"
    weights = sorted(reg.rglob("*.pt"))
    if [p.relative_to(reg).as_posix() for p in weights] != [f"model_{i}.pt" for i in range(5)]:
        raise SolubilityError("incomplete_or_unrecognized_regression_ensemble")
    manifest = {str(p): p.hash for p in dist.files or ()}
    definition = root / "data" / "admet.csv"
    digests = {}
    for path in [definition, *weights]:
        entry = manifest.get("admet_ai/resources/" + path.relative_to(root).as_posix())
        digest = hashlib.sha256(path.read_bytes()).digest()
        if not entry or entry.mode != "sha256" or base64.urlsafe_b64encode(digest).decode().rstrip("=") != entry.value:
            raise SolubilityError("installed_artifact_digest_mismatch")
        digests[path.name] = digest.hex()
    with definition.open(encoding="utf-8-sig", newline="") as source:
        rows = [row for row in csv.DictReader(source) if row.get("id") == ENDPOINT]
    if len(rows) != 1 or (rows[0].get("task_type"), rows[0].get("units")) != ("regression", "log(mol/L)"):
        raise SolubilityError("endpoint_definition_mismatch")

    from admet_ai import ADMETModel

    model = ADMETModel(include_physchem=False, drugbank_path=None, num_workers=0)
    owners = [i for i, tasks in enumerate(model.task_lists) for task in tasks if task == ENDPOINT]
    if len(owners) != 1 or len(model.model_lists[owners[0]]) != len(weights) or model.device != "cpu":
        raise SolubilityError("provider_ensemble_or_device_mismatch")
    return model, {
        "admet_ai": dist.version,
        "chemprop": metadata.version("chemprop"),
        "rdkit": metadata.version("rdkit"),
        "device": model.device,
        "verified_sha256": digests,
    }


def _values(frame: DataFrame, smiles: list[str]) -> dict[str, float | str]:
    """Join on molecular identity, since the native provider may omit input rows."""
    if list(frame.columns).count(ENDPOINT) != 1:
        raise SolubilityError("missing_or_duplicate_endpoint")
    counts = Counter(frame.index)
    if counts.keys() - set(smiles):
        raise SolubilityError("unexpected_provider_molecule")
    results = {}
    for key in smiles:
        if counts[key] != 1:
            results[key] = "missing_provider_row" if not counts[key] else "duplicate_provider_row"
            continue
        value = frame.at[key, ENDPOINT]
        if isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value):
            results[key] = float(value)
        else:
            results[key] = "nonfinite_or_nonnumeric_prediction"
    return results


def _blocker(error: Exception) -> str:
    if isinstance(error, SolubilityError):
        return str(error)
    if isinstance(error, metadata.PackageNotFoundError | ImportError):
        return "provider_dependency_unavailable"
    return f"provider_failed_{type(error).__name__}"


def _score_batch(model: ADMETModel, batch: list[str]) -> dict[str, float | str]:
    try:
        return _values(model.predict(batch), batch)
    except SolubilityError as error:
        return dict.fromkeys(batch, str(error))
    except (ValueError, RuntimeError, IndexError, KeyError, OverflowError) as error:
        # A molecule that the provider cannot featurize must not erase its batch mates.
        if len(batch) > 1:
            middle = len(batch) // 2
            return {**_score_batch(model, batch[:middle]), **_score_batch(model, batch[middle:])}
        return dict.fromkeys(batch, _blocker(error))
    except Exception as error:
        return dict.fromkeys(batch, _blocker(error))


def _infer(smiles: list[str]) -> dict:
    try:
        model, identity = _model()
    except Exception as error:
        return {"blocker": _blocker(error)}
    values = {}
    for start in range(0, len(smiles), _BATCH):
        batch = smiles[start : start + _BATCH]
        values.update(_score_batch(model, batch))
    return {"provider": identity, "values": values}


def _local_predictions(smiles: list[str]) -> dict:
    # Lightning writes progress to stdout and selects its device at import/initialization.
    # Keep both out of the host service. The worker reuses one ensemble across all batches.
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "HF_HUB_OFFLINE": "1",
    }
    try:
        run = subprocess.run(
            [sys.executable, "-m", __name__],
            input=json.dumps(smiles),
            text=True,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True,
            timeout=600 + 60 * math.ceil(len(smiles) / _BATCH),
        )
        return json.loads(run.stdout)
    except subprocess.TimeoutExpired:
        return {"blocker": "provider_timeout"}
    except (OSError, subprocess.SubprocessError, ValueError):
        return {"blocker": "provider_worker_failed"}


def _canonical(smiles: str, chem: ModuleType) -> str:
    # Disable RDKit's permissive trailing-name parsing and preserve salt, charge,
    # isotope and stereochemistry. Atom-map numbers label atoms in reactions.
    if not smiles or not smiles.isascii() or not smiles.isprintable() or " " in smiles or "|" in smiles:
        raise SolubilityError("invalid_smiles")
    params = chem.SmilesParserParams()
    params.parseName = params.allowCXSMILES = False
    try:
        mol = chem.MolFromSmiles(smiles, params)
    except (ValueError, RuntimeError, OverflowError) as error:
        raise SolubilityError("invalid_smiles") from error
    if mol is None or not mol.GetNumAtoms() or any(a.GetAtomicNum() == 0 for a in mol.GetAtoms()):
        raise SolubilityError("invalid_smiles")
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return chem.MolToSmiles(mol, isomericSmiles=True)


def profile(candidates: Iterable[Sequence[str]], *, scope: str) -> dict:
    """Return one signed prediction per distinct graph and coverage for every nomination."""
    rows = list(candidates)
    record = {
        "method": METHOD,
        "endpoint": ENDPOINT,
        "unit": UNIT,
        "scope": scope,
        "status": "untested",
        "attempted": len(rows),
        "unique_compounds": 0,
        "unique_predicted": 0,
        "inputs": [],
        "structures": {},
        "compounds": [],
        "corpus_provenance": declare(METHOD, against=ENDPOINT),
    }
    if not rows:
        return {**record, "blocker": "no_candidates"}
    try:
        from rdkit import Chem
    except (ImportError, OSError):
        Chem = None
    counts = Counter(row[0] for row in rows if row)
    keys = {}
    for row in rows:
        name = row[0] if row else ""
        item = {"candidate_id": name}
        record["inputs"].append(item)
        if len(row) != 2:
            item["blocker"] = "invalid_candidate_table_width"
            continue
        if not name or counts[name] != 1:
            item["blocker"] = "missing_candidate_id" if not name else "duplicate_candidate_id"
            continue
        if Chem is None:
            item["blocker"] = "rdkit_unavailable"
            continue
        try:
            key = _canonical(row[1], Chem)
        except SolubilityError as error:
            item["blocker"] = str(error)
            continue
        if key not in keys:
            keys[key] = "smiles_sha256_" + hashlib.sha256(key.encode("ascii")).hexdigest()
        item["compound_id"] = keys[key]
    if not keys:
        return {**record, "blocker": "no_usable_candidates"}
    record["unique_compounds"] = len(keys)
    record["structures"] = {compound_id: key for key, compound_id in keys.items()}
    prediction = _local_predictions(list(keys))
    if "provider" in prediction:
        record["provider"] = prediction["provider"]
    for key, compound_id in keys.items():
        item = {"compound_id": compound_id, "method": METHOD, "endpoint": ENDPOINT, "unit": UNIT}
        value = prediction.get("values", {}).get(key, prediction.get("blocker", "missing_provider_row"))
        if isinstance(value, float) and math.isfinite(value):
            item.update(status="predicted", predicted_log_s=value)
            record["unique_predicted"] += 1
        else:
            item.update(status="untested", blocker=value if isinstance(value, str) else "invalid_provider_result")
        record["compounds"].append(item)
    if record["unique_predicted"]:
        complete = record["unique_predicted"] == len(keys) and all("compound_id" in r for r in record["inputs"])
        record["status"] = "predicted" if complete else "partial"
    else:
        record["blocker"] = prediction.get("blocker", "no_valid_predictions")
    return record


def screen_run(directory: Path) -> dict | None:
    from rfproteina.disease.ligand_inputs import read_smiles_candidates

    scope, candidates, blocker = read_smiles_candidates(directory)
    if candidates is None:
        return None
    result = profile(candidates, scope=scope)
    if blocker:
        result["blocker"] = blocker
    return result


if __name__ == "__main__":
    requested = json.load(sys.stdin)
    with redirect_stdout(sys.stderr):
        response = _infer(requested)
    print(json.dumps(response, allow_nan=False))
