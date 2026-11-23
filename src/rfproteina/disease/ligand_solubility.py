"""Predict aqueous log10 molar solubility for explicitly supplied small molecules."""

from __future__ import annotations

import base64
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from collections.abc import Iterable, Sequence
from contextlib import redirect_stdout
from importlib import metadata
from io import StringIO
from numbers import Real
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from admet_ai import ADMETModel
    from pandas import DataFrame
    from rdkit import Chem

from rfproteina.disease.corpus_provenance import declare
from rfproteina.disease.ligand_inputs import parse_nominated_smiles

METHOD = "admet_ai_aqsoldb_v2"
ENDPOINT = "Solubility_AqSolDB"
UNIT = "log10(mol/L)"
BATCH_SIZE = 64


class SolubilityError(ValueError):
    """A named reason the declared measurement could not be made."""


def installed_assets() -> tuple[metadata.Distribution, Path, list[Path]]:
    """Locate the run's endpoint definition and all five local regression members."""
    dist = metadata.distribution("admet-ai")
    if dist.version.split(".")[0] != "2":
        raise SolubilityError("unsupported_admet_ai_version")
    root = Path(dist.locate_file("admet_ai/resources"))
    reg = root / "models" / "admet_regression"
    weights = sorted(reg.rglob("*.pt"))
    if [p.relative_to(reg).as_posix() for p in weights] != [f"model_{i}.pt" for i in range(5)]:
        raise SolubilityError("incomplete_or_unrecognized_regression_ensemble")
    definition = root / "data" / "admet.csv"
    if not definition.is_file() or any(not p.is_file() or p.stat().st_size == 0 for p in weights):
        raise SolubilityError("installed_artifact_missing")
    return dist, definition, weights


def verified_assets(ensemble_dir: Path | None = None) -> tuple[metadata.Distribution, list[Path], dict[str, str]]:
    """Verify the released endpoint and regression members, optionally staging only those members for inference."""
    dist, definition, weights = installed_assets()
    root = definition.parent.parent
    manifest = {p.as_posix(): p for p in dist.files or ()}
    digests = {}
    definition_bytes = b""
    for path in [definition, *weights]:
        entry = manifest.get("admet_ai/resources/" + path.relative_to(root).as_posix())
        if entry is None or not entry.hash or entry.hash.mode != "sha256":
            raise SolubilityError("installed_artifact_digest_mismatch")
        if entry.size is not None and path.stat().st_size != entry.size:
            raise SolubilityError("installed_artifact_digest_mismatch")
        if path == definition:
            definition_bytes = path.read_bytes()
            digest = hashlib.sha256(definition_bytes).digest()
        else:
            checked = path
            if ensemble_dir is not None:
                checked = ensemble_dir / path.name
                shutil.copyfile(path, checked)
            with checked.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").digest()
        if base64.urlsafe_b64encode(digest).decode().rstrip("=") != entry.hash.value:
            raise SolubilityError("installed_artifact_digest_mismatch")
        digests[path.name] = digest.hex()
    with StringIO(definition_bytes.decode("utf-8-sig"), newline="") as source:
        reader = csv.DictReader(source, strict=True)
        if any((reader.fieldnames or []).count(name) != 1 for name in ("id", "task_type", "units")):
            raise SolubilityError("endpoint_definition_mismatch")
        rows = [row for row in reader if row.get("id") == ENDPOINT]
    if len(rows) != 1 or (rows[0].get("task_type"), rows[0].get("units")) != ("regression", "log(mol/L)"):
        raise SolubilityError("endpoint_definition_mismatch")
    return dist, weights, digests


def _model() -> tuple[ADMETModel, dict]:
    # The public constructor loads every ensemble under models_dir. Supply only the checked regression files.
    with tempfile.TemporaryDirectory(prefix="admet-solubility-") as temporary:
        ensemble_dir = Path(temporary) / "admet_regression"
        ensemble_dir.mkdir()
        dist, weights, digests = verified_assets(ensemble_dir)
        from admet_ai import ADMETModel

        model = ADMETModel(models_dir=temporary, include_physchem=False, drugbank_path=None, num_workers=0)
    owners = [i for i, tasks in enumerate(model.task_lists) for task in tasks if task == ENDPOINT]
    if (
        owners != [0] or len(model.model_lists) != 1 or len(model.model_lists[0]) != len(weights)
        or model.device != "cpu"
    ):
        raise SolubilityError("provider_ensemble_or_device_mismatch")
    return model, {
        "admet_ai": dist.version,
        "chemprop": metadata.version("chemprop"),
        "rdkit": metadata.version("rdkit"),
        "device": model.device,
        "loaded_ensembles": len(model.model_lists),
        "regression_members": len(weights),
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
    for start in range(0, len(smiles), BATCH_SIZE):
        batch = smiles[start : start + BATCH_SIZE]
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
            timeout=600 + 60 * math.ceil(len(smiles) / BATCH_SIZE),
        )
        return json.loads(run.stdout)
    except subprocess.TimeoutExpired:
        return {"blocker": "provider_timeout"}
    except (OSError, subprocess.SubprocessError, ValueError):
        return {"blocker": "provider_worker_failed"}


def molecular_identity(smiles: str, chem: ModuleType) -> tuple[Chem.Mol, str]:
    """Return the submitted graph and the canonical key shared by selection and prediction."""
    try:
        mol = parse_nominated_smiles(smiles, chem, remove_hs=True)
    except (ValueError, RuntimeError, OverflowError) as error:
        raise SolubilityError("invalid_smiles") from error
    if mol is None or not mol.GetNumAtoms() or any(a.GetAtomicNum() == 0 for a in mol.GetAtoms()):
        raise SolubilityError("invalid_smiles")
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return mol, chem.MolToSmiles(mol, isomericSmiles=True)


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
            _, key = molecular_identity(row[1], Chem)
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


def screen_run(directory: Path, *, candidate: str | None = None) -> dict | None:
    from rfproteina.disease.ligand_inputs import nominated_rows, read_smiles_candidates

    scope, candidates, blocker = read_smiles_candidates(directory)
    if candidates is None:
        return None
    if candidate is not None and scope != "candidate_table":
        return {**profile([], scope=scope), "blocker": blocker or "candidate_requires_named_nomination"}
    result = profile(nominated_rows(candidates, candidate), scope=scope)
    if blocker:
        result["blocker"] = blocker
    return result


if __name__ == "__main__":
    requested = json.load(sys.stdin)
    with redirect_stdout(sys.stderr):
        response = _infer(requested)
    print(json.dumps(response, allow_nan=False))
