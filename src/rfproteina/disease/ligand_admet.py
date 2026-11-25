"""Shared bounded CPU execution and attribution for local ADMET-AI endpoints."""

from __future__ import annotations

import base64
import csv
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from contextlib import redirect_stdout
from dataclasses import dataclass
from importlib import metadata
from io import StringIO
from itertools import islice
from numbers import Real
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

from rfproteina.disease import ligand_inputs
from rfproteina.disease.corpus_provenance import declare

if TYPE_CHECKING:
    from admet_ai import ADMETModel
    from pandas import DataFrame
    from rdkit import Chem

BATCH_SIZE = 64
MAX_INPUT_BYTES = 1024 * 1024
MAX_CANDIDATES = 256
_MAX_SMILES = 4096
_MAX_ATOMS = 256
_MAX_ARTIFACT_BYTES = 64 * 1024 * 1024
MAX_RESPONSE_BYTES = 2 * MAX_INPUT_BYTES
_WORKER_TIMEOUT = 600
_CPU_ENV = {
    "CUDA_VISIBLE_DEVICES": "",
    "HIP_VISIBLE_DEVICES": "",
    "ROCR_VISIBLE_DEVICES": "",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "HF_HUB_OFFLINE": "1",
}
_CANDIDATE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
_MEMBERS = tuple(f"model_{i}.pt" for i in range(5))
_PREDICTION_BLOCKERS = frozenset({"missing_provider_row", "duplicate_provider_row", "invalid_provider_result"})
_WORKER_BLOCKERS = frozenset({
    "provider_dependency_unavailable", "trained_model_unavailable", "provider_timeout", "provider_worker_failed",
})


class AdmetError(ValueError):
    """A named reason a candidate or local release could not be used."""


@dataclass(frozen=True)
class Endpoint:
    """Output scale and release definition for a single exposed provider head."""

    method: str
    endpoint: str
    unit: str
    value_field: str
    task_type: str
    manifest_unit: str
    maximum: float
    structure_blocker: str
    input_blocker: str
    resource_blocker: str
    species: str | None = None

    @property
    def ensemble(self) -> str:
        return "admet_" + self.task_type

    @property
    def identity(self) -> dict[str, str]:
        return {"method": self.method, "endpoint": self.endpoint, "unit": self.unit}

    def accepts(self, value: object) -> bool:
        if not isinstance(value, Real) or isinstance(value, bool):
            return False
        try:
            numeric = float(value)
        except (ValueError, OverflowError):
            return False
        return math.isfinite(numeric) and 0 <= numeric <= self.maximum


def verified_assets(
    endpoint: Endpoint, ensemble_dir: Path | None = None,
) -> tuple[metadata.Distribution, list[Path], dict[str, str]]:
    """Check the installed release's endpoint and five ensemble members against its wheel RECORD.

    The RECORD attests consistency with the installed wheel, not the publisher's identity.
    When given a private worker directory, stage the exact bytes checked here for the loader.
    """
    dist = metadata.distribution("admet-ai")
    if dist.version.split(".", 1)[0] != "2":
        raise AdmetError("unsupported_admet_ai_version")
    root = Path(dist.locate_file("admet_ai/resources"))
    ensemble = root / "models" / endpoint.ensemble
    weights = [ensemble / name for name in _MEMBERS]
    if sorted(p.relative_to(ensemble).as_posix() for p in ensemble.rglob("*.pt")) != list(_MEMBERS):
        raise AdmetError(f"incomplete_or_unrecognized_{endpoint.task_type}_ensemble")
    definition = root / "data" / "admet.csv"
    entries = {p.as_posix(): p for p in dist.files or ()}
    remaining = _MAX_ARTIFACT_BYTES
    digests = {}
    definition_bytes = b""
    for path in [definition, *weights]:
        entry = entries.get("admet_ai/resources/" + path.relative_to(root).as_posix())
        if entry is None or entry.hash is None or entry.hash.mode != "sha256" or not entry.size:
            raise AdmetError("installed_artifact_digest_mismatch")
        if entry.size > remaining:
            raise AdmetError(endpoint.resource_blocker)
        if path.is_symlink() or not path.is_file() or path.stat().st_size != entry.size:
            raise AdmetError("installed_artifact_digest_mismatch")
        with path.open("rb") as stream:
            data = stream.read(entry.size + 1)
        digest = hashlib.sha256(data).digest()
        if len(data) != entry.size or base64.urlsafe_b64encode(digest).decode().rstrip("=") != entry.hash.value:
            raise AdmetError("installed_artifact_digest_mismatch")
        remaining -= len(data)
        digests[path.name] = digest.hex()
        if path == definition:
            definition_bytes = data
        elif ensemble_dir is not None:
            (ensemble_dir / path.name).write_bytes(data)
    with StringIO(definition_bytes.decode("utf-8-sig"), newline="") as source:
        reader = csv.DictReader(source, strict=True)
        fields = ["id", "task_type", "units"]
        if endpoint.species is not None:
            fields.extend(("species", "minimum", "maximum"))
        if any((reader.fieldnames or []).count(field) != 1 for field in fields):
            raise AdmetError("endpoint_definition_mismatch")
        rows = [row for row in reader if row.get("id") == endpoint.endpoint]
    if len(rows) != 1 or (rows[0].get("task_type"), rows[0].get("units")) != (
        endpoint.task_type, endpoint.manifest_unit
    ):
        raise AdmetError("endpoint_definition_mismatch")
    if endpoint.species is not None:
        try:
            bounds = (float(rows[0]["minimum"]), float(rows[0]["maximum"]))
        except (ValueError, TypeError) as error:
            raise AdmetError("endpoint_definition_mismatch") from error
        if rows[0]["species"] != endpoint.species or bounds != (0.0, endpoint.maximum):
            raise AdmetError("endpoint_definition_mismatch")
    return dist, weights, digests


def load_model(endpoint: Endpoint) -> tuple[ADMETModel, dict]:
    with tempfile.TemporaryDirectory(prefix="local-admet-") as working:
        directory = Path(working) / endpoint.ensemble
        directory.mkdir()
        dist, weights, digests = verified_assets(endpoint, directory)
        from admet_ai import ADMETModel

        model = ADMETModel(models_dir=working, include_physchem=False, drugbank_path=None, num_workers=0)
    if (
        model.device != "cpu" or len(model.task_lists) != 1 or model.task_lists[0].count(endpoint.endpoint) != 1
        or len(model.model_lists) != 1 or len(model.model_lists[0]) != len(weights)
    ):
        raise AdmetError("provider_ensemble_or_device_mismatch")
    return model, {
        "admet_ai": dist.version,
        "chemprop": metadata.version("chemprop"),
        "rdkit": metadata.version("rdkit"),
        "device": "cpu",
        "ensemble_members": len(weights),
        "verified_sha256": digests,
    }


def values(endpoint: Endpoint, frame: DataFrame, smiles: list[str]) -> dict[str, float | str]:
    if list(frame.columns).count(endpoint.endpoint) != 1:
        raise AdmetError("missing_or_duplicate_endpoint")
    counts = Counter(frame.index)
    if counts.keys() - set(smiles):
        raise AdmetError("unexpected_provider_molecule")
    values = {}
    for key in smiles:
        if not counts[key]:
            values[key] = "missing_provider_row"
        elif counts[key] != 1:
            values[key] = "duplicate_provider_row"
        else:
            value = frame.at[key, endpoint.endpoint]
            values[key] = float(value) if endpoint.accepts(value) else "invalid_provider_result"
    return values


def score_batch(endpoint: Endpoint, model: ADMETModel, batch: list[str]) -> dict[str, float | str]:
    try:
        return values(endpoint, model.predict(batch), batch)
    except AdmetError:
        return dict.fromkeys(batch, "invalid_provider_result")
    except (ValueError, RuntimeError, IndexError, KeyError, OverflowError):
        if len(batch) > 1:
            middle = len(batch) // 2
            return {**score_batch(endpoint, model, batch[:middle]), **score_batch(endpoint, model, batch[middle:])}
        return dict.fromkeys(batch, "invalid_provider_result")


def infer(endpoint: Endpoint, smiles: list[str]) -> dict:
    try:
        model, identity = load_model(endpoint)
    except (ImportError, metadata.PackageNotFoundError):
        return {"blocker": "provider_dependency_unavailable"}
    except AdmetError as error:
        blocker = endpoint.resource_blocker if str(error) == endpoint.resource_blocker else "trained_model_unavailable"
        return {"blocker": blocker}
    except (ValueError, OSError, RuntimeError, AttributeError, csv.Error):
        return {"blocker": "trained_model_unavailable"}
    values = {}
    for start in range(0, len(smiles), BATCH_SIZE):
        values.update(score_batch(endpoint, model, smiles[start : start + BATCH_SIZE]))
    return {"provider": identity, "values": values}


def local_predictions(module: str, smiles: list[str], *, run_process: Callable = subprocess.run) -> dict:
    try:
        run = run_process(
            [sys.executable, "-m", module], input=json.dumps(smiles), text=True, env={**os.environ, **_CPU_ENV},
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True, timeout=_WORKER_TIMEOUT,
        )
        if len(run.stdout) <= MAX_RESPONSE_BYTES:
            response = json.loads(run.stdout)
            if isinstance(response, dict):
                return response
    except subprocess.TimeoutExpired:
        return {"blocker": "provider_timeout"}
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    return {"blocker": "provider_worker_failed"}


def _structure_blocker(mol: Chem.Mol, chem: ModuleType, unsupported: str) -> str | None:
    if not mol.GetNumAtoms():
        return "invalid_nominated_smiles"
    atoms = mol.GetAtoms()
    if (
        mol.GetNumAtoms() > _MAX_ATOMS or len(chem.GetMolFrags(mol)) != 1
        or not any(atom.GetAtomicNum() == 6 for atom in atoms)
        or any(atom.GetAtomicNum() == 0 or atom.GetNumRadicalElectrons() for atom in atoms)
    ):
        return unsupported
    return None


def molecular_identity(endpoint: Endpoint, smiles: str, chem: ModuleType) -> tuple[Chem.Mol, str]:
    return molecular_graph(smiles, chem, unsupported=endpoint.structure_blocker)


def molecular_graph(smiles: str, chem: ModuleType, *, unsupported: str) -> tuple[Chem.Mol, str]:
    """Keep the submitted charge and stereochemistry. Never silently choose a mixture fragment."""
    try:
        mol = ligand_inputs.parse_nominated_smiles(smiles, chem, remove_hs=True) if len(smiles) <= _MAX_SMILES else None
    except (ValueError, RuntimeError, OverflowError) as error:
        raise AdmetError("invalid_nominated_smiles") from error
    if mol is None:
        raise AdmetError("invalid_nominated_smiles")
    if blocker := _structure_blocker(mol, chem, unsupported):
        raise AdmetError(blocker)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return mol, chem.MolToSmiles(mol, isomericSmiles=True)


def untested_record(endpoint: Endpoint, scope: str, attempted: int = 0) -> dict:
    return {
        **endpoint.identity, "scope": scope, "status": "untested",
        "attempted": attempted, "unique_compounds": 0, "unique_predicted": 0, "inputs": [], "compounds": [],
        "corpus_provenance": declare(endpoint.method, against=endpoint.endpoint),
    }


def valid_candidate_id(name: str) -> bool:
    """Require a short, stable identifier that can be represented safely in reports."""
    return bool(_CANDIDATE_ID.fullmatch(name))


def prepare_nominations(
    candidates: Iterable[Sequence[str]],
    report: dict,
    *,
    identify: Callable[[str, ModuleType], tuple[Chem.Mol, str]],
    limit_blocker: str,
) -> dict[str, str]:
    """Attach validated nominations and return private model inputs keyed to opaque IDs."""
    rows = list(islice(candidates, MAX_CANDIDATES + 1))
    if len(rows) > MAX_CANDIDATES:
        report["blocker"] = limit_blocker
        return {}
    report["attempted"] = len(rows)
    if not rows:
        report["blocker"] = "no_candidates"
        return {}
    try:
        from rdkit import Chem, rdBase
    except (ImportError, OSError):
        Chem = None
    counts = Counter(row[0] for row in rows if row)
    keys: dict[str, str] = {}
    for row in rows:
        name = row[0] if row else ""
        valid_name = valid_candidate_id(name)
        item = {"candidate_id": name if valid_name else ""}
        report["inputs"].append(item)
        if len(row) != 2:
            item["blocker"] = "invalid_candidate_table_width"
        elif not name:
            item["blocker"] = "missing_candidate_id"
        elif not valid_name:
            item["blocker"] = "invalid_or_duplicate_nomination"
        elif counts[name] != 1:
            item["blocker"] = "duplicate_candidate_id"
        elif Chem is None:
            item["blocker"] = "rdkit_unavailable"
        else:
            try:
                with rdBase.BlockLogs():
                    _, key = identify(row[1], Chem)
            except AdmetError as error:
                item["blocker"] = str(error)
            else:
                if key not in keys:
                    keys[key] = "smiles_sha256_" + hashlib.sha256(key.encode("ascii")).hexdigest()
                item["compound_id"] = keys[key]
    if not keys:
        report["blocker"] = "no_usable_candidates"
    report["unique_compounds"] = len(keys)
    return keys


def profile(
    endpoint: Endpoint, candidates: Iterable[Sequence[str]], *, scope: str, predict: Callable[[list[str]], dict],
) -> dict:
    """Predict once per canonical graph and attribute the result to every accepted nomination."""
    result = untested_record(endpoint, scope)
    keys = prepare_nominations(
        candidates, result, identify=lambda smiles, chem: molecular_identity(endpoint, smiles, chem),
        limit_blocker=endpoint.input_blocker,
    )
    if not keys:
        return result

    prediction = predict(list(keys))
    report_blocker = prediction.get("blocker")
    if not isinstance(report_blocker, str) or report_blocker not in (_WORKER_BLOCKERS | {endpoint.resource_blocker}):
        report_blocker = None
    values = prediction.get("values", {})
    if not isinstance(values, dict) or values.keys() - keys.keys():
        report_blocker, values = "provider_worker_failed", {}
    if isinstance(prediction.get("provider"), dict) and not report_blocker:
        result["provider"] = prediction["provider"]
    for key, compound_id in keys.items():
        item = {"compound_id": compound_id, **endpoint.identity}
        value = values.get(key, "missing_provider_row")
        if not report_blocker and endpoint.accepts(value):
            item.update(status="predicted")
            item[endpoint.value_field] = float(value)
            result["unique_predicted"] += 1
        else:
            code = value if isinstance(value, str) and value in _PREDICTION_BLOCKERS else "invalid_provider_result"
            item.update(status="untested", blocker=report_blocker or code)
        result["compounds"].append(item)
    if result["unique_predicted"]:
        complete = result["unique_predicted"] == len(keys) and all("compound_id" in r for r in result["inputs"])
        result["status"] = "predicted" if complete else "partial"
    else:
        result["blocker"] = report_blocker or "no_attributable_prediction"
    return result


def screen_inputs(
    endpoint: Endpoint, directory: Path, candidate: str | None,
) -> tuple[str, list[Sequence[str]], str | None]:
    return bounded_inputs(directory, candidate, limit_blocker=endpoint.input_blocker)


def bounded_inputs(
    directory: Path, candidate: str | None, *, limit_blocker: str,
) -> tuple[str, list[Sequence[str]], str | None]:
    """Use the same bounded source and candidate scope for plan and execution."""
    scope, rows, blocker = ligand_inputs.bounded_smiles_candidates(
        directory, max_bytes=MAX_INPUT_BYTES, max_rows=MAX_CANDIDATES
    )
    if blocker == "ligand_input_exceeds_interactive_size":
        blocker = limit_blocker
    if not blocker and candidate is not None and scope != "candidate_table":
        blocker = "candidate_requires_named_nomination"
    return scope, list(ligand_inputs.nominated_rows(rows, candidate)), blocker


def screen_run(
    endpoint: Endpoint, directory: Path, *, candidate: str | None, predict: Callable[[list[str]], dict],
) -> dict:
    """Score a finished run or one nominated candidate without changing the run's verdict."""
    scope, rows, blocker = screen_inputs(endpoint, directory, candidate)
    if blocker:
        return {**untested_record(endpoint, scope), "blocker": blocker}
    return profile(endpoint, rows, scope=scope, predict=predict)


def run_worker(endpoint: Endpoint) -> None:
    run_worker_request(lambda request: infer(endpoint, request))


def run_worker_request(predict: Callable[[list[str]], dict]) -> None:
    """Read one bounded private request and emit one JSON response, with devices hidden."""
    os.environ.update(_CPU_ENV)
    raw = sys.stdin.read(MAX_RESPONSE_BYTES + 1)
    try:
        request = json.loads(raw) if len(raw) <= MAX_RESPONSE_BYTES else None
    except (ValueError, RecursionError):
        request = None
    valid = (
        isinstance(request, list) and 0 < len(request) <= MAX_CANDIDATES
        and all(isinstance(key, str) and 0 < len(key) <= _MAX_SMILES for key in request)
    )
    with redirect_stdout(sys.stderr):
        response = predict(request) if valid else {"blocker": "provider_worker_failed"}
    print(json.dumps(response, allow_nan=False))
