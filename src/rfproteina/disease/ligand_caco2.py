"""Advisory native Caco-2 permeability and efflux scores from the pinned AZ release."""

from __future__ import annotations

import hashlib
import math
import os
import pickle
import stat
from collections.abc import Iterable, Sequence
from importlib import metadata, util
from io import BytesIO
from numbers import Real
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

from rfproteina.disease import ligand_admet as admet
from rfproteina.disease.corpus_provenance import declare

if TYPE_CHECKING:
    from chemprop.models import MPNN
    from rdkit import Chem

METHOD_ID = "ligand_caco2"
METHOD = "az_gnn_mtl_caco2_v1"
SCALE = "native_model_score"
RELEASE = "zenodo:16948542"
CHECKPOINT_BYTES = 1_283_341
CHECKPOINT_SHA256 = "c979bd0dab17a324e2c0c52c3da0bbe25d93415b7c6b49d62f0e01f6f2a487de"
_CHEMPROP_VERSION = "2.3.1"  # Runtime qualified against the published Chemprop 2.1.0 artifact.
_RELEASE_OUTPUTS = ("Caco2ER", "Caco2Papp", "MDCKER", "NIHMDCKER")
HEADS = (("efflux_ratio", _RELEASE_OUTPUTS[0]), ("apparent_permeability", _RELEASE_OUTPUTS[1]))
BATCH_SIZE = admet.BATCH_SIZE
MAX_INPUT_BYTES = admet.MAX_INPUT_BYTES
MAX_CANDIDATES = admet.MAX_CANDIDATES
valid_candidate_id = admet.valid_candidate_id
CacoError = admet.AdmetError
_PROVIDER_BLOCKERS = frozenset({
    "caco2_checkpoint_missing", "caco2_checkpoint_invalid", "caco2_runtime_unavailable",
    "caco2_output_contract_mismatch", "provider_timeout", "provider_worker_failed",
})
_HEAD_BLOCKERS = frozenset({"invalid_provider_result", "missing_provider_row", "caco2_output_contract_mismatch"})


def _checkpoint_path() -> Path | None:
    configured = os.environ.get("RFD4_CACO2_CHECKPOINT")
    if configured:
        return Path(configured)
    root = os.environ.get("DATA_PATH")
    return Path(root) / "caco2" / "az_gnn_mtl_v1.pt" if root else None


def verified_checkpoint() -> bytes:
    """Read only the pinned bytes from a regular local file, never fetching or trusting arbitrary weights."""
    path = _checkpoint_path()
    if path is None:
        raise CacoError("caco2_checkpoint_missing")
    try:
        original = path.lstat()
        if not stat.S_ISREG(original.st_mode) or original.st_size != CHECKPOINT_BYTES:
            raise CacoError("caco2_checkpoint_invalid")
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0)
        with os.fdopen(os.open(path, flags), "rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or not os.path.samestat(original, info):
                raise CacoError("caco2_checkpoint_invalid")
            data = stream.read(CHECKPOINT_BYTES + 1)
    except FileNotFoundError as error:
        raise CacoError("caco2_checkpoint_missing") from error
    except OSError as error:
        raise CacoError("caco2_checkpoint_invalid") from error
    if len(data) != CHECKPOINT_BYTES or hashlib.sha256(data).hexdigest() != CHECKPOINT_SHA256:
        raise CacoError("caco2_checkpoint_invalid")
    return data


def _runtime_version() -> str:
    version = metadata.version("chemprop")
    if version != _CHEMPROP_VERSION:
        raise CacoError("caco2_runtime_unavailable")
    return version


def asset_available() -> bool:
    """Check dependencies and exact asset bytes without importing a model runtime."""
    try:
        _runtime_version()
        return all(util.find_spec(name) is not None for name in ("rdkit", "chemprop", "torch", "lightning")) and bool(
            verified_checkpoint()
        )
    except (ImportError, OSError, ValueError):
        return False


def _load_model() -> tuple[MPNN, dict]:
    # Load only bytes already checked above. Do not use Chemprop's unrestricted pickle loader.
    raw = verified_checkpoint()
    version = _runtime_version()
    import torch
    from chemprop.models import MPNN
    from chemprop.nn import BondMessagePassing, MeanAggregation, RegressionFFN, UnscaleTransform
    from chemprop.nn.utils import Activation

    # Older Lightning serializes the hyperparameter mappings as AttributeDict.
    # Decode that mapping as a plain dict: weights_only does not accept SETITEMS
    # into the subclass, and inference does not need its attribute behavior.
    classes = [
        BondMessagePassing, MeanAggregation, RegressionFFN, UnscaleTransform, Activation,
        (dict, "lightning.fabric.utilities.data.AttributeDict"),
    ]
    with torch.serialization.safe_globals(classes):
        saved = torch.load(BytesIO(raw), weights_only=True, map_location="cpu")
    hparams = dict(saved["hyper_parameters"])
    state = saved["state_dict"]
    transform = hparams["predictor"]["output_transform"]
    if (
        saved.get("output_columns") is not None or hparams["predictor"]["n_tasks"] != len(_RELEASE_OUTPUTS)
        or type(transform) is not UnscaleTransform or hparams.get("X_d_transform") is not None
        or hparams["predictor"].get("criterion") is not None or hparams.get("metrics") is not None
    ):
        raise CacoError("caco2_output_contract_mismatch")
    submodules = (("message_passing", BondMessagePassing), ("agg", MeanAggregation), ("predictor", RegressionFFN))
    for key, expected in submodules:
        arguments = dict(hparams[key])
        if arguments.pop("cls") is not expected:
            raise CacoError("caco2_output_contract_mismatch")
        hparams[key] = expected(**arguments)
    model = MPNN(**hparams)
    model.load_state_dict(state, strict=True)
    model.cpu().eval()
    return model, {
        "chemprop": version, "torch": metadata.version("torch"), "rdkit": metadata.version("rdkit"),
        "device": "cpu", "checkpoint_sha256": CHECKPOINT_SHA256, "release": RELEASE,
    }


def valid_score(value: object) -> bool:
    """The raw scale is signed. Values on it are not physical ratios or permeabilities."""
    if not isinstance(value, Real) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except (OverflowError, ValueError):
        return False


def _native_values(rows: Iterable[Sequence[object]], smiles: list[str]) -> dict[str, list[float | str]]:
    matrix = list(rows)
    if len(matrix) != len(smiles) or any(
        not isinstance(row, Sequence) or isinstance(row, str | bytes) or len(row) != len(_RELEASE_OUTPUTS)
        for row in matrix
    ):
        raise CacoError("caco2_output_contract_mismatch")
    # Columns 2 and 3 are distinct MDCK assays and are deliberately not exposed here.
    return {
        key: [float(value) if valid_score(value) else "invalid_provider_result" for value in row[:len(HEADS)]]
        for key, row in zip(smiles, matrix, strict=True)
    }


def _predict_rows(model: MPNN, smiles: list[str]) -> list[list[float]]:
    import torch
    from chemprop.data import MoleculeDatapoint, MoleculeDataset, build_dataloader

    dataset = MoleculeDataset([MoleculeDatapoint.from_smi(smi) for smi in smiles])
    loader = build_dataloader(dataset, batch_size=BATCH_SIZE, num_workers=0, shuffle=False, drop_last=False)
    rows = []
    with torch.inference_mode():
        for index, batch in enumerate(loader):
            result = model.predict_step(batch, index)
            if result.ndim != 2 or result.shape[1] != len(_RELEASE_OUTPUTS):
                raise CacoError("caco2_output_contract_mismatch")
            rows.extend(result.detach().cpu().tolist())
    return rows


def _score_batch(model: MPNN, smiles: list[str]) -> dict[str, list[float | str]]:
    try:
        return _native_values(_predict_rows(model, smiles), smiles)
    except CacoError:
        return {key: ["caco2_output_contract_mismatch"] * len(HEADS) for key in smiles}
    except (ValueError, RuntimeError, IndexError, KeyError, OverflowError):
        if len(smiles) > 1:
            split = len(smiles) // 2
            return {**_score_batch(model, smiles[:split]), **_score_batch(model, smiles[split:])}
        return {key: ["invalid_provider_result"] * len(HEADS) for key in smiles}


def _infer(smiles: list[str]) -> dict:
    try:
        model, provider = _load_model()
    except CacoError as error:
        return {"blocker": str(error)}
    except (
        OSError, ImportError, RuntimeError, AttributeError, TypeError, ValueError, KeyError, pickle.UnpicklingError,
    ):
        return {"blocker": "caco2_runtime_unavailable"}
    values = {}
    for offset in range(0, len(smiles), BATCH_SIZE):
        values.update(_score_batch(model, smiles[offset : offset + BATCH_SIZE]))
    return {"provider": provider, "values": values}


def _local_predictions(smiles: list[str]) -> dict:
    return admet.local_predictions(__name__, smiles)


def molecular_identity(smiles: str, chem: ModuleType) -> tuple[Chem.Mol, str]:
    """Keep the submitted graph and reject mixed or unsupported molecular forms."""
    return admet.molecular_graph(smiles, chem, unsupported="unsupported_caco2_structure")


def _report(scope: str) -> dict:
    return {
        "method": METHOD, "scale": SCALE, "scope": scope, "status": "untested", "attempted": 0,
        "unique_compounds": 0, "unique_predicted": 0, "unique_complete": 0, "inputs": [], "compounds": [],
        "corpus_provenance": declare(METHOD),
    }


def _compound(compound_id: str, scores: Sequence[object], blocker: str | None) -> tuple[dict, int]:
    heads = {}
    count = 0
    for (name, endpoint), value in zip(HEADS, scores, strict=True):
        result = {"endpoint": endpoint, "scale": SCALE, "status": "untested"}
        if not blocker and valid_score(value):
            result.update(status="predicted", model_score=float(value))
            count += 1
        else:
            code = value if isinstance(value, str) and value in _HEAD_BLOCKERS else "invalid_provider_result"
            result["blocker"] = blocker or code
        heads[name] = result
    status = "predicted" if count == len(HEADS) else "partial" if count else "untested"
    return {"method": METHOD, "compound_id": compound_id, "status": status, "heads": heads}, count


def profile(candidates: Iterable[Sequence[str]], *, scope: str) -> dict:
    """Predict once per canonical graph, retaining coverage independently for both Caco-2 heads."""
    report = _report(scope)
    keys = admet.prepare_nominations(
        candidates, report, identify=molecular_identity, limit_blocker="caco2_input_limit_exceeded",
    )
    if not keys:
        return report

    response = _local_predictions(list(keys))
    blocker = response.get("blocker")
    if blocker is not None and (not isinstance(blocker, str) or blocker not in _PROVIDER_BLOCKERS):
        blocker = "provider_worker_failed"
    values = response.get("values", {})
    if not isinstance(values, dict) or values.keys() - keys.keys():
        blocker, values = "provider_worker_failed", {}
    if isinstance(response.get("provider"), dict) and not blocker:
        report["provider"] = response["provider"]
    for key, compound_id in keys.items():
        scores = values.get(key, ["missing_provider_row"] * len(HEADS))
        if not isinstance(scores, list) or len(scores) != len(HEADS):
            scores = ["invalid_provider_result"] * len(HEADS)
        row, count = _compound(compound_id, scores, blocker)
        report["compounds"].append(row)
        report["unique_predicted"] += count > 0
        report["unique_complete"] += count == len(HEADS)
    if report["unique_predicted"]:
        complete = report["unique_complete"] == len(keys) and all("compound_id" in r for r in report["inputs"])
        report["status"] = "predicted" if complete else "partial"
    else:
        report["blocker"] = blocker or "no_attributable_prediction"
    return report


def screen_inputs(directory: Path, candidate: str | None) -> tuple[str, list[Sequence[str]], str | None]:
    """Use the same bounded nomination source for both the planner and runtime."""
    return admet.bounded_inputs(directory, candidate, limit_blocker="caco2_input_limit_exceeded")


def screen_run(directory: Path, *, candidate: str | None = None) -> dict:
    """Expose only advisory native model scores for selected ligand nominations."""
    scope, rows, blocker = screen_inputs(directory, candidate)
    if blocker:
        return {**_report(scope), "blocker": blocker}
    return profile(rows, scope=scope)


if __name__ == "__main__":
    admet.run_worker_request(_infer)
