"""Local-only advisory DELPHI PSR scores for explicitly nominated antibody pairs."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import re
import stat
import subprocess
import sys
from collections.abc import Sequence
from contextlib import redirect_stdout
from importlib import metadata
from pathlib import Path

from rfproteina.disease import delphi_inputs
from rfproteina.disease.corpus_provenance import declare
from rfproteina.disease.delphi_inputs import AMINO_ACIDS as AMINO_ACIDS

METHOD_ID = "delphi_psr"
METHOD = "ipi_delphi_onehot_psr_v4"
ENDPOINT = "psr_filter"
UNIT = "probability"
VALUE_FIELD = "predicted_psr_pass_probability"
RELEASE = "zenodo:21823887"
CHECKPOINT_NAME = "FINAL_psr_filter_onehot_transformer_onehot_ipi_psr_trainset.pt"
CHECKPOINT_BYTES = 7_204_447
CHECKPOINT_SHA256 = "5bcd6accfd832c88f582b8206a27b5930f3cbf3a442eba0c59695ebb6f6d8b81"

_CPU_ENV = {
    "CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": "", "ROCR_VISIBLE_DEVICES": "",
    "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "HF_HUB_OFFLINE": "1",
}
_WORKER_CALL = "from rfproteina.disease.delphi_psr import _main; _main()"
_WORKER_LIMIT = 64 * 1024
_WORKER_TIMEOUT = 600
_BATCH_SIZE = 8
_VERSION = re.compile(r"[0-9][0-9A-Za-z.+!_-]{0,63}")
_MODEL_BLOCKERS = frozenset({
    "delphi_checkpoint_missing", "delphi_checkpoint_invalid", "delphi_runtime_unavailable",
    "delphi_inference_failed", "delphi_worker_timeout",
})


class DelPhiError(ValueError):
    """A named asset or model error that can safely reach a screen record."""


def checkpoint_path() -> Path:
    """Resolve only system configuration, never a path supplied with a nomination."""
    if override := os.environ.get("RFD4_DELPHI_PSR_CHECKPOINT"):
        return Path(override)
    if data_root := os.environ.get("DATA_PATH"):
        return Path(data_root) / "delphi" / CHECKPOINT_NAME
    raise DelPhiError("delphi_checkpoint_missing")


def verify_checkpoint(raw: bytes) -> bytes:
    if len(raw) != CHECKPOINT_BYTES or hashlib.sha256(raw).hexdigest() != CHECKPOINT_SHA256:
        raise DelPhiError("delphi_checkpoint_invalid")
    return raw


def verified_checkpoint(path: Path | None = None) -> bytes:
    """Read at most the published size and verify the exact bytes to be loaded."""
    chosen = checkpoint_path() if path is None else path
    try:
        before = chosen.lstat()
        if not stat.S_ISREG(before.st_mode) or before.st_size != CHECKPOINT_BYTES:
            raise DelPhiError("delphi_checkpoint_invalid")
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0)
        with os.fdopen(os.open(chosen, flags), "rb") as file:
            opened = os.fstat(file.fileno())
            if not stat.S_ISREG(opened.st_mode) or not os.path.samestat(before, opened):
                raise DelPhiError("delphi_checkpoint_invalid")
            return verify_checkpoint(file.read(CHECKPOINT_BYTES + 1))
    except FileNotFoundError as error:
        raise DelPhiError("delphi_checkpoint_missing") from error
    except OSError as error:
        raise DelPhiError("delphi_checkpoint_invalid") from error


def asset_available() -> bool:
    """Cheap discovery without importing torch or running antibody numbering."""
    try:
        if not all(importlib.util.find_spec(name) for name in ("torch", "anarcii")):
            return False
        verified_checkpoint()
        return True
    except (ImportError, ValueError):
        return False


def _score(value: object) -> bool:
    return type(value) in (int, float) and 0 <= value <= 1 and math.isfinite(value)


def _local_predictions(pairs: Sequence[delphi_inputs.Pair]) -> dict:
    private = [(pair.candidate_id, pair.vh, pair.vl) for pair in pairs]
    try:
        run = subprocess.run(
            [sys.executable, "-c", _WORKER_CALL], input=json.dumps(private), text=True,
            env={**os.environ, **_CPU_ENV}, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            timeout=_WORKER_TIMEOUT, check=True,
        )
        if len(run.stdout) <= _WORKER_LIMIT:
            response = json.loads(run.stdout)
            if isinstance(response, dict):
                return response
    except subprocess.TimeoutExpired:
        return {"blocker": "delphi_worker_timeout"}
    except (OSError, subprocess.SubprocessError, ValueError, RecursionError):
        pass
    return {"blocker": "delphi_inference_failed"}


def _provider(value: object) -> dict[str, object] | None:
    fingerprints = {"device": "cpu", "release": RELEASE, "checkpoint_sha256": CHECKPOINT_SHA256}
    if not isinstance(value, dict) or any(value.get(key) != item for key, item in fingerprints.items()):
        return None
    versions = {key: value.get(key) for key in ("torch", "anarcii")}
    if any(not isinstance(version, str) or not _VERSION.fullmatch(version) for version in versions.values()):
        return None
    return {**fingerprints, **versions}


def screen_run(directory: Path, *, candidate: str | None = None) -> dict:
    """Score the nominated pairs in a finished run without changing its verdict."""
    record = {
        "method": METHOD, "endpoint": ENDPOINT, "unit": UNIT, "scope": "candidate" if candidate is not None else "run",
        "status": "untested", "attempted": 0, "predicted": 0, "pairs": [],
        "corpus_provenance": declare(METHOD, against=ENDPOINT),
    }
    raw, blocker = delphi_inputs.read_manifest(directory)
    if raw is None:
        return {**record, "blocker": blocker or "delphi_pairs_unreadable"}
    record["input_sha256"] = hashlib.sha256(raw).hexdigest()
    pairs, blocker = delphi_inputs.parse_manifest(raw, candidate)
    if blocker:
        return {**record, "blocker": blocker}
    eligible = [pair for pair in pairs if pair.blocker is None]
    values: dict = {}
    provider = None
    if eligible:
        reply = _local_predictions(eligible)
        valid_ids = {pair.candidate_id for pair in eligible}
        reason = reply.get("blocker")
        if isinstance(reason, str) and reason in _MODEL_BLOCKERS:
            blocker = reason
        elif isinstance(results := reply.get("values"), dict) and results.keys() == valid_ids:
            values = results
            provider = _provider(reply.get("provider"))
            if any(_score(value) for value in values.values()) and provider is None:
                blocker, values = "delphi_inference_failed", {}
        else:
            blocker = "delphi_inference_failed"
    rows = []
    for pair in pairs:
        item = {"candidate_id": pair.candidate_id}
        value = values.get(pair.candidate_id)
        if pair.blocker is not None:
            item.update(status="untested", blocker=pair.blocker)
        elif _score(value):
            item.update(status="predicted", **{VALUE_FIELD: float(value)})
        else:
            if isinstance(value, str) and value in delphi_inputs.NUMBERING_BLOCKERS | _MODEL_BLOCKERS:
                reason = value
            else:
                reason = blocker or "delphi_inference_failed"
            item.update(status="untested", blocker=reason)
        rows.append(item)
    predicted = sum(row["status"] == "predicted" for row in rows)
    record.update(attempted=len(rows), predicted=predicted, pairs=rows)
    if predicted:
        record.update(status="predicted" if predicted == len(rows) else "partial", provider=provider)
    else:
        record["blocker"] = blocker or "delphi_no_predictions"
    return record


def _infer(pairs: list[delphi_inputs.Pair]) -> dict:
    try:
        raw = verified_checkpoint()
        numbered = delphi_inputs.number_pairs(pairs)
        ready = [(pair.candidate_id, pair.vh, pair.vl, numbered[pair.candidate_id].cdr3) for pair in pairs]
        values = {name: numbered[name].blocker for name, _, _, cdr3 in ready if cdr3 is None}
        to_score = [(name, vh, vl, cdr3) for name, vh, vl, cdr3 in ready if cdr3 is not None]
        if not to_score:
            return {"values": values}
        from rfproteina.disease import delphi_psr_model as backend

        provider = {
            "device": "cpu", "release": RELEASE, "checkpoint_sha256": CHECKPOINT_SHA256,
            "torch": metadata.version("torch"), "anarcii": metadata.version("anarcii"),
        }
        model = backend.load_model(raw)
        for index in range(0, len(to_score), _BATCH_SIZE):
            batch = to_score[index:index + _BATCH_SIZE]
            scores = backend.predict(model, [(vh, vl, cdr3) for _, vh, vl, cdr3 in batch])
            for (name, *_), score in zip(batch, scores, strict=True):
                values[name] = score if _score(score) else "delphi_inference_failed"
        return {"values": values, "provider": provider}
    except DelPhiError as error:
        return {"blocker": str(error)}
    except (ImportError, metadata.PackageNotFoundError):
        return {"blocker": "delphi_runtime_unavailable"}
    except (ValueError, KeyError, TypeError, OSError, RuntimeError):
        return {"blocker": "delphi_inference_failed"}


def _worker_pairs(data: object) -> list[delphi_inputs.Pair]:
    if not isinstance(data, list) or not 0 < len(data) <= delphi_inputs.MAX_ROWS:
        return []
    names: set[str] = set()
    results = []
    for row in data:
        if not isinstance(row, list) or len(row) != 3 or not all(isinstance(part, str) for part in row):
            return []
        name, vh, vl = row
        if not delphi_inputs.valid_candidate_id(name) or name in names or delphi_inputs.sequence_blocker(vh, vl):
            return []
        results.append(delphi_inputs.Pair(name, vh, vl))
        names.add(name)
    return results


def _main() -> None:
    # Set these again for direct invocation. Neither import nor numbering may
    # select GPU devices or fetch a model over the network.
    os.environ.update(_CPU_ENV)
    raw = sys.stdin.read(_WORKER_LIMIT + 1)
    try:
        pairs = _worker_pairs(json.loads(raw)) if len(raw) <= _WORKER_LIMIT else []
    except (ValueError, RecursionError):
        pairs = []
    with redirect_stdout(sys.stderr):
        response = _infer(pairs) if pairs else {"blocker": "delphi_inference_failed"}
    print(json.dumps(response, allow_nan=False))
