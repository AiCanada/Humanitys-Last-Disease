"""Verify and run the slim native ESP-Sim scoring provider locally."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
from importlib.util import find_spec
from pathlib import Path

from rfproteina.disease import espsim_worker

RELEASE = espsim_worker.RELEASE
ARCHIVE_URL = espsim_worker.ARCHIVE_URL
ARCHIVE_BYTES = espsim_worker.ARCHIVE_BYTES
ARCHIVE_SHA256 = espsim_worker.ARCHIVE_SHA256
SOURCE_SHA256 = ARCHIVE_SHA256
FILES = espsim_worker.FILES
_UNAVAILABLE = "espsim_runtime_unavailable"
_INTEGRITY = "espsim_runtime_integrity"
_TIMEOUT = "espsim_timeout"
_FAILURE = "espsim_provider_failure"
_OUTPUT = "espsim_output_unreadable"
BLOCKERS = frozenset({_UNAVAILABLE, _INTEGRITY, _TIMEOUT, _FAILURE, _OUTPUT, espsim_worker.CHARGE_UNAVAILABLE})
_RUN_TIMEOUT = 180
_MAX_OUTPUT_BYTES = 4096
_WORKER = Path(espsim_worker.__file__).resolve()


class EspsimRuntimeError(RuntimeError):
    """A local ESP-Sim operation stopped with a named blocker."""

    def __init__(self, blocker: str) -> None:
        self.blocker = blocker
        super().__init__(blocker)


def release_directory() -> Path:
    """Return the exact provider source directory in this environment."""
    return espsim_worker.release_directory()


def _require_provider() -> None:
    release = release_directory()
    try:
        release.lstat()
    except FileNotFoundError as error:
        raise EspsimRuntimeError(_UNAVAILABLE) from error
    except OSError as error:
        raise EspsimRuntimeError(_INTEGRITY) from error
    try:
        espsim_worker.verify_assets()
    except (OSError, ValueError) as error:
        raise EspsimRuntimeError(_INTEGRITY) from error
    try:
        available = all(find_spec(name) is not None for name in ("rdkit", "numpy"))
    except (ImportError, ValueError) as error:
        raise EspsimRuntimeError(_UNAVAILABLE) from error
    if not available:
        raise EspsimRuntimeError(_UNAVAILABLE)


def _invoke(payload: bytes) -> bytes:
    environment = {key: value for key, value in os.environ.items() if key.upper() in {"SYSTEMROOT", "WINDIR"}}
    environment.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    try:
        with (
            tempfile.TemporaryDirectory(prefix="rfproteina-espsim-") as directory,
            tempfile.TemporaryFile(dir=directory) as output,
        ):
            result = subprocess.run(
                [sys.executable, "-I", "-B", str(_WORKER)],
                input=payload,
                stdout=output,
                stderr=subprocess.DEVNULL,
                cwd=directory,
                env=environment,
                timeout=_RUN_TIMEOUT,
                check=False,
            )
            if result.returncode != 0:
                raise EspsimRuntimeError(_FAILURE)
            output.seek(0)
            data = output.read(_MAX_OUTPUT_BYTES + 1)
    except subprocess.TimeoutExpired as error:
        raise EspsimRuntimeError(_TIMEOUT) from error
    except OSError as error:
        raise EspsimRuntimeError(_FAILURE) from error
    if not 0 < len(data) <= _MAX_OUTPUT_BYTES:
        raise EspsimRuntimeError(_OUTPUT)
    return data


def installed() -> bool:
    """Report availability of the exact local ESP-Sim scoring source."""
    try:
        _require_provider()
    except EspsimRuntimeError:
        return False
    return True


def _unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("duplicate result field")
    return result


def _numeric(value: object, minimum: float, maximum: float) -> bool:
    return type(value) in {int, float} and minimum <= value <= maximum and math.isfinite(value)


def predict(candidate: bytes, reference: bytes) -> dict:
    """Return separate shape and electrostatic results for a prepared pose pair."""
    if not all(
        isinstance(data, bytes) and 0 < len(data) <= espsim_worker.MAX_SDF_BYTES for data in (candidate, reference)
    ):
        raise EspsimRuntimeError(_FAILURE)
    _require_provider()
    payload = espsim_worker.HEADER.pack(len(candidate), len(reference)) + candidate + reference
    try:
        row = json.loads(_invoke(payload).decode("ascii"), object_pairs_hook=_unique)
        if not isinstance(row, dict) or not _numeric(row.get("shape_tanimoto"), 0, 1):
            raise ValueError("invalid shape result")
        if row.keys() == {"shape_tanimoto", "blocker"} and row["blocker"] == espsim_worker.CHARGE_UNAVAILABLE:
            return row
        if row.keys() != {"shape_tanimoto", "electrostatic_carbo"} or not _numeric(row["electrostatic_carbo"], -1, 1):
            raise ValueError("invalid electrostatic result")
    except (ArithmeticError, RecursionError, ValueError) as error:
        raise EspsimRuntimeError(_OUTPUT) from error
    return row
