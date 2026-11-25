"""Verify the isolated Jazzy environment and execute its bounded worker."""

from __future__ import annotations

import functools
import json
import math
import os
import platform
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

from rfproteina.disease import jazzy_worker

VERSIONS = jazzy_worker.VERSIONS
RELEASE = VERSIONS["jazzy"]
WHEEL_SHA256 = {
    "click": "255bc9599cf7748b4b1a446ccc735421bd08a2ae529a8b88597d3de5664ee360",
    "jazzy": "b5d50a65c2d90cb6648b963425fb4121a2b80b811f00ee6aa2b4c615d0dfe6e6",
    "kallisto": "d7e908f909f65d57552c6fb7f2948eb5db6aaf6c5019b77d06c8c2cdc1123b71",
    "numpy": "675d61ffbfa78604709862923189bad94014bef562cc35cf61d3a07bba02a7ed",
    "pillow": "42d63be3af4f9b951b199b0a9d23365d37ab14687e6d77b74940432c8515c4ca",
    "rdkit": "cba519aa88683a0aff43486cb61625c735a696931180c0edd2f676bc5b86e7bd",
    "scipy": "de3ade0e53bc1f21358aa74ff4830235d716211d7d077e340c7349bc3542e884",
}
_UNAVAILABLE = "jazzy_runtime_unavailable"
_INTEGRITY = "jazzy_runtime_integrity"
_TIMEOUT = "jazzy_timeout"
_FAILURE = "jazzy_provider_failure"
_OUTPUT = "jazzy_output_unreadable"
_ROW_BLOCKERS = frozenset({jazzy_worker.GEOMETRY_UNAVAILABLE, jazzy_worker.CALCULATION_FAILED})
BLOCKERS = frozenset({_UNAVAILABLE, _INTEGRITY, _TIMEOUT, _FAILURE, _OUTPUT, *_ROW_BLOCKERS})
_RECEIPT = "install-receipt.json"
_MAX_RECEIPT_BYTES = 4096
_MAX_OUTPUT_BYTES = 64 * 1024
_PROBE_TIMEOUT = 30
_RUN_TIMEOUT = 90
_ENVIRONMENT = {
    "LC_ALL": "C",
    "PATH": "",
    "OMP_NUM_THREADS": "1",
    "OMP_THREAD_LIMIT": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


class JazzyRuntimeError(RuntimeError):
    """A local Jazzy operation stopped with a named blocker."""

    def __init__(self, blocker: str) -> None:
        self.blocker = blocker
        super().__init__(blocker)


def release_directory() -> Path:
    """Return the isolated Jazzy release directory."""
    return Path(sys.prefix) / "share" / "jazzy" / RELEASE


def _signature(info: os.stat_result) -> tuple[int, ...]:
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def _unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("duplicate field")
    return result


def _verify_receipt(path: Path) -> tuple[int, ...]:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= _MAX_RECEIPT_BYTES:
        raise JazzyRuntimeError(_INTEGRITY)
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as receipt:
        signature = _signature(os.fstat(receipt.fileno()))
        if signature != _signature(info):
            raise JazzyRuntimeError(_INTEGRITY)
        data = receipt.read(_MAX_RECEIPT_BYTES + 1)
        if len(data) != info.st_size or signature != _signature(os.fstat(receipt.fileno())):
            raise JazzyRuntimeError(_INTEGRITY)
    payload = json.loads(data.decode("ascii"), object_pairs_hook=_unique)
    if (
        not isinstance(payload, dict)
        or payload.keys() != {"version", "wheels"}
        or type(payload["version"]) is not int
        or payload["version"] != 1
        or payload["wheels"] != WHEEL_SHA256
    ):
        raise JazzyRuntimeError(_INTEGRITY)
    return signature


def _python() -> tuple[Path, tuple[tuple[int, ...], ...]]:
    if sys.platform != "linux" or platform.machine().lower() not in {"x86_64", "amd64"}:
        raise JazzyRuntimeError(_UNAVAILABLE)
    release = release_directory()
    try:
        info = release.lstat()
    except FileNotFoundError as error:
        raise JazzyRuntimeError(_UNAVAILABLE) from error
    except OSError as error:
        raise JazzyRuntimeError(_INTEGRITY) from error
    try:
        if not stat.S_ISDIR(info.st_mode) or any(
            not stat.S_ISDIR(path.lstat().st_mode) for path in (release.parent.parent, release.parent, release / "bin")
        ):
            raise JazzyRuntimeError(_INTEGRITY)
        receipt = _verify_receipt(release / _RECEIPT)
        python = release / "bin" / "python"
        effective = python.stat()
        launcher = python.lstat()
        config = (release / "pyvenv.cfg").lstat()
        if (
            not stat.S_ISREG(effective.st_mode)
            or not os.access(python, os.X_OK)
            or not stat.S_ISREG(config.st_mode)
            or not 0 < config.st_size <= _MAX_RECEIPT_BYTES
        ):
            raise JazzyRuntimeError(_INTEGRITY)
        return python, (receipt, _signature(config), _signature(launcher), _signature(effective))
    except (OSError, RecursionError, ValueError) as error:
        raise JazzyRuntimeError(_INTEGRITY) from error


def _invoke(python: Path, payload: bytes, args: tuple[str, ...], timeout: int) -> bytes:
    script = Path(jazzy_worker.__file__).resolve()
    try:
        with (
            tempfile.TemporaryDirectory(prefix="rfproteina-jazzy-") as directory,
            tempfile.TemporaryFile(dir=directory) as output,
        ):
            result = subprocess.run(
                [str(python), "-I", "-B", str(script), *args],
                input=payload,
                stdout=output,
                stderr=subprocess.DEVNULL,
                cwd=directory,
                env=_ENVIRONMENT,
                timeout=timeout,
                check=False,
            )
            if result.returncode != 0:
                raise JazzyRuntimeError(_FAILURE)
            output.seek(0)
            data = output.read(_MAX_OUTPUT_BYTES + 1)
    except subprocess.TimeoutExpired as error:
        raise JazzyRuntimeError(_TIMEOUT) from error
    except OSError as error:
        raise JazzyRuntimeError(_FAILURE) from error
    if not 0 < len(data) <= _MAX_OUTPUT_BYTES:
        raise JazzyRuntimeError(_OUTPUT)
    return data


@functools.lru_cache(maxsize=4)
def _check_worker(python: Path, signature: tuple[tuple[int, ...], ...]) -> None:
    try:
        ready = _invoke(python, b"", ("--check-installation",), _PROBE_TIMEOUT)
    except JazzyRuntimeError as error:
        if error.blocker == _TIMEOUT:
            raise
        raise JazzyRuntimeError(_INTEGRITY) from error
    if ready.strip() != jazzy_worker.READY.encode():
        raise JazzyRuntimeError(_INTEGRITY)


def installed() -> bool:
    """Report availability of the exact local provider."""
    try:
        _check_worker(*_python())
    except JazzyRuntimeError:
        return False
    return True


def predict(smiles: list[str]) -> list[dict]:
    """Return one finite hydration calculation or named blocker per input."""
    if (
        not isinstance(smiles, list)
        or not 0 < len(smiles) <= jazzy_worker.MAX_ROWS
        or not all(map(jazzy_worker.valid_smiles, smiles))
    ):
        raise JazzyRuntimeError(_FAILURE)
    payload = json.dumps(smiles, ensure_ascii=True, separators=(",", ":")).encode("ascii")
    if len(payload) > jazzy_worker.MAX_INPUT_BYTES:
        raise JazzyRuntimeError(_FAILURE)
    python, signature = _python()
    _check_worker(python, signature)
    data = _invoke(python, payload, (), _RUN_TIMEOUT)
    try:
        rows = json.loads(data.decode("ascii"), object_pairs_hook=_unique)
        if not isinstance(rows, list) or len(rows) != len(smiles):
            raise ValueError("wrong result count")
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("invalid result row")
            if row.keys() == {"blocker"} and isinstance(row["blocker"], str) and row["blocker"] in _ROW_BLOCKERS:
                continue
            if row.keys() != jazzy_worker.RESULT_FIELDS or any(
                type(value) not in {int, float} or not math.isfinite(value) for value in row.values()
            ):
                raise ValueError("invalid result values")
    except (ArithmeticError, RecursionError, ValueError) as error:
        raise JazzyRuntimeError(_OUTPUT) from error
    return rows
