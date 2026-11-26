"""Verify the released SBML and run the CPU solver outside the application process."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from functools import lru_cache
from pathlib import Path

from rfproteina.disease import canagliflozin_worker as worker
from rfproteina.disease.ligand_inputs import read_regular

MODEL_RELEASE = "0.8.0"
MODEL_REVISION = "0100cbb3f033d88067a36d565aee55abe6177de9"
MODEL_FILENAME = "canagliflozin_body_flat.xml"
MODEL_URL = (
    f"https://raw.githubusercontent.com/matthiaskoenig/canagliflozin-model/{MODEL_REVISION}/models/{MODEL_FILENAME}"
)
MODEL_SIZE = 252_849
MODEL_SHA256 = "440c48dffd6722bf6e6358c2ce1c85cbe7103b3d4b117f5a9c26eaaef9cf1a13"
BLOCKERS = frozenset({
    "canagliflozin_model_missing", "canagliflozin_model_integrity", "canagliflozin_solver_unavailable",
    "canagliflozin_timeout", "canagliflozin_simulation_failed", "canagliflozin_invalid_report",
})


class CanagliflozinRuntimeError(ValueError):
    """The pinned model or numerical runtime did not produce a verified result."""


def install_directory() -> Path:
    return Path(sys.prefix) / "share" / "canagliflozin_pbpk"


def unique_fields(pairs: list[tuple]) -> dict:
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("duplicate fields")
    return result


def verify_model(raw: bytes) -> bytes:
    if len(raw) != MODEL_SIZE or hashlib.sha256(raw).hexdigest() != MODEL_SHA256:
        raise CanagliflozinRuntimeError("canagliflozin_model_integrity")
    return raw


def model_bytes(path: Path) -> bytes:
    try:
        raw = read_regular(path, MODEL_SIZE, "canagliflozin_model", CanagliflozinRuntimeError)
    except CanagliflozinRuntimeError as error:
        if str(error) == "canagliflozin_model_missing":
            raise
        raise CanagliflozinRuntimeError("canagliflozin_model_integrity") from error
    return verify_model(raw)


def interpreter(root: Path) -> Path:
    return root / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def invoke(root: Path, model: bytes | None) -> dict:
    """Feed the solver the same in-memory SBML snapshot that passed verification."""
    python = interpreter(root)
    if not python.is_file():
        raise CanagliflozinRuntimeError("canagliflozin_solver_unavailable")
    command = [str(python), "-I", "-B", str(Path(worker.__file__).resolve())]
    if model is None:
        command.append("--check-installation")
    try:
        with tempfile.TemporaryDirectory(prefix="canagliflozin-") as scratch, tempfile.TemporaryFile() as output:
            env = {k: os.environ[k] for k in ("SYSTEMROOT", "WINDIR") if k in os.environ}
            env.update({
                "HOME": scratch, "TMPDIR": scratch, "TEMP": scratch, "TMP": scratch,
                "PATH": os.defpath, "LC_ALL": "C", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
            })
            result = subprocess.run(
                command, input=model or b"", stdout=output, stderr=subprocess.DEVNULL,
                timeout=90 if model is not None else 20, cwd=scratch, env=env, check=False,
            )
            if result.returncode == worker.SOLVER_UNAVAILABLE or (model is None and result.returncode):
                raise CanagliflozinRuntimeError("canagliflozin_solver_unavailable")
            if result.returncode:
                raise CanagliflozinRuntimeError("canagliflozin_simulation_failed")
            output.seek(0)
            raw = output.read(8193)
    except subprocess.TimeoutExpired as error:
        raise CanagliflozinRuntimeError("canagliflozin_timeout") from error
    except OSError as error:
        raise CanagliflozinRuntimeError("canagliflozin_solver_unavailable") from error
    try:
        if len(raw) > 8192:
            raise ValueError("oversize worker result")
        response = json.loads(raw.decode("utf-8"), object_pairs_hook=unique_fields)
        if model is None:
            if response != {"libroadrunner": worker.SOLVER_VERSION}:
                raise CanagliflozinRuntimeError("canagliflozin_solver_unavailable")
            return response
        return worker.validated_result(response)
    except CanagliflozinRuntimeError:
        raise
    except (ValueError, RecursionError) as error:
        raise CanagliflozinRuntimeError("canagliflozin_invalid_report") from error


@lru_cache(maxsize=4)
def _check_solver(root: Path, signature: tuple) -> bool:
    """Reuse successful import probes while the local solver and package files stay the same."""
    invoke(root, None)
    return True


def _solver_signature(root: Path) -> tuple:
    site = root / "venv" / ("Lib" if os.name == "nt" else "lib/python3.12") / "site-packages"
    suffix = ".pyd" if os.name == "nt" else ".so"
    paths = [
        interpreter(root), root / "venv" / "pyvenv.cfg", Path(worker.__file__), Path(worker.profiles.__file__),
        site / "roadrunner" / "__init__.py", site / "roadrunner" / "roadrunner.py",
        site / "numpy" / "__init__.py",
        *(site / f"{name}-{version}.dist-info" / "METADATA" for name, version in worker.VERSIONS.items()),
    ]
    # Extension filenames can include a Python ABI and host platform tag.
    paths.extend((site / "roadrunner").glob(f"_roadrunner*{suffix}"))
    paths.extend((site / "numpy" / "_core").glob(f"_multiarray_umath*{suffix}"))
    # Wheel repair tools put required native libraries (including OpenBLAS) in sibling .libs directories.
    for name in worker.VERSIONS:
        paths.extend((site / f"{name}.libs").glob("*"))
    signature = []
    for path in sorted(paths):
        stat = path.stat()
        signature.append((path, stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns))
    return tuple(signature)


def installed() -> bool:
    root = install_directory()
    try:
        model_bytes(root / "model" / MODEL_FILENAME)
        return _check_solver(root, _solver_signature(root))
    except (CanagliflozinRuntimeError, OSError):
        return False


def simulate() -> dict:
    root = install_directory()
    return invoke(root, model_bytes(root / "model" / MODEL_FILENAME))
