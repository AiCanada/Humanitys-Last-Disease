"""Load only the verified public fastsolv ensemble in a bounded CPU worker."""

from __future__ import annotations

import functools
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

RELEASE = "1.0.1"
ZENODO_RELEASE = "13943074"
CHECKPOINT_SIZE = 224_346_392
CHECKPOINT_MD5 = {
    "fastsolv_1701_1.ckpt": "8d0672a025eb0b183f27c238a47c7cfe",
    "fastsolv_1701_2.ckpt": "eb34f9898d00ec6eea1d1a33443968d8",
    "fastsolv_1701_3.ckpt": "6731ee7f42509ed2bdedcdad415940f4",
    "fastsolv_1701_4.ckpt": "e418b389f68d5004508315028f665df6",
}
_CHECKPOINT_SHA256 = {
    "fastsolv_1701_1.ckpt": "964aaf609f473ae704d4476f2aa4610b4736079bdba524cfad7c57885e1914fd",
    "fastsolv_1701_2.ckpt": "96af2a14662be6bd8d10258256f88de37911ccb0026ccfdfde911efc2a1a94e2",
    "fastsolv_1701_3.ckpt": "f21f73ac00c2e7b2d2b9743abebfae757c8ec0431f884153b8f5311509582a18",
    "fastsolv_1701_4.ckpt": "42fa948d6f3c915154e4772ff2deb04c2820deea15cea4085f3ba091a979589e",
}
_REQUIRED = {"rdkit": "2025.9.6", "mordredcommunity": "2.0.7", "numpy": "2.2.6"}
_UNAVAILABLE = "fastsolv_runtime_unavailable"
_INTEGRITY = "fastsolv_runtime_integrity"
_TIMEOUT = "fastsolv_timeout"
_FAILURE = "fastsolv_provider_failure"
_OUTPUT = "fastsolv_output_unreadable"
BLOCKERS = frozenset({_UNAVAILABLE, _INTEGRITY, _TIMEOUT, _FAILURE, _OUTPUT})
_MAX_BYTES = 128 * 1024
_TIMEOUT_SECONDS = 180


class FastsolvRuntimeError(RuntimeError):
    """An unavailable, unverified or unsuccessful local fastsolv runtime."""


def checkpoint_digests() -> dict[str, str]:
    return dict(_CHECKPOINT_SHA256)


def release_directory() -> Path:
    return Path(sys.prefix) / "share" / "fastsolv" / RELEASE


def _stamp(info: os.stat_result) -> tuple[int, ...]:
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def _reader(path: Path, expected: tuple[int, ...]) -> tuple[str, str]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0)
    md5 = hashlib.md5(usedforsecurity=False)  # Publisher transport digest, paired with SHA-256.
    sha256 = hashlib.sha256()
    with os.fdopen(os.open(path, flags), "rb") as stream:
        opened = os.fstat(stream.fileno())
        if not stat.S_ISREG(opened.st_mode) or _stamp(opened) != expected:
            raise FastsolvRuntimeError(_INTEGRITY)
        read = 0
        while block := stream.read(1024 * 1024):
            read += len(block)
            if read > CHECKPOINT_SIZE:
                raise FastsolvRuntimeError(_INTEGRITY)
            sha256.update(block)
            md5.update(block)
        if read != CHECKPOINT_SIZE or _stamp(os.fstat(stream.fileno())) != expected:
            raise FastsolvRuntimeError(_INTEGRITY)
    return md5.hexdigest(), sha256.hexdigest()


@functools.lru_cache(maxsize=4)
def _verified(directory: Path, signature: tuple[tuple[int, ...], ...]) -> tuple[Path, ...]:
    paths = tuple(directory / name for name in CHECKPOINT_MD5)
    for path, stamp in zip(paths, signature, strict=True):
        actual = _reader(path, stamp)
        if actual != (CHECKPOINT_MD5[path.name], _CHECKPOINT_SHA256[path.name]):
            raise FastsolvRuntimeError(_INTEGRITY)
    return paths


def verify_directory(directory: Path) -> tuple[Path, ...]:
    try:
        if CHECKPOINT_MD5.keys() != _CHECKPOINT_SHA256.keys() or not stat.S_ISDIR(directory.lstat().st_mode):
            raise FastsolvRuntimeError(_INTEGRITY)
        stamps = []
        for name in CHECKPOINT_MD5:
            info = (directory / name).lstat()
            if not stat.S_ISREG(info.st_mode) or info.st_size != CHECKPOINT_SIZE:
                raise FastsolvRuntimeError(_INTEGRITY)
            stamps.append(_stamp(info))
        return _verified(directory, tuple(stamps))
    except FileNotFoundError as error:
        raise FastsolvRuntimeError(_UNAVAILABLE) from error
    except OSError as error:
        raise FastsolvRuntimeError(_INTEGRITY) from error


def verified_assets() -> tuple[Path, ...]:
    return verify_directory(release_directory())


def _dependencies() -> bool:
    try:
        torch = re.match(r"([0-9]+)\.([0-9]+)", version("torch"))
        return bool(
            torch
            and tuple(map(int, torch.groups())) >= (2, 6)
            and all(version(name) == expected for name, expected in _REQUIRED.items())
        )
    except PackageNotFoundError:
        return False


def installed() -> bool:
    if not _dependencies():
        return False
    try:
        verified_assets()
    except FastsolvRuntimeError:
        return False
    return True


def predict(batch: list[dict]) -> list:
    if not _dependencies():
        raise FastsolvRuntimeError(_UNAVAILABLE)
    verified_assets()
    payload = json.dumps(batch, allow_nan=False).encode("ascii")
    if not 0 < len(batch) <= 128 or len(payload) > _MAX_BYTES:
        raise FastsolvRuntimeError(_FAILURE)
    script = Path(__file__).with_name("fastsolv_worker.py").resolve()
    environment = {key: value for key, value in os.environ.items() if key.upper() in ("SYSTEMROOT", "WINDIR")}
    environment.update(
        CLUSTER="local",
        PYTHON_DOTENV_DISABLED="1",
        CUDA_VISIBLE_DEVICES="-1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
    )
    try:
        with tempfile.TemporaryDirectory(prefix="rfproteina-fastsolv-") as directory:
            result = subprocess.run(
                [sys.executable, "-I", "-B", str(script)],
                env=environment,
                cwd=directory,
                input=payload,
                capture_output=True,
                check=True,
                timeout=_TIMEOUT_SECONDS,
            )
    except subprocess.TimeoutExpired as error:
        raise FastsolvRuntimeError(_TIMEOUT) from error
    except (OSError, subprocess.CalledProcessError) as error:
        raise FastsolvRuntimeError(_FAILURE) from error
    try:
        if len(result.stdout) > _MAX_BYTES:
            raise ValueError("Overlong worker output.")
        return json.loads(result.stdout)
    except (ValueError, RecursionError) as error:
        raise FastsolvRuntimeError(_OUTPUT) from error
