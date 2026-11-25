"""Verify and run the pinned Linux CREST build with fixed CPU settings."""

from __future__ import annotations

import hashlib
import os
import platform
import signal
import stat
import subprocess
import sys
import tempfile
import time
from contextlib import suppress
from pathlib import Path

RELEASE = "3.0.2"
ARCHIVE_URL = f"https://github.com/crest-lab/crest/releases/download/v{RELEASE}/crest-gnu-12-ubuntu-latest.tar.xz"
ARCHIVE_BYTES = 8_764_012
ARCHIVE_SHA256 = "8e5bd18b06f99741ebd7bb71b3a996295f391b2f076aaeab739740f709e9554d"
EXECUTABLE_BYTES = 58_561_984
EXECUTABLE_SHA256 = "201db7940796c4af3652245ba7c8e1a4234767f4e19223772fbd83098d5f2142"
THREADS = 2
TIMEOUT_SECONDS = 240

_UNAVAILABLE = "crest_runtime_unavailable"
_INTEGRITY = "crest_runtime_integrity"
_TIMEOUT = "crest_timeout"
_DISK = "crest_scratch_limit"
_FAILURE = "crest_provider_failure"
_OUTPUT = "crest_output_unreadable"
BLOCKERS = frozenset({_UNAVAILABLE, _INTEGRITY, _TIMEOUT, _DISK, _FAILURE, _OUTPUT})
_MAX_INPUT = 64 * 1024
_MAX_OUTPUT = 8 * 1024 * 1024
_MAX_FILE = 64 * 1024 * 1024
_MAX_SCRATCH = 256 * 1024 * 1024
_MAX_ENTRIES = 4096
_LOG_TAIL = 128 * 1024
_COMPLETION_LINE = b"CREST terminated normally."


class CrestRuntimeError(RuntimeError):
    """A bounded CREST run stopped with a public blocker code."""

    def __init__(self, blocker: str) -> None:
        self.blocker = blocker
        super().__init__(blocker)


def supported_platform() -> bool:
    """Return whether the pinned binary supports this host."""
    return sys.platform == "linux" and platform.machine().lower() in {"x86_64", "amd64"}


def release_directory() -> Path:
    """Return the release directory in the current Python environment."""
    return Path(sys.prefix) / "share" / "crest" / RELEASE


def _verified_executable() -> Path:
    if not supported_platform():
        raise CrestRuntimeError(_UNAVAILABLE)
    directory = release_directory()
    path = directory / "crest"
    try:
        for parent in (directory.parent.parent, directory.parent, directory):
            if not stat.S_ISDIR(parent.lstat().st_mode):
                raise CrestRuntimeError(_INTEGRITY)
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_size != EXECUTABLE_BYTES or not os.access(path, os.X_OK):
            raise CrestRuntimeError(_INTEGRITY)
        with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as executable:
            if not os.path.samestat(info, os.fstat(executable.fileno())):
                raise CrestRuntimeError(_INTEGRITY)
            if hashlib.file_digest(executable, "sha256").hexdigest() != EXECUTABLE_SHA256:
                raise CrestRuntimeError(_INTEGRITY)
    except FileNotFoundError as error:
        raise CrestRuntimeError(_UNAVAILABLE) from error
    except OSError as error:
        raise CrestRuntimeError(_INTEGRITY) from error
    return path


def installed() -> bool:
    """Return whether the exact release executable is available."""
    try:
        _verified_executable()
    except CrestRuntimeError:
        return False
    return True


def _check_scratch(directory: Path) -> None:
    pending = [directory]
    count = 0
    total = 0
    while pending:
        try:
            with os.scandir(pending.pop()) as entries:
                for entry in entries:
                    count += 1
                    if count > _MAX_ENTRIES:
                        raise CrestRuntimeError(_DISK)
                    try:
                        info = entry.stat(follow_symlinks=False)
                    except FileNotFoundError:
                        continue
                    if stat.S_ISDIR(info.st_mode):
                        pending.append(entry.path)
                    elif stat.S_ISREG(info.st_mode):
                        total += info.st_size
                        if info.st_size >= _MAX_FILE or total >= _MAX_SCRATCH:
                            raise CrestRuntimeError(_DISK)
                    else:
                        raise CrestRuntimeError(_FAILURE)
        except FileNotFoundError:
            # CREST removes scratch material as it advances between stages.
            continue


def _limit_files(process: subprocess.Popen[bytes]) -> None:
    # Import only on Linux. The application can still report installation status on Windows.
    import resource

    try:
        resource.prlimit(process.pid, resource.RLIMIT_CORE, (0, 0))
        resource.prlimit(process.pid, resource.RLIMIT_FSIZE, (_MAX_FILE, _MAX_FILE))
    except ProcessLookupError:
        # A process that already exited is checked by the caller.
        pass


def _terminate(process: subprocess.Popen[bytes]) -> None:
    with suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)
    process.wait()


def _run(executable: Path, directory: Path) -> None:
    log_path = directory / "crest.log"
    environment = {
        "PATH": "",
        "LC_ALL": "C",
        "TMPDIR": str(directory),
        "OMP_NUM_THREADS": str(THREADS),
        "OMP_THREAD_LIMIT": str(THREADS),
        "OMP_DYNAMIC": "FALSE",
        "OPENBLAS_NUM_THREADS": str(THREADS),
        "MKL_NUM_THREADS": str(THREADS),
    }
    deadline = time.monotonic() + TIMEOUT_SECONDS
    with log_path.open("xb") as log:
        process = subprocess.Popen(
            [str(executable), "input.xyz", "--quick", "--gfn2", "--imtdgc", "-T", str(THREADS)],
            cwd=directory,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            _limit_files(process)
            while process.poll() is None:
                _check_scratch(directory)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise CrestRuntimeError(_TIMEOUT)
                time.sleep(min(remaining, 0.1))
        finally:
            if process.returncode is None:
                _terminate(process)
    _check_scratch(directory)
    if process.returncode == -signal.SIGXFSZ:
        raise CrestRuntimeError(_DISK)
    if process.returncode != 0:
        raise CrestRuntimeError(_FAILURE)
    with log_path.open("rb") as log:
        log.seek(max(0, log_path.stat().st_size - _LOG_TAIL))
        if not any(line.strip() == _COMPLETION_LINE for line in log.read(_LOG_TAIL).splitlines()):
            raise CrestRuntimeError(_FAILURE)


def _read_ensemble(directory: Path) -> bytes:
    path = directory / "crest_conformers.xyz"
    try:
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= _MAX_OUTPUT:
            raise CrestRuntimeError(_OUTPUT)
        with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as output:
            if not os.path.samestat(info, os.fstat(output.fileno())):
                raise CrestRuntimeError(_OUTPUT)
            data = output.read(_MAX_OUTPUT + 1)
        if len(data) != info.st_size:
            raise CrestRuntimeError(_OUTPUT)
    except OSError as error:
        raise CrestRuntimeError(_OUTPUT) from error
    return data


def predict(xyz: bytes) -> bytes:
    """Run CREST on prepared XYZ and return its bounded conformer ensemble."""
    if not isinstance(xyz, bytes) or not 0 < len(xyz) <= _MAX_INPUT:
        raise CrestRuntimeError(_FAILURE)
    executable = _verified_executable()
    try:
        with tempfile.TemporaryDirectory(prefix="rfproteina-crest-") as temporary:
            directory = Path(temporary)
            (directory / "input.xyz").write_bytes(xyz)
            _run(executable, directory)
            return _read_ensemble(directory)
    except OSError as error:
        raise CrestRuntimeError(_FAILURE) from error
