"""Run the verified MetFrag release on a local list inside a network namespace."""

from __future__ import annotations

import functools
import hashlib
import math
import os
import re
import signal
import stat
import subprocess
import sys
import tempfile
import time
from contextlib import suppress
from pathlib import Path

RELEASE = "2.6.11"
JAR_NAME = f"MetFragCommandLine-{RELEASE}.jar"
JAR_URL = f"https://github.com/ipb-halle/MetFragRelaunched/releases/download/v{RELEASE}/{JAR_NAME}"
JAR_BYTES = 98_616_254
JAR_SHA256 = "8c55322fbc706c76df109dd886257262029695837bb1fb2211d4b88d945d8cfb"
TIMEOUT_SECONDS = 120
THREADS = 2

_UNAVAILABLE = "metfrag_runtime_unavailable"
_INTEGRITY = "metfrag_runtime_integrity"
_TIMEOUT = "metfrag_timeout"
_DISK = "metfrag_scratch_limit"
_FAILURE = "metfrag_provider_failure"
_OUTPUT = "metfrag_output_unreadable"
BLOCKERS = frozenset({_UNAVAILABLE, _INTEGRITY, _TIMEOUT, _DISK, _FAILURE, _OUTPUT})
_UNSHARE = Path("/usr/bin/unshare")
_NAMESPACE = ("--user", "--map-root-user", "--net", "--")
_JAVA_VERSION = re.compile(rb'(?:openjdk|java) version "21(?:[.\-+"])')
_MAX_INPUT = 256 * 1024
_MAX_OUTPUT = 4 * 1024 * 1024
_MAX_FILE = 32 * 1024 * 1024
_MAX_SCRATCH = 96 * 1024 * 1024
_MAX_ENTRIES = 1024
_ENVIRONMENT = {"PATH": "", "LC_ALL": "C", "LANG": "C"}


class MetfragRuntimeError(RuntimeError):
    """A MetFrag execution stopped with a named blocker."""

    def __init__(self, blocker: str) -> None:
        self.blocker = blocker
        super().__init__(blocker)


def release_directory() -> Path:
    """Return the pinned release directory under the current Python prefix."""
    return Path(sys.prefix) / "share" / "metfrag" / RELEASE


def _signature(info: os.stat_result) -> tuple[int, ...]:
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


@functools.lru_cache(maxsize=4)
def _check_digest(jar: Path, signature: tuple[int, ...]) -> None:
    with os.fdopen(os.open(jar, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as stream:
        if _signature(os.fstat(stream.fileno())) != signature:
            raise MetfragRuntimeError(_INTEGRITY)
        if hashlib.file_digest(stream, "sha256").hexdigest() != JAR_SHA256:
            raise MetfragRuntimeError(_INTEGRITY)
        if _signature(os.fstat(stream.fileno())) != signature:
            raise MetfragRuntimeError(_INTEGRITY)


def _verified_jar() -> Path:
    if sys.platform != "linux":
        raise MetfragRuntimeError(_UNAVAILABLE)
    directory = release_directory()
    jar = directory / JAR_NAME
    try:
        for parent in (directory.parent.parent, directory.parent, directory):
            if not stat.S_ISDIR(parent.lstat().st_mode):
                raise MetfragRuntimeError(_INTEGRITY)
        info = jar.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_size != JAR_BYTES:
            raise MetfragRuntimeError(_INTEGRITY)
        _check_digest(jar, _signature(info))
    except FileNotFoundError as error:
        raise MetfragRuntimeError(_UNAVAILABLE) from error
    except OSError as error:
        raise MetfragRuntimeError(_INTEGRITY) from error
    return jar


@functools.lru_cache(maxsize=8)
def _java_21(java: Path, signature: tuple[int, ...]) -> bool:
    try:
        with tempfile.TemporaryFile() as output:
            result = subprocess.run(
                [str(_UNSHARE), *_NAMESPACE, str(java), "-version"],
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.STDOUT,
                timeout=8,
                env=_ENVIRONMENT,
                check=False,
            )
            output.seek(0)
            version = output.read(8192)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0 and _JAVA_VERSION.search(version) is not None


def _java() -> Path:
    if not _UNSHARE.is_file() or not os.access(_UNSHARE, os.X_OK):
        raise MetfragRuntimeError(_UNAVAILABLE)
    choices = [Path(sys.prefix) / "share" / "p2rank" / "java" / "bin" / "java"]
    if home := os.environ.get("JAVA_HOME"):
        choices.append(Path(home) / "bin" / "java")
    choices.extend(Path(directory) / "java" for directory in os.get_exec_path() if directory)
    for choice in dict.fromkeys(choices):
        if not choice.is_absolute():
            continue
        try:
            java = choice.resolve(strict=True)
            info = java.stat()
        except (OSError, RuntimeError):
            continue
        if stat.S_ISREG(info.st_mode) and os.access(java, os.X_OK) and _java_21(java, _signature(info)):
            return java
    raise MetfragRuntimeError(_UNAVAILABLE)


def installed() -> bool:
    """Verify the local jar and the isolated Java 21 runtime."""
    try:
        _verified_jar()
        _java()
    except MetfragRuntimeError:
        return False
    return True


def _settings(precursor_mz: float, fragment_ppm: float, fragment_da: float) -> bytes:
    return (
        "PeakListPath = peaks.txt\n"
        "MetFragPeakListReader = de.ipbhalle.metfraglib.peaklistreader.TandemMassPeakListReader\n"
        "MetFragDatabaseType = LocalCSV\n"
        "LocalDatabasePath = candidates.csv\n"
        f"IonizedPrecursorMass = {precursor_mz}\n"
        "PrecursorIonMode = 1\n"
        "IsPositiveIonMode = true\n"
        f"FragmentPeakMatchRelativeMassDeviation = {fragment_ppm}\n"
        f"FragmentPeakMatchAbsoluteMassDeviation = {fragment_da}\n"
        "MetFragScoreTypes = FragmenterScore\n"
        "MetFragScoreWeights = 1.0\n"
        "MaximumTreeDepth = 2\n"
        f"NumberThreads = {THREADS}\n"
        "UseSmiles = true\n"
        "MetFragCandidateWriter = CSV\n"
        "SampleName = result\n"
        "ResultsPath = .\n"
    ).encode("ascii")


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
                        raise MetfragRuntimeError(_DISK)
                    try:
                        info = entry.stat(follow_symlinks=False)
                    except FileNotFoundError:
                        continue
                    if stat.S_ISDIR(info.st_mode):
                        pending.append(entry.path)
                    elif stat.S_ISREG(info.st_mode):
                        total += info.st_size
                        if info.st_size >= _MAX_FILE or total >= _MAX_SCRATCH:
                            raise MetfragRuntimeError(_DISK)
                    else:
                        raise MetfragRuntimeError(_FAILURE)
        except FileNotFoundError:
            # A native dependency may remove its temporary file while loading.
            continue


def _run(java: Path, jar: Path, directory: Path) -> None:
    import resource

    deadline = time.monotonic() + TIMEOUT_SECONDS
    command = [
        str(_UNSHARE),
        *_NAMESPACE,
        str(java),
        "-Xms32m",
        "-Xmx512m",
        "-XX:ActiveProcessorCount=2",
        "-Djava.io.tmpdir=.",
        "-Duser.home=.",
        "-Djava.net.useSystemProxies=false",
        "-Dfile.encoding=UTF-8",
        "-jar",
        str(jar),
        "settings.txt",
    ]
    with (directory / "provider.log").open("xb") as log:
        process = subprocess.Popen(
            command,
            cwd=directory,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=_ENVIRONMENT,
            start_new_session=True,
        )
        try:
            with suppress(ProcessLookupError):
                resource.prlimit(process.pid, resource.RLIMIT_CORE, (0, 0))
                resource.prlimit(process.pid, resource.RLIMIT_FSIZE, (_MAX_FILE, _MAX_FILE))
            while process.poll() is None:
                _check_scratch(directory)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise MetfragRuntimeError(_TIMEOUT)
                time.sleep(min(remaining, 0.1))
        finally:
            if process.returncode is None:
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait()
    _check_scratch(directory)
    if process.returncode == -signal.SIGXFSZ:
        raise MetfragRuntimeError(_DISK)
    if process.returncode != 0:
        raise MetfragRuntimeError(_FAILURE)


def _output(directory: Path) -> bytes:
    path = directory / "result.csv"
    try:
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= _MAX_OUTPUT:
            raise MetfragRuntimeError(_OUTPUT)
        with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as stream:
            if not os.path.samestat(info, os.fstat(stream.fileno())):
                raise MetfragRuntimeError(_OUTPUT)
            data = stream.read(_MAX_OUTPUT + 1)
        if len(data) != info.st_size:
            raise MetfragRuntimeError(_OUTPUT)
    except OSError as error:
        raise MetfragRuntimeError(_OUTPUT) from error
    return data


def predict(
    peaks: bytes, database: bytes, *, precursor_mz: float, precursor_ppm: float, fragment_ppm: float, fragment_da: float
) -> bytes:
    """Score a prepared local list and return its bounded native result table."""
    if not all(isinstance(data, bytes) and 0 < len(data) <= _MAX_INPUT for data in (peaks, database)):
        raise MetfragRuntimeError(_FAILURE)
    if (
        not all(
            type(value) in (int, float) and 0 <= value <= limit and math.isfinite(value)
            for value, limit in ((precursor_mz, 4000), (precursor_ppm, 1000), (fragment_ppm, 1000), (fragment_da, 1))
        )
        or precursor_mz <= 1.0073
        or precursor_ppm == 0
        or fragment_ppm == fragment_da == 0
    ):
        raise MetfragRuntimeError(_FAILURE)
    jar, java = _verified_jar(), _java()
    try:
        with tempfile.TemporaryDirectory(prefix="rfproteina-metfrag-") as temporary:
            directory = Path(temporary)
            (directory / "peaks.txt").write_bytes(peaks)
            (directory / "candidates.csv").write_bytes(database)
            (directory / "settings.txt").write_bytes(_settings(precursor_mz, fragment_ppm, fragment_da))
            _run(java, jar, directory)
            return _output(directory)
    except OSError as error:
        raise MetfragRuntimeError(_FAILURE) from error
