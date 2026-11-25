"""Run only the verified local P2Rank release with a fixed CPU prediction command."""

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
from pathlib import Path

RELEASE = "2.5.1"
ARCHIVE_SHA256 = "d243f2d9036ac053fefb9407b5fe1c85f4fe077c519fd975ac585e995feab274"
FILES_SHA256 = "c12fdba25f8598be28f664e3cca51ebb0681aa7eaaaf3c6b8907f032b0400e0d"
RECEIPT = "rfproteina-install.json"
_MAX_RECEIPT = 128 * 1024
_MAX_INPUT = 16 * 1024 * 1024
_MAX_OUTPUT = 8 * 1024 * 1024
_TIMEOUT = 120
_PROFILES = ("default", "alphafold")
_SUFFIXES = (".pdb", ".cif")
_RELEASE_JAR = "bin/p2rank.jar"
_UNAVAILABLE = "p2rank_runtime_unavailable"
_INTEGRITY = "p2rank_runtime_integrity"
_FAILURE = "p2rank_provider_failure"
_OUTPUT = "p2rank_output_unreadable"

_JAVA_OPTIONS = {
    "JAVA_OPTS", "JAVA_TOOL_OPTIONS", "JDK_JAVA_OPTIONS", "_JAVA_OPTIONS", "CLASSPATH",
    "LD_PRELOAD", "LD_LIBRARY_PATH", "DYLD_INSERT_LIBRARIES", "DYLD_LIBRARY_PATH",
}
_JAVA_VERSION = re.compile(rb'^\s*(?:openjdk|java) version "([0-9]{1,2})(?:[.\-+\"]|$)', re.MULTILINE)
_FILE_NAME = re.compile(r"[A-Za-z0-9_./+\-]+")
_HASH = re.compile(r"[a-f0-9]{64}")

# In the pinned release PdbUtils' attempt to disable BioJava fetching is a
# no-op. Select its bundled reduced chemical dictionary before parsing a
# receptor. The caller supplies protein-only coordinates for this scope.
_OFFLINE_CONFIG = b"""

org.biojava.nbio.structure.chem.ChemCompGroupFactory.setChemCompProvider(
    new org.biojava.nbio.structure.chem.ReducedChemCompProvider()
)
"""


def _install_root() -> Path:
    return Path(sys.prefix) / "share" / "p2rank" / RELEASE


def _links_or_reparse(info: os.stat_result) -> bool:
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
    )


def _directory(path: Path) -> None:
    info = path.lstat()
    if _links_or_reparse(info) or not stat.S_ISDIR(info.st_mode):
        raise RuntimeError(_INTEGRITY)


def _regular(path: Path, blocker: str) -> os.stat_result:
    info = path.lstat()
    if _links_or_reparse(info) or not stat.S_ISREG(info.st_mode):
        raise RuntimeError(blocker)
    return info


def _read(path: Path, limit: int, blocker: str) -> bytes:
    try:
        if not 0 < _regular(path, blocker).st_size <= limit:
            raise RuntimeError(blocker)
        with path.open("rb") as handle:
            data = handle.read(limit + 1)
        if not 0 < len(data) <= limit:
            raise RuntimeError(blocker)
    except OSError as error:
        raise RuntimeError(blocker) from error
    return data


def _unique(keys: list[tuple[str, object]]) -> dict[str, object]:
    obj = dict(keys)
    if len(obj) != len(keys):
        raise ValueError("duplicate receipt keys")
    return obj


def _receipt_files(root: Path) -> dict[str, str]:
    try:
        receipt = json.loads(_read(root / RECEIPT, _MAX_RECEIPT, _INTEGRITY), object_pairs_hook=_unique)
        if (
            not isinstance(receipt, dict) or receipt.keys() != {"version", "archive_sha256", "files"}
            or receipt["version"] != RELEASE or receipt["archive_sha256"] != ARCHIVE_SHA256
        ):
            raise RuntimeError(_INTEGRITY)
        files = receipt["files"]
        if not isinstance(files, dict) or not files or not all(
            isinstance(name, str) and _FILE_NAME.fullmatch(name)
            and all(part not in {"", ".", ".."} for part in name.split("/"))
            and isinstance(digest, str) and _HASH.fullmatch(digest)
            for name, digest in files.items()
        ):
            raise RuntimeError(_INTEGRITY)
        canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("ascii")
        if hashlib.sha256(canonical).hexdigest() != FILES_SHA256:
            raise RuntimeError(_INTEGRITY)
    except (ValueError, RecursionError) as error:
        raise RuntimeError(_INTEGRITY) from error
    return files


def _executed_assets(files: dict[str, str]) -> set[str]:
    required = {
        _RELEASE_JAR,
        *(f"config/{profile}.groovy" for profile in _PROFILES),
        *(f"models/{profile}/{leaf}" for profile in _PROFILES for leaf in ("model.zst", "features.txt")),
        *(f"models/_score_transform/{profile}_{scale}.json"
          for profile in _PROFILES for scale in ("ZscoreTpTransformer", "ProbabilityScoreTransformer")),
    }
    required.update(name for name in files if name.startswith("bin/lib/") and name.lower().endswith(".jar"))
    if not required <= files.keys():
        raise RuntimeError(_INTEGRITY)
    return required


def _verify_digest(path: Path, expected: str) -> None:
    with path.open("rb") as handle:
        if hashlib.file_digest(handle, "sha256").hexdigest() != expected:
            raise RuntimeError(_INTEGRITY)


def _verified_release() -> Path:
    root = _install_root()
    try:
        root.lstat()
    except FileNotFoundError as error:
        raise RuntimeError(_UNAVAILABLE) from error
    except OSError as error:
        raise RuntimeError(_INTEGRITY) from error

    try:
        for path in (root.parent.parent, root.parent, root):
            _directory(path)
        files = _receipt_files(root)
        required = _executed_assets(files)
        checked_dirs = {root}
        for name in sorted(required):
            path = root / name
            for directory in reversed(path.relative_to(root).parents):
                absolute = root / directory
                if absolute not in checked_dirs:
                    _directory(absolute)
                    checked_dirs.add(absolute)
            _regular(path, _INTEGRITY)
            _verify_digest(path, files[name])
        present_jars = {
            path.relative_to(root).as_posix()
            for path in (root / "bin" / "lib").iterdir() if path.suffix.lower() == ".jar"
        }
        if present_jars != {name for name in required if name.startswith("bin/lib/")}:
            raise RuntimeError(_INTEGRITY)
    except OSError as error:
        raise RuntimeError(_INTEGRITY) from error
    return root


def _java_env() -> dict[str, str]:
    return {key: value for key, value in os.environ.items() if key.upper() not in _JAVA_OPTIONS}


@functools.lru_cache(maxsize=8)
def _supported_java(exe: Path, signature: tuple[int, int, int, int, int]) -> bool:
    # The signature invalidates this short version probe when Java is replaced.
    try:
        result = subprocess.run(
            [str(exe), "-version"], env=_java_env(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            timeout=8, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    match = _JAVA_VERSION.search(result.stdout)
    return result.returncode == 0 and match is not None and 17 <= int(match.group(1)) <= 24


def _java() -> Path:
    name = "java.exe" if os.name == "nt" else "java"
    roots = [Path(sys.prefix) / "share" / "p2rank" / "java" / "bin"]
    if home := os.environ.get("JAVA_HOME"):
        roots.append(Path(home) / "bin")
    roots.extend(Path(root) for root in os.get_exec_path() if root)
    for directory in dict.fromkeys(roots):
        # An empty or relative PATH entry could select a file from a run.
        if not directory.is_absolute():
            continue
        try:
            candidate = (directory / name).resolve(strict=True)
            sig = candidate.stat()
        except (OSError, RuntimeError):
            continue
        stamp = (sig.st_dev, sig.st_ino, sig.st_size, sig.st_mtime_ns, sig.st_ctime_ns)
        if stat.S_ISREG(sig.st_mode) and os.access(candidate, os.X_OK) and _supported_java(candidate, stamp):
            return candidate
    raise RuntimeError(_UNAVAILABLE)


def installed() -> bool:
    """Whether the pinned release files and a compatible Java are available."""
    try:
        _verified_release()
        _java()
    except RuntimeError:
        return False
    return True


def _run(root: Path, java: Path, work: Path, data: bytes, suffix: str, profile: str) -> bytes:
    structure = work / f"receptor{suffix}"
    structure.write_bytes(data)
    output = work / "output"
    output.mkdir()
    config = work / "predict.groovy"
    profile_settings = (root / "config" / "alphafold.groovy").read_bytes() if profile == "alphafold" else b""
    config.write_bytes(profile_settings + _OFFLINE_CONFIG)
    classpath = os.pathsep.join((str(root / _RELEASE_JAR), str(root / "bin" / "lib" / "*")))
    command = [
        str(java), "-Xmx2048m", "-XX:ActiveProcessorCount=1", "-Djava.awt.headless=true",
        f"-Djava.io.tmpdir={work}", f"-Duser.home={work}", "-cp", classpath, "cz.siret.prank.program.Main",
        "predict", "-f", str(structure), "-o", str(output), "-c", str(config),
        "-m", str(root / "models" / profile), "-threads", "1", "-visualizations", "0", "-label_residues", "0",
        "-log_to_file", "0", "-log_to_console", "0",
    ]
    try:
        result = subprocess.run(
            command, cwd=work, env=_java_env(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=_TIMEOUT, check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("p2rank_timeout") from error
    except OSError as error:
        raise RuntimeError(_FAILURE) from error
    if result.returncode != 0:
        raise RuntimeError(_FAILURE)
    return _read(output / f"{structure.name}_predictions.csv", _MAX_OUTPUT, _OUTPUT)


def predict(data: bytes, suffix: str, profile: str) -> bytes:
    """Run on private validated receptor bytes and return this invocation's prediction CSV."""
    if (
        type(data) is not bytes or not 0 < len(data) <= _MAX_INPUT
        or suffix not in _SUFFIXES or profile not in _PROFILES
    ):
        raise RuntimeError(_FAILURE)
    root, java = _verified_release(), _java()
    try:
        with tempfile.TemporaryDirectory(prefix="rfp2rank-") as directory:
            return _run(root, java, Path(directory), data, suffix, profile)
    except OSError as error:
        raise RuntimeError(_FAILURE) from error
