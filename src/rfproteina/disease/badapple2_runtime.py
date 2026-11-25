"""Read the verified Badapple 2 scaffold snapshot and run its local HierS query."""

from __future__ import annotations

import json
import os
import re
import sqlite3
import stat
import subprocess
import sys
from contextlib import closing
from hashlib import file_digest
from pathlib import Path

from rfproteina.disease.ligand_inputs import read_regular

SNAPSHOT = "2025-08-08"
DUMP_SHA256 = "3c01e4b00ab86892aa838e5e770f7ba861af04630c07cfb705506d1a18382c9c"
API_REVISION = "1b56cb729d7cbaed46d1275fc52cd1f7d37d672e"
RDKIT = "2024.3.3"
SCAFFOLDGRAPH = "1.1.1"
MAX_SCAFFOLDS = 128
MAX_CANDIDATES = 32
MAX_SMILES = 512
COUNT_FIELDS = tuple(f"{stem}_{kind}" for stem in ("nass", "ncpd", "nsub", "nsam") for kind in ("active", "tested"))
_SQL = "SELECT id, pscore, in_drug, " + ", ".join(COUNT_FIELDS) + " FROM scaffold WHERE scafsmi = ? LIMIT 2"
_HASH = re.compile(r"[a-f0-9]{64}")
_WORKER_BLOCKERS = frozenset({
    "badapple2_scaffold_engine_unavailable", "badapple2_scaffold_unsupported", "badapple2_scaffold_limit",
})
BLOCKERS = _WORKER_BLOCKERS | {
    "badapple2_database_unavailable", "badapple2_database_integrity", "badapple2_timeout",
    "badapple2_provider_failure", "badapple2_invalid_report",
}


class BadappleRuntimeError(ValueError):
    """An offline asset, chemistry worker, or response could not be used."""


def _root() -> Path:
    return Path(sys.prefix) / "share" / "badapple2"


def _python() -> Path:
    name = ("Scripts", "python.exe") if os.name == "nt" else ("bin", "python")
    return (_root() / "chem-env").joinpath(*name)


def _unique(pairs: list[tuple[str, object]]) -> dict:
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("Duplicate JSON field.")
    return result


def _signature(info: os.stat_result) -> tuple[int, int, int, int, int]:
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def _database() -> tuple[Path, int, tuple[int, int, int, int, int]]:
    directory = _root() / SNAPSHOT
    path = directory / "scaffolds.sqlite"
    try:
        for location in (_root(), directory):
            if not stat.S_ISDIR(location.lstat().st_mode):
                raise BadappleRuntimeError("badapple2_database_integrity")
        receipt = json.loads(
            read_regular(directory / "receipt.json", 4096, "badapple2_receipt", BadappleRuntimeError),
            object_pairs_hook=_unique,
        )
        if not (
            isinstance(receipt, dict)
            and receipt.keys() == {"schema_version", "snapshot", "dump_sha256", "sqlite_sha256", "row_count"}
            and type(receipt["schema_version"]) is int and receipt["schema_version"] == 1
            and receipt["snapshot"] == SNAPSHOT and receipt["dump_sha256"] == DUMP_SHA256
            and isinstance(receipt["sqlite_sha256"], str) and _HASH.fullmatch(receipt["sqlite_sha256"])
            and type(receipt["row_count"]) is int and 0 < receipt["row_count"] <= 2_000_000
        ):
            raise BadappleRuntimeError("badapple2_database_integrity")
        info = path.lstat()
        snapshot = _signature(info)
        if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= 512 * 1024 * 1024:
            raise BadappleRuntimeError("badapple2_database_integrity")
        if _file_sha256(path, snapshot) != receipt["sqlite_sha256"]:
            raise BadappleRuntimeError("badapple2_database_integrity")
    except FileNotFoundError as error:
        raise BadappleRuntimeError("badapple2_database_unavailable") from error
    except BadappleRuntimeError as error:
        if str(error) == "badapple2_receipt_missing":
            raise BadappleRuntimeError("badapple2_database_unavailable") from error
        raise BadappleRuntimeError("badapple2_database_integrity") from error
    except (OSError, ValueError, RecursionError) as error:
        raise BadappleRuntimeError("badapple2_database_integrity") from error
    return path, receipt["row_count"], snapshot


def _file_sha256(path: Path, snapshot: tuple[int, int, int, int, int]) -> str:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
    with os.fdopen(os.open(path, flags), "rb") as stream:
        before = os.fstat(stream.fileno())
        if _signature(before) != snapshot:
            raise BadappleRuntimeError("badapple2_database_integrity")
        digest = file_digest(stream, "sha256").hexdigest()
        after = os.fstat(stream.fileno())
        if _signature(after) != snapshot:
            raise BadappleRuntimeError("badapple2_database_integrity")
    return digest


def _chemistry_available() -> bool:
    probe = "import importlib.metadata as m; print(m.version('rdkit'), m.version('scaffoldgraph'))"
    try:
        result = subprocess.run(
            [str(_python()), "-I", "-B", "-c", probe], capture_output=True, timeout=8, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0 and result.stdout.strip() == f"{RDKIT} {SCAFFOLDGRAPH}".encode()


def installed() -> bool:
    try:
        _database()
    except BadappleRuntimeError:
        return False
    return _chemistry_available()


def valid_scaffold(value: object) -> bool:
    return (
        isinstance(value, str) and 0 < len(value) <= MAX_SMILES and value.isascii() and value.isprintable()
        and not any(char in value for char in " |.\t")
        and re.fullmatch(r"[A-Za-z0-9@+\-#=()\[\]%/\\:*]+", value) is not None
    )


def _scaffolds(inputs: list[str]) -> dict:
    if not _python().is_file():
        raise BadappleRuntimeError("badapple2_scaffold_engine_unavailable")
    program = Path(__file__).with_name("badapple2_scaffolds.py").resolve()
    try:
        result = subprocess.run(
            [str(_python()), "-I", "-B", str(program)], input=json.dumps(inputs, ensure_ascii=True).encode("ascii"),
            capture_output=True, check=True, timeout=60,
        )
    except subprocess.TimeoutExpired as error:
        raise BadappleRuntimeError("badapple2_timeout") from error
    except (OSError, subprocess.CalledProcessError) as error:
        raise BadappleRuntimeError("badapple2_provider_failure") from error
    try:
        if len(result.stdout) > 4 * 1024 * 1024:
            raise ValueError("Too many output bytes.")
        reply = json.loads(result.stdout, object_pairs_hook=_unique)
        if not isinstance(reply, dict) or reply.keys() != set(inputs):
            raise ValueError("Mismatched candidates.")
    except (ValueError, RecursionError) as error:
        raise BadappleRuntimeError("badapple2_invalid_report") from error
    return reply


def _read_matches(database: sqlite3.Connection, scaffold: str) -> list[dict]:
    rows = []
    for raw in database.execute(_SQL, (scaffold,)):
        row = dict(raw)
        if type(row["in_drug"]) is int and row["in_drug"] in (0, 1):
            row["in_drug"] = bool(row["in_drug"])
        rows.append(row)
    return rows


def lookup(inputs: list[str]) -> dict:
    reply = _scaffolds(inputs)
    path, row_count, snapshot = _database()
    matches: dict[str, list[dict]] = {}
    results = {}
    try:
        with closing(sqlite3.connect(path.as_uri() + "?mode=ro&immutable=1", uri=True)) as database:
            database.execute("PRAGMA query_only=ON")
            database.execute("PRAGMA trusted_schema=OFF")
            metadata = dict(database.execute("SELECT key, value FROM metadata"))
            if metadata != {
                "schema_version": "1", "snapshot": SNAPSHOT, "dump_sha256": DUMP_SHA256,
                "row_count": str(row_count),
            }:
                raise BadappleRuntimeError("badapple2_database_integrity")
            database.row_factory = sqlite3.Row
            for parent, raw in reply.items():
                if not isinstance(raw, dict):
                    results[parent] = {"blocker": "badapple2_invalid_report"}
                    continue
                blocker, keys = raw.get("blocker"), raw.get("keys")
                if raw.keys() == {"blocker"} and isinstance(blocker, str) and blocker in _WORKER_BLOCKERS:
                    results[parent] = {"blocker": blocker}
                elif (
                    raw.keys() == {"keys"} and isinstance(keys, list) and len(keys) <= MAX_SCAFFOLDS
                    and all(valid_scaffold(key) for key in keys) and len(set(keys)) == len(keys)
                ):
                    rows = []
                    for scaffold in keys:
                        if scaffold not in matches:
                            matches[scaffold] = _read_matches(database, scaffold)
                        rows.append({"scafsmi": scaffold, "matches": matches[scaffold]})
                    results[parent] = {"scaffolds": rows}
                else:
                    results[parent] = {"blocker": "badapple2_invalid_report"}
        if _signature(path.lstat()) != snapshot:
            raise BadappleRuntimeError("badapple2_database_integrity")
    except (sqlite3.Error, OSError) as error:
        raise BadappleRuntimeError("badapple2_database_integrity") from error
    return results
