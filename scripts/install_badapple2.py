#!/usr/bin/env python3
"""Install the pinned Badapple 2.0 scaffold table as a local SQLite database."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sqlite3
import stat
import subprocess
import sys
import threading
from collections.abc import Iterator
from contextlib import closing, contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import BinaryIO
from urllib.request import urlopen

SNAPSHOT = "2025-08-08"
SCHEMA_VERSION = 1
DUMP_URL = "https://unmtid-dbs.net/download/Badapple2/badapple2.pgdump"
DUMP_BYTES = 134_146_465
DUMP_SHA256 = "3c01e4b00ab86892aa838e5e770f7ba861af04630c07cfb705506d1a18382c9c"
DATABASE = "scaffolds.sqlite"
RECEIPT = "receipt.json"
_MAX_EXPORT_BYTES = 1024 * 1024 * 1024
_MAX_LINE_BYTES = 8192
_RESTORE_TIMEOUT = 180
_COUNT_COLUMNS = (
    "ncpd_total", "ncpd_tested", "ncpd_active", "nsub_total", "nsub_tested", "nsub_active",
    "nass_tested", "nass_active", "nsam_tested", "nsam_active",
)
_TAIL_COLUMNS = (*_COUNT_COLUMNS, "in_drug", "pscore", "prank")
_COLUMNS = ("id", "scafsmi", *_TAIL_COLUMNS)
_COPY_COLUMNS = ("id", "scafsmi", "kekule_scafsmi", "scaftree", *_TAIL_COLUMNS)
_COPY_HEADER = f"COPY public.scaffold ({', '.join(_COPY_COLUMNS)}) FROM stdin;".encode("ascii")
_INSERT = f"INSERT INTO scaffold ({', '.join(_COLUMNS)}) VALUES ({', '.join('?' for _ in _COLUMNS)})"
_NULL = rb"\N"


class InstallError(ValueError):
    """A dump, tool, or destination outside the supported install contract."""


@contextmanager
def _dump_source(local: Path | None) -> Iterator[BinaryIO]:
    if local is None:
        with urlopen(DUMP_URL, timeout=60) as response:
            yield response
        return

    info = local.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_size != DUMP_BYTES:
        raise InstallError("the local dump must be a regular file of the pinned snapshot size")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    flags |= getattr(os, "O_BINARY", 0)
    with os.fdopen(os.open(local, flags), "rb") as source:
        if not os.path.samestat(info, os.fstat(source.fileno())):
            raise InstallError("the local dump changed before it could be opened")
        yield source


def _stage_dump(local: Path | None, dump: Path) -> None:
    digest = hashlib.sha256()
    with _dump_source(local) as source, dump.open("xb") as destination:
        remaining = DUMP_BYTES
        while remaining:
            chunk = source.read(min(remaining, 1024 * 1024))
            if not chunk:
                raise InstallError("the dump is shorter than the pinned snapshot")
            destination.write(chunk)
            digest.update(chunk)
            remaining -= len(chunk)
        if source.read(1):
            raise InstallError("the dump is larger than the pinned snapshot")
    if digest.hexdigest() != DUMP_SHA256:
        raise InstallError("the dump SHA-256 does not match the pinned snapshot")


@contextmanager
def _export(restore: Path, dump: Path) -> Iterator[BinaryIO]:
    command = [
        str(restore), "--data-only", "--schema=public", "--table=scaffold",
        "--no-owner", "--no-privileges", "--file=-", str(dump),
    ]
    timed_out = threading.Event()
    with subprocess.Popen(
        command, cwd=dump.parent, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    ) as process:
        def expire() -> None:
            if process.poll() is None:
                timed_out.set()
                process.kill()

        timer = threading.Timer(_RESTORE_TIMEOUT, expire)
        timer.start()
        try:
            assert process.stdout is not None
            yield process.stdout
            code = process.wait()
        except BaseException:
            if process.poll() is None:
                process.kill()
            if timed_out.is_set():
                raise InstallError("pg_restore exceeded the export timeout") from None
            raise
        finally:
            timer.cancel()
            timer.join()
        if timed_out.is_set():
            raise InstallError("pg_restore exceeded the export timeout")
        if code:
            raise InstallError(f"pg_restore could not export the scaffold table (exit {code})")


def _integer(raw: bytes) -> int | None:
    if raw == _NULL:
        return None
    if not raw.isdigit() or len(raw) > 10:
        raise InstallError("the scaffold export contains an invalid count or identifier")
    value = int(raw)
    if value > 2**31 - 1:
        raise InstallError("the scaffold export contains an invalid count or identifier")
    return value


def _row(line: bytes) -> tuple:
    raw = line.split(b"\t")
    if len(raw) != len(_COPY_COLUMNS):
        raise InstallError("the scaffold export contains a row with an invalid field count")
    fields = dict(zip(_COPY_COLUMNS, raw, strict=True))
    identifier = _integer(fields["id"])
    # In COPY text, a literal backslash in a SMILES bond is encoded as two.
    segments = fields["scafsmi"].split(rb"\\")
    smiles = b"\\".join(segments).decode("ascii")
    if identifier is None or not 0 < len(smiles) <= 512 or not smiles.isprintable() or (
        any(b"\\" in segment for segment in segments) or any(char.isspace() for char in smiles)
    ):
        raise InstallError("the scaffold export contains an invalid canonical key or identifier")

    raw_score = fields["pscore"]
    score = None if raw_score == _NULL else float(raw_score)
    if score is not None and (not math.isfinite(score) or score < 0):
        raise InstallError("the scaffold export contains an invalid pScore")
    in_drug = {b"t": 1, b"f": 0, _NULL: None}
    if fields["in_drug"] not in in_drug:
        raise InstallError("the scaffold export contains an invalid in_drug flag")
    counts = tuple(_integer(fields[key]) for key in _COUNT_COLUMNS)
    return identifier, smiles, *counts, in_drug[fields["in_drug"]], score, _integer(fields["prank"])


def _scaffold_rows(source: BinaryIO) -> Iterator[tuple]:
    started = finished = False
    total = 0
    while raw := source.readline(_MAX_LINE_BYTES + 1):
        total += len(raw)
        if total > _MAX_EXPORT_BYTES or len(raw) > _MAX_LINE_BYTES or not raw.endswith(b"\n"):
            raise InstallError("the scaffold export exceeded its stream limits or ended mid-line")
        line = raw.rstrip(b"\r\n")
        if started and not finished:
            if line == rb"\.":
                finished = True
            else:
                yield _row(line)
        elif line.startswith(b"COPY "):
            if started or line != _COPY_HEADER:
                raise InstallError("the exported COPY schema differs from the pinned scaffold table")
            started = True
    if not finished:
        raise InstallError("pg_restore did not provide a complete scaffold COPY section")


def _build_database(restore: Path, dump: Path, target: Path) -> int:
    count_schema = ", ".join(f"{name} INTEGER" for name in _COUNT_COLUMNS)
    with closing(sqlite3.connect(target)) as database, database:
        database.execute("PRAGMA page_size = 4096")
        database.execute("PRAGMA max_page_count = 524288")
        database.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        database.execute(
            f"CREATE TABLE scaffold (id INTEGER PRIMARY KEY, scafsmi TEXT COLLATE BINARY NOT NULL, "
            f"{count_schema}, in_drug INTEGER, pscore REAL, prank INTEGER) STRICT"
        )
        database.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL) STRICT")
        with _export(restore, dump) as source:
            database.executemany(_INSERT, _scaffold_rows(source))
        count = database.execute("SELECT COUNT(*) FROM scaffold").fetchone()[0]
        if not count:
            raise InstallError("the scaffold table is empty")
        database.execute("CREATE INDEX scaffold_scafsmi ON scaffold (scafsmi)")
        database.executemany("INSERT INTO metadata VALUES (?, ?)", (
            ("schema_version", str(SCHEMA_VERSION)), ("snapshot", SNAPSHOT),
            ("dump_sha256", DUMP_SHA256), ("row_count", str(count)),
        ))
    return count


def _require_new(path: Path) -> None:
    try:
        path.lstat()
    except FileNotFoundError:
        return
    raise InstallError("the snapshot destination already exists")


def install(local_dump: Path | None, pg_restore: Path | None) -> Path:
    executable = shutil.which("pg_restore") if pg_restore is None else pg_restore
    if executable is None:
        raise InstallError("install the PostgreSQL client tools or supply --pg-restore PATH")
    restore = Path(executable).resolve(strict=True)
    if not restore.is_file() or not os.access(restore, os.X_OK):
        raise InstallError("pg_restore must be an executable file")

    share = Path(sys.prefix) / "share"
    snapshots = share / "badapple2"
    for directory in (share, snapshots):
        directory.mkdir(exist_ok=True)
        if not stat.S_ISDIR(directory.lstat().st_mode) or directory.is_junction():
            raise InstallError("the installation parent must be a regular directory")

    destination = snapshots / SNAPSHOT
    _require_new(destination)
    lock = snapshots / f".install-{SNAPSHOT}.lock"
    try:
        lock.mkdir()
    except FileExistsError as error:
        raise InstallError("the snapshot installation is locked") from error
    try:
        with TemporaryDirectory(prefix=".badapple2-", dir=snapshots) as temporary:
            work = Path(temporary)
            dump = work / "badapple2.pgdump"
            _stage_dump(local_dump, dump)
            staged = work / SNAPSHOT
            staged.mkdir()
            target = staged / DATABASE
            count = _build_database(restore, dump, target)
            with target.open("rb") as source:
                sqlite_sha256 = hashlib.file_digest(source, "sha256").hexdigest()
            receipt = {
                "schema_version": SCHEMA_VERSION, "snapshot": SNAPSHOT, "dump_sha256": DUMP_SHA256,
                "sqlite_sha256": sqlite_sha256, "row_count": count,
            }
            with (staged / RECEIPT).open("x", encoding="utf-8", newline="\n") as file:
                json.dump(receipt, file, indent=2, sort_keys=True)
                file.write("\n")
            _require_new(destination)
            staged.rename(destination)
    finally:
        lock.rmdir()
    return destination / DATABASE


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", type=Path, help="use the official slim Badapple 2.0 dump already on disk")
    parser.add_argument("--pg-restore", type=Path, help="path to the PostgreSQL pg_restore client")
    args = parser.parse_args()
    try:
        database = install(args.dump, args.pg_restore)
    except (OSError, ValueError, sqlite3.Error, subprocess.SubprocessError) as error:
        parser.exit(1, f"Badapple 2.0 installation failed: {error}\n")
    print(f"Installed Badapple 2.0 snapshot {SNAPSHOT} at {database}")


if __name__ == "__main__":
    main()
