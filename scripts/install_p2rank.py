#!/usr/bin/env python3
"""Install the checksum-verified P2Rank release in the current Python environment."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import stat
import sys
import tarfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path, PureWindowsPath
from tempfile import TemporaryDirectory
from typing import BinaryIO
from urllib.request import urlopen

VERSION = "2.5.1"
ARCHIVE_ROOT = f"p2rank_{VERSION}"
ARCHIVE_URL = f"https://github.com/rdk/p2rank/releases/download/{VERSION}/{ARCHIVE_ROOT}.tar.gz"
ARCHIVE_BYTES = 275_625_956
ARCHIVE_SHA256 = "d243f2d9036ac053fefb9407b5fe1c85f4fe077c519fd975ac585e995feab274"
RECEIPT = "rfproteina-install.json"
_MAX_EXPANDED_BYTES = 768 * 1024 * 1024
_MAX_MEMBERS = 4096
_CHUNK = 1024 * 1024
_REQUIRED = {"bin/p2rank.jar", "config/default.groovy", "models/default/model.zst"}


class InstallError(ValueError):
    """A release or destination outside the pinned installation contract."""


@contextmanager
def _archive_source(local_archive: Path | None) -> Iterator[BinaryIO]:
    if local_archive is None:
        with urlopen(ARCHIVE_URL, timeout=60) as response:
            yield response
        return

    info = local_archive.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_size != ARCHIVE_BYTES:
        raise InstallError("the local archive must be a regular file of the pinned release size")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    flags |= getattr(os, "O_BINARY", 0)
    with os.fdopen(os.open(local_archive, flags), "rb") as source:
        if not os.path.samestat(info, os.fstat(source.fileno())):
            raise InstallError("the local archive changed before it could be opened")
        yield source


def _copy(source: BinaryIO, destination: BinaryIO, size: int) -> str:
    digest = hashlib.sha256()
    remaining = size
    while remaining:
        data = source.read(min(remaining, _CHUNK))
        if not data:
            raise InstallError("the archive or member was shorter than expected")
        destination.write(data)
        digest.update(data)
        remaining -= len(data)
    if source.read(1):
        raise InstallError("the archive or member was longer than expected")
    return digest.hexdigest()


class _BoundedTarInput:
    """Limit decompressed bytes, including tar headers and extension records."""

    def __init__(self, source: BinaryIO):
        self._source = source
        self._remaining = _MAX_EXPANDED_BYTES

    def read(self, size: int) -> bytes:
        data = self._source.read(min(size, self._remaining + 1))
        self._remaining -= len(data)
        if self._remaining < 0:
            raise InstallError("the expanded archive exceeded its size limit")
        return data


def _member_parts(member: tarfile.TarInfo) -> tuple[str, ...]:
    name = member.name.removesuffix("/") if member.isdir() else member.name
    parts = name.split("/")
    if len(name) > 512 or parts[0] != ARCHIVE_ROOT or any(
        not part or part in (".", "..") or part != part.strip() or part.endswith(".")
        or not part.isprintable() or any(char in part for char in '<>:"\\|?*')
        or PureWindowsPath(part).is_reserved()
        for part in parts
    ):
        raise InstallError("the archive contains an unsafe path")
    return tuple(parts[1:])


def _unpack(archive: Path, staging: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    seen: set[str] = set()
    total = 0
    with gzip.open(archive, "rb") as decoded, tarfile.open(fileobj=_BoundedTarInput(decoded), mode="r|") as bundle:
        for count, member in enumerate(bundle, 1):
            if count > _MAX_MEMBERS:
                raise InstallError("the archive contains too many members")
            parts = _member_parts(member)
            relative = "/".join(parts)
            normalized = relative.casefold()
            if normalized in seen or normalized == RECEIPT.casefold():
                raise InstallError("the archive contains a duplicate or reserved path")
            seen.add(normalized)
            target = staging.joinpath(*parts)
            if member.isdir() and member.size == 0:
                target.mkdir(exist_ok=True)
                continue
            if (
                not parts or member.type not in (tarfile.REGTYPE, tarfile.AREGTYPE)
                or not 0 <= member.size <= _MAX_EXPANDED_BYTES - total
            ):
                raise InstallError("the archive contains an unsupported or oversized member")
            source = bundle.extractfile(member)
            if source is None:
                raise InstallError("the archive file could not be read")
            with source, target.open("xb") as destination:
                hashes[relative] = _copy(source, destination, member.size)
            if member.mode & 0o111:
                target.chmod(target.stat().st_mode | 0o100)
            total += member.size
    if not _REQUIRED.issubset(hashes) or not any(
        name.startswith("bin/lib/") and name.endswith(".jar") for name in hashes
    ):
        raise InstallError("the archive is missing files required by the pinned runtime")
    return hashes


def _require_new(destination: Path) -> None:
    try:
        destination.lstat()
    except FileNotFoundError:
        return
    raise InstallError("the release destination already exists")


def install(local_archive: Path | None) -> Path:
    prefix = Path(sys.prefix)
    share = prefix / "share"
    releases = share / "p2rank"
    for directory in (share, releases):
        directory.mkdir(exist_ok=True)
        if not stat.S_ISDIR(directory.lstat().st_mode) or directory.is_junction():
            raise InstallError("the installation parent must be a regular directory")

    destination = releases / VERSION
    _require_new(destination)
    lock = releases / f".install-{VERSION}.lock"
    try:
        lock.mkdir()
    except FileExistsError as error:
        raise InstallError("the release installation is locked") from error
    try:
        with TemporaryDirectory(prefix=".p2rank-", dir=releases) as temporary:
            work = Path(temporary)
            archive = work / "release.tar.gz"
            with _archive_source(local_archive) as source, archive.open("xb") as staging_file:
                if _copy(source, staging_file, ARCHIVE_BYTES) != ARCHIVE_SHA256:
                    raise InstallError("the archive SHA-256 did not match the pinned release")
            staging = work / VERSION
            staging.mkdir()
            files = _unpack(archive, staging)
            receipt = {"version": VERSION, "archive_sha256": ARCHIVE_SHA256, "files": files}
            with (staging / RECEIPT).open("x", encoding="utf-8", newline="\n") as record:
                json.dump(receipt, record, indent=2, sort_keys=True)
                record.write("\n")
            _require_new(destination)
            staging.rename(destination)
    finally:
        lock.rmdir()
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="use the official release archive already on disk")
    args = parser.parse_args()
    try:
        target = install(args.archive)
    except (InstallError, OSError, tarfile.TarError, EOFError) as error:
        parser.exit(1, f"P2Rank installation failed: {error}\n")
    print(f"Installed P2Rank {VERSION} at {target}")


if __name__ == "__main__":
    main()
