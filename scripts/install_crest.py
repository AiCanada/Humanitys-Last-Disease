#!/usr/bin/env python3
"""Install the checksum-verified Linux CREST release in this Python environment."""

from __future__ import annotations

import argparse
import hashlib
import io
import lzma
import os
import stat
import tarfile
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlopen

from rfproteina.disease import crest_runtime

_MEMBERS = {
    "crest/crest": crest_runtime.EXECUTABLE_BYTES,
    "crest/LICENSE": 35_149,
    "crest/LICENSE.LESSER": 7_652,
}


class InstallError(ValueError):
    """The release asset or destination failed verification."""


def _read_archive(archive: Path | None) -> bytes:
    size = crest_runtime.ARCHIVE_BYTES
    if archive is None:
        with urlopen(crest_runtime.ARCHIVE_URL, timeout=60) as source:
            data = source.read(size + 1)
    else:
        info = archive.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_size != size:
            raise InstallError("the local archive must be a regular file of the release size")
        with os.fdopen(os.open(archive, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as source:
            if not os.path.samestat(info, os.fstat(source.fileno())):
                raise InstallError("the local archive changed before it could be opened")
            data = source.read(size + 1)
    if len(data) != size or hashlib.sha256(data).hexdigest() != crest_runtime.ARCHIVE_SHA256:
        raise InstallError("the archive size or SHA-256 did not match the pinned release")
    return data


def _unpack(data: bytes, destination: Path) -> None:
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:xz") as bundle:
        members = bundle.getmembers()
        by_name = {member.name: member for member in members}
        if (
            len(members) != len(_MEMBERS) + 1
            or by_name.keys() != {"crest", *_MEMBERS}
            or not by_name["crest"].isdir()
            or by_name["crest"].size != 0
        ):
            raise InstallError("the archive had an unexpected member layout")
        for name, size in _MEMBERS.items():
            member = by_name[name]
            if member.type not in {tarfile.REGTYPE, tarfile.AREGTYPE} or member.size != size:
                raise InstallError("the archive contained an unexpected file")
            source = bundle.extractfile(member)
            if source is None:
                raise InstallError("an archive file could not be read")
            path = destination / name.split("/")[1]
            digest = hashlib.sha256()
            count = 0
            with source, path.open("xb") as target:
                while chunk := source.read(1024 * 1024):
                    count += len(chunk)
                    if count > size:
                        raise InstallError("an archive file exceeded its expected size")
                    digest.update(chunk)
                    target.write(chunk)
                if count != size:
                    raise InstallError("an archive file was shorter than expected")
                if name == "crest/crest":
                    if digest.hexdigest() != crest_runtime.EXECUTABLE_SHA256:
                        raise InstallError("the executable SHA-256 did not match the pinned release")
                    os.fchmod(target.fileno(), 0o755)
                target.flush()
                os.fsync(target.fileno())


def _existing(destination: Path) -> bool:
    try:
        destination.lstat()
    except FileNotFoundError:
        return False
    if not crest_runtime.installed():
        raise InstallError("an unverified destination already exists")
    return True


def install(archive: Path | None = None) -> Path:
    """Install CREST under the current Linux Python prefix."""
    if not crest_runtime.supported_platform():
        raise InstallError("the pinned CREST release requires Linux or WSL on x86-64")
    destination = crest_runtime.release_directory()
    for directory in (destination.parent.parent, destination.parent):
        directory.mkdir(exist_ok=True)
        if not stat.S_ISDIR(directory.lstat().st_mode):
            raise InstallError("the installation parent must be a regular directory")
    if _existing(destination):
        return destination
    lock = destination.parent / f".install-{crest_runtime.RELEASE}.lock"
    try:
        lock.mkdir()
    except FileExistsError as error:
        raise InstallError("the release installation is locked") from error
    try:
        if not _existing(destination):
            data = _read_archive(archive)
            with TemporaryDirectory(prefix=".crest-", dir=destination.parent) as temporary:
                staging = Path(temporary) / crest_runtime.RELEASE
                staging.mkdir()
                _unpack(data, staging)
                if not _existing(destination):
                    staging.rename(destination)
    finally:
        lock.rmdir()
    return destination


def main() -> None:
    """Run the CREST installation command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="use the official release archive already on disk")
    args = parser.parse_args()
    try:
        destination = install(args.archive)
    except (InstallError, OSError, tarfile.TarError, EOFError, lzma.LZMAError) as error:
        parser.exit(1, f"CREST installation failed: {error}\n")
    print(f"Installed CREST {crest_runtime.RELEASE} at {destination}")


if __name__ == "__main__":
    main()
