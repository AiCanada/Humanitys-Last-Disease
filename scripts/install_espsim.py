#!/usr/bin/env python3
"""Install the pinned ESP-Sim analytical scorer and its MIT license."""

from __future__ import annotations

import argparse
import hashlib
import io
import os
import stat
import tarfile
from http.client import HTTPMessage
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import BinaryIO
from urllib.request import HTTPRedirectHandler, Request, build_opener

from rfproteina.disease import espsim_runtime
from rfproteina.disease.ligand_inputs import read_regular


class InstallError(ValueError):
    """The published source or installation location failed verification."""


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(
        self, req: Request, fp: BinaryIO, code: int, msg: str, headers: HTTPMessage, newurl: str
    ) -> None:
        return None


def _release(archive: Path | None) -> bytes:
    size = espsim_runtime.ARCHIVE_BYTES
    if archive is None:
        with build_opener(_NoRedirect()).open(espsim_runtime.ARCHIVE_URL, timeout=45) as source:
            data = source.read(size + 1)
    else:
        data = read_regular(archive, size, "espsim_archive", InstallError)
    if len(data) != size or hashlib.sha256(data).hexdigest() != espsim_runtime.ARCHIVE_SHA256:
        raise InstallError("source archive size or SHA-256 did not match the pinned release")
    return data


def _unpack(data: bytes, destination: Path) -> None:
    expected = {
        f"espsim-{espsim_runtime.RELEASE}/{name}": (name, size, digest)
        for name, (size, digest) in espsim_runtime.FILES.items()
    }
    (destination / "espsim").mkdir()
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as release:
        seen = set()
        for member in release:
            if member.name not in expected:
                continue
            if member.name in seen:
                raise InstallError("duplicate source member in the release")
            seen.add(member.name)
            name, size, digest = expected[member.name]
            if member.type not in {tarfile.REGTYPE, tarfile.AREGTYPE} or member.size != size:
                raise InstallError(f"unexpected archive entry for {name}")
            source = release.extractfile(member)
            if source is None:
                raise InstallError(f"cannot read archive entry for {name}")
            with source:
                raw = source.read(size + 1)
            if len(raw) != size or hashlib.sha256(raw).hexdigest() != digest:
                raise InstallError(f"archive entry size or SHA-256 did not match {name}")
            with (destination / name).open("xb") as target:
                target.write(raw)
                target.flush()
                os.fsync(target.fileno())
        if seen != expected.keys():
            raise InstallError("the release was missing required source files")


def _existing(destination: Path) -> bool:
    try:
        destination.lstat()
    except FileNotFoundError:
        return False
    if not espsim_runtime.installed():
        raise InstallError("an unverified ESP-Sim destination already exists")
    return True


def install(archive: Path | None = None) -> Path:
    """Install the verified scoring source under the active Python prefix."""
    destination = espsim_runtime.release_directory()
    for directory in (destination.parent.parent, destination.parent):
        directory.mkdir(exist_ok=True)
        if not stat.S_ISDIR(directory.lstat().st_mode):
            raise InstallError("the installation parent must be a regular directory")
    if _existing(destination):
        return destination
    lock = destination.parent / f".install-{espsim_runtime.RELEASE}.lock"
    try:
        lock.mkdir()
    except FileExistsError as exc:
        raise InstallError("the ESP-Sim installation is locked") from exc
    try:
        if not _existing(destination):
            data = _release(archive)
            with TemporaryDirectory(prefix=".espsim-", dir=destination.parent) as name:
                staging = Path(name) / espsim_runtime.RELEASE
                staging.mkdir()
                _unpack(data, staging)
                if not _existing(destination):
                    staging.rename(destination)
    finally:
        lock.rmdir()
    return destination


def main() -> None:
    """Install from the pinned PyPI archive or an already downloaded copy."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="path to the exact published source archive")
    args = parser.parse_args()
    try:
        destination = install(args.archive)
    except (InstallError, OSError, EOFError, tarfile.TarError) as exc:
        parser.exit(1, f"ESP-Sim installation failed: {exc}\n")
    print(f"Installed ESP-Sim {espsim_runtime.RELEASE} at {destination}")


if __name__ == "__main__":
    main()
