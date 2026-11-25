#!/usr/bin/env python3
"""Install the checksum-verified MetFrag jar in this Linux Python environment."""

from __future__ import annotations

import argparse
import hashlib
import os
import stat
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import BinaryIO
from urllib.request import urlopen

from rfproteina.disease import metfrag_runtime


class InstallError(ValueError):
    """The release asset or destination failed verification."""


@contextmanager
def _source(local_jar: Path | None) -> Iterator[BinaryIO]:
    if local_jar is None:
        with urlopen(metfrag_runtime.JAR_URL, timeout=60) as download:
            yield download
        return
    info = local_jar.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_size != metfrag_runtime.JAR_BYTES:
        raise InstallError("the local jar must be a regular file of the release size")
    with os.fdopen(os.open(local_jar, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as source:
        if not os.path.samestat(info, os.fstat(source.fileno())):
            raise InstallError("the local jar changed before it could be opened")
        yield source


def _copy(local_jar: Path | None, destination: Path) -> None:
    remaining = metfrag_runtime.JAR_BYTES
    digest = hashlib.sha256()
    with _source(local_jar) as source, destination.open("xb") as target:
        while remaining:
            chunk = source.read(min(remaining, 1024 * 1024))
            if not chunk:
                raise InstallError("the jar was shorter than expected")
            target.write(chunk)
            digest.update(chunk)
            remaining -= len(chunk)
        if source.read(1) or digest.hexdigest() != metfrag_runtime.JAR_SHA256:
            raise InstallError("the jar size or SHA-256 did not match the pinned release")
        target.flush()
        os.fsync(target.fileno())


def _existing(destination: Path) -> bool:
    try:
        destination.lstat()
    except FileNotFoundError:
        return False
    try:
        metfrag_runtime._verified_jar()
    except metfrag_runtime.MetfragRuntimeError as error:
        raise InstallError("an unverified installation destination already exists") from error
    return True


def install(local_jar: Path | None = None) -> Path:
    """Install the exact release without replacing an existing installation."""
    if sys.platform != "linux":
        raise InstallError("this MetFrag integration requires Linux or WSL network isolation")
    destination = metfrag_runtime.release_directory()
    for directory in (destination.parent.parent, destination.parent):
        directory.mkdir(exist_ok=True)
        if not stat.S_ISDIR(directory.lstat().st_mode):
            raise InstallError("the installation parent must be a regular directory")
    if _existing(destination):
        return destination
    lock = destination.parent / f".install-{metfrag_runtime.RELEASE}.lock"
    try:
        lock.mkdir()
    except FileExistsError as error:
        raise InstallError("the release installation is locked") from error
    try:
        if not _existing(destination):
            with TemporaryDirectory(prefix=".metfrag-", dir=destination.parent) as temporary:
                staging = Path(temporary) / metfrag_runtime.RELEASE
                staging.mkdir()
                _copy(local_jar, staging / metfrag_runtime.JAR_NAME)
                if not _existing(destination):
                    staging.rename(destination)
    finally:
        lock.rmdir()
    return destination


def main() -> None:
    """Run the MetFrag installer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jar", type=Path, help="use the official release jar already on disk")
    args = parser.parse_args()
    try:
        destination = install(args.jar)
    except (InstallError, OSError) as error:
        parser.exit(1, f"MetFrag installation failed: {error}\n")
    print(f"Installed MetFrag {metfrag_runtime.RELEASE} at {destination}")


if __name__ == "__main__":
    main()
