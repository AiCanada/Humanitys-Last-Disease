#!/usr/bin/env python3
"""Install the pinned OpenAntigens human-target evidence in this Python environment."""

from __future__ import annotations

import argparse
import os
import stat
from http.client import HTTPMessage
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import BinaryIO
from urllib.request import HTTPRedirectHandler, Request, build_opener

from rfproteina.disease import openantigens_snapshot as snapshot
from rfproteina.disease.ligand_inputs import read_regular


class InstallError(ValueError):
    """The installation could not be verified or safely completed."""


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(
        self, req: Request, fp: BinaryIO, code: int, msg: str, headers: HTTPMessage, newurl: str
    ) -> None:
        return None


def _read_asset(url: str, size: int, local: Path | None) -> bytes:
    if local is not None:
        return read_regular(local, size, "local_asset", InstallError)
    with build_opener(_NoRedirect()).open(url, timeout=30) as source:
        return source.read(size + 1)


def _existing(destination: Path) -> bool:
    try:
        destination.lstat()
    except FileNotFoundError:
        return False
    snapshot.verify_directory(destination)
    return True


def install(*, index: Path | None = None, accession_map: Path | None = None, metadata: Path | None = None) -> Path:
    """Install upstream assets from their official URLs or supplied local files."""
    destination = snapshot.release_directory()
    for directory in (destination.parent.parent, destination.parent):
        directory.mkdir(exist_ok=True)
        if not stat.S_ISDIR(directory.lstat().st_mode):
            raise InstallError("the installation parent must be a regular directory")
    if _existing(destination):
        return destination

    local_assets = {
        snapshot.INDEX_NAME: index,
        snapshot.ACCESSION_MAP_NAME: accession_map,
        snapshot.METADATA_NAME: metadata,
    }
    lock = destination.parent / f".install-{snapshot.RELEASE}.lock"
    try:
        lock.mkdir()
    except FileExistsError as exc:
        raise InstallError("the release installation is locked") from exc
    try:
        if not _existing(destination):
            with TemporaryDirectory(prefix=".openantigens-", dir=destination.parent) as temporary:
                staged = Path(temporary) / snapshot.RELEASE
                staged.mkdir()
                for name, url, size, _ in snapshot.ASSETS:
                    data = _read_asset(url, size, local_assets[name])
                    if len(data) != size:
                        raise InstallError(f"the pinned release size did not match {name}")
                    with (staged / name).open("xb") as target:
                        target.write(data)
                        target.flush()
                        os.fsync(target.fileno())
                snapshot.verify_directory(staged)
                if not _existing(destination):
                    staged.rename(destination)
    finally:
        lock.rmdir()
    return destination


def main() -> None:
    """Run the OpenAntigens evidence installation command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, help="use an already downloaded portal index")
    parser.add_argument("--accession-map", type=Path, help="use an already downloaded human accession map")
    parser.add_argument("--metadata", type=Path, help="use an already downloaded portal metadata file")
    args = parser.parse_args()
    try:
        destination = install(index=args.index, accession_map=args.accession_map, metadata=args.metadata)
    except (InstallError, OSError, snapshot.SnapshotError) as exc:
        parser.exit(1, f"OpenAntigens installation failed: {exc}\n")
    print(f"Installed OpenAntigens {snapshot.RELEASE} at {destination}")


if __name__ == "__main__":
    main()
