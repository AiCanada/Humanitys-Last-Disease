#!/usr/bin/env python3
"""Install the checksum-verified public fastsolv ensemble in this environment."""

from __future__ import annotations

import argparse
import os
import shutil
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import BinaryIO
from urllib.request import urlopen

from rfproteina.disease import fastsolv_runtime as runtime


@contextmanager
def _source(name: str, local: Path | None) -> Iterator[BinaryIO]:
    if local is None:
        url = f"https://zenodo.org/records/{runtime.ZENODO_RELEASE}/files/{name}"
        with urlopen(url, timeout=60) as stream:
            yield stream
        return
    path = local / name
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_size != runtime.CHECKPOINT_SIZE:
        raise ValueError(f"The local checkpoint {name} has the wrong type or size.")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0)
    with os.fdopen(os.open(path, flags), "rb") as stream:
        if not os.path.samestat(info, os.fstat(stream.fileno())):
            raise ValueError(f"The local checkpoint {name} changed before it could be opened.")
        yield stream


def install(local: Path | None = None) -> tuple[Path, ...]:
    destination = runtime.release_directory()
    if destination.exists() or destination.is_symlink():
        return runtime.verify_directory(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not stat.S_ISDIR(destination.parent.lstat().st_mode):
        raise ValueError("The destination must be a regular directory.")
    required = len(runtime.CHECKPOINT_MD5) * runtime.CHECKPOINT_SIZE
    if shutil.disk_usage(destination.parent).free < required:
        raise ValueError("The destination lacks space for the verified four-checkpoint ensemble.")
    with TemporaryDirectory(prefix=".fastsolv-", dir=destination.parent) as temporary:
        staging = Path(temporary) / "release"
        staging.mkdir()
        for name in runtime.CHECKPOINT_MD5:
            with _source(name, local) as source, (staging / name).open("xb") as output:
                remaining = runtime.CHECKPOINT_SIZE
                while remaining:
                    block = source.read(min(remaining, 1024 * 1024))
                    if not block:
                        raise ValueError(f"The checkpoint {name} is shorter than the release size.")
                    output.write(block)
                    remaining -= len(block)
                if source.read(1):
                    raise ValueError(f"The checkpoint {name} is longer than the release size.")
        runtime.verify_directory(staging)
        try:
            staging.rename(destination)
        except OSError:
            # A concurrent installer can finish first. Accept only the exact same release.
            if not destination.exists():
                raise
    return runtime.verify_directory(destination)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, help="use the four official files from this local directory")
    args = parser.parse_args()
    try:
        files = install(args.checkpoints)
    except (ValueError, RuntimeError, OSError) as error:
        print(f"fastsolv installation failed ({type(error).__name__}): {error}")
        return 2
    print(f"Verified {len(files)} fastsolv checkpoints in the active Python environment.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
