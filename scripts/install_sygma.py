#!/usr/bin/env python
"""Install checksum-verified SyGMa rule data for local screening."""

import argparse
import os
import tempfile
from pathlib import Path
from urllib.request import urlopen

from rfproteina.disease.sygma_engine import ARCHIVE_URL, ScreenError, archive_path, read_archive


def install(source: Path | None) -> Path:
    if source is None:
        with urlopen(ARCHIVE_URL, timeout=60) as response:
            data = response.read(1024 * 1024)
    else:
        with source.open("rb") as file:
            data = file.read(1024 * 1024)
    read_archive(data)
    target = archive_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent) as staging:
        tmp = Path(staging) / target.name
        with tmp.open("xb") as file:
            file.write(data)
            file.flush()
            os.fsync(file.fileno())
        os.replace(tmp, target)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-archive", type=Path, help="use the released SyGMa 1.1.0 source archive from disk")
    args = parser.parse_args()
    try:
        target = install(args.from_archive)
    except (OSError, ScreenError) as error:
        parser.exit(1, f"SyGMa installation failed: {error}\n")
    print(f"Installed both SyGMa rule tables at {target}")


if __name__ == "__main__":
    main()
