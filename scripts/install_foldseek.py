#!/usr/bin/env python3
"""Install the Foldseek static binary, and optionally one structure database.

Foldseek is GPL-3.0 and its pre-built databases are CC-BY-4.0. Both are run
here, never redistributed: the binary lands under ``DATA_PATH/foldseek``, which
is outside the checkout, and nothing in this script uploads anything.

The binary goes to ``DATA_PATH/foldseek/foldseek``, which is where
``rfproteina.disease.novelty`` looks after ``FOLDSEEK`` and before ``PATH``.
The official tarball unpacks to ``foldseek/bin/foldseek``; the executable is
lifted out so the install matches the MMseqs2 one beside it.

A database is a separate, explicit step, because it is the part that costs
gigabytes:

    python scripts/install_foldseek.py                     # the binary alone
    python scripts/install_foldseek.py --list-databases
    python scripts/install_foldseek.py --database PDB
    python scripts/install_foldseek.py --have

**No database over 25 GB is downloaded.** That is this project's install limit,
and it is enforced here rather than left to care: ``Alphafold/UniProt50`` is
114.1 GB and its name gives no hint of that. The size is read from the figure
``novelty.FOLDSEEK_DATABASES`` records for each set, and a size that cannot be
read refuses the download rather than guessing it is small.

AVX2 is used when the CPU reports it, otherwise the SSE4.1 build. There is no
SSE2 build published.
"""

from __future__ import annotations

import argparse
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

_AVX2 = "https://mmseqs.com/foldseek/foldseek-linux-avx2.tar.gz"
_SSE41 = "https://mmseqs.com/foldseek/foldseek-linux-sse41.tar.gz"

#: The project's install limit, in gigabytes, applied to a database download.
DOWNLOAD_CAP_GB = 25.0

#: ``"pdb100.tar.gz 2.2 GB"`` -> 2.2. Anchored on the unit so a version number
#: in the file name cannot be read as the size.
_SIZE = re.compile(r"(\d+(?:\.\d+)?)\s*(GB|MB|TB)\b", re.IGNORECASE)
_SCALE = {"MB": 1 / 1024, "GB": 1.0, "TB": 1024.0}


def _unquote(value: str) -> str:
    text = value.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        return text[1:-1]
    return text


def _data_path_from_file() -> str:
    env_file = Path(__file__).resolve().parent.parent / ".env.local"
    if not env_file.is_file():
        return ""
    for raw in env_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() == "DATA_PATH":
            return _unquote(value)
    return ""


def _ensure_data_path() -> None:
    if os.environ.get("DATA_PATH", "").strip():
        return
    found = _data_path_from_file()
    if not found:
        raise SystemExit("DATA_PATH is unset and .env.local does not name one.")
    os.environ["DATA_PATH"] = found


def _install_dir() -> Path:
    return Path(os.environ["DATA_PATH"]) / "foldseek"


def _cpu_url() -> str:
    if platform.machine().lower() not in {"x86_64", "amd64"}:
        raise SystemExit(f"No static Foldseek build is selected for {platform.machine()}.")
    flags = Path("/proc/cpuinfo").read_text(encoding="utf-8") if Path("/proc/cpuinfo").is_file() else ""
    return _AVX2 if "avx2" in flags else _SSE41


def _binary_inside(extracted: Path) -> Path:
    found = [path for path in extracted.rglob("foldseek") if path.is_file()]
    if len(found) != 1:
        names = ", ".join(str(path) for path in found) or "none"
        raise SystemExit(f"Expected one foldseek binary in the archive, found {names}.")
    return found[0]


def declared_gb(database: str) -> float | None:
    """The recorded download size in GB, or None when the record cannot be read.

    None means refuse, never "assume it is small": the whole reason the limit
    is in code is that ``afdb50.tar.gz 114.1 GB`` and ``pdb100.tar.gz 2.2 GB``
    look alike until the figure is read.
    """
    from rfproteina.disease.novelty import FOLDSEEK_DATABASES

    entry = FOLDSEEK_DATABASES.get(database)
    if entry is None:
        return None
    match = _SIZE.search(str(entry.get("download", "")))
    if match is None:
        return None
    return float(match.group(1)) * _SCALE[match.group(2).upper()]


def database_alias(database: str) -> str | None:
    """The short name the recorded command stores the set under.

    ``foldseek databases PDB pdb tmp`` -> ``pdb``. Taken from the command
    rather than invented so the directory on disk matches what the handoff
    tells an operator to type.
    """
    from rfproteina.disease.novelty import FOLDSEEK_DATABASES

    entry = FOLDSEEK_DATABASES.get(database)
    if entry is None:
        return None
    parts = str(entry.get("command", "")).split()
    return parts[3] if len(parts) >= 4 else None


def _list_databases() -> int:
    from rfproteina.disease.novelty import FOLDSEEK_DATABASES

    print(f"cap  {DOWNLOAD_CAP_GB:.0f} GB per download\n")
    for name, entry in FOLDSEEK_DATABASES.items():
        size = declared_gb(name)
        if size is None:
            verdict = "REFUSED (size unreadable)"
        elif size > DOWNLOAD_CAP_GB:
            verdict = f"REFUSED ({size:.1f} GB, over the cap)"
        else:
            verdict = f"allowed ({size:.1f} GB)"
        print(f"  {name:22s} {verdict}")
        print(f"  {'':22s} {entry['download']}  [{entry['licence']}]")
    return 0


def _have(destination: Path) -> int:
    """Report the binary wherever the screen would actually find it."""
    from rfproteina.disease.novelty import foldseek_command

    found = foldseek_command()
    if found is None:
        print(f"binary  {destination}")
        print("status  absent")
        return 1
    if str(destination) != found:
        print(f"binary  {found}")
        print(f"        (not at the install destination {destination}; resolved by the screen's own lookup)")
    else:
        print(f"binary  {found}")
    result = subprocess.run([found, "version"], capture_output=True, text=True, check=False)
    print(f"version {result.stdout.strip() or result.stderr.strip()}")

    databases = _install_dir() / "databases"
    built = sorted(path.name for path in databases.glob("*") if path.is_dir()) if databases.is_dir() else []
    print(f"databases {databases}")
    print(f"          {', '.join(built) if built else 'none'}")
    return 0 if result.returncode == 0 else 1


def _install_binary(destination: Path) -> None:
    url = _cpu_url()
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"fetching {url}")
    with tempfile.TemporaryDirectory(prefix="foldseek_") as scratch:
        archive = Path(scratch) / "foldseek.tar.gz"
        # The URL is one of the two constants above, never caller-supplied.
        urllib.request.urlretrieve(url, archive)
        with tarfile.open(archive, "r:gz") as packed:
            packed.extractall(scratch, filter="data")
        shutil.copy2(_binary_inside(Path(scratch)), destination)
    destination.chmod(destination.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    result = subprocess.run([str(destination), "version"], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise SystemExit(result.stderr.strip() or "foldseek version failed")
    print(result.stdout.strip())
    print(f"ready  {destination}")


def _install_database(binary: Path, database: str) -> None:
    from rfproteina.disease.novelty import FOLDSEEK_DATABASES

    if database not in FOLDSEEK_DATABASES:
        known = ", ".join(FOLDSEEK_DATABASES)
        raise SystemExit(f"{database} is not one of the recorded databases: {known}")
    size = declared_gb(database)
    if size is None:
        raise SystemExit(
            f"{database} records no readable download size, so it is refused rather than guessed at. "
            f"Record one in novelty.FOLDSEEK_DATABASES first."
        )
    if size > DOWNLOAD_CAP_GB:
        raise SystemExit(
            f"{database} is {size:.1f} GB, over this project's {DOWNLOAD_CAP_GB:.0f} GB download limit. "
            f"Nothing was downloaded. Smaller sets: "
            f"{', '.join(n for n in FOLDSEEK_DATABASES if (declared_gb(n) or 1e9) <= DOWNLOAD_CAP_GB)}."
        )
    alias = database_alias(database)
    if alias is None:
        raise SystemExit(f"{database} records no command to read its alias from.")

    root = _install_dir() / "databases" / alias
    root.mkdir(parents=True, exist_ok=True)
    target = root / alias
    scratch = root / "tmp"
    print(f"fetching {database} ({size:.1f} GB, {FOLDSEEK_DATABASES[database]['licence']}) into {target}")
    command = [str(binary), "databases", database, str(target), str(scratch)]
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise SystemExit(f"foldseek databases {database} failed with exit {result.returncode}")
    if scratch.is_dir():
        shutil.rmtree(scratch, ignore_errors=True)
    on_disk = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    print(f"ready  {target}  ({on_disk / 1024**3:.2f} GB on disk)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--have", action="store_true", help="report the installed binary and databases, then stop")
    parser.add_argument(
        "--list-databases",
        action="store_true",
        help="print the recorded databases with their sizes and whether the cap allows them",
    )
    parser.add_argument(
        "--database",
        metavar="NAME",
        help="also fetch this structure database, if its recorded size is within the cap",
    )
    args = parser.parse_args(argv)

    if args.list_databases:
        return _list_databases()

    _ensure_data_path()
    destination = _install_dir() / "foldseek"
    if args.have:
        return _have(destination)

    from rfproteina.disease.novelty import foldseek_command

    if destination.is_file():
        print(f"have   {destination}")
    else:
        _install_binary(destination)

    if args.database:
        resolved = foldseek_command() or str(destination)
        _install_database(Path(resolved), args.database)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
