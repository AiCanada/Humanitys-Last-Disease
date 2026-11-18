#!/usr/bin/env python3
"""Install the MMseqs2 static binary under ``DATA_PATH/mmseqs``.

The human proteome FASTA is already in ``DATA_PATH/screening_tables``. This
script does not download it again. It fetches the official Linux static build
(MIT) and leaves the executable at ``DATA_PATH/mmseqs/mmseqs``, which is where
``rfproteina.disease.offtarget`` looks after ``MMSEQS`` and before ``PATH``.

    python scripts/install_mmseqs.py
    python scripts/install_mmseqs.py --have

AVX2 is used when the CPU reports it. Otherwise the SSE4.1 build is used.
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

_AVX2 = "https://mmseqs.com/latest/mmseqs-linux-avx2.tar.gz"
_SSE41 = "https://mmseqs.com/latest/mmseqs-linux-sse41.tar.gz"


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
    return Path(os.environ["DATA_PATH"]) / "mmseqs"


def _cpu_url() -> str:
    if platform.machine().lower() not in {"x86_64", "amd64"}:
        raise SystemExit(f"No static MMseqs2 build is selected for {platform.machine()}.")
    flags = Path("/proc/cpuinfo").read_text(encoding="utf-8") if Path("/proc/cpuinfo").is_file() else ""
    if "avx2" in flags:
        return _AVX2
    return _SSE41


def _binary_inside(extracted: Path) -> Path:
    found = [path for path in extracted.rglob("mmseqs") if path.is_file() and path.name == "mmseqs"]
    if len(found) != 1:
        names = ", ".join(str(path) for path in found) or "none"
        raise SystemExit(f"Expected one mmseqs binary in the archive, found {names}.")
    return found[0]


def _have(destination: Path) -> int:
    print(f"binary  {destination}")
    if not destination.is_file():
        print("status  absent")
        return 1
    result = subprocess.run([str(destination), "version"], capture_output=True, text=True, check=False)
    print(result.stdout.strip() or result.stderr.strip())
    return 0 if result.returncode == 0 else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--have", action="store_true", help="report the installed binary, then stop")
    args = parser.parse_args(argv)
    _ensure_data_path()
    destination = _install_dir() / "mmseqs"
    if args.have:
        return _have(destination)
    url = _cpu_url()
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"fetching {url}")
    with tempfile.TemporaryDirectory(prefix="mmseqs_") as scratch:
        archive = Path(scratch) / "mmseqs.tar.gz"
        urllib.request.urlretrieve(url, archive)  # noqa: S310 — the URL is one of the two constants above
        with tarfile.open(archive, "r:gz") as packed:
            packed.extractall(scratch, filter="data")
        binary = _binary_inside(Path(scratch))
        shutil.copy2(binary, destination)
    destination.chmod(destination.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    result = subprocess.run([str(destination), "version"], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise SystemExit(result.stderr.strip() or "mmseqs version failed")
    print(result.stdout.strip())
    print(f"ready  {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
