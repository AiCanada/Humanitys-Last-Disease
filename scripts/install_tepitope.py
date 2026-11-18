#!/usr/bin/env python3
"""Download the Unlicense TEPITOPE matrices into ``DATA_PATH/tepitope``.

The scorer is ``rfproteina.disease.tepitope``. ``scripts/tepitope.py`` is the
upstream command-line program on Python 3. This fetches the matrix files.

    python scripts/install_tepitope.py
    python scripts/install_tepitope.py --have
"""

from __future__ import annotations

import argparse
import io
import os
import tarfile
import urllib.request
from pathlib import Path

_ARCHIVE = "https://github.com/BenPorebski/tepitope/archive/refs/heads/master.tar.gz"


def _unquote(value: str) -> str:
    text = value.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        return text[1:-1]
    return text


def _ensure_data_path() -> None:
    if os.environ.get("DATA_PATH", "").strip():
        return
    env_file = Path(__file__).resolve().parent.parent / ".env.local"
    if not env_file.is_file():
        raise SystemExit("DATA_PATH is unset and .env.local does not name one.")
    for raw in env_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() == "DATA_PATH":
            os.environ["DATA_PATH"] = _unquote(value)
            return
    raise SystemExit("DATA_PATH is unset and .env.local does not name one.")


def _destination() -> Path:
    return Path(os.environ["DATA_PATH"]) / "tepitope" / "matrices"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--have", action="store_true", help="report the matrices, then stop")
    args = parser.parse_args(argv)
    _ensure_data_path()
    destination = _destination()
    if args.have:
        count = len(list(destination.glob("*.csv"))) if destination.is_dir() else 0
        print(f"matrices  {destination}")
        print(f"count     {count}")
        return 0 if count else 1
    print(f"fetching {_ARCHIVE}")
    # The URL is _ARCHIVE, the constant above, never caller-supplied.
    with urllib.request.urlopen(_ARCHIVE, timeout=120) as response:
        payload = response.read()
    destination.mkdir(parents=True, exist_ok=True)
    written = 0
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as packed:
        for member in packed.getmembers():
            name = Path(member.name)
            if member.isfile() and name.parent.name == "matrices" and name.suffix == ".csv":
                extracted = packed.extractfile(member)
                if extracted is None:
                    continue
                (destination / name.name).write_bytes(extracted.read())
                written += 1
    if written == 0:
        raise SystemExit("The archive contained no matrix files.")
    print(f"ready  {destination}  ({written} matrices)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
