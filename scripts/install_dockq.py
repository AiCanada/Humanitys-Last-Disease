#!/usr/bin/env python3
"""Install DockQ v2 (MIT) into its own virtual environment under ``DATA_PATH``.

DockQ scores a modelled complex against a reference: one number per interface
from Fnat, iRMSD and LRMSD, with the chain mapping searched automatically.
``rfproteina.disease.interactions`` already computes Fnat natively and calls
this for the full score; the executable is what was missing.

**It is deliberately not installed into the project environment.** Every
released DockQ pins ``numpy<2.0`` and this project runs numpy 2.x, so a plain
``uv pip install DockQ`` resolves by downgrading numpy underneath torch,
bionemo and everything else. Isolating it is the same choice
``scripts/install_mhcflurry.py`` makes for mhcflurry, and the runner finds it
through ``DOCKQ_EXECUTABLE`` or this directory.

About 25 MB installed, including its own numpy 1.26 and Biopython. No weights
and no network at run time.

    python scripts/install_dockq.py
    python scripts/install_dockq.py --have
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rfproteina.disease.interactions import (
    DOCKQ_COMMAND_ENV,
    DOCKQ_REQUIREMENT,
    dockq_command,
    dockq_install_root,
    dockq_version,
)


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


def _venv_python(venv: Path) -> Path:
    posix = venv / "bin" / "python"
    if posix.is_file():
        return posix
    windows = venv / "Scripts" / "python.exe"
    if windows.is_file():
        return windows
    raise SystemExit(f"No Python interpreter in {venv}.")


def _venv_dockq(venv: Path) -> Path:
    for candidate in (venv / "bin" / "DockQ", venv / "Scripts" / "DockQ.exe"):
        if candidate.is_file():
            return candidate
    raise SystemExit(f"DockQ is not installed beside {_venv_python(venv)}.")


def _have() -> int:
    root = dockq_install_root()
    command = dockq_command()
    print(f"install root  {root}")
    print(f"executable    {command or 'absent'}")
    print(f"resolved by   {DOCKQ_COMMAND_ENV} / {root}/venv / PATH")
    if command is None:
        print("version       absent")
        return 1
    print(f"version       {dockq_version(command) or 'unknown'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--have", action="store_true", help="report the install, then stop")
    parser.add_argument("--force", action="store_true", help="reinstall even where DockQ is already present")
    args = parser.parse_args(argv)
    _ensure_data_path()
    if args.have:
        return _have()

    root = dockq_install_root()
    if root is None:  # pragma: no cover - _ensure_data_path raises first
        raise SystemExit("DATA_PATH is unset.")
    venv = root / "venv"
    root.mkdir(parents=True, exist_ok=True)
    print(f"install root  {root}")

    if not (venv / "pyvenv.cfg").is_file():
        print(f"creating virtual environment  {venv}")
        # sys.executable is this project's interpreter; the child environment is
        # independent of it and gets its own numpy, which is the whole point.
        subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
    python = _venv_python(venv)

    already = None
    try:
        already = _venv_dockq(venv)
    except SystemExit:
        already = None
    if already is not None and not args.force:
        print(f"have          {already}")
    else:
        print(f"installing {DOCKQ_REQUIREMENT} into {python}")
        subprocess.run([str(python), "-m", "pip", "install", "--upgrade", "pip"], check=True)
        subprocess.run([str(python), "-m", "pip", "install", "--upgrade", DOCKQ_REQUIREMENT], check=True)

    command = _venv_dockq(venv)
    version = dockq_version(str(command)) or "unknown"
    print(f"ready         {command}  (DockQ {version})")
    # The runner finds this without being told, but say so: a reader who sets
    # the variable to something else should know it wins.
    print(f"note          {DOCKQ_COMMAND_ENV} overrides this path if set")
    resolved = dockq_command()
    if resolved != str(command):
        print(f"warning       {DOCKQ_COMMAND_ENV} or PATH resolves to {resolved} instead")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
