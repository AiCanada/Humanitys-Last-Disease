#!/usr/bin/env python3
"""Install Commec and download its reference databases under ``DATA_PATH/commec``.

The package goes in its own virtual environment. The databases are what
``commec setup`` fetches from databases.commec.io: biorisk, best-match,
low-concern, and the control lists. IBBIS states that download is under 10 GB.
This script does not fetch NCBI ``nr`` or ``core_nt``.

    python scripts/install_commec.py
    python scripts/install_commec.py --have

``biosecurity.py`` looks for the executable at ``DATA_PATH/commec/venv`` and
for the databases at ``DATA_PATH/commec/databases``.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
from pathlib import Path

_ZSTD_URL = "https://github.com/facebook/zstd/releases/download/v1.5.7/zstd-1.5.7.tar.gz"


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


def _root() -> Path:
    return Path(os.environ["DATA_PATH"]) / "commec"


def _python313() -> str:
    """An interpreter new enough for the current Commec command line.

    ``commec`` passes ``deprecated=`` to ``argparse``, which Python 3.12
    rejects. The virtual environment is therefore built with 3.13.
    """
    found = shutil.which("python3.13")
    if found:
        return found
    probe = subprocess.run(["uv", "python", "find", "3.13"], capture_output=True, text=True, check=False)
    if probe.returncode == 0 and probe.stdout.strip():
        return probe.stdout.strip()
    subprocess.run(["uv", "python", "install", "3.13"], check=True)
    probe = subprocess.run(["uv", "python", "find", "3.13"], capture_output=True, text=True, check=True)
    return probe.stdout.strip()


def _venv_is_current(venv: Path) -> bool:
    python = venv / "bin" / "python"
    if not python.is_file():
        python = venv / "Scripts" / "python.exe"
    if not python.is_file():
        return False
    probe = subprocess.run([str(python), "-c", "import sys; raise SystemExit(0 if sys.version_info >= (3, 13) else 1)"], check=False)
    return probe.returncode == 0


def _venv_python(venv: Path) -> Path:
    windows = venv / "Scripts" / "python.exe"
    if windows.is_file():
        return windows
    posix = venv / "bin" / "python"
    if posix.is_file():
        return posix
    raise SystemExit(f"No Python interpreter in {venv}.")


def _commec(python: Path) -> Path:
    for name in ("commec", "commec.exe"):
        candidate = python.parent / name
        if candidate.is_file():
            return candidate
    raise SystemExit(f"commec is not installed beside {python}.")


def _ensure_zstd(root: Path) -> str:
    """A ``zstd`` executable. Commec's archives are ``.tar.zst`` and ``tar`` calls ``zstd``."""
    found = shutil.which("zstd")
    if found:
        return found
    destination = root / "bin" / "zstd"
    if destination.is_file() and os.access(destination, os.X_OK):
        return str(destination)
    print(f"building zstd from {_ZSTD_URL}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="zstd_") as scratch:
        archive = Path(scratch) / "zstd.tar.gz"
        urllib.request.urlretrieve(_ZSTD_URL, archive)  # noqa: S310 — the URL is the constant above
        with tarfile.open(archive, "r:gz") as packed:
            packed.extractall(scratch, filter="data")
        source = next(Path(scratch).glob("zstd-*"))
        subprocess.run(["make", "-C", str(source), "-j4"], check=True)
        built = source / "programs" / "zstd"
        if not built.is_file():
            built = source / "zstd"
        shutil.copy2(built, destination)
    destination.chmod(destination.stat().st_mode | 0o755)
    return str(destination)


def _have(root: Path) -> int:
    databases = root / "databases"
    binary = root / "venv" / "bin" / "commec"
    windows = root / "venv" / "Scripts" / "commec.exe"
    print(f"install root  {root}")
    print(f"executable    {binary if binary.is_file() else windows}")
    populated = databases.is_dir() and any(databases.iterdir())
    print(f"databases     {'present' if populated else 'absent'}  ({databases})")
    return 0 if populated and (binary.is_file() or windows.is_file()) else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--have", action="store_true", help="report the install and databases, then stop")
    args = parser.parse_args(argv)
    _ensure_data_path()
    root = _root()
    if args.have:
        return _have(root)
    venv = root / "venv"
    databases = root / "databases"
    databases.mkdir(parents=True, exist_ok=True)
    print(f"install root  {root}")
    if not _venv_is_current(venv):
        if venv.exists():
            print(f"replacing {venv} with a Python 3.13 environment")
            shutil.rmtree(venv)
        interpreter = _python313()
        print(f"creating virtual environment  {venv}  ({interpreter})")
        subprocess.run([interpreter, "-m", "venv", str(venv)], check=True)
    python = _venv_python(venv)
    # PyPI has no commec distribution, and the repository does not declare its
    # Python dependencies. They are listed in its conda environment file.
    source = "git+https://github.com/ibbis-bio/common-mechanism.git"
    print(f"installing {source} into {python}")
    subprocess.run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--upgrade",
            "pyyaml",
            "biopython",
            "numpy",
            "pandas",
            "pycountry",
            "flask",
            "openpyxl",
            "xlrd",
            "werkzeug",
            "psutil",
            source,
        ],
        check=True,
    )
    command = _commec(python)
    zstd = _ensure_zstd(root)
    environment = os.environ.copy()
    environment["PATH"] = str(Path(zstd).parent) + os.pathsep + environment.get("PATH", "")
    print(f"fetching reference databases into {databases}")
    subprocess.run([str(command), "setup", "-d", str(databases)], check=True, env=environment)
    if not any(databases.iterdir()):
        raise SystemExit(f"commec setup finished but {databases} is empty.")
    print(f"ready  {command}")
    print(f"ready  {databases}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
