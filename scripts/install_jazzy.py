#!/usr/bin/env python3
"""Install the verified Jazzy wheel stack in an isolated Python environment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import stat
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import HTTPRedirectHandler, Request, build_opener

from rfproteina.disease import jazzy_runtime, jazzy_worker
from rfproteina.disease.ligand_inputs import read_regular


@dataclass(frozen=True)
class Wheel:
    version: str
    tag: str
    asset: str
    size: int


_MANYLINUX_2014 = "cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64"
_WHEELS = {
    "click": Wheel(
        "8.5.0", "py3-none-any", "58/50/6c0d534c5f134586a8e1ba4e330569e32f057e33372ae556463212fb4cd3", 125_251
    ),
    "jazzy": Wheel(
        "0.1.4", "py3-none-any", "01/8a/055258eacbbdd3bc3e24443283796e5860c2e7ca776e34e522d703d43b53", 24_272
    ),
    "kallisto": Wheel(
        "1.0.10", "py3-none-any", "7a/90/e90f62d1d8d315c50119d183b663c87d8ea14b6084c143fdf5a5ec7f82d5", 104_667
    ),
    "numpy": Wheel(
        "1.26.4", _MANYLINUX_2014, "0f/50/de23fde84e45f5c4fda2488c759b69990fd4512387a8632860f3ac9cd225", 17_950_613
    ),
    "pillow": Wheel(
        "13.0.0",
        "cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64",
        "e6/99/fdf65cc539e266ab93546c261bdf98411fe165c6e4e13f9dbf597097192b",
        7_144_542,
    ),
    "rdkit": Wheel(
        "2024.3.1", _MANYLINUX_2014, "28/22/d7267282b251b4f2866f8bc67af27303f3660e48f21518a93fd67cff80ff", 35_008_885
    ),
    "scipy": Wheel(
        "1.13.1", _MANYLINUX_2014, "88/ab/6ecdc526d509d33814835447bbbeedbebdec7cca46ef495a61b00a35b4bf", 38_218_311
    ),
}

_SMOKE = """
import importlib.metadata
import json
import math
import sys
from numbers import Real

from jazzy.api import molecular_vector_from_smiles

expected = json.loads(sys.argv[1])
if any(importlib.metadata.version(package) != version for package, version in expected.items()):
    raise SystemExit("the isolated Jazzy package versions did not match")
result = molecular_vector_from_smiles(
    "CC(=O)NC1=CC=C(C=C1)O", embedding_type="3D", embedding_seed=11,
    embedding_max_iterations=64, minimisation_method="MMFF94",
)
if set(result) != {"sdc", "sdx", "sa", "dga", "dgp", "dgtot"} or not all(
    isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value) for value in result.values()
):
    raise SystemExit("the Jazzy paracetamol check did not return finite descriptors")
"""


class InstallError(ValueError):
    """The selected wheel or destination could not be verified."""


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req: Request, fp: object, code: int, msg: str, headers: object, newurl: str) -> None:
        return None


def _stage_wheels(directory: Path, source: Path | None) -> Path:
    wheels = directory / "wheels"
    wheels.mkdir()
    requirements = []
    opener = build_opener(_NoRedirect())
    for name, wheel in _WHEELS.items():
        filename = f"{name}-{wheel.version}-{wheel.tag}.whl"
        checksum = jazzy_runtime.WHEEL_SHA256[name]
        if source is None:
            url = f"https://files.pythonhosted.org/packages/{wheel.asset}/{filename}"
            with opener.open(url, timeout=30) as response:
                raw = response.read(wheel.size + 1)
        else:
            raw = read_regular(source / filename, wheel.size, "jazzy_wheel", InstallError)
        if len(raw) != wheel.size or hashlib.sha256(raw).hexdigest() != checksum:
            raise InstallError(f"wheel size or SHA-256 did not match {filename}")
        (wheels / filename).write_bytes(raw)
        requirements.append(f"{name}=={wheel.version} --hash=sha256:{checksum}\n")
    (directory / "requirements.txt").write_text("".join(requirements), encoding="ascii")
    return wheels


def _uv() -> Path:
    bundled = Path(sys.executable).parent / "uv"
    if bundled.is_file() and os.access(bundled, os.X_OK):
        return bundled
    system = shutil.which("uv")
    if system is None:
        raise InstallError("uv is required to provision the isolated Jazzy environment")
    return Path(system).absolute()


def _run(command: list[str | Path], directory: Path) -> None:
    env = {
        "HOME": str(directory),
        "LANG": "C.UTF-8",
        "PATH": os.defpath,
        "TMPDIR": str(directory),
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
    }
    subprocess.run(command, cwd=directory, env=env, check=True, timeout=180)


def _build(directory: Path, wheelhouse: Path | None) -> Path:
    wheel_dir = _stage_wheels(directory, wheelhouse)
    prefix = directory / jazzy_runtime.RELEASE
    uv = [_uv(), "--no-config", "--no-cache", "--offline"]
    _run(
        [*uv, "venv", "--no-project", "--no-python-downloads", "--relocatable", "--python", sys.executable, prefix],
        directory,
    )
    python = prefix / "bin" / "python"
    _run(
        [
            *uv,
            "pip",
            "install",
            "--python",
            python,
            "--no-index",
            "--find-links",
            wheel_dir,
            "--only-binary",
            ":all:",
            "--require-hashes",
            "--no-deps",
            "-r",
            directory / "requirements.txt",
        ],
        directory,
    )
    _run([*uv, "pip", "check", "--python", python], directory)
    _run([python, "-IB", Path(jazzy_worker.__file__).resolve(), "--check-installation"], directory)
    _run([python, "-IB", "-c", _SMOKE, json.dumps(jazzy_runtime.VERSIONS)], directory)
    receipt = {"version": 1, "wheels": dict(jazzy_runtime.WHEEL_SHA256)}
    with (prefix / "install-receipt.json").open("x", encoding="ascii") as output:
        json.dump(receipt, output, sort_keys=True)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    return prefix


def _existing(destination: Path) -> bool:
    try:
        destination.lstat()
    except FileNotFoundError:
        return False
    if not jazzy_runtime.installed():
        raise InstallError("an unverified Jazzy destination already exists")
    return True


def install(wheelhouse: Path | None = None) -> Path:
    """Install the CPU wheel stack without modifying the active environment's packages."""
    if not (
        sys.platform == "linux"
        and platform.machine() == "x86_64"
        and sys.implementation.name == "cpython"
        and sys.version_info[:2] == (3, 12)
    ):
        raise InstallError("this Jazzy wheel stack requires Linux x86-64 with CPython 3.12")
    destination = jazzy_runtime.release_directory()
    for directory in (destination.parent.parent, destination.parent):
        directory.mkdir(exist_ok=True)
        if not stat.S_ISDIR(directory.lstat().st_mode):
            raise InstallError("the installation parent must be a regular directory")
    if _existing(destination):
        return destination
    lock = destination.parent / f".install-{jazzy_runtime.RELEASE}.lock"
    try:
        lock.mkdir()
    except FileExistsError as exc:
        raise InstallError("the Jazzy installation is locked") from exc
    try:
        if not _existing(destination):
            with TemporaryDirectory(prefix=".jazzy-", dir=destination.parent) as name:
                staged = _build(Path(name), wheelhouse)
                if not _existing(destination):
                    staged.rename(destination)
    finally:
        lock.rmdir()
    return destination


def main() -> None:
    """Install Jazzy from exact official wheels or an existing wheelhouse."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheelhouse", type=Path, help="directory containing the seven pinned wheel files")
    args = parser.parse_args()
    try:
        destination = install(args.wheelhouse)
    except (InstallError, OSError, subprocess.SubprocessError) as exc:
        parser.exit(1, f"Jazzy installation failed: {exc}\n")
    print(f"Installed Jazzy {jazzy_runtime.RELEASE} at {destination}")


if __name__ == "__main__":
    main()
