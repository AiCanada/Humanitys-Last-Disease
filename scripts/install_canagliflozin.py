#!/usr/bin/env python3
"""Install the fixed canagliflozin SBML and verify its reference simulation in an isolated CPU environment."""

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
from urllib.request import urlopen

from filelock import FileLock

from rfproteina.disease import canagliflozin_runtime as runtime
from rfproteina.disease import canagliflozin_worker as worker
from rfproteina.disease.ligand_inputs import read_regular


@dataclass(frozen=True)
class Wheel:
    tag: str
    size: int
    sha256: str
    asset: str


WHEELS = {
    "linux": {
        "libroadrunner": Wheel(
            "manylinux_2_28_x86_64", 50_798_266, "bc8736c90bdfaa98d52311bdda6a133aaaa919085473e0b6d4ae59984dd236be",
            "41/21/1e556f8cdd74a2fcfa8c8287c3dd8141be260594d7ed38968af7f452bc2f",
        ),
        "numpy": Wheel(
            "manylinux_2_27_x86_64.manylinux_2_28_x86_64", 16_717_410,
            "b7e18c623bb5c95acb3b3328861272816ba199fb531921c5d6d0b675f1fde9e3",
            "65/af/aa78d1a88805456e212b65461354cd943197fb9acecc4c90fd12295123a3",
        ),
    },
    "win32": {
        "libroadrunner": Wheel(
            "win_amd64", 25_092_542, "5713ac61d37556853c433c3ef285108b5860539ed1773770be9f0ecf83823632",
            "a2/16/93b747fda29dd15171832a595f9145d1292528a865cc3156670f99d8c94f",
        ),
        "numpy": Wheel(
            "win_amd64", 12_567_828, "0a59a421a32580a009e8a1751345bf829631b990dc1794b80514ab722b435def",
            "3c/a1/accf6d4f0c80c5d9ba9735d6b1550e444180599f34dec69ca01360f717ad",
        ),
    },
}


class InstallError(ValueError):
    """The destination or a required asset failed verification."""


def _download(url: str, size: int) -> bytes:
    with urlopen(url, timeout=60) as response:
        return response.read(size + 1)


def _stage_wheels(temp: Path, wheelhouse: Path | None) -> Path:
    wheel_dir = temp / "wheels"
    wheel_dir.mkdir()
    requirements = []
    for name, wheel in WHEELS[sys.platform].items():
        version = worker.VERSIONS[name]
        filename = f"{name}-{version}-cp312-cp312-{wheel.tag}.whl"
        if wheelhouse is None:
            raw = _download(f"https://files.pythonhosted.org/packages/{wheel.asset}/{filename}", wheel.size)
        else:
            raw = read_regular(wheelhouse / filename, wheel.size, "canagliflozin_wheel", InstallError)
        if len(raw) != wheel.size or hashlib.sha256(raw).hexdigest() != wheel.sha256:
            raise InstallError(f"invalid size or SHA-256 for {filename}")
        (wheel_dir / filename).write_bytes(raw)
        requirements.append(f"{name}=={version} --hash=sha256:{wheel.sha256}\n")
    (temp / "requirements.txt").write_text("".join(requirements), encoding="ascii")
    return wheel_dir


def _uv() -> str:
    search_path = os.pathsep.join([str(Path(sys.executable).parent), os.environ.get("PATH", "")])
    executable = shutil.which("uv", path=search_path)
    if executable is None:
        raise InstallError("uv is required to create the isolated CPU environment")
    return executable


def _stage(temp: Path, model: bytes, wheelhouse: Path | None, executable: str) -> Path:
    bundle = temp / "bundle"
    (bundle / "model").mkdir(parents=True)
    (bundle / "model" / runtime.MODEL_FILENAME).write_bytes(model)
    wheels = _stage_wheels(temp, wheelhouse)
    uv = [executable, "--no-config", "--no-cache", "--offline"]
    python = runtime.interpreter(bundle)
    commands = (
        [
            *uv, "venv", "--no-project", "--no-python-downloads", "--relocatable",
            "--python", sys.executable, bundle / "venv",
        ],
        [
            *uv, "pip", "install", "--python", python, "--no-index", "--find-links", wheels, "--only-binary", ":all:",
            "--no-deps", "--require-hashes", "-r", temp / "requirements.txt",
        ],
        [*uv, "pip", "check", "--python", python],
    )
    for command in commands:
        subprocess.run(command, cwd=temp, check=True, timeout=180)
    runtime.invoke(bundle, model)
    receipt = {
        "version": 1, "model_revision": runtime.MODEL_REVISION, "model_sha256": runtime.MODEL_SHA256,
        "wheels": {name: wheel.sha256 for name, wheel in WHEELS[sys.platform].items()},
    }
    (bundle / "install-receipt.json").write_text(json.dumps(receipt, sort_keys=True) + "\n", encoding="ascii")
    return bundle


def _verified_existing(destination: Path) -> bool:
    try:
        info = destination.lstat()
    except FileNotFoundError:
        return False
    if not stat.S_ISDIR(info.st_mode):
        raise InstallError("the installation destination must be a directory")
    # Exercise the exact model and solver even when another installer published it.
    try:
        model = runtime.model_bytes(destination / "model" / runtime.MODEL_FILENAME)
        runtime.invoke(destination, model)
    except runtime.CanagliflozinRuntimeError as error:
        raise InstallError(f"existing destination failed verification ({error})") from error
    return True


def install(model: Path | None = None, wheelhouse: Path | None = None) -> Path:
    if (
        sys.platform not in WHEELS or platform.machine().lower() not in {"x86_64", "amd64"}
        or sys.implementation.name != "cpython" or sys.version_info[:2] != (3, 12)
    ):
        raise InstallError("the pinned CPU wheels require CPython 3.12 on Linux x86-64 or Windows x64")
    destination = runtime.install_directory()
    destination.parent.mkdir(exist_ok=True)
    if not stat.S_ISDIR(destination.parent.lstat().st_mode):
        raise InstallError("the installation parent must be a directory")
    if _verified_existing(destination):
        return destination
    executable = _uv()
    # Concurrent callers share one download and build. An interrupted process releases the OS lock.
    with FileLock(destination.parent / f".{destination.name}.install.lock"):
        if _verified_existing(destination):
            return destination
        model_data = (
            runtime.model_bytes(model) if model is not None
            else runtime.verify_model(_download(runtime.MODEL_URL, runtime.MODEL_SIZE))
        )
        with TemporaryDirectory(dir=destination.parent, prefix=".canagliflozin-") as directory:
            staged = _stage(Path(directory), model_data, wheelhouse, executable)
            try:
                staged.rename(destination)
            except OSError:
                if not _verified_existing(destination):
                    raise
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, help="local copy of the released flat SBML for offline installation")
    parser.add_argument("--wheelhouse", type=Path, help="local directory containing both pinned platform wheels")
    args = parser.parse_args()
    try:
        destination = install(args.model, args.wheelhouse)
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        parser.exit(1, f"Canagliflozin installation failed: {error}\n")
    print(f"Verified canagliflozin {runtime.MODEL_RELEASE} at {destination}")


if __name__ == "__main__":
    main()
