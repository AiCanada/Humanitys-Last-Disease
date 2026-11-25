#!/usr/bin/env python3
"""Install the verified CCTBX 2025.11 protein validation reference subset offline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
import tarfile
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

import zstandard

VERSION = "2025.11"
ARCHIVE_BYTES = 1_132_738_508
ARCHIVE_SHA256 = "b157567eeb89a7228c2aa9f9c6a4e6fa4a04250ff063d733950fa6b0d1b5e8f9"
MEMBER = "pkg-chem_data-2025.11-pyhe0d8492_0.tar.zst"
MARKER = ".rfd4-molprobity-data"
INSTALL_ID = "protein-reference-v1 " + ARCHIVE_SHA256
REQUIRED = (
    "mon_lib/list/mon_lib_list.cif", "mon_lib/ener_lib.cif", "rotarama_data/rota8000-arg.data",
    "chemical_components/a/data_ALA.cif", "chemical_components/t/data_THR.cif",
)
MAX_SELECTED_BYTES = 256 * 1024 * 1024
_CANONICAL = frozenset(
    "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL HOH WAT DOD".split()
)
_GEOSTD_INIT = frozenset({
    "chain_link_rna2p.cif", "chain_link_rna3p.cif", "mod_rna2p.cif", "mod_rna3p.cif",
    "mod_rna2p_pur.cif", "mod_rna3p_pur.cif", "mod_rna2p_pyr.cif", "mod_rna3p_pyr.cif",
})
_LOCATE = (
    "import json, sysconfig; from importlib.metadata import version; "
    "print(json.dumps([version('cctbx-base'), sysconfig.get_path('purelib')]))"
)
_BUILD = """import sys
import libtbx.load_env
libtbx.env.repository_paths.insert(0, libtbx.env.as_relocatable_path(sys.argv[1]))
from mmtbx.chemical_components import get_cif_dictionary
from mmtbx.command_line.rebuild_rotarama_cache import run
from mmtbx.rotamer.rotamer_eval import RotamerEval
run()
RotamerEval(data_version='8000')
assert all(get_cif_dictionary(name) is not None for name in sys.argv[2:])
"""


class ReferenceInstallError(ValueError):
    """A runtime or archive outside the repeatable local install contract."""


def selected_path(member: str) -> PurePosixPath | None:
    path = PurePosixPath(member)
    if path.is_absolute() or ".." in path.parts or "\\" in member:
        return None
    if path.parts[:2] != ("site-packages", "chem_data") or len(path.parts) < 4:
        return None
    group, *parts = path.parts[2:]
    if group in ("mon_lib", "rotarama_data"):
        return PurePosixPath(group, *parts)
    if group == "chemical_components" and len(parts) == 2:
        leaf = PurePosixPath(parts[-1])
        if leaf.suffix == ".cif" and leaf.stem.removeprefix("data_").upper() in _CANONICAL:
            return PurePosixPath(group, *parts)
    if group == "geostd":
        leaf = PurePosixPath(parts[-1])
        # CCTBX initializes these shared link definitions even for protein-only inputs.
        if len(parts) == 2 and parts[0] == "rna_dna" and leaf.name in _GEOSTD_INIT:
            return PurePosixPath(group, *parts)
        if parts[0] == "list" or (len(parts) == 1 and (leaf.suffix == ".cif" or leaf.name == "COPYRIGHT.txt")):
            return PurePosixPath(group, *parts)
        if leaf.suffix == ".cif" and leaf.stem.removeprefix("data_").upper() in _CANONICAL:
            return PurePosixPath(group, *parts)
    return None


def _extract(archive: Path, output: Path) -> None:
    with archive.open("rb") as file:
        before = os.fstat(file.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size != ARCHIVE_BYTES:
            raise ReferenceInstallError("wrong chemistry archive size")
        if hashlib.file_digest(file, "sha256").hexdigest() != ARCHIVE_SHA256:
            raise ReferenceInstallError("chemistry archive SHA-256 did not match the pinned release")
        file.seek(0)
        total = 0
        with (
            zipfile.ZipFile(file) as bundle,
            bundle.open(MEMBER) as compressed,
            zstandard.ZstdDecompressor(max_window_size=2**27).stream_reader(compressed) as decoded,
            tarfile.open(fileobj=decoded, mode="r|") as package,
        ):
            for item in package:
                relative = selected_path(item.name)
                if relative is None or item.isdir():
                    continue
                if not item.isreg() or item.size < 0 or item.size > MAX_SELECTED_BYTES - total:
                    raise ReferenceInstallError("unexpected or oversized protein reference in archive")
                target = output.joinpath(*relative.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with package.extractfile(item) as source, target.open("xb") as dest:
                    shutil.copyfileobj(source, dest, length=1024 * 1024)
                os.utime(target, (item.mtime, item.mtime))
                total += item.size
        after = os.fstat(file.fileno())
        if (before.st_mtime_ns, before.st_ctime_ns, before.st_size) != (
            after.st_mtime_ns, after.st_ctime_ns, after.st_size,
        ):
            raise ReferenceInstallError("chemistry archive changed during extraction")
    if not all((output / path).is_file() for path in REQUIRED):
        raise ReferenceInstallError("the expected native protein reference files were not found")


def install(archive: Path, python: Path) -> bool:
    python = python.absolute()  # Do not resolve a venv Python symlink to the base interpreter.
    process = subprocess.run([str(python), "-I", "-c", _LOCATE], capture_output=True, check=True, text=True, timeout=30)
    version, result = json.loads(process.stdout)
    if version != VERSION:
        raise ReferenceInstallError("runtime must contain cctbx-base 2025.11")
    site = Path(result)
    target = site / "chem_data"
    if (target / MARKER).is_file() and (target / MARKER).read_text(encoding="ascii").strip() == INSTALL_ID:
        if not all((target / filename).is_file() for filename in (*REQUIRED, "rotarama_data/rota8000-arg.pickle")):
            raise ReferenceInstallError("existing reference install is incomplete; it was not overwritten")
        return False
    if target.exists() or target.is_symlink():
        raise ReferenceInstallError("runtime already has a chemistry library; it was not overwritten")
    lock = site / ".rfd4-molprobity-install.lock"
    lock.mkdir()
    try:
        with tempfile.TemporaryDirectory(prefix=".rfd4-molprobity-data-", dir=site) as directory:
            root = Path(directory)
            staged = root / "chem_data"
            _extract(archive, staged)
            expected = sorted(_CANONICAL - {"WAT", "DOD"})
            subprocess.run([str(python), "-I", "-c", _BUILD, str(root), *expected], check=True, timeout=300, cwd=root)
            (staged / MARKER).write_text(INSTALL_ID + "\n", encoding="ascii")
            if target.exists() or target.is_symlink():
                raise ReferenceInstallError("runtime already has a chemistry library; it was not overwritten")
            staged.rename(target)
    finally:
        lock.rmdir()
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="pinned official chem_data .conda archive")
    parser.add_argument("--python", type=Path, required=True, help="trusted isolated CCTBX 2025.11 interpreter")
    args = parser.parse_args()
    try:
        changed = install(args.archive, args.python)
    except (OSError, ValueError, subprocess.SubprocessError, zipfile.BadZipFile, tarfile.TarError,
            zstandard.ZstdError) as error:
        parser.exit(1, f"MolProbity data installation failed ({type(error).__name__}): {error}\n")
    print("Installed verified protein validation references." if changed else "References already installed.")


if __name__ == "__main__":
    main()
