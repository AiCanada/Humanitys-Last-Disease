"""The offline reference installer extracts only its pinned native data subset."""

from __future__ import annotations

import hashlib
import io
import runpy
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

import pytest
import zstandard

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "install_molprobity_data.py"


@pytest.fixture
def installer():
    return runpy.run_path(str(SCRIPT))


def test_only_protein_validation_and_native_initialization_dependencies_are_selected(installer):
    select = installer["selected_path"]
    base = "site-packages/chem_data/"
    for name in (
        "mon_lib/list/mon_lib_list.cif", "rotarama_data/rota8000-arg.data", "geostd/a/data_ALA.cif",
        "chemical_components/t/data_THR.cif", "geostd/rna_dna/chain_link_rna2p.cif",
    ):
        assert select(base + name) == PurePosixPath(name)
    for name in (
        "../../outside", "chemical_components/x/data_XYZ.cif", "geostd/q/data_QQQ.cif", "segment_lib/a.dat",
        "geostd/rna_dna/unneeded.cif", "mon_lib/../retarget", r"mon_lib\evil",
    ):
        assert select(base + name) is None
    assert select("/" + base + "mon_lib/ener_lib.cif") is None


def test_archive_digest_precedes_extraction_and_foreign_entries_are_skipped(installer, tmp_path, monkeypatch):
    path = tmp_path / "pinned.conda"
    stream = io.BytesIO()
    entries = [*installer["REQUIRED"], "segment_lib/large-unused.data", "../../foreign"]
    raw = b"test reference"
    with tarfile.open(fileobj=stream, mode="w") as archive:
        for name in entries:
            member = tarfile.TarInfo("site-packages/chem_data/" + name)
            member.size = len(raw)
            archive.addfile(member, io.BytesIO(raw))
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(installer["MEMBER"], zstandard.ZstdCompressor().compress(stream.getvalue()))
    extract = installer["_extract"]
    monkeypatch.setitem(extract.__globals__, "ARCHIVE_BYTES", path.stat().st_size)
    output = tmp_path / "chem_data"
    with pytest.raises(installer["ReferenceInstallError"], match="SHA-256"):
        extract(path, output)
    assert not output.exists()
    monkeypatch.setitem(extract.__globals__, "ARCHIVE_SHA256", hashlib.sha256(path.read_bytes()).hexdigest())
    extract(path, output)
    assert {p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()} == set(installer["REQUIRED"])
    assert not (tmp_path / "foreign").exists()
