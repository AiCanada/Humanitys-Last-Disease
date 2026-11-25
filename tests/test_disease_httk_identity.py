"""A candidate manifest cannot create or replace the pinned EPA substance identity reference."""

import hashlib
import zipfile

import pytest

from rfproteina.disease import httk_identity as identity
from scripts import install_httk_identity as installer

CAFFEINE = ("58-08-2", "DTXSID0020232")
ACETAMINOPHEN = ("103-90-2", "DTXSID2020006")
CAFFEINE_KEY = "RYYVLZVUVIJVGH-UHFFFAOYSA-N"
ACETAMINOPHEN_KEY = "RZVAJINKPMORJF-UHFFFAOYSA-N"


@pytest.fixture
def source(tmp_path, monkeypatch):
    archive = tmp_path / "epa.zip"
    text = (
        "DTXSID,CASRN,INCHIKEY,QSAR_READY_SMILES\n"
        f"{CAFFEINE[1]},{CAFFEINE[0]},{CAFFEINE_KEY},test\n"
        f"{ACETAMINOPHEN[1]},{ACETAMINOPHEN[0]},{ACETAMINOPHEN_KEY},test\n"
        "DTXSID9999999,50-78-2,,no-structure\n"
        "DTXSID1234567,50-78-2,UNKNOWN,missing\n"
        f"DTXSID0033333,50-78-2,{CAFFEINE_KEY},duplicate\n"
        f"DTXSID0033333,50-78-2,{ACETAMINOPHEN_KEY},duplicate\n"
    )
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr(installer._MEMBER, text.encode())
    ambiguous_or_missing = {("50-78-2", sid) for sid in ("DTXSID9999999", "DTXSID1234567", "DTXSID0033333")}
    pairs = {CAFFEINE, ACETAMINOPHEN} | ambiguous_or_missing
    reference = (
        f"{ACETAMINOPHEN[0]}\t{ACETAMINOPHEN[1]}\t{ACETAMINOPHEN_KEY}\n"
        f"{CAFFEINE[0]}\t{CAFFEINE[1]}\t{CAFFEINE_KEY}\n"
    ).encode()
    path = tmp_path / "cache" / "index.tsv"
    monkeypatch.setattr(identity, "index_path", lambda: path)
    monkeypatch.setattr(identity, "ARCHIVE_BYTES", archive.stat().st_size)
    monkeypatch.setattr(identity, "ARCHIVE_SHA256", hashlib.sha256(archive.read_bytes()).hexdigest())
    monkeypatch.setattr(identity, "INDEX_SHA256", hashlib.sha256(reference).hexdigest())
    monkeypatch.setattr(installer, "_reference_pairs", lambda: pairs)
    return archive, path


def test_setup_uses_exact_unambiguous_pairs_and_is_idempotent(source):
    archive, target = source
    assert installer.install(archive) == target
    assert identity.load_index() == {CAFFEINE: CAFFEINE_KEY, ACETAMINOPHEN: ACETAMINOPHEN_KEY}
    archive.unlink()
    assert installer.install(None) == target


@pytest.mark.parametrize("pin", ["ARCHIVE_BYTES", "ARCHIVE_SHA256", "INDEX_SHA256"])
def test_rejects_wrong_releases_before_publishing_index(source, monkeypatch, pin):
    archive, target = source
    monkeypatch.setattr(identity, pin, 1 if pin == "ARCHIVE_BYTES" else "f" * 64)
    with pytest.raises(ValueError):
        installer.install(archive)
    assert not target.exists() and list(target.parent.iterdir()) == []


def test_tampered_local_index_is_rejected(source):
    archive, target = source
    installer.install(archive)
    target.write_bytes(target.read_bytes().replace(CAFFEINE_KEY.encode(), ACETAMINOPHEN_KEY.encode()))
    with pytest.raises(identity.IdentityReferenceError):
        identity.load_index()
