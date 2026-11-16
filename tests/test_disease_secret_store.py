"""The encrypted blob store that holds a credential which is not an API key.

``key_store`` already encrypts the provider API keys. A session credential for a
web service is the same kind of secret with none of the same shape: it is not
one of the five providers, it has no model tied to it, and it expires. It was
going to end up in a plain file in the run directory, which is readable by
anything on the machine and one ``git add`` from a public repository.

Three properties here are load-bearing and none of them is visible in review:

1. **Writing one secret may not lose the other.** ``save_key_slots`` replaces
   the whole decrypted body of ``api_keys.json``. A blob added as another key
   inside that body would be destroyed the next time somebody typed an API key,
   and the only symptom would be a sign-in that mysteriously stopped lasting.
   So the blob gets its own file, and this pins that the two writers do not
   interfere in either order.

2. **A damaged blob is not an empty one.** ``load_secret`` returning ``{}`` for
   a file that will not decrypt would read as "no credential stored", the caller
   would sign in again, and the overwrite would take the damaged file's contents
   with it. Absent returns empty; unreadable raises.

3. **The plaintext is not on disk.** The whole reason this module exists.
"""

from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path

import pytest

from rfproteina.disease.key_store import (
    KeySlot,
    forget_secret,
    key_directory,
    load_api_keys,
    load_key_slots,
    load_secret,
    save_api_keys,
    save_key_slots,
    save_secret,
)
from rfproteina.disease.publish import RepositoryKeyError

#: A value that is obviously not a real credential but is long and distinctive
#: enough that finding it in a file proves the file holds plaintext.
FAKE = "af3-session-not-a-real-credential-0123456789"


@pytest.fixture
def store(monkeypatch, tmp_path: Path) -> Path:
    """An empty key directory outside any checkout."""
    directory = tmp_path / "keys"
    monkeypatch.setenv("RFD4_KEY_DIR", str(directory))
    return directory


def test_nothing_stored_reads_as_empty(store: Path) -> None:
    """A caller with no credential is told so, not handed an exception."""
    assert load_secret("af3_session") == {}
    assert not store.exists()


def test_a_saved_blob_comes_back_unchanged(store: Path) -> None:
    """Every JSON type the caller may put in a credential record survives."""
    body = {
        "cookie": FAKE,
        "expires": 1763000000,
        "verified": True,
        "scopes": ["submit", "retrieve"],
        "account": None,
    }
    save_secret("af3_session", body)
    assert load_secret("af3_session") == body


def test_the_file_on_disk_does_not_contain_the_credential(store: Path) -> None:
    """The point of the module: ciphertext, not a readable cookie."""
    save_secret("af3_session", {"cookie": FAKE})
    written = [path for path in store.iterdir() if path.is_file()]
    assert written, "nothing was written"
    for path in written:
        assert FAKE not in path.read_bytes().decode("utf-8", "replace"), f"{path.name} holds the plaintext"


def test_an_unknown_name_is_refused_by_name(store: Path) -> None:
    """A typo must not be written happily to a file nothing reads.

    Without this, ``save_secret("af3_sesion", ...)`` succeeds and the matching
    ``load_secret("af3_session")`` reports no credential -- indistinguishable
    from a save that never happened.
    """
    with pytest.raises(ValueError, match="Known: af3_session"):
        save_secret("af3_sesion", {"cookie": FAKE})
    with pytest.raises(ValueError, match="Known: af3_session"):
        load_secret("af3_sesion")
    with pytest.raises(ValueError, match="Known: af3_session"):
        forget_secret("af3_sesion")


def test_an_empty_body_removes_the_file(store: Path) -> None:
    """Forgetting a credential and never having had one leave the same state."""
    save_secret("af3_session", {"cookie": FAKE})
    assert load_secret("af3_session") != {}
    save_secret("af3_session", {})
    assert load_secret("af3_session") == {}
    assert not (store / "af3_session.enc.json").exists()


def test_forget_reports_whether_there_was_anything_to_forget(store: Path) -> None:
    """A sign-out that found no session says so rather than claiming success."""
    assert forget_secret("af3_session") is False
    save_secret("af3_session", {"cookie": FAKE})
    assert forget_secret("af3_session") is True
    assert forget_secret("af3_session") is False


def test_a_damaged_blob_raises_instead_of_reading_as_empty(store: Path) -> None:
    """Unreadable is not absent. Treating it as absent overwrites the evidence."""
    save_secret("af3_session", {"cookie": FAKE})
    path = store / "af3_session.enc.json"
    path.write_text(json.dumps({"version": 1, "token": "not-a-fernet-token"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_secret("af3_session")


def test_a_blob_whose_master_key_is_gone_raises(store: Path) -> None:
    """The ciphertext alone is not a credential, and cannot be read as one."""
    save_secret("af3_session", {"cookie": FAKE})
    (store / "master.key").unlink()
    with pytest.raises(ValueError):
        load_secret("af3_session")


@pytest.mark.parametrize("missing", ["version", "token"])
def test_a_blob_with_a_broken_envelope_raises(store: Path, missing: str) -> None:
    """A file that is JSON but not this module's envelope is not silently empty."""
    save_secret("af3_session", {"cookie": FAKE})
    path = store / "af3_session.enc.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.pop(missing)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_secret("af3_session")


def test_saving_api_keys_does_not_lose_the_credential(store: Path) -> None:
    """The defect this module's separate file exists to prevent.

    ``save_key_slots`` replaces the entire decrypted body of ``api_keys.json``.
    A credential kept as another key inside that body is gone the next time
    anybody types an API key on the landing page.
    """
    save_secret("af3_session", {"cookie": FAKE})
    save_api_keys({"grok": "xai-fake-key-for-this-test"})
    save_key_slots({"claude": (KeySlot("sk-fake-key-for-this-test", "claude-opus-5"),)})
    assert load_secret("af3_session") == {"cookie": FAKE}


def test_saving_the_credential_does_not_lose_the_api_keys(store: Path) -> None:
    """And the same in the other direction, which is just as easy to break."""
    save_key_slots({"claude": (KeySlot("sk-fake-key-for-this-test", "claude-opus-5"),)})
    save_secret("af3_session", {"cookie": FAKE})
    assert load_api_keys() == {"claude": "sk-fake-key-for-this-test"}
    assert load_key_slots()["claude"][0].model == "claude-opus-5"


def test_both_secrets_share_one_master_key(store: Path) -> None:
    """One master key, created by whichever writer runs first, used by both."""
    save_secret("af3_session", {"cookie": FAKE})
    created = (store / "master.key").read_bytes()
    save_api_keys({"grok": "xai-fake-key-for-this-test"})
    assert (store / "master.key").read_bytes() == created
    assert load_secret("af3_session") == {"cookie": FAKE}
    assert load_api_keys() == {"grok": "xai-fake-key-for-this-test"}


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX file modes")
def test_the_blob_is_private_to_this_user(store: Path) -> None:
    """0600, like the API keys. A world-readable credential is not stored safely."""
    save_secret("af3_session", {"cookie": FAKE})
    mode = stat.S_IMODE(os.stat(store / "af3_session.enc.json").st_mode)
    assert mode == 0o600, f"mode is {mode:o}"


def test_the_credential_is_refused_inside_a_repository(monkeypatch, tmp_path: Path) -> None:
    """Same guard the API keys have: a store in a checkout gets committed."""
    checkout = tmp_path / "checkout"
    (checkout / ".git").mkdir(parents=True)
    (checkout / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    monkeypatch.setenv("RFD4_KEY_DIR", str(checkout / "secrets"))
    assert key_directory() == checkout / "secrets"
    with pytest.raises(RepositoryKeyError):
        save_secret("af3_session", {"cookie": FAKE})
