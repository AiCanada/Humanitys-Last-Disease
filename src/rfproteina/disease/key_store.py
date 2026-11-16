"""Encrypted storage for API keys typed on the landing page.

The ciphertext and the encryption secret are separate files, mode 0600, under
``~/.config/Humanitys-Last-Disease``. The secret is mixed with this machine's id before use,
so the ciphertext file does not contain a usable key on its own. Neither file
is written into a command, a job, or a research report.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
from dataclasses import dataclass
from pathlib import Path
from secrets import token_bytes

from cryptography.fernet import Fernet, InvalidToken

from rfproteina.disease.publish import refuse_repository_key_store

_PROVIDERS = ("grok", "claude", "openai", "nvidia", "openrouter")
_UNREADABLE = "Saved API keys could not be read. Enter them again on the first page."
_UNWRITABLE = "Saved API keys could not be written."

#: Blobs a caller may store beside the API keys, each in its own encrypted file.
#: Closed for the same reason :data:`_PROVIDERS` is: a misspelled name would
#: write a second file that nothing ever reads, and the caller would be told the
#: secret was saved.
_SECRET_FILES = ("af3_session",)
_SECRET_UNREADABLE = "A saved credential could not be read. Sign in again."
_SECRET_UNWRITABLE = "A credential could not be written."


@dataclass(frozen=True)
class KeySlot:
    """One saved key and the model it is sent with.

    ``model`` is empty when the key was saved before keys were tied to models.
    That key is still used for every model of its provider.
    """

    secret: str
    model: str


def key_directory() -> Path:
    """Directory for the encrypted store. ``RFD4_KEY_DIR`` selects another one."""
    override = os.environ.get("RFD4_KEY_DIR", "").strip()
    if override:
        return Path(override)
    return Path.home() / ".config" / "Humanitys-Last-Disease"


def load_api_keys() -> dict[str, str]:
    """Return one saved key per provider. A missing store is empty.

    A provider with several keys contributes its first key. A bad store raises
    ``ValueError``.
    """
    body = _read_body()
    if body is None:
        return {}
    if body.get("version") == 2:
        return {name: rows[0].secret for name, rows in _slots_from_body(body).items() if rows}
    return _clean(body)


def load_key_slots() -> dict[str, tuple[KeySlot, ...]]:
    """Return every saved key, in the order it was entered, with its model.

    A store written before keys were tied to models yields one slot per provider
    and an empty model. A missing store is empty. A bad store raises ``ValueError``.
    """
    body = _read_body()
    if body is None:
        return {}
    if body.get("version") == 2:
        return _slots_from_body(body)
    return {name: (KeySlot(secret, ""),) for name, secret in _clean(body).items()}


def save_key_slots(slots: dict[str, tuple[KeySlot, ...] | list[KeySlot]]) -> None:
    """Encrypt every key and the model tied to it, replacing the store."""
    payload: dict[str, list[dict[str, str]]] = {}
    for name in _PROVIDERS:
        rows = slots.get(name, ())
        if not isinstance(rows, list | tuple):
            continue
        kept: list[dict[str, str]] = []
        for row in rows:
            if not isinstance(row, KeySlot):
                continue
            secret = row.secret.strip()
            if secret:
                kept.append({"secret": secret, "model": row.model.strip()})
        if kept:
            payload[name] = kept
    refuse_repository_key_store(key_directory())
    try:
        _write_store({"version": 2, "slots": payload})
    except (OSError, UnicodeError, ValueError):
        raise ValueError(_UNWRITABLE) from None


def load_model_choices() -> dict[str, str]:
    """Return saved model ids. A missing or unreadable file is empty.

    Model ids are not secrets. They are stored beside the encrypted keys, not
    inside the ciphertext.
    """
    path = key_directory() / "models.json"
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    body = payload.get("models") if isinstance(payload, dict) else None
    if not isinstance(body, dict):
        return {}
    return _clean(body)


def save_model_choices(models: dict[str, str]) -> bool:
    """Store model ids beside the encrypted keys. Returns whether the file changed."""
    incoming = _clean(models)
    if not incoming:
        return False
    current = load_model_choices()
    merged = dict(current)
    merged.update(incoming)
    if merged == current:
        return False
    refuse_repository_key_store(key_directory())
    document = json.dumps({"version": 1, "models": merged}, indent=2) + "\n"
    try:
        _write_private(key_directory() / "models.json", document.encode("utf-8"))
    except OSError:
        raise ValueError("Saved model choices could not be written.") from None
    return True


def save_api_keys(keys: dict[str, str]) -> None:
    """Encrypt ``keys`` and replace the store. Empty values are not written."""
    cleaned = _clean(keys)
    if not cleaned:
        return
    refuse_repository_key_store(key_directory())
    try:
        _write_store(cleaned)
    except (OSError, UnicodeError, ValueError):
        raise ValueError(_UNWRITABLE) from None


def load_secret(name: str) -> dict[str, object]:
    """The decrypted blob stored under ``name``, or empty when none is stored.

    A credential that is not one provider's API key still has to live somewhere,
    and a plain file in the run directory is the wrong somewhere: it survives
    the run, it is readable by anything on the machine, and it is one ``git add``
    away from a public repository. This gives such a credential the same
    treatment the API keys already get -- Fernet over a 0600 secret mixed with
    the machine id, in a directory outside the checkout.

    Its own file rather than another key inside ``api_keys.json``, so that
    writing one cannot lose the other. The two writers would otherwise have to
    agree about the whole body, and :func:`save_key_slots` replaces it.

    An unreadable blob raises ``ValueError``; an *absent* one does not. The
    difference matters to the caller: nothing stored means "sign in", while a
    blob that will not decrypt means the store is damaged and signing in again
    would overwrite whatever else is in it.
    """
    path = _secret_path(name)
    if not path.is_file():
        return {}
    directory = key_directory()
    secret = _read_master(directory)
    if secret is None:
        raise ValueError(_SECRET_UNREADABLE)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        token = payload.get("token") if isinstance(payload, dict) and payload.get("version") == 1 else None
        if not isinstance(token, str) or not token:
            raise ValueError(_SECRET_UNREADABLE)
        body = json.loads(_cipher(secret).decrypt(token.encode("ascii")).decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, InvalidToken, ValueError):
        raise ValueError(_SECRET_UNREADABLE) from None
    if not isinstance(body, dict):
        raise ValueError(_SECRET_UNREADABLE)
    return body


def save_secret(name: str, body: dict[str, object]) -> None:
    """Encrypt ``body`` and replace the blob stored under ``name``.

    An empty body removes the file rather than writing an empty one, so
    forgetting a credential and never having had one leave the same state. A
    half-written store is not possible: :func:`_write_private` renames a
    complete temporary file over the target.
    """
    path = _secret_path(name)
    directory = key_directory()
    refuse_repository_key_store(directory)
    if not body:
        path.unlink(missing_ok=True)
        return
    try:
        secret = _read_master(directory)
        if secret is None:
            secret = token_bytes(32)
            _write_private(directory / "master.key", secret)
        token = _cipher(secret).encrypt(json.dumps(body, separators=(",", ":"), sort_keys=True).encode("utf-8"))
        _write_private(path, (json.dumps({"version": 1, "token": token.decode("ascii")}) + "\n").encode("ascii"))
    except (OSError, UnicodeError, ValueError, TypeError):
        raise ValueError(_SECRET_UNWRITABLE) from None


def forget_secret(name: str) -> bool:
    """Delete the blob stored under ``name``. Returns whether a file was removed."""
    path = _secret_path(name)
    existed = path.is_file()
    try:
        path.unlink(missing_ok=True)
    except OSError:
        raise ValueError(_SECRET_UNWRITABLE) from None
    return existed


def _secret_path(name: str) -> Path:
    """Where one named blob lives. An unknown name is refused, not invented.

    A typo would otherwise be written happily to its own file and read back as
    "nothing stored" by the code that meant to use it, which looks exactly like
    a credential that never saved.
    """
    if name not in _SECRET_FILES:
        raise ValueError(f"Unknown credential name {name!r}. Known: {', '.join(_SECRET_FILES)}.")
    return key_directory() / f"{name}.enc.json"


def _read_body() -> dict[str, object] | None:
    """The decrypted store, or None when the file is absent. A bad file raises."""
    directory = key_directory()
    store = directory / "api_keys.json"
    if not store.is_file():
        return None
    secret = _read_master(directory)
    if secret is None:
        raise ValueError(_UNREADABLE)
    try:
        payload = json.loads(store.read_text(encoding="utf-8"))
        token = payload.get("token") if isinstance(payload, dict) and payload.get("version") == 1 else None
        if not isinstance(token, str) or not token:
            raise ValueError(_UNREADABLE)
        raw = _cipher(secret).decrypt(token.encode("ascii"))
        body = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, InvalidToken, ValueError):
        raise ValueError(_UNREADABLE) from None
    if not isinstance(body, dict):
        raise ValueError(_UNREADABLE)
    return body


def _slots_from_body(body: dict[str, object]) -> dict[str, tuple[KeySlot, ...]]:
    raw = body.get("slots")
    if not isinstance(raw, dict):
        raise ValueError(_UNREADABLE)
    found: dict[str, tuple[KeySlot, ...]] = {}
    for name in _PROVIDERS:
        rows = raw.get(name, [])
        if rows in (None, []):
            continue
        if not isinstance(rows, list):
            raise ValueError(_UNREADABLE)
        parsed: list[KeySlot] = []
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError(_UNREADABLE)
            secret = row.get("secret", "")
            model = row.get("model", "")
            if not isinstance(secret, str) or not isinstance(model, str):
                raise ValueError(_UNREADABLE)
            if secret.strip():
                parsed.append(KeySlot(secret.strip(), model.strip()))
        if parsed:
            found[name] = tuple(parsed)
    return found


def _write_store(body: dict[str, object]) -> None:
    directory = key_directory()
    refuse_repository_key_store(directory)
    secret = _read_master(directory)
    if secret is None:
        secret = token_bytes(32)
        _write_private(directory / "master.key", secret)
    token = _cipher(secret).encrypt(json.dumps(body, separators=(",", ":")).encode("utf-8"))
    document = json.dumps({"version": 1, "token": token.decode("ascii")}) + "\n"
    _write_private(directory / "api_keys.json", document.encode("ascii"))


def _clean(keys: dict[str, object]) -> dict[str, str]:
    cleaned: dict[str, str] = {}
    for name in _PROVIDERS:
        value = keys.get(name, "")
        if isinstance(value, str) and value.strip():
            cleaned[name] = value.strip()
    return cleaned


def _read_master(directory: Path) -> bytes | None:
    path = directory / "master.key"
    if not path.is_file():
        return None
    secret = path.read_bytes()
    if len(secret) < 16:
        return None
    return secret


def _cipher(secret: bytes) -> Fernet:
    mixed = hmac.new(_machine_tag(), secret, hashlib.sha256).digest()
    return Fernet(base64.urlsafe_b64encode(mixed))


#: Last-resort machine tag, used only where there is no machine id and no
#: ``os.uname`` -- in practice native Windows. It is an input to the key
#: derivation below, so changing this value changes the Fernet key on those
#: platforms and makes a store written there undecryptable. A store written
#: on Linux or WSL derives from the machine id and never reaches this, which
#: is why the rename below was safe for the existing one.
_TAG_FALLBACK = b"Humanitys-Last-Disease"


def _machine_tag() -> bytes:
    machine = Path("/etc/machine-id")
    if machine.is_file():
        text = machine.read_text(encoding="utf-8").strip()
        if text:
            return text.encode("utf-8")
    node = os.uname().nodename.strip() if hasattr(os, "uname") else ""
    return node.encode("utf-8") or _TAG_FALLBACK


def _write_private(path: Path, data: bytes) -> None:
    directory = path.parent
    directory.mkdir(parents=True, exist_ok=True)
    os.chmod(directory, 0o700)
    temporary = directory / f".{path.name}.tmp"
    try:
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.write(fd, data)
        finally:
            os.close(fd)
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise
