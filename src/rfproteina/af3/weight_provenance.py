"""Verify the parameter family shared by the AF3 host and its standalone worker."""

import hashlib
import json
import os
import re
from functools import lru_cache
from pathlib import Path

STAMP = "of3_ported_weights.provenance.json"
_OF3_CHECKPOINT = "of3_ported_weights.bin.zst"
# Published converted file in scripts/oracle_artifacts.sh's pinned Hub revision.
_RELEASE_BYTES = 1_369_238_320
_RELEASE_SHA256 = "7ae00e49f8507593d6a8c5a7c0389e2151fea9a79062069979c42ebb5bbf14d2"


def _version(info: os.stat_result) -> tuple[int, ...]:
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


@lru_cache(maxsize=4)
def _digest(path: Path, version: tuple[int, ...]) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        if _version(os.fstat(handle.fileno())) != version:
            raise ValueError("OpenFold3 checkpoint changed before verification")
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
        if _version(os.fstat(handle.fileno())) != version or _version(path.stat()) != version:
            raise ValueError("OpenFold3 checkpoint changed during verification")
    return digest.hexdigest()


def openfold3_checkpoint(model_dir: Path) -> str | None:
    """Return the verified checkpoint digest, or None for an unmarked AF3 directory."""
    stamp = model_dir / STAMP
    if not stamp.exists():
        if (model_dir / _OF3_CHECKPOINT).exists():
            raise ValueError("OpenFold3 checkpoint is missing its provenance record")
        return None
    provenance = json.loads(stamp.read_text(encoding="utf-8"))
    if not isinstance(provenance, dict) or provenance.get("weight_family") != "openfold3":
        raise ValueError("OpenFold3 provenance has an unexpected weight family")
    record = provenance.get("converted_checkpoint")
    if not isinstance(record, dict):
        raise ValueError("OpenFold3 provenance is missing converted_checkpoint")
    name, size, expected = record.get("filename"), record.get("bytes"), record.get("sha256")
    if not isinstance(name, str) or not name or name in {".", ".."} or "/" in name or "\\" in name or ":" in name:
        raise ValueError("OpenFold3 provenance has an invalid checkpoint filename")
    if expected is None and name == _OF3_CHECKPOINT and size == _RELEASE_BYTES:
        expected = _RELEASE_SHA256
    if (
        type(size) is not int or size <= 0 or not isinstance(expected, str)
        or not re.fullmatch(r"[a-fA-F0-9]{64}", expected)
    ):
        raise ValueError("OpenFold3 provenance requires a byte count and SHA-256")
    path = model_dir / name
    if not path.is_file() or path.stat().st_size != size:
        raise ValueError("OpenFold3 checkpoint is missing or has a different size")
    actual = _digest(path.resolve(), _version(path.stat()))
    if actual != expected.lower():
        raise ValueError("OpenFold3 checkpoint SHA-256 disagrees with its provenance")
    return actual


def require_weight_family(expected: str | None, actual: str | None, checkpoint: str | None) -> None:
    if expected is None:
        return
    if actual != expected:
        raise ValueError(f"Expected {expected} weight family, found {actual or 'unreported'}")
    if expected == "openfold3" and (not isinstance(checkpoint, str) or not re.fullmatch(r"[0-9a-f]{64}", checkpoint)):
        raise ValueError("OpenFold3 result is missing its verified checkpoint SHA-256")
