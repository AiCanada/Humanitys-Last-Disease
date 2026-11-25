"""Exact generated-model selection and bounded snapshots for MolProbity."""

from __future__ import annotations

import gzip
import os
import re
import stat
import zlib
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from importlib.util import find_spec
from io import BytesIO, StringIO
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rfproteina.disease.validation_checks import ValidationCandidate

MAX_MODELS = 8
MAX_STRUCTURE_BYTES = 16 * 1024 * 1024
MAX_ENCODED_BYTES = 8 * 1024 * 1024
BLOCKERS = frozenset({
    "molprobity_models_missing", "molprobity_too_many_models", "molprobity_candidate_not_found",
    "molprobity_candidate_ambiguous", "molprobity_structure_unreadable", "molprobity_input_over_limit",
    "molprobity_zstd_unavailable", "molprobity_structure_invalid",
})
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
# A zstd block emits at most 128 KiB and consumes at least its block header.
# Limit each decompressor call as well as the accumulated decoded result.
_ZSTD_STEP = 64


@dataclass(frozen=True, slots=True)
class ModelInput:
    candidate_id: str
    path: Path | None
    blocker: str | None = None


class MolProbityInputError(ValueError):
    """A coordinate snapshot could not be read safely or exactly."""


def valid_candidate_id(value: object) -> bool:
    return isinstance(value, str) and _ID.fullmatch(value) is not None


def _encoded_limit(path: Path) -> int:
    return MAX_ENCODED_BYTES if path.name.endswith((".gz", ".gzip", ".zst")) else MAX_STRUCTURE_BYTES


def _meta(path: Path) -> os.stat_result:
    """Inspect the authoritative path without opening or following an alias."""
    # Check the leaf first so a special file cannot be mistaken for a usable
    # path. Comparing to the absolute spelling also checks all parent names.
    details = path.lstat()
    if not stat.S_ISREG(details.st_mode) or path.resolve(strict=True) != path.absolute():
        raise MolProbityInputError("molprobity_structure_unreadable")
    if details.st_size > _encoded_limit(path):
        raise MolProbityInputError("molprobity_input_over_limit")
    if details.st_size == 0:
        raise MolProbityInputError("molprobity_structure_invalid")
    return details


def _has_zstd() -> bool:
    try:
        return find_spec("zstandard") is not None
    except (ImportError, ValueError):
        return False


def resolve_inputs(
    directory: Path, candidate: str | None = None, *, candidates: Iterable[ValidationCandidate] | None = None,
) -> tuple[list[ModelInput], str | None]:
    """Select exact generated mmCIFs using the normal candidate inventory and metadata only."""
    from rfproteina.disease.screen import _DESIGN_SUFFIXES, _design_suffix, _sample_id
    from rfproteina.disease.validation_checks import _candidates, _inventory

    if candidate is not None and not valid_candidate_id(candidate):
        return [], "molprobity_candidate_not_found"
    try:
        root = directory.resolve()
        if candidates is None:
            inventory = _inventory(directory)
            if inventory.structure_blocker:
                return [], "molprobity_structure_unreadable"
            candidates = _candidates(inventory)
        items = tuple(candidates)
    except (OSError, RuntimeError, ValueError):
        return [], "molprobity_structure_unreadable"
    selected = items if candidate is None else tuple(item for item in items if item.example_id == candidate)
    if not selected:
        return [], "molprobity_models_missing" if candidate is None else "molprobity_candidate_not_found"
    if candidate is not None and len(selected) != 1:
        return [], "molprobity_candidate_ambiguous"
    if len(selected) > MAX_MODELS:
        return [], "molprobity_too_many_models"

    ids = Counter(item.example_id for item in items)
    preferred = [
        max(item.files, key=lambda path: _DESIGN_SUFFIXES.index(_design_suffix(path)), default=None) for item in items
    ]
    short = Counter(_sample_id(path) for path in preferred if path is not None)
    # The normal fallback can still yield the bare name for a root sample with
    # a same-named nested sample. Do not accept that bare selection here.
    rows = []
    for item, path in zip(items, preferred, strict=True):
        identifier = item.example_id
        if candidate is not None and identifier != candidate:
            continue
        if (
            not valid_candidate_id(identifier) or ids[identifier] != 1
            or (path is not None and _sample_id(path) == identifier and short[identifier] > 1)
        ):
            rows.append(ModelInput("", None, "molprobity_candidate_ambiguous"))
            continue
        assert identifier is not None
        if path is None or not path.name.endswith(_DESIGN_SUFFIXES) or not path.absolute().is_relative_to(root):
            rows.append(ModelInput(identifier, None, "molprobity_structure_unreadable"))
            continue
        try:
            _meta(path)
            if path.name.endswith(".zst") and not _has_zstd():
                raise MolProbityInputError("molprobity_zstd_unavailable")
        except (OSError, RuntimeError):
            rows.append(ModelInput(identifier, None, "molprobity_structure_unreadable"))
        except MolProbityInputError as exc:
            rows.append(ModelInput(identifier, None, str(exc)))
        else:
            rows.append(ModelInput(identifier, path))
    return rows, None


def _open(path: Path) -> int:
    """On POSIX anchor every component to avoid following a raced parent alias."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0)
    if os.open not in os.supports_dir_fd or not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        return os.open(path, flags)
    parent_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    parent = os.open(path.anchor, parent_flags)
    try:
        for part in path.parts[1:-1]:
            child = os.open(part, parent_flags, dir_fd=parent)
            os.close(parent)
            parent = child
        return os.open(path.name, flags, dir_fd=parent)
    finally:
        os.close(parent)


def _signature(info: os.stat_result) -> tuple[int, ...]:
    return info.st_dev, info.st_ino, info.st_mode, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def _snapshot(path: Path) -> bytes:
    path = path.absolute()
    limit = _encoded_limit(path)
    with os.fdopen(_open(path), "rb") as file:
        opened = os.fstat(file.fileno())
        if not stat.S_ISREG(opened.st_mode) or _signature(_meta(path)) != _signature(opened):
            raise MolProbityInputError("molprobity_structure_unreadable")
        if opened.st_size > limit:
            raise MolProbityInputError("molprobity_input_over_limit")
        raw = file.read(limit + 1)
        if len(raw) > limit:
            raise MolProbityInputError("molprobity_input_over_limit")
        if (
            len(raw) != opened.st_size or _signature(os.fstat(file.fileno())) != _signature(opened)
            or _signature(_meta(path)) != _signature(opened)
        ):
            raise MolProbityInputError("molprobity_structure_unreadable")
    return raw


def _zstd_decode(raw: bytes) -> bytes:
    try:
        import zstandard
    except (ImportError, OSError) as exc:
        raise MolProbityInputError("molprobity_zstd_unavailable") from exc
    decoded = bytearray()
    try:
        context = zstandard.ZstdDecompressor(max_window_size=MAX_STRUCTURE_BYTES)
        decoder = context.decompressobj()
        for start in range(0, len(raw), _ZSTD_STEP):
            part = raw[start:start + _ZSTD_STEP]
            while part:
                if decoder.eof:
                    decoder = context.decompressobj()
                decoded.extend(decoder.decompress(part))
                if len(decoded) > MAX_STRUCTURE_BYTES:
                    raise MolProbityInputError("molprobity_input_over_limit")
                part = decoder.unused_data
        if not decoder.eof:
            raise MolProbityInputError("molprobity_structure_invalid")
    except zstandard.ZstdError as exc:
        raise MolProbityInputError("molprobity_structure_invalid") from exc
    return bytes(decoded)


def read_structure(path: Path) -> bytes:
    """Return the original bounded, decoded mmCIF bytes without modifying coordinates."""
    try:
        _meta(path)
        raw = _snapshot(path)
    except (OSError, RuntimeError) as exc:
        raise MolProbityInputError("molprobity_structure_unreadable") from exc
    try:
        if path.name.endswith((".gz", ".gzip")):
            with gzip.GzipFile(fileobj=BytesIO(raw)) as reader:
                raw = reader.read(MAX_STRUCTURE_BYTES + 1)
        elif path.name.endswith(".zst"):
            raw = _zstd_decode(raw)
        if len(raw) > MAX_STRUCTURE_BYTES:
            raise MolProbityInputError("molprobity_input_over_limit")
        decoded = raw.decode("utf-8")
        with StringIO(decoded) as stream:
            title = next((line.strip().split(maxsplit=1)[0] for line in stream if line.strip() and not
                          line.lstrip().startswith("#")), "")
        if not title.casefold().startswith("data_") or len(title) <= len("data_"):
            raise MolProbityInputError("molprobity_structure_invalid")
    except (EOFError, gzip.BadGzipFile, UnicodeError, zlib.error) as exc:
        raise MolProbityInputError("molprobity_structure_invalid") from exc
    return raw
