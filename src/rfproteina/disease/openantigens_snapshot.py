"""Verify and read a pinned OpenAntigens human-target evidence snapshot."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import stat
import sys
from collections import Counter
from collections.abc import Iterator
from functools import lru_cache
from io import StringIO
from pathlib import Path

from rfproteina.disease.ligand_inputs import read_regular

RELEASE = "2026-11-01"
INDEX_NAME = "agdesign2_portal_index.tsv"
METADATA_NAME = "portal_metadata.json"
ACCESSION_MAP_NAME = "accessible_human_topology_refined_accessible_only.csv"
INDEX_URL = f"https://openantigens.org/downloads/{INDEX_NAME}"
METADATA_URL = f"https://openantigens.org/{METADATA_NAME}"
ACCESSION_MAP_COMMIT = "49dcc77e006975ca08a29c95a4371683ada893a5"
ACCESSION_MAP_URL = (
    f"https://raw.githubusercontent.com/proteininnovation/OpenAntigens/{ACCESSION_MAP_COMMIT}"
    f"/data/inputs/{ACCESSION_MAP_NAME}"
)
INDEX_SHA256 = "72aed9d005a6634acfaf5f46fca6fb6dbd42a7944a4bd9332cad79acb9061117"
METADATA_SHA256 = "b2f85251493e8c75f688e8a5b4923d0f66c29ecd834228e8016ba1610f0ebe75"
ACCESSION_MAP_SHA256 = "ddecaede6d2cf7fcb665f652a5ae61dd83da568d3806eccf6323a095877f7d12"
ASSETS = (
    (INDEX_NAME, INDEX_URL, 1_275_884, INDEX_SHA256),
    (ACCESSION_MAP_NAME, ACCESSION_MAP_URL, 4_168_724, ACCESSION_MAP_SHA256),
    (METADATA_NAME, METADATA_URL, 1_844, METADATA_SHA256),
)
TARGET_COUNT = 5_328

_UNAVAILABLE = "openantigens_snapshot_unavailable"
_INTEGRITY = "openantigens_snapshot_integrity"
BLOCKERS = frozenset({_UNAVAILABLE, _INTEGRITY})
_ACCESSION = re.compile(r"(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})")
_ENTRY = re.compile(r"[A-Z0-9]+_HUMAN")
_TOPOLOGIES = frozenset({"GPI", "Multipass", "Secreted", "Single-pass"})
_FLAGS = (
    "has_alphafold_structure",
    "has_pdb",
    "has_interpro",
    "has_family_context",
    "has_assembly_requirements",
)


class SnapshotError(RuntimeError):
    """The pinned human-target snapshot is absent or failed verification."""


def release_directory() -> Path:
    """Resolve the data location for the active Python environment."""
    return Path(sys.prefix) / "share" / "openantigens" / RELEASE


def valid_accession(accession: object) -> bool:
    """Recognize an exact, unsuffixed UniProt accession."""
    return isinstance(accession, str) and _ACCESSION.fullmatch(accession) is not None


def _signature(directory: Path) -> tuple[tuple[int, ...], ...]:
    try:
        if not stat.S_ISDIR(directory.lstat().st_mode):
            raise SnapshotError(_INTEGRITY)
    except FileNotFoundError as exc:
        raise SnapshotError(_UNAVAILABLE) from exc
    except OSError as exc:
        raise SnapshotError(_INTEGRITY) from exc

    signatures = []
    for name, _, size, _ in ASSETS:
        try:
            info = (directory / name).lstat()
        except OSError as exc:
            raise SnapshotError(_INTEGRITY) from exc
        if not stat.S_ISREG(info.st_mode) or info.st_size != size:
            raise SnapshotError(_INTEGRITY)
        signatures.append((info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns))
    return tuple(signatures)


def _rows(raw: bytes, delimiter: str, required: set[str]) -> Iterator[dict[str, str]]:
    reader = csv.DictReader(StringIO(raw.decode("utf-8"), newline=""), delimiter=delimiter, strict=True)
    fields = reader.fieldnames
    if not fields or len(fields) != len(set(fields)) or not required.issubset(fields):
        raise SnapshotError(_INTEGRITY)
    for row in reader:
        if None in row or None in row.values():
            raise SnapshotError(_INTEGRITY)
        yield row


def _parse(data: dict[str, bytes]) -> dict[str, dict]:
    metadata = json.loads(data[METADATA_NAME])
    if (
        not isinstance(metadata, dict)
        or metadata.get("database") != "OpenAntigens"
        or metadata.get("portal_build_date") != RELEASE
        or metadata.get("target_count") != TARGET_COUNT
        or metadata.get("status_counts") != {"ok": TARGET_COUNT}
    ):
        raise SnapshotError(_INTEGRITY)

    source = {}
    accessions = set()
    for row in _rows(data[ACCESSION_MAP_NAME], ",", {"accession", "uniprot_entry_name", "simplified_topology_bucket"}):
        name, accession, topology = row["uniprot_entry_name"], row["accession"], row["simplified_topology_bucket"]
        if (
            not valid_accession(accession)
            or not _ENTRY.fullmatch(name)
            or topology not in _TOPOLOGIES
            or name in source
            or accession in accessions
        ):
            raise SnapshotError(_INTEGRITY)
        source[name] = accession, topology
        accessions.add(accession)

    targets = {}
    entry_names = set()
    topology_counts = Counter()
    required = {"query", "source_column", "entry_name", "topology_bucket", "status", *_FLAGS}
    for row in _rows(data[INDEX_NAME], "\t", required):
        name, topology = row["entry_name"], row["topology_bucket"]
        # The query preserves the reviewed entry name supplied to the portal build.
        if row["query"] not in source or row["source_column"] != "uniprot_name" or row["status"] != "ok":
            raise SnapshotError(_INTEGRITY)
        accession, source_topology = source.pop(row["query"])
        if (
            not _ENTRY.fullmatch(name)
            or name in entry_names
            or topology != source_topology
            or any(row[flag] not in {"0", "1"} for flag in _FLAGS)
        ):
            raise SnapshotError(_INTEGRITY)
        entry_names.add(name)
        topology_counts[topology] += 1
        targets[accession] = {
            "entry_name": name,
            "topology": topology,
            **{flag: row[flag] == "1" for flag in _FLAGS},
        }

    if source or len(targets) != TARGET_COUNT or topology_counts != metadata.get("topology_counts"):
        raise SnapshotError(_INTEGRITY)
    return targets


@lru_cache(maxsize=2)
def _verified(directory: Path, signature: tuple[tuple[int, ...], ...]) -> dict[str, dict]:
    try:
        data = {}
        for name, _, size, digest in ASSETS:
            raw = read_regular(directory / name, size, "openantigens_snapshot", ValueError)
            if len(raw) != size or hashlib.sha256(raw).hexdigest() != digest:
                raise SnapshotError(_INTEGRITY)
            data[name] = raw
        records = _parse(data)
    except (ValueError, csv.Error) as exc:
        raise SnapshotError(_INTEGRITY) from exc
    try:
        unchanged = signature == _signature(directory)
    except SnapshotError as exc:
        raise SnapshotError(_INTEGRITY) from exc
    if not unchanged:
        raise SnapshotError(_INTEGRITY)
    return records


def verify_directory(directory: Path) -> None:
    """Validate staged snapshot files before installation."""
    _verified(directory, _signature(directory))


def installed() -> bool:
    """Report whether the complete pinned snapshot is ready to use."""
    try:
        verify_directory(release_directory())
    except SnapshotError:
        return False
    return True


def target(accession: str) -> dict | None:
    """Return published evidence availability for an exact human accession."""
    directory = release_directory()
    evidence = _verified(directory, _signature(directory)).get(accession)
    return evidence.copy() if evidence is not None else None
