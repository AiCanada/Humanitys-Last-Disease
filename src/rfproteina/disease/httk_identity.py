"""Offline identity crosswalk for the intersection of two pinned EPA releases."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from rfproteina.disease import ligand_inputs
from rfproteina.disease.screening_tools import data_dir

SOURCE = "EPA DSSTox CCD 2025-12-09"
ARCHIVE_URL = "https://clowder.edap-cluster.com/api/files/69529775e4b0731a616efc4b"
ARCHIVE_SHA256 = "66fc9d4d3bda053ab60ee01e2ff0ed4ba030339588c40711b6939faede6cf4c3"
ARCHIVE_BYTES = 289_824_966
INDEX_SHA256 = "fae55e4e02f741e7cb00db5d407061a8818540ebcb2fd6dcdb5a1f804ae01604"
STANDARD_KEY = re.compile(r"[A-Z]{14}-[A-Z]{8}SA-[A-Z]")


class IdentityReferenceError(ValueError):
    """The local structure crosswalk failed availability or release integrity checks."""


def index_path() -> Path:
    return data_dir(".cache", "httk", "2.7.4", "dsstox-20251209.tsv")


def load_index() -> dict[tuple[str, str], str]:
    """Verify the complete small index before trusting any single chemical association."""
    try:
        raw = ligand_inputs.read_regular(index_path(), 2 * 1024**2, "httk_identity", ValueError)
        if hashlib.sha256(raw).hexdigest() != INDEX_SHA256:
            raise ValueError("unverified index")
    except ValueError as exc:
        raise IdentityReferenceError("httk_identity_reference_unavailable") from exc
    return {(cas, sid): key for cas, sid, key in (line.split("\t") for line in raw.decode("ascii").splitlines())}
