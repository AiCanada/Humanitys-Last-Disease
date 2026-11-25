"""Review existing red-cell assay evidence for nominated therapeutic peptide samples."""

from __future__ import annotations

import hashlib
import math
import os
import re
import stat
from collections import Counter
from collections.abc import Collection, Sequence
from decimal import Decimal, InvalidOperation
from io import StringIO
from pathlib import Path

from rfproteina.disease.corpus_provenance import declare

METHOD_ID = "peptide_rbc_evidence"
MANIFEST = "peptide_rbc_evidence.tsv"
METHOD = "supplied_peptide_rbc_assay_v1"
ENDPOINT = "observed_hemolysis"
UNIT = "percent"
MAX_BYTES = 256 * 1024
MAX_ROWS = 256

HEADER = (
    "candidate_id", "observation_id", "formulation_id", "source_id", "species", "matrix", "context_id",
    "concentration", "concentration_unit", "exposure_minutes", "hemolysis_percent", "controls_qc",
)
ID_FIELDS = ("candidate_id", "observation_id", "formulation_id", "source_id", "context_id")
SPECIES = frozenset({"human", "mouse", "rat", "rabbit", "dog", "pig", "sheep", "cow", "cynomolgus"})
MATRICES = frozenset({"washed_rbc", "whole_blood"})
CONCENTRATION_UNITS = frozenset({"uM", "ug/mL", "mg/mL"})
BLOCKERS = frozenset({
    "peptide_rbc_missing", "peptide_rbc_unreadable", "peptide_rbc_over_limit", "peptide_rbc_header_invalid",
    "peptide_rbc_empty", "peptide_rbc_candidates_unavailable", "peptide_rbc_candidate_unknown",
    "peptide_rbc_candidate_not_nominated", "peptide_rbc_identifier_invalid", "peptide_rbc_duplicate_observation",
    "peptide_rbc_row_width_invalid", "peptide_rbc_species_unsupported", "peptide_rbc_matrix_unsupported",
    "peptide_rbc_controls_failed", "peptide_rbc_controls_unreported", "peptide_rbc_controls_invalid",
    "peptide_rbc_exposure_invalid", "peptide_rbc_unit_unsupported", "peptide_rbc_readout_invalid",
    "peptide_rbc_no_qualified_observations",
})
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}")
_DECIMAL = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")


def valid_identifier(value: object) -> bool:
    """Use opaque short evidence and sample labels, not free text or paths."""
    return isinstance(value, str) and _ID.fullmatch(value) is not None


def is_number(value: object) -> bool:
    """Reject booleans, non-finite values, and overflowing forged integers."""
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _decimal(text: str, *, upper: int | None = None) -> float | None:
    if len(text) > 32 or not _DECIMAL.fullmatch(text):
        return None
    try:
        exact = Decimal(text)
    except InvalidOperation:
        return None
    if exact < 0 or (upper is not None and exact > upper):
        return None
    number = float(exact)
    if not is_number(number):
        return None
    # Do not turn underflow into measured zero or silently change a boundary value.
    if Decimal(str(number)) != exact:
        return None
    return 0.0 if number == 0 else number


def read_manifest(directory: Path) -> tuple[bytes | None, str | None]:
    """Read the bounded local TSV as a stable regular file, without following symlinks."""
    source = directory / MANIFEST
    try:
        before = source.lstat()
        if not stat.S_ISREG(before.st_mode):
            return None, "peptide_rbc_unreadable"
        if before.st_size > MAX_BYTES:
            return None, "peptide_rbc_over_limit"
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0)
        with os.fdopen(os.open(source, flags), "rb") as stream:
            opened = os.fstat(stream.fileno())
            if not stat.S_ISREG(opened.st_mode) or not os.path.samestat(before, opened):
                return None, "peptide_rbc_unreadable"
            if opened.st_size > MAX_BYTES:
                return None, "peptide_rbc_over_limit"
            raw = stream.read(MAX_BYTES + 1)
            after = os.fstat(stream.fileno())
    except FileNotFoundError:
        return None, "peptide_rbc_missing"
    except OSError:
        return None, "peptide_rbc_unreadable"
    if len(raw) > MAX_BYTES:
        return None, "peptide_rbc_over_limit"
    if (
        len(raw) != opened.st_size
        or (opened.st_size, opened.st_mtime_ns, opened.st_ctime_ns)
        != (after.st_size, after.st_mtime_ns, after.st_ctime_ns)
    ):
        return None, "peptide_rbc_unreadable"
    return raw, None


def _measurement(fields: dict[str, str]) -> tuple[dict, str | None]:
    if not all(valid_identifier(fields[key]) for key in ID_FIELDS):
        return {}, "peptide_rbc_identifier_invalid"
    if fields["species"] not in SPECIES:
        return {}, "peptide_rbc_species_unsupported"
    if fields["matrix"] not in MATRICES:
        return {}, "peptide_rbc_matrix_unsupported"
    controls = fields["controls_qc"]
    if controls != "accepted":
        blocker = {
            "failed": "peptide_rbc_controls_failed", "not_reported": "peptide_rbc_controls_unreported",
            "": "peptide_rbc_controls_unreported",
        }.get(controls, "peptide_rbc_controls_invalid")
        return {}, blocker
    if fields["concentration_unit"] not in CONCENTRATION_UNITS:
        return {}, "peptide_rbc_unit_unsupported"
    concentration = _decimal(fields["concentration"])
    time = _decimal(fields["exposure_minutes"])
    if concentration is None or time is None or concentration <= 0 or time <= 0:
        return {}, "peptide_rbc_exposure_invalid"
    percent = _decimal(fields["hemolysis_percent"], upper=100)
    if percent is None:
        return {}, "peptide_rbc_readout_invalid"
    return {**fields, "concentration": concentration, "exposure_minutes": time, "hemolysis_percent": percent}, None


def _one_row(row: Sequence[str], known: Collection[str], duplicates: Counter[tuple[str, str]]) -> dict:
    # Only bounded, opaque identifiers survive a refused row, never raw values.
    raw_ids = dict(zip(HEADER[:3], row[:3], strict=False))
    ids = {key: value for key, value in raw_ids.items() if valid_identifier(value)}
    if len(row) != len(HEADER):
        blocker = "peptide_rbc_row_width_invalid"
    elif not valid_identifier(row[0]) or not valid_identifier(row[1]):
        blocker = "peptide_rbc_identifier_invalid"
    elif row[0] not in known:
        blocker = "peptide_rbc_candidate_unknown"
    elif duplicates[(row[0], row[1])] != 1:
        blocker = "peptide_rbc_duplicate_observation"
    else:
        measurement, blocker = _measurement(dict(zip(HEADER, row, strict=True)))
        if blocker is None:
            return {**measurement, "status": "reported"}
    return {**ids, "status": "untested", "blocker": blocker}


def _parse(
    raw: bytes, candidate: str | None, known_candidates: Collection[str] | None,
) -> tuple[list[dict], str | None]:
    try:
        with StringIO(raw.decode("utf-8-sig"), newline=None) as stream:
            if tuple(next(stream, "").removesuffix("\n").split("\t", len(HEADER))) != HEADER:
                return [], "peptide_rbc_header_invalid"
            rows = []
            for source in stream:
                line = source.removesuffix("\n")
                if not line:
                    continue
                if len(rows) >= MAX_ROWS:
                    return [], "peptide_rbc_over_limit"
                rows.append(line.split("\t", len(HEADER)))
    except UnicodeError:
        return [], "peptide_rbc_unreadable"
    if not rows:
        return [], "peptide_rbc_empty"
    if not known_candidates:
        return [], "peptide_rbc_candidates_unavailable"
    if candidate is not None:
        if not valid_identifier(candidate) or candidate not in known_candidates:
            return [], "peptide_rbc_candidate_unknown"
        rows = [row for row in rows if row[0] == candidate]
        if not rows:
            return [], "peptide_rbc_candidate_not_nominated"
    occurrences = Counter((row[0], row[1]) for row in rows if len(row) >= 2)
    return [_one_row(row, known_candidates, occurrences) for row in rows], None


def read_observations(
    directory: Path, *, candidate: str | None = None, known_candidates: Collection[str] | None = None,
) -> tuple[list[dict], str | None]:
    """Read qualified observations and explicit refusals without deriving any score."""
    raw, blocker = read_manifest(directory)
    if raw is None:
        return [], blocker
    return _parse(raw, candidate, known_candidates)


def screen_run(
    directory: Path, *, candidate: str | None = None, known_candidates: Collection[str] | None = None,
) -> dict:
    """Report supplied assay values as evidence, with no threshold or safety decision."""
    output = {
        "method": METHOD, "endpoint": ENDPOINT, "unit": UNIT, "scope": "run" if candidate is None else "candidate",
        "status": "untested", "attempted": 0, "reported": 0, "human_reported": 0, "observations": [],
        "corpus_provenance": declare(METHOD),
    }
    raw, blocker = read_manifest(directory)
    if raw is None:
        return {**output, "blocker": blocker or "peptide_rbc_unreadable"}
    output["input_sha256"] = hashlib.sha256(raw).hexdigest()
    observations, blocker = _parse(raw, candidate, known_candidates)
    if blocker:
        return {**output, "blocker": blocker}
    reported = [row for row in observations if row["status"] == "reported"]
    output.update(
        attempted=len(observations),
        reported=len(reported),
        human_reported=sum(row["species"] == "human" for row in reported),
        observations=observations,
    )
    if not reported:
        output["blocker"] = "peptide_rbc_no_qualified_observations"
    else:
        output["status"] = "reported" if len(reported) == len(observations) else "partial"
    return output
