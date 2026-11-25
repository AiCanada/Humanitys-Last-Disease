"""Bounded DELPHI nominations and complete paired-variable-domain validation."""

from __future__ import annotations

import os
import re
import stat
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path

MANIFEST = "delphi_pairs.tsv"
MAX_BYTES = 1024 * 1024
MAX_ROWS = 64
MAX_DOMAIN_LENGTH = 135
MAX_CDR3_LENGTH = 25
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
NUMBERING_BLOCKERS = frozenset({
    "delphi_domain_numbering_unavailable", "delphi_domain_numbering_failed",
    "delphi_chain_roles_invalid", "delphi_domain_incomplete", "delphi_cdr3_unsupported",
})
_HEADER = ["candidate_id", "vh", "vl"]
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
_CANONICAL = frozenset(AMINO_ACIDS)
_INSERTION = re.compile(r"[A-Z]?")


@dataclass(frozen=True)
class Pair:
    """One candidate's explicitly ordered domain sequences, or its named input blocker."""

    candidate_id: str
    vh: str = field(default="", repr=False)
    vl: str = field(default="", repr=False)
    blocker: str | None = None


@dataclass(frozen=True)
class NumberedPair:
    """A complete VH/VL pair's model-compatible HCDR3, or its named blocker."""

    cdr3: str | None = field(default=None, repr=False)
    blocker: str | None = None


def valid_candidate_id(value: object) -> bool:
    return isinstance(value, str) and _ID.fullmatch(value) is not None


def read_manifest(directory: Path) -> tuple[bytes | None, str | None]:
    """Read only a bounded, stable, regular nomination file, without following symlinks."""
    path = directory / MANIFEST
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return None, "delphi_pairs_missing"
    except OSError:
        return None, "delphi_pairs_unreadable"
    if not stat.S_ISREG(metadata.st_mode):
        return None, "delphi_pairs_unreadable"
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0)
    try:
        with os.fdopen(os.open(path, flags), "rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode) or not os.path.samestat(metadata, before):
                return None, "delphi_pairs_unreadable"
            if before.st_size > MAX_BYTES:
                return None, "delphi_pairs_over_limit"
            raw = stream.read(MAX_BYTES + 1)
            after = os.fstat(stream.fileno())
    except OSError:
        return None, "delphi_pairs_unreadable"
    if len(raw) > MAX_BYTES:
        return None, "delphi_pairs_over_limit"
    if (
        len(raw) != before.st_size
        or (before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        != (after.st_size, after.st_mtime_ns, after.st_ctime_ns)
    ):
        return None, "delphi_pairs_unreadable"
    return raw, None


def sequence_blocker(vh: str, vl: str) -> str | None:
    """Check that both sequences fit the released DELPHI encoding contract."""
    if not vh or not vl:
        return "delphi_domains_required"
    if max(len(vh), len(vl)) > MAX_DOMAIN_LENGTH:
        return "delphi_domain_length_out_of_range"
    if not (set(vh) <= _CANONICAL and set(vl) <= _CANONICAL):
        return "delphi_domain_residues_unsupported"
    return None


def read_pairs(directory: Path, candidate: str | None = None) -> tuple[list[Pair], str | None]:
    """Read ordered, explicitly nominated VH/VL pairs in run or exact candidate scope."""
    raw, blocker = read_manifest(directory)
    if raw is None:
        return [], blocker
    return parse_manifest(raw, candidate)


def parse_manifest(raw: bytes, candidate: str | None = None) -> tuple[list[Pair], str | None]:
    """Decode a nomination snapshot obtained by :func:`read_manifest`."""
    if len(raw) > MAX_BYTES:
        return [], "delphi_pairs_over_limit"
    try:
        with StringIO(raw.decode("utf-8-sig"), newline=None) as stream:
            if next(stream, "").removesuffix("\n").split("\t", 3) != _HEADER:
                return [], "delphi_pair_header_invalid"
            rows = []
            for line in stream:
                fields = line.removesuffix("\n")
                if not fields:
                    continue
                if len(rows) == MAX_ROWS:
                    return [], "delphi_pairs_over_limit"
                rows.append(fields.split("\t", 3))
    except UnicodeError:
        return [], "delphi_pairs_unreadable"
    if not rows:
        return [], "delphi_pairs_empty"
    if candidate is not None and not valid_candidate_id(candidate):
        return [], "delphi_candidate_not_nominated"
    counts = Counter(row[0] for row in rows)
    pairs: list[Pair] = []
    for row in rows:
        name, *values = row
        if candidate is not None and name != candidate:
            continue
        if not valid_candidate_id(name):
            pairs.append(Pair("", blocker="delphi_candidate_id_invalid"))
        elif counts[name] != 1:
            pairs.append(Pair(name, blocker="delphi_candidate_id_duplicate"))
        elif len(values) != 2:
            pairs.append(Pair(name, blocker="delphi_pair_width_invalid"))
        elif blocker := sequence_blocker(*values):
            pairs.append(Pair(name, blocker=blocker))
        else:
            pairs.append(Pair(name, *values))
    return (pairs, None) if pairs else ([], "delphi_candidate_not_nominated")


def _domain(
    result: object, sequence: str, *, heavy: bool,
) -> tuple[tuple[tuple[int, str], ...], str | None]:
    if not isinstance(result, Mapping) or result.get("error") or result.get("scheme") != "imgt":
        return (), "delphi_domain_numbering_failed"
    roles, end = (("H",), 128) if heavy else (("K", "L"), 127)
    if result.get("chain_type") not in roles:
        return (), "delphi_chain_roles_invalid"
    numbering = result.get("numbering")
    if not isinstance(numbering, list | tuple) or len(numbering) > 128 + MAX_DOMAIN_LENGTH:
        return (), "delphi_domain_numbering_failed"
    seen: set[tuple[int, str]] = set()
    occupied: list[tuple[int, str]] = []
    previous = 0
    for entry in numbering:
        if not (isinstance(entry, list | tuple) and len(entry) == 2):
            return (), "delphi_domain_numbering_failed"
        label, aa = entry
        if not (isinstance(label, list | tuple) and len(label) == 2 and isinstance(aa, str)):
            return (), "delphi_domain_numbering_failed"
        position, insertion = label
        if (
            type(position) is not int or not 1 <= position <= 128 or position < previous
            or not isinstance(insertion, str) or _INSERTION.fullmatch(insertion.strip()) is None
        ):
            return (), "delphi_domain_numbering_failed"
        position_id = (position, insertion.strip())
        if position_id in seen:
            return (), "delphi_domain_numbering_failed"
        seen.add(position_id)
        previous = position
        if aa != "-":
            if aa not in _CANONICAL:
                return (), "delphi_domain_numbering_failed"
            occupied.append((position, aa))
    # The reported end is inclusive. ANARCII outputs unoccupied terminal gaps too,
    # so exact sequence equality and covered framework positions are both required.
    minimum = {1, 23, 41, 104, *range(118, end + 1)}
    if (
        result.get("query_start") != 0 or result.get("query_end") != len(sequence) - 1
        or "".join(aa for _, aa in occupied) != sequence
        or not minimum <= {position for position, _ in occupied}
        or occupied[-1][0] != end
    ):
        return (), "delphi_domain_incomplete"
    return tuple(occupied), None


def number_pairs(
    pairs: Sequence[Pair], *, number: Callable[[dict[str, str]], object] | None = None,
) -> dict[str, NumberedPair]:
    """Number each unique domain on CPU, preserve pairing by keys, and extract IMGT HCDR3."""
    reports: dict[str, NumberedPair] = {}
    counts = Counter(pair.candidate_id for pair in pairs)
    eligible = []
    for pair in pairs:
        blocker = pair.blocker or sequence_blocker(pair.vh, pair.vl)
        if counts[pair.candidate_id] != 1:
            blocker = "delphi_candidate_id_duplicate"
        if not valid_candidate_id(pair.candidate_id):
            continue
        if blocker:
            reports[pair.candidate_id] = NumberedPair(blocker=blocker)
        else:
            eligible.append(pair)
    if not eligible:
        return reports
    keys: dict[str, str] = {}
    for pair in eligible:
        for seq in (pair.vh, pair.vl):
            if seq not in keys:
                keys[seq] = f"domain_{len(keys)}"
    try:
        if number is None:
            from anarcii import Anarcii

            number = Anarcii(seq_type="antibody", batch_size=8, cpu=True, ncpu=1, verbose=False).number
        result = number({label: seq for seq, label in keys.items()})
    except (ImportError, OSError):
        blocker = "delphi_domain_numbering_unavailable"
    except (RuntimeError, ValueError, TypeError, KeyError, IndexError, OverflowError):
        blocker = "delphi_domain_numbering_failed"
    else:
        blocker = None if isinstance(result, Mapping) else "delphi_domain_numbering_failed"
    if blocker:
        reports.update((pair.candidate_id, NumberedPair(blocker=blocker)) for pair in eligible)
        return reports
    for pair in eligible:
        heavy, vh_blocker = _domain(result.get(keys[pair.vh]), pair.vh, heavy=True)
        _, vl_blocker = _domain(result.get(keys[pair.vl]), pair.vl, heavy=False)
        blocker = vh_blocker or vl_blocker
        if blocker:
            reports[pair.candidate_id] = NumberedPair(blocker=blocker)
            continue
        cdr3 = "".join(aa for position, aa in heavy if 105 <= position <= 117)
        # DELPHI's encoder drops an initial C as a legacy anchor normalization.
        # IMGT 104 is already excluded, so doing so here would remove a real residue.
        if not 1 <= len(cdr3) <= MAX_CDR3_LENGTH or cdr3.startswith("C"):
            reports[pair.candidate_id] = NumberedPair(blocker="delphi_cdr3_unsupported")
        else:
            reports[pair.candidate_id] = NumberedPair(cdr3=cdr3)
    return reports
