"""Alignment-based proteome off-target scan, via MMseqs2, extending the 9-mer axis.

``liability.py`` already carries a ``proteome_offtarget`` axis: exact, distinct
9-mer identity of a design against the human reference proteome. That is a strict
"is any 9-residue window an exact substring of a human protein" test. It is fast
and has no false alignment, but it sees nothing below an exact 9-mer -- a design
that is 80% identical to a human protein over 200 residues, with no exact 9-mer
run, clears it.

This screen is the complementary view, not a replacement: a sensitive local
alignment of each design against the same human proteome FASTA with MMseqs2,
reporting per design the best off-target hit's sequence identity, coverage,
E-value and bit score. It is registered as its own axis
(:data:`AXIS_NAME`) so it never displaces the exact-9-mer one.

A design's intended target may itself be a human protein, and that on-target
would be the top hit here. Pass its UniProt accession(s) in ``ignore_targets`` so
the self-match is not counted as an off-target; the record says when the top hit
was excluded for that reason.

The human proteome FASTA (``UP000005640_9606.fasta.gz``, 7.38 MB, CC-BY-4.0) is
already on disk in the data folder; MMseqs2 (MIT) reads it gzipped. Nothing here
is downloaded. The binary is ``MMSEQS``, then ``DATA_PATH/mmseqs/mmseqs``, then
``PATH``. If it is absent, or the proteome path is absent, the screen returns
``untested`` with a blocker naming what is missing.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from rfproteina.disease.combinations import rollup_axis
from rfproteina.disease.screening_tools import (
    CLEAR,
    HIT,
    UNTESTED,
    ScreeningInputError,
    axis_record,
    run_tool,
    tool_code_path,
    write_fasta,
)

MMSEQS = "mmseqs"
#: An executable path. It wins over the copy installed under ``DATA_PATH``.
MMSEQS_ENV = "MMSEQS"

#: The axis this screen writes to. Distinct from ``proteome_offtarget`` so the
#: exact-9-mer axis in ``liability.py`` is left untouched.
AXIS_NAME = "proteome_offtarget_alignment"

#: MMseqs2 ``easy-search`` output columns, in order.
SEARCH_COLUMNS = ("query", "target", "fident", "alnlen", "qcov", "tcov", "evalue", "bits")

#: Thresholds with provenance. None of these is a published off-target cut; they
#: are this program's screening defaults, and a reviewer should calibrate them.
OFFTARGET_THRESHOLDS: dict[str, dict[str, object]] = {
    "identity_min": {
        "value": 0.5,
        "status": "suggested",
        "source": "program default pending calibration; no published cross-reactivity identity cut",
    },
    "coverage_min": {
        "value": 0.5,
        "status": "suggested",
        "source": "program default pending calibration (min of query and target coverage)",
    },
    "evalue_max": {
        "value": 1e-3,
        "status": "suggested",
        "source": "program default pending calibration",
    },
}

#: Free, operator-fetched structure database for the structural off-target path,
#: with size read from foldseek.steineggerlab.workers.dev on 2026-11-16.
STRUCTURE_OFFTARGET_DATABASE = {
    "name": "Alphafold/Proteome (human subset)",
    "command": "foldseek databases Alphafold/Proteome afdb_prot tmp",
    "download": "afdb_proteome.tar.gz 1.7 GB (whole-proteome set; CC-BY-4.0)",
    "licence": "CC-BY-4.0",
    "note": "Not on disk. The structural off-target path is unavailable until this is fetched.",
}


@dataclass(frozen=True)
class OffTargetHit:
    """One alignment of a design against a proteome entry."""

    query: str
    target: str
    fident: float
    alnlen: int
    qcov: float
    tcov: float
    evalue: float
    bits: float

    @property
    def coverage(self) -> float:
        """The smaller of query and target coverage, the conservative figure."""
        return min(self.qcov, self.tcov)


class OffTargetUnavailableError(RuntimeError):
    """MMseqs2 or the proteome is absent. Carries the blocker to record as untested."""

    def __init__(self, blocker: str) -> None:
        super().__init__(blocker)
        self.blocker = blocker


def _executable(path: Path) -> bool:
    return path.is_file() and os.access(path, os.X_OK)


def mmseqs_command() -> str | None:
    """The MMseqs2 executable, or None when this machine has none.

    ``MMSEQS`` wins, then the binary wherever
    :func:`screening_tools.tool_code_path` finds it -- ``DATA_PATH/mmseqs`` if
    a copy is still there, otherwise the checkout's
    ``screening subpackage/mmseqs`` -- then ``PATH``.

    MMseqs2 is a compiled binary and therefore third-party *code*, so it moved
    into the checkout; nothing it reads moved with it.
    """
    named = os.environ.get(MMSEQS_ENV, "").strip()
    if named:
        candidate = Path(named)
        if _executable(candidate):
            return str(candidate)
        found = shutil.which(named)
        if found:
            return found
    found = tool_code_path(MMSEQS, "mmseqs", "mmseqs.exe", "bin/mmseqs")
    if found is not None and _executable(found):
        return str(found)
    return shutil.which(MMSEQS)


def mmseqs_available() -> bool:
    """Whether an MMseqs2 executable can be resolved. Checked, never assumed."""
    return mmseqs_command() is not None


def accession_of(target_id: str) -> str:
    """The UniProt accession in a FASTA id like ``sp|P69905|HBA_HUMAN``; else the id itself."""
    parts = target_id.split("|")
    if len(parts) >= 3 and parts[0] in {"sp", "tr"}:
        return parts[1]
    return target_id


def build_search_command(
    query_fasta: Path,
    proteome_fasta: Path,
    result_out: Path,
    tmp_dir: Path,
    *,
    mmseqs: str = MMSEQS,
    sensitivity: float = 7.5,
    evalue_max: float = 1e-3,
) -> list[str]:
    """The ``mmseqs easy-search`` argument list. ``-s 7.5`` is MMseqs2's most sensitive preset."""
    return [
        mmseqs,
        "easy-search",
        str(query_fasta),
        str(proteome_fasta),
        str(result_out),
        str(tmp_dir),
        "-s",
        str(sensitivity),
        "-e",
        str(evalue_max),
        "--format-output",
        ",".join(SEARCH_COLUMNS),
    ]


def parse_search_table(text: str, columns: tuple[str, ...] = SEARCH_COLUMNS) -> dict[str, list[OffTargetHit]]:
    """All hits per query from an MMseqs2 tab-separated table.

    A row whose numeric fields do not parse is skipped rather than read as a
    zero. Hits are kept per query so the caller can exclude the on-target before
    choosing the worst off-target.
    """
    index = {name: position for position, name in enumerate(columns)}
    hits: dict[str, list[OffTargetHit]] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        fields = line.split("\t")
        if len(fields) < len(columns):
            continue
        query = fields[index["query"]]
        try:
            hit = OffTargetHit(
                query=query,
                target=fields[index["target"]],
                fident=float(fields[index["fident"]]),
                alnlen=int(float(fields[index["alnlen"]])),
                qcov=float(fields[index["qcov"]]),
                tcov=float(fields[index["tcov"]]),
                evalue=float(fields[index["evalue"]]),
                bits=float(fields[index["bits"]]),
            )
        except (KeyError, ValueError):
            continue
        hits.setdefault(query, []).append(hit)
    return hits


def worst_offtarget(
    hits: list[OffTargetHit],
    *,
    ignore_targets: frozenset[str],
    identity_min: float,
    coverage_min: float,
    evalue_max: float,
) -> tuple[OffTargetHit | None, bool]:
    """The strongest off-target hit over threshold, and whether an on-target was skipped.

    An on-target is a hit whose accession is in ``ignore_targets``; it is removed
    before thresholding so a binder of a human protein is not flagged for hitting
    its own target. "Strongest" is the highest bit score among hits that clear
    identity, coverage and E-value.
    """
    skipped_on_target = False
    qualifying: list[OffTargetHit] = []
    for hit in hits:
        if accession_of(hit.target) in ignore_targets or hit.target in ignore_targets:
            skipped_on_target = True
            continue
        if hit.fident >= identity_min and hit.coverage >= coverage_min and hit.evalue <= evalue_max:
            qualifying.append(hit)
    if not qualifying:
        return None, skipped_on_target
    return max(qualifying, key=lambda item: item.bits), skipped_on_target


def verdict(
    hits: list[OffTargetHit],
    *,
    ignore_targets: frozenset[str] = frozenset(),
    identity_min: float = 0.5,
    coverage_min: float = 0.5,
    evalue_max: float = 1e-3,
) -> dict[str, object]:
    """One design's off-target record from its hits.

    A qualifying off-target makes the axis a hit, with the identity as the
    measurement so a roll-up reads the most cross-reactive design. No qualifying
    off-target is clear.
    """
    worst, skipped = worst_offtarget(
        hits,
        ignore_targets=ignore_targets,
        identity_min=identity_min,
        coverage_min=coverage_min,
        evalue_max=evalue_max,
    )
    if worst is None:
        return axis_record(CLEAR, 0.0, None, off_target=None, skipped_on_target=skipped)
    return axis_record(
        HIT,
        round(worst.fident, 4),
        None,
        off_target=worst.target,
        identity=round(worst.fident, 4),
        coverage=round(worst.coverage, 4),
        evalue=worst.evalue,
        bits=worst.bits,
        skipped_on_target=skipped,
    )


def offtarget_scan(
    designs: list[tuple[str, str]],
    proteome_fasta: Path,
    work_dir: Path,
    *,
    ignore_targets: frozenset[str] = frozenset(),
    mmseqs: str = MMSEQS,
    identity_min: float = 0.5,
    coverage_min: float = 0.5,
    evalue_max: float = 1e-3,
    timeout_s: float = 3600.0,
) -> dict[str, dict[str, object]]:
    """An off-target record per design ``(id, sequence)``.

    Raises :class:`ScreeningInputError` when no design carries a sequence.
    Raises :class:`OffTargetUnavailableError` when MMseqs2 is not installed or the
    proteome path is absent, so the caller records every design ``untested`` with
    that one blocker instead of a made-up score.
    """
    usable = [(design_id, seq) for design_id, seq in designs if seq and seq.strip()]
    if not usable:
        raise ScreeningInputError("no_design_sequences")
    if mmseqs == MMSEQS:
        resolved = mmseqs_command()
        if resolved is None:
            raise OffTargetUnavailableError("mmseqs_not_installed")
    else:
        resolved = mmseqs
    if not Path(proteome_fasta).exists():
        raise OffTargetUnavailableError("proteome_missing")
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    query_fasta = work_dir / "designs.fasta"
    write_fasta(usable, query_fasta)
    result_out = work_dir / "offtarget.m8"
    tmp_dir = work_dir / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    command = build_search_command(
        query_fasta, Path(proteome_fasta), result_out, tmp_dir, mmseqs=resolved, evalue_max=evalue_max
    )
    result = run_tool(command, timeout_s=timeout_s)
    if not result.ok:
        raise OffTargetUnavailableError(result.blocker or "mmseqs_failed")
    text = result_out.read_text(encoding="utf-8") if result_out.is_file() else result.stdout
    hits = parse_search_table(text)
    records: dict[str, dict[str, object]] = {}
    for design_id, _seq in usable:
        records[design_id] = verdict(
            hits.get(design_id, []),
            ignore_targets=ignore_targets,
            identity_min=identity_min,
            coverage_min=coverage_min,
            evalue_max=evalue_max,
        )
    return records


def rollup(records: dict[str, dict[str, object]]) -> dict[str, object]:
    """One off-target axis for the whole batch, by the shared roll-up rule."""
    empty = axis_record(UNTESTED, None, "designs_absent", off_target=None)
    return rollup_axis(list(records), records, empty)


def unavailable_axis(blocker: str) -> dict[str, object]:
    """The axis a caller records when the scan could not run at all."""
    return axis_record(UNTESTED, None, blocker, off_target=None)


def structure_offtarget_requirements() -> dict[str, str]:
    """What the structural (Foldseek) off-target path needs. It is not on disk."""
    return dict(STRUCTURE_OFFTARGET_DATABASE)
