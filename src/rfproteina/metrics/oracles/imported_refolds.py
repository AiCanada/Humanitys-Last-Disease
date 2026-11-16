"""Import a structure folded somewhere else, so the existing metrics can use it.

A refolding oracle normally folds a design itself. Not everyone can run one:
AF3 weights are licence-gated, a container runtime may be unavailable, and some
services return a file to a browser rather than to a program. What those cases
all have in common is that the *fold already exists* -- it is simply on disk
instead of coming out of a subprocess.

This places such a fold where the oracle would have put it. Every oracle treats
a design as already refolded when its confidence JSON is at the canonical path
(:py:meth:`OracleMetric.already_refolded`), and reads the pair back through
:py:meth:`OracleMetric._load_compute_result`. So two files at two canonical
paths are the whole contract, and nothing downstream has to know the fold came
from outside: the refolding metrics, the declared pass criteria and the disease
screen's ``refold`` report all run unchanged.

Deliberately oracle-agnostic. It asks the configured oracle for its own paths
rather than hardcoding any, so the same call works for AF2, AF3, OpenFold3 or
anything added later, under whichever ``oracle_name`` and output encoding that
oracle is configured with.

What this does **not** do is decide whether you are permitted to use a
particular fold. A service's terms are between you and that service; this reads
a file you already have.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from rfproteina.utils.common import design_output_dir
from rfproteina.utils.structure_files import is_structure_file, structure_file_stem

if TYPE_CHECKING:  # pragma: no cover - typing only
    from rfproteina.metrics.oracles.oracle_metric import OracleMetric

#: Marks a confidence record this module wrote rather than an oracle. A reader
#: that finds it knows the numbers beside it were not produced by the oracle
#: whose name is on the file.
IMPORT_MARK = "imported_refold"
#: What the record says when the external fold came with no confidence values.
#: The confidence metrics are then simply absent, which every pass criterion
#: already reports as an unmet bound rather than a failed one.
NO_CONFIDENCE = "not_supplied"
#: A confidence file past this is not a confidence file.
CONFIDENCE_BYTE_LIMIT = 64 * 1024 * 1024
#: Entries one import may examine while walking ``source``. A download
#: directory is a handful of files. A path given by mistake -- a home
#: directory, a whole drive -- is not, and walking it would hang the import
#: rather than fail it. Reaching this is reported, never silent.
SCAN_LIMIT = 20000

#: ``<job>_model_<rank>`` and ``<job>_model``. AF3 writes the second spelling;
#: the AlphaFold Server numbers every file in the download it hands back.
_MODEL_RANKED = re.compile(r"_model_(\d+)$")
_MODEL_PLAIN = re.compile(r"_model$")
#: A rank a folding tool put in a filename: ``_model_0``, ``_sample_2``,
#: ``_rank0``. The rightmost one wins, so a design id that itself ends in a
#: number is not mistaken for a rank.
_RANK_IN_NAME = re.compile(r"_(?:model|sample|rank)_?(\d+)(?=$|[^0-9])")


@dataclass(frozen=True)
class PathOracle:
    """An oracle's canonical paths, without the oracle.

    Importing a fold and asking whether one has arrived are both questions
    about *where files go*. Answering them needs no model, no weights and no
    GPU -- but the real :py:class:`OracleMetric` brings torch with it, and the
    disease CLI is deliberately free of it.

    So this mirrors the four path members and nothing else. The mirroring is
    the risk, and it is covered by a test that calls ``OracleMetric``'s own
    unbound methods with one of these as ``self``: if the canonical layout ever
    changes, that test fails rather than this drifting quietly out of step.

    ``oracle_name`` must match the oracle the metrics are configured with,
    since it is in every filename.
    """

    oracle_name: str = "af3"
    output_cif_suffix: str = ".cif"

    def output_stem(self, example_id: str, save_id: str | None = None) -> str:
        return (
            f"{example_id}_{save_id}_{self.oracle_name}" if save_id is not None else f"{example_id}_{self.oracle_name}"
        )

    def confidence_path(self, out_dir: Path, example_id: str, save_id: str | None = None) -> Path:
        return design_output_dir(out_dir, example_id) / f"{self.output_stem(example_id, save_id)}_confidences.json"

    def refolded_cif_path(self, out_dir: Path, example_id: str, save_id: str | None = None) -> Path:
        return (
            design_output_dir(out_dir, example_id)
            / f"{self.output_stem(example_id, save_id)}_refolded{self.output_cif_suffix}"
        )

    def already_refolded(self, out_dir: Path, example_id: str, save_id: str | None = None) -> bool:
        return self.confidence_path(out_dir, example_id, save_id).exists()


@dataclass(frozen=True)
class ImportedRefold:
    """Where one imported fold landed, or why it did not."""

    example_id: str
    blocker: str | None = None
    structure: Path | None = None
    confidence: Path | None = None
    save_id: str | None = None

    @property
    def placed(self) -> bool:
        return self.blocker is None


@dataclass
class ImportReport:
    """Every design the import was asked about."""

    placed: list[ImportedRefold] = field(default_factory=list)
    refused: list[ImportedRefold] = field(default_factory=list)
    unmatched: list[str] = field(default_factory=list)
    #: Further folds of a design that was imported from another file. A tool
    #: that returns five models per job leaves four of them here.
    superseded: list[str] = field(default_factory=list)
    #: True when the walk hit :py:data:`SCAN_LIMIT`, so ``source`` was only
    #: partly read and a design may be missing for that reason alone.
    scan_truncated: bool = False

    def as_dict(self) -> dict[str, object]:
        """A record shaped for a run log: counts, and every reason by name.

        ``superseded`` and ``scan_truncated`` appear only when they happened.
        An import of one fold per design is the ordinary case, and a key that
        is always empty there is noise in every record that does not need it.
        """
        record: dict[str, object] = {
            "imported": len(self.placed),
            "refused": [{"example_id": item.example_id, "blocker": item.blocker} for item in self.refused],
            "unmatched_files": sorted(self.unmatched),
        }
        if self.superseded:
            record["superseded_files"] = sorted(self.superseded)
        if self.scan_truncated:
            record["scan_truncated"] = True
        return record


def place_refold(
    oracle: OracleMetric,
    out_dir: Path,
    example_id: str,
    structure: Path,
    *,
    save_id: str | None = None,
    confidence: Path | None = None,
) -> ImportedRefold:
    """Put one externally folded structure where ``oracle`` would have put it.

    The structure is copied rather than moved or linked: the caller's file stays
    theirs, a later re-run cannot be changed by editing it, and a symlink into a
    directory the run later cleans would take the original with it.

    It keeps its own extension. The refold is looked up by stem through
    :py:func:`find_structure_file`, so a plain ``.cif`` is found even when the
    oracle writes ``.cif.zst`` for folds of its own -- converting it here would
    mean re-encoding a file to no purpose.

    Returns a record. A refusal never writes anything, because a half-placed
    refold is worse than none: the confidence JSON is what marks a design done,
    so writing it beside a structure that cannot be read would make the design
    look refolded and measure nothing.
    """
    if not structure.is_file():
        return ImportedRefold(example_id, "structure_not_found", save_id=save_id)
    if not is_structure_file(structure, include_pdb=True):
        return ImportedRefold(example_id, "not_a_structure_file", save_id=save_id)
    try:
        if structure.stat().st_size == 0:
            return ImportedRefold(example_id, "structure_empty", save_id=save_id)
    except OSError:
        return ImportedRefold(example_id, "structure_unreadable", save_id=save_id)

    payload = _confidence_payload(oracle, confidence)
    if isinstance(payload, str):
        return ImportedRefold(example_id, payload, save_id=save_id)

    try:
        directory = design_output_dir(out_dir, example_id)
    except ValueError:
        # design_output_dir refuses an id that would write outside the run.
        return ImportedRefold(example_id, "example_id_not_a_directory_name", save_id=save_id)

    nominal = oracle.refolded_cif_path(out_dir, example_id, save_id)
    destination = nominal.parent / f"{structure_file_stem(nominal)}{_suffix(structure)}"
    confidence_destination = oracle.confidence_path(out_dir, example_id, save_id)
    try:
        directory.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(structure, destination)
        # Written last. It is the marker the oracle checks, so until it exists
        # the design is not yet claimed as refolded.
        confidence_destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except OSError:
        return ImportedRefold(example_id, "write_failed", save_id=save_id)

    if not oracle.already_refolded(out_dir, example_id, save_id):
        # The oracle still does not recognise it. AF2, for one, wants its own
        # model name in the record. Said rather than left to be discovered when
        # the metrics come back empty.
        return ImportedRefold(example_id, "oracle_did_not_accept", save_id=save_id)
    return ImportedRefold(
        example_id,
        None,
        structure=destination,
        confidence=confidence_destination,
        save_id=save_id,
    )


def import_refolds(
    oracle: OracleMetric,
    out_dir: Path,
    source: Path,
    *,
    example_ids: list[str] | None = None,
    save_id: str | None = None,
    recursive: bool = True,
    bases: dict[str, Path] | None = None,
) -> ImportReport:
    """Import every fold in ``source`` that names a design under ``out_dir``.

    A file is matched to a design when its stem **contains** that design's id
    at a token boundary. Containment rather than a prefix because tools put
    their own word first: the AlphaFold Server hands back
    ``fold_binder_0_model_0.cif``, which no prefix rule would match. The
    boundary is what keeps that permissive, not loose -- see
    :py:func:`_match_example_id`.

    ``source`` is walked into subdirectories by default, because a download is
    usually an archive that unpacks to a directory per job. Pass
    ``recursive=False`` for one flat directory.

    When ``example_ids`` is given, only those are considered and anything else
    is reported unmatched; without it, the design directories already under
    ``out_dir`` are the candidates. Either way a file that matches nothing is
    **named**, not skipped quietly -- an import that silently did nothing is
    the failure this is most likely to have.

    Where several files match one design -- a tool that returns five models per
    job -- one is chosen by :py:func:`_best_fold` and the rest are reported as
    ``superseded``. Importing them in turn would leave whichever sorted last in
    place, which for a ranked set is the worst model of the five.

    ``bases`` says where each design's files go, for a run that does not hold
    every design directly under ``out_dir``. A design absent from it falls back
    to ``out_dir``.

    A confidence JSON beside the structure is used for that design; the names
    looked for are in :py:func:`_confidence_beside`.
    """
    report = ImportReport()
    if not source.is_dir():
        report.unmatched.append(str(source))
        return report

    wanted = example_ids if example_ids is not None else _designs_under(out_dir)
    paths, report.scan_truncated = _structure_files(source, out_dir, recursive)

    candidates: dict[str, list[Path]] = {}
    for path in paths:
        example_id = _match_example_id(structure_file_stem(path), wanted)
        if example_id is None:
            report.unmatched.append(_named(path, source))
            continue
        candidates.setdefault(example_id, []).append(path)

    for example_id in sorted(candidates):
        chosen, rest = _best_fold(candidates[example_id])
        report.superseded.extend(_named(path, source) for path in rest)
        found = place_refold(
            oracle,
            bases.get(example_id, out_dir) if bases else out_dir,
            example_id,
            chosen,
            save_id=save_id,
            confidence=_confidence_beside(chosen),
        )
        (report.placed if found.placed else report.refused).append(found)
    return report


def _structure_files(source: Path, out_dir: Path, recursive: bool) -> tuple[list[Path], bool]:
    """Structure files under ``source``, and whether the walk was cut short.

    Anything inside ``out_dir`` is skipped. A caller who points ``source`` at
    the run itself would otherwise re-import refolds this module already
    placed: those are named for their design, so they match it, and one would
    supersede the fold it was made from.
    """
    found: list[Path] = []
    examined = 0
    try:
        resolved_out = out_dir.resolve()
    except OSError:
        resolved_out = out_dir
    try:
        for path in source.rglob("*") if recursive else source.iterdir():
            examined += 1
            if examined > SCAN_LIMIT:
                return sorted(found), True
            if not path.is_file() or not is_structure_file(path, include_pdb=True):
                continue
            if _inside(path, resolved_out):
                continue
            found.append(path)
    except OSError:
        # A directory that cannot be read yields what was found before it. The
        # designs that are then missing are reported by their absence from
        # `placed`, which is the same signal as a file that never arrived.
        pass
    return sorted(found), False


def _inside(path: Path, directory: Path) -> bool:
    try:
        return directory == path.resolve().parent or directory in path.resolve().parents
    except OSError:
        return False


def _named(path: Path, source: Path) -> str:
    """How a file is reported: its path under ``source``, in posix spelling."""
    try:
        return path.relative_to(source).as_posix()
    except ValueError:
        return path.name


def _best_fold(paths: list[Path]) -> tuple[Path, list[Path]]:
    """The fold to import out of several for one design, and the rest.

    Ranked lowest-first. Both AF3 and the AlphaFold Server write the
    best-scoring model as rank 0 and number the rest downwards in quality, so
    the lowest rank in a filename is the one to keep. Files carrying no rank
    sort after those that do, and the name breaks a tie, so the choice is the
    same on every machine and on every re-run.
    """
    ordered = sorted(paths, key=_fold_order)
    return ordered[0], ordered[1:]


def _fold_order(path: Path) -> tuple[bool, int, str]:
    rank = _rank_in_name(structure_file_stem(path))
    return (rank is None, rank if rank is not None else 0, path.name)


def _rank_in_name(stem: str) -> int | None:
    """The rank a folding tool put in this filename, if it put one there."""
    found = _RANK_IN_NAME.findall(stem)
    return int(found[-1]) if found else None


def _suffix(path: Path) -> str:
    """The structure's full extension, compound ones included (``.cif.zst``)."""
    return path.name[len(structure_file_stem(path)) :]


def _confidence_payload(oracle: OracleMetric, confidence: Path | None) -> dict[str, object] | str:
    """The record to write, or a blocker name when the supplied file is unusable."""
    base: dict[str, object] = {IMPORT_MARK: True, "oracle_name": oracle.oracle_name}
    if confidence is None:
        base["confidence"] = NO_CONFIDENCE
        return base
    try:
        if confidence.stat().st_size > CONFIDENCE_BYTE_LIMIT:
            return "confidence_too_large"
        loaded = json.loads(confidence.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError, RecursionError):
        return "confidence_unreadable"
    if not isinstance(loaded, dict):
        return "confidence_not_an_object"
    # The supplied values win, and the mark stays: a reader can tell which
    # numbers came from outside even when they are real confidence values.
    return {**loaded, **base}


def _confidence_beside(structure: Path) -> Path | None:
    """A confidence JSON named for this structure, if its author left one.

    Four spellings, because the file is rarely named after the structure. AF3
    writes ``<job>_model.cif`` beside ``<job>_summary_confidences.json``, and
    the AlphaFold Server numbers both halves of that pair. Deriving the partner
    by substituting the one segment that differs finds it under either, and
    missing it is not a loud failure -- the fold imports with its confidence
    recorded as absent, and the criteria that needed those numbers then report
    bounds they could not check.
    """
    stem = structure_file_stem(structure)
    names = [f"{stem}_confidences.json", f"{stem}.json"]
    for pattern, replacement in (
        (_MODEL_RANKED, r"_summary_confidences_\1"),
        (_MODEL_PLAIN, "_summary_confidences"),
    ):
        swapped, count = pattern.subn(replacement, stem)
        if count:
            names.append(f"{swapped}.json")
    for candidate in names:
        path = structure.parent / candidate
        if path.is_file():
            return path
    return None


def _designs_under(out_dir: Path) -> list[str]:
    """Design ids that already have an output directory."""
    try:
        return sorted(path.name for path in out_dir.iterdir() if path.is_dir())
    except OSError:
        return []


def _match_example_id(stem: str, wanted: list[str]) -> str | None:
    """The design this filename names, when exactly one design fits.

    External tools decorate a name on both sides: ``binder_0_model_1``,
    ``binder_0_rank0``, and the AlphaFold Server's ``fold_binder_0_model_0``.
    So the id is looked for anywhere in the stem rather than only at the front.

    The longest candidate wins, and an ambiguous stem matches nothing rather
    than the first one tried.
    """
    hits = [name for name in wanted if _holds_id(stem, name)]
    if not hits:
        return None
    longest = max(len(name) for name in hits)
    best = [name for name in hits if len(name) == longest]
    return best[0] if len(best) == 1 else None


def _holds_id(stem: str, name: str) -> bool:
    """Whether ``stem`` contains ``name`` bounded by non-alphanumerics.

    The boundary is what makes containment safe. ``binder_1`` does occur inside
    ``binder_10``, so a bare substring test would let one design claim another
    design's fold -- the mistake that is both the easiest to make here and the
    hardest to notice, since the file is real and the numbers are plausible.
    Requiring a non-alphanumeric character (or the end of the name) on each
    side rejects that: what follows ``binder_1`` in ``binder_10`` is a digit.
    """
    if not name:
        return False
    start = 0
    while True:
        index = stem.find(name, start)
        if index < 0:
            return False
        before = stem[index - 1] if index else ""
        after = stem[index + len(name) : index + len(name) + 1]
        if not before.isalnum() and not after.isalnum():
            return True
        start = index + 1
