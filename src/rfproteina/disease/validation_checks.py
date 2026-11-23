"""Plan and run read-only checks of the files produced by registered local runs."""

from __future__ import annotations

import hashlib
import json
import re
import stat
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from rfproteina.disease.screen import (
    _DESIGN_SUFFIXES,
    _ZSTD_SUFFIX,
    _design_stem,
    _design_suffix,
    _design_text,
    _metrics_path,
    _metrics_table,
    _sample_id,
    _sample_ids,
    _samples_root,
    _zstandard,
)
from rfproteina.disease.trace import VALIDATION_FILE
from rfproteina.utils.metrics_files import ARCHIVE, PENDING, RECEIPT, file_sha256, job_tables

SYNTHESIS_REVIEW_RECEIPT = "synthesis_review.json"
_VALIDATION_FILE_LIMIT = 16 * 1024 * 1024


@dataclass(frozen=True)
class ValidationCheck:
    """A named check and the description shown to the operator."""

    key: str
    label: str
    description: str


@dataclass(frozen=True)
class ValidationChoice(ValidationCheck):
    """A check's installation and recommendation for the current run selection."""

    installed: bool
    recommended: bool
    reason: str
    workload: str


@dataclass(frozen=True)
class ValidationCandidate:
    """Identity and files for one candidate within a run."""

    key: str
    label: str
    example_id: str | None
    files: tuple[Path, ...]


CHECKS = (
    ValidationCheck("structure_files", "Structure file readability", "Check generated plain and gzip CIF files."),
    ValidationCheck("metrics_table", "Metrics table integrity", "Check CSV layout and aggregation provenance."),
    ValidationCheck("zstd_files", "Zstandard structure readability", "Check generated Zstandard-compressed CIF files."),
    ValidationCheck(
        "experimental_record", "Therapeutic experimental evidence", "Check the layout of attached bench records."
    ),
    ValidationCheck(
        "synthesis_receipt", "DNA-synthesis review receipt", "Check whether a separate review receipt is attached."
    ),
    ValidationCheck(
        "checksums", "File checksums", "Calculate full SHA-256 digests of generated structures and metrics."
    ),
)


@dataclass(frozen=True)
class _RunFiles:
    directory: Path
    structures: tuple[Path, ...]
    structure_blocker: str | None
    metrics: Path | None
    metric_inputs: tuple[Path, ...]
    metrics_blocker: str | None


def _local(path: Path, root: Path) -> bool:
    """Keep nested file references inside the registered run directory."""
    try:
        return path.resolve().is_relative_to(root)
    except (OSError, RuntimeError):
        return False


def _structure_files(root: Path) -> tuple[tuple[Path, ...], str | None]:
    samples = _samples_root(root)
    if not _local(samples, root):
        return (), "The structure directory points outside the run."
    files = []
    for directory, subdirs, names in samples.walk():
        subdirs[:] = [name for name in subdirs if _local(directory / name, root)]
        for name in names:
            path = directory / name
            if name.endswith(_DESIGN_SUFFIXES) and _local(path, root) and path.is_file():
                files.append(path)
    return tuple(sorted(files)), None


def _metrics_files(root: Path) -> tuple[Path | None, tuple[Path, ...], str | None]:
    metrics_dirs = (root / "inference_metrics", root / "evaluation_metrics")
    artifacts = (*metrics_dirs, *(folder / ARCHIVE for folder in metrics_dirs))
    if any(not _local(path, root) for path in artifacts):
        return None, (), "A metrics directory points outside the run."

    metrics = _metrics_path(root)
    publication_files = [
        path
        for folder in metrics_dirs
        for path in (folder / PENDING, folder / RECEIPT, *job_tables(folder))
    ]
    if metrics is not None:
        publication_files.append(metrics)
    if any(not _local(path, root) for path in publication_files):
        return None, (), "A metrics file points outside the run."
    if any(path.exists() and not path.is_file() for path in publication_files):
        return None, (), "A metrics artifact is not a regular file."
    return metrics, tuple(path for path in dict.fromkeys(publication_files) if path.is_file()), None


def _inventory(directory: Path) -> _RunFiles:
    root = directory.resolve()
    try:
        structures, structure_blocker = _structure_files(root)
    except OSError:
        structures, structure_blocker = (), "The structure directory could not be inspected."
    try:
        metrics, metric_inputs, metrics_blocker = _metrics_files(root)
    except OSError:
        metrics, metric_inputs, metrics_blocker = None, (), "The metrics directory could not be inspected."
    return _RunFiles(root, structures, structure_blocker, metrics, metric_inputs, metrics_blocker)


def _inventories(run_dirs: Sequence[Path]) -> tuple[_RunFiles, ...]:
    """Inspect names and metadata only, and keep a failed directory visible."""
    runs = []
    seen = set()
    for directory in run_dirs:
        try:
            item = _inventory(directory)
        except (OSError, RuntimeError):
            reason = "The run files could not be inspected."
            item = _RunFiles(directory, (), reason, None, (), reason)
        if item.directory not in seen:
            seen.add(item.directory)
            runs.append(item)
    return tuple(runs)


def _candidates(run: _RunFiles) -> tuple[ValidationCandidate, ...]:
    groups: dict[tuple[Path, str], list[Path]] = {}
    for path in run.structures:
        identity = (path.parent.relative_to(run.directory), _design_stem(path.name))
        groups.setdefault(identity, []).append(path)
    ordered = sorted(groups.items(), key=lambda item: (str(item[0][0]), item[0][1]))
    paths = [max(files, key=lambda p: _DESIGN_SUFFIXES.index(_design_suffix(p))) for _, files in ordered]
    example_ids = _sample_ids(run.directory, paths)
    short_ids = [_sample_id(path) for path in paths]
    counts = Counter(short_ids)
    # A corrupt or unsupported compressed file may have been excluded when
    # the screen assigned its IDs. In that case a collision cannot be joined
    # by a guessed ID, while unique short IDs still identify their own file.
    plain_only = all(path.name.endswith("_generated.cif") for path in paths)
    candidates = []
    identified = zip(ordered, example_ids, short_ids, strict=True)
    for index, (((parent, stem), files), name, short_id) in enumerate(identified, 1):
        key = hashlib.sha256(f"{run.directory.as_posix()}\0{parent.as_posix()}\0{stem}".encode()).hexdigest()[:24]
        example_id = name if plain_only or (short_id == name and counts[short_id] == 1) else None
        label = f"Candidate {index}"
        if short_id and re.fullmatch(r"binder_\d+", short_id):
            label = f"{short_id} ({label.lower()})"
        candidates.append(ValidationCandidate(key, label, example_id, tuple(files)))
    return tuple(candidates)


def _select_candidate(candidates: Sequence[ValidationCandidate], key: str) -> ValidationCandidate:
    for candidate in candidates:
        if candidate.key == key:
            return candidate
    raise ValueError("Choose an available candidate from the list.")


def _checksum_files(run: _RunFiles) -> tuple[Path, ...]:
    return run.structures if run.metrics is None else (*run.structures, run.metrics)


def _workload(files: Sequence[Path], sizes: dict[Path, int | None]) -> str:
    if not files:
        return ""
    for path in files:
        if path not in sizes:
            try:
                sizes[path] = path.stat().st_size
            except OSError:
                sizes[path] = None
    count = f"{len(files)} {'file' if len(files) == 1 else 'files'}"
    if any(sizes[path] is None for path in files):
        return f"{count}, total size unavailable."
    return f"{count}, {sum(sizes[path] or 0 for path in files):,} bytes on disk."


def _plan(
    runs: Sequence[_RunFiles], *, candidate: ValidationCandidate | None = None, show_workload: bool = False
) -> tuple[ValidationChoice, ...]:
    zstd_installed = _zstandard() is not None
    standard = [p for run in runs for p in run.structures if not p.name.endswith(_ZSTD_SUFFIX)]
    zstd = [p for run in runs for p in run.structures if p.name.endswith(_ZSTD_SUFFIX)]
    metrics = [p for run in runs for p in run.metric_inputs]
    benches = [p for run in runs if (p := _attached_record(run.directory, VALIDATION_FILE)[0]) is not None]
    standard_reason = "Plain or gzip CIF files found." if standard else "No plain or gzip CIF files found."
    metrics_reason = "Metrics artifacts found." if metrics else "No metrics artifacts found."
    zstd_reason = "Zstandard CIF files found." if zstd else "No Zstandard CIF files found."
    if not zstd_installed:
        zstd_reason = "Zstandard decoder is unavailable."
    experiment_recommended = bool(benches)
    experiment_reason = "Bench record found." if benches else "No bench record attached."
    if candidate is not None and benches:
        record, detail = _read_bench_record(runs[0].directory)
        if record is None:
            experiment_recommended, experiment_reason = False, detail
        else:
            experiment_recommended, experiment_reason = _candidate_bench_status(record, candidate)
    if candidate is not None and metrics:
        metrics_reason = "Optional run-level check of the shared metrics table."
    if candidate is not None and standard and zstd_installed and zstd:
        zstd_reason = "Optional compressed copy of this candidate's structure."
    zstd_recommended = zstd_installed and bool(zstd) and (candidate is None or not standard)
    status = {
        "structure_files": (True, bool(standard), standard_reason),
        "metrics_table": (True, bool(metrics) and candidate is None, metrics_reason),
        "zstd_files": (zstd_installed, zstd_recommended, zstd_reason),
        "experimental_record": (True, experiment_recommended, experiment_reason),
        "synthesis_receipt": (True, False, "Optional administrative check."),
        "checksums": (True, False, "Optional full file read for a reproducibility record."),
    }
    workload = {}
    if show_workload:
        inputs = {
            "structure_files": standard,
            "metrics_table": metrics,
            "zstd_files": zstd,
            "experimental_record": benches,
            "checksums": [p for run in runs for p in _checksum_files(run)],
        }
        sizes: dict[Path, int | None] = {}
        workload = {key: _workload(files, sizes) for key, files in inputs.items()}
        workload["synthesis_receipt"] = "Metadata only."
    return tuple(
        ValidationChoice(check.key, check.label, check.description, *status[check.key], workload.get(check.key, ""))
        for check in CHECKS
    )


def plan_validation_checks(run_dirs: Sequence[Path]) -> tuple[ValidationChoice, ...]:
    """Report which checks are installed and which match the available files."""
    return _plan(_inventories(run_dirs), show_workload=True)


def candidate_validation_plan(
    run_dirs: Sequence[Path], candidate_key: str | None
) -> tuple[tuple[ValidationCandidate, ...], tuple[ValidationChoice, ...]]:
    """Prepare the candidate picker and file-based recommendations together."""
    runs = _inventories(run_dirs)
    candidates = _candidates(runs[0]) if len(run_dirs) == len(runs) == 1 else ()
    if candidate_key is None:
        return candidates, _plan(runs, show_workload=True)
    try:
        candidate = _select_candidate(candidates, candidate_key)
    except ValueError:
        return candidates, ()
    scoped = replace(runs[0], structures=candidate.files)
    return candidates, _plan((scoped,), candidate=candidate, show_workload=True)


def _readability(run: _RunFiles, *, zstd: bool, name_files: bool = True) -> str:
    if run.structure_blocker is not None:
        return run.structure_blocker
    files = [path for path in run.structures if path.name.endswith(_ZSTD_SUFFIX) == zstd]
    if not files:
        return "No matching structure files found."
    failures = []
    for path in files:
        data = _design_text(path)
        if not data or not data.strip():
            failures.append(path.relative_to(run.directory).as_posix())
    result = f"{len(files) - len(failures)} of {len(files)} files readable as UTF-8."
    if failures and name_files:
        result += " Unreadable or empty files include " + ", ".join(failures) + "."
    return result


def _metrics(run: _RunFiles) -> str:
    if run.metrics_blocker is not None:
        return run.metrics_blocker
    if not run.metric_inputs:
        return "No metrics artifacts found."
    rows, blocker = _metrics_table(run.directory, run.metrics)
    if blocker is not None:
        return blocker.replace("_", " ").capitalize() + "."
    return f"Validated {len(rows)} metrics rows."


def _attached_record(directory: Path, name: str) -> tuple[Path | None, str]:
    path = directory / name
    if not _local(path, directory):
        return None, "The attachment points outside the run."
    try:
        if path.is_file():
            return path, "Attachment present."
        if path.exists():
            return None, "The attachment must be a regular file."
    except OSError:
        return None, "The attachment could not be inspected."
    return None, "No attachment found."


def _unique_record(pairs: list[tuple[str, object]]) -> dict[str, object]:
    fields = dict(pairs)
    if len(fields) != len(pairs):
        raise ValueError("Duplicate record keys.")
    return fields


def _has_status(record: dict[str, object]) -> bool:
    status = record.get("status")
    return isinstance(status, str) and bool(status.strip())


def _read_bench_record(directory: Path) -> tuple[dict[str, object] | None, str]:
    path, detail = _attached_record(directory, VALIDATION_FILE)
    if path is None:
        return None, detail
    try:
        with path.open("rb") as handle:
            payload = handle.read(_VALIDATION_FILE_LIMIT + 1)
        if len(payload) > _VALIDATION_FILE_LIMIT:
            return None, "The bench record exceeds 16 MiB."
        record = json.loads(payload.decode("utf-8-sig"), object_pairs_hook=_unique_record)
    except (OSError, ValueError, RecursionError):
        return None, "The bench record could not be read as JSON with unique keys."
    if not isinstance(record, dict) or not record:
        return None, "The bench record needs a run-level status or sample records with a status."
    return record, ""


def _candidate_bench_status(record: dict[str, object], candidate: ValidationCandidate) -> tuple[bool, str]:
    entry = record.get(candidate.example_id) if candidate.example_id else None
    if isinstance(entry, dict):
        if _has_status(entry):
            return True, "The selected candidate's bench record contains a status."
        return True, "The selected candidate has a bench record with a missing status."
    if _has_status(record):
        return True, "A run-level bench record contains a status."
    if candidate.example_id is None:
        return False, "The candidate could not be matched to a bench record by ID."
    return False, "No bench record matches the selected candidate."


def _experimental_record(run: _RunFiles, candidate: ValidationCandidate | None = None) -> str:
    record, detail = _read_bench_record(run.directory)
    if record is None:
        return detail
    if candidate is not None:
        return _candidate_bench_status(record, candidate)[1]
    if _has_status(record):
        return "The run-level bench record contains a status."
    complete = sum(isinstance(entry, dict) and _has_status(entry) for entry in record.values())
    return f"{complete} of {len(record)} sample records contain a status."


def synthesis_receipt_attached(directory: Path) -> bool:
    """Check for a regular receipt file using its directory entry only."""
    try:
        return stat.S_ISREG((directory / SYNTHESIS_REVIEW_RECEIPT).lstat().st_mode)
    except OSError:
        return False


def _synthesis_receipt(run: _RunFiles) -> str:
    return "Attachment present." if synthesis_receipt_attached(run.directory) else "No attachment found."


def _checksums(run: _RunFiles, *, name_files: bool = True) -> str:
    digests = []
    for index, path in enumerate(_checksum_files(run), 1):
        try:
            digest = file_sha256(path)
        except OSError:
            digest = "Unreadable"
        label = path.relative_to(run.directory).as_posix() if name_files else f"File {index}"
        digests.append(f"{label} {digest}")
    for blocker in (run.structure_blocker, run.metrics_blocker):
        if blocker is not None and blocker not in digests:
            digests.append(blocker)
    return "\n".join(digests) if digests else "No generated structures or metrics found."


def _limit_validation_reads(runs: Sequence[_RunFiles], selected: set[str], byte_limit: int) -> None:
    """Check the on-disk workload of a remote request before opening its inputs."""
    files: set[Path] = set()
    for run in runs:
        if "structure_files" in selected:
            files.update(path for path in run.structures if not path.name.endswith(_ZSTD_SUFFIX))
        if "zstd_files" in selected:
            files.update(path for path in run.structures if path.name.endswith(_ZSTD_SUFFIX))
        if "checksums" in selected:
            files.update(_checksum_files(run))
        if "metrics_table" in selected:
            files.update(run.metric_inputs)
        if "experimental_record" in selected:
            record, _ = _attached_record(run.directory, VALIDATION_FILE)
            if record is not None:
                files.add(record)
    if len(files) > 512:
        raise ValueError("This validation request covers too many files. Select one candidate or fewer checks.")
    try:
        total = sum(path.stat().st_size for path in files)
    except OSError:
        raise ValueError("A validation input changed before it could be inspected. Refresh the plan.") from None
    if total > byte_limit:
        raise ValueError("This validation request exceeds the read budget. Select one candidate or fewer checks.")


def run_validation_checks(
    run_dirs: Sequence[Path],
    keys: Sequence[str],
    *,
    candidate_key: str | None = None,
    name_files: bool = True,
    read_budget: int | None = None,
) -> dict[str, str]:
    """Validate the selection before reading files and return results in registry order."""
    selected = set(keys)
    unknown = sorted(selected.difference(check.key for check in CHECKS))
    if unknown:
        raise ValueError(f"Unknown validation checks were selected ({', '.join(unknown)}).")
    if not selected:
        return {}

    runs = _inventories(run_dirs)
    candidate = None
    if candidate_key is not None:
        if len(runs) != 1:
            raise ValueError("Choose one finished run to validate a candidate.")
        candidate = _select_candidate(_candidates(runs[0]), candidate_key)
        runs = (replace(runs[0], structures=candidate.files),)
    choices = _plan(runs)
    unavailable = [choice.label for choice in choices if choice.key in selected and not choice.installed]
    if unavailable:
        raise ValueError(f"Unavailable validation checks were selected ({', '.join(unavailable)}).")
    if read_budget is not None:
        _limit_validation_reads(runs, selected, read_budget)
    readers = {
        "structure_files": lambda run: _readability(run, zstd=False, name_files=name_files),
        "metrics_table": _metrics,
        "zstd_files": lambda run: _readability(run, zstd=True, name_files=name_files),
        "experimental_record": lambda run: _experimental_record(run, candidate),
        "synthesis_receipt": _synthesis_receipt,
        "checksums": lambda run: _checksums(run, name_files=name_files),
    }
    return {
        key: "\n".join(
            f"{(run.directory.name if name_files else 'Run') or 'Run'} — {readers[key](run)}" for run in runs
        )
        or "No registered runs selected."
        for key in readers
        if key in selected
    }
