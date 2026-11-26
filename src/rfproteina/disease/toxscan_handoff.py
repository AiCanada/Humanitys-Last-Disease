"""Prepare a local author-platform request and attach its verified, operator-supplied export."""

from __future__ import annotations

import json
import os
import stat
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from rfproteina.disease import toxscan, toxscan_csv
from rfproteina.disease import toxscan_inputs as inputs


def prepare_request(directory: Path, candidate: str) -> dict:
    smiles = inputs.candidate_structure(directory, candidate)
    return {
        "candidate_id": candidate, "canonical_smiles": smiles,
        "structure_sha256": inputs.sha256(smiles.encode("ascii")),
        "source": inputs.SOURCE,
        "notice": "Submitting the SMILES on the ToxScan website shares this molecular structure with its operator. "
                  "Copy the SMILES if you wish to submit it, then download the results CSV.",
    }


@contextmanager
def _writer_lock(directory: Path) -> Iterator[None]:
    # Keep a stable lock inode. Closing the descriptor releases the lock after a crash, and opening it never
    # truncates existing content. Checking the opened object also protects platforms without O_NOFOLLOW.
    path = directory / ".toxscan-import.lock"
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or not os.path.samestat(info, os.fstat(descriptor)):
            raise inputs.ToxScanError("toxscan_import_conflict")
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(descriptor, msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            deadline = time.monotonic() + 10
            while True:
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(0.05)
        yield
    finally:
        os.close(descriptor)


def _pending_files(directory: Path, manifest: dict, data: bytes) -> dict[str, bytes]:
    pending = {}
    try:
        existing = inputs.read_export_file(directory / inputs.SOURCE_FILE)
    except inputs.ToxScanError as error:
        if str(error) != "toxscan_export_missing":
            raise inputs.ToxScanError("toxscan_import_conflict") from error
        pending[inputs.SOURCE_FILE] = data
    else:
        if existing != data:
            raise inputs.ToxScanError("toxscan_import_conflict")
    try:
        recorded = inputs.read_manifest(directory)
    except inputs.ToxScanError as error:
        if str(error) != "toxscan_manifest_missing":
            raise inputs.ToxScanError("toxscan_import_conflict") from error
        pending[inputs.MANIFEST] = (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
    else:
        expected = manifest.copy()
        if recorded["version"] == 1:
            expected["version"] = 1
            del expected["structure_sha256"]
        if recorded != expected:
            raise inputs.ToxScanError("toxscan_import_conflict")
    return pending


@contextmanager
def _publish(directory: Path, pending: dict[str, bytes]) -> Iterator[None]:
    # Publish complete files without replacing a prior file. The manifest is the commit point: until it exists,
    # normal readers cannot report a measurement. A process killed before it commits can retry the same CSV.
    if not pending:
        yield
        return
    with tempfile.TemporaryDirectory(dir=directory, prefix=".toxscan-import-") as staging_dir:
        staged = Path(staging_dir)
        published = []
        try:
            for name, payload in pending.items():
                with (staged / name).open("xb") as stream:
                    (staged / name).chmod(0o600)
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
            for name in pending:
                os.link(staged / name, directory / name)
                published.append(name)
            yield
        except BaseException:
            # Remove only links created by this transaction, including on interrupted writes. A manual replacement
            # that raced the importer belongs to its author. Never remove it or roll back a pre-existing file.
            for name in reversed(published):
                target = directory / name
                try:
                    if os.path.samestat(target.lstat(), (staged / name).stat()):
                        target.unlink()
                except FileNotFoundError:
                    continue
            raise


def _verified_result(directory: Path, manifest: dict) -> dict:
    report = toxscan.screen_run(directory, manifest["candidate_id"])
    if report["status"] != "complete":
        raise inputs.ToxScanError(report["blocker"])
    if report["structure_sha256"] != manifest["structure_sha256"]:
        raise inputs.ToxScanError("toxscan_nomination_changed")
    if report["source_sha256"] != manifest["export_sha256"]:
        raise inputs.ToxScanError("toxscan_export_digest_mismatch")
    return report


def import_export(
    directory: Path, candidate: str, data: bytes, *, expected_structure_sha256: str | None = None,
) -> dict:
    if expected_structure_sha256 is not None and (
        not isinstance(expected_structure_sha256, str) or not inputs.SHA256.fullmatch(expected_structure_sha256)
    ):
        raise inputs.ToxScanError("toxscan_invalid_request_digest")
    if not isinstance(data, bytes) or not 0 < len(data) <= inputs.MAX_EXPORT_BYTES:
        raise inputs.ToxScanError("toxscan_invalid_export")
    structure = inputs.candidate_structure(directory, candidate)
    structure_digest = inputs.sha256(structure.encode("ascii"))
    if expected_structure_sha256 is not None and expected_structure_sha256 != structure_digest:
        raise inputs.ToxScanError("toxscan_nomination_changed")
    toxscan_csv.read_export(data, structure)
    digest = inputs.sha256(data)
    manifest = {"version": 2, "candidate_id": candidate, "source": inputs.SOURCE,
                "export_sha256": digest, "structure_sha256": structure_digest}
    # An identical complete receipt can also be verified in a read-only archive.
    if not _pending_files(directory, manifest, data):
        return _verified_result(directory, manifest)
    try:
        with _writer_lock(directory):
            if inputs.candidate_structure(directory, candidate) != structure:
                raise inputs.ToxScanError("toxscan_nomination_changed")
            pending = _pending_files(directory, manifest, data)
            with _publish(directory, pending):
                return _verified_result(directory, manifest)
    except FileExistsError as error:
        raise inputs.ToxScanError("toxscan_import_conflict") from error
    except OSError as error:
        raise inputs.ToxScanError("toxscan_import_write_failed") from error
