"""Keep a copy of every finished generation for later search or training.

The live ``rfd4-generate`` directory is left where the run wrote it. This
module adds ``generations/<disease name>/<target>/`` under the output
directory and copies the finished designs there. The folder is created when
it is missing. Nothing already stored is overwritten.
"""

from __future__ import annotations

import hashlib
import re
import shutil
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from rfproteina.disease.publish import scrub_for_upload, scrubbed_json

#: Directory name under the output directory. The main page shows this folder.
GENERATION_ARCHIVE = "generations"
#: Shape of ``generation.json``. Raised when a key is added or changes meaning,
#: so a reader can tell which record it is holding instead of inferring it from
#: which keys happen to be present.
RECORD_VERSION = 1
#: Text formats rewritten through ``scrub_for_upload`` as they are archived.
#: Coordinates are left alone: a CIF carries no path and is large.
_SCRUBBED_SUFFIXES = frozenset({".csv", ".json", ".yaml", ".yml", ".txt", ".md"})
#: Bytes a file may have before it is archived unread. Scrubbing holds the file
#: in memory, and a metrics table is kilobytes.
_SCRUB_LIMIT = 8_388_608
#: Coordinates are not scrubbed whole -- they carry no path and they are large
#: -- but the writer stamps the account it ran under into this one item, and an
#: account name is personal data wherever it sits. The value is replaced with
#: the mmCIF null rather than the file being rewritten. A loop header of the
#: same name has no value on its line and does not match.
_AUTHOR_ITEM = re.compile(r"^(_entry\.author[ \t]+)(\S.*?)[ \t]*$", re.MULTILINE)
_AUTHOR_SUFFIXES = frozenset({".cif"})

_ILLEGAL = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_RESERVED = frozenset(
    {
        "con",
        "prn",
        "aux",
        "nul",
        *(f"com{number}" for number in range(1, 10)),
        *(f"lpt{number}" for number in range(1, 10)),
    }
)


def generation_archive_root(out_dir: str | Path) -> Path:
    """The folder that holds every saved generation for this output directory.

    When the output directory is already that folder, it is used as itself so
    a click on it in the file list does not nest another copy inside.
    """
    path = Path(out_dir)
    if path.name == GENERATION_ARCHIVE:
        return path
    return path / GENERATION_ARCHIVE


def archive_segment(text: str, fallback: str) -> str:
    """One path component. Separators and reserved names cannot leave the archive."""
    cleaned = _ILLEGAL.sub("-", text.strip()).strip(" .-")
    if not cleaned or cleaned in {".", ".."} or cleaned.casefold() in _RESERVED:
        return fallback
    return cleaned[:80]


def save_generation(
    out_dir: Path,
    output_dir: Path | None,
    *,
    disease_name: str,
    disease_id: str,
    target_id: str,
    group: str,
    entry_id: str,
    chain: str,
    binder_kind: str,
) -> Path:
    """Copy one finished generation under its disease name and target.

    ``output_dir`` is the directory ``rfd4-generate`` wrote. The copy is a new
    folder. A later run of the same target is saved beside it. The source
    files stay in place.
    """
    disease_folder = archive_segment(disease_name, "disease")
    target_folder = archive_segment(target_id or f"{entry_id}_{chain}", "target")
    source_stem = f"{group}_{entry_id}_{chain}"
    folder_name = archive_segment(source_stem, "generation")
    parent = generation_archive_root(out_dir) / disease_folder / target_folder
    parent.mkdir(parents=True, exist_ok=True)
    dest = _unique_directory(parent, folder_name)
    dest.mkdir()
    if output_dir is not None and output_dir.is_dir() and not _contains(output_dir, dest):
        _copy_children(output_dir, dest)
    _copy_inputs(Path(out_dir), source_stem, dest)
    scrubbed = _scrub_archived_text(dest)
    saved_at = datetime.now(UTC).isoformat()
    record = {
        "record_version": RECORD_VERSION,
        "run_id": _run_id((disease_id, target_id, group, entry_id, chain, binder_kind, saved_at)),
        "disease_name": disease_name,
        "disease_id": disease_id,
        "target_id": target_id,
        "group": group,
        "entry_id": entry_id,
        "chain": chain,
        "binder_kind": binder_kind,
        # The directory the run wrote, scrubbed. It is where this copy came
        # from on this machine, not where the structure came from: the database
        # that served the coordinates is in the research trace beside this file.
        "source": "" if output_dir is None else scrub_for_upload(str(output_dir)),
        "saved_at": saved_at,
        "scrubbed_files": scrubbed,
    }
    (dest / "generation.json").write_text(scrubbed_json(record), encoding="utf-8")
    return dest


def _scrub_archived_text(dest: Path) -> int:
    """Rewrite the archived text with machine paths and secrets removed.

    The live run keeps exactly what the tools wrote. This copy is the one that
    leaves the machine, and the generator's own metrics table carries this
    checkout's absolute path in four of its columns -- ``designed_structure_path``,
    ``input_json_path``, ``output_base_dir`` and ``dataset_name``. AGENTS.md
    names that as personal data that must not leave, and those paths are
    already wrong here anyway: they point at the run directory, not at the copy
    they now sit in.

    Coordinates take a narrower pass: the CIF writer stamps the account it ran
    under into ``_entry.author``, so that one value is replaced with the mmCIF
    null and the rest of the file is left as it was written.

    A symlink is skipped rather than followed. ``copytree`` preserves them, and
    writing through one would edit a file outside the archive entirely.
    """
    cleaned = 0
    for path in sorted(dest.rglob("*")):
        suffix = path.suffix.casefold()
        if path.is_symlink() or suffix not in _SCRUBBED_SUFFIXES | _AUTHOR_SUFFIXES:
            continue
        try:
            if not path.is_file() or path.stat().st_size > _SCRUB_LIMIT:
                continue
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        if suffix in _AUTHOR_SUFFIXES:
            scrubbed = _AUTHOR_ITEM.sub(r"\1.", text) if "_entry.author" in text else text
        else:
            scrubbed = scrub_for_upload(text)
        if scrubbed == text:
            continue
        try:
            path.write_text(scrubbed, encoding="utf-8")
        except OSError:
            continue
        cleaned += 1
    return cleaned


def _run_id(parts: Sequence[str]) -> str:
    """A name for this archived run that survives leaving its folder.

    The folder name was the only handle a saved generation had, and a folder
    can be renamed, copied or merged: a second archive of the same target lands
    beside the first as ``name-2``, with nothing inside either file saying
    which run it holds, so two generations could not be told apart or cited.

    It is derived from the record's own fields, the archive time among them, so
    it is reproducible from the file it sits in and a second archive of the
    same target gets its own id rather than repeating the first one's.
    """
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:16]


def _unique_directory(parent: Path, name: str) -> Path:
    """A new directory name. An existing generation is kept."""
    candidate = parent / name
    if not candidate.exists():
        return candidate
    number = 2
    while (parent / f"{name}-{number}").exists():
        number += 1
        if number > 10000:
            raise OSError(f"No free archive folder for {name}.")
    return parent / f"{name}-{number}"


def _contains(parent: Path, child: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
    except (OSError, ValueError):
        return False
    return True


def _copy_children(source: Path, dest: Path) -> None:
    for child in source.iterdir():
        target = dest / child.name
        if target.exists():
            continue
        if child.is_dir():
            shutil.copytree(child, target, symlinks=True)
        elif child.is_file() or child.is_symlink():
            shutil.copy2(child, target, follow_symlinks=False)


def _copy_inputs(out_dir: Path, stem: str, dest: Path) -> None:
    """Copy the conditioning files that sit beside the generation directory."""
    for name in (
        f"{stem}.cif",
        f"{stem}.annotated.cif",
        f"{stem}.yaml",
        f"{stem}.job.json",
        # Where the coordinates came from and the digest of the bytes that
        # arrived. Without it the archived structure cannot be shown to be the
        # file that was fetched.
        f"{stem}.source.json",
    ):
        source = out_dir / name
        target = dest / name
        if source.is_file() and not target.exists() and not _contains(dest, source):
            shutil.copy2(source, target)
