"""Read combination labels from a caller-supplied directory.

The directory is not searched for on disk. Training supplies the path. Coarse
``high_affinity_clean`` and ``high_affinity_toxic`` names still load, and a
file that names one binder kind and one axis loads as that axis. ``file_limit``
keeps a partial directory valid. A coarse name does not set an axis.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rfproteina.disease.combinations import parse_class_filename
from rfproteina.disease.tokens import BAD_CLASS, GOOD_CLASS

CLASS_NAMES = (GOOD_CLASS, BAD_CLASS)


@dataclass(frozen=True)
class ClassFile:
    """One structure file and the combination its name declares."""

    path: Path
    label: str
    coarse: bool = False
    binder_kind: str | None = None
    axis: str | None = None
    state: str | None = None


def class_from_filename(name: str) -> str | None:
    """Return the class named by ``name``, or ``None`` when it names neither."""
    parsed = parse_class_filename(name)
    if parsed is None:
        return None
    return parsed.label


def load_class_files(directory: str | Path | None, file_limit: int | None = None) -> list[ClassFile]:
    """Load some class files from ``directory``.

    A missing directory raises ``FileNotFoundError`` whose message contains the
    path. ``file_limit`` keeps the first files in sorted order. A file that
    names a shared axis without a binder kind raises ``ValueError``.
    """
    if directory is None or str(directory).strip() == "":
        raise ValueError("Set disease_class_data.directory to the class-file directory.")
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"Class-file directory not found: {root}")
    if file_limit is not None and file_limit < 0:
        raise ValueError(f"file_limit must be >= 0, got {file_limit}")

    found: list[ClassFile] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if file_limit is not None and len(found) >= file_limit:
            break
        parsed = parse_class_filename(path.name)
        if parsed is None:
            continue
        found.append(
            ClassFile(
                path=path,
                label=parsed.label,
                coarse=parsed.coarse,
                binder_kind=parsed.binder_kind,
                axis=parsed.axis,
                state=parsed.state,
            )
        )
    return found
