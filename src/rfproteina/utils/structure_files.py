"""Compression-independent structure discovery and filenames."""

from collections.abc import Iterable
from functools import cache
from os import PathLike
from pathlib import Path


@cache
def _all_suffixes() -> tuple[str, ...]:
    """Every recognized structure suffix, including PDB. Resolved on first use.

    ``CIF_LIKE_EXTENSIONS`` is a tuple of strings, but reaching it imports
    ``atomworks.io``, which pulls in the atomworks stack and torch -- 39 s on a
    checkout mounted from Windows. Everything else in this module is pure path
    handling, so a caller that only needs to recognize a filename should not pay
    for a model framework: :py:mod:`~rfproteina.metrics.oracles.imported_refolds`
    reads refolds off disk and needs no model, and this import was the whole of
    its remaining startup cost.

    Deferred rather than hardcoded, because the set of extensions is atomworks'
    to define and a copy here would drift from it silently.
    """
    from atomworks.io.utils.io_utils import CIF_LIKE_EXTENSIONS

    # In this codebase, jsons are never structure files, and instead may be ConditionSpec inputs
    return tuple(suffix for suffix in CIF_LIKE_EXTENSIONS if not suffix.startswith(".json"))


@cache
def _structure_suffixes() -> tuple[str, ...]:
    """Recognized structure suffixes with PDB left out."""
    return tuple(suffix for suffix in _all_suffixes() if not suffix.startswith(".pdb"))


def __getattr__(name: str) -> object:
    """Resolve ``STRUCTURE_FILE_SUFFIXES`` on first access, then cache it.

    It was a module-level constant and stays reachable under that name, so an
    importer of it is unaffected by the deferral above. Anything else raises
    :py:class:`AttributeError`, as a missing attribute must.
    """
    if name == "STRUCTURE_FILE_SUFFIXES":
        value = _structure_suffixes()
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def is_structure_file(path: PathLike | str, *, include_pdb: bool = False) -> bool:
    """Recognize exact structure suffixes, excluding backup and temporary files."""
    return str(path).lower().endswith(_all_suffixes() if include_pdb else _structure_suffixes())


def structure_file_stem(path: PathLike | str) -> str:
    """Remove the structure format and compression suffix, preserving dots in IDs."""
    name = Path(path).name
    for suffix in _all_suffixes():
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    raise ValueError(f"Unsupported structure filename: {path}")


def unique_structure_files(paths: Iterable[PathLike | str]) -> list[Path]:
    """Sort structures and reject multiple encodings of the same sibling design."""
    result = sorted(Path(path) for path in paths)
    identities: dict[Path, Path] = {}
    for path in result:
        key = path.absolute().with_name(structure_file_stem(path))
        if key in identities:
            raise ValueError(f"Ambiguous structure files for {key}: {identities[key]} and {path}. Keep one encoding.")
        identities[key] = path
    return result


def iter_structure_files(
    directory: PathLike | str, *, include_pdb: bool = False, stem_suffix: str | None = None, pattern: str = "**/*"
) -> list[Path]:
    """Discover mixed encodings by glob pattern (recursive by default) and optional artifact role."""
    return unique_structure_files(
        path
        for path in Path(directory).glob(pattern)
        if is_structure_file(path, include_pdb=include_pdb)
        and (stem_suffix is None or structure_file_stem(path).endswith(stem_suffix))
        and path.is_file()
    )


def find_structure_file(nominal_path: PathLike | str) -> Path | None:
    """Find one sibling encoding of a nominal structure path, or return None."""
    nominal_path = Path(nominal_path)
    stem = structure_file_stem(nominal_path)
    if not nominal_path.parent.is_dir():
        return None
    matches = unique_structure_files(
        path
        for path in nominal_path.parent.iterdir()
        if path.is_file() and is_structure_file(path, include_pdb=True) and structure_file_stem(path) == stem
    )
    return matches[0] if matches else None
