#!/usr/bin/env python3
"""Download the IUPHAR/BPS Guide to PHARMACOLOGY release and reduce it, once.

Three files, because the identifiers live apart: ``interactions.csv`` has the
affinities and a ligand id, ``ligands.csv`` the structures and the
approved/withdrawn marking, ``targets_and_families.csv`` the family each target
belongs to. Only the first was in the registry, so nothing in this repository
could fetch the other two and neither reduction could run.

Two reductions are written beside the downloads, both already in nanomolar
because GtoPdb's ``Original Affinity Median nm`` column is nanomolar as it
stands:

* ``gtopdb_panel_subset.tsv`` -- the ten ligand-safety axes, in the shared
  ``PANEL_SUBSET_COLUMNS`` format ``bindingdb_panel_subset`` writes, so the two
  merge and reach ``ligand_safety``'s per-ligand lookup and read-across. Take
  the volume literally: this is curation, not scale.
* ``gtopdb_offtarget_subset.tsv`` -- every target rather than ten, which is
  what the download is actually for. ``ligand_safety.offtarget_profile`` reads
  it, and it carries what a bare affinity throws away: the action, the
  curator's selectivity call, ``Primary Target``, and whether the compound is
  an approved or a withdrawn drug.

The merge into one panel subset is attempted when a BindingDB subset is on
disk. It is reported, never forced: a subset written by an older build of
``bindingdb_panel_subset`` can be missing a column this one writes, and
rewriting somebody else's derived table to fit is not this script's business.

Licence: Open Database License (ODbL) 1.0. Share-alike reaches any derivative
that is *redistributed*, which these reductions would be; nothing here uploads
them.

    python scripts/install_gtopdb.py
    python scripts/install_gtopdb.py --have
    python scripts/install_gtopdb.py --refresh
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from rfproteina.disease.tables import (
    BINDINGDB_PANEL_SUBSET_FILE,
    GTOPDB_FILES,
    SOURCES,
    GtoPdbRelease,
    Source,
    TooLargeError,
    default_directory,
    fetch,
    gtopdb_offtarget_subset,
    gtopdb_panel_subset,
    gtopdb_release,
    merge_panel_subsets,
    missing_gtopdb_files,
    readable,
)


def _unquote(value: str) -> str:
    text = value.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        return text[1:-1]
    return text


def _ensure_data_path() -> None:
    if os.environ.get("DATA_PATH", "").strip():
        return
    env_file = Path(__file__).resolve().parent.parent / ".env.local"
    if not env_file.is_file():
        raise SystemExit("DATA_PATH is unset and .env.local does not name one.")
    for raw in env_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() == "DATA_PATH":
            os.environ["DATA_PATH"] = _unquote(value)
            return
    raise SystemExit("DATA_PATH is unset and .env.local does not name one.")


def _sources() -> list[Source]:
    """The registry entries for the three files, in the order the reducers take them.

    Looked up by ``file`` rather than by name: the name is prose and the file
    name is the contract ``fetch`` writes and ``downloads`` checks.
    """
    by_file = {source.file: source for source in SOURCES}
    absent = [name for name in GTOPDB_FILES if name not in by_file]
    if absent:
        raise SystemExit(f"the registry has no entry for {', '.join(absent)}")
    return [by_file[name] for name in GTOPDB_FILES]


def _have(downloads_dir: Path, subsets_dir: Path) -> int:
    print(f"downloads  {downloads_dir}")
    for name in GTOPDB_FILES:
        path = downloads_dir / name
        size = path.stat().st_size if path.is_file() else 0
        print(f"  {'present' if size else 'absent ':8s} {name:26s} {size if size else '':>10}")
    print(f"subsets    {subsets_dir}")
    built = []
    for name in ("gtopdb_panel_subset.tsv", "gtopdb_offtarget_subset.tsv"):
        path = subsets_dir / name
        size = path.stat().st_size if path.is_file() else 0
        built.append(bool(size))
        print(f"  {'present' if size else 'absent ':8s} {name:26s} {size if size else '':>10}")
    complete = not missing_gtopdb_files(downloads_dir) and all(built)
    print(f"status     {'ready' if complete else 'incomplete'}")
    return 0 if complete else 1


def _fetch(sources: list[Source], downloads_dir: Path, *, refresh: bool) -> None:
    for source in sources:
        path = downloads_dir / source.file
        if path.is_file() and path.stat().st_size > 0 and not refresh:
            print(f"have   {path}  ({path.stat().st_size} bytes)")
            continue
        try:
            got = fetch(source, downloads_dir)
        except TooLargeError as error:
            raise SystemExit(str(error)) from error
        except OSError as error:
            raise SystemExit(f"{source.file} could not be downloaded: {error}") from error
        arrived = got.stat().st_size
        print(f"ready  {got}  ({arrived} bytes, {readable(arrived)})")
        if source.size is not None and arrived != source.size:
            # Not fatal: GtoPdb publishes quarterly and a new release is a
            # bigger file, not a broken one. It does make the registry's pinned
            # size wrong, and a test asserts that size, so say so here rather
            # than let it surface as an unexplained failure later.
            print(
                f"  note: {source.file} arrived at {arrived} bytes, and the registry pins "
                f"{source.size}. The release has moved; update Source.size for it."
            )


def _reduce(release: GtoPdbRelease, subsets_dir: Path) -> list[Path]:
    written: list[Path] = []
    panel = gtopdb_panel_subset(release.interactions, release.ligands, subsets_dir)
    if panel.blocker:
        raise SystemExit(f"the panel reduction failed: {panel.blocker}")
    print(
        f"ready  {panel.path}  ({panel.rows} rows over {panel.ligands} ligands, "
        f"{panel.scanned} interactions scanned)"
    )
    if panel.skipped:
        print(f"  skipped  {dict(panel.skipped)}")
    written.append(panel.path)

    offtarget = gtopdb_offtarget_subset(*release.as_arguments(), subsets_dir)
    if offtarget.blocker:
        raise SystemExit(f"the off-target reduction failed: {offtarget.blocker}")
    print(
        f"ready  {offtarget.path}  ({offtarget.rows} rows over {offtarget.targets} targets "
        f"and {offtarget.ligands} compounds)"
    )
    written.append(offtarget.path)
    return written


def _merge(panel_subset: Path, data_root: Path) -> None:
    bindingdb = data_root / "bindingdb" / BINDINGDB_PANEL_SUBSET_FILE
    if not bindingdb.is_file():
        print(
            f"merge  skipped: no BindingDB subset at {bindingdb}. The GtoPdb subset stands on "
            "its own; merging adds BindingDB's volume to this release's curation."
        )
        return
    destination = data_root / "panel_subset"
    merged = merge_panel_subsets([bindingdb, panel_subset], destination)
    if merged.blocker:
        print(f"merge  refused: {merged.blocker}")
        if merged.blocker.startswith("column_absent:"):
            print(
                f"       {bindingdb.name} predates this build of bindingdb_panel_subset and is "
                "missing a column it now writes. Regenerate it from BindingDB_All rather than "
                "editing it; the GtoPdb subset above is unaffected and readable on its own."
            )
        return
    print(f"ready  {merged.path}  ({merged.rows} rows over {merged.ligands} ligands)")
    if merged.skipped:
        print(f"  skipped  {dict(merged.skipped)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--have", action="store_true", help="report the downloads and the subsets, then stop")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="download again even when a file is already there, for a new quarterly release",
    )
    args = parser.parse_args(argv)
    _ensure_data_path()

    downloads_dir = default_directory()
    data_root = Path(os.environ["DATA_PATH"])
    subsets_dir = data_root / "gtopdb"

    if args.have:
        return _have(downloads_dir, subsets_dir)

    print(f"loading the GtoPdb release into {downloads_dir}")
    _fetch(_sources(), downloads_dir, refresh=args.refresh)

    release = gtopdb_release(downloads_dir)
    if release is None:
        raise SystemExit(f"still absent after the fetch: {', '.join(missing_gtopdb_files(downloads_dir))}")

    subsets_dir.mkdir(parents=True, exist_ok=True)
    panel_subset, _offtarget = _reduce(release, subsets_dir)
    _merge(panel_subset, data_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
