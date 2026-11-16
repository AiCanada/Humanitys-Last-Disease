#!/usr/bin/env python3
"""Fetch the human reference tables the disease screen reads, and stage a run.

Thin: the registry, the size limit, the download, the conversion and the
staging all live in ``rfproteina.disease.tables``. This chooses what to do and
prints what happened.

    python scripts/fetch_screening_tables.py --list
    python scripts/fetch_screening_tables.py
    python scripts/fetch_screening_tables.py --have
    python scripts/fetch_screening_tables.py --stage runs/my_run --disease "Breast cancer"
    python scripts/fetch_screening_tables.py --coverage runs/my_run

Three directories, and they are not interchangeable. Downloads land in one of
them under the names their servers give; a run reads tables out of its own
output directory under the screen's stems; and ``--stage`` is the step between,
because nothing else joins them. ``--have`` reports the first, ``--coverage``
the second.

Only human subsets are fetched. Every source here has a whole-database
counterpart between 11 GB and 1.39 TB sitting beside it under a similar name;
those are refused by the registry and answered either by a smaller file or by a
live query, and ``--list`` prints both sides so the choice is visible.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rfproteina.disease.tables import (
    QUERIES,
    REFUSED,
    SIZE_LIMIT,
    SOURCES,
    TooLargeError,
    coverage,
    declared_size,
    default_directory,
    downloads,
    fetch,
    readable,
    sample_line,
    stage_for_run,
)

#: What each ``handoff`` means for an operator holding the file.
_HANDOFF = {
    "convert": "converted into {stem} by --stage",
    "operator": "not staged as a table; read by --stage and passed on a flag",
    "none": "NO path to {stem} -- see the hazard below",
}


def show_registry() -> None:
    print("Downloaded (human subsets)")
    for source in SOURCES:
        size = f"{readable(source.size)} over the wire" if source.size is not None else "size not declared"
        reaches = _HANDOFF[source.handoff].format(stem=source.stem or "any table")
        print(f"  {source.name}")
        print(f"      lands as  {source.file}   ({size})   {source.licence}")
        print(f"      reaches the screen: {reaches}")
        if source.note:
            print(f"      {source.note}")
        if source.hazard:
            # The one thing an operator is most likely to do with a file whose
            # stem is printed beside it, and the reason not to.
            print(f"      HAZARD: {source.hazard}")
    print()
    print("Queried, not downloaded (the screen already calls these)")
    for query in QUERIES:
        print(f"  {query.axis}: {query.name}")
        print(f"      {query.endpoint}")
        print(f"      narrowed to {query.narrowed}")
    print()
    print(f"Refused, over the {readable(SIZE_LIMIT)} limit")
    for refused in REFUSED:
        print(f"  {refused.name}  {readable(refused.size)}")
        print(f"      instead: {refused.instead}")


def show_downloads(directory: Path) -> int:
    """What is on disk in a download directory, by the names ``fetch`` writes."""
    found = downloads(directory)
    print(f"Downloads in {directory}")
    for source, path in found.present:
        try:
            size = readable(path.stat().st_size)
        except OSError:
            size = "unreadable"
        print(f"  present  {source.file:34} {size:>10}")
    for source in found.absent:
        print(f"  ABSENT   {source.file:34} {source.name}")
    print()
    print("  all registry sources present" if found.complete else "  incomplete")
    print("  A download is not a staged run: use --stage to convert what can be converted.")
    return 0 if found.complete else 1


def show_coverage(directory: Path) -> int:
    """What a RUN directory can answer from disk, and what it will still ask."""
    found = coverage(directory)
    print(f"Screen tables in run {directory}")
    for stem, path in found.present:
        print(f"  present  {stem:16} {path.name}")
    for stem, name in found.missing:
        print(f"  MISSING  {stem:16} stage it from: {name}")
    if not found.present and not found.missing:
        print("  no axis in this screen reads a downloadable table")
    print()
    print("  Still asked online for every run, whatever is on disk:")
    for query in found.online:
        print(f"    {query.axis:18} {query.endpoint.split(',')[0]}")
    print()
    print("  complete" if found.complete else "  incomplete")
    return 0 if found.complete else 1


def show_staging(run: Path, *, disease: str, source_directory: Path, targets: list[str], release: str) -> int:
    """Convert and place what can be placed, and name everything that cannot."""
    made = stage_for_run(
        run,
        disease=disease,
        source_directory=source_directory,
        targets=targets or None,
        release=release,
    )
    print(f"Staging {run}")
    print(f"  from     {source_directory}")
    print(f"  disease  {disease}")
    if targets:
        print(f"  targets  {', '.join(targets)}")
    print()
    for stem, path in made.staged:
        print(f"  staged   {stem:16} {path.name}")
    for stem, reason in made.unstaged:
        marker = "SHORT  " if (stem, reason) in made.shortfalls else "n/a    "
        print(f"  {marker}  {stem:16} {reason}")
    if made.fold is not None and made.fold.written:
        fold = made.fold
        print()
        print(f"  fold table: {fold.rows} row(s), {fold.genes} gene(s), {fold.mapped} with a UniProt accession")
        if not fold.answers_accessions:
            # The failure this whole step exists to make visible: a readable
            # table that every real lookup misses.
            print("    An accession lookup against this table cannot match. A run whose")
            print("    condition spec names a UniProt accession will read untested.")
    if made.proteome is not None:
        print()
        print(f"  off-target axis can read {made.proteome.name} where it lies (not staged)")
    else:
        print()
        print("  off-target axis has no proteome: fetch the UniProt human reference proteome,")
        print("  or name one with --proteome on the screen.")
    if made.fold is not None and made.fold.available:
        print(f"  diseases in the source table include: {sample_line(made.fold.available)}")
    if made.fold is not None and made.fold.available_genes:
        # The disease matched and the targets did not, which is a different fix
        # from a disease that is absent, so it gets a different line.
        print(f"  genes this disease does report include: {sample_line(made.fold.available_genes)}")
    print()
    print("  Asked online on every run regardless:")
    for query in made.online:
        print(f"    {query.axis:18} {query.endpoint.split(',')[0]}")
    print()
    print("  equipped" if made.equipped else "  NOT equipped -- see the SHORT lines above")
    return 0 if made.equipped else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", action="store_true", help="Show every source, query and refusal; fetch nothing.")
    parser.add_argument("--to", type=Path, default=None, help=f"Where downloads go. Default: {default_directory()}")
    parser.add_argument("--only", action="append", default=[], help="Fetch just this source by name. Repeatable.")
    parser.add_argument("--have", action="store_true", help="Report which downloads are on disk in --to.")
    parser.add_argument("--coverage", type=Path, help="Report which screen tables a RUN directory holds.")
    parser.add_argument("--stage", type=Path, help="Convert the downloads into screen tables in this RUN directory.")
    parser.add_argument("--disease", default="", help="The one disease the fold table is for. Required by --stage.")
    parser.add_argument(
        "--gene",
        action="append",
        default=[],
        help="Narrow the fold table to this gene symbol or UniProt accession. Repeatable.",
    )
    parser.add_argument("--release", default="", help="Source release to record in the provenance column.")
    parser.add_argument("--limit", type=int, default=SIZE_LIMIT, help="Byte limit per download.")
    parser.add_argument("--dry-run", action="store_true", help="Ask each server its size and stop.")
    arguments = parser.parse_args(argv)

    destination = arguments.to or default_directory()

    if arguments.list:
        show_registry()
        return 0
    if arguments.have:
        return show_downloads(destination)
    if arguments.coverage is not None:
        return show_coverage(arguments.coverage)
    if arguments.stage is not None:
        if not arguments.disease.strip():
            # Not defaulted on purpose. The fold table holds one disease
            # because the axis asks how far the target moves in THAT disease,
            # so a guess here answers a question the run did not ask.
            print("--stage needs --disease: the fold table is for one disease.", file=sys.stderr)
            print('For example: --stage runs/my_run --disease "Breast cancer"', file=sys.stderr)
            return 2
        return show_staging(
            arguments.stage,
            disease=arguments.disease,
            source_directory=destination,
            targets=arguments.gene,
            release=arguments.release,
        )

    wanted = list(SOURCES)
    if arguments.only:
        wanted = [source for source in SOURCES if source.name in set(arguments.only)]
        unknown = set(arguments.only) - {source.name for source in SOURCES}
        if unknown:
            print(f"No such source: {', '.join(sorted(unknown))}", file=sys.stderr)
            print("Use --list for the names.", file=sys.stderr)
            return 2

    print(f"Fetching {len(wanted)} human table(s) into {destination}")
    print(f"Limit: {readable(arguments.limit)} per download. Nothing larger is taken.")
    print()

    failures = 0
    for source in wanted:
        announced = declared_size(source.url)
        shown = readable(announced) if announced is not None else "no size declared"
        print(f"{source.name}  ({shown})")
        if arguments.dry_run:
            continue
        try:
            landed = fetch(source, destination, limit=arguments.limit)
        except TooLargeError as exc:
            print(f"  REFUSED  {exc}")
            failures += 1
        except OSError as exc:
            print(f"  FAILED   {exc}")
            failures += 1
        else:
            print(f"  got      {landed}")
    print()
    if arguments.dry_run:
        # Said plainly: a dry run asked each server its size and fetched
        # nothing, and must not report arrival for files that are not there.
        print(f"Dry run: {len(wanted)} source(s) reachable, nothing downloaded.")
    elif failures:
        print(f"{failures} of {len(wanted)} did not arrive.")
    else:
        print("All requested tables arrived.")
    print()
    print("A download is not a staged run. Three of these five feed no screen table at")
    print("all (--list says which, and what dropping them in does); the fourth has to be")
    print("converted. Run --stage next.")
    print()
    print("These three axes are asked online on every run, whatever is on disk:")
    for query in QUERIES:
        print(f"  {query.axis:18} {query.endpoint.split(',')[0]}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
