"""Turn an MHCflurry run into the ``mhc_i`` table the screen already reads.

``tables.py`` can download IEDB's ``mhc_ligand_full.tsv`` and then refuses to
convert it: the registry marks that source ``handoff="none"`` with the hazard
note that it is 112 columns of assay records about other people's peptides, not
a prediction over this design. Worse, dropping it next to a run under the
``mhc_i`` stem *displaces* the computation that would otherwise run. So the
class I axis stays untested no matter how much IEDB is on disk.

The missing piece is a predictor. MHCflurry 2.x is Apache-2.0 and its weights
are downloaded by its own ``mhcflurry-downloads fetch`` command, so nothing
here ships a model. This module does three things and no more:

1. cuts a protein sequence into the peptide windows a class I predictor takes,
2. runs the ``mhcflurry-predict`` command when it is on PATH, and
3. renames MHCflurry's output columns onto the ones ``liability.py`` reads.

Point 3 is the part that is easy to get wrong. ``liability.py`` reads a rank
from ``percentile_rank`` and a concentration only from a column that says
nanomolar: ``ic50_nm``, ``affinity_nm`` or ``aff_nm``, or a bare ``ic50`` with
a unit column that says nM. MHCflurry writes ``mhcflurry_affinity`` in
nanomolar and ``mhcflurry_affinity_percentile`` as a percentile. The rename
below is written out explicitly so a reviewer can check it against both sides,
and the unit column is written as ``nM`` rather than assumed.

Nothing is invented here. If MHCflurry is not installed the functions raise;
they never emit a table of made-up scores. The thresholds that then apply are
the ones already in ``liability.py`` (class I rank <= 1 or IC50 <= 500 nM,
IEDB's published convention), and this module does not restate or change them.
"""

from __future__ import annotations

import csv
import io
import os
import shutil
import subprocess
import tempfile
from collections.abc import Iterable, Sequence
from pathlib import Path

#: Peptide widths a class I predictor is normally run over.
CLASS_I_WIDTHS = (8, 9, 10, 11)
#: The twenty standard residues. A window holding anything else is dropped,
#: because a predictor's answer on an X is not an answer.
STANDARD_RESIDUES = frozenset("ACDEFGHIKLMNPQRSTVWY")
#: MHCflurry column -> the column ``liability.py`` reads.
COLUMN_MAP = {
    "peptide": "peptide",
    "allele": "allele",
    "mhcflurry_affinity": "ic50_nm",
    "mhcflurry_affinity_percentile": "percentile_rank",
    "mhcflurry_presentation_percentile": "presentation_percentile",
    "mhcflurry_presentation_score": "presentation_score",
    "mhcflurry_processing_score": "processing_score",
}
#: Columns written out, in order. ``unit`` is written so no reader has to
#: assume what ``ic50_nm`` is in.
OUTPUT_COLUMNS = ("peptide", "allele", "percentile_rank", "ic50_nm", "unit", "method", "start")
METHOD = "mhcflurry"


class PredictorMissingError(Exception):
    """MHCflurry is not installed, or its command failed. No table is written."""


def windows(sequence: str, widths: Sequence[int] = CLASS_I_WIDTHS) -> list[tuple[int, str]]:
    """Every peptide window of the given widths, as ``(1-based start, peptide)``.

    Windows containing a non-standard residue are dropped rather than passed to
    a predictor that has no value for them.
    """
    text = "".join((sequence or "").split()).upper()
    found: list[tuple[int, str]] = []
    for width in widths:
        if width <= 0:
            continue
        for start in range(0, max(0, len(text) - width + 1)):
            piece = text[start : start + width]
            if set(piece) <= STANDARD_RESIDUES:
                found.append((start + 1, piece))
    return found


def convert_mhcflurry_csv(text: str, *, starts: dict[str, int] | None = None) -> str:
    """MHCflurry prediction CSV -> a TSV with the columns ``liability.py`` reads.

    ``starts`` optionally carries the 1-based position each peptide was cut
    from, so a reviewer can map a hit back onto the design. A peptide that
    appears at several positions is reported at the first.

    Raises ``ValueError`` when the input does not carry a peptide and an
    allele, because a renamed table that is missing those reads as "no numbers
    in it" rather than as a broken conversion.
    """
    reader = csv.DictReader(io.StringIO(text))
    field_names = [name.strip() for name in (reader.fieldnames or [])]
    if "peptide" not in field_names or "allele" not in field_names:
        raise ValueError("mhcflurry_csv_missing_peptide_or_allele")
    rows: list[dict[str, str]] = []
    for row in reader:
        clean = {key.strip(): (value or "").strip() for key, value in row.items() if key}
        peptide = clean.get("peptide", "").upper()
        if not peptide:
            continue
        affinity = clean.get("mhcflurry_affinity", "")
        percentile = clean.get("mhcflurry_affinity_percentile", "")
        rows.append(
            {
                "peptide": peptide,
                "allele": clean.get("allele", ""),
                "percentile_rank": percentile,
                "ic50_nm": affinity,
                "unit": "nM" if affinity else "",
                "method": METHOD,
                "start": str(starts.get(peptide, "")) if starts else "",
            }
        )
    if not rows:
        raise ValueError("mhcflurry_csv_no_rows")
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=list(OUTPUT_COLUMNS), delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue()


def write_mhc_i_table(
    csv_text: str,
    directory: Path,
    *,
    starts: dict[str, int] | None = None,
    stem: str = "mhc_i",
) -> Path:
    """Write the converted table beside a run, where ``liability.py`` looks for it."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{stem}.tsv"
    path.write_text(convert_mhcflurry_csv(csv_text, starts=starts), encoding="utf-8")
    return path


#: An environment variable naming the ``mhcflurry-predict`` executable. This
#: exists because MHCflurry 2.3 requires ``pandas>=3`` while this project pins
#: ``pandas`` 2.x, so installing it into the project environment would move a
#: dependency under every other screen. It is installed into its own virtual
#: environment instead and pointed at from here. Nothing is downloaded by this
#: module either way.
MHCFLURRY_COMMAND_ENV = "MHCFLURRY_PREDICT"
#: What MHCflurry's percentile column is on. Measured, not assumed: on
#: mhcflurry 2.3.29 with ``models_class1_presentation`` (release 20260928),
#: ``SIINFEKL`` against ``HLA-A*02:01`` returns an affinity of 6249.06 nM and an
#: affinity percentile of 7.96. A percentile of 7.96 beside a 6 uM affinity can
#: only be a percentage, so the column is 0-100 and not 0-1. ``liability.py``
#: compares a class I rank against 1.0, which is the same convention.
PERCENTILE_SCALE = "percent_0_100"
#: One observed row, kept so a reviewer can re-run it and so a test can assert
#: the conversion rescales nothing. Numbers are from the run described above.
OBSERVED_EXAMPLE = {
    "mhcflurry_version": "2.3.29",
    "models": "models_class1_presentation, release 20260928",
    "observed": "2026-11-16 (America/Halifax)",
    "rows": (
        ("HLA-A*02:01", "SIINFEKL", 6249.061165259931, 7.964161947308878),
        ("HLA-A*02:01", "NLVPMVATV", 34.716636716484736, 0.18943906717683803),
        ("HLA-A*02:01", "GILGFVFTL", 25.841361314444036, 0.1308625820725428),
    ),
}


def predictor_command() -> str | None:
    """The ``mhcflurry-predict`` executable to run, or None when there is none.

    ``MHCFLURRY_PREDICT`` wins over ``PATH`` so an installation kept out of the
    project environment can be used without changing ``PATH`` for every other
    process. A value that is not an executable file is ignored rather than
    passed to ``subprocess``, so a stale variable reads as "not installed"
    instead of as a crash.
    """
    named = os.environ.get(MHCFLURRY_COMMAND_ENV, "").strip()
    if named:
        if os.path.isfile(named) and os.access(named, os.X_OK):
            return named
        found = shutil.which(named)
        if found:
            return found
    return shutil.which("mhcflurry-predict")


def predictor_available() -> bool:
    """True when a ``mhcflurry-predict`` executable can be found. Checked, never assumed."""
    return predictor_command() is not None


def run_mhcflurry(
    peptides: Iterable[tuple[int, str]],
    alleles: Sequence[str],
    *,
    timeout_s: int = 1800,
) -> tuple[str, dict[str, int]]:
    """Run ``mhcflurry-predict`` over the windows and return its CSV plus the start map.

    Raises :class:`PredictorMissingError` when the command is absent or exits
    non-zero. Install with ``pip install mhcflurry`` (Apache-2.0) and fetch the
    weights once with ``mhcflurry-downloads fetch models_class1_presentation``;
    this function downloads nothing itself.
    """
    pairs = list(peptides)
    if not pairs:
        raise ValueError("no_peptides")
    if not alleles:
        raise ValueError("no_alleles")
    command = predictor_command()
    if command is None:
        raise PredictorMissingError(f"mhcflurry-predict not on PATH and ${MHCFLURRY_COMMAND_ENV} unset")
    starts: dict[str, int] = {}
    for start, peptide in pairs:
        starts.setdefault(peptide, start)
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["allele", "peptide"])
    for allele in alleles:
        for peptide in starts:
            writer.writerow([allele, peptide])
    # Written into a temporary directory rather than handed to the command as
    # ``/dev/stdin``. That path is POSIX-only and the previous version of this
    # function failed on Windows outside WSL for that reason alone. The
    # directory and its file go away with the context manager, including on a
    # timeout.
    with tempfile.TemporaryDirectory(prefix="mhcflurry_") as scratch:
        request = Path(scratch) / "peptides.csv"
        request.write_text(buffer.getvalue(), encoding="utf-8")
        try:
            finished = subprocess.run(
                [command, str(request)],
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise PredictorMissingError(f"mhcflurry_failed:{type(error).__name__}") from error
    if finished.returncode != 0:
        raise PredictorMissingError(f"mhcflurry_exit_{finished.returncode}:{finished.stderr.strip()[:200]}")
    return finished.stdout, starts


def predict_sequence(
    sequence: str,
    alleles: Sequence[str],
    *,
    widths: Sequence[int] = CLASS_I_WIDTHS,
    directory: Path | None = None,
    timeout_s: int = 1800,
) -> dict[str, object]:
    """Design sequence in, ``mhc_i.tsv`` out. The whole class I path in one call.

    Cuts the windows, runs the predictor, renames the columns and -- when
    ``directory`` is given -- writes the table where ``liability.py`` looks for
    it. Returns the table text, the path written, and the counts a reader needs
    to see that the scan covered the chain: a partial scan clears the axis on
    the part that was looked at, and ``liability.py`` has its own coverage rule
    that reads these same rows.

    Raises :class:`PredictorMissingError` when no predictor is installed. It
    never writes a table in that case: an empty or invented ``mhc_i.tsv``
    beside a run *displaces* the computation, which is the exact failure this
    module was written to undo.
    """
    pairs = windows(sequence, widths)
    if not pairs:
        raise ValueError("no_peptides")
    csv_text, starts = run_mhcflurry(pairs, alleles, timeout_s=timeout_s)
    table = convert_mhcflurry_csv(csv_text, starts=starts)
    path: Path | None = None
    if directory is not None:
        path = Path(directory) / "mhc_i.tsv"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(table, encoding="utf-8")
    return {
        "method": METHOD,
        "percentile_scale": PERCENTILE_SCALE,
        "affinity_unit": "nM",
        "alleles": list(alleles),
        "widths": list(widths),
        "peptides": len(starts),
        "windows": len(pairs),
        "rows": max(0, len(table.splitlines()) - 1),
        "table": table,
        "path": str(path) if path is not None else None,
    }


def contract() -> dict[str, object]:
    """What this module promises ``liability.py``, in one readable record.

    Written out so a reviewer can diff it against both sides rather than
    reading two files. If MHCflurry renames a column or changes its percentile
    convention, the rename below is where that breaks, and the test
    ``test_the_conversion_rescales_nothing`` is what catches it.
    """
    return {
        "column_map": dict(COLUMN_MAP),
        "output_columns": list(OUTPUT_COLUMNS),
        "affinity_unit": "nM",
        "percentile_scale": PERCENTILE_SCALE,
        "read_by": {
            "ic50": "liability._IC50_COLUMNS, nanomolar only, cut at 500 nM",
            "rank": "liability._RANK_COLUMNS, cut at 1.0 for class I",
        },
        "observed_example": OBSERVED_EXAMPLE,
    }


#: The three proteins ``liability.py`` will accept as the tolerated reference
#: for a protein binder: serum albumin, the IgG1 constant region and ubiquitin
#: (``liability._REFERENCE_PROTEINS``). A class I table for a protein binder
#: cannot reach a verdict without a reference scanned the same way, which is
#: what ``reference_not_configured`` means on that axis.
REFERENCE_ACCESSIONS = ("P02768", "P01857", "P0CG47")
#: Columns the reference table carries on top of the design table's. The
#: protein column attributes a row, and ``length`` is what the density is
#: divided by. ``liability._protein_lengths`` ignores a stated length at or
#: below the longest peptide, so the real protein length has to be written.
REFERENCE_COLUMNS = (
    "peptide",
    "allele",
    "percentile_rank",
    "ic50_nm",
    "unit",
    "method",
    "protein",
    "start",
    "length",
)


def convert_reference_csv(text: str, *, protein: str, length: int, starts: dict[str, int]) -> list[dict[str, str]]:
    """One protein's MHCflurry CSV -> reference rows, with the protein attributed.

    Separate from :func:`convert_mhcflurry_csv` because the reference table
    carries two columns the design table does not, and because the design
    table must *not* carry them: a design row naming a protein would be read as
    belonging to that protein.
    """
    reader = csv.DictReader(io.StringIO(text))
    names = [name.strip() for name in (reader.fieldnames or [])]
    if "peptide" not in names or "allele" not in names:
        raise ValueError("mhcflurry_csv_missing_peptide_or_allele")
    rows: list[dict[str, str]] = []
    for row in reader:
        clean = {key.strip(): (value or "").strip() for key, value in row.items() if key}
        peptide = clean.get("peptide", "").upper()
        if not peptide:
            continue
        affinity = clean.get("mhcflurry_affinity", "")
        rows.append(
            {
                "peptide": peptide,
                "allele": clean.get("allele", ""),
                "percentile_rank": clean.get("mhcflurry_affinity_percentile", ""),
                "ic50_nm": affinity,
                "unit": "nM" if affinity else "",
                "method": METHOD,
                "protein": protein,
                "start": str(starts.get(peptide, "")),
                "length": str(length),
            }
        )
    if not rows:
        raise ValueError("mhcflurry_csv_no_rows")
    return rows


def predict_reference_table(
    proteins: dict[str, str],
    alleles: Sequence[str],
    *,
    widths: Sequence[int] = CLASS_I_WIDTHS,
    directory: Path | None = None,
    stem: str = "mhc_i",
    timeout_s: int = 7200,
) -> dict[str, object]:
    """Scan the tolerated reference proteins and write ``mhc_i_reference.tsv``.

    ``proteins`` maps an accession to its sequence. The sequences are not
    downloaded here: the caller supplies them, because this module does not
    fetch anything, and because a silently fetched sequence is a silently
    chosen reference. Only the three accessions in :data:`REFERENCE_ACCESSIONS`
    will be accepted by ``liability.py``; any other row is dropped there, so
    one is refused here rather than written and ignored.

    The scan must match the design's: same predictor, same alleles, same
    widths. ``liability._reference_density`` refuses the comparison otherwise,
    and those refusals are what make the bar meaningful.
    """
    unknown = [name for name in proteins if name.upper() not in REFERENCE_ACCESSIONS]
    if unknown:
        raise ValueError(f"not_a_tolerated_reference_protein:{','.join(sorted(unknown))}")
    if not proteins:
        raise ValueError("no_proteins")
    rows: list[dict[str, str]] = []
    scanned: dict[str, int] = {}
    for accession, sequence in proteins.items():
        text = "".join((sequence or "").split()).upper()
        pairs = windows(text, widths)
        if not pairs:
            raise ValueError(f"no_peptides:{accession}")
        csv_text, starts = run_mhcflurry(pairs, alleles, timeout_s=timeout_s)
        rows.extend(convert_reference_csv(csv_text, protein=accession.upper(), length=len(text), starts=starts))
        scanned[accession.upper()] = len(text)
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=list(REFERENCE_COLUMNS), delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    table = out.getvalue()
    path: Path | None = None
    if directory is not None:
        path = Path(directory) / f"{stem}_reference.tsv"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(table, encoding="utf-8")
    return {
        "method": METHOD,
        "proteins": scanned,
        "alleles": list(alleles),
        "widths": list(widths),
        "rows": len(rows),
        "table": table,
        "path": str(path) if path is not None else None,
    }
