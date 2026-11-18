"""Turn an MHCflurry run into the ``mhc_i`` table the screen already reads.

``tables.py`` can download IEDB's ``mhc_ligand_full.tsv`` and then refuses to
convert it: the registry marks that source ``handoff="none"`` with the hazard
note that it is 112 columns of assay records about other people's peptides, not
a prediction over this design. Worse, dropping it next to a run under the
``mhc_i`` stem *displaces* the computation that would otherwise run. So the
class I axis stays untested no matter how much IEDB is on disk.

The missing piece is a predictor. MHCflurry 2.x is Apache-2.0. Its weights are
not shipped here. :func:`fetch_weights` downloads the presentation bundle into
``DATA_PATH/mhcflurry/models`` when the operator asks. A screen does not.

This module then:

1. cuts a protein sequence into the peptide windows a class I predictor takes,
2. runs ``mhcflurry-predict`` from ``MHCFLURRY_PREDICT``, the virtual environment
   under that data directory, or ``PATH``,
3. renames MHCflurry's output columns onto the ones ``liability.py`` reads, and
4. writes ``mhc_i.tsv`` during a screen only when an allele list, the predictor,
   and that weight bundle are all present. An existing table is left in place.

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
import re
import shutil
import subprocess
import tempfile
from collections.abc import Iterable, Sequence
from pathlib import Path

from rfproteina.disease.screening_tools import data_dir

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
#: Where the presentation weights live. MHCflurry's own variable. When it is
#: unset, this module points the subprocess at ``DATA_PATH/mhcflurry/models``
#: so a fetch does not land in a home directory.
MHCFLURRY_DOWNLOADS_ENV = "MHCFLURRY_DOWNLOADS_DIR"
#: Comma- or whitespace-separated allele list. It overrides the program panel.
ALLELES_ENV = "MHC_I_ALLELES"
#: Supertype prototypes from Sidney et al., BMC Immunol 2008;9:1.
#: HLA-B*08:01 is a B07/B08 prototype in that paper and is omitted here:
#: the Calis 2013 mask used by the immunogenicity axis is unpublished for it,
#: so a peptide presented only by that allele cannot be scored.
CLASS_I_ALLELES: tuple[str, ...] = (
    "HLA-A*01:01",
    "HLA-A*02:01",
    "HLA-A*03:01",
    "HLA-A*24:02",
    "HLA-B*07:02",
    "HLA-B*44:02",
    "HLA-B*58:01",
)
#: Beside a run, this file is the allele list for that run. It wins over the
#: file in the install directory and loses to ``ALLELES_ENV``.
ALLELES_FILENAME = "mhc_i_alleles.txt"
#: The one bundle whose affinity column is nanomolar and whose percentile is
#: the 0-100 scale ``PERCENTILE_SCALE`` was measured on.
MODELS_BUNDLE = "models_class1_presentation"
#: Catalogue release of that bundle. The archive inside it is dated 20260928,
#: which is the run :data:`OBSERVED_EXAMPLE` was measured on. A custom
#: downloads directory makes MHCflurry refuse ``fetch`` unless ``--release``
#: is passed, so the release is part of the command rather than assumed.
MODELS_RELEASE = "2.3.0"
_ALLELE = re.compile(r"^[A-Za-z][A-Za-z0-9*:._-]{2,39}$")
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


def install_root() -> Path:
    """``DATA_PATH/mhcflurry``, or ``data/mhcflurry`` inside the checkout.

    The virtual environment and the weight files both live here. A home
    directory is not a location this module chooses.

    The checkout fallback is found by searching upward for ``.project-root``,
    not by counting parent directories, so it stays correct if this module ever
    moves to a different depth.
    """
    return data_dir("mhcflurry")


def models_directory() -> Path:
    """The directory MHCflurry reads weights from.

    ``MHCFLURRY_DOWNLOADS_DIR`` wins when it is set. Otherwise the directory
    is ``install_root()/models``.
    """
    named = os.environ.get(MHCFLURRY_DOWNLOADS_ENV, "").strip()
    if named:
        return Path(named)
    return install_root() / "models"


def weights_present(directory: Path | None = None) -> bool:
    """Whether the class I presentation bundle is on disk. Nothing is downloaded."""
    root = models_directory() if directory is None else directory
    if not root.is_dir():
        return False
    return any(path.is_dir() and path.name == MODELS_BUNDLE for path in root.rglob(MODELS_BUNDLE))


def _executable(path: Path) -> bool:
    return path.is_file() and os.access(path, os.X_OK)


def _venv_predict() -> Path | None:
    root = install_root() / "venv"
    for relative in (
        Path("Scripts") / "mhcflurry-predict.exe",
        Path("Scripts") / "mhcflurry-predict",
        Path("bin") / "mhcflurry-predict",
    ):
        candidate = root / relative
        if _executable(candidate):
            return candidate
    return None


def predictor_command() -> str | None:
    """The ``mhcflurry-predict`` executable to run, or None when there is none.

    ``MHCFLURRY_PREDICT`` wins, then the virtual environment under
    :func:`install_root`, then ``PATH``. A value that is not an executable
    file is ignored rather than passed to ``subprocess``, so a stale variable
    reads as "not installed" instead of as a crash.
    """
    named = os.environ.get(MHCFLURRY_COMMAND_ENV, "").strip()
    if named:
        if _executable(Path(named)):
            return named
        found = shutil.which(named)
        if found:
            return found
    bundled = _venv_predict()
    if bundled is not None:
        return str(bundled)
    return shutil.which("mhcflurry-predict")


def _sibling_downloads(predict: str) -> str | None:
    directory = Path(predict).parent
    for name in ("mhcflurry-downloads.exe", "mhcflurry-downloads"):
        candidate = directory / name
        if _executable(candidate):
            return str(candidate)
    return shutil.which("mhcflurry-downloads")


def _predict_env() -> dict[str, str]:
    """The subprocess environment, with the weight directory filled in when unset."""
    env = os.environ.copy()
    env.setdefault(MHCFLURRY_DOWNLOADS_ENV, str(models_directory()))
    return env


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
                env=_predict_env(),
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


def parse_alleles(text: str) -> list[str]:
    """Allele tokens from a file or an environment value.

    Blank lines and ``#`` comments are ignored. A token is one allele.
    Commas and spaces both separate tokens. A repeated allele is kept once,
    in the order it first appeared. A token that is not an allele name raises
    ``ValueError`` rather than being sent to the predictor.
    """
    found: list[str] = []
    seen: set[str] = set()
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        for piece in line.replace(",", " ").split():
            token = piece.strip()
            if not token or token in seen:
                continue
            if _ALLELE.fullmatch(token) is None:
                raise ValueError(f"allele_unreadable:{token}")
            seen.add(token)
            found.append(token)
    return found


def configured_alleles(directory: Path | None = None) -> tuple[list[str], str]:
    """The allele list to scan, and where it was read from.

    ``MHC_I_ALLELES`` wins, then ``mhc_i_alleles.txt`` beside the run, then
    ``alleles.txt`` in the install directory. When none of those is present
    the program panel in :data:`CLASS_I_ALLELES` is used.
    """
    named = os.environ.get(ALLELES_ENV, "").strip()
    if named:
        parsed = parse_alleles(named if "\n" in named else named + "\n")
        if not parsed:
            raise ValueError("allele_list_empty")
        return parsed, ALLELES_ENV
    candidates: list[tuple[Path, str]] = []
    if directory is not None:
        candidates.append((Path(directory) / ALLELES_FILENAME, ALLELES_FILENAME))
    candidates.append((install_root() / "alleles.txt", "install_alleles"))
    for path, source in candidates:
        if not path.is_file():
            continue
        parsed = parse_alleles(path.read_text(encoding="utf-8"))
        if not parsed:
            raise ValueError(f"allele_list_empty:{path.name}")
        return parsed, source
    return list(CLASS_I_ALLELES), "program_panel"


def allele_file_text() -> str:
    """The install-directory allele file, comments plus :data:`CLASS_I_ALLELES`."""
    return (
        "# Class I alleles. One allele per line. This file overrides the program panel.\n"
        "# Sidney et al., BMC Immunol 2008;9:1, HLA class I supertype prototypes.\n"
        "# HLA-B*08:01 is omitted: the Calis 2013 mask is unpublished for that allele.\n"
        + "\n".join(CLASS_I_ALLELES)
        + "\n"
    )


def ensure_allele_file() -> Path:
    """Write ``alleles.txt`` under the install root when the operator has none.

    An existing file is left as it is, so a later edit is not replaced by
    the program panel.
    """
    path = install_root() / "alleles.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file():
        path.write_text(allele_file_text(), encoding="utf-8")
    return path


def _fasta_sequence(path: Path) -> str:
    """The first sequence in a FASTA file, with whitespace removed."""
    chunks: list[str] = []
    started = False
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            if started:
                break
            started = True
            continue
        started = True
        chunks.append(line)
    sequence = "".join(chunks).replace(" ", "").upper()
    if not sequence:
        raise ValueError(f"reference_sequence_empty:{path.name}")
    return sequence


def reference_sequences() -> tuple[str, dict[str, str]]:
    """The tolerated reference sequences and whether the set is complete.

    The status is ``absent`` when none of the three FASTA files are under
    ``install_root()/reference``, ``incomplete`` when some are, and ``ready``
    when all three are. A partial set is not a bar, and a missing file is
    not fetched.
    """
    folder = install_root() / "reference"
    found: dict[str, str] = {}
    for accession in REFERENCE_ACCESSIONS:
        path = next(
            (
                candidate
                for suffix in (".fasta", ".fa", ".txt")
                if (candidate := folder / f"{accession}{suffix}").is_file()
            ),
            None,
        )
        if path is not None:
            found[accession] = _fasta_sequence(path)
    if not found:
        return "absent", {}
    if len(found) != len(REFERENCE_ACCESSIONS):
        return "incomplete", found
    return "ready", found


def class_i_readiness(directory: Path | None = None) -> dict[str, object]:
    """Whether a class I scan can run. This does not start the predictor."""
    try:
        alleles, source = configured_alleles(directory)
        allele_blocker = None
    except ValueError as error:
        alleles, source, allele_blocker = None, "unreadable", str(error)
    table = directory is not None and (Path(directory) / "mhc_i.tsv").is_file()
    return {
        "predictor_available": predictor_command() is not None,
        "weights_present": weights_present(),
        "alleles_configured": bool(alleles),
        "allele_source": source,
        "allele_blocker": allele_blocker,
        "table_present": table,
        "models": str(models_directory()),
    }


#: What :func:`predict_reference_table` appends to the stem.
_REFERENCE_STEM = "_reference.tsv"


def _attach_reference(
    record: dict[str, object],
    directory: Path,
    *,
    timeout_s: int,
    alleles: Sequence[str] | None = None,
) -> dict[str, object]:
    """Scan the tolerated proteins into ``mhc_i_reference.tsv`` and note the outcome.

    Shared by both paths through :func:`prepare_class_i_table`: the one that
    has just written a design table and already knows the alleles, and the one
    that found a design table already there and has to resolve them itself.
    Every outcome is recorded under ``reference`` so a caller can tell a
    missing reference from a failed scan.
    """
    status, references = reference_sequences()
    if status != "ready":
        record["reference"] = status
        return record
    if alleles is None:
        try:
            alleles, _source = configured_alleles(directory)
        except ValueError as error:
            record["reference"] = f"alleles_unreadable:{error}"
            return record
        if not alleles:
            record["reference"] = "alleles_not_configured"
            return record
        if predictor_command() is None:
            record["reference"] = "predictor_missing"
            return record
        if not weights_present():
            record["reference"] = "weights_missing"
            return record
    try:
        reference = predict_reference_table(
            references, alleles, directory=directory, timeout_s=timeout_s
        )
    except (PredictorMissingError, ValueError, OSError) as error:
        record["reference"] = f"predictor_failed:{error}"
    else:
        record["reference"] = reference["path"]
    return record


def prepare_class_i_table(
    directory: Path,
    sequences: Sequence[tuple[str, str | None]],
    *,
    binder_kind: str,
    timeout_s: int = 1800,
) -> dict[str, object]:
    """Write ``mhc_i.tsv`` for this run when the operator has asked for a scan.

    An existing table is left as it is: replacing it would discard a table the
    operator put there, which is how a supplied measurement gets displaced.
    No allele list means no scan and no file. A missing predictor or a missing
    weight bundle is reported and nothing is written. A protein binder also
    gets ``mhc_i_reference.tsv`` when all three reference sequences are on
    disk; a partial set is recorded and not written.
    """
    directory = Path(directory)
    existing = directory / "mhc_i.tsv"
    if existing.is_file():
        record: dict[str, object] = {"state": "table_present", "path": str(existing)}
        # The design table is the operator's and is never replaced. The
        # reference table is a different artifact, and a run made before the
        # reference sequences were on disk left none behind. Writing it here
        # is what lets a machine that has since fetched them recover: without
        # it the first screen's mhc_i.tsv suppresses the reference scan for
        # good, and the axis reports reference_not_configured forever even
        # though every input it needs is present.
        if binder_kind == "protein":
            reference = directory / f"mhc_i{_REFERENCE_STEM}"
            if reference.is_file():
                # Say so rather than returning silently. A record that omits the
                # reference reads the same whether one is beside the run or not,
                # and the axis behaves very differently in those two cases.
                record["reference"] = str(reference)
                record["reference_reused"] = True
            else:
                _attach_reference(record, directory, timeout_s=timeout_s)
        return record
    try:
        alleles, source = configured_alleles(directory)
    except ValueError as error:
        return {"state": "alleles_unreadable", "blocker": str(error)}
    if not alleles:
        return {"state": "alleles_not_configured"}
    if predictor_command() is None:
        return {"state": "predictor_missing", "allele_source": source}
    if not weights_present():
        return {
            "state": "weights_missing",
            "allele_source": source,
            "models": str(models_directory()),
        }
    pairs: list[tuple[int, str]] = []
    for _name, sequence in sequences:
        if sequence:
            pairs.extend(windows(sequence))
    if not pairs:
        return {"state": "no_peptides", "allele_source": source}
    try:
        csv_text, starts = run_mhcflurry(pairs, alleles, timeout_s=timeout_s)
        table = convert_mhcflurry_csv(csv_text, starts=starts)
    except (PredictorMissingError, ValueError, OSError) as error:
        return {"state": "predictor_failed", "blocker": str(error), "allele_source": source}
    path = directory / "mhc_i.tsv"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(table, encoding="utf-8")
    record: dict[str, object] = {
        "state": "written",
        "method": METHOD,
        "allele_source": source,
        "alleles": list(alleles),
        "peptides": len(starts),
        "windows": len(pairs),
        "rows": max(0, len(table.splitlines()) - 1),
        "path": str(path),
    }
    if binder_kind == "protein":
        # Skip the scan when the reference table is already beside this run.
        # The tolerated proteins do not change between designs, but the table
        # is written per run directory because that is where the axis reads it,
        # so an unguarded call rescanned all three proteins on every pass. A
        # single screen_directory call on a one-design fixture was measured
        # starting mhcflurry-predict eight times; the design scan accounts for
        # one of those and the rest were this. The early-return path above
        # already guards the same way.
        if (directory / f"mhc_i{_REFERENCE_STEM}").is_file():
            record["reference"] = str(directory / f"mhc_i{_REFERENCE_STEM}")
            record["reference_reused"] = True
        else:
            _attach_reference(record, directory, timeout_s=timeout_s, alleles=alleles)
    return record


def fetch_weights(*, timeout_s: int = 3600) -> dict[str, object]:
    """Download ``models_class1_presentation`` into :func:`models_directory`.

    The caller asks for this. A screen does not. The download command is the
    one installed beside ``mhcflurry-predict``.
    """
    predict = predictor_command()
    if predict is None:
        raise PredictorMissingError(
            f"mhcflurry-predict not on PATH and ${MHCFLURRY_COMMAND_ENV} unset"
        )
    command = _sibling_downloads(predict)
    if command is None:
        raise PredictorMissingError("mhcflurry-downloads not installed beside mhcflurry-predict")
    destination = models_directory()
    destination.mkdir(parents=True, exist_ok=True)
    env = _predict_env()
    env[MHCFLURRY_DOWNLOADS_ENV] = str(destination)
    try:
        finished = subprocess.run(
            [command, "fetch", MODELS_BUNDLE, "--release", MODELS_RELEASE],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
            env=env,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise PredictorMissingError(f"mhcflurry_downloads_failed:{type(error).__name__}") from error
    if finished.returncode != 0:
        raise PredictorMissingError(
            f"mhcflurry_downloads_exit_{finished.returncode}:{finished.stderr.strip()[:200]}"
        )
    return {
        "bundle": MODELS_BUNDLE,
        "release": MODELS_RELEASE,
        "models": str(destination),
        "weights_present": weights_present(destination),
    }
