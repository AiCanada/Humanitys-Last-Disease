"""TEPITOPE class II binding scores, in Python 3.

The matrices are the Unlicense release at
https://github.com/BenPorebski/tepitope. ``scripts/tepitope.py`` is that
repository's command-line script ported to Python 3. Each allele is a 9-column positional
profile. A numerical score is also published as the cutoff for the top 1%
through 10% of that profile. This module reports that bin.

It does not report an IEDB consensus percentile rank. The ADA axis in
``liability.py`` applies Paul et al. 2015 only to a table whose method says
consensus, so a method of ``tepitope`` leaves that axis untested. The class II
binding axis can still read the bin: its rank cutoff of 10 then means the
TEPITOPE 10% threshold, which is this tool's own scale.
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from rfproteina.disease.immunogenicity import DR_PANEL, windows
from rfproteina.disease.mhc_tables import reference_sequences
from rfproteina.disease.screening_tools import data_dir

METHOD = "tepitope"
#: Below every published threshold. It is not an IEDB rank. It is above the
#: class II rank cutoff, so the peptide is scored and is not called a binder.
_BELOW_TOP_DECILE = 100
_CORE = 9
_STANDARD = frozenset("ACDEFGHIKLMNPQRSTVWY")


@dataclass(frozen=True)
class Matrix:
    """One allele profile and the percent bins published with it."""

    allele: str
    weights: dict[str, tuple[float, ...]]
    thresholds: tuple[tuple[int, float], ...]

    def score(self, core: str) -> float:
        """Sum the nine positions. An unknown residue contributes nothing."""
        total = 0.0
        for index, residue in enumerate(core):
            column = self.weights.get(residue)
            if column is None:
                continue
            total += column[index]
        return total

    def percentile_bin(self, score: float) -> int:
        """The tightest published percent this score meets, or 100 when none."""
        met = [percent for percent, cutoff in self.thresholds if score >= cutoff]
        if not met:
            return _BELOW_TOP_DECILE
        return min(met)


def matrices_dir() -> Path:
    """``DATA_PATH/tepitope/matrices``, or ``data/tepitope/matrices`` in the checkout.

    The checkout is found by searching upward for ``.project-root``, not by a
    fixed parent count, which would point at the wrong directory without
    raising if this module ever changed depth.
    """
    return data_dir("tepitope", "matrices")


def allele_of(stem: str) -> str:
    """``HLA-DRB1_0301`` to the panel spelling ``DRB1*03:01``."""
    text = stem[4:] if stem.upper().startswith("HLA-") else stem
    if "_" not in text:
        return text
    gene, digits = text.split("_", 1)
    if len(digits) == 4 and digits.isdigit():
        return f"{gene}*{digits[:2]}:{digits[2:]}"
    return text.replace("_", "*")


def load_matrix(path: Path) -> Matrix:
    """One matrix file. A bare ``-`` in a position is an unscored pocket, added as zero."""
    weights: dict[str, tuple[float, ...]] = {}
    percents: list[int] = []
    scores: list[float] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        cells = [cell.strip() for cell in raw.split(",")]
        label = cells[0]
        if label.endswith(":") and len(label) == 2 and label[0] in _STANDARD:
            values: list[float] = []
            for cell in cells[1:10]:
                if cell in {"", "-"}:
                    values.append(0.0)
                else:
                    values.append(float(cell))
            if len(values) != _CORE:
                raise ValueError(f"matrix_width:{path.name}")
            weights[label[0]] = tuple(values)
        elif label == "Percent Threshold":
            percents = [int(cell.replace("%", "")) for cell in cells[1:] if cell]
        elif label == "Numerical Score":
            scores = [float(cell) for cell in cells[1:] if cell]
    if not weights or len(percents) != len(scores) or not percents:
        raise ValueError(f"matrix_unreadable:{path.name}")
    return Matrix(allele_of(path.stem), weights, tuple(zip(percents, scores, strict=True)))


def load_panel(directory: Path | None = None) -> tuple[list[Matrix], list[str]]:
    """Matrices for the DR panel, and the panel alleles that have none."""
    folder = matrices_dir() if directory is None else directory
    by_allele: dict[str, Matrix] = {}
    if folder.is_dir():
        for path in sorted(folder.glob("*.csv")):
            matrix = load_matrix(path)
            by_allele.setdefault(matrix.allele, matrix)
    present = [by_allele[name] for name in DR_PANEL if name in by_allele]
    missing = [name for name in DR_PANEL if name not in by_allele]
    return present, missing


def _best_core(matrix: Matrix, peptide: str) -> tuple[str, float]:
    """The highest-scoring 9-mer core inside ``peptide``."""
    best_core = peptide[:_CORE]
    best_score = matrix.score(best_core)
    for start in range(1, len(peptide) - _CORE + 1):
        core = peptide[start : start + _CORE]
        score = matrix.score(core)
        if score > best_score:
            best_core, best_score = core, score
    return best_core, best_score


def score_sequence(sequence: str, matrices: list[Matrix]) -> list[dict[str, str]]:
    """One row per 15-mer and allele. The peptide is the 15-mer, not the core."""
    rows: list[dict[str, str]] = []
    for start, peptide in windows(sequence):
        for matrix in matrices:
            _core, score = _best_core(matrix, peptide)
            rows.append(
                {
                    "peptide": peptide,
                    "allele": matrix.allele,
                    "percentile_rank": str(matrix.percentile_bin(score)),
                    "method": METHOD,
                    "start": str(start),
                    "tepitope_score": f"{score:.4f}",
                }
            )
    return rows


def _write(path: Path, rows: list[dict[str, str]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t", lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_class_ii_reference(directory: Path, matrices: Sequence[object]) -> dict[str, object]:
    """Score the tolerated proteins into ``mhc_ii_reference.tsv``.

    Shared by both paths through :func:`prepare_class_ii_table`, so a run that
    finds a design table already present still gets the reference table it
    needs. The status is returned either way, so a caller can tell a reference
    that is absent from one that could not be written.
    """
    status, proteins = reference_sequences()
    if status != "ready":
        return {"reference": status}
    rows: list[dict[str, str]] = []
    for accession, sequence in proteins.items():
        for row in score_sequence(sequence, matrices):
            row["protein"] = accession
            row["length"] = str(len(sequence))
            rows.append(row)
    reference = directory / "mhc_ii_reference.tsv"
    _write(
        reference,
        rows,
        ["peptide", "allele", "percentile_rank", "method", "protein", "start", "length", "tepitope_score"],
    )
    return {"reference": str(reference)}


def prepare_class_ii_table(
    directory: Path,
    sequences: list[tuple[str, str | None]],
    *,
    binder_kind: str,
) -> dict[str, object]:
    """Write ``mhc_ii.tsv`` when the matrices are on disk and no table is there yet.

    An existing table is kept. The method is ``tepitope``, which does not
    declare the IEDB consensus scale, so the ADA axis stays untested.
    """
    directory = Path(directory)
    existing = directory / "mhc_ii.tsv"
    if existing.is_file():
        record: dict[str, object] = {"state": "table_present", "path": str(existing)}
        # The design table is kept, but the reference table is a separate
        # artifact: a run made before the reference sequences were fetched
        # left none, and the early return above then suppressed the reference
        # scan on every later run, pinning the class II axis at
        # reference_not_configured even once the sequences were present.
        if binder_kind == "protein" and not (directory / "mhc_ii_reference.tsv").is_file():
            matrices, _missing = load_panel()
            if matrices:
                record.update(_write_class_ii_reference(directory, matrices))
        return record
    matrices, missing = load_panel()
    if not matrices:
        return {"state": "matrices_missing", "missing_alleles": missing, "models": str(matrices_dir())}
    rows: list[dict[str, str]] = []
    for _name, sequence in sequences:
        if sequence:
            rows.extend(score_sequence(sequence, matrices))
    if not rows:
        return {"state": "no_peptides", "missing_alleles": missing}
    _write(
        existing,
        rows,
        ["peptide", "allele", "percentile_rank", "method", "start", "tepitope_score"],
    )
    record: dict[str, object] = {
        "state": "written",
        "method": METHOD,
        "rows": len(rows),
        "alleles": [matrix.allele for matrix in matrices],
        "missing_alleles": missing,
        "path": str(existing),
        "ada": "untested_not_consensus_rank",
    }
    if binder_kind == "protein":
        record.update(_write_class_ii_reference(directory, matrices))
    return record
