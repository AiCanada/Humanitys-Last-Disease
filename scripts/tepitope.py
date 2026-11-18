#!/usr/bin/env python3
"""Python 3 port of BenPorebski/tepitope ``tepitope.py`` (The Unlicense).

The upstream file is Python 2: ``print`` statements, ``print >>``, a ``map``
stored and later indexed, and a CSV opened in binary mode. Those are the
changes. The upstream loop also stopped at ``len(sequence) - 9`` and dropped
the final 9-mer. Every 9-mer is scored.

Matrices are ``--matrices``, then ``./matrices``, then the ``matrices``
directory beside this file. ``scripts/install_tepitope.py`` puts them in
``DATA_PATH/tepitope/matrices``.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import sys
from pathlib import Path


class HLAmatrix:
    """One TEPITOPE positional matrix and the percent threshold it was opened at."""

    def __init__(self, csv_file: str, threshold: int) -> None:
        self.matrix_data: dict[str, list[float]] = {}
        self.threshold = threshold
        self.threshold_score = 0.0
        self.matrix_file = csv_file
        self.allele = os.path.splitext(os.path.basename(csv_file))[0]
        self.load_matrix(csv_file)

    def load_matrix(self, matrix_file: str) -> None:
        """Load the residue rows and the numerical score at ``self.threshold``."""
        percentage_threshold_tmp: list[str] = []
        scoring_tmp: list[str] = []
        with open(matrix_file, encoding="utf-8") as matrix_handle:
            for line in matrix_handle:
                # A residue label is one letter and a colon. "-" is an unscored
                # pocket; the upstream rewrite turns it into "-0", which float()
                # accepts, and turns "-1.30" into "-01.30", which float() also
                # accepts. Kept so a matrix scores the same number as before.
                if len(line.split(",")[0]) <= 2:
                    parsed = line.replace("-", "-0")
                    cells = parsed.strip().split(",")[1:]
                    self.matrix_data[parsed.split(",")[0][0]] = [float(cell) for cell in cells]
                if line.split(",")[0] == "Percent Threshold":
                    percentage_threshold_tmp = line.rstrip().split(",")[1:]
                if line.split(",")[0] == "Numerical Score":
                    scoring_tmp = line.rstrip().split(",")[1:]
        for index, threshold in enumerate(percentage_threshold_tmp):
            thresh_clean = threshold.split("%")[0]
            if int(thresh_clean) == self.threshold:
                self.threshold_score = float(scoring_tmp[index])

    def score(self, nonamer: str) -> tuple[str, float]:
        """Sum one 9-mer. A residue the matrix does not list is skipped."""
        total = 0.0
        for index, residue in enumerate(nonamer):
            try:
                total += self.matrix_data[residue][index]
            except (KeyError, IndexError):
                continue
        return nonamer, total

    def score_sequence(self, sequence: str) -> list[tuple[str, float, int, int]]:
        """9-mer windows at or above the threshold, highest score first."""
        returned: list[tuple[str, float, int, int]] = []
        for start in range(len(sequence) - 8):
            peptide, value = self.score(sequence[start : start + 9])
            if value >= self.threshold_score:
                returned.append((peptide, value, start, start + 8))
        return sorted(returned, key=lambda item: item[1], reverse=True)


def _matrices(explicit: str | None) -> list[str]:
    if explicit:
        found = sorted(glob.glob(str(Path(explicit) / "*.csv")))
    else:
        found = sorted(glob.glob("matrices/*.csv"))
        if not found:
            beside = Path(__file__).resolve().parent / "matrices"
            found = sorted(glob.glob(str(beside / "*.csv")))
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TEPITOPE MHC class II binding scores.")
    parser.add_argument("-t", "--threshold", dest="threshold", type=int, default=5)
    parser.add_argument("-c", "--csv", dest="csv_filename", default=None, help="filename for CSV output")
    parser.add_argument(
        "--no-header",
        dest="csv_header",
        action="store_false",
        help="omit the header row from the CSV file",
    )
    parser.add_argument(
        "-f",
        "--file",
        "--sequence-file",
        dest="filename",
        default=None,
        help="file which contains the sequence",
    )
    parser.add_argument(
        "-i",
        "--starting-index",
        dest="base_index",
        default=1,
        type=int,
        help="the index of the first peptide in the sequence",
    )
    parser.add_argument(
        "--matrices",
        default=None,
        help="directory of allele CSV files (default: ./matrices, then beside this script)",
    )
    parser.add_argument("sequence", default=None, nargs="?", help="the peptide sequence to analyze, if not using -f")
    args = parser.parse_args(argv)

    if args.sequence is not None and args.filename is not None:
        print(
            "You must specify only one of -f, --file, --sequence-file, or the sequence as a command-line argument",
            file=sys.stderr,
        )
        return 1
    if args.sequence is None and args.filename is None:
        print(
            "You must specify -f, --file, --sequence-file, or put the sequence as a command-line argument",
            file=sys.stderr,
        )
        return 1
    if args.sequence:
        sequence = args.sequence
    else:
        with open(args.filename, encoding="utf-8") as handle:
            sequence = re.sub(r"\s", "", handle.read())

    matrix_files = _matrices(args.matrices)
    if not matrix_files:
        print("No matrix CSV files found. Pass --matrices or run from a directory that contains matrices/.", file=sys.stderr)
        return 1

    results: set[tuple[float, int, int, str, str]] = set()
    for matrix_file in matrix_files:
        matrix = HLAmatrix(matrix_file, args.threshold)
        for epitope, score, begin, end in matrix.score_sequence(sequence):
            results.add((score, begin + args.base_index, end + args.base_index, epitope, matrix.allele))
    ordered = sorted(results, key=lambda row: row[0], reverse=True)

    if args.csv_filename is not None:
        with open(args.csv_filename, "w", encoding="utf-8", newline="") as csv_file:
            out = csv.writer(csv_file, lineterminator=os.linesep)
            if args.csv_header:
                out.writerow(("score", "begin", "end", "epitope", "allele"))
            for result in ordered:
                out.writerow(result)
    else:
        for score, begin, end, epitope, allele in ordered:
            print(f"{score:.2f}\t{begin}\t{end}\t{epitope}\t{allele}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
