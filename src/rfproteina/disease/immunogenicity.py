"""MHC class II, as far as a free-and-open program can honestly take it.

There is no permissively licensed class II binding predictor. That is the
finding, and this module is written around it rather than against it. Every
option that exists was checked and each one is either academic-use-only or
non-commercial; the licence text for each is quoted in
:data:`CLASS_II_OPTIONS` with the date it was read. Nothing academic-only is
wired in, nothing is shelled out to, and no class II table is ever written
from a guess.

What is here instead:

* **The interface.** :func:`register_backend` lets whoever holds a licence
  plug their own predictor in at runtime. The repository ships no backend, so
  :func:`predict_class_ii` raises :class:`ClassIIUnavailableError` out of the
  box, and the error names the options and their licences rather than just
  failing.
* **The unavailable path, documented.** :func:`class_ii_status` returns a
  record a screen can report: not available, why, what would change it, and
  what each alternative would cost in licence terms. An axis that reads
  ``untested`` because no predictor exists is a different thing from one that
  reads ``untested`` because a file is missing, and a reviewer should be able
  to tell them apart.
* **The converter.** Someone who runs a licensed predictor themselves gets
  back a CSV. :func:`write_class_ii_table` renames it onto the columns
  ``liability.py`` reads and warns about the two mistakes that would otherwise
  be silent: a rank on a scale the ADA threshold does not belong to, and an
  incomplete DR panel.

The ADA axis in ``liability.py`` is the one that consumes this. Its rule is
Paul et al. (2015): a peptide whose *median* IEDB consensus percentile rank
across the seven-allele DR panel is at most 20 is a promiscuous binder. The
threshold belongs to the consensus scale, so a NetMHCIIpan eluted-ligand rank
is not interchangeable with it -- ``liability._declares_consensus`` refuses a
table that does not say it is consensus, and this module writes the method
through rather than papering over it.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from rfproteina.disease.liability import _DR_MEDIAN_MAX, _DR_PANEL

#: The seven DR alleles the ADA rule is defined over, re-exported from
#: ``liability.py`` so a caller runs its predictor on exactly the panel the
#: threshold was validated on. A partial panel leaves the axis untested.
DR_PANEL: tuple[str, ...] = tuple(_DR_PANEL)
DR_MEDIAN_MAX: float = _DR_MEDIAN_MAX
#: Class II binding cores are 9-mers, presented in longer peptides, and class
#: II predictors are conventionally run over 15-mer windows. 15 is a
#: **suggested** default here, not a threshold from a publication.
CLASS_II_WIDTH = 15
STANDARD_RESIDUES = frozenset("ACDEFGHIKLMNPQRSTVWY")
#: Columns written, matching what ``liability.py`` reads for the class II and
#: ADA axes: a peptide, an allele, a rank, and a method that states its scale.
OUTPUT_COLUMNS = ("peptide", "allele", "percentile_rank", "ic50_nm", "unit", "method", "start")
#: What ``liability._declares_consensus`` looks for in the method column.
CONSENSUS_MARK = "consensus"


class ClassIIUnavailableError(Exception):
    """No class II predictor is registered. No table is written and nothing is guessed."""


@dataclass(frozen=True)
class ClassIIOption:
    """One class II predictor, and the licence that decides whether it can be used."""

    name: str
    licence: str
    licence_quote: str
    licence_read_on: str
    commercial_use: str
    rank_scale: str
    note: str


#: Every class II route found, with its licence quoted from the distributor on
#: the date given. None of these is installed or called by this repository.
CLASS_II_OPTIONS: tuple[ClassIIOption, ...] = (
    ClassIIOption(
        name="NetMHCIIpan 4.3 (DTU Health Tech)",
        licence="academic-use download; commercial use by separate arrangement",
        licence_quote=(
            "the service page states the software is on a 'download page for academic users; other users are "
            "requested to contact Health Tech Software Package Manager at health-software@dtu.dk'"
        ),
        licence_read_on="2026-11-16, https://services.healthtech.dtu.dk/services/NetMHCIIpan-4.3/",
        commercial_use="not granted",
        rank_scale="NetMHCIIpan EL and BA percentile ranks, bands 1 per cent strong and 5 per cent weak",
        note=(
            "Not the consensus scale the ADA threshold of 20 belongs to. Even with a licence, its rank cannot be "
            "compared against that threshold without recalibration."
        ),
    ),
    ClassIIOption(
        name="MixMHC2pred 2.0 (Ludwig Institute for Cancer Research)",
        licence="academic non-commercial",
        licence_quote=(
            "the repository LICENSE opens 'SOFTWARE LICENSE AGREEMENT FOR ACADEMIC NON-COMMERCIAL RESEARCH "
            "PURPOSES ONLY'"
        ),
        licence_read_on="2026-11-16, https://github.com/GfellerLab/MixMHC2pred/blob/master/LICENSE",
        commercial_use="expressly excluded",
        rank_scale="its own percentile rank",
        note="Excluded on licence grounds.",
    ),
    ClassIIOption(
        name="MHCnuggets 2.4.1 (Johns Hopkins University)",
        licence="JHU Academic Software License; non-commercial only",
        licence_quote=(
            "the repository LICENSE grants 'a non-exclusive, nontransferable license ... for academic, "
            "non-profit, or government-sponsored research purposes. Use of the Software under this License is "
            "restricted to non-commercial purposes.'"
        ),
        licence_read_on="2026-11-16, https://github.com/KarchinLab/mhcnuggets/blob/master/LICENSE",
        commercial_use="requires a separately executed written agreement",
        rank_scale="predicted IC50 in nM; no consensus rank",
        note=(
            "Worth flagging: the PyPI metadata for mhcnuggets says 'Apache License', which contradicts the "
            "LICENSE file in the repository. The repository file is the one that governs, and it is "
            "non-commercial. A licence read off PyPI alone would have wired this in by mistake."
        ),
    ),
    ClassIIOption(
        name="IEDB class II consensus tools",
        licence="mixed; the suite bundles NetMHCIIpan, which carries the DTU terms above",
        licence_quote="not quoted here: the bundle components carry different terms and were not all read",
        licence_read_on="not read in this work",
        commercial_use="unverified",
        rank_scale="IEDB consensus percentile rank -- the scale the ADA threshold of 20 is defined on",
        note=(
            "This is the only route whose rank scale matches the threshold already in liability.py. If the ADA "
            "axis is to reach a verdict, this is the one whose terms are worth reading properly."
        ),
    ),
    ClassIIOption(
        name="epytope 4.0.1 (BSD-3-Clause) / mhctools 4.9.5",
        licence="the framework is permissive",
        licence_quote="PyPI metadata gives epytope BSD-3-Clause and mhctools Apache-2.0 with bundled data licences",
        licence_read_on="2026-11-16, PyPI JSON metadata",
        commercial_use="granted for the framework itself",
        rank_scale="whatever the wrapped predictor writes",
        note=(
            "A permissive wrapper is not a permissive predictor. These call the tools above, so the licence that "
            "binds is still the predictor's. Installing one of these would not solve the problem."
        ),
    ),
)

_BACKENDS: dict[str, Callable[[Sequence[str], Sequence[str]], str]] = {}


def register_backend(name: str, backend: Callable[[Sequence[str], Sequence[str]], str]) -> None:
    """Register a class II predictor the caller holds a licence for.

    ``backend(peptides, alleles)`` returns a CSV with at least a peptide, an
    allele and a percentile rank. Registration is a runtime act by whoever has
    the licence; nothing in this repository calls it.
    """
    if not name or backend is None:
        raise ValueError("backend_name_and_callable_required")
    _BACKENDS[name] = backend


def registered_backends() -> list[str]:
    """Which backends this process has been given, if any."""
    return sorted(_BACKENDS)


def class_ii_status() -> dict[str, object]:
    """Whether a class II prediction is possible here, and what it would take.

    This is the documented unavailable path. It is deliberately verbose: the
    cost of a missing class II axis is that a design ships without an
    anti-drug-antibody read, and a record that only says ``untested`` hides
    that the reason is a licence rather than a missing file.
    """
    backends = registered_backends()
    return {
        "available": bool(backends),
        "backends": backends,
        "reason": None if backends else "no_permissively_licensed_class_ii_predictor",
        "panel": list(DR_PANEL),
        "panel_rule": (
            "Paul et al., J Immunol Methods 422:28-34 (2015): a peptide is promiscuous when its median IEDB "
            f"consensus percentile rank across the panel is at most {DR_MEDIAN_MAX}."
        ),
        "scale_required_for_ada": "IEDB consensus percentile rank",
        "options": [dict(option.__dict__) for option in CLASS_II_OPTIONS],
        "what_would_unblock": (
            "Either a licence for one of the options above plus register_backend(), or a class II table produced "
            "elsewhere and written with write_class_ii_table()."
        ),
    }


def windows(sequence: str, width: int = CLASS_II_WIDTH) -> list[tuple[int, str]]:
    """Class II windows as ``(1-based start, peptide)``, dropping non-standard residues."""
    if width <= 0:
        raise ValueError("width_must_be_positive")
    text = "".join((sequence or "").split()).upper()
    found: list[tuple[int, str]] = []
    for start in range(0, max(0, len(text) - width + 1)):
        piece = text[start : start + width]
        if set(piece) <= STANDARD_RESIDUES:
            found.append((start + 1, piece))
    return found


def predict_class_ii(
    sequence: str,
    alleles: Sequence[str] = DR_PANEL,
    *,
    backend: str | None = None,
    width: int = CLASS_II_WIDTH,
) -> str:
    """Run a registered backend over the windows. Raises when there is none.

    The raise carries the reason, so a caller that logs the exception logs why
    rather than just that.
    """
    backends = registered_backends()
    if not backends:
        raise ClassIIUnavailableError(
            "no_permissively_licensed_class_ii_predictor: every option is academic-use-only or non-commercial; "
            "see immunogenicity.CLASS_II_OPTIONS"
        )
    chosen = backend or backends[0]
    if chosen not in _BACKENDS:
        raise ClassIIUnavailableError(f"backend_not_registered:{chosen}")
    pairs = windows(sequence, width)
    if not pairs:
        raise ValueError("no_peptides")
    return _BACKENDS[chosen]([peptide for _start, peptide in pairs], list(alleles))


def convert_class_ii_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    method: str,
    starts: Mapping[str, int] | None = None,
) -> tuple[str, list[str]]:
    """Predictor rows -> the TSV ``liability.py`` reads, plus the warnings it will act on.

    Each row needs ``peptide``, ``allele`` and ``percentile_rank``; an
    ``ic50_nm`` is optional and is written with an explicit ``nM`` unit, never
    rescaled from anything else. The returned warnings are not cosmetic: they
    say in advance which axis will refuse the table and why, so a caller does
    not read a refusal as a clear.
    """
    if not rows:
        raise ValueError("no_rows")
    if not method:
        raise ValueError("method_required")
    warnings: list[str] = []
    if CONSENSUS_MARK not in method.casefold():
        warnings.append(
            f"ada_axis_will_report_rank_scale_unknown: the median threshold of {DR_MEDIAN_MAX} is defined on the "
            f"IEDB consensus rank, and this table declares method '{method}'"
        )
    written: list[dict[str, str]] = []
    for row in rows:
        peptide = str(row.get("peptide", "")).strip().upper()
        allele = str(row.get("allele", "")).strip()
        rank = row.get("percentile_rank", row.get("rank", ""))
        if not peptide or not allele:
            raise ValueError("peptide_and_allele_required")
        affinity = row.get("ic50_nm", "")
        has_affinity = affinity not in (None, "")
        written.append(
            {
                "peptide": peptide,
                "allele": allele,
                "percentile_rank": "" if rank is None else str(rank).strip(),
                "ic50_nm": str(affinity).strip() if has_affinity else "",
                "unit": "nM" if has_affinity else "",
                "method": method,
                "start": str(starts.get(peptide, "")) if starts else "",
            }
        )
    covered = {row["allele"] for row in written}
    missing = [allele for allele in DR_PANEL if allele not in covered]
    if missing:
        warnings.append(f"ada_axis_will_report_allele_panel_incomplete: missing {', '.join(missing)}")
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=list(OUTPUT_COLUMNS), delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(written)
    return out.getvalue(), warnings


def write_class_ii_table(
    rows: Sequence[Mapping[str, object]],
    directory: Path,
    *,
    method: str,
    starts: Mapping[str, int] | None = None,
    stem: str = "mhc_ii",
) -> dict[str, object]:
    """Write ``mhc_ii.tsv`` beside a run, and report the warnings with it."""
    table, warnings = convert_class_ii_rows(rows, method=method, starts=starts)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{stem}.tsv"
    path.write_text(table, encoding="utf-8")
    return {"path": str(path), "rows": max(0, len(table.splitlines()) - 1), "method": method, "warnings": warnings}
