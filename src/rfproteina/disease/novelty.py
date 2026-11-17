"""Structural novelty of a design against known structures, via Foldseek.

A generator can rediscover a protein that already exists. This screen asks, for
each design structure, "is there a known structure that this essentially is?" It
runs Foldseek's ``easy-search`` over the design against a structure database
(the PDB, or an AlphaFold set) and reads back the best match's TM-score.

The verdict is deliberately a reported one, not a hard gate. A design whose best
match reaches the same-fold TM-score of 0.5 is marked ``hit`` to mean *a known
structure of this fold exists*, and one below it ``clear`` to mean *structurally
novel*. Whether "known" is bad depends on the campaign -- a scaffolded binder is
meant to reuse a fold -- so the integration note asks for this to be surfaced as
an informational axis rather than wired into a pass bar. The 0.5 same-fold cut is
cited; everything else is labelled a suggestion.

Nothing here downloads a database. Foldseek (GPL-3.0) and its pre-built
databases (CC-BY-4.0) are fetched by an operator with ``foldseek databases``;
the sizes are in :data:`FOLDSEEK_DATABASES` and in the handoff. If Foldseek is
not on PATH, or the database path is absent, the screen returns ``untested`` with
a blocker that names what is missing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rfproteina.disease.combinations import rollup_axis
from rfproteina.disease.screening_tools import (
    CLEAR,
    HIT,
    UNTESTED,
    ScreeningInputError,
    axis_record,
    run_tool,
    tool_available,
)

FOLDSEEK = "foldseek"

#: Structure file extensions Foldseek reads as a query. Compression is handled by
#: Foldseek itself, so the gzipped forms are listed too.
STRUCTURE_SUFFIXES = (".pdb", ".cif", ".mmcif", ".pdb.gz", ".cif.gz", ".mmcif.gz", ".bcif")

#: The columns requested from Foldseek, in order. ``alntmscore`` is the novelty
#: signal; the rest let a reviewer trace a flag back to the matched entry.
SEARCH_COLUMNS = ("query", "target", "fident", "alntmscore", "evalue", "bits")

#: TM-score at or above which two structures share a fold. Xu & Zhang,
#: Bioinformatics 26(7):889-895 (2010): a TM-score > 0.5 corresponds to the same
#: fold with high probability, independent of length. Used here as the cut above
#: which a design is "a known structure", i.e. not novel.
SAME_FOLD_TM = 0.5

#: Thresholds with provenance. ``cited`` names a source; ``suggested`` is this
#: program's default pending calibration on its own designs.
NOVELTY_THRESHOLDS: dict[str, dict[str, object]] = {
    "same_fold_tmscore": {
        "value": SAME_FOLD_TM,
        "status": "cited",
        "source": "Xu & Zhang, Bioinformatics 26(7):889-895 (2010): same-fold TM-score > 0.5",
    },
    "evalue_max": {
        "value": 1e-3,
        "status": "suggested",
        "source": "program default pending calibration; Foldseek's own default E-value is 10",
    },
}

#: Free, operator-fetched structure databases, with sizes read from
#: foldseek.steineggerlab.workers.dev on 2026-11-16 and their download commands.
#: Download size, not extracted. All CC-BY-4.0. Nothing here is fetched by code.
FOLDSEEK_DATABASES: dict[str, dict[str, str]] = {
    "PDB": {"command": "foldseek databases PDB pdb tmp", "download": "pdb100.tar.gz 2.2 GB", "licence": "CC-BY-4.0"},
    "Alphafold/Swiss-Prot": {
        "command": "foldseek databases Alphafold/Swiss-Prot afdb_sp tmp",
        "download": "afdb_swissprot.tar.gz 1.5 GB",
        "licence": "CC-BY-4.0",
    },
    "Alphafold/Proteome": {
        "command": "foldseek databases Alphafold/Proteome afdb_prot tmp",
        "download": "afdb_proteome.tar.gz 1.7 GB",
        "licence": "CC-BY-4.0",
    },
    "Alphafold/UniProt50": {
        "command": "foldseek databases Alphafold/UniProt50 afdb50 tmp",
        "download": "afdb50.tar.gz 114.1 GB (large; state before fetching)",
        "licence": "CC-BY-4.0",
    },
}


@dataclass(frozen=True)
class StructureHit:
    """The best structural match found for one design."""

    query: str
    target: str
    fident: float
    tmscore: float
    evalue: float
    bits: float


class NoveltyUnavailableError(RuntimeError):
    """Foldseek or its database is absent. Carries the blocker to record as untested."""

    def __init__(self, blocker: str) -> None:
        super().__init__(blocker)
        self.blocker = blocker


def foldseek_available() -> bool:
    """Whether ``foldseek`` is on PATH. Checked, never assumed."""
    return tool_available(FOLDSEEK)


def query_structures(query_dir: Path) -> list[Path]:
    """Every structure file under ``query_dir`` Foldseek can read, sorted by name."""
    found: list[Path] = []
    for path in sorted(query_dir.rglob("*")):
        if not path.is_file():
            continue
        name = path.name.lower()
        if any(name.endswith(suffix) for suffix in STRUCTURE_SUFFIXES):
            found.append(path)
    return found


def build_search_command(
    query: Path,
    database: Path,
    alignment_out: Path,
    tmp_dir: Path,
    *,
    foldseek: str = FOLDSEEK,
    evalue_max: float = 1e-3,
) -> list[str]:
    """The ``foldseek easy-search`` argument list. ``--alignment-type 1`` gives a TM-score."""
    return [
        foldseek,
        "easy-search",
        str(query),
        str(database),
        str(alignment_out),
        str(tmp_dir),
        "--alignment-type",
        "1",
        "-e",
        str(evalue_max),
        "--format-output",
        ",".join(SEARCH_COLUMNS),
    ]


def parse_search_table(text: str, columns: tuple[str, ...] = SEARCH_COLUMNS) -> dict[str, StructureHit]:
    """Best structural match per query from a Foldseek tab-separated table.

    "Best" is the highest TM-score. A row whose numeric fields do not parse is
    skipped rather than read as a zero, which would look like a confident
    non-match. The query key is the design's own id as Foldseek reports it.
    """
    index = {name: position for position, name in enumerate(columns)}
    best: dict[str, StructureHit] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        fields = line.split("\t")
        if len(fields) < len(columns):
            continue
        query = fields[index["query"]]
        try:
            hit = StructureHit(
                query=query,
                target=fields[index["target"]],
                fident=float(fields[index["fident"]]),
                tmscore=float(fields[index["alntmscore"]]),
                evalue=float(fields[index["evalue"]]),
                bits=float(fields[index["bits"]]),
            )
        except (KeyError, ValueError):
            continue
        current = best.get(query)
        if current is None or hit.tmscore > current.tmscore:
            best[query] = hit
    return best


def _query_id(path: Path) -> str:
    """The id Foldseek reports for a query file: its name without a structure suffix."""
    name = path.name
    for suffix in sorted(STRUCTURE_SUFFIXES, key=len, reverse=True):
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def verdict(best: StructureHit | None, *, same_fold_tm: float = SAME_FOLD_TM) -> dict[str, object]:
    """One design's novelty record from its best match.

    A match at or above the same-fold TM-score is a hit meaning *not novel*; a
    weaker best match, or none, is clear meaning *structurally novel*. The
    measurement is the best TM-score so a roll-up reads the least-novel design.
    """
    if best is None:
        return axis_record(CLEAR, 0.0, None, novel=True, best_target=None, reason="no_match_within_evalue")
    state = HIT if best.tmscore >= same_fold_tm else CLEAR
    return axis_record(
        state,
        round(best.tmscore, 4),
        None,
        novel=best.tmscore < same_fold_tm,
        best_target=best.target,
        fident=round(best.fident, 4),
        evalue=best.evalue,
        bits=best.bits,
    )


def novelty_search(
    query_dir: Path,
    database: Path,
    work_dir: Path,
    *,
    foldseek: str = FOLDSEEK,
    same_fold_tm: float = SAME_FOLD_TM,
    evalue_max: float = 1e-3,
    timeout_s: float = 3600.0,
) -> dict[str, dict[str, object]]:
    """A novelty record per design structure under ``query_dir``.

    Raises :class:`ScreeningInputError` when there is no structure to search.
    Raises :class:`NoveltyUnavailableError` when Foldseek is not installed or the
    database path does not exist, so a caller records every design ``untested``
    with that one blocker instead of a made-up score.
    """
    query_dir = Path(query_dir)
    structures = query_structures(query_dir)
    if not structures:
        raise ScreeningInputError("no_query_structures")
    if not foldseek_available():
        raise NoveltyUnavailableError("foldseek_not_installed")
    if not Path(database).exists():
        raise NoveltyUnavailableError("foldseek_database_missing")
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    alignment_out = work_dir / "novelty.m8"
    tmp_dir = work_dir / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    command = build_search_command(
        query_dir, Path(database), alignment_out, tmp_dir, foldseek=foldseek, evalue_max=evalue_max
    )
    result = run_tool(command, timeout_s=timeout_s)
    if not result.ok:
        raise NoveltyUnavailableError(result.blocker or "foldseek_failed")
    text = alignment_out.read_text(encoding="utf-8") if alignment_out.is_file() else result.stdout
    best = parse_search_table(text)
    records: dict[str, dict[str, object]] = {}
    for path in structures:
        design_id = _query_id(path)
        records[design_id] = verdict(best.get(design_id), same_fold_tm=same_fold_tm)
    return records


def rollup(records: dict[str, dict[str, object]]) -> dict[str, object]:
    """One novelty axis for the whole batch, by the shared roll-up rule."""
    empty = axis_record(UNTESTED, None, "designs_absent", novel=None, best_target=None)
    return rollup_axis(list(records), records, empty)


def unavailable_axis(blocker: str) -> dict[str, object]:
    """The axis a caller records when the search could not run at all."""
    return axis_record(UNTESTED, None, blocker, novel=None, best_target=None)


def database_requirements() -> dict[str, dict[str, str]]:
    """The free databases this screen can use, with sizes, licences and commands."""
    return dict(FOLDSEEK_DATABASES)
