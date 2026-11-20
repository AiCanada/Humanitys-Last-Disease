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
databases (CC-BY-4.0) are fetched by an operator -- ``scripts/install_foldseek.py``
for the binary and for one named database, or ``foldseek databases`` by hand;
the sizes are in :data:`FOLDSEEK_DATABASES` and in the handoff. If no Foldseek
can be resolved, or the database path is absent, the screen returns ``untested``
with a blocker that names what is missing.

The binary is ``FOLDSEEK``, then ``DATA_PATH/foldseek``, then the checkout's
``screening subpackage/foldseek``, then ``PATH`` -- see :func:`foldseek_command`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from rfproteina.disease.combinations import rollup_axis
from rfproteina.disease.screening_tools import (
    CLEAR,
    HIT,
    UNTESTED,
    ScreeningInputError,
    axis_record,
    resolve_tool,
    run_tool,
)

FOLDSEEK = "foldseek"

#: An executable path, or a bare command on PATH. It wins over any installed copy.
FOLDSEEK_ENV = "FOLDSEEK"

#: Where the binary sits inside ``<root>/foldseek``. The official release tarball
#: unpacks to ``foldseek/bin/foldseek``; ``scripts/install_foldseek.py`` lifts it
#: out to ``foldseek/foldseek``, so both spellings are tried.
FOLDSEEK_RELATIVES = ("foldseek", "bin/foldseek")

#: Structure file extensions Foldseek reads as a query. Compression is handled by
#: Foldseek itself, so the gzipped forms are listed too.
STRUCTURE_SUFFIXES = (".pdb", ".cif", ".mmcif", ".pdb.gz", ".cif.gz", ".mmcif.gz", ".bcif")

#: The columns requested from Foldseek, in order. ``qtmscore`` and ``ttmscore``
#: are the novelty signal: TM-scores normalised by the query chain's own length
#: and by the target chain's own length, which is the normalisation Xu & Zhang's
#: same-fold statement concerns.
#:
#: ``alntmscore``, requested here before, normalises by the *aligned span*
#: instead -- Foldseek's own ``--tmscore-threshold-mode`` names the three as
#: "0: alignment, 1: query 2: target length". A match covering half a design
#: therefore read near 1.0 on it (values above 1.0 were observed, which a
#: TM-score cannot take), so the cited 0.5 cut was being applied to a quantity
#: the citation is not about. ``metrics/cath_metric.py`` already asks for these
#: two columns and scores on the larger of them; this screen was the outlier.
SEARCH_COLUMNS = ("query", "target", "fident", "qtmscore", "ttmscore", "evalue", "bits")

#: TM-score at or above which two structures share a fold. Xu & Zhang,
#: Bioinformatics 26(7):889-895 (2010): a TM-score > 0.5 corresponds to the same
#: fold with high probability, independent of length. Used here as the cut above
#: which a design is "a known structure", i.e. not novel.
#:
#: The cited statement is strict and :func:`verdict` is inclusive: a best match
#: of exactly 0.5 is a hit here and is not one under the paper's wording. The
#: difference is a single point on a continuous measure and no stored verdict
#: has ever turned on it, but it is the program's boundary rather than the
#: paper's and is recorded as such -- the same fidelity the instability cut is
#: held to, where ProtParam says "smaller than 40" and the code tests ``< 40``.
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
        "value": 1e6,
        "status": "suggested",
        "source": (
            "deliberately permissive and not calibrated. With --alignment-type 1 Foldseek applies "
            "-e at its structurealign stage, which runs before TMalign, so a tight cut decides this "
            "axis instead of the TM-score that is meant to. At the former 1e-3 an atom-exact copy of "
            "a PDB entry returned no rows at all and read novel; Foldseek's own default of 10 is "
            "still too tight for that case. metrics/foldseek_novelty.py searches at 1000000 for the "
            "same reason, so this matches the convention already in the repository."
        ),
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


def foldseek_command() -> str | None:
    """The Foldseek executable, or None when this machine has none.

    ``FOLDSEEK`` wins, then a copy under ``DATA_PATH/foldseek`` (where
    ``scripts/install_foldseek.py`` puts it) or the checkout's
    ``screening subpackage/foldseek``, then ``PATH`` -- the precedence
    :func:`~rfproteina.disease.screening_tools.resolve_tool` holds for every
    external-tool screen.

    This screen used to ask ``PATH`` alone, which is not the same question: an
    installed Foldseek that was not also on ``PATH`` left the novelty axis
    reporting ``foldseek_not_installed`` with the binary on disk.
    """
    return resolve_tool(FOLDSEEK, FOLDSEEK_ENV, *FOLDSEEK_RELATIVES)


def foldseek_available() -> bool:
    """Whether a Foldseek executable can be resolved. Checked, never assumed."""
    return foldseek_command() is not None


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
    evalue_max: float = 1e6,
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
                tmscore=max(float(fields[index["qtmscore"]]), float(fields[index["ttmscore"]])),
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
    """A query file's design id: its name without a structure suffix.

    Not necessarily the id Foldseek reports. ``createdb`` names an entry after
    the file and appends the chain whenever the file holds more than one, so a
    two-chain design comes back as ``<design_id>_A`` and ``<design_id>_E`` and
    never as ``<design_id>``. :func:`best_per_design` maps the reported ids
    back onto the ids this returns.
    """
    name = path.name
    for suffix in sorted(STRUCTURE_SUFFIXES, key=len, reverse=True):
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def best_per_design(best: dict[str, StructureHit], design_ids: Sequence[str]) -> dict[str, StructureHit | None]:
    """The strongest match reported for each design, across every chain of it.

    Reading the table by file id alone matched a single-chain query and nothing
    else, so every multi-chain design -- which is what the PPI pipeline writes
    -- was recorded ``clear, novel`` because nothing had been *looked up*,
    rather than because nothing was similar. The same chain scored 0.976 on its
    own and 0.0 inside its own complex.

    A reported id is attributed to the longest design id it equals or extends
    with an underscore, so a design whose own name ends in ``_something`` keeps
    its rows. The best TM-score over a complex's chains is kept, which reads the
    least novel chain -- the direction :func:`rollup` already takes.

    An id belonging to no query file is a broken naming assumption, not an
    absent match, so it raises rather than being dropped into silence.
    """
    matched: dict[str, StructureHit | None] = {design_id: None for design_id in design_ids}
    for query, hit in best.items():
        candidates = [name for name in matched if query == name or query.startswith(f"{name}_")]
        if not candidates:
            raise NoveltyUnavailableError(f"foldseek_query_id_unrecognized:{query}")
        design_id = max(candidates, key=len)
        current = matched[design_id]
        if current is None or hit.tmscore > current.tmscore:
            matched[design_id] = hit
    return matched


def verdict(best: StructureHit | None, *, same_fold_tm: float = SAME_FOLD_TM) -> dict[str, object]:
    """One design's novelty record from its best match.

    A match at or above the same-fold TM-score is a hit meaning *not novel*; a
    weaker best match, or none, is clear meaning *structurally novel*. The
    measurement is the best TM-score so a roll-up reads the least-novel design.

    ``matched_entry`` names the Foldseek query entry the match came from, which
    for a multi-chain file carries the chain. **Read it before reading the
    state.** A file holding a binder and its target is answered by whichever
    chain matches best, and a target matches its own PDB entry, so a real IL33
    binder complex reports ``hit`` at 0.969 against ``4kc3`` on its *target*
    chain while saying nothing about the binder. Scoring binder novelty means
    searching the chain of interest alone; this field records which chain
    answered, so that distinction is visible rather than implied.
    """
    if best is None:
        return axis_record(
            CLEAR, 0.0, None, novel=True, best_target=None, matched_entry=None, reason="no_match_within_evalue"
        )
    state = HIT if best.tmscore >= same_fold_tm else CLEAR
    return axis_record(
        state,
        round(best.tmscore, 4),
        None,
        novel=best.tmscore < same_fold_tm,
        best_target=best.target,
        matched_entry=best.query,
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
    evalue_max: float = 1e6,
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
    if foldseek == FOLDSEEK:
        # Run the binary this machine actually has, which may be an installed
        # copy that is not on PATH. Falling back to the bare name keeps the
        # behaviour a caller gets when availability has been stubbed.
        foldseek = foldseek_command() or FOLDSEEK
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
    design_ids = [_query_id(path) for path in structures]
    matched = best_per_design(best, design_ids)
    records: dict[str, dict[str, object]] = {}
    for design_id in design_ids:
        records[design_id] = verdict(matched[design_id], same_fold_tm=same_fold_tm)
    return records


def rollup(records: dict[str, dict[str, object]]) -> dict[str, object]:
    """One novelty axis for the whole batch, by the shared roll-up rule."""
    empty = axis_record(UNTESTED, None, "designs_absent", novel=None, best_target=None, matched_entry=None)
    return rollup_axis(list(records), records, empty)


def unavailable_axis(blocker: str) -> dict[str, object]:
    """The axis a caller records when the search could not run at all."""
    return axis_record(UNTESTED, None, blocker, novel=None, best_target=None, matched_entry=None)


def database_requirements() -> dict[str, dict[str, str]]:
    """The free databases this screen can use, with sizes, licences and commands."""
    return dict(FOLDSEEK_DATABASES)
