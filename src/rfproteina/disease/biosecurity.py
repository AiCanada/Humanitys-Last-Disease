"""Biosecurity screening of synthesis constructs, via Commec (the Common Mechanism).

Before a designed construct is ordered as DNA, it should be screened for sequences
of concern -- toxins and regulated-pathogen genes. This stage wraps Commec, the
International Biosecurity and Biosafety Initiative for Science (IBBIS) Common
Mechanism: a free, open-source (MIT), locally run screener that transfers no data
off the machine.

``commec screen`` runs a nucleotide FASTA through a biorisk HMM search, a
regulated-pathogen taxonomy search (BLASTX/DIAMOND against NCBI nr and BLASTN
against core_nt), and a low-concern clear-back step, writing one
``<record>.screen.json`` per sequence. ``commec flag`` collates those into a CSV
whose recommendation per record is ``Pass``, ``Flag`` or ``Warn``. This module
maps those onto axis states and rolls them up.

The mapping is conservative on purpose: ``Warn`` is treated as a hit, not a clear,
because for a biosecurity gate a missed concern is the costly error. The mapping
is a module constant (:data:`RECOMMENDATION_STATE`) so a reviewer can see and
change it.

Nothing here downloads a database. The biorisk and low-concern databases are
small (IBBIS states the full Commec download is under 10 GB) and fetched with
``commec setup``; the regulated-taxonomy step additionally needs the NCBI ``nr``
and ``core_nt`` databases, which are large and operator-provided -- run with
``--skip-tx`` to screen on the biorisk and low-concern databases alone. If Commec
is not installed, or the database directory is absent, the stage returns
``untested`` with a blocker naming what is missing. See :func:`requirements`.
"""

from __future__ import annotations

from pathlib import Path

from rfproteina.disease.combinations import rollup_axis
from rfproteina.disease.screening_tools import (
    CLEAR,
    HIT,
    UNTESTED,
    ScreeningInputError,
    alphabet_of,
    axis_record,
    read_fasta,
    run_tool,
    tool_available,
)

COMMEC = "commec"

#: Commec recommendation -> axis state. ``warn`` maps to a hit deliberately: a
#: lower-confidence concern still needs human review, and for a safety gate the
#: conservative direction is to flag it. ``error`` is not a verdict, so it is
#: untested. Keys are compared case-folded.
RECOMMENDATION_STATE: dict[str, str] = {
    "pass": CLEAR,
    "flag": HIT,
    "warn": HIT,
    "error": UNTESTED,
}

#: Candidate header names for the record-id column in a ``commec flag`` CSV. The
#: exact spelling is parsed flexibly and should be verified against a real run.
_ID_COLUMNS = ("name", "query", "query_name", "record", "sequence", "sequence_name", "id", "target")
#: Candidate header names for the recommendation column.
_RECOMMENDATION_COLUMNS = ("recommendation", "recommend", "status", "outcome", "overall", "screen_status")


class BiosecurityUnavailableError(RuntimeError):
    """Commec or its database is absent. Carries the blocker to record as untested."""

    def __init__(self, blocker: str) -> None:
        super().__init__(blocker)
        self.blocker = blocker


def commec_available() -> bool:
    """Whether ``commec`` is on PATH. Checked, never assumed."""
    return tool_available(COMMEC)


def build_screen_command(
    fasta: Path,
    databases_dir: Path,
    output_json: Path,
    *,
    commec: str = COMMEC,
    skip_tx: bool = True,
    threads: int = 1,
) -> list[str]:
    """The ``commec screen`` argument list. ``--skip-tx`` omits the large NCBI taxonomy step."""
    command = [
        commec,
        "screen",
        "-d",
        str(databases_dir),
        "-o",
        str(output_json),
        "-t",
        str(threads),
    ]
    if skip_tx:
        command.append("--skip-tx")
    command.append(str(fasta))
    return command


def build_flag_command(screen_dir: Path, csv_out: Path, *, commec: str = COMMEC) -> list[str]:
    """The ``commec flag`` argument list: collate ``.screen.json`` files into a CSV."""
    return [commec, "flag", "-i", str(screen_dir), "-o", str(csv_out)]


def _header_index(header: list[str], candidates: tuple[str, ...]) -> int | None:
    lowered = [name.strip().casefold() for name in header]
    for candidate in candidates:
        if candidate in lowered:
            return lowered.index(candidate)
    # A looser match for a recommendation column whose header embeds the word.
    for position, name in enumerate(lowered):
        if any(candidate in name for candidate in candidates):
            return position
    return None


def parse_flag_csv(text: str) -> dict[str, str]:
    """Map each record id to its recommendation string from a ``commec flag`` CSV.

    The id and recommendation columns are found by name, case-insensitively,
    because the exact header spelling is Commec's to change. A file whose header
    carries neither raises :class:`ScreeningInputError`, rather than returning an
    empty map that would read as "everything screened clear".
    """
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        raise ScreeningInputError("empty_flag_csv")
    header = next(iter(lines)).split(",")
    id_index = _header_index(header, _ID_COLUMNS)
    rec_index = _header_index(header, _RECOMMENDATION_COLUMNS)
    if id_index is None or rec_index is None:
        raise ScreeningInputError("flag_csv_missing_columns")
    recommendations: dict[str, str] = {}
    for line in lines[1:]:
        fields = line.split(",")
        if len(fields) <= max(id_index, rec_index):
            continue
        record_id = fields[id_index].strip()
        if record_id:
            recommendations[record_id] = fields[rec_index].strip()
    return recommendations


def verdict(recommendation: str) -> dict[str, object]:
    """One record's biosecurity axis from its Commec recommendation.

    An unrecognised recommendation is untested, not clear, so a changed Commec
    vocabulary surfaces as "not screened" rather than as a silent pass.
    """
    key = recommendation.strip().casefold()
    state = RECOMMENDATION_STATE.get(key)
    if state is None:
        return axis_record(UNTESTED, None, f"unknown_recommendation:{key or 'blank'}", recommendation=recommendation)
    blocker = None if state != UNTESTED else f"commec_{key}"
    return axis_record(state, None, blocker, recommendation=recommendation)


def biosecurity_screen(
    fasta_path: Path,
    databases_dir: Path,
    work_dir: Path,
    *,
    commec: str = COMMEC,
    skip_tx: bool = True,
    threads: int = 1,
    timeout_s: float = 7200.0,
) -> dict[str, dict[str, object]]:
    """A biosecurity record per record in ``fasta_path``.

    Raises :class:`ScreeningInputError` when the FASTA holds no record. Raises
    :class:`BiosecurityUnavailableError` when Commec is not installed or the
    database directory is absent, so the caller records every input ``untested``
    with that one blocker. A record Commec does not report on is ``untested`` with
    a ``no_commec_verdict`` blocker, never assumed to have passed.
    """
    fasta_path = Path(fasta_path)
    if not fasta_path.is_file():
        raise ScreeningInputError("fasta_missing")
    records = read_fasta(fasta_path.read_text(encoding="utf-8"))
    if not records:
        raise ScreeningInputError("no_fasta_records")
    if not commec_available():
        raise BiosecurityUnavailableError("commec_not_installed")
    if not Path(databases_dir).is_dir():
        raise BiosecurityUnavailableError("commec_databases_missing")
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    screen_out = work_dir / "screen"
    screen_out.mkdir(parents=True, exist_ok=True)
    csv_out = work_dir / "flags.csv"
    screen_result = run_tool(
        build_screen_command(
            fasta_path, Path(databases_dir), screen_out, commec=commec, skip_tx=skip_tx, threads=threads
        ),
        timeout_s=timeout_s,
    )
    if not screen_result.ok:
        raise BiosecurityUnavailableError(screen_result.blocker or "commec_screen_failed")
    flag_result = run_tool(build_flag_command(screen_out, csv_out, commec=commec), timeout_s=timeout_s)
    if not flag_result.ok:
        raise BiosecurityUnavailableError(flag_result.blocker or "commec_flag_failed")
    text = csv_out.read_text(encoding="utf-8") if csv_out.is_file() else flag_result.stdout
    recommendations = parse_flag_csv(text)
    out: dict[str, dict[str, object]] = {}
    for record_id, sequence in records:
        recommendation = recommendations.get(record_id)
        if recommendation is None:
            out[record_id] = axis_record(
                UNTESTED, None, "no_commec_verdict", input_alphabet=alphabet_of(sequence)
            )
            continue
        record = verdict(recommendation)
        record["input_alphabet"] = alphabet_of(sequence)
        out[record_id] = record
    return out


def rollup(records: dict[str, dict[str, object]]) -> dict[str, object]:
    """One biosecurity axis for the whole batch, by the shared roll-up rule."""
    empty = axis_record(UNTESTED, None, "designs_absent")
    return rollup_axis(list(records), records, empty)


def unavailable_axis(blocker: str) -> dict[str, object]:
    """The axis a caller records when the stage could not run at all."""
    return axis_record(UNTESTED, None, blocker)


def requirements() -> dict[str, object]:
    """Exactly what the Commec stage needs, so an unavailable path is actionable."""
    return {
        "install": "pip install commec  (MIT; https://github.com/ibbis-bio/common-mechanism)",
        "databases_biorisk_benign": (
            "commec setup -d <dir>  (biorisk + low-concern + control-list databases; "
            "IBBIS states the full Commec download is under 10 GB; MIT)"
        ),
        "databases_taxonomy": (
            "NCBI nr (BLASTX/DIAMOND) and core_nt (BLASTN), operator-provided and large; "
            "needed only without --skip-tx"
        ),
        "external_tools": "hmmer, diamond or blast, cmscan (Commec calls these; install per Commec docs)",
        "skip_tx_mode": "run with skip_tx=True to screen on biorisk + low-concern databases alone",
        "input": "nucleotide FASTA (DNA/RNA); back-translate protein designs before screening",
        "alternative": (
            "SecureDNA (securedna.org): free hosted screening via a client with issued keys; "
            "not a local-only free database, and it sends a hashed form off-machine, so Commec is "
            "the local-and-free option implemented here"
        ),
    }
