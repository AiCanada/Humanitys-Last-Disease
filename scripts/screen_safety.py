"""CLI over the novelty, off-target and biosecurity screens (worker B).

One entry point so an agent can run any of the three screens and read a JSON
verdict without importing the modules. Mirrors ``scripts/screen_extras.py``: it
shells out to the external tools only when they are installed, and prints an
``untested`` record with a blocker when they are not. Nothing is downloaded.

    python scripts/screen_safety.py check
    python scripts/screen_safety.py novelty    --query-dir DIR --db PATH --work DIR
    python scripts/screen_safety.py offtarget  --fasta designs.fasta --proteome UP000005640_9606.fasta.gz --work DIR
    python scripts/screen_safety.py biosecurity --fasta constructs.fasta --db COMMEC_DB_DIR --work DIR
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rfproteina.disease import biosecurity, novelty, offtarget
from rfproteina.disease.screening_tools import read_fasta


def _emit(payload: dict[str, object]) -> None:
    print(json.dumps(payload, indent=2, default=str))


def _cmd_check(_args: argparse.Namespace) -> int:
    _emit(
        {
            "foldseek_available": novelty.foldseek_available(),
            "mmseqs_available": offtarget.mmseqs_available(),
            "commec_available": biosecurity.commec_available(),
            "novelty_databases": novelty.database_requirements(),
            "offtarget_structure_database": offtarget.structure_offtarget_requirements(),
            "biosecurity_requirements": biosecurity.requirements(),
        }
    )
    return 0


def _cmd_novelty(args: argparse.Namespace) -> int:
    try:
        records = novelty.novelty_search(Path(args.query_dir), Path(args.db), Path(args.work))
    except novelty.NoveltyUnavailableError as error:
        _emit({"axis": "structural_novelty", "unavailable": novelty.unavailable_axis(error.blocker)})
        return 0
    _emit({"axis": "structural_novelty", "per_design": records, "rollup": novelty.rollup(records)})
    return 0


def _cmd_offtarget(args: argparse.Namespace) -> int:
    designs = read_fasta(Path(args.fasta).read_text(encoding="utf-8"))
    ignore = frozenset(token for token in (args.ignore or "").split(",") if token)
    try:
        records = offtarget.offtarget_scan(designs, Path(args.proteome), Path(args.work), ignore_targets=ignore)
    except offtarget.OffTargetUnavailableError as error:
        _emit({"axis": offtarget.AXIS_NAME, "unavailable": offtarget.unavailable_axis(error.blocker)})
        return 0
    _emit({"axis": offtarget.AXIS_NAME, "per_design": records, "rollup": offtarget.rollup(records)})
    return 0


def _cmd_biosecurity(args: argparse.Namespace) -> int:
    try:
        records = biosecurity.biosecurity_screen(
            Path(args.fasta), Path(args.db), Path(args.work), skip_tx=not args.full_taxonomy
        )
    except biosecurity.BiosecurityUnavailableError as error:
        _emit({"axis": "biosecurity", "unavailable": biosecurity.unavailable_axis(error.blocker)})
        return 0
    _emit({"axis": "biosecurity", "per_record": records, "rollup": biosecurity.rollup(records)})
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Novelty, off-target and biosecurity screens.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("check", help="Tool availability and what each screen needs.").set_defaults(func=_cmd_check)

    p_novelty = sub.add_parser("novelty", help="Foldseek structural novelty.")
    p_novelty.add_argument("--query-dir", required=True)
    p_novelty.add_argument("--db", required=True)
    p_novelty.add_argument("--work", required=True)
    p_novelty.set_defaults(func=_cmd_novelty)

    p_offtarget = sub.add_parser("offtarget", help="MMseqs2 proteome off-target scan.")
    p_offtarget.add_argument("--fasta", required=True)
    p_offtarget.add_argument("--proteome", required=True)
    p_offtarget.add_argument("--work", required=True)
    p_offtarget.add_argument("--ignore", default="", help="Comma-separated on-target UniProt accessions to skip.")
    p_offtarget.set_defaults(func=_cmd_offtarget)

    p_biosec = sub.add_parser("biosecurity", help="Commec biosecurity screen.")
    p_biosec.add_argument("--fasta", required=True)
    p_biosec.add_argument("--db", required=True)
    p_biosec.add_argument("--work", required=True)
    p_biosec.add_argument("--full-taxonomy", action="store_true", help="Run the large NCBI taxonomy step.")
    p_biosec.set_defaults(func=_cmd_biosecurity)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
