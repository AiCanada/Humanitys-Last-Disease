"""One CLI over the antibody developability, humanness and immunogenicity screens.

Nothing here judges a design. Each subcommand prints one JSON record with the
method, the provenance and the blockers attached, so another program can read
it without parsing prose.

    python scripts/screen_antibody.py liabilities --heavy EVQ... --light DIQ...
    python scripts/screen_antibody.py humanness --heavy EVQ... --light DIQ...
    python scripts/screen_antibody.py mhc1 --sequence EVQ... --allele 'HLA-A*02:01' --out runs/x
    python scripts/screen_antibody.py mhc2-status

``mhc1`` needs MHCflurry. It is not in this project's environment because
MHCflurry 2.3 requires pandas 3 and this project pins pandas 2; install it
separately and point ``MHCFLURRY_PREDICT`` at its ``mhcflurry-predict``:

    uv venv ~/.venvs/mhcflurry && VIRTUAL_ENV=~/.venvs/mhcflurry uv pip install 'mhcflurry>=2.3,<2.4'
    ~/.venvs/mhcflurry/bin/mhcflurry downloads fetch models_class1_presentation
    export MHCFLURRY_PREDICT=~/.venvs/mhcflurry/bin/mhcflurry-predict
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rfproteina.disease import developability, humanness, immunogenicity, mhc_tables


def _liabilities(args: argparse.Namespace) -> dict[str, object]:
    sequences = {name: value for name, value in (("H", args.heavy), ("L", args.light)) if value}
    if not sequences:
        raise SystemExit("give --heavy, --light, or both")
    try:
        chains = developability.number_chains(sequences)
    except developability.NumberingUnavailableError as error:
        return {"developability_blocker": str(error)}
    return developability.developability_report(chains, cdr_only=args.cdr_only)


def _humanness(args: argparse.Namespace) -> dict[str, object]:
    sequences = {name: value for name, value in (("H", args.heavy), ("L", args.light)) if value}
    if not sequences:
        raise SystemExit("give --heavy, --light, or both")
    pairs = [("design", args.heavy or "", args.light or "")] if args.likeness else None
    return humanness.humanness_report(sequences, pairs=pairs)


def _mhc1(args: argparse.Namespace) -> dict[str, object]:
    try:
        record = mhc_tables.predict_sequence(
            args.sequence,
            args.allele,
            widths=args.width,
            directory=Path(args.out) if args.out else None,
        )
    except mhc_tables.PredictorMissingError as error:
        return {"mhc_i_blocker": str(error), "install": "see the module docstring"}
    record.pop("table", None)
    return record


def _mhc2(_args: argparse.Namespace) -> dict[str, object]:
    return immunogenicity.class_ii_status()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    liabilities = subparsers.add_parser("liabilities", help="IMGT numbering, CDRs and chemical liability motifs")
    liabilities.add_argument("--heavy")
    liabilities.add_argument("--light")
    liabilities.add_argument("--cdr-only", action="store_true")
    liabilities.set_defaults(run=_liabilities)

    human = subparsers.add_parser("humanness", help="OASis identity, optionally AbLang-2 antibody-likeness")
    human.add_argument("--heavy")
    human.add_argument("--light")
    human.add_argument("--likeness", action="store_true", help="also load AbLang-2 (slow, torch)")
    human.set_defaults(run=_humanness)

    mhc1 = subparsers.add_parser("mhc1", help="MHCflurry class I over a sequence, written as mhc_i.tsv")
    mhc1.add_argument("--sequence", required=True)
    mhc1.add_argument("--allele", action="append", required=True)
    mhc1.add_argument("--width", action="append", type=int, default=None)
    mhc1.add_argument("--out", help="directory to write mhc_i.tsv into")
    mhc1.set_defaults(run=_mhc1)

    mhc2 = subparsers.add_parser("mhc2-status", help="why class II is unavailable, and what each option costs")
    mhc2.set_defaults(run=_mhc2)

    args = parser.parse_args(argv)
    if getattr(args, "width", None) is None and args.command == "mhc1":
        args.width = list(mhc_tables.CLASS_I_WIDTHS)
    print(json.dumps(args.run(args), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
