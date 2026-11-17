#!/usr/bin/env python
"""Run the added screens from the command line, one design at a time.

These are the screens that had no runner: interface quality on a finished
complex, RDKit structure alerts on a ligand, class I peptide windows and the
MHCflurry-to-table conversion, and construct synthesis feasibility.

Nothing here downloads anything, calls a service, or needs a key. The MHCflurry
subcommands shell out to ``mhcflurry-predict`` only when it is already
installed, and say so plainly when it is not.

Examples::

    python scripts/screen_extras.py interface run/binder_0.cif --binder A
    python scripts/screen_extras.py ligand "CCOc1ccccc1C(=O)N"
    python scripts/screen_extras.py mhc-windows MKTIIALSYIFCLVFA --widths 9
    python scripts/screen_extras.py mhc-convert predictions.csv --into run/
    python scripts/screen_extras.py synthesis ATGGCC... --forbidden BsaI=GGTCTC
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rfproteina.disease import interface_quality, ligand_alerts, mhc_tables, synthesis


def _print(payload: object) -> int:
    print(json.dumps(payload, indent=2, sort_keys=False))
    return 0


def _interface(args: argparse.Namespace) -> int:
    targets = args.target.split(",") if args.target else None
    try:
        measured = interface_quality.measure_interface(Path(args.structure), args.binder, targets)
    except interface_quality.InterfaceUnreadableError as error:
        return _print({"error": str(error)}) or 2
    return _print({"measured": measured.as_record(), "verdicts": interface_quality.verdicts(measured)})


def _ligand(args: argparse.Namespace) -> int:
    try:
        screened = ligand_alerts.screen_smiles(args.smiles)
    except ligand_alerts.LigandUnreadableError as error:
        return _print({"error": str(error)}) or 2
    return _print({"measured": screened.as_record(), "verdicts": ligand_alerts.verdicts(screened)})


def _mhc_windows(args: argparse.Namespace) -> int:
    widths = tuple(int(value) for value in args.widths.split(",")) if args.widths else mhc_tables.CLASS_I_WIDTHS
    found = mhc_tables.windows(args.sequence, widths)
    return _print(
        {
            "windows": [{"start": start, "peptide": peptide} for start, peptide in found],
            "count": len(found),
            "predictor_available": mhc_tables.predictor_available(),
            "install": "pip install mhcflurry && mhcflurry-downloads fetch models_class1_presentation",
        }
    )


def _mhc_convert(args: argparse.Namespace) -> int:
    text = Path(args.predictions).read_text(encoding="utf-8-sig")
    try:
        if args.into:
            path = mhc_tables.write_mhc_i_table(text, Path(args.into))
            return _print({"written": str(path)})
        return _print({"table": mhc_tables.convert_mhcflurry_csv(text)})
    except ValueError as error:
        return _print({"error": str(error)}) or 2


def _synthesis(args: argparse.Namespace) -> int:
    forbidden: dict[str, str] = {}
    for item in args.forbidden or []:
        name, _, site = item.partition("=")
        if site:
            forbidden[name] = site
    try:
        report = synthesis.screen_sequence(args.sequence, forbidden=forbidden or None)
    except synthesis.SequenceUnreadableError as error:
        return _print({"error": str(error)}) or 2
    return _print({"measured": report.as_record(), "verdicts": synthesis.verdicts(report)})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    one = sub.add_parser("interface", help="buried area, interface residues and polar satisfaction")
    one.add_argument("structure", help="a CIF or PDB file holding the complex")
    one.add_argument("--binder", required=True, help="chain id of the binder")
    one.add_argument("--target", default="", help="comma separated target chain ids, default every other chain")
    one.set_defaults(run=_interface)

    two = sub.add_parser("ligand", help="RDKit PAINS, BRENK and NIH alerts plus drug-likeness")
    two.add_argument("smiles")
    two.set_defaults(run=_ligand)

    three = sub.add_parser("mhc-windows", help="class I peptide windows for a predictor")
    three.add_argument("sequence")
    three.add_argument("--widths", default="", help="comma separated widths, default 8,9,10,11")
    three.set_defaults(run=_mhc_windows)

    four = sub.add_parser("mhc-convert", help="MHCflurry CSV into the mhc_i.tsv the screen reads")
    four.add_argument("predictions")
    four.add_argument("--into", default="", help="run directory to write mhc_i.tsv into")
    four.set_defaults(run=_mhc_convert)

    five = sub.add_parser("synthesis", help="GC windows, repeats, hairpins and homopolymers")
    five.add_argument("sequence")
    five.add_argument("--forbidden", action="append", help="NAME=SITE, may be repeated")
    five.set_defaults(run=_synthesis)

    args = parser.parse_args(argv)
    return int(args.run(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
