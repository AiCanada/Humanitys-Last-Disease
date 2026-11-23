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
    python scripts/screen_extras.py prodigy run/complex.cif --binder H,L --target A
    python scripts/screen_extras.py tnp run/domain.cif --chain H --sequence-sha256 DIGEST
    python scripts/screen_extras.py ligand "CCOc1ccccc1C(=O)N"
    python scripts/screen_extras.py admet-solubility "CC(=O)Oc1ccccc1C(=O)O"
    python scripts/screen_extras.py qupkake "c1ccncc1"
    python scripts/screen_extras.py sygma "CC(=O)Nc1ccc(O)cc1"
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
from rfproteina.disease.prodigy import METHOD, AffinityInputError, Partners, predict_file


def _print(payload: object) -> int:
    print(json.dumps(payload, indent=2, sort_keys=False))
    return 0


def _tnp(args: argparse.Namespace) -> int:
    from rfproteina.disease.screen import _cif_text
    from rfproteina.disease.tnp import profile, provenance

    settings = {"label_asym_id": args.chain, "sequence_sha256": args.sequence_sha256, "domain_type": "VHH"}
    result = profile(_cif_text(args.structure), settings)
    _print({**result, "corpus_provenance": provenance()})
    return 0 if result["status"] == "profiled" else 2


def _interface(args: argparse.Namespace) -> int:
    targets = args.target.split(",") if args.target else None
    try:
        measured = interface_quality.measure_interface(Path(args.structure), args.binder, targets)
    except interface_quality.InterfaceUnreadableError as error:
        return _print({"error": str(error)}) or 2
    return _print({"measured": measured.as_record(), "verdicts": interface_quality.verdicts(measured)})


def _prodigy(args: argparse.Namespace) -> int:
    try:
        partners = Partners.parse(
            {
                "binder_chains": [chain.strip() for chain in args.binder.split(",")],
                "target_chains": [chain.strip() for chain in args.target.split(",")],
                "temperature_c": args.temperature_c,
                "chain_namespace": args.chain_namespace,
                "model_index": args.model_index,
                "altloc": args.altloc,
            }
        )
    except AffinityInputError as error:
        _print({"method": METHOD, "status": "untested", "blocker": str(error)})
        return 2
    record = predict_file(Path(args.structure), partners)
    _print(record)
    return 0 if record["status"] == "predicted" else 2


def _ligand(args: argparse.Namespace) -> int:
    try:
        screened = ligand_alerts.screen_smiles(args.smiles)
    except ligand_alerts.LigandUnreadableError as error:
        return _print({"error": str(error)}) or 2
    return _print({"measured": screened.as_record(), "verdicts": ligand_alerts.verdicts(screened)})


def _admet_solubility(args: argparse.Namespace) -> int:
    from rfproteina.disease.ligand_solubility import profile

    result = profile(((str(i), s) for i, s in enumerate(args.smiles, 1)), scope="standalone")
    _print(result)
    return 0 if result["status"] == "predicted" else 2


def _qupkake(args: argparse.Namespace) -> int:
    from rfproteina.disease import qupkake

    if args.sdf:
        result = qupkake.profile_sdf(args.sdf, tautomerize=args.tautomerize)
    else:
        result = qupkake.profile([(str(i), s) for i, s in enumerate(args.smiles, 1)], tautomerize=args.tautomerize)
    _print(result)
    return 0 if result["status"] == "complete" else 2


def _sygma(args: argparse.Namespace) -> int:
    from rfproteina.disease import ligand_metabolites

    if args.sdf:
        result = ligand_metabolites.profile_sdf(args.sdf, phase1=args.phase1_cycles, phase2=args.phase2_cycles)
    else:
        rows = [(str(i), s) for i, s in enumerate(args.smiles, 1)]
        result = ligand_metabolites.profile(rows, phase1=args.phase1_cycles, phase2=args.phase2_cycles)
    _print(result)
    return 0 if result["status"] == "complete" else 2


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

    tnp = sub.add_parser("tnp", help="profile a finished, explicitly identified single-domain VHH with local TNP")
    tnp.add_argument("structure", type=Path, help="finished mmCIF, optionally gzip or zstd compressed")
    tnp.add_argument("--chain", required=True, help="exact mmCIF label_asym_id of the isolated VHH domain")
    tnp.add_argument("--sequence-sha256", required=True, help="SHA-256 of the nominated canonical domain sequence")
    tnp.set_defaults(run=_tnp)

    one = sub.add_parser("interface", help="buried area, interface residues and polar satisfaction")
    one.add_argument("structure", help="a CIF or PDB file holding the complex")
    one.add_argument("--binder", required=True, help="chain id of the binder")
    one.add_argument("--target", default="", help="comma separated target chain ids, default every other chain")
    one.set_defaults(run=_interface)

    affinity = sub.add_parser("prodigy", help="estimate affinity between two protein partners")
    affinity.add_argument("structure", help="PDB or mmCIF, optionally gzip or zstd compressed")
    affinity.add_argument("--binder", required=True, help="comma-separated binder chain identifiers")
    affinity.add_argument("--target", required=True, help="comma-separated target chain identifiers")
    affinity.add_argument("--temperature-c", type=float, default=25.0, help="Celsius for deriving Kd (default 25)")
    affinity.add_argument("--chain-namespace", choices=("author", "label"), default="author")
    affinity.add_argument("--model-index", type=int, help="one-based model ordinal for an ensemble")
    affinity.add_argument("--altloc", help="alternate-location identifier when residues have multiple conformers")
    affinity.set_defaults(run=_prodigy)

    two = sub.add_parser("ligand", help="RDKit PAINS, BRENK and NIH alerts plus drug-likeness")
    two.add_argument("smiles")
    two.set_defaults(run=_ligand)

    solubility = sub.add_parser("admet-solubility", help="predict aqueous log10 molar solubility with local ADMET-AI")
    solubility.add_argument("smiles", nargs="+", help="one quoted SMILES per candidate")
    solubility.set_defaults(run=_admet_solubility)

    pka = sub.add_parser("qupkake", help="predict aqueous microscopic pKa for individual proton-transfer sites")
    pka.add_argument("smiles", nargs="*", help="one quoted molecular SMILES per nomination")
    pka.add_argument("--sdf", type=Path, help="input SDF with one named record per nomination")
    pka.add_argument("--tautomerize", action="store_true", help="rank enumerated tautomers using GFN2-xTB in water")
    pka.set_defaults(run=_qupkake)

    metabolites = sub.add_parser("sygma", help="rank possible human metabolites for each nominated small molecule")
    metabolites.add_argument("smiles", nargs="*", help="one quoted parent SMILES per nomination")
    metabolites.add_argument("--sdf", type=Path, help="SDF containing named parent structures")
    metabolites.add_argument("--phase1-cycles", type=int, default=1)
    metabolites.add_argument("--phase2-cycles", type=int, default=1)
    metabolites.set_defaults(run=_sygma)

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
    if args.command in {"qupkake", "sygma"} and bool(args.sdf) == bool(args.smiles):
        parser.error(f"{args.command} requires either one or more SMILES or --sdf FILE")
    return int(args.run(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
