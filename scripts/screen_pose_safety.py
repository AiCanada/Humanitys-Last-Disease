#!/usr/bin/env python
"""Run the pose, interaction and ligand-safety screens from the command line.

These are the screens worker D added: PoseBusters plus RDKit pose validation,
a native (or ProLIF) interaction fingerprint, Fnat against a reference complex,
and the ChEMBL measured-nM safety panel. Each prints JSON.

Nothing here downloads a model or needs a key. ``safety`` is the only subcommand
that touches the network, and only to read ChEMBL's free REST API.

Examples::

    python scripts/screen_pose_safety.py pose ligand.sdf --protein pocket.pdb
    python scripts/screen_pose_safety.py ifp pocket.pdb ligand.sdf
    python scripts/screen_pose_safety.py fnat model.pdb --reference native.pdb --chains A,B
    python scripts/screen_pose_safety.py safety "CC(=O)Oc1ccccc1C(=O)O"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rfproteina.disease import interactions, ligand_safety, pose_validation


def _print(payload: object) -> int:
    print(json.dumps(payload, indent=2, sort_keys=False))
    return 0


def _pose(args: argparse.Namespace) -> int:
    try:
        result = pose_validation.validate_pose(
            Path(args.ligand), Path(args.protein) if args.protein else None, require_posebusters=not args.allow_missing
        )
    except (pose_validation.PoseUnreadableError, pose_validation.PoseBustersUnavailableError) as error:
        return _print({"error": str(error)}) or 2
    return _print({"measured": result.as_record(), "verdicts": pose_validation.verdicts(result)})


def _ifp(args: argparse.Namespace) -> int:
    from rdkit import Chem

    mol = Chem.MolFromMolFile(args.ligand, removeHs=False) if args.ligand.lower().endswith(".mol") else None
    if mol is None:
        supplier = Chem.SDMolSupplier(args.ligand, removeHs=False)
        mol = next((item for item in supplier if item is not None), None)
    if mol is None:
        return _print({"error": "ligand_unparsable"}) or 2
    try:
        fingerprint = interactions.native_fingerprint(Path(args.protein), mol)
    except interactions.StructureUnreadableError as error:
        return _print({"error": str(error)}) or 2
    return _print({"measured": fingerprint.as_record(), "verdicts": interactions.native_verdicts(fingerprint)})


def _fnat(args: argparse.Namespace) -> int:
    chain_a, _, chain_b = args.chains.partition(",")
    try:
        comparison = interactions.compare_to_reference(Path(args.model), Path(args.reference), chain_a, chain_b)
    except interactions.StructureUnreadableError as error:
        return _print({"error": str(error)}) or 2
    bar = args.fnat_min if args.fnat_min is not None else None
    return _print({"measured": comparison.as_record(), "verdicts": interactions.comparison_verdicts(comparison, bar)})


def _safety(args: argparse.Namespace) -> int:
    try:
        measured = ligand_safety.fetch_measured_nm(args.smiles)
    except ligand_safety.LigandUnreadableError as error:
        return _print({"error": str(error)}) or 2
    record = measured.as_record()
    ligand_safety.assert_no_probability_in_nm(record)
    return _print({"measured": record, "toxicology_rows": ligand_safety.to_toxicology_rows(measured)})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    pose = sub.add_parser("pose", help="PoseBusters + RDKit pose validation")
    pose.add_argument("ligand", help="an SDF, MOL, MOL2 or PDB holding the pose")
    pose.add_argument("--protein", default="", help="protein PDB/CIF for dock-mode checks and native clashes")
    pose.add_argument("--allow-missing", action="store_true", help="do not fail if PoseBusters is absent")
    pose.set_defaults(run=_pose)

    ifp = sub.add_parser("ifp", help="native interaction fingerprint of a ligand against a protein")
    ifp.add_argument("protein", help="protein PDB/CIF")
    ifp.add_argument("ligand", help="ligand SDF or MOL with 3D coordinates")
    ifp.set_defaults(run=_ifp)

    fnat = sub.add_parser("fnat", help="Fnat of a model complex against a reference")
    fnat.add_argument("model", help="model complex PDB/CIF")
    fnat.add_argument("--reference", required=True, help="reference complex PDB/CIF")
    fnat.add_argument("--chains", default="A,B", help="the two chain ids, comma separated")
    fnat.add_argument("--fnat-min", type=float, default=None, help="optional pass bar, recorded as caller_supplied")
    fnat.set_defaults(run=_fnat)

    safety = sub.add_parser("safety", help="measured nM safety panel from ChEMBL")
    safety.add_argument("smiles")
    safety.set_defaults(run=_safety)

    args = parser.parse_args(argv)
    return int(args.run(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
