#!/usr/bin/env python
"""Run the pose, interaction and ligand-safety screens from the command line.

These are the screens worker D added: PoseBusters plus RDKit pose validation,
a native (or ProLIF) interaction fingerprint, Fnat against a reference complex,
the ChEMBL measured-nM safety panel, the local BayeshERG classifier, and a reader
for completed OpenFE relative binding free-energy results. The TI command
analyses archived alchemlyb derivatives for one stage. The pmx-bar command
verifies a completed switching-work report. Each command prints JSON.

Nothing here downloads a model or needs a key. ``safety`` is the only subcommand
that touches the network, and only to read ChEMBL's free REST API.

Examples::

    python scripts/screen_pose_safety.py pose ligand.sdf --protein pocket.pdb
    python scripts/screen_pose_safety.py spyrmsd controls --sample redock_1
    python scripts/screen_pose_safety.py ifp pocket.pdb ligand.sdf
    python scripts/screen_pose_safety.py fnat model.pdb --reference native.pdb --chains A,B
    python scripts/screen_pose_safety.py safety "CC(=O)Oc1ccccc1C(=O)O"
    python scripts/screen_pose_safety.py bayesherg CCO --release-dir /path/to/BayeshERG --python /path/to/env/bin/python
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rfproteina.disease import interactions, ligand_safety, pose_validation
from rfproteina.disease.openfe_results import read_openfe_rbfe


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
    try:
        mol = pose_validation.read_ligand(args.ligand)
        measure = interactions.prolif_fingerprint if args.engine == "prolif" else interactions.native_fingerprint
        fingerprint = measure(Path(args.protein), mol)
    except (interactions.StructureUnreadableError, interactions.ProLIFUnavailableError,
            pose_validation.PoseUnreadableError, OSError, ValueError) as error:
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


def _bayesherg(args: argparse.Namespace) -> int:
    try:
        result = ligand_safety.bayesherg_probability(args.smiles, release_dir=args.release_dir, python=args.python)
    except (ligand_safety.LigandUnreadableError, ligand_safety.SafetyBackendUnavailableError) as error:
        return _print({"error": str(error)}) or 2
    record = result.as_record()
    ligand_safety.assert_no_probability_in_nm(record)
    return _print({"predicted": record})


def _rbfe(args: argparse.Namespace) -> int:
    try:
        record = read_openfe_rbfe(args.ligand_a, args.ligand_b, args.result_files)
    except ValueError as error:
        _print({"method": "openfe_completed_rbfe", "state": "untested", "reason": str(error)})
        return 2
    return _print(record)


def _ti(args: argparse.Namespace) -> int:
    from rfproteina.disease.alchemical_ti import analyze_ti_stage

    try:
        record = analyze_ti_stage(args.table, args.states, discard_before=args.discard_before)
    except (OSError, ValueError) as error:
        _print({"method": "alchemlyb_thermodynamic_integration", "state": "untested", "reason": str(error)})
        return 2
    return _print(record)


def _pmx_bar(args: argparse.Namespace) -> int:
    from rfproteina.disease.pmx_results import read_pmx_bar

    try:
        record = read_pmx_bar(args.report, args.forward, args.reverse)
    except (OSError, ValueError) as error:
        _print({"method": "pmx_completed_bar", "state": "untested", "reason": str(error)})
        return 2
    return _print(record)


def _spyrmsd(args: argparse.Namespace) -> int:
    from rfproteina.disease.spyrmsd_pose import measure

    result = measure(args.directory, args.sample, standalone=True)
    _print(result)
    return 0 if result["status"] == "reported" else 2


def _vina(args: argparse.Namespace) -> int:
    from rfproteina.disease.screen import _cif_text
    from rfproteina.disease.vina_pose import RunPoses, corpus_provenance

    row = RunPoses.read(args.manifest).score(args.sample, _cif_text(args.structure))
    _print({**row, "corpus_provenance": corpus_provenance()})
    return 0 if row["status"] == "scored" else 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    vina = sub.add_parser("vina", help="score one supplied small-molecule pose against its verified rigid receptor")
    vina.add_argument("structure", type=Path, help="finished mmCIF, optionally gzip or zstd compressed")
    vina.add_argument("--manifest", type=Path, required=True, help="vina_poses.json with prepared inputs and box")
    vina.add_argument("--sample", required=True, help="exact sample ID in the manifest")
    vina.set_defaults(run=_vina)

    redock = sub.add_parser("spyrmsd", help="measure symmetry-corrected RMSD for matched redocking controls")
    redock.add_argument("directory", type=Path, help="directory containing spyrmsd_controls.json and its ligand files")
    redock.add_argument("--sample", help="exact control ID in the manifest (default: all nominated controls)")
    redock.set_defaults(run=_spyrmsd)

    pose = sub.add_parser("pose", help="PoseBusters + RDKit pose validation")
    pose.add_argument("ligand", help="one ligand pose with 3D coordinates in SDF, MOL, MOL2 or PDB format")
    pose.add_argument(
        "--protein", default="", help="receptor PDB/mmCIF (optionally compressed), without the query ligand"
    )
    pose.add_argument("--allow-missing", action="store_true", help="do not fail if PoseBusters is absent")
    pose.set_defaults(run=_pose)

    ifp = sub.add_parser("ifp", help="interaction fingerprint of one ligand pose against a protein")
    ifp.add_argument("protein", help="protein PDB/CIF")
    ifp.add_argument("ligand", help="ligand SDF or MOL with 3D coordinates")
    ifp.add_argument("--engine", choices=("native", "prolif"), default="native")
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

    bayesherg = sub.add_parser("bayesherg", help="local hERG blocker probability and uncertainty variances")
    bayesherg.add_argument("smiles")
    bayesherg.add_argument("--release-dir", type=Path, help="BayeshERG checkout, defaults to BAYESHERG_ROOT")
    bayesherg.add_argument("--python", type=Path, help="BayeshERG interpreter, defaults to BAYESHERG_PYTHON")
    bayesherg.set_defaults(run=_bayesherg)

    rbfe = sub.add_parser("rbfe", help="audit a completed OpenFE binding free-energy comparison")
    rbfe.add_argument("ligand_a", help="recorded initial ligand name")
    rbfe.add_argument("ligand_b", help="recorded final ligand name")
    rbfe.add_argument("result_files", type=Path, nargs="+", help="complex and solvent result JSONs for this edge")
    rbfe.set_defaults(run=_rbfe)

    ti = sub.add_parser("ti", help="thermodynamic integration of one archived alchemlyb stage")
    ti.add_argument("table", type=Path, help="combined one-coordinate dH/d-lambda Parquet table")
    ti.add_argument("--states", nargs="+", type=float, required=True, help="increasing lambda schedule from the study")
    ti.add_argument("--discard-before", type=float, help="optional time cut in the source table's time units")
    ti.set_defaults(run=_ti)

    pmx = sub.add_parser("pmx-bar", help="verify a published pmx BAR estimate against integrated work")
    pmx.add_argument("report", type=Path, help="pmx results.txt with original trajectory lists")
    pmx.add_argument("forward", type=Path, help="forward integrated-work file")
    pmx.add_argument("reverse", type=Path, help="reverse integrated-work file in pmx's sign convention")
    pmx.set_defaults(run=_pmx_bar)

    args = parser.parse_args(argv)
    return int(args.run(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
