"""One CLI over the antibody developability, humanness and immunogenicity screens.

Nothing here judges a design. Each subcommand prints one JSON record with the
method, the provenance and the blockers attached, so another program can read
it without parsing prose.

    python scripts/screen_antibody.py liabilities --heavy EVQ... --light DIQ...
    python scripts/screen_antibody.py sap --heavy EVQ... --light DIQ...
    python scripts/screen_antibody.py humanness --heavy EVQ... --light DIQ...
    python scripts/screen_antibody.py mhc1-fetch
    python scripts/screen_antibody.py mhc1 --sequence EVQ... --allele 'HLA-A*02:01' --out runs/x
    python scripts/screen_antibody.py mhc2-status

``mhc1`` needs MHCflurry. It is not in this project's environment because
MHCflurry 2.3 requires pandas 3 and this project pins pandas 2. Install it
under ``DATA_PATH/mhcflurry``. ``scripts/install_mhcflurry.py`` creates
that environment and downloads the presentation weights.
``MHCFLURRY_PREDICT`` overrides the executable it installs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rfproteina.disease import (
    class_ii,
    deepsp,
    deepviscosity,
    developability,
    humanness,
    immunogenicity,
    mhc_tables,
)


def _liabilities(args: argparse.Namespace) -> dict[str, object]:
    sequences = {name: value for name, value in (("H", args.heavy), ("L", args.light)) if value}
    if not sequences:
        raise SystemExit("give --heavy, --light, or both")
    try:
        chains = developability.number_chains(sequences)
    except developability.NumberingUnavailableError as error:
        return {"developability_blocker": str(error)}
    return developability.developability_report(chains, cdr_only=args.cdr_only)


def _sap(args: argparse.Namespace) -> dict[str, object]:
    """DeepSP's aggregation and charge descriptors. Both chains, or neither."""
    sequences = {name: value for name, value in (("H", args.heavy), ("L", args.light)) if value}
    if len(sequences) != 2:
        raise SystemExit("give both --heavy and --light: DeepSP scores an Fv, not a lone chain")
    return deepsp.deepsp_report(
        sequences,
        Path(args.models) if args.models else None,
        name=args.name,
    )


def _viscosity(args: argparse.Namespace) -> dict[str, object]:
    """DeepViscosity's high-concentration viscosity class. Both chains, or neither."""
    sequences = {name: value for name, value in (("H", args.heavy), ("L", args.light)) if value}
    if len(sequences) != 2:
        raise SystemExit("give both --heavy and --light: DeepViscosity scores an Fv, not a lone chain")
    return deepviscosity.deepviscosity_report(
        sequences,
        Path(args.models) if args.models else None,
        Path(args.deepsp_models) if args.deepsp_models else None,
        name=args.name,
    )


def _humanness(args: argparse.Namespace) -> dict[str, object]:
    sequences = {name: value for name, value in (("H", args.heavy), ("L", args.light)) if value}
    if not sequences:
        raise SystemExit("give --heavy, --light, or both")
    pairs = [("design", args.heavy or "", args.light or "")] if args.likeness else None
    return humanness.humanness_report(sequences, pairs=pairs)


def _mhc1_fetch(_args: argparse.Namespace) -> dict[str, object]:
    try:
        return mhc_tables.fetch_weights()
    except mhc_tables.PredictorMissingError as error:
        return {"mhc_i_blocker": str(error), "install": "see the module docstring"}


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


def _mhc2_status(_args: argparse.Namespace) -> dict[str, object]:
    return immunogenicity.class_ii_status()


def _mhc2_train(args: argparse.Namespace) -> dict[str, object]:
    """Fit the class II matrices from the IEDB corpus and save them.

    The AUC comes back with the model rather than on request: a predictor
    whose worth is unreported is the thing this axis was missing, not a
    predictor.
    """
    directory = Path(args.out) if args.out else class_ii.default_model_dir()
    corpus = Path(args.corpus) if args.corpus else directory / "class_ii_dr_panel.tsv"
    report = class_ii.load_corpus(corpus)
    model = class_ii.train(report)
    return {
        "corpus": str(corpus),
        "rows_read": report.read,
        "pairs_kept": len(report.rows),
        "dropped": dict(report.dropped),
        "pairs_from_repeats": report.aggregated,
        "covers_panel": report.covers_panel,
        "model": str(class_ii.save(model, directory)),
        "method": model.method,
        "mean_held_out_auc": model.mean_auc,
        "per_allele": {
            allele: {"rows": fitted.trained_rows, "binders": fitted.binders, "held_out_auc": fitted.auc}
            for allele, fitted in sorted(model.alleles.items())
        },
        "panel_alleles_missing": list(model.missing),
        "notes": list(model.notes),
    }


def _mhc2(args: argparse.Namespace) -> dict[str, object]:
    """Score one chain's 15-mer windows against the panel."""
    return class_ii.class_ii_report(
        args.sequence,
        directory=Path(args.models) if args.models else None,
        out=Path(args.out) if args.out else None,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    liabilities = subparsers.add_parser("liabilities", help="IMGT numbering, CDRs and chemical liability motifs")
    liabilities.add_argument("--heavy")
    liabilities.add_argument("--light")
    liabilities.add_argument("--cdr-only", action="store_true")
    liabilities.set_defaults(run=_liabilities)

    sap = subparsers.add_parser(
        "sap",
        help="DeepSP spatial aggregation propensity and charge maps from sequence (needs install_deepsp.py)",
    )
    sap.add_argument("--heavy", required=True)
    sap.add_argument("--light", required=True)
    sap.add_argument("--name", default="design", help="what to call this antibody in the record")
    sap.add_argument("--models", help="model directory; defaults to DATA_PATH/deepsp/models")
    sap.set_defaults(run=_sap)

    viscosity = subparsers.add_parser(
        "viscosity",
        help=(
            "DeepViscosity high-concentration viscosity class at 150 mg/mL "
            "(needs install_deepviscosity.py and install_deepsp.py)"
        ),
    )
    viscosity.add_argument("--heavy", required=True)
    viscosity.add_argument("--light", required=True)
    viscosity.add_argument("--name", default="design", help="what to call this antibody in the record")
    viscosity.add_argument("--models", help="model directory; defaults to DATA_PATH/deepviscosity/models")
    viscosity.add_argument(
        "--deepsp-models",
        help="DeepSP's models, which produce the descriptors; defaults to DATA_PATH/deepsp/models",
    )
    viscosity.set_defaults(run=_viscosity)

    human = subparsers.add_parser("humanness", help="OASis identity, optionally AbLang-2 antibody-likeness")
    human.add_argument("--heavy")
    human.add_argument("--light")
    human.add_argument("--likeness", action="store_true", help="also load AbLang-2 (slow, torch)")
    human.set_defaults(run=_humanness)

    fetch = subparsers.add_parser(
        "mhc1-fetch",
        help="download the MHCflurry class I presentation weights into DATA_PATH/mhcflurry/models",
    )
    fetch.set_defaults(run=_mhc1_fetch)

    mhc1 = subparsers.add_parser("mhc1", help="MHCflurry class I over a sequence, written as mhc_i.tsv")
    mhc1.add_argument("--sequence", required=True)
    mhc1.add_argument("--allele", action="append", required=True)
    mhc1.add_argument("--width", action="append", type=int, default=None)
    mhc1.add_argument("--out", help="directory to write mhc_i.tsv into")
    mhc1.set_defaults(run=_mhc1)

    status = subparsers.add_parser("mhc2-status", help="why class II is unavailable, and what each option costs")
    status.set_defaults(run=_mhc2_status)

    train = subparsers.add_parser(
        "mhc2-train",
        help="fit the class II matrices from the IEDB corpus and report their held-out AUC",
    )
    train.add_argument("--corpus", help="class_ii_dr_panel.tsv; defaults to the one beside the model")
    train.add_argument("--out", help="directory for class_ii_pssm.json; defaults to DATA_PATH/iedb")
    train.set_defaults(run=_mhc2_train)

    mhc2 = subparsers.add_parser("mhc2", help="class II ranks over a sequence, optionally written as mhc_ii.tsv")
    mhc2.add_argument("--sequence", required=True)
    mhc2.add_argument("--models", help="directory holding class_ii_pssm.json")
    mhc2.add_argument("--out", help="directory to write mhc_ii.tsv into")
    mhc2.set_defaults(run=_mhc2)

    args = parser.parse_args(argv)
    if getattr(args, "width", None) is None and args.command == "mhc1":
        args.width = list(mhc_tables.CLASS_I_WIDTHS)
    print(json.dumps(args.run(args), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
