#!/usr/bin/env python3
"""Fit and run the nanomolar regressors for the three uncovered panel axes.

``nav1_5``, ``bcrp`` and ``bsep`` are the ligand-safety axes ADMET-AI has no
endpoint for, so for a designed ligand they have no answer at all. BindingDB
has thousands of measurements against those accessions, and
``tables.bindingdb_panel_subset`` already reduces the download to them. This
fits a regressor per axis on that subset and predicts from it.

    python scripts/train_ligand_regressor.py train
    python scripts/train_ligand_regressor.py train --backend chemprop
    python scripts/train_ligand_regressor.py predict --smiles "CC(=O)Oc1ccccc1C(=O)O"
    python scripts/train_ligand_regressor.py have

Everything printed under ``predicted_nanomolar`` is a model output. It is not a
measurement and must never be copied into an ``*_nm`` column; the module
enforces that separation rather than relying on this paragraph.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rfproteina.disease import ligand_regressor as lr


def _unquote(value: str) -> str:
    text = value.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        return text[1:-1]
    return text


def _ensure_data_path() -> None:
    if os.environ.get("DATA_PATH", "").strip():
        return
    env_file = Path(__file__).resolve().parent.parent / ".env.local"
    if not env_file.is_file():
        return
    for raw in env_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() == "DATA_PATH":
            os.environ["DATA_PATH"] = _unquote(value)
            return


def _train(args: argparse.Namespace) -> dict[str, object]:
    subset = Path(args.subset) if args.subset else lr.default_subset()
    out = Path(args.out) if args.out else lr.default_model_dir(args.backend)
    lr.check_output_backend(out, args.backend)
    report = lr.load_corpus(subset, axes=args.axes)
    model = lr.train(report, backend=args.backend, replicates=args.replicates, evaluate_first=not args.skip_scores)
    manifest = lr.save(model, out)
    artifacts = json.loads(manifest.read_text(encoding="utf-8"))["axes"].values()
    size = manifest.stat().st_size + sum(
        (out / axis[key]).stat().st_size for axis in artifacts for key in ("estimator", "fingerprints") if axis[key]
    )
    return {
        "subset": str(subset),
        "rows_read": report.read,
        "pairs_kept": len(report.rows),
        "pairs_from_repeats": report.aggregated,
        "dropped": dict(report.dropped),
        "per_axis": report.per_axis,
        "measurement_scope": report.measurement_scope,
        "covers_uncovered_axes": report.covers_uncovered_axes,
        "backend": model.backend,
        "method": model.method,
        "manifest": str(manifest),
        "artifact_megabytes": round(size / 1e6, 1),
        "scores": {axis: score.reported_metrics() for axis, score in model.scores.items()},
        "axes_still_unanswered": list(model.missing),
        "notes": list(model.notes),
    }


def _predict(args: argparse.Namespace) -> dict[str, object]:
    directory = Path(args.models) if args.models else lr.default_model_dir(args.backend)
    return lr.regressor_report(args.smiles, directory=directory)


def _have(args: argparse.Namespace) -> dict[str, object]:
    """What is on disk, and what each backend still needs."""
    subset = Path(args.subset) if args.subset else lr.default_subset()
    out = Path(args.out) if args.out else lr.default_model_dir(args.backend)
    backends: dict[str, object] = {}
    for name, modules in (("forest", ("sklearn", "joblib")), ("chemprop", ("chemprop", "lightning"))):
        missing = []
        for module in modules:
            try:
                __import__(module)
            except ImportError as error:
                missing.append(f"{module}:{error.__class__.__name__}")
        backends[name] = "ready" if not missing else f"missing {', '.join(missing)}"
    # The scaffold splitter comes from chemprop, so training either backend
    # needs it even though only one backend trains a graph network with it.
    try:
        __import__("chemprop")
        splitter = "ready"
    except ImportError:
        splitter = "absent: training needs chemprop for the scaffold splitter"
    record: dict[str, object] = {
        "subset": str(subset),
        "subset_present": subset.is_file(),
        "model_dir": str(out),
        "backends": backends,
        "scaffold_splitter": splitter,
    }
    try:
        model = lr.load_regressor(out)
    except lr.RegressorUnavailableError as error:
        record["model"] = f"absent: {error}"
        return record
    record["model"] = {
        "backend": model.backend,
        "method": model.method,
        "axes": list(model.covered),
        "axes_still_unanswered": list(model.missing),
        "scores": {axis: score.reported_metrics() for axis, score in model.scores.items()},
    }
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="fit one regressor per axis and report its held-out error")
    train.add_argument("--subset", help="bindingdb_panel_subset.tsv; defaults under DATA_PATH")
    train.add_argument("--out", help="directory for the models; defaults to the backend's directory under DATA_PATH")
    train.add_argument("--axes", nargs="+", choices=lr.UNCOVERED_AXES, default=lr.UNCOVERED_AXES)
    train.add_argument(
        "--backend",
        choices=lr.BACKENDS,
        default="forest",
        help="forest (default; measured better on all three axes) or chemprop",
    )
    train.add_argument("--replicates", type=int, default=lr.REPLICATES, help="scaffold splits to pool")
    train.add_argument(
        "--skip-scores",
        action="store_true",
        help="fit without measuring held-out error -- the resulting model cannot say what it is worth",
    )
    train.set_defaults(run=_train)

    predict = sub.add_parser("predict", help="predicted nanomolar for one SMILES, with its error and reach")
    predict.add_argument("--smiles", required=True)
    predict.add_argument("--models", help="directory holding ligand_regressor.json")
    predict.add_argument("--backend", choices=lr.BACKENDS, default="forest", help="select the default model directory")
    predict.set_defaults(run=_predict)

    have = sub.add_parser("have", help="what is on disk and what each backend needs")
    have.add_argument("--subset")
    have.add_argument("--out")
    have.add_argument("--backend", choices=lr.BACKENDS, default="forest")
    have.set_defaults(run=_have)

    args = parser.parse_args(argv)
    needs_defaults = not args.models if args.command == "predict" else not (args.subset and args.out)
    if needs_defaults and not os.environ.get("DATA_PATH"):
        _ensure_data_path()
    try:
        result = args.run(args)
    except (lr.RegressorCorpusError, lr.RegressorModelError, lr.RegressorUnavailableError, ValueError) as error:
        print(json.dumps({"error": str(error)}))
        return 2
    print(json.dumps(result, indent=2, default=str, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
