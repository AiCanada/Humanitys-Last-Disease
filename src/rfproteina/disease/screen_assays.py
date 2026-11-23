"""Dispatch existing local analytical readers on explicitly supplied experimental inputs."""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

from rfproteina.disease import ligand_inputs

# The first existing file is authoritative, including an unreadable file.
FILES = {
    "posebusters": ("ligand_candidates.sdf",),
    "octant": ("ligand_candidates.tsv", "ligand_smiles.txt"),
    "bindingdb_exact": ("ligand_candidates.tsv", "ligand_smiles.txt"),
    "bindingdb_fingerprint": ("ligand_candidates.tsv", "ligand_smiles.txt"),
    "gtopdb": ("ligand_candidates.tsv", "ligand_smiles.txt"),
    "bindingdb_forest": ("ligand_candidates.tsv", "ligand_smiles.txt"),
    "bindingdb_chemprop": ("ligand_candidates.tsv", "ligand_smiles.txt"),
    "toxicology_table": ("toxicology.tsv", "toxicology.csv"),
    "openfe": ("openfe_inputs.json",),
    "pmx": ("pmx_inputs.json",),
}
PURPOSE = {
    "posebusters": "Check the internal geometry of the nominated 3D ligand with PoseBusters in molecule mode.",
    "octant": "Look up the nominated compound in the installed Octant CYP3A4 measurements.",
    "bindingdb_exact": "Read BindingDB measurements matched to the nominated compound's full InChIKey.",
    "bindingdb_fingerprint": "Review measured BindingDB neighbours with their fingerprint similarities.",
    "gtopdb": "Review reported secondary pharmacology for similar compounds from GtoPdb.",
    "bindingdb_forest": "Review panel estimates and training-set similarity from the installed ExtraTrees model.",
    "bindingdb_chemprop": "Review panel estimates and training-set similarity from the installed Chemprop model.",
    "toxicology_table": "Read the supplied panel concentrations with their units and qualifiers.",
    "openfe": "Verify the supplied completed OpenFE complex and solvent legs.",
    "pmx": "Verify one completed pmx BAR report against both sets of work measurements.",
}
LOOKUPS = {"octant", "bindingdb_exact", "bindingdb_fingerprint", "gtopdb"}
EXACT_LOOKUPS = {"octant", "bindingdb_exact"}
REGRESSORS = {"bindingdb_forest": "forest", "bindingdb_chemprop": "chemprop"}
MOLECULAR_QUERIES = LOOKUPS | REGRESSORS.keys()
READERS = {"openfe", "pmx", "toxicology_table"} | LOOKUPS


def _unique(pairs: list) -> dict:
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("The input manifest contains a duplicate field.")
    return result


def inputs(directory: Path, key: str) -> dict:
    path = directory / FILES[key][0]
    if path.is_symlink() or not path.is_file():
        raise ValueError("The result manifest must be a regular local file.")
    with path.open("rb") as stream:
        raw = stream.read(64 * 1024 + 1)
    if len(raw) > 64 * 1024:
        raise ValueError("The result manifest exceeds 64 KiB.")
    entry = json.loads(raw, object_pairs_hook=_unique)
    expected = (
        {"candidate_id", "ligand_a", "ligand_b", "results"}
        if key == "openfe"
        else {"candidate_id", "report", "forward", "reverse"}
    )
    if not isinstance(entry, dict) or entry.keys() != expected:
        raise ValueError("The result manifest must identify the candidate and the required source files.")
    for name in expected - {"results"}:
        if not isinstance(entry[name], str) or entry[name].strip() != entry[name] or not 0 < len(entry[name]) <= 255:
            raise ValueError("The result manifest contains an invalid identifier.")
    if key == "openfe":
        names = entry["results"]
        if not isinstance(names, list) or not 2 <= len(names) <= 32:
            raise ValueError("Supply between two and 32 completed OpenFE result files.")
    else:
        names = [entry["report"], entry["forward"], entry["reverse"]]
    if not all(
        isinstance(n, str) and n and len(n) <= 255 and n not in {".", ".."} and not any(c in n for c in "/\\:\x00")
        for n in names
    ) or len(set(names)) != len(names):
        raise ValueError("Supply distinct result filenames directly inside the run.")
    total = 0
    for name in names:
        data = directory / name
        if data.is_symlink() or not data.is_file():
            raise ValueError("A required result file is missing or is not a regular file.")
        size = data.stat().st_size
        if not size:
            raise ValueError("A required result file is empty.")
        total += size
    if total > 256 * 1024 * 1024:
        raise ValueError("The completed result files exceed the 256 MiB interactive input limit.")
    return entry


def _result_rows(key: str, samples: list[dict]) -> dict:
    completed = sum("blocker" not in row for row in samples)
    status = "untested"
    if completed:
        incomplete = key == "ligand_alerts" and any(row.get("missing_catalogues") for row in samples)
        status = "reported" if completed == len(samples) and not incomplete else "partial"
    return {"method": key, "status": status, "attempted": len(samples), "measured": completed, "samples": samples}


def _lookup(key: str, smiles: str) -> dict:
    if key == "ligand_alerts":
        from rfproteina.disease import ligand_alerts

        try:
            screen = ligand_alerts.screen_smiles(smiles, nominated=True)
        except ligand_alerts.LigandUnreadableError:
            return {"blocker": "invalid_nominated_smiles"}
        return {**screen.as_record(), "verdicts": ligand_alerts.verdicts(screen)}

    from rfproteina.disease import ligand_safety, octant
    from rfproteina.disease.ligand_regressor import default_subset

    if key == "gtopdb":
        subset = Path(os.environ["DATA_PATH"]) / "gtopdb" / "gtopdb_offtarget_subset.tsv"
        profile = ligand_safety.offtarget_profile(smiles, subset)
        if profile.analogues:
            return {
                "provider_method": profile.method,
                "reference_compounds": profile.analogues,
                "reference_target_families": profile.families,
            }
    elif key == "bindingdb_fingerprint":
        profile = ligand_safety.nearest_measured_neighbours(smiles, default_subset())
        if any(profile.neighbours.values()):
            return {
                "provider_method": profile.method,
                "reference_neighbours": profile.neighbours,
                "fingerprint": profile.fingerprint,
                "min_similarity": profile.min_similarity,
            }
    else:
        if key == "octant":
            subset = octant.installed_subset()
            if subset is None:
                raise ligand_safety.SafetyBackendUnavailableError("octant_subset_absent")
            panel = octant.measured_nm_from_octant(smiles, subset)
        else:
            panel = ligand_safety.measured_nm_from_bindingdb(smiles, default_subset())
        if any(r.get("value_nm") is not None or r.get("opinions") for r in panel.axes.values()):
            return {"provider_method": panel.method, "unit": panel.unit, "axes": panel.axes}
    return {"blocker": "no_matching_reference_measurements"}


def _ligand_rows(directory: Path, key: str, example: str | None) -> dict:
    from rfproteina.disease import ligand_regressor

    model = None
    if key in REGRESSORS:
        try:
            model = ligand_regressor.load_regressor(ligand_regressor.default_model_dir(REGRESSORS[key]))
        except (ligand_regressor.RegressorUnavailableError, ligand_regressor.RegressorModelError, ValueError, OSError):
            return {"method": key, "status": "untested", "blocker": "trained_model_unavailable"}
    if key == "posebusters":
        from rfproteina.disease.pose_validation import (
            PoseBustersResultError,
            PoseBustersUnavailableError,
            PoseUnreadableError,
            read_nominated_pose,
            run_posebusters,
        )

        rows = list(ligand_inputs.nominated_rows(ligand_inputs.read_sdf_candidates(directory / FILES[key][0]), example))
    else:
        from rfproteina.disease.ligand_regressor import RegressorCorpusError
        from rfproteina.disease.ligand_safety import LigandUnreadableError, SafetyBackendUnavailableError

        _, rows, problem = ligand_inputs.nominated_inputs(directory, "smiles", example)
        if problem:
            return {"method": key, "status": "untested", "blocker": problem}

    counts = Counter(row[0] for row in rows if row)
    samples = []
    for row in rows:
        name = row[0] if row else ""
        sample = {"candidate_id": name}
        samples.append(sample)
        if not name or counts[name] != 1 or len(row) != (3 if key == "posebusters" else 2):
            sample["blocker"] = "invalid_or_duplicate_nomination"
        elif key in MOLECULAR_QUERIES and not _valid_nomination(row[1]):
            sample["blocker"] = "invalid_nominated_smiles"
        elif key == "posebusters":
            try:
                mode, checks = run_posebusters(read_nominated_pose(row[1], row[2]))
                sample.update(mode=mode, checks=checks)
                if not checks or any(value is None for value in checks.values()):
                    sample["blocker"] = "pose_checks_incomplete"
            except PoseUnreadableError as error:
                sample["blocker"] = str(error)
            except (
                PoseBustersResultError,
                PoseBustersUnavailableError,
                RuntimeError,
                ValueError,
                TimeoutError,
            ):
                sample["blocker"] = "ligand_geometry_unavailable"
        elif model is not None:
            try:
                predictions = ligand_regressor.predict(model, row[1])
                if predictions.nanomolar:
                    sample.update(
                        provider_method=predictions.method,
                        unit=predictions.unit,
                        predictions=predictions.predictions,
                        uncovered_axes=predictions.uncovered_axes,
                    )
                else:
                    sample["blocker"] = "no_attributable_prediction"
            except (ligand_regressor.RegressorModelError, ligand_regressor.RegressorUnavailableError, ValueError):
                sample["blocker"] = "compound_prediction_unavailable"
        else:
            try:
                sample.update(_lookup(key, row[1]))
            except (
                LigandUnreadableError,
                SafetyBackendUnavailableError,
                RegressorCorpusError,
                KeyError,
                OSError,
                ValueError,
            ):
                sample["blocker"] = "compound_or_reference_unreadable"
    return _result_rows(key, samples)


def _valid_nomination(smiles: str) -> bool:
    from rfproteina.disease import ligand_alerts

    try:
        ligand_alerts.parse_molecule(smiles, nominated=True)
    except ligand_alerts.LigandUnreadableError:
        return False
    return True


def supplied_panel(directory: Path) -> dict:
    from rfproteina.disease.toxicology import measure_toxicology

    _, rows, problem = ligand_inputs.nominated_inputs(directory, "smiles", None, max_bytes=16 * 1024 * 1024)
    smiles = None
    if not problem and len(rows) == 1 and len(rows[0]) == 2 and rows[0][0] and _valid_nomination(rows[0][1]):
        smiles = rows[0][1]
    readings = measure_toxicology(directory, ligand_smiles=smiles)
    measured = 0
    complete = True
    attributable = bool(smiles)
    for record in readings.values():
        opinions = record.get("opinions", [record])
        values = [row for row in opinions if row.get("measurement") is not None]
        measured += bool(values)
        attributable &= all(row.get("inchikey") and row.get("assay_source") and row.get("endpoint") for row in values)
        complete &= all(row.get("measurement") is not None or row.get("blocker") == "toxicology_not_reported"
                        for row in opinions)
    return {
        "method": "toxicology_table", "status": ("reported" if complete else "partial") if measured else "untested",
        "measured_axes": measured, "attributed_assays": bool(measured and attributable), "axes": readings,
    }


def measure(directory: Path, key: str, example: str | None) -> dict:
    if key in {"posebusters", "ligand_alerts"} or key in LOOKUPS or key in REGRESSORS:
        return _ligand_rows(directory, key, example)
    if key == "toxicology_table":
        return supplied_panel(directory)
    entry = inputs(directory, key)
    if example is not None and entry["candidate_id"] != example:
        raise ValueError("The result manifest names another candidate.")
    if key == "openfe":
        from rfproteina.disease.openfe_results import read_openfe_rbfe

        record = read_openfe_rbfe(entry["ligand_a"], entry["ligand_b"], [directory / n for n in entry["results"]])
    else:
        from rfproteina.disease.pmx_results import read_pmx_bar

        record = read_pmx_bar(*(directory / entry[n] for n in ("report", "forward", "reverse")))
    return {"method": key, "status": "reported", "candidate_id": entry["candidate_id"], "result": record}


def safe_measure(directory: Path, key: str, example: str | None) -> dict:
    try:
        return measure(directory, key, example)
    except ValueError as error:
        from rfproteina.disease.publish import scrub_for_upload

        return {
            "method": key,
            "status": "untested",
            "blocker": "input_validation_failed",
            "detail": scrub_for_upload(str(error))[:256],
        }
    except (ImportError, OSError, RuntimeError, OverflowError):
        return {"method": key, "status": "untested", "blocker": "prepared_input_or_provider_unavailable"}
