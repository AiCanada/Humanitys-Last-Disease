"""Other post-generation producers and the evidence they publish for experimental review."""

from __future__ import annotations

import json
import math
import os
import shutil
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property
from importlib import metadata, util
from pathlib import Path


@dataclass(frozen=True)
class Source:
    key: str
    label: str
    provider: str
    kind: str = ""
    metric: str = ""
    field: str = ""
    priority: bool = False


# A field names a numerical observation in a saved verdict. A metric names the exact
# producer prefix in the published table. Standalone results can be attached by method ID.
SOURCES = (
    Source(
        "posebusters",
        "PoseBusters docking and ligand checks",
        "posebusters",
        "ligand",
        "posebusters_metrics",
        priority=True,
    ),
    Source("dockq", "DockQ", "dockq", "polymer"),
    Source("prolif", "ProLIF typed interactions", "prolif", "ligand"),
    Source("geometric_fingerprint", "Geometric interaction fingerprint", "biotite", "ligand"),
    Source("fnat", "Fraction of native interface contacts", "biotite", "polymer"),
    Source("interface_geometry", "Biotite interface geometry", "biotite", "polymer"),
    Source(
        "aggrescan", "AGGRESCAN sequence score", "builtin", "polymer", field="modality_screen.aggrescan.na4vss",
        priority=True,
    ),
    Source("aggrescan3d", "Aggrescan3D static structure score", "aggrescan3d", "polymer"),
    Source("octant", "Octant exact measurement lookup", "octant", "ligand"),
    Source("chembl_panel", "ChEMBL measured safety panel", "web_client", "ligand"),
    Source("bindingdb_exact", "BindingDB exact measurement lookup", "bindingdb", "ligand"),
    Source("bindingdb_fingerprint", "BindingDB fingerprint lookup", "bindingdb", "ligand"),
    Source("gtopdb", "GtoPdb secondary pharmacology", "gtopdb", "ligand"),
    Source("bayesherg", "BayeshERG", "bayesherg", "ligand"),
    Source("boltz_affinity", "Boltz-2 supplied affinity result reader", "builtin", "ligand"),
    Source("foldseek_search", "Foldseek structural search", "foldseek_search", "polymer"),
    Source("foldseek_clustering", "Foldseek structural clustering", "foldseek", "polymer"),
    Source("mmseqs", "MMseqs2 human proteome search", "mmseqs", "polymer"),
    Source("gravy", "GRAVY mean hydropathy", "builtin", "polymer", field="gravy", priority=True),
    Source("instability", "Instability index", "builtin", "polymer", field="instability_index"),
    Source(
        "protein_composition",
        "Protein composition and mass",
        "builtin",
        "polymer",
        field="modality_screen.molecular_weight",
        priority=True,
    ),
    Source(
        "protein_charge",
        "Protein charge and isoelectric point",
        "builtin",
        "polymer",
        field="chemical_liability.isoelectric_point",
        priority=True,
    ),
    Source("sequence_diversity", "Sequence diversity", "builtin", "polymer", priority=True),
    Source("anarcii", "ANARCII numbering", "anarcii", "polymer"),
    Source("antibody_chemistry", "Antibody chemical liability motifs", "anarcii", "polymer"),
    Source("deepsp", "DeepSP formulation descriptors", "deepsp", "polymer"),
    Source("deepviscosity", "DeepViscosity", "deepviscosity", "polymer"),
    Source("bindingdb_chemprop", "BindingDB Chemprop D-MPNN", "chemprop", "ligand"),
    Source("bindingdb_forest", "BindingDB ExtraTrees regressor", "forest", "ligand"),
    Source(
        "ligand_composition",
        "Ligand component composition",
        "builtin",
        "ligand",
        field="modality_screen.formula_weight",
        priority=True,
    ),
    Source("openmm", "OpenMM single-structure potential difference", "openmm", "polymer"),
    Source("openfe", "OpenFE completed free-energy result reader", "builtin", "ligand"),
    Source("alchemlyb", "alchemlyb thermodynamic integration", "alchemlyb", "ligand"),
    Source("pmx", "pmx completed work and BAR result reader", "pymbar", "ligand"),
    Source("dynamics", "Trajectory dynamics", "dynamics"),
    Source("hpa", "Human Protein Atlas plasma abundance", "web_client"),
    Source("string_network", "STRING physical network lookup", "web_client"),
    Source("uniref", "UniRef50 human cluster members", "web_client"),
    Source("ensembl_paralog", "Ensembl Compara paralog lookup", "web_client"),
    Source("plasma_fold", "Supplied plasma fold change", "builtin"),
    Source("psea", "P-SEA secondary structure", "structure_metrics", "polymer", "sse_distribution", priority=True),
    Source("lddt", "Local distance difference test", "structure_metrics", metric="structure_prediction_lddt"),
    Source("ca_rmsd", "C-alpha RMSD to input", "structure_metrics", "polymer", "ca_rmsd_to_input", "ca_rmsd"),
    Source("cath_s95", "CATH S95 domain classification", "cath", "polymer"),
    Source("gearnet", "GearNet CATH classifier", "gearnet", "polymer"),
    Source("stereochemistry", "Stereochemistry", "structure_metrics", metric="chiral_accuracy", priority=True),
    Source("ccd_geometry", "CCD internal bond geometry", "structure_metrics", metric="internal_residue_bond"),
    Source(
        "clashes",
        "Steric clash fraction",
        "structure_metrics",
        metric="clash_metrics",
        field="clash_fraction",
        priority=True,
    ),
    Source("bond_length", "Element-dependent bond lengths", "structure_metrics", metric="bond_length_metrics",
           priority=True),
    Source(
        "contact_surface",
        "Contact molecular surface",
        "structure_metrics",
        metric="contact_molecular_surface",
        field="contact_ms",
    ),
    Source("toxicology_table", "Supplied toxicology concentration reader", "builtin", "ligand"),
)
BY_KEY = {source.key: source for source in SOURCES}
ATTACHMENT = "screening_observations.json"
_MAX_RECORD = 16 * 1024 * 1024


def _file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _modules(*names: str) -> bool:
    return all(util.find_spec(name) is not None for name in names)


def _configured_file(variable: str, suffix: str = "") -> bool:
    path = os.environ.get(variable)
    return bool(path and _file(Path(path + suffix)))


def _regressor(backend: str) -> bool:
    from rfproteina.disease.ligand_regressor import RegressorUnavailableError, default_model_dir, regressor_manifest

    try:
        _, _, actual = regressor_manifest(default_model_dir(backend))
    except (OSError, RuntimeError, ValueError, RegressorUnavailableError):
        return False
    return actual == backend and _modules("rdkit", "chemprop" if backend == "chemprop" else "sklearn")


def _provider(name: str) -> bool:
    from rfproteina.disease import deepsp, deepviscosity, interactions, novelty, octant, offtarget, tables

    if name in {"builtin", "web_client"}:
        return True
    if name == "posebusters":
        from rfproteina.utils.pose_checks import molecule_checks_available

        return molecule_checks_available()
    if name in {"biotite", "openmm", "alchemlyb", "prolif"}:
        return _modules(name)
    if name == "pymbar":
        return _modules("pymbar", "numpy", "scipy")
    if name == "structure_metrics":
        return _modules("atomworks", "biotite", "torch")
    if name == "dynamics":
        return _modules("numpy", "biotite")
    if name == "dockq":
        return interactions.dockq_available()[0]
    if name in {"foldseek", "foldseek_search"}:
        binary = novelty.foldseek_available()
        if name == "foldseek" or not binary:
            return binary
        if os.environ.get("FOLDSEEK_DB"):
            return _configured_file("FOLDSEEK_DB", ".dbtype") and _configured_file("FOLDSEEK_DB")
        data = os.environ.get("DATA_PATH")
        banks = Path(data) / "foldseek" / "databases" if data else None
        return bool(
            banks
            and any(
                _file(prefix) and _file(prefix.with_suffix(".dbtype"))
                for folder in banks.glob("*")
                if folder.is_dir()
                for prefix in (folder / folder.name,)
            )
        )
    if name == "mmseqs":
        return offtarget.mmseqs_available() and (
            _configured_file("HUMAN_PROTEOME") or tables.downloaded_proteome() is not None
        )
    if name == "bindingdb":
        data = os.environ.get("DATA_PATH")
        return bool(data and _modules("rdkit") and _file(Path(data) / "bindingdb" / tables.BINDINGDB_PANEL_SUBSET_FILE))
    if name == "gtopdb":
        data = os.environ.get("DATA_PATH")
        return bool(data and _modules("rdkit") and _file(Path(data) / "gtopdb" / "gtopdb_offtarget_subset.tsv"))
    if name == "octant":
        return _modules("rdkit") and _file(octant.octant_directory() / octant.SUBSET_FILE)
    if name in {"chemprop", "forest"}:
        return _regressor(name)
    if name == "gearnet":
        return _modules("torch") and all(
            _configured_file(env) for env in ("GEARNET_CATH_CKPT_PATH", "GEARNET_CATH_LABEL_MAPPING_PATH")
        )
    if name == "cath":
        return bool(shutil.which(os.environ.get("FOLDSEEK_EXEC", "foldseek"))) and (
            _configured_file("CATH_FOLDSEEK_DB", ".dbtype") and _configured_file("CATH_DOMAIN_LIST")
        )
    if name == "bayesherg":
        root = os.environ.get("BAYESHERG_ROOT")
        runtime = os.environ.get("BAYESHERG_PYTHON")
        return bool(
            root
            and runtime
            and os.access(runtime, os.X_OK)
            and all(_file(Path(root) / p) for p in ("main.py", "model/model_weights.pth", "model/BayeshERG_model.py"))
        )
    if name in {"deepsp", "deepviscosity"}:
        root = deepsp.default_models_dir()
        if not root or not _modules("torch", "h5py", "anarcii"):
            return False
        if not all(_file(root / f) for files in deepsp.DEEPSP_WEIGHTS.values() for f in files):
            return False
        if name == "deepsp":
            return True
        root = deepviscosity.default_models_dir()
        return bool(root and all(_file(root / f) for f in (*deepviscosity.model_names(), deepviscosity.SCALER_NAME)))
    if name == "anarcii":
        dist = metadata.distribution("anarcii")
        return _modules("anarcii", "torch") and any(
            p.suffix in (".pt", ".pth") and _file(Path(dist.locate_file(p))) for p in dist.files or ()
        )
    if name == "aggrescan3d":
        return bool(shutil.which("aggrescan") or shutil.which("a3d"))
    raise KeyError(name)


def installations(keys: set[str] | None = None) -> dict[str, bool]:
    sources = [s for s in SOURCES if keys is None or s.key in keys]
    states = {}
    for provider in dict.fromkeys(s.provider for s in sources):
        try:
            states[provider] = _provider(provider)
        except (OSError, ValueError, ImportError, RuntimeError):
            states[provider] = False
    return {s.key: states[s.provider] for s in sources}


def _json(path: Path) -> dict:
    try:
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_size > _MAX_RECORD:
            return {}
        with path.open("rb") as stream:
            raw = stream.read(_MAX_RECORD + 1)
        value = json.loads(raw) if len(raw) <= _MAX_RECORD else None
    except (OSError, ValueError, RecursionError):
        return {}
    return value if isinstance(value, dict) else {}


def _numeric(value: object) -> bool:
    try:
        return not isinstance(value, bool) and math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


class Evidence:
    """Read each published source once. Return coverage without pooling scales or exposing candidate sequences."""

    def __init__(self, directory: Path, example: str | None, record: Mapping | None = None):
        from rfproteina.disease.screen import _metrics_path, _metrics_table

        self.directory = directory
        rows = (record if record is not None else _json(directory / "screen.json")).get("samples")
        self.samples = (
            [row for row in rows if isinstance(row, dict) and (example is None or row.get("example_id") == example)]
            if isinstance(rows, list)
            else []
        )
        path = _metrics_path(directory)
        self.metrics = []
        if path is not None and self.safe_table(path):
            metrics, blocker = _metrics_table(directory, path)
            if not blocker:
                self.metrics = [row for row in metrics if example is None or row.get("example_id") == example]
        self.attached = _json(directory / ATTACHMENT)
        self.example = example

    @cached_property
    def candidate_ids(self) -> set[str] | None:
        from rfproteina.disease.validation_checks import _candidates, _inventory

        candidates = _candidates(_inventory(self.directory))
        return {c.example_id for c in candidates if c.example_id} if candidates else None

    @staticmethod
    def safe_table(path: Path) -> bool:
        try:
            info = path.lstat()
            return stat.S_ISREG(info.st_mode) and info.st_size <= _MAX_RECORD and path.resolve() == path.absolute()
        except (OSError, RuntimeError):
            return False

    def coverage(self, source: Source, *, include_values: bool = False) -> dict:
        observations = []
        unmatched = 0

        def add(candidate: object, endpoint: str, value: object, origin: str, unit: str | None = None) -> int:
            nonlocal unmatched
            if not isinstance(candidate, str) or not candidate.strip() or (
                self.candidate_ids is not None and candidate not in self.candidate_ids
            ):
                unmatched += 1
                return 0
            if include_values and len(observations) < 512:
                row = {"candidate_id": candidate, "endpoint": endpoint, "value": float(value), "source": origin}
                if unit is not None:
                    row["unit"] = unit
                observations.append(row)
            return 1

        fields = 0
        if source.field:
            for sample in self.samples:
                value = sample
                for part in source.field.split("."):
                    value = value.get(part) if isinstance(value, dict) else None
                if _numeric(value):
                    fields += add(sample.get("example_id"), source.field, value, "screen.json")
        cells = 0
        if source.metric:
            for row in self.metrics:
                for key, value in row.items():
                    if key.startswith(source.metric + ".") and _numeric(value):
                        cells += add(row.get("example_id"), key, value, "published_metrics")
        raw = self.attached.get(source.key)
        count = 0
        if isinstance(raw, list):
            for row in raw:
                if not isinstance(row, dict) or not all(
                    isinstance(row.get(k), str) and 0 < len(row[k].strip()) <= 256
                    for k in ("candidate_id", "endpoint", "unit")
                ):
                    continue
                if (self.example is None or row["candidate_id"] == self.example) and _numeric(row.get("value")):
                    count += add(row["candidate_id"], row["endpoint"], row["value"], ATTACHMENT, row["unit"])
        total = count + cells + fields
        result = {
            "method": source.key,
            "status": "recorded" if total else "untested",
            "sample_observations": fields,
            "metric_cells": cells,
            "attached_observations": count,
        }
        if unmatched:
            result["unmatched_observations"] = unmatched
        if total and self.candidate_ids is None:
            result["candidate_scope_unverified"] = True
        if include_values:
            result.update(observations=observations, omitted_observations=total - len(observations))
        return result

    def has_chain(self) -> bool:
        return any(
            isinstance(row.get("chain_of_interest"), str)
            and row["chain_of_interest"].strip().lower() not in {"", ".", "?", "nan", "none", "null"}
            for row in self.metrics
        )
