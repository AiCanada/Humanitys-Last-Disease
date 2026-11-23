"""Plan optional post-generation measurements from local installations and run inputs."""

from __future__ import annotations

import csv
import os
import stat
import subprocess
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from importlib import util
from pathlib import Path

from rfproteina.disease import (
    ligand_alerts,
    ligand_solubility,
    qupkake,
    screen_assays,
    screen_criteria,
    screen_inventory,
    screen_physical,
    sygma_engine,
    tnp,
    vina_pose,
)


@dataclass(frozen=True)
class Method:
    key: str
    label: str
    kinds: tuple[str, ...]
    inputs: tuple[str, ...]
    workload: str
    purpose: str
    default: bool = True
    operation: str = "compute"


PROTEIN = ("protein", "peptide")
LIGAND = ("ligand",)
SMILES = ("ligand_candidates.tsv", "ligand_smiles.txt")
METHODS = (
    Method(
        "ligand_alerts",
        "RDKit assay-interference alerts and descriptors",
        LIGAND,
        SMILES,
        "CPU descriptors",
        "Flag assay-interference motifs before testing a nominated compound.",
    ),
    Method(
        "ligand_solubility",
        "ADMET-AI aqueous solubility",
        LIGAND,
        SMILES,
        "CPU model ensemble",
        "Estimate aqueous logS for planning a solubility measurement.",
    ),
    Method(
        "qupkake",
        "QupKake microscopic pKa",
        LIGAND,
        ("qupkake.json",),
        "Quantum chemistry and CPU models",
        "Compare site ionization with a pH-dependent assay.",
    ),
    Method(
        "sygma",
        "SyGMa human metabolite prediction",
        LIGAND,
        ("sygma.json",),
        "CPU reaction enumeration",
        "Identify possible metabolites for analytical confirmation.",
    ),
    Method(
        "vina_pose",
        "AutoDock Vina supplied-pose score",
        LIGAND,
        ("vina_poses.json",),
        "CPU pose scoring",
        "Review an explicitly prepared small-molecule pose.",
        False,
    ),
    Method(
        "prodigy_affinity",
        "PRODIGY supplied protein-pair estimate",
        PROTEIN,
        ("prodigy_partners.json",),
        "CPU structure calculation",
        "Review the two partners identified in the run's settings.",
        False,
    ),
    Method(
        "tnp_profile",
        "Therapeutic Nanobody Profiler",
        ("protein",),
        ("tnp_domains.json",),
        "CPU model ensemble and structure refinement",
        "Characterize an explicitly identified VHH domain.",
        False,
    ),
    Method(
        "protein_sol",
        "Protein-Sol result reader",
        PROTEIN,
        ("seq_prediction.txt",),
        "Local result reader",
        "Match native expression-solubility predictions to the corresponding sample.",
        operation="native_result_reader",
    ),
    Method(
        "solubility",
        "Supplied solubility and aggregation measurements",
        (),
        ("solubility.tsv", "solubility.csv"),
        "Local result reader",
        "Review the supplied method, study and measurement scale.",
        operation="native_result_reader",
    ),
)
NATIVE_IDS = tuple(method.key for method in METHODS)
_PURPOSES = {
    "dockq": "Review stored DockQ with the reference complex and the model-to-reference chain assignment.",
    "prolif": "Review stored ProLIF interaction types with their ligand pose, partner and frame.",
    "geometric_fingerprint": "Review stored geometric contacts with the ligand pose and distance definition.",
    "fnat": "Review the native contact fraction with its reference interface and chain correspondence.",
    "interface_geometry": "Review the stored interface measurement with the identities of both partners.",
    "chembl_panel": "Review the recorded ChEMBL assay, compound identity, target and concentration units.",
    "bayesherg": "Review the stored BayeshERG model output with its release and endpoint scale.",
    "boltz_affinity": "Review the stored Boltz affinity endpoint with its ligand, target and output scale.",
    "foldseek_search": "Review the recorded structural match with the query chain and reference database.",
    "foldseek_clustering": "Review the recorded Foldseek cluster against its input cohort and clustering settings.",
    "mmseqs": "Review the recorded alignment with its query chain, coverage and human reference release.",
    "anarcii": "Review the stored domain annotation with its chain identity and numbering scheme.",
    "antibody_chemistry": (
        "Review the stored chemical observations with the annotated antibody domain and numbering scheme."
    ),
    "deepsp": "Review stored DeepSP descriptors with their model release and paired-chain identities.",
    "deepviscosity": "Review the recorded viscosity endpoint with its concentration and formulation conditions.",
    "openmm": "Review the recorded potential-energy difference with its structures, force field and units.",
    "alchemlyb": "Review thermodynamic integration with the transformation, simulation states and uncertainty.",
    "dynamics": "Review stored trajectory statistics with the atom selection, trajectory and sampling interval.",
    "hpa": "Review recorded plasma abundance with the target protein, assay and concentration units.",
    "string_network": "Review recorded physical-network evidence with the protein identifiers and STRING release.",
    "uniref": "Review the recorded human cluster membership with the input accession and UniRef release.",
    "ensembl_paralog": (
        "Review the recorded paralog relationship with its species, gene identifiers and Compara release."
    ),
    "plasma_fold": "Review the recorded plasma change with the target analyte, comparator and fold-change scale.",
    "lddt": "Review the recorded local distance score with its structure pair, atom selection and scale.",
    "ca_rmsd": "Review C-alpha displacement with the input structure, alignment and matched residue set.",
    "cath_s95": "Review the recorded domain assignment with its boundaries and CATH reference release.",
    "gearnet": "Review the stored domain class with the model checkpoint and class mapping.",
    "ccd_geometry": "Review stored internal bond geometry with the component identifier and CCD reference.",
    "contact_surface": "Review the stored contact surface with its two partners, surface definition and units.",
    "gravy": "Measure mean hydropathy of the designed chain for comparison with handling and solubility observations.",
    "instability": "Report the sequence instability index alongside experimental stability measurements.",
    "protein_composition": "Calculate residue counts and the expected unmodified mass for identity checks.",
    "protein_charge": "Estimate the designed chain's isoelectric point and charge at pH 7.4.",
    "aggrescan": "Report the sequence aggregation score alongside aggregation assay results.",
    "psea": "Measure helix, sheet and loop fractions in the supplied coordinates.",
    "clashes": "Check the supplied coordinates for unusually close atomic contacts.",
    "bond_length": "Check observed bond lengths against the element-dependent geometry rules.",
    "ligand_composition": "Read the ligand component's formula weight for chemical identity checks.",
    "sequence_diversity": "Count distinct designed chains before choosing replicate samples for a bench panel.",
}
_REQUIREMENTS = {
    "ligand_alerts": "Requires RDKit's molecular descriptors and FilterCatalog support.",
    "aggrescan3d": "Requires the Aggrescan3D executable on the service path.",
    "bayesherg": "Requires release weights under BAYESHERG_ROOT and the BAYESHERG_PYTHON interpreter.",
    "bindingdb_chemprop": "Requires Chemprop and the trained graph-model files in the configured model directory.",
    "alchemlyb": "Requires an importable alchemlyb installation in this runtime.",
    "pmx": "Requires PyMBAR, NumPy and SciPy in the result reader's runtime.",
    "cath_s95": "Requires Foldseek, the CATH S95 database and its domain list.",
    "gearnet": "Requires PyTorch, the GearNet classifier checkpoint and its label mapping.",
    "qupkake": "Requires the configured QupKake runtime, microscopic pKa checkpoints and xTB.",
    "tnp_profile": "Requires the TNP runtime, NanoBodyBuilder2 checkpoints and DSSP.",
    "ligand_solubility": "Requires the ADMET-AI regression ensemble and its endpoint definitions.",
    "sygma": "Requires RDKit and the released SyGMa rule tables.",
}
METHODS += tuple(
    Method(
        source.key,
        source.label,
        PROTEIN if source.kind == "polymer" else (source.kind,) if source.kind else (),
        (),
        "CPU structure measurement" if source.key in screen_physical.METHODS else "Stored evidence review",
        _PURPOSES.get(source.key, "Review this producer's recorded observations for the selected candidate."),
        source.priority,
        "structure_measurement" if source.key in screen_physical.METHODS else "recorded_evidence",
    )
    for source in screen_inventory.SOURCES
)
METHOD_IDS = tuple(method.key for method in METHODS)


@dataclass(frozen=True)
class Choice:
    method: Method
    installed: bool
    applicable: bool
    recommended: bool
    reason: str

    def as_record(self) -> dict:
        return {
            "id": self.method.key,
            "label": self.method.label,
            "installed": self.installed,
            "applicable": self.applicable,
            "recommended": self.recommended,
            "reason": self.reason,
            "workload": self.method.workload,
            "purpose": self.method.purpose,
            "operation": self.method.operation,
        }


def _modules(*names: str) -> bool:
    return all(util.find_spec(name) is not None for name in names)


def _worker_ready(python: Path | None, module: str) -> bool:
    if python is None or not os.access(python, os.X_OK):
        return False
    # Keep an isolated environment's Python symlink intact.
    worker = Path(__file__).with_name(module + "_worker.py")
    result = subprocess.run(
        [str(python), "-I", str(worker), "--check-installation"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=12,
        check=False,
    )
    return result.returncode == 0


_BATCH: ContextVar[dict[str, bool] | None] = ContextVar("screening_installations", default=None)


@contextmanager
def installation_batch() -> Iterator[None]:
    """Reuse software discovery while processing one command's batch of finished runs."""
    token = _BATCH.set({})
    try:
        yield
    finally:
        _BATCH.reset(token)


def installations(keys: set[str] | None = None) -> dict[str, bool]:
    """Check the provider, executable and required assets without loading a model or starting a screen."""
    batch = _BATCH.get()
    requested = set(METHOD_IDS) if keys is None else keys
    missing = requested if batch is None else requested.difference(batch)
    if not missing:
        return {k: batch[k] for k in requested} if batch is not None else {}

    def rdkit_alerts() -> bool:
        ligand_alerts.missing_catalogues()
        return True

    def admet() -> bool:
        if not _modules("rdkit", "chemprop", "admet_ai", "torch"):
            return False
        ligand_solubility.verified_assets()
        return True

    def sygma() -> bool:
        from rdkit.Chem import inchi, rdChemReactions, rdMolDescriptors

        return (
            inchi.INCHI_AVAILABLE
            and callable(rdChemReactions.ReactionFromSmarts)
            and callable(rdMolDescriptors.CalcExactMolWt)
            and bool(sygma_engine.installed_rule_tables())
        )

    probes = {
        "ligand_alerts": rdkit_alerts,
        "ligand_solubility": admet,
        "sygma": sygma,
        "qupkake": lambda: _modules("rdkit") and _worker_ready(qupkake._runtime(), "qupkake"),
        "vina_pose": lambda: _modules("gemmi") and bool(vina_pose.installed_executable()),
        "prodigy_affinity": lambda: _modules("Bio", "prodigy_prot", "freesasa"),
        "tnp_profile": lambda: _modules("gemmi") and _worker_ready(tnp._interpreter(), "tnp"),
        "protein_sol": lambda: True,
        "solubility": lambda: True,
    }
    found = {}
    for key, probe in probes.items():
        if key not in missing:
            continue
        try:
            found[key] = probe()
        except (ImportError, AttributeError, OSError, RuntimeError, ValueError, csv.Error, subprocess.SubprocessError):
            found[key] = False
    found.update(screen_inventory.installations(missing))
    if batch is not None:
        batch.update(found)
        return {k: batch[k] for k in requested}
    return found


def _regular(path: Path) -> bool:
    try:
        return stat.S_ISREG(path.lstat().st_mode)
    except OSError:
        return False


_READ_LIMIT = 16 * 1024 * 1024
_RUN_LEVEL = {"toxicology_table"}
_PREPARED_STRUCTURES = {"tnp_profile", "vina_pose", "prodigy_affinity"}
_CRITERIA = {
    "ligand_alerts": screen_criteria.rdkit_alerts,
    "ligand_solubility": screen_criteria.admet_solubility,
}


def _input_reason(directory: Path, method: Method, kind: str, example: str | None) -> str | None:
    if method.kinds and kind not in method.kinds:
        return "Requires a " + " or ".join(method.kinds) + " candidate."
    if example is not None and (not example or method.key in _RUN_LEVEL):
        return "The measurement requires an identifiable candidate or a run-level selection."
    inputs = method.inputs
    if method.key in {"posebusters", "qupkake", "sygma"}:
        return None  # Each provider inspects its manifest and selected molecular source together.
    # A present invalid higher-priority input must never fall through to another candidate source.
    path = next((directory / n for n in inputs if (directory / n).exists() or (directory / n).is_symlink()), None)
    if path is None or not _regular(path):
        return "Requires a regular " + " or ".join(inputs) + " file."
    if path.stat().st_size > _READ_LIMIT:
        return "The supplied input exceeds the size limit for interactive screening."
    if method.key == "protein_sol":
        fasta = directory / "protein_sol.fasta"
        if (fasta.exists() or fasta.is_symlink()) and not screen_inventory.Evidence.safe_table(fasta):
            return "Requires a readable regular protein_sol.fasta for sample attribution."
    return None


def plan(
    directory: Path,
    kind: str,
    *,
    example: str | None = None,
    available: dict[str, bool] | None = None,
    evidence: screen_inventory.Evidence | None = None,
) -> tuple[Choice, ...]:
    """Plan the supplied installation inventory, or discover the full catalogue when it is omitted."""
    from rfproteina.disease.screen import _canonical_kind

    kind = _canonical_kind(kind)
    available = installations() if available is None else available
    choices = []
    evidence = evidence if evidence is not None else screen_inventory.Evidence(directory, example)
    from rfproteina.disease.validation_checks import _candidates, _inventory

    needs_structure = screen_physical.METHODS | _PREPARED_STRUCTURES | {"solubility", "protein_sol", "posebusters"}
    candidates = () if needs_structure.isdisjoint(available) else _candidates(_inventory(directory))
    has_structure = any(c.example_id and (example is None or c.example_id == example) for c in candidates)
    candidate = next((c for c in candidates if c.example_id == example), None) if example else None
    structures = screen_physical.Structures(directory, candidate)
    for definition in METHODS:
        if definition.key not in available:
            continue
        method = definition
        decision = None
        installed = available[method.key]
        if method.key in screen_assays.FILES:
            names = screen_assays.FILES[method.key]
            supplied = any((directory / name).exists() or (directory / name).is_symlink() for name in names)
            if supplied or evidence.coverage(screen_inventory.BY_KEY[method.key])["status"] != "recorded":
                reader = method.key in screen_assays.READERS
                method = replace(
                    method,
                    inputs=names,
                    operation="native_result_reader" if reader else "compute",
                    workload=(
                        "Local result reader"
                        if reader
                        else "CPU model inference"
                        if method.key in screen_assays.REGRESSORS
                        else "CPU ligand geometry checks"
                    ),
                    purpose=screen_assays.PURPOSE[method.key],
                    default=method.key in {"posebusters", "openfe", "pmx", "toxicology_table"}
                    or method.key in screen_assays.EXACT_LOOKUPS,
                )
        if method.operation == "structure_measurement":
            missing = None
            if method.kinds and kind not in method.kinds:
                missing = "Requires a " + " or ".join(method.kinds) + " candidate."
            elif not has_structure:
                missing = "Requires a generated structure for this candidate."
            elif method.key in {"ligand_composition", "sequence_diversity", *screen_physical.POLYPEPTIDE_WARNINGS}:
                try:
                    decision = (
                        screen_criteria.ligand_component(structures, _READ_LIMIT) if method.key == "ligand_composition"
                        else screen_criteria.polypeptide_measurement(method.key, structures)
                    )
                    if not decision.applicable:
                        missing = decision.reason
                except (ValueError, OSError, RuntimeError):
                    missing = "The candidate structure records could not be inspected."
            elif method.key in screen_physical.COORDINATES:
                try:
                    decision = screen_criteria.coordinate_measurement(method.key, structures)
                    missing = None if decision.applicable else decision.reason
                except (ImportError, ValueError, OSError, RuntimeError):
                    missing = "The coordinates could not be assigned to the selected structures."
            elif method.key in screen_physical.SEQUENCE and not evidence.has_chain():
                missing = "Requires an identified designed chain in the metrics table."
        elif method.operation == "recorded_evidence":
            source = screen_inventory.BY_KEY[method.key]
            if method.kinds and kind not in method.kinds:
                missing = "Requires a " + " or ".join(method.kinds) + " candidate."
            elif (coverage := evidence.coverage(source))["status"] == "recorded":
                missing = None
                if coverage.get("unmatched_observations") or coverage.get("candidate_scope_unverified"):
                    decision = screen_criteria.Decision(
                        True, False, "Review the candidate attribution of the supplied observations."
                    )
                elif method.key == "posebusters":
                    decision = screen_criteria.recorded_pose_checks(evidence, {c.example_id for c in candidates})
                elif method.key == "stereochemistry":
                    decision = screen_criteria.recorded_stereochemistry(evidence)
            else:
                missing = "No attributable result from this producer is attached to the selected scope."
        else:
            try:
                missing = _input_reason(directory, method, kind, example)
                if missing is None and method.key == "qupkake":
                    decision = screen_criteria.qupkake_nominations(directory, example, _READ_LIMIT)
                elif missing is None and method.key == "sygma":
                    decision = screen_criteria.sygma_nominations(directory, example, _READ_LIMIT)
                elif missing is None and method.key == "posebusters":
                    decision = screen_criteria.posebusters_nominations(directory, example, _READ_LIMIT)
                elif missing is None and method.key in screen_assays.MOLECULAR_QUERIES:
                    decision = screen_criteria.reference_nominations(method.key, directory, example, _READ_LIMIT)
                elif missing is None and method.key in {"openfe", "pmx"}:
                    decision = screen_criteria.completed_result(method.key, evidence, _READ_LIMIT)
                elif missing is None and method.key == "toxicology_table":
                    decision = screen_criteria.supplied_panel(directory)
                elif missing is None and method.key in _PREPARED_STRUCTURES and has_structure:
                    decision = screen_criteria.prepared_structure(method.key, structures, _READ_LIMIT)
                elif missing is None and method.key == "solubility":
                    generated_ids = {c.example_id for c in candidates if c.example_id}
                    decision = screen_criteria.supplied_solubility(directory, example, generated_ids)
                elif missing is None and method.key == "protein_sol":
                    if example is not None and candidate is None:
                        missing = "No generated sample matches this candidate."
                    else:
                        decision = screen_criteria.protein_sol(directory, kind, structures, _READ_LIMIT)
                elif missing is None and method.key in _CRITERIA:
                    decision = _CRITERIA[method.key](directory, example)
                if decision is not None and not decision.applicable:
                    missing = decision.reason
                if (
                    missing is None
                    and method.key in _PREPARED_STRUCTURES
                    and not has_structure
                ):
                    missing = "Requires a generated structure for this candidate."
            except (ImportError, AttributeError, OSError, ValueError, csv.Error, RuntimeError):
                missing = "The candidate inputs could not be inspected."
        applicable = missing is None
        reason = missing or (decision.reason if decision is not None else method.purpose)
        if not installed:
            required = _REQUIREMENTS.get(method.key, "Requires the local provider and its configured assets.")
            reason = f"Unavailable in this runtime. {required} {reason}"
        default = method.default
        if method.key in _PREPARED_STRUCTURES:
            default = _regular(directory / method.inputs[0])
        if decision is not None:
            default = default and decision.recommended
        choices.append(Choice(method, installed, applicable, installed and applicable and default, reason))
    return tuple(choices)


def execution_plan(
    directory: Path,
    kind: str,
    mode: str,
    requested: Sequence[str],
    *,
    example: str | None = None,
    evidence: screen_inventory.Evidence | None = None,
) -> tuple[tuple[Choice, ...], tuple[str, ...]]:
    """Check only providers whose installation can change this execution decision."""
    if mode == "explicit":
        select((), mode, requested)
        if not requested:
            return (), ()
        keys = set(requested)
        found = installations(keys)
        choices = plan(directory, kind, example=example, available={k: found[k] for k in keys}, evidence=evidence)
        return choices, select(choices, mode, requested)
    draft = plan(directory, kind, example=example, available=dict.fromkeys(METHOD_IDS, True), evidence=evidence)
    keys = set(select(draft, mode, requested))
    available = installations(keys)
    choices = tuple(
        replace(c, installed=available[c.method.key], recommended=c.recommended and available[c.method.key])
        for c in draft
        if c.method.key in keys
    )
    return choices, select(choices, mode, requested)


def select(choices: Sequence[Choice], mode: str, explicit: Sequence[str] = ()) -> tuple[str, ...]:
    """Validate the whole request before invoking any of its providers."""
    if mode not in ("recommended", "all_installed", "explicit"):
        raise ValueError("Choose recommended, all_installed or explicit screening.")
    if mode != "explicit" and explicit:
        raise ValueError("Choose one screening mode or individual methods.")
    if isinstance(explicit, str | bytes) or any(not isinstance(key, str) for key in explicit):
        raise ValueError("Screen methods must be a list of method IDs.")
    if len(set(explicit)) != len(explicit) or set(explicit).difference(METHOD_IDS):
        raise ValueError("Choose distinct screen IDs from the current plan.")
    unavailable = [c.method.label for c in choices if c.method.key in explicit and not c.installed]
    if unavailable:
        raise ValueError("Unavailable screening methods: " + ", ".join(unavailable))
    return tuple(
        c.method.key
        for c in choices
        if (
            (mode == "recommended" and c.recommended)
            or (mode == "all_installed" and c.installed)
            or (mode == "explicit" and c.method.key in explicit)
        )
    )
