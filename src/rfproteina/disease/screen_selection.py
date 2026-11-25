"""Plan optional post-generation measurements from local installations and run inputs."""

from __future__ import annotations

import csv
import os
import stat
import subprocess
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from importlib import util
from pathlib import Path

from rfproteina.disease import (
    badapple2,
    cellprofiler_runtime,
    cellprofiler_stain,
    crest,
    cypstrate,
    cypstrate_runtime,
    dimorphite,
    delphi_inputs,
    delphi_psr,
    espsim,
    fastsolv,
    httk,
    jazzy,
    ligand_alerts,
    ligand_ames,
    ligand_caco2,
    ligand_dili,
    ligand_microsomal_clearance,
    ligand_solubility,
    metfrag,
    molprobity_geometry,
    openantigens,
    p2rank,
    propka,
    peptide_rbc,
    pksim,
    pksim_runtime,
    qupkake,
    screen_assays,
    screen_criteria,
    screen_inventory,
    screen_performance,
    screen_physical,
    scscore,
    scscore_model,
    spyrmsd_pose,
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
        badapple2.METHOD_ID,
        "Badapple 2 historical scaffold assay evidence",
        LIGAND,
        badapple2.SOURCES,
        "Local CPU scaffold lookup",
        "Review PubChem assay history when choosing interference counterscreens for cyclic small-molecule leads.",
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
        ligand_ames.METHOD_ID,
        "ADMET-AI Ames mutagenicity prediction",
        LIGAND,
        SMILES,
        "CPU classification ensemble",
        "Prioritize bacterial reverse-mutation testing for nominated small-molecule leads.",
    ),
    Method(
        ligand_microsomal_clearance.METHOD_ID,
        "ADMET-AI human liver microsomal intrinsic clearance",
        LIGAND,
        SMILES,
        "CPU regression ensemble",
        "Prioritize a human liver microsomal stability assay for nominated systemic small-molecule leads.",
    ),
    Method(
        ligand_caco2.METHOD_ID,
        "AstraZeneca GNN-MTL Caco-2 permeability and efflux",
        LIGAND,
        SMILES,
        "CPU graph model",
        "Review model scores to plan a Caco-2 permeability and efflux assay for oral small-molecule leads.",
    ),
    Method(
        ligand_dili.METHOD_ID,
        "ADMET-AI drug-induced liver injury prediction",
        LIGAND,
        SMILES,
        "CPU classification ensemble",
        "Prioritize liver-safety follow-up for systemically exposed small-molecule leads.",
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
        dimorphite.METHOD_ID,
        "Dimorphite-DL ligand protonation states",
        LIGAND,
        (dimorphite.MANIFEST,),
        "Local CPU substructure rules",
        "Enumerate protonation states for nominated ligands at an explicitly declared assay pH.",
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
        p2rank.METHOD_ID,
        "P2Rank receptor binding pockets",
        LIGAND,
        (p2rank.MANIFEST,),
        "Local CPU pocket prediction",
        "Rank candidate pockets for the explicitly nominated receptor when choosing a binding site.",
    ),
    Method(
        propka.METHOD_ID,
        "PROPKA structure-conditioned pKa",
        ("protein", "peptide", "ligand"),
        (propka.MANIFEST,),
        "Local CPU pKa calculation",
        "Estimate group pKa on the nominated structure for pH-dependent assay planning.",
    ),
    Method(
        crest.METHOD_ID,
        "CREST conformer sampling",
        LIGAND,
        (crest.MANIFEST,),
        "Local CPU conformer search",
        "Sample alternative conformations of one nominated ligand for later modeling.",
    ),
    Method(
        openantigens.METHOD_ID,
        "OpenAntigens human target evidence",
        PROTEIN,
        (openantigens.MANIFEST,),
        "Local reference lookup",
        "Review human target topology and public structure or assembly evidence before experimental validation.",
    ),
    Method(
        metfrag.METHOD_ID,
        "MetFrag MS/MS fragment evidence",
        LIGAND,
        (metfrag.MANIFEST,),
        "Local CPU fragment scoring",
        "Compare nominated structures against one measured tandem mass spectrum.",
    ),
    Method(
        fastsolv.METHOD_ID,
        "fastsolv organic-solvent solubility",
        LIGAND,
        (fastsolv.MANIFEST,),
        "Local CPU solubility predictions",
        "Compare solubility in nominated pure organic solvents at defined temperatures.",
    ),
    Method(
        jazzy.METHOD_ID,
        "Jazzy hydration energetics",
        LIGAND,
        jazzy.SOURCES,
        "Local CPU hydration and hydrogen-bond estimates",
        "Estimate water hydration free energy and donor and acceptor strengths for the nominated molecular form.",
    ),
    Method(
        espsim.METHOD_ID,
        "ESP-Sim aligned shape and electrostatics",
        LIGAND,
        (espsim.MANIFEST,),
        "Local CPU comparison of one aligned pair",
        "Compare a nominated ligand pose and its prealigned reference on two separate similarity scales.",
    ),
    Method(
        cellprofiler_stain.METHOD_ID,
        "CellProfiler regional DAB staining",
        (),
        (cellprofiler_stain.MANIFEST,),
        "CPU brightfield image analysis",
        "Compare staining in annotated regions of candidate and control images from human tissue or chip experiments.",
    ),
    Method(
        pksim.METHOD_ID,
        "PK-Sim whole-body exposure",
        LIGAND,
        (pksim.MANIFEST,),
        "Local CPU pharmacokinetic simulation",
        "Simulate concentration over time for a small molecule with a prepared human PBPK model.",
    ),
    Method(
        httk.METHOD_ID,
        "EPA httk population steady-state exposure",
        LIGAND,
        (httk.MANIFEST,),
        "Local CPU population toxicokinetics",
        "Estimate human oral steady-state plasma concentration for ligands matched to EPA reference structures.",
        default=False,
    ),
    Method(
        cypstrate.METHOD_ID,
        "CYPstrate substrate profiling",
        LIGAND,
        (cypstrate.MANIFEST,),
        "Local CPU substrate classifiers",
        "Classify a systemic small molecule against its nominated human CYP isoforms.",
    ),
    Method(
        scscore.METHOD_ID,
        "SCScore synthetic complexity",
        LIGAND,
        (scscore.MANIFEST,),
        "CPU molecular fingerprint and NumPy model",
        "Compare the synthetic complexity of otherwise comparable small molecules before chemistry review.",
    ),
    Method(
        spyrmsd_pose.METHOD,
        "sPyRMSD redocking pose recovery",
        LIGAND,
        (spyrmsd_pose.MANIFEST,),
        "CPU symmetry-corrected RMSD",
        "Measure heavy-atom displacement from a matched experimental ligand pose in its receptor frame.",
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
        delphi_psr.METHOD_ID,
        "DELPHI antibody polyreactivity (PSR)",
        ("protein",),
        (delphi_inputs.MANIFEST,),
        "CPU sequence classifier",
        "Prioritize experimental polyreactivity testing for explicitly paired antibody variable domains.",
    ),
    Method(
        peptide_rbc.METHOD_ID,
        "Supplied peptide red-cell compatibility evidence",
        ("peptide",),
        (peptide_rbc.MANIFEST,),
        "Local result reader",
        "Review supplied red-cell measurements for nominated therapeutic peptide candidates.",
        operation="native_result_reader",
    ),
    Method(
        molprobity_geometry.METHOD_ID,
        "MolProbity protein geometry",
        PROTEIN,
        (),
        "CPU all-atom and torsion validation",
        "Review Ramachandran angles, side-chain rotamers and all-atom clashes in the whole coordinate model.",
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
    cellprofiler_stain.METHOD_ID: "Requires Pillow, NumPy and the pinned CellProfiler image on a local Docker engine.",
    "ligand_alerts": "Requires RDKit's molecular descriptors and FilterCatalog support.",
    "aggrescan3d": "Requires the Aggrescan3D executable on the service path.",
    "bayesherg": "Requires release weights under BAYESHERG_ROOT and the BAYESHERG_PYTHON interpreter.",
    "bindingdb_chemprop": "Requires Chemprop and the trained graph-model files in the configured model directory.",
    "alchemlyb": "Requires an importable alchemlyb installation in this runtime.",
    "pmx": "Requires PyMBAR, NumPy and SciPy in the result reader's runtime.",
    "cath_s95": "Requires Foldseek, the CATH S95 database and its domain list.",
    "gearnet": "Requires PyTorch, the GearNet classifier checkpoint and its label mapping.",
    "qupkake": "Requires the configured QupKake runtime, microscopic pKa checkpoints and xTB.",
    dimorphite.METHOD_ID: "Requires Dimorphite-DL 2.1.0 with its verified bundled rules and RDKit in this runtime.",
    badapple2.METHOD_ID: "Requires the verified Badapple 2 snapshot, its local HierS runtime, and RDKit.",
    p2rank.METHOD_ID: "Requires the verified P2Rank 2.5.1 release, compatible Java and Gemmi in this runtime.",
    propka.METHOD_ID: "Requires PROPKA 3.5.1 with the verified parameter files and Gemmi in this runtime.",
    crest.METHOD_ID: "Requires the verified CREST 3.0.2 Linux release and RDKit in this runtime.",
    openantigens.METHOD_ID: "Requires the verified local OpenAntigens human reference snapshot.",
    metfrag.METHOD_ID: "Requires the verified MetFrag 2.6.11 release, Java 21, and a Linux offline runtime.",
    fastsolv.METHOD_ID: "Requires four verified fastsolv checkpoints and the local CPU descriptor runtime.",
    jazzy.METHOD_ID: "Requires the isolated Jazzy 0.1.4 CPU runtime with its pinned RDKit and Kallisto versions.",
    espsim.METHOD_ID: "Requires the verified ESP-Sim 0.0.1 scoring sources and the local scientific Python runtime.",
    "tnp_profile": "Requires the TNP runtime, NanoBodyBuilder2 checkpoints and DSSP.",
    delphi_psr.METHOD_ID: "Requires the verified DELPHI PSR checkpoint and local CPU antibody runtime.",
    molprobity_geometry.METHOD_ID: "Requires the local MolProbity CCTBX runtime and its geometry reference data.",
    "ligand_solubility": "Requires the ADMET-AI regression ensemble and its endpoint definitions.",
    ligand_ames.METHOD_ID: "Requires the verified local ADMET-AI classification ensemble and AMES definition.",
    ligand_caco2.METHOD_ID: "Requires the verified local AstraZeneca checkpoint and a compatible CPU runtime.",
    ligand_dili.METHOD_ID: "Requires the verified local ADMET-AI classification ensemble and DILI definition.",
    ligand_microsomal_clearance.METHOD_ID: (
        "Requires the verified local ADMET-AI regression ensemble and human microsomal clearance definition."
    ),
    "sygma": "Requires RDKit and the released SyGMa rule tables.",
    spyrmsd_pose.METHOD: "Requires the spyrmsd extra with RDKit and a working graph backend.",
    scscore.METHOD_ID: "Requires RDKit, NumPy and the verified SCScore 1024-bit Boolean checkpoint.",
    pksim.METHOD_ID: "Requires RDKit and the supported local PK-Sim simulation runtime.",
    httk.METHOD_ID: "Requires RDKit, a compatible local EPA httk R runtime and its verified reference data.",
    cypstrate.METHOD_ID: "Requires RDKit and the pinned CYPstrate v0.0.25 image on a local Docker engine.",
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
    resolved: bool = False
    producer_installed: bool | None = None

    @property
    def performance(self) -> dict[str, str]:
        return screen_performance.for_method(self.method.key, self.method.operation, installed=self.installed)

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
            "performance": self.performance,
            "operation": self.method.operation,
            **({"producer_installed": self.producer_installed} if self.producer_installed is not None else {}),
        }


def _with_installation(choice: Choice, available: bool) -> Choice:
    """A stored measurement can be read locally after its producer was removed."""
    recorded = choice.method.operation == "recorded_evidence"
    usable = available or (recorded and choice.applicable)
    reason = choice.reason
    if not available and usable:
        reason = f"{reason} The local reader can open these saved results. The producing software is unavailable."
    elif not available and not choice.resolved:
        required = _REQUIREMENTS.get(choice.method.key, "Requires the local provider and its configured assets.")
        reason = f"Unavailable in this runtime. {required} {reason}"
    return replace(
        choice, installed=usable, recommended=usable and choice.recommended,
        producer_installed=available if recorded else None, reason=reason,
    )


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

    def admet(verify: Callable[[], object]) -> bool:
        if not _modules("rdkit", "chemprop", "admet_ai", "torch"):
            return False
        verify()
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
        cellprofiler_stain.METHOD_ID: lambda: _modules("PIL.Image", "numpy") and cellprofiler_runtime.installed(),
        pksim.METHOD_ID: lambda: _modules("rdkit") and pksim_runtime.installed(),
        httk.METHOD_ID: httk.asset_available,
        delphi_psr.METHOD_ID: delphi_psr.asset_available,
        molprobity_geometry.METHOD_ID: molprobity_geometry.asset_available,
        peptide_rbc.METHOD_ID: lambda: True,
        "ligand_alerts": rdkit_alerts,
        "ligand_solubility": lambda: admet(ligand_solubility.verified_assets),
        ligand_ames.METHOD_ID: lambda: admet(ligand_ames.verified_assets),
        ligand_caco2.METHOD_ID: ligand_caco2.asset_available,
        ligand_dili.METHOD_ID: lambda: admet(ligand_dili.verified_assets),
        ligand_microsomal_clearance.METHOD_ID: lambda: admet(ligand_microsomal_clearance.verified_assets),
        "sygma": sygma,
        "qupkake": lambda: _modules("rdkit") and _worker_ready(qupkake._runtime(), "qupkake"),
        dimorphite.METHOD_ID: dimorphite.installed,
        badapple2.METHOD_ID: badapple2.installed,
        p2rank.METHOD_ID: p2rank.installed,
        propka.METHOD_ID: propka.installed,
        crest.METHOD_ID: crest.installed,
        openantigens.METHOD_ID: openantigens.installed,
        metfrag.METHOD_ID: metfrag.installed,
        fastsolv.METHOD_ID: fastsolv.installed,
        jazzy.METHOD_ID: jazzy.installed,
        espsim.METHOD_ID: espsim.installed,
        "vina_pose": lambda: _modules("gemmi") and bool(vina_pose.installed_executable()),
        spyrmsd_pose.METHOD: spyrmsd_pose.installed,
        scscore.METHOD_ID: scscore_model.installed,
        cypstrate.METHOD_ID: cypstrate_runtime.installed,
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
_RUN_LEVEL = {"toxicology_table", openantigens.METHOD_ID}
_PREPARED_STRUCTURES = {"tnp_profile", "vina_pose", "prodigy_affinity"}
_CRITERIA = {
    cellprofiler_stain.METHOD_ID: screen_criteria.cellprofiler_staining,
    pksim.METHOD_ID: screen_criteria.pksim_exposure,
    httk.METHOD_ID: screen_criteria.httk_exposure,
    cypstrate.METHOD_ID: screen_criteria.cyp_substrates,
    delphi_psr.METHOD_ID: screen_criteria.delphi_antibody_pairs,
    scscore.METHOD_ID: screen_criteria.scscore_comparison,
    dimorphite.METHOD_ID: screen_criteria.dimorphite_nominations,
    badapple2.METHOD_ID: screen_criteria.badapple2_nominations,
    p2rank.METHOD_ID: screen_criteria.p2rank_nomination,
    propka.METHOD_ID: screen_criteria.propka_nomination,
    crest.METHOD_ID: screen_criteria.crest_nomination,
    openantigens.METHOD_ID: screen_criteria.openantigens_target,
    metfrag.METHOD_ID: screen_criteria.metfrag_nominations,
    fastsolv.METHOD_ID: screen_criteria.fastsolv_conditions,
    jazzy.METHOD_ID: screen_criteria.jazzy_nominations,
    espsim.METHOD_ID: screen_criteria.espsim_pair,
    spyrmsd_pose.METHOD: screen_criteria.redocking_controls,
    "ligand_alerts": screen_criteria.rdkit_alerts,
    "bayesherg": screen_criteria.bayesherg_nominations,
    "ligand_solubility": screen_criteria.admet_solubility,
    ligand_ames.METHOD_ID: screen_criteria.admet_ames,
    ligand_caco2.METHOD_ID: screen_criteria.caco2_permeability,
    ligand_dili.METHOD_ID: screen_criteria.admet_dili,
    ligand_microsomal_clearance.METHOD_ID: screen_criteria.admet_microsomal_clearance,
}


def _input_requirement(directory: Path, method: Method, kind: str, example: str | None) -> str | None:
    if method.kinds and kind not in method.kinds:
        return "Requires a " + " or ".join(method.kinds) + " candidate."
    if example is not None and (not example or method.key in _RUN_LEVEL):
        return "The measurement requires an identifiable candidate or a run-level selection."
    inputs = method.inputs
    if method.key == openantigens.METHOD_ID:
        return None  # The archived run or job can identify the target without a separate nomination.
    if method.key in {delphi_psr.METHOD_ID, molprobity_geometry.METHOD_ID, peptide_rbc.METHOD_ID}:
        return None  # These readers check their authoritative input and its own bounds.
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
    evidence = evidence if evidence is not None else screen_inventory.Evidence(directory, example, kind=kind)
    from rfproteina.disease.validation_checks import _candidates, _inventory

    needs_structure = screen_physical.METHODS | _PREPARED_STRUCTURES | {"solubility", "protein_sol", "posebusters"}
    if kind in PROTEIN:
        needs_structure = needs_structure | {molprobity_geometry.METHOD_ID}
    if kind == "peptide" and _regular(directory / peptide_rbc.MANIFEST):
        needs_structure = needs_structure | {peptide_rbc.METHOD_ID}
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
        if method.operation == "recorded_evidence" and method.key in screen_assays.FILES:
            names = screen_assays.FILES[method.key]
            supplied = any((directory / name).exists() or (directory / name).is_symlink() for name in names)
            use_fresh = supplied and installed
            if not use_fresh:
                coverage = evidence.coverage(screen_inventory.BY_KEY[method.key])
                use_fresh = coverage["status"] != "recorded" or (
                    supplied and bool(coverage.get("candidate_scope_unverified"))
                )
            if use_fresh:
                reader = method.key in screen_assays.READERS
                method = replace(
                    method,
                    inputs=names,
                    operation="native_result_reader" if reader else "compute",
                    workload=(
                        "Local result reader"
                        if reader
                        else "CPU model inference"
                        if method.key in screen_assays.REGRESSORS or method.key == "bayesherg"
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
                    total = sum(coverage[k] for k in ("sample_observations", "metric_cells", "attached_observations"))
                    noun = "observation" if total == 1 else "observations"
                    decision = screen_criteria.Decision(
                        True, method.default, f"{total} saved {noun} in this selection. {method.purpose}"
                    )
            else:
                missing = "No attributable result from this producer is attached to the selected scope."
        else:
            try:
                missing = _input_requirement(directory, method, kind, example)
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
                elif missing is None and method.key == peptide_rbc.METHOD_ID:
                    ids = screen_criteria.peptide_rbc_candidate_ids(candidates)
                    decision = screen_criteria.peptide_rbc_observations(directory, example, ids)
                elif missing is None and method.key == molprobity_geometry.METHOD_ID:
                    decision = screen_criteria.molprobity_geometry_inputs(directory, example, candidates)
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
        resolved = decision is not None and decision.resolved
        default = method.default
        if method.key in _PREPARED_STRUCTURES:
            default = _regular(directory / method.inputs[0])
        if decision is not None:
            default = default and decision.recommended
        choice = Choice(method, installed, applicable, applicable and default, reason, resolved)
        choices.append(_with_installation(choice, installed))
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
    if mode in {"explicit", "all_installed"}:
        select((), mode, requested)
        if mode == "explicit" and not requested:
            return (), ()
        keys = set(requested if mode == "explicit" else METHOD_IDS)
        found = installations(keys)
        choices = plan(directory, kind, example=example, available={k: found[k] for k in keys}, evidence=evidence)
        return choices, select(choices, mode, requested)
    from rfproteina.disease.screen import _canonical_kind

    kind = _canonical_kind(kind)
    evidence = evidence if evidence is not None else screen_inventory.Evidence(directory, example, kind=kind)
    draft = plan(directory, kind, example=example, available=dict.fromkeys(METHOD_IDS, True), evidence=evidence)
    keys = set(select(draft, mode, requested))
    # Some methods can read saved results when their producing tool is absent. In particular,
    # well-attributed saved pose checks can be recommended even if fresh inputs fail the pose criteria.
    saved_keys = {
        c.method.key for c in draft
        if c.method.operation != "recorded_evidence" and c.method.key in screen_assays.FILES
        and (source := screen_inventory.BY_KEY.get(c.method.key)) is not None
        and (c.method.key in keys or source.priority) and (not c.method.kinds or kind in c.method.kinds)
        and evidence.coverage(source)["status"] == "recorded"
    }
    saved = {}
    if saved_keys:
        offline = plan(directory, kind, example=example, available=dict.fromkeys(saved_keys, False), evidence=evidence)
        saved = {c.method.key: c for c in offline}
    keys.update(key for key, c in saved.items() if c.recommended)
    available = installations(keys)
    choices = tuple(
        saved[c.method.key] if c.method.key in saved and not available[c.method.key]
        else _with_installation(c, available[c.method.key])
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
    unavailable = [c.method.label for c in choices if c.method.key in explicit and not c.installed and not c.resolved]
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
