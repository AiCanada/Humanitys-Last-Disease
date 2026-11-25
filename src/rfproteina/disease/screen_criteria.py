"""Decide whether a nominated molecule is ready for a particular post-generation screen."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

from rfproteina.disease import (
    badapple2,
    cellprofiler_stain,
    crest,
    cypstrate,
    delphi_inputs,
    dimorphite,
    espsim,
    fastsolv,
    httk,
    jazzy,
    ligand_alerts,
    ligand_ames,
    ligand_caco2,
    ligand_dili,
    ligand_inputs,
    ligand_metabolites,
    ligand_microsomal_clearance,
    ligand_solubility,
    metfrag,
    openantigens,
    p2rank,
    peptide_rbc,
    pksim,
    propka,
    qupkake,
    screen_inventory,
    screen_physical,
    screening_context,
    scscore,
    spyrmsd_pose,
)

if TYPE_CHECKING:
    from rfproteina.disease.validation_checks import ValidationCandidate


@dataclass(frozen=True)
class Decision:
    applicable: bool
    recommended: bool
    reason: str
    resolved: bool = False


def _real_example_file(path: Path) -> bool:
    try:
        return path.resolve(strict=True) == path and path.is_file()
    except (OSError, RuntimeError):
        return False


def peptide_rbc_candidate_ids(candidates: Iterable[ValidationCandidate]) -> frozenset[str]:
    """An absent, aliased, or colliding example cannot identify experimental evidence."""
    from rfproteina.disease.screen import _sample_id

    identities = []
    for candidate in candidates:
        if candidate.files and _real_example_file(path := screen_physical._file(candidate)):
            identities.append((candidate.example_id, _sample_id(path)))
    id_counts = Counter(name for name, _ in identities)
    short_counts = Counter(short for _, short in identities)
    # The run's root example can still inherit a collision-prone bare name
    # from the structure inventory. Do not bind a lab result to that name
    # when it could refer to another actual example with the same short ID.
    return frozenset(
        name for name, short in identities
        if name and id_counts[name] == 1 and (name != short or short_counts[short] == 1)
    )


def peptide_rbc_known_candidates(directory: Path) -> frozenset[str]:
    """Resolve real generated examples without reading sequences or coordinates."""
    from rfproteina.disease.validation_checks import _candidates, _inventory

    return peptide_rbc_candidate_ids(_candidates(_inventory(directory)))


def peptide_rbc_observations(directory: Path, example: str | None, known: frozenset[str]) -> Decision:
    """Recommend review of complete human observations, not a safety decision."""
    rows, blocker = peptide_rbc.read_observations(directory, candidate=example, known_candidates=known)
    requirement = f"Requires attributable experimental peptide observations in {peptide_rbc.MANIFEST}."
    if blocker is not None:
        return Decision(False, False, f"{requirement} The evidence is unavailable ({blocker}).")
    reported = sum(row.get("status") == "reported" for row in rows)
    if not reported:
        attributable = any(row.get("candidate_id") in known for row in rows)
        if attributable:
            return Decision(True, False, "No usable observation is available. Select explicitly to review blockers.")
        return Decision(False, False, f"{requirement} No observation matches this selection.")
    human = sum(row.get("status") == "reported" and row.get("species") == "human" for row in rows)
    if human == len(rows):
        return Decision(True, True, "Review the supplied, QC-accepted human red-cell measurements separately.")
    return Decision(
        True, False,
        f"{reported} of {len(rows)} observations are attributable and QC-accepted, {human} human. Select explicitly.",
    )


def molprobity_geometry_inputs(
    directory: Path, example: str | None, candidates: Iterable[ValidationCandidate],
) -> Decision:
    """Recommend one coordinate model from metadata, without parsing or running CCTBX."""
    from rfproteina.disease import molprobity_inputs

    models, blocker = molprobity_inputs.resolve_inputs(directory, example, candidates=candidates)
    if blocker is not None:
        return Decision(False, False, f"Requires an eligible finished protein coordinate model ({blocker}).")
    ready = sum(model.blocker is None for model in models)
    if not ready:
        reason = next((model.blocker for model in models if model.blocker), None)
        suffix = f" ({reason})" if reason else ""
        return Decision(False, False, f"No eligible finished protein coordinate model{suffix}.")
    if ready == len(models) == 1:
        return Decision(True, True, "Review geometry of the whole nominated model, including any partner chains.")
    return Decision(
        True, False,
        f"{ready} of {len(models)} models are ready. Select explicitly to review each whole model separately.",
    )


def delphi_antibody_pairs(directory: Path, example: str | None) -> Decision:
    """Check declared VH/VL nominations cheaply. Inference verifies domain identities."""
    rows, blocker = delphi_inputs.read_pairs(directory, candidate=example)
    requirement = f"Requires named VH/VL pairs in {delphi_inputs.MANIFEST}. VHH-only inputs are not supported."
    if blocker:
        return Decision(False, False, f"{requirement} The nomination is unusable ({blocker}).")
    eligible = sum(row.blocker is None for row in rows)
    if not eligible:
        return Decision(False, False, f"No usable nomination matches this selection. {requirement}")
    purpose = "Prioritize experimental PSR testing for explicitly nominated paired antibody variable domains."
    if eligible == len(rows) == 1:
        return Decision(True, True, purpose)
    reason = f"{eligible} of {len(rows)} VH/VL nominations are eligible for domain verification. Select explicitly."
    return Decision(True, False, f"{purpose} {reason}")


def _smiles_decision(
    directory: Path, example: str | None, classify: Callable[[str], Decision], purpose: str
) -> Decision:
    scope, records, problem = ligand_inputs.read_smiles_candidates(directory)
    if problem:
        return Decision(False, False, f"The molecular nomination could not be read ({problem}).")
    if example is not None and scope != "candidate_table":
        return Decision(False, False, "A run-level molecule has no individual candidate attribution.")
    return _decide_smiles_rows(list(ligand_inputs.nominated_rows(records or (), example)), classify, purpose)


def _decide_smiles_rows(
    rows: Sequence[Sequence[str]], classify: Callable[[str], Decision], purpose: str, *,
    validate_id: Callable[[str], bool] = bool,
) -> Decision:
    if not rows:
        return Decision(False, False, "No molecular nomination matches this selection.")
    counts = Counter(row[0] for row in rows if row)
    seen: dict[str, Decision] = {}
    review: set[str] = set()
    eligible = 0
    for row in rows:
        if len(row) != 2 or not validate_id(row[0]) or counts[row[0]] != 1:
            review.add("Requires a unique candidate ID and exactly one SMILES field.")
            continue
        smiles = row[1]
        if smiles not in seen:
            seen[smiles] = classify(smiles)
        result = seen[smiles]
        eligible += result.applicable
        if not result.recommended:
            review.add(result.reason)
    if not review:
        return Decision(True, True, purpose)
    reasons = " ".join(sorted(review))
    if len(rows) == 1:
        return Decision(bool(eligible), False, reasons)
    return Decision(bool(eligible), False, f"{eligible} of {len(rows)} nominations can be evaluated. {reasons}")


def rdkit_alerts(directory: Path, example: str | None) -> Decision:
    from rdkit import Chem, rdBase

    def classify(smiles: str) -> Decision:
        try:
            # RDKit otherwise echoes rejected inputs to the service log on every picker refresh.
            with rdBase.BlockLogs():
                mol = ligand_alerts.parse_molecule(smiles, nominated=True)
        except ligand_alerts.LigandUnreadableError:
            return Decision(False, False, "Requires a complete molecular graph in a valid SMILES field.")
        if len(Chem.GetMolFrags(mol)) != 1:
            return Decision(True, False, "Confirm the intended salt or mixture before screening this structure.")
        return Decision(True, True, "")

    decision = _smiles_decision(
        directory, example, classify, "The nominated molecule is ready for an assay-interference and descriptor check."
    )
    if decision.applicable and (missing := ligand_alerts.missing_catalogues()):
        reason = "Review partial RDKit catalogue coverage. Missing " + ", ".join(missing) + "."
        return Decision(True, False, reason if decision.recommended else f"{decision.reason} {reason}")
    return decision


def bayesherg_nominations(directory: Path, example: str | None) -> Decision:
    from rfproteina.disease import screen_assays

    plan, molecules = screen_assays.bayesherg_nominations(directory, example)
    if not molecules:
        return Decision(False, False, f"No supported cardiac safety nomination ({plan['blocker']}).")
    eligible = sum("compound_id" in row for row in plan["inputs"])
    return Decision(
        True, False, f"{eligible} of {plan['attempted']} nominations are eligible. "
        f"The CPU model uses 30 dropout samples per distinct molecule ({len(molecules)} total). "
        "Select it when a hERG patch-clamp assay is under consideration.",
    )


def admet_solubility(directory: Path, example: str | None) -> Decision:
    from rdkit import Chem, rdBase

    compounds = set()

    def classify(smiles: str) -> Decision:
        try:
            with rdBase.BlockLogs():
                mol, identity = ligand_solubility.molecular_identity(smiles, Chem)
        except ligand_solubility.SolubilityError:
            return Decision(False, False, "Requires an explicit molecular graph accepted by the solubility model.")
        compounds.add(identity)
        if len(Chem.GetMolFrags(mol)) > 1:
            return Decision(True, False, "Confirm the salt or mixture and the form used in the solubility assay.")
        if Chem.GetFormalCharge(mol):
            return Decision(True, False, "Confirm the charged form and the pH planned for the solubility assay.")
        if not any(atom.GetAtomicNum() == 6 for atom in mol.GetAtoms()):
            return Decision(True, False, "Review use of the drug solubility model for this carbon-free structure.")
        return Decision(True, True, "")

    decision = _smiles_decision(
        directory,
        example,
        classify,
        "Estimate aqueous logS for the nominated compound to plan a solubility measurement.",
    )
    if len(compounds) > ligand_solubility.BATCH_SIZE:
        reason = f"{len(compounds)} distinct molecules require multiple CPU batches. Select this ensemble explicitly."
        return Decision(True, False, reason if decision.recommended else f"{decision.reason} {reason}")
    return decision


def admet_ames(directory: Path, example: str | None) -> Decision:
    return _bounded_admet_decision(
        directory, example, ligand_ames, ligand_ames.AmesError,
        label="Ames", purpose="Prioritize bacterial reverse-mutation testing for the nominated compound.",
    )


def admet_microsomal_clearance(directory: Path, example: str | None) -> Decision:
    return _bounded_admet_decision(
        directory, example, ligand_microsomal_clearance, ligand_microsomal_clearance.ClearanceError,
        label="human microsomal clearance",
        purpose="Estimate human microsomal intrinsic clearance to prioritize a metabolic stability assay.",
    )


def caco2_permeability(directory: Path, example: str | None) -> Decision:
    purpose = "Select explicitly to plan Caco-2 measurements for an oral small-molecule lead."
    return _bounded_admet_decision(
        directory, example, ligand_caco2, ligand_caco2.CacoError,
        label="Caco-2", purpose=purpose,
    )


def admet_dili(directory: Path, example: str | None) -> Decision:
    # Molecular identity alone does not establish the intended systemic exposure.
    purpose = "Select explicitly to prioritize liver-safety follow-up for a systemically exposed small-molecule lead."
    return _bounded_admet_decision(
        directory, example, ligand_dili, ligand_dili.DiliError,
        label="DILI", purpose=purpose,
    )


def _bounded_admet_decision(
    directory: Path, example: str | None, provider: ModuleType, parse_error: type[ValueError], *,
    label: str, purpose: str,
) -> Decision:
    """Review bounded nominations without loading checkpoints or making predictions."""
    from rdkit import Chem, rdBase

    _, rows, blocker = provider.screen_inputs(directory, example)
    if blocker:
        return Decision(False, False, f"The {label} nominations could not be evaluated ({blocker}).")
    compounds: set[str] = set()

    def classify(smiles: str) -> Decision:
        try:
            with rdBase.BlockLogs():
                mol, identity = provider.molecular_identity(smiles, Chem)
        except parse_error:
            return Decision(False, False, "Requires a supported single connected organic molecular graph.")
        compounds.add(identity)
        if any(atom.GetFormalCharge() for atom in mol.GetAtoms()):
            return Decision(True, False, f"Confirm the nominated charged form before selecting {label} prediction.")
        return Decision(True, True, "")

    decision = _decide_smiles_rows(rows, classify, purpose, validate_id=provider.valid_candidate_id)
    if len(compounds) > provider.BATCH_SIZE:
        reason = f"{len(compounds)} distinct molecules require multiple CPU batches. Select this ensemble explicitly."
        return Decision(True, False, reason if decision.recommended else f"{decision.reason} {reason}")
    if decision.recommended and provider in (ligand_caco2, ligand_dili, ligand_microsomal_clearance):
        facts, issue = screening_context.for_candidates(directory, (row[0] for row in rows))
        if issue:
            return Decision(True, False, issue)
        if any(item.systemic is False for item in facts):
            return Decision(True, False, f"Non-systemic exposure is declared. Review {label} manually.")
        if provider is ligand_caco2:
            if all(item.route == "oral" for item in facts):
                return Decision(True, True, "Oral administration is declared throughout this selection. "
                                "Prioritize a Caco-2 permeability and efflux assay.")
            return Decision(True, False, purpose)
        if provider is ligand_dili:
            if all(item.systemic is True for item in facts):
                return Decision(True, True, "Systemic exposure is declared throughout this selection. "
                                "Prioritize liver-safety follow-up for the nominated molecules.")
            return Decision(True, False, purpose)
    return decision


def reference_nominations(key: str, directory: Path, example: str | None, budget: int) -> Decision:
    from rdkit import Chem, rdBase

    from rfproteina.disease import ligand_regressor, ligand_safety, octant, screen_assays

    compounds = {}

    def classify(smiles: str) -> Decision:
        try:
            with rdBase.BlockLogs():
                mol = ligand_alerts.parse_molecule(smiles, nominated=True)
        except ligand_alerts.LigandUnreadableError:
            return Decision(False, False, "Requires a complete nominated molecular graph.")
        compounds[Chem.MolToSmiles(mol)] = smiles
        if len(Chem.GetMolFrags(mol)) != 1:
            return Decision(True, False, "Confirm the form of the salt or mixture before interpreting the lookup.")
        return Decision(True, True, "")

    decision = _smiles_decision(directory, example, classify, screen_assays.PURPOSE[key])
    if not decision.applicable:
        return decision
    if key not in screen_assays.EXACT_LOOKUPS:
        reason = screen_assays.PURPOSE[key]
        if not decision.recommended:
            reason += " " + decision.reason
        return Decision(True, False, reason)
    if len(compounds) != 1:
        return Decision(True, False, "Each distinct compound requires a separate reference lookup.")
    try:
        subset = octant.installed_subset() if key == "octant" else ligand_regressor.default_subset()
        if subset is None or subset.stat().st_size > budget:
            return Decision(True, False, "Select to verify the exact match in the configured reference table.")
        result = screen_assays._lookup(key, next(iter(compounds.values())))
    except (OSError, ligand_regressor.RegressorCorpusError, ligand_safety.SafetyBackendUnavailableError,
            ligand_safety.LigandUnreadableError):
        return Decision(False, False, "The configured exact-measurement reference could not verify this compound.")
    if result.get("blocker"):
        return Decision(False, False, "The installed reference contains no exact measurement for the nominated form.")
    return decision


def supplied_solubility(directory: Path, example: str | None, generated_ids: set[str]) -> Decision:
    from rfproteina.disease.solubility import read_solubility

    report = read_solubility(directory, candidate=example)
    if blocker := report.get("solubility_blocker"):
        reasons = {
            "solubility_run_only": "The table has no candidate_id column and belongs to the whole run.",
            "solubility_candidate_missing": "No solubility or aggregation row identifies this candidate.",
            "solubility_residue_profile": "A residue profile cannot supply a whole-candidate measurement.",
            "cell_unparsed": "A score or its scale contains invalid or conflicting values.",
            "solubility_not_reported": "Requires a finite score with its method and source.",
            "solubility_truncated": "The table exceeds the native reader's 32-row limit.",
            "solubility_too_large": "The table exceeds the native reader's 1 MiB limit.",
        }
        return Decision(False, False, reasons.get(blocker, "The supplied solubility table could not be read."))
    reasons = []
    if example is None and any(
        row["candidate_id"] not in generated_ids for row in report["scores"] if "candidate_id" in row
    ):
        reasons.append("Some named observations do not match a generated candidate in this run.")
    if report.get("incomplete_rows"):
        reasons.append("Some rows lack a score, method, source or candidate identity.")
    if any(not row.get("unit") for row in report["scores"]):
        reasons.append("Confirm the unit or scoring scale for each reported value.")
    if reasons:
        return Decision(True, False, " ".join(reasons))
    return Decision(True, True, "Review the supplied values with each method, source and unit or scale.")


def protein_sol(directory: Path, kind: str, structures: screen_physical.Structures, budget: int) -> Decision:
    from rfproteina.disease.solubility import match_protein_sol, protein_sol_inputs

    try:
        inputs = protein_sol_inputs(directory)
    except ValueError as error:
        return Decision(False, False, f"Invalid native Protein-Sol input ({error}).")
    if inputs is None:
        return Decision(False, False, "Requires a native Protein-Sol prediction report.")
    if not structures.paths:
        return Decision(False, False, "Requires a generated sample with an identifiable designed chain.")
    predictions, fasta = inputs
    if not (structures.paths.keys() & predictions.keys() & fasta.keys()) and not any(
        re.fullmatch(r"sha256:[0-9a-f]{64}", name) for name in predictions
    ):
        return Decision(False, False, "Requires a matching FASTA ID or a native prediction named by sequence hash.")
    if sum(path.stat().st_size for path in structures.paths.values()) > budget:
        return Decision(True, False, "Sequence matching exceeds the interactive budget. Select to verify manually.")
    samples = match_protein_sol(structures.chains, *inputs).get("samples", [])
    matched = sum("score" in row for row in samples)
    if not matched:
        reasons = sorted({row["blocker"] for row in samples})
        detail = ", ".join(reasons) or "protein_sol_samples_missing"
        return Decision(False, False, f"No native prediction matches a resolved designed sequence ({detail}).")
    if matched < len(samples):
        return Decision(True, False, f"Only {matched} of {len(samples)} designed sequences match native predictions.")
    if kind == "peptide":
        return Decision(True, False, "Confirm that the cell-free protein-expression endpoint applies to this peptide.")
    return Decision(True, True, "The native expression-solubility prediction matches the complete designed sequence.")


def _nomination_blocker(blocker: str | None) -> Decision:
    if blocker == "ligand_input_exceeds_interactive_size":
        return Decision(True, False, "The nomination file exceeds the interactive size budget. Select to verify it.")
    return Decision(False, False, f"Requires an attributable molecular nomination ({blocker or 'no_candidates'}).")


def _nomination_coverage(scope: str, rows: list[Sequence[str]], inputs: list[dict]) -> list[str]:
    reasons = []
    eligible = sum("compound_id" in row for row in inputs)
    if eligible != len(inputs):
        reasons.append(f"Only {eligible} of {len(inputs)} nominations meet the provider's input requirements.")
    if scope == "sdf" and any(not row[0] for row in rows):
        reasons.append("An unnamed SDF record can only be reviewed by its position in the run.")
    return reasons


def badapple2_nominations(directory: Path, example: str | None) -> Decision:
    try:
        _, nominations = badapple2.prepare(directory, example)
    except badapple2.BadappleError as error:
        return Decision(False, False, f"The Badapple 2 nominations cannot be used ({error}).")
    ready = sum(candidate.blocker is None for candidate in nominations)
    if not ready:
        return Decision(False, False, "No selected nomination has a supported cyclic small-molecule structure.")
    return Decision(
        True, ready == len(nominations),
        f"Historical scaffold lookup covers {ready} of {len(nominations)} nominations. "
        "Use the assay history to choose interference counterscreens.",
    )


def p2rank_nomination(directory: Path, example: str | None) -> Decision:
    try:
        receptor = p2rank.prepare(directory)
    except p2rank.P2RankError as error:
        return Decision(False, False, f"The P2Rank receptor cannot be used ({error}).")
    detail = f"{len(receptor.residue_ids)} receptor residues with the {receptor.profile} model."
    return Decision(True, True, f"{detail} All ligands in the run share this receptor pocket map.")


def propka_nomination(directory: Path, example: str | None) -> Decision:
    try:
        plan = propka.prepare(directory, example)
    except propka.PropkaError as error:
        return Decision(False, False, f"The PROPKA structure cannot be used ({error}).")
    if plan.incomplete_termini:
        return Decision(
            True, False, "The structure has incomplete termini. Review missing atoms before using its pKa estimates.",
        )
    reason = "Estimate group pKa on the nominated structure to choose titration conditions."
    if plan.assay_ph is not None:
        reason += f" The nominated assay pH is {plan.assay_ph:g}."
    return Decision(True, True, reason)


def crest_nomination(directory: Path, example: str | None) -> Decision:
    try:
        crest.prepare(directory, example)
    except crest.CrestError as error:
        return Decision(False, False, f"The CREST structure cannot be used ({error}).")
    return Decision(True, True, "Sample gas-phase starting geometries for the nominated ligand's molecular form.")


def openantigens_target(directory: Path, example: str | None) -> Decision:
    from rfproteina.disease.openantigens_snapshot import SnapshotError

    try:
        _, evidence = openantigens.prepare(directory, example)
    except (openantigens.OpenAntigensError, SnapshotError) as error:
        return Decision(False, False, f"The OpenAntigens target cannot be used ({error}).")
    return Decision(
        True, True, f"Use the target's {evidence['topology'].lower()} topology to plan how to present it in an assay.",
    )


def metfrag_nominations(directory: Path, example: str | None) -> Decision:
    try:
        plan = metfrag.prepare(directory, example)
    except metfrag.MetfragError as error:
        return Decision(False, False, f"The MetFrag inputs cannot be used ({error}).")
    compatible = sum(not item.blocker for item in plan.candidates)
    if not compatible:
        return Decision(False, False, "No supported nomination matches the measured MetFrag precursor.")
    note = (
        f"{compatible} of {len(plan.candidates)} nominations match the measured precursor. "
        f"Compare their fragments against {plan.input_peak_count} supplied peaks."
    )
    distinct = len(plan.representatives())
    if distinct == 1:
        note += " Only one structure is eligible, so its score cannot distinguish alternative identities."
    else:
        note += f" These nominations represent {distinct} distinct structures for fragmentation."
    return Decision(True, True, note)


def fastsolv_conditions(directory: Path, example: str | None) -> Decision:
    try:
        plan = fastsolv.prepare(directory, example)
    except fastsolv.FastsolvError as error:
        return Decision(False, False, f"The fastsolv inputs cannot be used ({error}).")
    ready = sum(smiles is not None for _, smiles in plan.candidates)
    if not ready:
        return Decision(False, False, "No nominated solute is within the supported fastsolv chemistry scope.")
    return Decision(
        True, ready == len(plan.candidates),
        f"Solubility screening covers {ready} of {len(plan.candidates)} nominations. The requested solvent and "
        f"temperature conditions bring the total number of predictions to {ready * len(plan.conditions)}.",
    )


def jazzy_nominations(directory: Path, example: str | None) -> Decision:
    try:
        _, nominations = jazzy.prepare(directory, example)
    except jazzy.JazzyError as error:
        return Decision(False, False, f"The Jazzy nomination cannot be used ({error}).")
    ready = sum(smiles is not None for _, smiles in nominations)
    if not ready:
        return Decision(False, False, "No nominated form meets the Jazzy chemistry and stereochemistry requirements.")
    return Decision(
        True, ready == len(nominations),
        f"Hydration estimates cover {ready} of {len(nominations)} nominations in their supplied neutral forms.",
    )


def espsim_pair(directory: Path, example: str | None) -> Decision:
    try:
        espsim.prepare(directory, example)
    except espsim.EspsimError as error:
        return Decision(False, False, f"The ESP-Sim nomination cannot be used ({error}).")
    return Decision(True, True, "Compare shape and electrostatics separately in the supplied common coordinate frame.")


def dimorphite_nominations(directory: Path, example: str | None) -> Decision:
    try:
        plan = dimorphite.prepare(directory, example)
    except dimorphite.DimorphiteError as error:
        return Decision(False, False, f"The Dimorphite-DL input cannot be used ({error}).")
    ready = sum(row.blocker is None for row in plan.nominations)
    if not ready:
        return Decision(False, False, "No selected nomination meets the Dimorphite-DL input requirements.")
    note = f"Declared pH {plan.ph_min:g} to {plan.ph_max:g}. {ready} of {len(plan.nominations)} nominations ready."
    return Decision(True, True, note)


def qupkake_nominations(directory: Path, example: str | None, budget: int) -> Decision:
    from rdkit import Chem, rdBase

    try:
        tautomerize, mode, configured = qupkake.settings_for(directory)
    except (OSError, ValueError, RecursionError):
        return Decision(False, False, "The QupKake manifest has invalid settings.")
    scope, rows, blocker = ligand_inputs.nominated_inputs(directory, mode, example, max_bytes=budget)
    if blocker or not rows:
        return _nomination_blocker(blocker)
    with rdBase.BlockLogs():
        inputs, graphs = qupkake.prepare_nominations(rows, sdf=mode == "sdf")
    if not graphs:
        blockers = ", ".join(sorted({row["blocker"] for row in inputs}))
        return Decision(False, False, f"No nomination meets QupKake's molecular input requirements ({blockers}).")
    reasons = _nomination_coverage(scope, rows, inputs)
    if not configured:
        reasons.append("Quantum pKa calculation is opt-in. The run has no QupKake manifest.")
    if len(graphs) > 1:
        reasons.append(f"{len(graphs)} distinct input states require separate quantum calculations.")
    else:
        mol = Chem.MolFromSmiles(next(iter(graphs.values())))
        if all(atom.GetAtomicNum() == 6 for atom in mol.GetAtoms()):
            reasons.append("Confirm the site-specific pKa question for this hydrocarbon.")
    if reasons:
        return Decision(True, False, " ".join(reasons))
    reason = "One supported molecular state is nominated for the requested microscopic pKa calculation."
    if tautomerize:
        reason += " The manifest also requests a tautomer search."
    return Decision(True, True, reason)


def sygma_nominations(directory: Path, example: str | None, budget: int) -> Decision:
    from rdkit import Chem, rdBase

    try:
        phase1, phase2, mode, configured = ligand_metabolites.settings_for(directory)
    except (OSError, ValueError, RecursionError):
        return Decision(False, False, "The SyGMa manifest has invalid settings.")
    scope, rows, blocker = ligand_inputs.nominated_inputs(directory, mode, example, max_bytes=budget)
    if blocker or not rows:
        return _nomination_blocker(blocker)
    with rdBase.BlockLogs():
        inputs, parents = ligand_metabolites.prepare_parents(rows, sdf=mode == "sdf")
    if not parents:
        blockers = ", ".join(sorted({row["blocker"] for row in inputs}))
        return Decision(False, False, f"No nomination meets SyGMa's parent requirements ({blockers}).")
    reasons = _nomination_coverage(scope, rows, inputs)
    if not configured:
        reasons.append("Metabolite enumeration is opt-in. The run has no SyGMa manifest.")
    if max(phase1, phase2) > 1:
        reasons.append("The scenario repeats a reaction phase. Select it to expand the search.")
    if len(parents) > 1:
        reasons.append(f"{len(parents)} distinct parents require separate metabolite rankings.")
    else:
        parent = Chem.MolFromSmiles(next(iter(parents.values())))
        if any(s.specified != Chem.StereoSpecified.Specified for s in Chem.FindPotentialStereo(parent)):
            reasons.append("Confirm the parent drug's unspecified stereochemistry before ranking metabolites.")
    if reasons:
        return Decision(True, False, " ".join(reasons))
    return Decision(True, True, "One identified parent is ready for metabolite ranking and analytical follow-up.")


def posebusters_nominations(directory: Path, example: str | None, budget: int) -> Decision:
    from rdkit import rdBase

    from rfproteina.disease.pose_validation import PoseUnreadableError, read_nominated_pose

    _, rows, blocker = ligand_inputs.nominated_inputs(directory, "sdf", example, max_bytes=budget)
    if blocker or not rows:
        return _nomination_blocker(blocker)
    names = Counter(row[0] for row in rows)
    errors = set()
    ready = 0
    with rdBase.BlockLogs():
        for name, structure, problem in rows:
            if not name or names[name] != 1:
                errors.add("missing_or_duplicate_pose_id")
                continue
            try:
                read_nominated_pose(structure, problem)
            except PoseUnreadableError as error:
                errors.add(str(error))
            except (RuntimeError, ValueError, OverflowError):
                errors.add("ligand_unparsable")
            else:
                ready += 1
    if errors:
        detail = ", ".join(sorted(errors))
        return Decision(bool(ready), False, f"{ready} of {len(rows)} nominated poses have usable 3D input ({detail}).")
    if ready > 1:
        return Decision(True, False, f"Each of the {ready} poses needs a separate geometry and energy calculation.")
    return Decision(True, True, "The nominated 3D pose is ready for the ligand-only geometry and energy checks.")


def cellprofiler_staining(directory: Path, example: str | None) -> Decision:
    return Decision(*cellprofiler_stain.readiness(directory, example))


def pksim_exposure(directory: Path, example: str | None) -> Decision:
    return Decision(*pksim.readiness(directory, example))


def httk_exposure(directory: Path, example: str | None) -> Decision:
    """Match nominated ligand structures to the EPA reference before offering the simulation."""
    try:
        plan = httk.prepare(directory, example)
    except (ImportError, OSError, RuntimeError, ValueError):
        return Decision(False, False, "The EPA httk scenario or molecular nominations could not be inspected.")
    if plan.blocker:
        return Decision(False, False, f"No attributable EPA httk scenario ({plan.blocker}).")
    ready = sum(member.blocker is None for member in plan.members)
    if not ready:
        blockers = sorted({member.blocker for member in plan.members if member.blocker in httk.BLOCKERS})
        reason = ", ".join(blockers) or "httk_candidate_not_nominated"
        return Decision(False, False, f"No candidate has a verified EPA reference structure ({reason}).")
    return Decision(
        True, False,
        f"{ready} of {len(plan.members)} candidate structures match their EPA reference identifiers. "
        "Select explicitly to compare modeled human oral steady-state exposure per unit intake.",
    )


def cyp_substrates(directory: Path, example: str | None) -> Decision:
    try:
        plan = cypstrate.prepare(directory, example)
    except cypstrate.CypstrateError as exc:
        return Decision(False, False, f"Check the CYPstrate metabolism hypotheses ({exc}).")
    if plan.blocker == "cypstrate_nonsystemic":
        return Decision(False, False, "The nominated molecules have a declared non-systemic exposure.", resolved=True)
    if plan.blocker:
        return Decision(False, False, f"No eligible systemic CYP substrate hypothesis ({plan.blocker}).")
    ready = sum(m.blocker is None for m in plan.members)
    if ready < len(plan.members):
        reasons = ", ".join(sorted({m.blocker for m in plan.members if m.blocker}))
        return Decision(bool(ready), False, f"{ready} of {len(plan.members)} CYPstrate nominations ready ({reasons}).")
    return Decision(True, True, "Systemic exposure and specific CYP metabolism hypotheses are declared.")


def scscore_comparison(directory: Path, example: str | None) -> Decision:
    try:
        comparison = scscore.prepare(directory, example)
    except scscore.SCScoreError as exc:
        return Decision(False, False, f"The SCScore comparison could not be read ({exc}).")
    if comparison.blocker == scscore.RESOLVED:
        route = comparison.evidence["verified_route_ref"]
        costs = comparison.evidence["procurement_costs_ref"]
        return Decision(
            False, False, f"Verified route {route} and procurement-cost record {costs} resolve this comparison.",
            resolved=True,
        )
    if comparison.blocker:
        return Decision(False, False, f"No eligible SCScore comparison ({comparison.blocker}).")
    members = comparison.selected(example)
    ready = sum(m.blocker is None for m in members)
    if ready != len(members):
        reasons = ", ".join(sorted({m.blocker for m in members if m.blocker}))
        return Decision(bool(ready), False, f"{ready} of {len(members)} selected nominations are eligible ({reasons}).")
    return Decision(True, True, "The nominated molecules are ready for a synthetic complexity tie-break.")


def redocking_controls(directory: Path, example: str | None) -> Decision:
    """Verify identity and frame in the picker. Only execution enumerates atom mappings."""
    try:
        blockers = [blocker for _, _, blocker in spyrmsd_pose.controls(directory, example)]
    except spyrmsd_pose.ControlError as exc:
        return Decision(False, False, f"No matched redocking control ({exc}).")
    ready = blockers.count(None)
    if ready < len(blockers):
        reasons = ", ".join(sorted({reason for reason in blockers if reason}))
        return Decision(bool(ready), False, f"{ready} of {len(blockers)} redocking controls are ready ({reasons}).")
    return Decision(True, True, "Ligand identities and declared receptor frames match the experimental references.")


def recorded_pose_checks(evidence: screen_inventory.Evidence, generated_ids: set[str]) -> Decision:
    from rfproteina.utils.pose_checks import POSEBUSTERS_DOCK_CHECKS

    record = evidence.coverage(screen_inventory.BY_KEY["posebusters"], include_values=True)
    rows = record["observations"]
    reason = "Review the scope, scale and measurement coverage of the stored PoseBusters result."
    if record["omitted_observations"] or any(row["candidate_id"] not in generated_ids for row in rows):
        return Decision(True, False, reason)
    published = {}
    for row in rows:
        if row["source"] == "published_metrics":
            values = published.setdefault(row["candidate_id"], {})
            if row["endpoint"] in values:
                return Decision(True, False, reason)
            values[row["endpoint"]] = row["value"]
        elif (
            row["endpoint"] not in POSEBUSTERS_DOCK_CHECKS
            or row["value"] not in (0, 1)
            or row.get("unit", "").casefold() not in {"boolean", "binary", "bool"}
        ):
            return Decision(True, False, reason)
    for values in published.values():
        count = values.get("posebusters_metrics.n_measured", 0)
        total = values.get("posebusters_metrics.n_ligands")
        rate = values.get("posebusters_metrics.pass_rate")
        if count <= 0 or not count.is_integer() or count != total or rate is None or not 0 <= rate <= 1:
            return Decision(True, False, reason)
    return Decision(True, True, "Review the attributable ligand-pose checks with their recorded pass or fail results.")


def recorded_stereochemistry(evidence: screen_inventory.Evidence) -> Decision:
    """The local ChiralAccuracy producer reports means on [0, 1], including its percent-named columns."""
    endpoints = {
        "tetrahedral_percent_correct_chirality", "non_tetrahedral_percent_correct_chirality",
        "double_bond_percent_correct_stereochemistry",
    }
    record = evidence.coverage(screen_inventory.BY_KEY["stereochemistry"], include_values=True)
    review = Decision(True, False, "Confirm the reference comparison, endpoint, scale and candidate coverage.")
    if record["omitted_observations"]:
        return review
    seen = set()
    for row in record["observations"]:
        endpoint = row["endpoint"].removeprefix("chiral_accuracy.")
        identity = (row["candidate_id"], endpoint)
        unit = "fraction" if row["source"] == "published_metrics" else row["unit"].strip().casefold()
        maximum = {"fraction": 1, "%": 100, "percent": 100}.get(unit)
        if endpoint not in endpoints or maximum is None or not 0 <= row["value"] <= maximum or identity in seen:
            return review
        seen.add(identity)
    expected = {evidence.example} if evidence.example else evidence.candidate_ids
    if not expected or {name for name, _ in seen} != expected:
        return review
    return Decision(True, True, "Each selected candidate has a reference-comparison result on a defined chiral scale.")


def completed_result(key: str, evidence: screen_inventory.Evidence, budget: int) -> Decision:
    from importlib.util import find_spec

    from rfproteina.disease import screen_assays

    entry = screen_assays.inputs(evidence.directory, key)
    name = entry["candidate_id"]
    if (evidence.example is not None and evidence.example != name) or (
        evidence.candidate_ids is not None and name not in evidence.candidate_ids
    ):
        return Decision(False, False, "The result manifest identifies a different candidate.")
    names = entry["results"] if key == "openfe" else [entry[n] for n in ("report", "forward", "reverse")]
    if sum((evidence.directory / n).stat().st_size for n in names) > budget:
        return Decision(True, False, "The completed results exceed the interactive byte budget. Select to verify.")
    if key == "pmx":
        if any(find_spec(module) is None for module in ("numpy", "scipy", "pymbar")):
            return Decision(True, False, "The work files are present. Independent BAR verification requires PyMBAR.")
        return Decision(True, False, "The report and both work files are present. "
                        "Select this method to fit BAR and verify the reported transformation and uncertainty.")
    report = screen_assays.safe_measure(evidence.directory, key, evidence.example)
    if report["status"] != "reported":
        return Decision(False, False, "The supplied results could not verify the requested transformation.")
    if evidence.candidate_ids is None:
        return Decision(True, False, "Review the result manifest's candidate identity against the original run.")
    if report["result"]["uncertainty"] is None:
        return Decision(True, False, "The completed edge lacks independent repeats for a between-repeat uncertainty.")
    return Decision(True, True, "The candidate has verified complex and solvent legs with repeat-based uncertainty.")


def supplied_panel(directory: Path) -> Decision:
    from rfproteina.disease import screen_assays

    report = screen_assays.supplied_panel(directory)
    if not report["measured_axes"]:
        return Decision(False, False, "No usable panel concentration matches the supplied compound and unit rules.")
    if not report["attributed_assays"] or report["status"] != "reported":
        return Decision(True, False, "Review compound identity, assay provenance or incomplete supplied readings.")
    return Decision(True, True, "The nominated compound has attributable assay concentrations on a defined scale.")


def prepared_structure(key: str, structures: screen_physical.Structures, budget: int) -> Decision:
    from rfproteina.disease import prodigy, screen, tnp, vina_pose

    providers = {
        "tnp_profile": tnp.RunProfiles, "vina_pose": vina_pose.RunPoses, "prodigy_affinity": prodigy.RunAffinity,
    }
    prepared = providers[key].configured(structures.directory)
    if prepared is None or prepared.blocker:
        return Decision(False, False, "Requires readable preparation settings for this method.")
    paths = structures.paths
    settings = None if key == "prodigy_affinity" else prepared.settings
    foreign = structures.candidate is None and settings is not None and bool(settings.keys() - paths.keys())
    ready = 0
    reasons = set()
    for name, path in paths.items():
        if settings is not None and name not in settings:
            reasons.add("sample_settings_absent")
            continue
        try:
            text = screen._design_text(path, max_bytes=budget)
            if not text:
                reasons.add("structure_unreadable")
                continue
            budget -= len(text.encode("utf-8"))
            if key == "tnp_profile":
                domain = tnp.Domain.parse(prepared.settings[name])
                sequence, observed = tnp._extract(text, domain)
                if observed < len(sequence):
                    reasons.add("partial_observed_domain")
            elif key == "vina_pose":
                pose = vina_pose.Pose.parse(structures.directory, prepared.settings[name])
                budget -= pose.receptor.stat().st_size + pose.ligand.stat().st_size
                if budget < 0:
                    raise screen.DesignTextBudgetError
                vina_pose.verified_inputs(pose, text, {})
            else:
                prodigy.verify_interface(text, prepared.partners)
        except screen.DesignTextBudgetError:
            return Decision(True, False, "The preparation exceeds the interactive byte budget. Select to verify it.")
        except (ValueError, OSError) as error:
            reason = str(error)
            reasons.add(reason if re.fullmatch(r"[a-z][a-z0-9_]{0,79}", reason) else "preparation_unreadable")
            continue
        ready += 1
    if not ready:
        detail = ", ".join(sorted(reasons)) or "no_matching_sample"
        return Decision(False, False, f"No preparation matches a selected structure ({detail}).")
    if foreign or reasons:
        return Decision(True, False, "Review partial coordinate coverage or unmatched preparation settings.")
    if ready > 1:
        return Decision(True, False, f"Each of the {ready} prepared candidates requires a separate calculation.")
    if key == "tnp_profile":
        return Decision(True, True, "The nominated VHH chain, sequence digest and observed domain agree.")
    if key == "prodigy_affinity":
        return Decision(True, True, "The selected model has two complete protein partners with a measurable interface.")
    return Decision(True, True, "The prepared ligand and receptor match the selected structure and scoring box.")


def ligand_component(structures: screen_physical.Structures, budget: int) -> Decision:
    from rfproteina.disease import modality, screen

    paths = structures.paths
    ready = 0
    warnings = set()
    for path in paths.values():
        try:
            text = screen._design_text(path, max_bytes=budget)
        except screen.DesignTextBudgetError:
            return Decision(True, False, "The component files exceed the interactive byte budget. Select to verify.")
        if not text:
            continue
        budget -= len(text.encode("utf-8"))
        weight = modality.identified_component_weight(screen._chem_components(text))
        if weight:
            ready += 1
            if warning := weight.get("input_warning"):
                warnings.add(warning)
    if not ready:
        return Decision(False, False, "Requires one named non-water ligand component with a usable formula weight.")
    if ready < len(paths) or warnings:
        detail = ", ".join(sorted(warnings)) or "missing or ambiguous components"
        return Decision(True, False, f"Weights available for {ready} of {len(paths)} candidates. Review {detail}.")
    return Decision(True, True, "Each selected candidate has one identified ligand component with a formula weight.")


def coordinate_measurement(key: str, structures: screen_physical.Structures) -> Decision:
    from rfproteina.disease.screen import DesignTextBudgetError

    try:
        arrays = structures.coordinate_arrays
    except DesignTextBudgetError:
        return Decision(True, False, "The structures exceed the interactive byte budget. Select to inspect them.")
    ready = 0
    short = False
    for name, atoms in arrays.items():
        if atoms is None:
            continue
        if key == "clashes":
            ready += screen_physical.has_clash_pair(atoms)
            continue
        if key == "bond_length":
            ready += screen_physical.has_measurable_bonds(atoms)
            continue
        _, span = screen_physical.psea_chain(atoms, structures.chain_ids[name])
        ready += span > 0
        short |= 0 < span <= 5
    if key == "bond_length":
        if not ready:
            return Decision(False, False, "Requires bonds in the parsed structure with finite endpoint coordinates.")
        if ready < len(arrays):
            return Decision(True, False, f"Only {ready} of {len(arrays)} structures have measurable bond lengths.")
        return Decision(True, True, "Each selected candidate has bonded atoms available for the geometry check.")
    if key == "clashes":
        if not ready:
            return Decision(False, False, "Requires finite coordinates and an eligible nonbonded atom pair.")
        if ready < len(arrays):
            return Decision(True, False, f"Only {ready} of {len(arrays)} structures have measurable nonbonded pairs.")
        return Decision(True, True, "The native clash check can compare nonbonded pairs in each selected structure.")
    if not ready:
        return Decision(False, False, "Requires observed amino-acid C-alpha coordinates on one identifiable chain.")
    if ready < len(arrays) or short:
        return Decision(True, False, "Review missing chain coverage or C-alpha traces of five residues or fewer.")
    return Decision(True, True, "The selected protein chains have consecutive C-alpha coordinates for P-SEA.")


def polypeptide_measurement(key: str, structures: screen_physical.Structures) -> Decision:
    from rfproteina.disease import chemical, modality
    from rfproteina.disease.screen import DesignTextBudgetError

    if key == "sequence_diversity" and (structures.candidate is not None or len(structures.paths) < 2):
        return Decision(False, False, "Requires a run-level comparison of at least two designed chains.")
    try:
        polymers = structures.planned_polypeptides
    except DesignTextBudgetError:
        return Decision(True, False, "Reading the designed chains exceeds the interactive byte budget.")
    ready = [polymer for polymer in polymers.values() if polymer is not None]
    if key == "sequence_diversity":
        from rfproteina.disease.diversity import DIVERSITY_DESIGN_LIMIT, KMER_WIDTH

        if len(ready) < 2:
            return Decision(False, False, "Fewer than two designed polypeptides have identifiable sequences.")
        if len(ready) != len(polymers) or any(p.modified for p in ready):
            return Decision(True, False, "The cohort contains unresolved chains or annotated modifications.")
        if len(ready) > DIVERSITY_DESIGN_LIMIT:
            return Decision(True, False, f"Above {DIVERSITY_DESIGN_LIMIT} chains the method reports counts only.")
        lengths = Counter(len(p.sequence) for p in ready)
        if max(lengths.values()) < 2 and sum(count for n, count in lengths.items() if n >= KMER_WIDTH) < 2:
            return Decision(True, False, "Only distinct counts are available. No two chains share a comparable scale.")
        return Decision(True, True, "Compare the identified designs and count exact replicates in this cohort.")
    if key == "instability":
        eligible = [polymer for polymer in ready if len(polymer.sequence) >= 2]
        if not eligible:
            return Decision(False, False, "Requires two adjacent identified amino acids on the designed chain.")
        if len(eligible) != len(polymers):
            return Decision(True, False, f"Only {len(eligible)} of {len(polymers)} designed chains support the index.")
        if any(polymer.modified for polymer in eligible):
            return Decision(True, False, "Only the unmodified sequence can supply the published dipeptide weights.")
        return Decision(True, False, "Select for an exploratory comparison using the protein instability index.")
    if key == "aggrescan":
        eligible = [p for p in ready if len(p.sequence) >= modality._aggrescan_window(len(p.sequence))]
        if not eligible:
            return Decision(False, False, "Requires a complete standard chain long enough for an AGGRESCAN window.")
        if len(eligible) != len(polymers) or any(p.modified for p in eligible):
            return Decision(True, False, "Review partial cohort coverage or the canonical form of a modified chain.")
        return Decision(True, True, "The identified polypeptide sequences support the published aggregation windows.")
    if not ready:
        return Decision(False, False, "Requires an identified designed polypeptide with a complete standard sequence.")
    if len(ready) != len(polymers):
        return Decision(True, False, f"Only {len(ready)} of {len(polymers)} designed chains are fully identified.")
    if key == "gravy":
        if any(polymer.modified for polymer in ready):
            return Decision(True, False, "The scale describes the unmodified residues of this annotated chain.")
        return Decision(True, True, "All identified residues have a Kyte-Doolittle hydropathy value.")
    if key == "protein_charge":
        if any(polymer.modified for polymer in ready):
            return Decision(True, False, "An annotated modification or disulfide may change the charge model.")
        if not all(chemical.has_isoelectric_crossing(p.sequence) for p in ready):
            return Decision(True, False, "Charge at pH 7.4 is available. A pI is unresolved between pH 0 and 14.")
        return Decision(True, True, "The identified chain supports charge at pH 7.4 and a theoretical pI.")
    if any(polymer.modified for polymer in ready):
        return Decision(True, False, "An annotated modification or disulfide needs a separate intact-mass assignment.")
    return Decision(True, True, "The identified chain supports residue counts and an unmodified linear-chain mass.")
