"""Offline evidence for interpreting the endpoint reported by a screening method."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class Evidence:
    status: str
    summary: str
    interpretation: str
    source: str = ""
    basis: str = field(kw_only=True, default="Unspecified evidence basis")

    def as_record(self) -> dict[str, str]:
        record = {
            "status": self.status,
            "basis": self.basis,
            "summary": self.summary,
            "interpretation": self.interpretation,
        }
        if self.source:
            record["source"] = self.source
        return record


def _bindingdb_benchmark(backend: str) -> Evidence | None:
    """Use the evaluation saved alongside the installed estimator without loading it."""
    try:
        from rfproteina.disease import ligand_regressor
    except (ImportError, OSError):
        return None
    try:
        _, manifest, actual = ligand_regressor.regressor_manifest(ligand_regressor.default_model_dir(backend))
    except (OSError, ValueError, RuntimeError, ligand_regressor.RegressorUnavailableError):
        return None
    if actual != backend:
        return None
    estimates = []
    missing = []
    for axis, label in (("nav1_5", "NaV1.5"), ("bcrp", "BCRP"), ("bsep", "BSEP")):
        score = manifest["axes"].get(axis, {}).get("score")
        if isinstance(score, dict) and score.get("axis") == axis and score.get("backend") == backend:
            error = score.get("rmse_log10")
            rows = score.get("test_rows")
            splits = score.get("replicates")
            if (
                type(error) in (float, int) and 0 <= error <= sys.float_info.max
                and type(rows) is int and type(splits) is int and 1 <= splits <= rows <= sys.maxsize
            ):
                split_label = "scaffold split" if splits == 1 else "scaffold splits"
                estimates.append(f"{label} {error:.3g} ({rows:,} test rows, {splits} {split_label})")
                continue
        missing.append(label)
    if not estimates:
        return None
    summary = "The local model manifest reports held-out RMSE in log10(nM). " + ", ".join(estimates) + "."
    if missing:
        summary += " No verified estimate for " + ", ".join(missing) + "."
    return Evidence(
        "local_benchmark", summary,
        "Test rows are pooled across splits and may include the same compound more than once. "
        "A difference of one log10 unit is tenfold in concentration. Confirm close predictions with a matched "
        "assay on that target before choosing a lead.",
        basis="Held-out scaffolds from local model evaluation",
    )


def for_method(key: str, operation: str, *, installed: bool = True) -> dict[str, str]:
    """Historical measurements can come from a different release than today's local producer."""
    evidence = _METHODS.get(key)
    if key == "toxscan_fingerprint_ames" and installed and operation == "compute":
        from rfproteina.disease import toxscan_ames_model

        metrics = toxscan_ames_model.benchmark()
        if metrics is not None:
            evidence = Evidence(
                "local_benchmark",
                f"The installed fingerprint model scored AUROC {metrics['roc_auc']:.3f} and "
                f"{metrics['accuracy']:.1%} accuracy at 0.5 on {metrics['n_test']:,} held-out Ames structures "
                f"({metrics['test_positives']:,} positive and {metrics['test_negatives']:,} negative). "
                f"{metrics['test_fingerprint_seen_in_train']:,} test structures shared a training fingerprint.",
                f"At 0.5 the model identified {metrics['tp'] / metrics['test_positives']:.1%} of positives and "
                f"{metrics['tn'] / metrics['test_negatives']:.1%} of negatives. Calling every compound positive "
                f"would score {metrics['test_positives'] / metrics['n_test']:.1%} accuracy. Use this score to "
                "prioritize confirmatory Ames testing.",
                basis="Held-out Ames structures from the released ToxScan test partition",
            )
    if operation == "compute" and installed and key in {"bindingdb_forest", "bindingdb_chemprop"}:
        evidence = _bindingdb_benchmark("forest" if key == "bindingdb_forest" else "chemprop") or evidence
    if evidence is None:
        evidence = Evidence(
            "not_verified", "A validation estimate for this method and endpoint has not been verified.",
            "Check this method against reference samples with known results before relying on it to select candidates.",
            basis="Validation estimate unavailable",
        )
    if operation == "recorded_evidence" and evidence.status == "published_benchmark":
        evidence = replace(evidence, summary=evidence.summary + " Check the producer version on the saved observation.")
    return evidence.as_record()


_METHODS = {
    "toxscan_fingerprint_ames": Evidence(
        "not_verified", "Install a verified local fit to display its held-out Ames accuracy and AUROC.",
        "This single-endpoint adaptation uses the study's fingerprint branch. The full model's published "
        "accuracy applies to a different predictor.", basis="Local fit required",
    ),
    "badapple2": Evidence(
        "source_evidence",
        "Badapple 2 drew on 906 NIH screening assays. Its scaffold pScores use compounds tested in at least "
        "50 assays. These counts describe coverage. A candidate-specific accuracy estimate is unavailable.",
        "A high pScore can justify an orthogonal counterscreen for a cyclic lead. Read the active and tested "
        "counts for its scaffold. Zero is a reported low score. An absent or unscored scaffold leaves the "
        "question unresolved.",
        "https://chiltepin.health.unm.edu/badapple2/about.html",
        basis="Historical PubChem assay coverage by scaffold",
    ),
    "metfrag": Evidence(
        "published_benchmark",
        "Using fragmentation alone against PubChem, MetFrag 2.2 ranked the correct structure first for 30 of "
        "473 merged spectra. The local 2.6.11 screen compares a user-supplied shortlist of protonated candidates.",
        "Use the explained peaks to choose reference standards for follow-up. Multiple labels for the same "
        "molecule represent one possible assignment. Different isomers can share the top score. Confirm identity "
        "against a standard under matched analytical conditions.",
        "https://doi.org/10.1186/s13321-016-0115-9",
        basis="Earlier release tested on reference spectra against PubChem",
    ),
    "fastsolv": Evidence(
        "published_benchmark",
        "For held-out solutes, the FASTPROP model underlying fastsolv had RMSE of 0.83 on SolProp and 0.95 on "
        "Leeds in log10(mol/L). Respectively, 78.1% and 73.8% were within tenfold of measured solubility. "
        "The authors reported those figures for the model before forming the four-model ensemble used here.",
        "Use large predicted differences to prioritize pure organic solvent and temperature conditions. Measure "
        "saturation solubility for close choices using the intended solid form. The four-model spread has no "
        "calibrated prediction-interval coverage.",
        "https://pmc.ncbi.nlm.nih.gov/articles/PMC12365040/",
        basis="Experimental solubility benchmarks with held-out solutes",
    ),
    "jazzy": Evidence(
        "published_benchmark",
        "On 3,313 compounds in the external Guthrie set, the authors reported mean absolute error of 5.89 kJ/mol "
        "and RMSE of 9.19 kJ/mol for total hydration free energy. These errors apply only to hydration energy.",
        "Use the estimates to compare gas-to-water transfer for related neutral molecular forms. Close "
        "predictions warrant a measured hydration reference. The donor and acceptor descriptors have their "
        "own scales, and solid solubility requires a separate measurement.",
        "https://pmc.ncbi.nlm.nih.gov/articles/PMC10011384/",
        basis="External experimental hydration free energies",
    ),
    "crest": Evidence(
        "reference_dependent",
        "The CREST 3.0.2 quick search samples gas-phase conformers with GFN2-xTB. A recovery rate against "
        "measured solution or bound conformations has not been established for this reduced search.",
        "Use distinct geometries as starting structures. Relative energies in kcal/mol compare conformers of "
        "the same molecular form in this run. Solution populations and the bound pose require separate evidence.",
        "https://crest-lab.github.io/crest-docs/page/documentation/keywords.html",
        basis="Reduced conformer search with no verified experimental recovery rate",
    ),
    "espsim": Evidence(
        "published_benchmark",
        "Across about 3,000 neutral molecules, the mean Carbó similarity was 0.78 when the authors compared "
        "each molecule's own Gasteiger and quantum-derived RESP electrostatic potentials.",
        "Use the two scores to inspect where prealigned ligands differ in shape or electrostatics. The study "
        "compared charge models on identical molecules. It supplies neither a ligand-pair cutoff nor an "
        "estimate of binding probability.",
        "https://doi.org/10.1021/acs.jcim.1c01535",
        basis="Agreement between charge models on the same molecule",
    ),
    "propka": Evidence(
        "published_benchmark",
        "The original PROPKA3 study reported RMS deviations from measured pKa of 0.79 units for Asp and Glu, "
        "0.75 for Tyr, 0.65 for Lys and 1.00 for His. These protein residue results do not validate bound-ligand "
        "groups or independently benchmark the local 3.5.1 release.",
        "When a group pKa is close to the assay pH, an error of this size can change which protonation state "
        "predominates. Use the structure to choose sites and pH values for titration, and review incomplete "
        "or coupled groups before interpreting a shift.",
        "https://pubmed.ncbi.nlm.nih.gov/26596171/",
        basis="Published comparison with measured protein residue pKa",
    ),
    "p2rank": Evidence(
        "published_benchmark",
        "On COACH420 and HOLO4K ligand-bound experimental structures, top-n pocket recovery was 71.6% and "
        "72.7% with the default model, and 71.2% and 72.1% without B-factors. The study used the number of "
        "ligands for n and counted a center within 4 Å of a ligand atom as a hit.",
        "Use the centers and residue IDs to choose receptor sites to examine. This benchmark measures site "
        "localization on experimental structures. It gives no candidate binding probability or accuracy "
        "estimate for a predicted receptor.",
        "https://pmc.ncbi.nlm.nih.gov/articles/PMC10353840/",
        basis="Pocket localization on two experimental structure sets",
    ),
    "openantigens": Evidence(
        "source_evidence",
        "The November 1, 2026 human index covers 5,328 cell-surface and secreted targets. This is coverage of "
        "topology and structural annotations. Candidate binding and tissue staining were not scored.",
        "Use the target accession and topology to choose how to present the target in a binding assay. To "
        "assess tissue cross-reactivity, compare candidate and control staining on a human tissue panel.",
        "https://openantigens.org/portal_metadata.json",
        basis="Release-specific human target index coverage",
    ),
    "dimorphite_dl": Evidence(
        "published_benchmark",
        "At pH 6.4 to 8.4 and precision factor 1.0, Dimorphite-DL v1.0 returned exactly the expected "
        "states for 58.8% of its reference compounds, excess states for 40.2% and incorrect states for 0.9%. "
        "This study did not test the configured v2.1.0 release.",
        "Use the returned states as alternatives for the assay pH. Their presence in the list gives no "
        "population fractions. If the dominant state matters to the decision, check experimental ionization data.",
        "https://doi.org/10.1186/s13321-019-0336-9",
        basis="Rule evaluation on the original development data",
    ),
    "qupkake": Evidence(
        "published_benchmark",
        "On five external experimental sets, the authors reported microscopic pKa RMSE of 0.5 to 0.8 pKa units. "
        "Error depends on the ionization site and chemical series.",
        "Near the planned assay pH, an error of this size can change which protonation state appears more abundant. "
        "Use the estimate to choose pH values to test. Measure pKa when that choice changes the experiment.",
        "https://doi.org/10.1021/acs.jctc.4c00328",
        basis="External experimental test sets",
    ),
    "sygma": Evidence(
        "published_benchmark",
        "The published rules recovered 68% of observed metabolites on an independent test set, with 30% of observed "
        "metabolites among the top three predictions. Recovery depends on the enumeration settings.",
        "Use the list to nominate compounds to look for by mass spectrometry. The study missed observed products, "
        "so keep the analytical search open to unexpected metabolites.",
        "https://doi.org/10.1002/cmdc.200700312",
        basis="Independent test of metabolite recovery",
    ),
    "prodigy_affinity": Evidence(
        "published_benchmark",
        "On the full 81-complex development set, Pearson r was 0.73 and binding free-energy RMSE "
        "was 1.89 kcal/mol against experiment.",
        "Small predicted affinity differences are difficult to resolve at this error scale. Prioritize a direct "
        "binding measurement when choosing between closely scored protein pairs.",
        "https://wenmr.science.uu.nl/prodigy/method",
        basis="Development-set fit",
    ),
    "toxscan": Evidence(
        "published_benchmark",
        "The authors reported an AUROC of 0.879 for Ames mutagenicity, 0.775 for carcinogenicity and 0.671 for "
        "the hERG IC50 cutoff of 10 micromolar (Supplementary Table S1).",
        "The study model separated positives from negatives better for Ames than for that hERG cutoff. "
        "Each reported classification includes its own test-set accuracy and AUROC. Use the applicable assay "
        "to confirm a call. The export omits checkpoint identity.",
        "https://doi.org/10.1021/acs.est.5c04614.s001",
        basis="Endpoint-specific performance on the authors' test data",
    ),
    "cypstrate": Evidence(
        "published_benchmark",
        "Matthews correlation across nine CYP isoforms ranged from 0.37 to 0.81 in best-performance mode "
        "with abstentions and from 0.32 to 0.70 in full-coverage mode on a stratified 20% holdout.",
        "Matthews correlation summarizes agreement across both classes. Reliability varied by enzyme. Use a call "
        "to nominate an isoform for a metabolism assay and keep abstentions unresolved.",
        "https://doi.org/10.3390/molecules26154678",
        basis="Internal holdout by compound",
    ),
    "bayesherg": Evidence(
        "published_benchmark",
        "On external set 1 (44 compounds, hERG IC50 cutoff 10 µM), the model identified 25 of 30 blockers "
        "(83.3% sensitivity) and 12 of 14 nonblockers (85.7% specificity).",
        "Five known blockers were missed in this small panel. A flag can prioritize patch-clamp testing, "
        "and a low score leaves residual uncertainty about channel blockade.",
        "https://doi.org/10.1093/bib/bbac211",
        basis="External hERG blocker classification test",
    ),
    "delphi_psr": Evidence(
        "published_benchmark",
        "The provider reports mean ROC AUC 0.8821 (SD 0.0312) in internal 10-fold validation of the one-hot PSR "
        "model on IPI paired antibodies. This measures discrimination on the polyreactivity assay.",
        "Higher scores favor passing the provider's PSR filter (lower polyreactivity). A low score can "
        "prioritize an antibody for a confirmatory polyreactivity assay. Calibration on new antibodies is unverified.",
        "https://github.com/proteininnovation/delphi/blob/f3ce7ff9f87c673dd7ad4677d142edf9166d5e81/"
        "config/model_registry.yaml#L209-L223",
        basis="Internal cross-validation",
    ),
    "ligand_caco2": Evidence(
        "published_benchmark",
        "On external NMMPDB macrocycles, the published baseline GNN-MTL had Spearman correlation 0.40 for permeability "
        "and 0.19 for efflux ratio. The released Chemprop 2.1 weights were retrained. Their exact performance "
        "on this set is unverified.",
        "On this external set, permeability rankings tracked measurements modestly and efflux rankings weakly. "
        "Use the results to nominate compounds for a bidirectional Caco-2 assay. Small score differences are "
        "a poor basis for dropping a lead.",
        "https://doi.org/10.1021/acsomega.5c04861.s001",
        basis="External test of an earlier model",
    ),
    "spyrmsd": Evidence(
        "published_benchmark",
        "On 40,439 redocked poses, fixed-frame symmetry-corrected RMSD differed from Open Babel by at most "
        "0.00005 Å. This measures agreement between calculations. Pose recovery still depends on the reference.",
        "For a redocking control, a large displacement is unlikely to arise from rounding in the RMSD code. "
        "Check the receptor alignment and experimental pose before interpreting a docking failure.",
        "https://doi.org/10.1186/s13321-020-00455-2",
        basis="Agreement between implementations",
    ),
    "protein_sol": Evidence(
        "published_benchmark",
        "The authors reported correlation 0.621 with measured E. coli cell-free expression solubility on the "
        "data used to fit the model. Error for a new therapeutic protein or formulation is unverified.",
        "Use the score to plan a soluble-expression check. For a new host, buffer or formulation, compare with "
        "measured soluble yield under those conditions.",
        "https://doi.org/10.1093/bioinformatics/btx345",
        basis="Development-set fit",
    ),
    "cellprofiler": Evidence(
        "reference_dependent",
        "The screen subtracts mean DAB intensity in a masked negative-control field from the candidate field. "
        "Use repeat fields and matched staining controls to estimate variability. Diagnostic accuracy is unverified.",
        "Use the signed difference to choose regions for pathologist review. Inspect the masks and control slides "
        "for that specimen. For chip images, assessing a treatment effect also requires a biological comparator.",
        "https://cellprofiler-manual.s3.amazonaws.com/CPmanual/UnmixColors.html",
        basis="Paired tissue or chip fields with matched staining controls",
    ),
    "epa_httk": Evidence(
        "published_benchmark",
        "An earlier HTTK study of 74 drugs and 11 environmental chemicals reported R² about 0.34 against human "
        "steady-state concentrations inferred at 1 mg/kg/day. The current simulation's population percentiles "
        "describe variability across adults. Their coverage of observed concentrations is unverified.",
        "The historical agreement was limited. Use modeled exposure to prioritize PK follow-up. If a safety "
        "margin depends on the concentration, obtain compound-specific exposure measurements.",
        "https://pmc.ncbi.nlm.nih.gov/articles/PMC4560038/",
        basis="Human exposure comparison for an earlier model",
    ),
    "canagliflozin_pbpk": Evidence(
        "reference_dependent",
        "The authors used 22 clinical studies for calibration and evaluation. For the 188 mg reference, local "
        "release 0.8.0 predicted 30.4% urinary and 57.6% fecal recovery at seven days. The study means were "
        "32.5% and 60.4% in six participants.",
        "This comparison checks a study used in model development. Compare chip concentrations at matching time "
        "points and measure tissue injury separately. The authors also reported overprediction of the effect of "
        "hepatic impairment on parent exposure in the one available study with plasma profiles.",
        "https://doi.org/10.20944/preprints202601.2095.v1",
        basis="Clinical pharmacokinetic calibration and evaluation for one reference drug",
    ),
    "pksim": Evidence(
        "reference_dependent",
        "Peak concentration is the highest value on the exported time grid. AUC integrates the sampled interval. "
        "Exposure error requires observed concentrations for the same analyte, dosing regimen and compartment.",
        "Match sampling times and units when comparing a simulated profile with plasma, tissue or chip data. "
        "A coarse time grid can miss a peak. Assess organ injury with a separate injury measurement.",
        "https://docs.open-systems-pharmacology.org/working-with-pk-sim/pk-sim-documentation/pk-sim-simulations",
        basis="Concentration-time profile from one configured human PBPK simulation",
    ),
    "scscore": Evidence(
        "published_benchmark",
        "The 1024-bit Boolean model scored the product above its reactant on 90.8% of held-out reaction pairs "
        "(Table S4). This tests relative complexity. Route feasibility requires separate evidence.",
        "Use relative complexity to break ties between otherwise comparable molecules before chemistry review. "
        "Check the actual route, starting materials and procurement costs before committing to synthesis.",
        "https://doi.org/10.1021/acs.jcim.7b00622.s001",
        basis="Held-out reaction ordering",
    ),
    "molprobity_geometry": Evidence(
        "reference_dependent",
        "Clash counts and torsion outlier rates describe these coordinates against reference geometry. Their "
        "percentiles depend on the reference set and do not estimate the probability of a correct structure.",
        "Inspect flagged residues against the coordinates and available density before using an apparent "
        "contact to interpret binding.",
        "https://pmc.ncbi.nlm.nih.gov/articles/PMC5734394/",
        basis="Experimental reference geometry",
    ),
    "tnp_profile": Evidence(
        "reference_dependent",
        "The published flag ranges were calibrated on 36 clinical-stage VHHs. Patch scores depend on the "
        "modeled structure. Accuracy for predicting a particular developability assay is unverified.",
        "Use an unusual patch score to choose a formulation or developability assay. Recheck the modeled "
        "surface before treating a borderline flag as reproducible.",
        "https://doi.org/10.1038/s42003-026-09594-y",
        basis="Calibration on a clinical-stage reference panel",
    ),
    "vina_pose": Evidence(
        "reference_dependent",
        "This operation scores one supplied pose. Assess affinity error with matched experimental binding data "
        "and the same receptor preparation. Redocking success rates measure a different task.",
        "Use the score to investigate a prepared binding pose. Check apparent affinity differences with a "
        "binding assay using the nominated target.",
        "https://doi.org/10.1002/jcc.21334",
        basis="Fixed-pose scoring function",
    ),
    "ligand_alerts": Evidence(
        "reference_dependent",
        "A hit identifies a catalogue motif in the supplied molecular graph. Assay-specific positive controls "
        "and counter-screens are needed to estimate how often a flagged compound interferes.",
        "A motif suggests which control to include, such as an orthogonal readout. Retain the compound for "
        "review until the assay establishes whether interference occurred.",
        "https://www.rdkit.org/docs/GettingStartedInPython.html#filtering-unwanted-substructures",
        basis="Substructure rule match",
    ),
    "peptide_rbc_evidence": Evidence(
        "source_evidence",
        "The reader reports each measured hemolysis percentage at its recorded concentration and exposure time "
        "when the source accepted the assay controls. Replicate observations remain separate.",
        "Compare a formulation with observations from the same species and matrix, such as human washed red cells. "
        "Check replicate spread and test the intended formulation before drawing a blood-compatibility conclusion.",
        basis="Formulation-specific red-cell assay observations",
    ),
    "solubility": Evidence(
        "source_evidence",
        "Each row carries the producer's solubility or aggregation score, method and source. Scores from "
        "different methods can use different scales.",
        "Check how the original method defines its score. Measure solubility in the intended formulation "
        "before using it to choose a dosing concentration.",
        basis="Supplied solubility or aggregation study",
    ),
    "toxicology_table": Evidence(
        "source_evidence",
        "Review each assay's concentration in nM and its qualifier. An inequality can resolve a panel cutoff "
        "only when every concentration allowed by the bound falls on the same side.",
        "Preserve each assay's endpoint and cutoff when planning a repeat. If the bound spans the cutoff, "
        "obtain a measurement that resolves it. Check the original assay report for exposure time and precision.",
        basis="Reported assay concentrations and bounds",
    ),
}

_ADMET_RELEASE = "https://github.com/swansonk14/admet_ai/blob/v_2.0.1/admet_ai/resources/data/admet.csv"
for _key, (_summary, _interpretation) in {
    "ligand_solubility": (
        "On the upstream AqSolDB task, ADMET-AI v2.0.1 reports R² 0.817 and mean absolute error "
        "0.692 log10(mol/L).",
        "A difference of 0.692 log units is about fivefold in concentration. Use the estimate to set a dilution "
        "range, then measure solubility in the intended buffer and molecular form.",
    ),
    "ligand_ames": (
        "On the upstream Ames mutagenicity task, ADMET-AI v2.0.1 reports ROC AUC 0.882 and average precision 0.896.",
        "The model ranked assay-positive compounds above negatives fairly often in this dataset. A higher score "
        "can prioritize bacterial reverse-mutation testing. Individual risk needs assay evidence.",
    ),
    "ligand_microsomal_clearance": (
        "On the upstream AstraZeneca human microsomal clearance task, ADMET-AI v2.0.1 reports R² 0.277 and "
        "mean absolute error 26.74 µL/min/mg.",
        "The model explains a limited share of the variation. If predicted differences between compounds are "
        "near that error scale, measure microsomal stability before choosing a lead.",
    ),
    "ligand_dili": (
        "On the upstream DILI classification task, ADMET-AI v2.0.1 reports ROC AUC 0.881 and average precision 0.878.",
        "Use the relative score to prioritize liver-safety follow-up for systemically exposed leads. The "
        "benchmark gives no individual patient's probability of liver injury.",
    ),
}.items():
    _METHODS[_key] = Evidence(
        "published_benchmark", _summary, _interpretation, _ADMET_RELEASE,
        basis="Upstream holdout tests, averaged across model replicates",
    )

# These are computations with an observable reference, or readings from the original
# assay. Their error is defined by that reference, measurement or experimental setting.
def _group(keys: str, status: str, basis: str, summary: str, interpretation: str) -> None:
    evidence = Evidence(status, summary, interpretation, basis=basis)
    for key in keys.split():
        _METHODS[key] = evidence


_group(
    "octant", "source_evidence", "Measured CYP3A4 inhibition for an exact compound match",
    "Octant matches the full InChIKey and summarizes CYP3A4 inhibition as the median IC50 in nM "
    "after converting the source pIC50 readings. The number of readings gives no estimate of their spread.",
    "Use a matching IC50 to choose concentrations for a CYP3A4 inhibition assay. Whether the enzyme metabolizes "
    "the compound and how rapidly the compound clears require separate measurements.",
)
_group(
    "chembl_panel", "source_evidence", "ChEMBL measurements grouped by assay and endpoint",
    "Medians summarize determinate nM values within the same ChEMBL assay and endpoint. "
    "Bounded readings are counted separately. Precision depends on the original assay.",
    "Check the activity records and repeat the relevant assay when testing a liability. A missing determinate "
    "value leaves that target's activity unresolved.",
)
_group(
    "bindingdb_exact", "source_evidence", "BindingDB measurements for the full InChIKey",
    "Determinate nM values from matching records are summarized by endpoint, curation and available study source. "
    "Bounded readings are excluded from each median.",
    "Check the target preparation and endpoint before choosing a follow-up concentration. Keep IC50 and binding "
    "affinity readings separate when comparing observations across studies.",
)
_group(
    "hpa", "source_evidence", "Reference plasma abundance",
    "The recorded concentration describes the target protein in plasma. Variation across donors or assays "
    "must be checked in the original source.",
    "Use the concentration to plan a soluble-target assay in plasma. Tissue binding needs separate evidence.",
)
_group(
    "plasma_fold", "source_evidence", "Disease versus control comparison",
    "This summarizes the magnitude of a target's plasma abundance difference. An increase and a decrease "
    "can have the same fold-change magnitude.",
    "Inspect the original direction, comparator and concentration before choosing an experimental exposure.",
)
_group(
    "string_network uniref ensembl_paralog cath_s95 foldseek_search mmseqs", "source_evidence",
    "Reference database record or match",
    "Check the matched reference entry, release and coverage. Database evidence for a match has no common "
    "experimental accuracy across queries.",
    "Review the source record before using a match to choose a biological comparator for experimental work.",
)
_group(
    "posebusters stereochemistry ccd_geometry clashes bond_length interface_geometry prolif "
    "geometric_fingerprint contact_surface", "reference_dependent", "Coordinate-based geometry",
    "The result depends on the supplied coordinates, chemical assignment and geometric definitions. "
    "Compare with a matched experimental structure to assess this model.",
    "Inspect flagged atoms or contacts in the coordinates before using them to explain an assay result.",
)
_group(
    "dockq lddt fnat ca_rmsd", "reference_dependent", "Comparison with a reference structure",
    "This measures agreement with a supplied reference. Its interpretation depends on the chain and atom "
    "correspondence, reference quality and coverage.",
    "Use disagreement to locate regions for inspection. Resolve correspondence and coverage before comparing models.",
)
_group(
    "openfe alchemlyb pmx", "reference_dependent", "Free-energy calculation",
    "Use the uncertainty and convergence evidence for the specific free-energy calculation. "
    "Experimental error estimates require matched measured transformations.",
    "If the calculated difference is comparable to its uncertainty, seek further sampling or a matched "
    "experimental measurement before choosing between compounds.",
)
_group(
    "openmm", "reference_dependent", "Single-structure implicit-solvent potential energies",
    "The reported difference subtracts receptor and ligand potential energies from the complex energy in kcal/mol. "
    "In local repeat runs, independent preparation of the same coordinates changed the sign.",
    "Check how atoms and hydrogens were prepared and compare the component energies to diagnose a discrepancy. "
    "Determining binding affinity requires an independent measurement.",
)
_group(
    "dynamics", "reference_dependent", "Coordinate variation across supplied ensemble frames",
    "The recorded fluctuations, structural spread or radius of gyration depend on the supplied frames and "
    "atom selection. A trajectory statistic alone gives no estimate of experimental error.",
    "Inspect which regions contribute to an apparent change and whether it persists across independent samples "
    "before choosing an experimental readout of flexibility.",
)
_group(
    "ligand_composition protein_composition sequence_diversity", "reference_dependent", "Calculation from the input",
    "This is a count or composition calculation from the identified input. Completeness, chemical identity "
    "and modifications determine agreement with the sample.",
    "Use it as an identity or diversity check before choosing samples for an experimental panel.",
)
_group(
    "gravy protein_charge instability aggrescan aggrescan3d", "reference_dependent", "Physicochemical descriptor",
    "This descriptor uses the method's scale. Agreement with a stability, charge, solubility or aggregation "
    "measurement requires a matched experimental endpoint and conditions.",
    "Use an unusual descriptor to choose a confirmatory assay under the intended handling or formulation conditions.",
)
_group(
    "bindingdb_chemprop bindingdb_forest", "not_verified", "Model-specific validation unavailable",
    "A held-out activity prediction error for this configured model, target and chemical series has not been verified.",
    "Measure positive and negative controls alongside new compounds before using the predictions to rank leads.",
)
_group(
    "anarcii antibody_chemistry", "reference_dependent", "Antibody domain assignment",
    "Verify domain assignment and numbering against a curated reference. Any motif's experimental effect "
    "depends on the antibody and assay conditions.",
    "Use the result to locate residues for analytical review and confirm any proposed liability experimentally.",
)
_group(
    "bindingdb_fingerprint", "reference_dependent", "Activity measured on a structural analogue",
    "The similarity score describes the molecular fingerprints. Activity in a matched analogue requires "
    "its own measurement to quantify transfer error.",
    "A similar structure can suggest a comparator. Measure activity for the nominated compound on the same target.",
)
_group(
    "gtopdb", "source_evidence", "Reference ligand pharmacology",
    "The lookup can return exact and structurally similar ligands. Check which molecule was assayed. "
    "A neighbour's measured activity does not quantify the candidate's potency.",
    "Use an analogue's measured off-target activity to nominate a counter-assay for the candidate.",
)
_group(
    "foldseek_clustering", "reference_dependent", "Clustering within the input set",
    "Cluster membership depends on the input structures, similarity threshold and coverage settings.",
    "Use cluster representatives to diversify an experimental panel and inspect borderline assignments.",
)
_group(
    "psea", "reference_dependent", "C-alpha trace assignment",
    "This assigns secondary structure to a C-alpha trace. Coordinate resolution and missing residues affect "
    "agreement with an experimental structure assignment.",
    "Inspect short or poorly resolved segments before interpreting a difference in helix or sheet content.",
)
_group(
    "deepsp deepviscosity boltz_affinity gearnet", "not_verified", "Model-specific validation unavailable",
    "A benchmark for the producing checkpoint, endpoint and validation population on the saved observation "
    "has not been verified.",
    "Check the producing model and validate its readout on matched references before using it to select a candidate.",
)
