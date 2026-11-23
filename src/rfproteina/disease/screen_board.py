"""The closed page's screen, one tab per check.

The sentences in ``format_screen_lines`` stay available as the technical
record. This view is what a person reads: the verdict, which side of the scale
is favorable, the stored number drawn on that scale, and -- for a check with no
result -- the file or flag that would give it one.

Every chart is drawn from numbers the screen already stored. Nothing here
measures anything, and a chart is never drawn from a number that is absent: an
axis with no measurement draws its scale and says so, because an empty gauge
that looks like a full one reads as a result.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from html import escape

from rfproteina.disease.chemical import MOTIF_NAMES
from rfproteina.disease.coevolution import (
    COEVOLUTION_FILE_LIMIT,
    COEVOLUTION_ROW_LIMIT,
    COEVOLUTION_SENTENCES,
    HUMAN_TOXICITY_SCOPE,
)
from rfproteina.disease.combinations import HIT_NM, PANEL_NM_SOURCE_DEFAULT, panel_cut, panel_cut_source
from rfproteina.disease.developability import ORACLE_FILE_LIMIT, ORACLE_ROW_LIMIT
from rfproteina.disease.liability import (
    _DENSITY_METHOD,
    _DR_MEDIAN_MAX,
    _MHC_I_IC50,
    _MHC_I_RANK,
    _MHC_II_IC50,
    _MHC_II_RANK,
    SUPPLIED_METHOD,
)
from rfproteina.disease.mcp_api import _AXIS_GROUPS, _CURATED_SOURCE
from rfproteina.disease.modality import HALF_LIFE_PHRASES, TRIAGE_MOTIFS, component_label
from rfproteina.disease.oligo import (
    _CONSENSUS_METHOD,
    _HUMAN_HEXAMER,
    _MURINE_MOTIF,
    SEED_WIDTH,
    TRANSCRIPTOME_FILE_LIMIT,
    TRANSCRIPTOME_MIN_BASES,
)
from rfproteina.disease.screen import _GRAVY_MAX, _INSTABILITY_MAX, _METRICS_FILE_LIMIT, one_token

#: Every axis that can have a tab, in the order the tabs appear: the sequence
#: axes, paralog, membrane beside its siblings, the plasma axes, then the two
#: oligo axes. ``_shown_axes`` keeps only the ones a screen actually recorded.
_CATALOG: tuple[tuple[str, str, str], ...] = (
    ("immunogenicity", "sequence_liability", "Immunogenicity"),
    ("proteome_offtarget", "sequence_liability", "Proteome off-target"),
    ("mhc_i", "sequence_liability", "MHC class I"),
    ("mhc_ii", "sequence_liability", "MHC class II"),
    ("ada", "sequence_liability", "ADA"),
    ("paralog", "target_liability", "Paralog"),
    ("membrane", "sequence_liability", "Membrane"),
    ("plasma_abundance", "target_liability", "Plasma abundance"),
    ("plasma_hub", "target_liability", "Plasma hub"),
    ("plasma_fold", "target_liability", "Plasma fold"),
    ("off_target_hybridization", "nucleic_liability", "Off-target hybridization"),
    ("innate_immune", "nucleic_liability", "Innate immune"),
)
#: The tabs a screen that recorded no states falls back to, which is the
#: protein set. ``membrane`` is a peptide axis and the two oligo axes are
#: nucleic, so neither belongs in a default standing in for "the screen did not
#: say": a protein page drawing a Membrane tab would be answering a question it
#: was never asked.
_DEFAULT_AXES = frozenset(
    {
        "immunogenicity",
        "proteome_offtarget",
        "mhc_i",
        "mhc_ii",
        "ada",
        "paralog",
        "plasma_abundance",
        "plasma_hub",
        "plasma_fold",
    }
)
_AXES: tuple[tuple[str, str, str], ...] = tuple(item for item in _CATALOG if item[0] in _DEFAULT_AXES)
_TOX_AXES: tuple[tuple[str, str, str], ...] = (
    ("herg", "toxicology", "hERG"),
    ("nav1_5", "toxicology", "Nav1.5"),
    ("cyp3a4", "toxicology", "CYP3A4"),
    ("cyp2d6", "toxicology", "CYP2D6"),
    ("cyp2c9", "toxicology", "CYP2C9"),
    ("cyp1a2", "toxicology", "CYP1A2"),
    ("cyp2c19", "toxicology", "CYP2C19"),
    ("pgp", "toxicology", "P-gp"),
    ("bcrp", "toxicology", "BCRP"),
    ("bsep", "toxicology", "BSEP"),
)
_TOX_NAMES = frozenset(item[0] for item in _TOX_AXES)
_TOX_ABOUT = "A concentration in nM from the toxicology table in the output folder."
#: The two published TLR9 hexamers ``oligo`` reads, taken from that module so a
#: motif it stores on a record is the one this page is willing to print.
_CPG_MOTIFS = frozenset({_MURINE_MOTIF, _HUMAN_HEXAMER})
_STATES = frozenset({"hit", "clear", "untested"})
#: The three words the panels already use for a state, so the status grid and
#: the tabs name a state exactly as its own tab does.
_STATE_WORD = {"clear": "Clear", "hit": "Hit", "untested": "Not decided"}
_BLOCKERS = {
    "duplicate_candidate_id": "Several candidate rows have the same identifier.",
    "foldseek_result_invalid": "A Foldseek result has an invalid identifier or score.",
    "independently_prepared_subsystems": "The separately prepared subsystems have no shared energy reference.",
    "invalid_candidate_table_width": "A candidate row has a different number of fields from its input format.",
    "invalid_molecular_structure": "The chemistry reader could not interpret the candidate structure.",
    "invalid_provider_result": "The predictor returned an unusable value for this compound.",
    "invalid_qupkake_manifest": "The QupKake input manifest could not be read or validated.",
    "invalid_sygma_manifest": "The SyGMa input manifest could not be read or validated.",
    "invalid_sygma_scenario": "The requested SyGMa phase counts are outside the supported scenario.",
    "ligand_binder_kind_required": "This screen requires a run whose candidate is a small-molecule ligand.",
    "ligand_input_missing": "The screen could not find the selected ligand input.",
    "missing_candidate_id": "A candidate row has no identifier to link it to a result.",
    "native_tnp_failed": "The Therapeutic Nanobody Profiler failed while analysing the sample.",
    "no_attributable_site_prediction": "QupKake could not assign a result to a specific site on this compound.",
    "no_candidates": "The selected input contains no candidate rows.",
    "no_completed_compounds": "No compound returned a completed prediction.",
    "no_completed_parents": "SyGMa could not finish a metabolite search for any parent compound.",
    "no_usable_candidates": "Every candidate row has an input error.",
    "protein_sol_ambiguous": "Several Protein-Sol predictions could be assigned to this sample.",
    "protein_sol_fasta_invalid": "The FASTA accompanying the Protein-Sol report could not be read or validated.",
    "protein_sol_input_unverified": "The Protein-Sol input sequence could not be matched to this sample.",
    "protein_sol_report_invalid": "The Protein-Sol report has invalid prediction headers or rows.",
    "protein_sol_report_unreadable": "The Protein-Sol prediction file could not be read within the file limit.",
    "protein_sol_sample_missing": "The Protein-Sol report has no prediction matching this sample.",
    "protein_sol_samples_missing": "The run contains no sample sequences to match to the Protein-Sol report.",
    "protein_sol_sequence_unresolved": "Protein-Sol requires at least 21 standard amino acids in the sample.",
    "provider_timeout": "The predictor exceeded the time allowed for this calculation.",
    "provider_worker_failed": "The predictor process ended before it returned a usable result.",
    "qupkake_response_identity_mismatch": "The QupKake result identifies a different input compound.",
    "qupkake_runtime_unavailable": "The QupKake runtime could not be loaded.",
    "qupkake_timeout": "QupKake exceeded the time allowed for this compound.",
    "qupkake_worker_failed": "The QupKake process failed while analysing this compound.",
    "qupkake_worker_invalid_report": "The QupKake process returned a report the reader could not validate.",
    "rdkit_unavailable": "The chemistry reader could not load RDKit.",
    "sdf_input_unreadable": "The selected SDF structure input could not be read.",
    "sequence_shorter_than_window": "The sequence is shorter than the AGGRESCAN scoring window.",
    "site_head_incomplete": "A QupKake site predictor failed before it could establish site coverage.",
    "structure_reader_unavailable": "The protein structure reader could not be loaded.",
    "sygma_dependency_unavailable": "The metabolite worker could not load a required chemistry dependency.",
    "sygma_invalid_worker_report": "The SyGMa worker returned a report the reader could not validate.",
    "sygma_parent_timeout": "SyGMa exceeded the time allowed for this parent compound.",
    "sygma_worker_failed": "The SyGMa process failed while analysing this parent compound.",
    "tnp_provider_unavailable": "The Therapeutic Nanobody Profiler runtime could not be loaded.",
    "tnp_runtime_io_error": "The Therapeutic Nanobody Profiler could not read or write its working files.",
    "unsupported_residue": "AGGRESCAN has no residue score for at least one sequence position.",
    "no_pass_cut": "A number was measured. No pass cut applies to it, so the check stays undecided.",
    "cpg_not_consensus": (
        "The design holds CpG and none of it is one of the two consensus hexamers. "
        "Zero hexamers is not an absence of innate-immune signal: the other sensors are not scored here."
    ),
    "predictor_not_configured": "No predictor table was found for this check.",
    "predictor_unreadable": "The predictor table could not be read.",
    "peptide_column_absent": "The table carries ranks with no peptide named.",
    "allele_column_absent": "The class II table has no allele column.",
    "allele_panel_incomplete": "The seven-allele DR panel is not fully covered.",
    "no_background_for_allele": (
        "The fitted class II model carries this allele with no background scores, so a peptide cannot be "
        "placed on a percentile rank and the missing alleles were not scored."
    ),
    "rank_absent": "The panel is covered, but no row carries a rank.",
    "rank_scale_unknown": "The table does not declare a consensus method.",
    "score_absent": "The table holds no usable number.",
    "table_not_from_this_design": "The peptides in the table are not windows of this design.",
    "proteome_not_configured": "No proteome file was given, so shared 9-mers were not counted.",
    "proteome_unreadable": "The proteome file could not be read.",
    "proteome_empty": "The proteome file holds no records.",
    "proteome_too_large": "The proteome file is larger than this check will read.",
    "proteome_too_small": "The proteome file is too small to support a clear.",
    "proteome_not_protein": "The proteome file reads as DNA or RNA, so it was not used.",
    "reference_not_configured": "No reference table was found, so there is no bar to compare against.",
    "reference_unreadable": "The reference table could not be read.",
    "reference_length_unknown": "The reference scan was filtered, so its protein length is unknown.",
    "reference_method_mismatch": "The design and the reference came from different tools.",
    "reference_method_unknown": "Exactly one of the two tables declares a method.",
    "reference_allele_mismatch": "The design and the reference used different alleles.",
    "reference_allele_unknown": "One of the two tables declares no allele.",
    "reference_width_mismatch": "The design and the reference scored different peptide lengths.",
    "reference_length_inconsistent": "One reference scan states a length its own rows contradict.",
    "reference_not_tolerated_protein": "The reference names no protein from the tolerated set.",
    "reference_rows_exceed_protein": "The reference holds more rows than one protein of that length can.",
    "reference_density_implausible": "A reference protein has more than a quarter of its windows binding.",
    "reference_ic50_not_nm": "An affinity column in the reference states no unit, so there is no bar.",
    "reference_too_short_for_width": "Every tolerated reference protein is shorter than the widths scored.",
    "cell_unparsed": "A rank, affinity or score cell is neither a number nor blank.",
    "reference_cell_unparsed": "A cell in the reference table is neither a number nor blank.",
    "network_not_permitted": "Target lookup was not requested, so this number was not fetched.",
    "threshold_not_set": "The number is shown. No cut was named, so the check stays undecided.",
    "fold_not_configured": "No plasma fold table was found in the output folder.",
    "fold_not_reported": "The plasma fold table has no usable number.",
    "fold_unreadable": "The plasma fold table could not be read.",
    "fold_target_not_in_table": "The plasma fold table labels its rows and holds none for this target.",
    "fold_too_large": "The plasma fold table is larger than this screen reads.",
    "source_not_named": "The fold table does not name the study it came from.",
    "toxicology_not_configured": "No toxicology table was found in the output folder.",
    "toxicology_unreadable": "The toxicology table could not be read.",
    "toxicology_not_reported": "The toxicology table has no usable nM reading for this check.",
    "chain_not_named": "Nothing in the run named the chain of interest, so the design was not measured.",
    "coordinates_absent": "The generated file holds a sequence and no coordinates, so no interface was measured.",
    "coordinates_unusable": "The structure loaded but its coordinates could not be used.",
    "binder_chain_absent": "The chain of interest is not in the generated structure.",
    "interface_failed": "The interface measurement did not complete for this design.",
    "structure_unreadable": "The generated structure file could not be opened or decoded.",
    "metrics_failed": "GRAVY and the instability index did not return a finite number for this chain.",
    "not_a_polypeptide": "GRAVY and the instability index are defined over amino acids. This design is neither.",
    "chain_shorter_than_dipeptide": "The instability index sums over dipeptides, and this chain is one residue.",
    "solubility_not_configured": "No solubility table was found in the output folder.",
    "solubility_unreadable": "The solubility table could not be read.",
    "solubility_not_reported": "The solubility table has no row naming a score, a method and a study.",
    "solubility_residue_profile": "The supplied table contains per-residue scores.",
    "solubility_truncated": "The solubility table has more rows than this screen will read.",
    "solubility_too_large": "The solubility file is larger than this screen will read.",
    "reading_not_decisive": "The cell bounds the reading without placing it on one side of the cut.",
    "reading_not_nm": "The cell is not a non-negative nM reading.",
    "coevolution_not_configured": "No co-evolution table was found in the output folder.",
    "coevolution_unreadable": "The co-evolution table could not be read.",
    "coevolution_missing_columns": "The study table contains none of the fields for this comparison.",
    "coevolution_not_reported": "The co-evolution table has no usable row for this check.",
    "coevolution_unparsed": "A co-evolution cell could not be read as yes or no, high or low, or toxic or non-toxic.",
    "coevolution_not_classified": "Co-evolution is not named high or low.",
    "toxicity_not_classified": "Other-species toxicity is not named toxic or non-toxic.",
    "same_shape_absent": "No row names the same shape in another species.",
    "coevolution_disagreement": "Rows with the same shape name different notes.",
    "coevolution_truncated": "The co-evolution table has more data rows than this screen will judge.",
    "coevolution_too_large": "The co-evolution file is larger than this screen will read.",
    "not_requested": "The refolding oracle was not asked about this design.",
    "oracle_not_configured": "The refolding oracle is not configured.",
    "no_manifest": "No refold requests have been written for this run.",
    "manifest_unreadable": "The refold request manifest could not be read.",
    "manifest_not_a_manifest": "The refold request manifest is not in the shape this reads.",
    "scan_incomplete": "The scan does not cover the whole chain, so a clear cannot be claimed.",
    "unscored_windows": "Some window of the chain was never scored, so a zero cannot be claimed.",
    "ic50_not_nm": "An affinity column states no unit, so not every presented peptide was read.",
    "width_without_cutoff": "Every peptide scored is of a length no published binding cut covers.",
    "calis_mask_unpublished_for_allele": "This molecule's anchors are not the ones the published score masks.",
    "no_presented_ligand": "The class I table presented nothing.",
    "no_nine_mer_ligand": "The table presented ligands, but none of length 9.",
    "presented_not_in_design": "The presented peptides are not in this chain.",
    "cluster_singleton_not_evidence": "The target is alone in its UniRef50 cluster. That is not a clear.",
    "gene_not_in_compara": "Ensembl Compara holds no gene tree for the target, so its paralogs could not be measured.",
    "gene_symbol_not_found": "No gene symbol was found for the accession, so Compara could not be asked for paralogs.",
    "cluster_not_found": "The paralog lookup found no cluster for the target.",
    "members_not_listed": "The cluster was found, but its members were not listed.",
    "members_truncated": "The member list came back truncated, so it cannot be counted.",
    "target_not_accession": "The target is a gene symbol and no condition spec resolved it.",
    "target_accession_ambiguous": "Two condition specs in this campaign name different accessions.",
    "target_not_named": "The run names no target at all.",
    "target_not_reported": "The atlas holds no entry for this target.",
    "concentration_not_reported": "The atlas holds no blood concentration for this target.",
    "concentration_invalid": "The atlas row contains a negative blood concentration.",
    "partners_not_listed": "The partner lookup returned no list.",
    "compara_identities_incomplete": "Some returned Ensembl records lack a usable percent identity.",
    "degree_censored": "The partner count came back censored.",
    "lookup_unavailable": "The lookup service did not answer.",
    "sequence_absent": "No designed sequence was available to measure.",
    "structure_unparsed": "A loop in the generated file could not be read as rows, so nothing was taken from it.",
    "designs_absent": "No generated structure file was found for per-design measurements.",
    "evaluation_structure_table_only": "The offline evaluator recorded inputs. This screen read only their table.",
    "designs_compressed": "The packed structure files could not be opened as text CIFs.",
    "metrics_not_configured": "No per-design metrics table was found in the output folder.",
    "metrics_aggregation_required": "The published metrics do not account for the current job tables.",
    "metrics_publication_pending": "A metrics writer has an unfinished publication in this directory.",
    "metrics_provenance_mismatch": "The published metrics differ from the CSV recorded at aggregation.",
    "metrics_provenance_unreadable": "The recorded aggregation sources could not be verified.",
    "not_evaluated": "No declared pass criterion was evaluated for this run.",
    "metrics_incomplete": "Some applicable designs lack a valid result or a required measurement.",
    "ligand_identity_mismatch": "The panel contains measurements for a different compound.",
    "ligand_identity_unverified": "The compound identity could not be verified against the panel.",
    "panel_opinions_incomplete": "At least one panel method has an unresolved reading.",
    "criteria_not_found": "The declared pass criteria could not be read.",
    "criteria_not_declared": "The pass criteria file declares none.",
    "thresholds_not_resolved": "This criterion borrows its bounds in a form this page does not resolve.",
    "metrics_not_reported": "The metrics table holds no rows.",
    "metrics_unreadable": "The metrics table could not be read.",
    "metrics_too_large": "The metrics file is larger than this screen will read.",
    "metrics_columns_repeated": "The metrics table names a column this screen reads more than once.",
    "metrics_row_too_long": "A metrics row carries more fields than its own header.",
    "metrics_row_too_short": "A metrics row stops short of its own header.",
    "metrics_row_absent": "No row of the metrics table names this design.",
    "metrics_row_ambiguous": "Two metrics rows name this design and disagree about it.",
    "metrics_row_shared": "The same metrics evidence matches multiple structure files.",
    "chain_disagreement": "The metrics table names more than one chain of interest.",
    "chain_shorter_than_window": "The chain is shorter than the 19-residue window, so it was not assessed.",
    "hydropathy_unknown": "The chain carries a letter with no hydropathy value.",
    "transcriptome_not_configured": "No reference was found for the off-target seed count.",
    "transcriptome_unreadable": "The reference could not be read.",
    "transcriptome_empty": "The reference holds no sequence.",
    "transcriptome_too_large": "The reference is larger than this screen will read.",
    "transcriptome_too_small": "The reference is too small to support a clear.",
    "transcriptome_not_nucleic": "The reference does not read as DNA or RNA.",
    "sequence_not_nucleic": "The designed chain is not a DNA or RNA sequence.",
    "chain_shorter_than_seed": "The design is shorter than one 15-base seed.",
    "design_windows_unscannable": "Every seed window of this design spans an ambiguity code, so nothing was searched.",
    "cpg_free_not_evidence": "The design carries no CpG at all. That is not a clear for the other innate sensors.",
    # The screening-extras checks: interface quality, construct synthesis, ligand
    # alerts, biosecurity, novelty, off-target, antibody developability and
    # humanness. See docs/screening-implementation-handoff.md.
    "empty_command": "A screening tool was asked to run with no command, so nothing was measured.",
    "interface_quality_failed": "The interface quality measurement did not complete for this design.",
    "module_unavailable": "The module this check runs could not be imported, so the check did not run.",
    "synthesis_failed": "The construct synthesis screen did not return a reading for this sequence.",
    "no_commec_verdict": "The biosecurity screen ran and its flag table names no recommendation for this design.",
    "no_numbered_chains": "No chain was IMGT-numbered, so there are no antibody regions to scan.",
    "oracle_unreadable": "The developability oracle's table could not be read.",
    "oracle_too_large": "The developability oracle's table is larger than this screen will read.",
    "oracle_columns_absent": (
        "The table defines none of the columns this oracle reports, so it is not that oracle's output."
    ),
    "oracle_truncated": "The developability oracle's table holds more rows than this screen will judge.",
    "oracle_not_reported": "The developability oracle's table holds no row carrying a usable score.",
    "no_rows_query_may_not_have_been_read": (
        "Foldseek returned nothing for this design. At this E-value a chain that was searched comes back with "
        "something, so the file was most likely never read, and nothing similar cannot be told from nothing searched."
    ),
    "no_scorable_peptides": "The domain yields no 9-mer of standard residues, so no human peptide was looked up.",
    "no_paired_chains_supplied": "Antibody-likeness is scored over a heavy and light pair, and none was supplied.",
}
#: What to do about a blocker. The wording follows the "What to do" column of
#: docs/reference/screen-inputs.md rather than new prose, so the panel and that
#: reference cannot drift apart.
_ACTIONS = {
    "duplicate_candidate_id": "Give each candidate row a unique identifier and run the screen again.",
    "foldseek_result_invalid": "Check the alignment table against the requested Foldseek output columns.",
    "independently_prepared_subsystems": "Check the preparation provenance before interpreting energy differences.",
    "invalid_candidate_table_width": "Check the row against the header and the delimiter of the selected format.",
    "invalid_molecular_structure": "Inspect the candidate's chemical structure in the source file.",
    "invalid_provider_result": "Inspect the predictor output for the affected compound.",
    "invalid_qupkake_manifest": "Check the selected QupKake input and its tautomer setting.",
    "invalid_sygma_manifest": "Check the selected ligand input and the phase counts in the SyGMa manifest.",
    "invalid_sygma_scenario": "Use supported integer counts for the phase I and phase II steps.",
    "ligand_binder_kind_required": "Select the run containing the small-molecule candidates.",
    "ligand_input_missing": "Check that the selected candidate file exists in the run.",
    "missing_candidate_id": "Assign a unique identifier to the row before retrying.",
    "native_tnp_failed": "Check the sample structure and the profiler's diagnostic output.",
    "no_attributable_site_prediction": "Inspect the compound's site mapping and any rejected site rows.",
    "no_candidates": "Check that the selected input file contains the intended compounds.",
    "no_completed_compounds": "Read the individual compound errors before retrying.",
    "no_completed_parents": "Inspect the individual parent errors before retrying the metabolite screen.",
    "no_usable_candidates": "Correct the input errors listed against the individual rows.",
    "protein_sol_ambiguous": "Keep one traceable prediction for the sample in the report.",
    "protein_sol_fasta_invalid": "Provide the FASTA used to run Protein-Sol, with unique headers.",
    "protein_sol_input_unverified": "Check the original predictor FASTA against the sample identifier and sequence.",
    "protein_sol_report_invalid": "Provide the native seq_prediction.txt with its prediction header.",
    "protein_sol_report_unreadable": "Check the encoding, size and read access of seq_prediction.txt.",
    "protein_sol_sample_missing": "Check that the predictor input and output both contain this sample.",
    "protein_sol_samples_missing": "Select a finished run with identifiable sample sequences.",
    "provider_timeout": "Inspect the predictor log for the affected input before retrying.",
    "provider_worker_failed": "Check the predictor runtime and its diagnostic output.",
    "qupkake_response_identity_mismatch": "Check the compound identifier against the worker input and repeat the run.",
    "qupkake_runtime_unavailable": "Check the configured QupKake environment and its dependencies.",
    "qupkake_timeout": "Inspect the QupKake log for this compound before retrying.",
    "qupkake_worker_failed": "Inspect the QupKake worker diagnostics for this compound.",
    "qupkake_worker_invalid_report": "Check that the QupKake worker and report reader use compatible versions.",
    "rdkit_unavailable": "Check the RDKit installation in the environment that runs this screen.",
    "sdf_input_unreadable": "Check the SDF location, read access and record format.",
    "site_head_incomplete": "Inspect the acid and base site predictor statuses for this compound.",
    "structure_reader_unavailable": "Check the structural analysis dependencies in the screening environment.",
    "sygma_dependency_unavailable": "Check the worker's chemistry dependencies and the pinned SyGMa rule files.",
    "sygma_invalid_worker_report": "Check the SyGMa worker version and inspect the failed parent report.",
    "sygma_parent_timeout": "Inspect the parent structure and its requested phase counts before retrying.",
    "sygma_worker_failed": "Inspect the SyGMa worker diagnostics for the affected parent.",
    "tnp_provider_unavailable": "Check the configured profiler runtime and its dependencies.",
    "tnp_runtime_io_error": "Check read and write access to the profiler's working directory.",
    "predictor_not_configured": "Add mhc_i.tsv or mhc_ii.tsv to the output folder.",
    "no_manifest": "Run the refold export first, or point --refold-dir at where it wrote.",
    "manifest_unreadable": "Check the file is readable JSON, or export the requests again.",
    "manifest_not_a_manifest": "Export the requests again so the manifest is rewritten.",
    "predictor_unreadable": "Check the encoding and the delimiter.",
    "peptide_column_absent": "Add a peptide column.",
    "allele_column_absent": "Add an allele column.",
    "allele_panel_incomplete": "Score the alleles named below.",
    "no_background_for_allele": (
        "Re-fit the class II model so every allele carries a background, or set RFD4_CLASS_II_MODEL to one that does."
    ),
    "no_rows_query_may_not_have_been_read": (
        "Check the design file parses: a chain-only CIF without occupancy and B-factor columns is dropped silently."
    ),
    "rank_absent": "The median is a percentile, so an affinity cannot stand in for it. Add a rank column.",
    "rank_scale_unknown": "Re-run with the IEDB consensus method, or add a method column.",
    "score_absent": "Check the rank or affinity column.",
    "table_not_from_this_design": "The table is another run's scan. Re-score this chain.",
    "proteome_not_configured": "Pass --proteome, or set HUMAN_PROTEOME.",
    "proteome_unreadable": "Check the file is a protein FASTA of reasonable size.",
    "proteome_empty": "Check the file is a protein FASTA of reasonable size.",
    "proteome_too_large": "Check the file is a protein FASTA of reasonable size.",
    "proteome_too_small": (
        "Use a whole protein proteome. A UniProtKB export without the Sequence column reads as one record."
    ),
    "proteome_not_protein": "This is cds.all.fa where pep.all.fa was meant.",
    "reference_not_configured": "Score the three reference proteins.",
    "reference_unreadable": "Check the encoding and the delimiter.",
    "reference_length_unknown": "Re-export the complete scan, or add a real length.",
    "reference_method_mismatch": "Score both with the same tool.",
    "reference_method_unknown": "Add a method column to both, or to neither.",
    "reference_allele_mismatch": "Score both over the same allele set.",
    "reference_allele_unknown": "Add an allele column to both, over the same alleles.",
    "reference_width_mismatch": "Score both over the same widths.",
    "reference_length_inconsistent": "Correct the stated length.",
    "reference_not_tolerated_protein": "Score albumin, IgG1 or ubiquitin.",
    "reference_rows_exceed_protein": "Keep each protein's own accession and length.",
    "reference_density_implausible": "Re-export the scan.",
    "reference_ic50_not_nm": "Name the unit in the reference's affinity column heading, in nM.",
    "reference_too_short_for_width": "Score the reference at a width its proteins are long enough for.",
    "cell_unparsed": "Most often a comma decimal from a German or French Excel export. Re-save with a dot.",
    "reference_cell_unparsed": "Most often a comma decimal from a German or French Excel export. Re-save with a dot.",
    "scan_incomplete": "Export the complete scan: every window, at every width scored.",
    "unscored_windows": "Score the windows the predictor declined, or accept that absence cannot be claimed.",
    "ic50_not_nm": "Name the unit in the affinity column heading, in nM, or drop the column.",
    "width_without_cutoff": "Score 8 to 11 residues for class I, or 13 to 25 for class II.",
    "calis_mask_unpublished_for_allele": "Report this allele on its own. No published score covers its anchors.",
    "no_nine_mer_ligand": "Include 9-mers in the scan.",
    "presented_not_in_design": "Re-score this chain.",
    "cluster_singleton_not_evidence": (
        "Name --paralog-identity-max to measure the closest paralog's identity, or curate "
        "the call into axis_evidence.json."
    ),
    "gene_not_in_compara": "Curate the call into axis_evidence.json.",
    "gene_symbol_not_found": "Retry, or curate the call into axis_evidence.json.",
    "cluster_not_found": "Retry, or curate the call into axis_evidence.json.",
    "members_not_listed": "Retry, or curate the call into axis_evidence.json.",
    "members_truncated": "Retry, or curate the call into axis_evidence.json.",
    "target_not_accession": "This check needs an accession, not a gene symbol.",
    "target_accession_ambiguous": "Name the target explicitly.",
    "target_not_named": "Supply generation.json, or a job file that names the target.",
    "partners_not_listed": "Retry.",
    "degree_censored": "Retry.",
    "network_not_permitted": "Pass --lookup-targets.",
    "lookup_unavailable": "Retry.",
    "fold_not_configured": "Add plasma_fold.tsv to the output folder.",
    "fold_not_reported": "Add a fold or log2_fold_change column with a number in it.",
    "fold_unreadable": "Check the encoding and the delimiter.",
    "fold_target_not_in_table": "Add a row for this target, or remove the label column.",
    "fold_too_large": "Keep the table under 1 MiB and 256 data rows.",
    "source_not_named": "Add a source column. A fold change without a provenance is a rumour.",
    "sequence_absent": "Check the run produced designs.",
    "structure_unparsed": "Check the file for a trailing comment or an unclosed quote on a loop row.",
    "designs_absent": "Check the run produced designs.",
    "evaluation_structure_table_only": "Use the evaluator table for each input's recorded measurements.",
    "designs_compressed": "Check the archive and its decoder, or export a readable text CIF alongside it.",
    "toxicology_not_configured": "Add toxicology.tsv to the output folder.",
    "toxicology_unreadable": "Check the encoding and the delimiter.",
    "toxicology_not_reported": "Add an nM column for this panel protein.",
    "chain_not_named": "Check the metrics file names a chain_of_interest.",
    "coordinates_absent": "Expected for an inverse-folding run, which writes sequences.",
    "coordinates_unusable": "Check the run wrote finite coordinates for this design.",
    "binder_chain_absent": "Check the chain_of_interest matches a chain in the design.",
    "interface_failed": "Check the design and the target are both in the generated file.",
    "structure_unreadable": "Check the run wrote a readable CIF for this design.",
    "metrics_failed": "Check the designed chain is a standard protein sequence.",
    "not_a_polypeptide": "Expected for a nucleic or small-molecule design. Read its own triage report instead.",
    "chain_shorter_than_dipeptide": "Expected for a one-residue design. The GRAVY beside it still applies.",
    "solubility_not_configured": "Add solubility.tsv to the output folder.",
    "solubility_unreadable": "Check the encoding and the delimiter.",
    "solubility_not_reported": "Give each row a score, the method that produced it, and a study.",
    "solubility_residue_profile": "Supply a separate whole-structure summary with its method and source.",
    "solubility_truncated": "Keep one row per method. The limit is 32 rows.",
    "solubility_too_large": "Keep the file within 1 MiB.",
    "reading_not_decisive": "Record the measured concentration, or a bound that falls clear of the cut.",
    "reading_not_nm": "Write the reading in nM.",
    "coevolution_not_configured": "Add paralog_coevolution.tsv to the output folder.",
    "coevolution_unreadable": "Check the encoding and the delimiter.",
    "coevolution_not_reported": "Name same shape, co-evolution, and other-species toxicity.",
    "coevolution_unparsed": "Use yes or no, high or low, and toxic or non-toxic.",
    "coevolution_not_classified": "Write high or low in the co-evolution column.",
    "toxicity_not_classified": "Write toxic or non-toxic.",
    "same_shape_absent": "The four transfer notes apply when the same shape is present.",
    "coevolution_disagreement": "Keep one class in the table.",
    "coevolution_truncated": f"Keep one row per species. The limit is {COEVOLUTION_ROW_LIMIT} data rows.",
    "coevolution_too_large": f"Keep the file within {COEVOLUTION_FILE_LIMIT // (1024 * 1024)} MiB.",
    "metrics_not_configured": "Evaluate the run to write inference_metrics/per_example_metrics.csv.",
    "metrics_aggregation_required": "Aggregate the metrics directory and screen again.",
    "metrics_publication_pending": "Allow the writer to finish or rerun aggregation in place to recover it.",
    "metrics_provenance_mismatch": (
        "Restore the verified table and receipt or aggregate the original jobs in a fresh directory."
    ),
    "metrics_provenance_unreadable": "Check the receipt and source files. Use verified copies to rebuild the report.",
    "not_evaluated": "Run the refolding oracle to evaluate them.",
    "metrics_incomplete": "Check the per-design metrics and the oracle results for the affected designs.",
    "ligand_identity_mismatch": "Supply measurements for the ligand named in this run.",
    "ligand_identity_unverified": "Check the run's ligand SMILES and the InChIKey on each measurement.",
    "panel_opinions_incomplete": "Inspect each method's readings and blocker.",
    "criteria_not_found": "Expected at configs/metrics/pass_criteria/all_pass_criteria.yaml.",
    "metrics_not_reported": "Check the run wrote a row per design.",
    "metrics_unreadable": "Check the encoding and the delimiter.",
    "metrics_too_large": f"Keep the file within {_METRICS_FILE_LIMIT // (1024 * 1024)} MiB.",
    "metrics_columns_repeated": "Keep one copy of each column and screen again.",
    "metrics_row_too_long": "Export the table again without the stray separator.",
    "metrics_row_too_short": "Export the table again with every row the width of its header.",
    "metrics_row_absent": "Check the table names this design in example_id.",
    "metrics_row_ambiguous": "Screen one job's table at a time, or drop the row that is out of date.",
    "metrics_row_shared": "Export a separate row with the structure path for each design.",
    "chain_disagreement": "Screen one target at a time, or name the chain in every row.",
    "chain_shorter_than_window": "A span cannot be claimed or ruled out below 19 residues.",
    "hydropathy_unknown": "Check the designed chain is a standard protein sequence.",
    "transcriptome_not_configured": "Add transcriptome.fa to the output folder, or set HUMAN_TRANSCRIPTOME.",
    "transcriptome_unreadable": "Check the file is a nucleotide FASTA of reasonable size.",
    "transcriptome_empty": "Check the file is a nucleotide FASTA of reasonable size.",
    "transcriptome_too_large": f"Keep the file within {TRANSCRIPTOME_FILE_LIMIT // (1024 * 1024)} MiB.",
    "transcriptome_too_small": f"Use a reference of at least {TRANSCRIPTOME_MIN_BASES} bases.",
    "transcriptome_not_nucleic": "This is a protein FASTA; supply the transcript sequences.",
    "sequence_not_nucleic": "Check the run wrote a nucleic chain for this design.",
    "chain_shorter_than_seed": f"A seed match cannot be claimed below {SEED_WIDTH} bases.",
    "design_windows_unscannable": "Resolve the degenerate positions, or accept that this design cannot be searched.",
    "cpg_free_not_evidence": "Curate the call into axis_evidence.json.",
    # The screening-extras checks, in the same order as _BLOCKERS above.
    "empty_command": "No input causes this. Report it as a defect in the screen.",
    "interface_quality_failed": "Check the design and the target are both in the generated file.",
    "module_unavailable": (
        "Reinstall this project's environment. The module ships in-tree, so what is missing is a dependency of it."
    ),
    "synthesis_failed": "Check the designed chain is a plain DNA or RNA sequence.",
    "no_commec_verdict": "Re-run the screen so every record gets a row, or screen this design on its own.",
    "no_numbered_chains": "Give --heavy, --light, or both, and check ANARCII accepted them.",
    "oracle_unreadable": "Check the encoding and the delimiter.",
    "oracle_too_large": f"Keep the file within {ORACLE_FILE_LIMIT // (1024 * 1024)} MiB.",
    "oracle_columns_absent": "Supply the oracle's own CSV, with its property columns named in the header.",
    "oracle_truncated": f"Keep one row per design. The limit is {ORACLE_ROW_LIMIT} rows.",
    "oracle_not_reported": "Check the property columns carry numbers.",
    "no_scorable_peptides": "Supply a variable domain of at least nine standard residues.",
    "no_paired_chains_supplied": "Pass --likeness with both --heavy and --light.",
}
#: Where one axis needs a narrower answer than the blocker alone can give --
#: which table, or which of the three --plasma cuts.
_AXIS_ACTIONS = {
    ("innate_immune", "cpg_not_consensus"): "Curate the call into axis_evidence.json.",
    ("immunogenicity", "predictor_not_configured"): "Add immunogenicity.tsv to the output folder.",
    ("immunogenicity", "no_pass_cut"): "Add mhc_i.tsv to the output folder, so the presented 9-mers are named.",
    ("mhc_i", "predictor_not_configured"): "Add mhc_i.tsv to the output folder.",
    ("mhc_ii", "predictor_not_configured"): "Add mhc_ii.tsv to the output folder.",
    ("ada", "predictor_not_configured"): "Add mhc_ii.tsv to the output folder.",
    ("plasma_abundance", "threshold_not_set"): "Pass --plasma-abundance-max.",
    ("plasma_hub", "threshold_not_set"): "Pass --plasma-hub-max.",
    ("plasma_fold", "threshold_not_set"): "Pass --plasma-fold-max.",
}
#: Where a blocker's shared sentence would be wrong on one axis, so that axis
#: states its own. Empty, and kept because the shape is worth having: the one
#: entry it held was a stopgap for ``innate_immune`` reporting ``no_pass_cut``
#: for a design holding non-consensus CpG, where a cut DOES apply and the page
#: therefore denied what its own High line asserts. ``oligo`` now names that
#: case ``cpg_not_consensus``, which is the fact being reported, so the sentence
#: belongs in ``_BLOCKERS`` with every other one and no override is needed.
_AXIS_BLOCKERS: dict[tuple[str, str], str] = {}
_SAMPLE_CAP = 8
#: Fill, edge and ink per state. One table, so a tab, a status grid cell and a
#: panel heading cannot disagree about which colour a state is.
_FILL = {"clear": "#d7eadf", "hit": "#f8dcd7", "untested": "#e2e9ef"}
_EDGE = {"clear": "#1f7a4d", "hit": "#c0392b", "untested": "#8aa0b2"}
_INK = {"clear": "#0b5c40", "hit": "#8f2318", "untested": "#3d4a57"}
#: Background of the selected tab. The darkest tone of each state, because the
#: label text turns white on it and has to stay readable.
_TAB_ON = {"clear": "#0b6e4f", "hit": "#9c2a1b", "untested": "#3d4a57"}
#: Dot fill. A verdict's dot is a solid disc of its own edge colour. The
#: undecided dot is paler than its border, so an open question reads as
#: quieter than an answer rather than as a third verdict.
_DOT = {"clear": _EDGE["clear"], "hit": _EDGE["hit"], "untested": "#cfd8e0"}
_CUT_INK = "#8a1f11"
_BAR_INK = "#3d6f8f"
_TRACK = "#e2e9ef"
_RULE = "#b7c3ce"


def render_closing(lines: Sequence[str], screens: Sequence[Mapping[str, object]]) -> str:
    """The Panel closed section. Screens become tabs. Other sentences stay visible."""
    parts = ['<section class="screen-board-page"><h2>Panel closed</h2>']
    if not screens:
        parts.append("".join(f"<p>{_esc(line)}</p>" for line in lines))
        parts.append("</section>")
        return "".join(parts)
    technical = [line for line in lines if _screen_sentence(line)]
    other = [line for line in lines if not _screen_sentence(line)]
    parts.append(_style())
    parts.append(
        '<p class="hint">Each tab is one check. '
        "Green is clear, red is a hit, and grey is not decided. "
        "The chart places the stored number on a low-to-high scale. "
        "A number on the favorable side can still be undecided.</p>"
    )
    for index, report in enumerate(screens):
        parts.append(_board(index, report))
    if other:
        parts.append("".join(f"<p>{_esc(line)}</p>" for line in other))
    if technical:
        rows = "".join(f"<p>{_esc(line)}</p>" for line in technical)
        parts.append(f"<details><summary>Technical record</summary>{rows}</details>")
    parts.append(_script())
    parts.append("</section>")
    return "".join(parts)


def _board(index: int, report: Mapping[str, object]) -> str:
    screen = report.get("screen")
    if not isinstance(screen, dict):
        return "<p>This screen could not be drawn.</p>"
    entry = report.get("entry_id")
    chain = report.get("chain")
    title = _design_title(entry if isinstance(entry, str) else "", chain if isinstance(chain, str) else "")
    name = f"board-{index}"
    prefix = f"b{index}"
    shown = _shown_axes(screen)
    declared = _declared_axes(screen)
    tabs: list[str] = []
    panels: list[str] = []
    states: list[tuple[str, str, bool]] = []
    for axis, bucket, label in shown:
        record = _axis_record(screen, axis, bucket)
        state = _state(record)
        states.append((label, state, _is_declared(record)))
        checked = " checked" if not tabs else ""
        field = f"{name}-{axis}"
        tabs.append(
            f'<input type="radio" name="{name}" id="{field}"{checked}>'
            f'<label for="{field}" class="tab-{state}">'
            f'<span class="dot dot-{state}"></span>{_esc(label)}</label>'
        )
        panels.append(_panel(axis, label, record, prefix))
    states.extend(_untabbed_states(screen, shown, declared))
    return (
        f'<article class="axis-board"><h3>{_esc(title)}</h3>'
        f"{_summary(screen, prefix)}"
        f"{_status_grid(states)}"
        f"{''.join(tabs)}"
        f'<div class="axis-panels">{"".join(panels)}</div></article>'
    )


def _untabbed_states(
    screen: Mapping[str, object],
    shown: Sequence[tuple[str, str, str]],
    declared: Mapping[str, Mapping[str, object]],
) -> list[tuple[str, str, bool]]:
    """Checks the screen recorded that this page has no tab for.

    Named from the record's own key, because the alternative is to drop a
    check for want of a catalog entry -- and the one it drops is as likely to
    be the hit the heading is announcing as any other.
    """
    states = screen.get("states")
    if not isinstance(states, dict):
        return []
    tabbed = {axis for axis, _bucket, _label in shown}
    found: list[tuple[str, str, bool]] = []
    for axis, state in states.items():
        if not isinstance(axis, str) or not axis or axis in tabbed:
            continue
        text = axis.replace("_", " ")
        word = state if isinstance(state, str) and state in _STATES else "untested"
        found.append((text[:1].upper() + text[1:], word, axis in declared))
    return found


def _shown_axes(screen: Mapping[str, object]) -> tuple[tuple[str, str, str], ...]:
    """The checks this screen recorded, in catalog order.

    Protein states name the sequence and plasma checks. Ligand states name the
    safety panel and the plasma checks, so those tabs are the panel readings.
    A screen with no ``states`` keeps the protein set.
    """
    states = screen.get("states")
    if not isinstance(states, dict) or not states:
        return _AXES
    shown = tuple(item for item in (*_TOX_AXES, *_CATALOG) if item[0] in states)
    return shown or _AXES


def _status_grid(states: Sequence[tuple[str, str, bool]]) -> str:
    """Every check and its state, before any tab is opened.

    Each tab answers one question. This answers the question a reader opens the
    page with -- which of them came back with anything -- without opening every
    one.

    It lists every check the screen recorded, including one this page has no
    tab for. A peptide run can hit an axis the tab catalog does not carry, and
    the grid then showed nine chips reading Clear or Not decided under a
    heading saying a check had found a problem, with nothing naming it.

    A declared state keeps its colour, because a person's clear is still a
    clear, and says so in the word, because it is not a measured one.
    """
    if not states:
        return ""
    cells = "".join(
        f'<li class="chip chip-{state}"><span>{_esc(label)}</span>'
        f"<strong>{_esc(_state_word(state, declared))}</strong></li>"
        for label, state, declared in states
    )
    return f'<ul class="status-grid">{cells}</ul>'


def _summary(screen: Mapping[str, object], prefix: str) -> str:
    sequence = _dict(screen.get("sequence_developability"))
    interface = _dict(screen.get("designed_interface"))
    samples = _sample_rows(screen)
    gravy = _as_float(sequence.get("gravy"))
    instability = _as_float(sequence.get("instability_index"))
    contact = _as_float(interface.get("contact_ms"))
    clash = _as_float(interface.get("clash_fraction"))
    parts = [f"<p><strong>{_esc(_call_text(screen.get('call')))}</strong></p>"]
    # Beside the call, not inside a tab. "Every check in this bar is clear"
    # read as a measured clear when one axis in it was a person's declaration
    # over a measurement that said hit, and the only place saying so was a
    # paragraph below the chart of the tab that had to be opened to see it.
    told = _declared_sentence(screen)
    if told:
        parts.append(f"<p><strong>{_esc(told)}</strong></p>")
    gauges = [
        block
        for block in (
            _gauge_block(
                "GRAVY",
                gravy,
                low="At or below 0.1 passes this half. Lower is the favorable side.",
                high="Above 0.1 fails this half.",
                vmin=-1.0,
                vmax=1.0,
                threshold=_GRAVY_MAX,
                better="low",
                slug=f"{prefix}-gravy",
            ),
            _gauge_block(
                "Instability",
                instability,
                # ExPASy ProtParam: "A protein whose instability index is
                # smaller than 40 is predicted as stable." Exactly 40 is not.
                low=f"Below {_num(_INSTABILITY_MAX)} passes this half. Lower is more stable.",
                high=f"At or above {_num(_INSTABILITY_MAX)} fails this half.",
                vmin=0.0,
                vmax=max(80.0, instability or 0.0),
                threshold=_INSTABILITY_MAX,
                better="low",
                slug=f"{prefix}-instability",
            ),
        )
        if block
    ]
    if gauges:
        parts.append("<h4>Design checks</h4>")
        parts.append('<div class="metric-row">')
        parts.extend(gauges)
        parts.append("</div>")
    sentence = _sequence_sentence(sequence.get("sequence_pass"), gravy, instability)
    # "Sequence check was not recorded" on its own left the reader to guess
    # among four causes. The record now names the one that applied.
    reason = " ".join(_blocker_lines(sequence.get("sequence_blocker")))
    if reason and sequence.get("sequence_pass") not in {True, False}:
        sentence = f"{sentence} {reason}"
    if sentence:
        parts.append(f"<p>{_esc(sentence)}</p>")
    parts.extend(_chemical_lines(_dict(screen.get("chemical_liability"))))
    parts.extend(_modality_lines(_dict(screen.get("modality_screen"))))
    if len(samples) > 1:
        charts = _sample_charts(samples)
        if charts:
            if gauges:
                parts.append("<p>The gauges are the first design. The charts are every design's own numbers.</p>")
            parts.append(charts)
    checks = [
        # First, because it is about which run this page is describing rather
        # than about any one check on it.
        *_job_mismatch_lines(screen.get("job_mismatch")),
        _flag_sentence("Checks before a refolding oracle", screen.get("pre_oracle_pass")),
        _ppi_sentence(screen),
        # Why the two flags above have nothing in them, when the table is why.
        *_blocker_lines(screen.get("metrics_blocker")),
        *_refold_lines(_dict(screen.get("refold"))),
        *_blocker_lines(screen.get("chain_blocker")),
        _contact_sentence(contact),
        _clash_sentence(clash),
        # Why the two above are empty, when they are.
        *_blocker_lines(interface.get("interface_blocker")),
        *_row_blocker_lines(screen.get("samples")),
    ]
    checks.extend(_solubility_lines(_dict(screen.get("solubility"))))
    checks.extend(_diversity_lines(_dict(screen.get("design_diversity"))))
    listed = "".join(f"<li>{_esc(row)}</li>" for row in checks if row)
    if listed:
        parts.append(f'<ul class="plain-metrics">{listed}</ul>')
    return "".join(parts)


def _job_mismatch_lines(record: object) -> list[str]:
    """What the job file recorded, when the screen was run against something else.

    The screen kept this and no reader ever saw it, so a run screened under the
    wrong kind or against the wrong target showed a verdict with nothing saying
    the two records describe different runs.
    """
    if not isinstance(record, Mapping) or not record:
        return []
    named = ", ".join(
        f"{one_token(key)} {one_token(value)}" for key, value in sorted(record.items()) if isinstance(key, str)
    )
    if not named:
        return []
    return [f"The job file for this run records {named}. This screen was run against other values."]


def _solubility_lines(record: Mapping[str, object]) -> list[str]:
    """Supplied solubility scores, or why there are none.

    Each displayed score retains its tool, reported unit and study.
    """
    if not record:
        return []
    found: list[str] = []
    scores = record.get("scores")
    if isinstance(scores, list):
        for item in scores:
            if not isinstance(item, Mapping):
                continue
            method = item.get("method")
            score = item.get("score")
            source = item.get("source")
            if (not isinstance(method, str) or not method.strip() or not isinstance(score, int | float)
                    or isinstance(score, bool) or not math.isfinite(score)):
                continue
            value = _num(float(score))
            unit = item.get("unit")
            if isinstance(unit, str) and unit.strip():
                value += f" {unit}"
            line = f"Solubility {value} by {method}"
            if isinstance(source, str) and source:
                line = f"{line}, from {source}"
            found.append(f"{line}. No cut applies to it.")
    dropped = record.get("incomplete_rows")
    if isinstance(dropped, int) and not isinstance(dropped, bool) and dropped:
        found.append(f"{dropped} solubility row(s) named no method or no study and were not read.")
    found.extend(_blocker_lines(record.get("solubility_blocker")))
    return found


def _diversity_lines(record: Mapping[str, object]) -> list[str]:
    """How many designs this run really produced."""
    if not record:
        return []
    designs = record.get("designs")
    distinct = record.get("distinct")
    if not isinstance(designs, int) or isinstance(designs, bool):
        return []
    found = [f"Designs {designs}, distinct sequences {distinct}."]
    groups = record.get("groups")
    if isinstance(groups, int) and not isinstance(groups, bool):
        largest = record.get("largest_group")
        line = f"Separate designs {groups}"
        if isinstance(largest, int) and not isinstance(largest, bool) and largest > 1:
            line = f"{line}, the largest group holding {largest}"
        found.append(f"{line}.")
    if record.get("pairwise") == "not_compared":
        found.append("No two designs share a length, so none could be compared.")
    return found


def _row_blocker_lines(samples: object) -> list[str]:
    """Designs that were not measured against exactly one metrics row.

    Counted rather than listed: the reason is the same for every design it
    applies to, and a run of two hundred would otherwise print it two
    hundred times. First-seen order, so the line does not move between two
    renders of the same screen.
    """
    if not isinstance(samples, list):
        return []
    counts: dict[str, int] = {}
    for sample in samples:
        if not isinstance(sample, Mapping):
            continue
        reason = sample.get("row_blocker")
        if isinstance(reason, str) and reason:
            counts[reason] = counts.get(reason, 0) + 1
    found: list[str] = []
    for reason, number in counts.items():
        for line in _blocker_lines(reason):
            found.append(f"{number} design(s): {line}")
    return found


def _curated_lines(record: Mapping[str, object]) -> list[str]:
    """What a curated call was curated on, or that nothing was stated.

    Some axes have no measured clear at all, so a curated one is the only
    way they ever clear and a reader has nothing else to go on. Saying the
    basis was not stated is part of that: it is a fact about the record.
    """
    found = record.get("curated")
    if isinstance(found, Mapping):
        lines: list[str] = []
        reason = found.get("reason")
        study = found.get("source")
        if isinstance(reason, str) and reason:
            lines.append(f"The stated reason is {reason}")
        if isinstance(study, str) and study:
            lines.append(f"The stated source is {study}")
        return lines
    if found == "basis_not_stated":
        return ["No reason or source was recorded with it."]
    return []


def _refold_lines(record: Mapping[str, object]) -> list[str]:
    """Each declared pass criterion this run was evaluated against.

    Only the ones that were evaluated. A run with no refolding oracle has
    eleven criteria with no column, and printing eleven lines saying so
    would bury the checks that do have an answer under the ones that cannot.
    The run-level reason is printed once instead.
    """
    if not record:
        return []
    found: list[str] = []
    criteria = record.get("criteria")
    if isinstance(criteria, list):
        for item in criteria:
            if not isinstance(item, Mapping):
                continue
            name = item.get("name")
            state = item.get("state")
            if isinstance(name, str) and state == "incomplete":
                measured, expected = item.get("rows_measured"), item.get("rows_expected")
                found.append(f"{name} has complete results for {measured} of {expected} applicable rows.")
                continue
            if not isinstance(name, str) or state not in {"pass", "fail"}:
                continue
            word = "passes" if state == "pass" else "fails"
            line = f"{name.replace('_', ' ')} {word} the criterion in the evaluation table"
            missed = item.get("failed_thresholds")
            if isinstance(missed, list) and missed:
                # Which bound, not just that one was missed: a ptm floor
                # and an aligned RMSD ask for different work.
                details = ", ".join(str(piece) for piece in missed)
                line = f"{line}. The checkout's default bounds were missed for {details}"
            found.append(f"{line}.")
    found.extend(_blocker_lines(record.get("refold_blocker")))
    return found


def _blocker_lines(blocker: object) -> list[str]:
    """One sentence naming a blocker and what to do about it."""
    if not isinstance(blocker, str) or not blocker:
        return []
    said = _BLOCKERS.get(blocker, blocker.replace("_", " "))
    action = _ACTIONS.get(blocker, "")
    return [f"{said} {action}".strip()]


def _panel(axis: str, label: str, record: Mapping[str, object], prefix: str) -> str:
    state = _state(record)
    low, high, about = _guidance(axis, record)
    measurement = _as_float(record.get("measurement"))
    parts = [
        '<article class="axis-panel">',
        f"<h3>{_esc(label)}</h3>",
        f'<p class="axis-status status-{state}">{_esc(_status(state, declared=_is_declared(record)))}</p>',
        f"<p>{_esc(about)}</p>",
        *_todo(axis, record),
        _axis_chart(axis, record, measurement, slug=f"{prefix}-{axis}"),
        '<div class="guide">',
        f"<p><strong>Low. </strong>{_esc(low)}</p>",
        f"<p><strong>High. </strong>{_esc(high)}</p>",
        "</div>",
        *_detail_lines(axis, record),
        "</article>",
    ]
    return "".join(parts)


def _todo(axis: str, record: Mapping[str, object]) -> list[str]:
    """Why a check has no result, and the file or flag that would give it one.

    Above the chart, because on a screen where nothing was configured this is
    the whole content of the tab and the chart below it is an empty scale.
    """
    blocker = record.get("blocker")
    if not isinstance(blocker, str) or not blocker:
        return []
    said = _AXIS_BLOCKERS.get((axis, blocker)) or _BLOCKERS.get(blocker, blocker.replace("_", " "))
    action = _AXIS_ACTIONS.get((axis, blocker)) or _ACTIONS.get(blocker, "")
    body = f"{said} {action}" if action else said
    return [f'<p class="axis-todo">{_esc(body)}</p>']


def _tox_guide(axis: str, record: Mapping[str, object]) -> tuple[str, str, str]:
    """The three guide sentences for one panel axis, stated at that axis's own cut.

    One number used to be written out for all ten panel proteins, so the page
    could state a bar the verdict had not been taken against. The cut is per
    axis, and ``toxicology._record`` stores the one each reading was judged
    against under ``threshold`` for exactly this reason -- reading
    :data:`rfproteina.disease.combinations.PANEL_NM` live instead meant that
    after the owner moved an axis's cut, a stored hit of 4000 nM was shown
    beside a guide saying anything above 3000 nM is clear. A record written
    before the per-axis cut existed carries no threshold, and the live table is
    then the only cut there is.

    It also says where that number came from: ``program_convention`` means a
    convention this program applies, with no publication behind it, and
    anything else is named as the record names it.
    """
    cut = _as_float(record.get("threshold"))
    if cut is None:
        cut = panel_cut(axis)
    source = record.get("threshold_source")
    if not isinstance(source, str) or not source:
        source = panel_cut_source(axis)
    if source == PANEL_NM_SOURCE_DEFAULT:
        basis = f"The cut is {cut:g} nM ({cut / 1000:g} uM), a program convention."
    else:
        basis = f"The cut is {cut:g} nM ({cut / 1000:g} uM); its source is recorded as {source}."
    return (
        f"At or below {cut:g} nM is a hit on this panel protein.",
        f"Above {cut:g} nM is clear on this panel protein.",
        f"{_TOX_ABOUT} {basis}",
    )


def _guidance(axis: str, record: Mapping[str, object]) -> tuple[str, str, str]:
    """Low text, high text, and what the check is.

    Low is the left end of the chart. High is the right end. For ADA the left
    end is the concerning side, because a lower rank is stronger binding. The
    record is read where one axis states its own cut rather than a constant.
    """
    if axis in _TOX_NAMES:
        return _tox_guide(axis, record)
    rank_i = _num(_MHC_I_RANK)
    ic50_i = _num(_MHC_I_IC50)
    rank_ii = _num(_MHC_II_RANK)
    ic50_ii = _num(_MHC_II_IC50)
    ada = _num(_DR_MEDIAN_MAX)
    guides = {
        "immunogenicity": (
            "At or below 0 is the favorable side. Every scored 9-mer at or below 0 can clear when the scan applies.",
            (
                "Above 0 is the concerning side. A presented 9-mer above 0 is a hit. "
                "A whole-chain score above 0 is reported and stays undecided "
                "until a class I table names the presented 9-mers."
            ),
            "Calis 2013 class I score of a 9-residue window. The stored number is the highest window.",
        ),
        "proteome_offtarget": (
            "Zero shared 9-mers is the only measured clear. One shared 9-mer is a hit.",
            "A higher count is more exact 9-mer identity with the proteome file.",
            (
                "Exact 9-mer identity between the design and a protein FASTA. "
                "A 9-mer that differs by one residue is not counted."
            ),
        ),
        "mhc_i": (
            (
                "For a protein, fewer binders per 100 windows than the tolerated proteins is the clear side. "
                f"A peptide is clear when nothing binds at rank {rank_i} or IC50 {ic50_i} nM."
            ),
            "More binders than the tolerated proteins is a hit. A strong rank or a tight IC50 is a predicted binder.",
            (
                "Class I binding from the table in the output folder, "
                "compared with tolerated proteins when that reference is present."
            ),
        ),
        "mhc_ii": (
            (
                "For a protein, fewer binders per 100 residues than the tolerated proteins is the clear side. "
                f"A peptide is clear when nothing binds at rank {rank_ii} or IC50 {ic50_ii} nM."
            ),
            "More binders than the tolerated proteins is a hit.",
            (
                "Class II binding from the table in the output folder, "
                "compared with tolerated proteins when that reference is present."
            ),
        ),
        "ada": (
            (
                f"At or below {ada} is the concerning side. "
                "A median rank that low is predicted to bind across the DR panel."
            ),
            f"A median rank above {ada} is the weaker-binding side.",
            "Paul 2015 median rank on a seven-allele DR panel. A lower rank is stronger binding.",
        ),
        "paralog": (
            (
                "Fewer other human proteins in the same UniRef50 cluster is the quieter side. "
                "A count of zero is not a clear."
            ),
            "One or more other human proteins in the cluster is a hit.",
            "Whether another reviewed human protein shares the target's UniRef50 cluster.",
        ),
        "membrane": (
            "No 19-residue window above 1.6 mean hydropathy is the soluble side.",
            "One such window is a span, and one span is a hit.",
            (
                "Kyte-Doolittle 1982 mean hydropathy over a 19-residue window. "
                "A chain shorter than the window is not assessed."
            ),
        ),
        "off_target_hybridization": (
            "No exact 15-base seed of the design occurs in the supplied reference.",
            "One seed occurring in the reference is a hit.",
            (
                "Exact 15-base seeds of the design and its reverse complement, counted "
                "against transcriptome.fa in the output folder. The clear is a clear "
                "against that file, whose size and digest are recorded with it."
            ),
        ),
        # Both published rules, because ``oligo.innate_immune_record`` reads
        # both and GTCGTT is not an instance of the murine formula -- its
        # second base is a pyrimidine. Naming only the murine one presented a
        # hit driven by the human hexamer as a purine-purine-CG-pyrimidine-
        # pyrimidine match the reader could not find in the chain. The motif
        # names and the method sentence come from that module so the page and
        # the record cannot drift apart.
        "innate_immune": (
            (
                f"Neither hexamer in the design: no {_MURINE_MOTIF} "
                f"(purine-purine-CG-pyrimidine-pyrimidine) and no {_HUMAN_HEXAMER}."
            ),
            (
                f"One hexamer of either rule is a hit. {_HUMAN_HEXAMER} is not an "
                f"instance of {_MURINE_MOTIF}, so a hit can be on either one alone."
            ),
            (
                f"{_CONSENSUS_METHOD}. A CpG-free design cannot present either, "
                "which is not a clear for the other innate sensors."
            ),
        ),
        "plasma_abundance": (
            "A lower blood concentration is less antigen already in circulation.",
            ("A higher concentration is more antigen in the blood. " "It becomes a hit only against a cut you name."),
            "Blood concentration of the target, in picograms per litre.",
        ),
        "plasma_hub": (
            "Fewer high-confidence physical partners is the quieter side.",
            (
                "A higher partner count means the target connects more proteins. "
                "It becomes a hit only against a cut you name."
            ),
            "How many physical interaction partners the target has at high confidence.",
        ),
        "plasma_fold": (
            "A magnitude of one means the groups have the same level.",
            "Larger magnitudes indicate a greater change between groups.",
            "The chart compares the linear fold magnitude with the configured cutoff.",
        ),
    }
    return guides[axis]


def _axis_chart(axis: str, record: Mapping[str, object], measurement: float | None, *, slug: str) -> str:
    """The chart for one check, drawn from what the record actually holds."""
    threshold = _as_float(record.get("threshold"))
    if axis in {"mhc_i", "mhc_ii"} and measurement is not None:
        references = _reference_rows(record)
        if references:
            # The bar is the one the record carries, which ``liability`` writes
            # only when the comparison was made. A scan that covered part of
            # the chain keeps the per-protein densities and no bar -- "under the
            # highest of three" was never established, and the error of a
            # partial scan runs toward clear. Taking the highest density as the
            # cut anyway drew that line and put the design on the clear side of
            # it under "The dashed line is the cut".
            bar = _as_float(record.get("reference_density"))
            rows = [("This design", measurement, True), *((name, value, False) for name, value in references)]
            if bar is None:
                beside = _bars(rows, title="Design beside the tolerated proteins", slug=slug)
                if not beside:
                    return ""
                return f"{beside}<p>No cut is drawn. This design was not compared with them.</p>"
            return _marked(_bars(rows, cut=bar, title="Design compared with tolerated proteins", slug=slug), bar)
    better = "low"
    vmin = 0.0
    vmax = max(1.0, measurement or 0.0, threshold or 0.0)
    cuts: tuple[float, ...] = () if threshold is None else (threshold,)
    if axis == "ada":
        better = "high"
        vmax = 100.0
        cuts = (_DR_MEDIAN_MAX,)
    elif axis == "immunogenicity":
        vmin = -1.0
        vmax = 1.0
        cuts = (0.0,)
    elif axis == "proteome_offtarget":
        vmax = max(1.0, measurement or 0.0)
        cuts = (0.0,)
    elif axis == "paralog":
        vmax = max(1.0, measurement or 0.0)
        cuts = ()
    elif axis == "plasma_fold":
        # Historical raw measurements did not specify their scale.
        measurement = _as_float(record.get("linear_fold_magnitude"))
        if measurement is None:
            return ""
        vmin = min(1.0, threshold if threshold is not None else 1.0)
        vmax = max(2.0, measurement, threshold or 0.0)
        better = "low" if threshold is not None else "none"
    elif axis in _TOX_NAMES:
        better = "high"
        cut = HIT_NM if threshold is None else threshold
        vmin = 0.0
        vmax = max(cut * 2.0, measurement or 0.0)
        cuts = (cut,)
    if measurement is None:
        return ""
    chart = _gauge(measurement, vmin=vmin, vmax=vmax, cuts=cuts, better=better, slug=slug)
    return _marked(chart, cuts[0] if cuts else None, count=len(cuts))


def _reference_rows(record: Mapping[str, object]) -> list[tuple[str, float]]:
    """Each tolerated protein's density, or the one number the bar was taken from.

    The per-protein densities are what the comparison was actually made over,
    and a design sitting under the highest of three says more than a design
    sitting under an unnamed bar.
    """
    densities = record.get("reference_densities")
    rows: list[tuple[str, float]] = []
    if isinstance(densities, dict):
        for name, value in densities.items():
            number = _as_float(value)
            if isinstance(name, str) and name and number is not None:
                rows.append((name, number))
    if rows:
        rows.sort(key=lambda item: item[0])
        return rows
    single = _as_float(record.get("reference_density"))
    return [] if single is None else [("Tolerated proteins", single)]


def _detail_lines(axis: str, record: Mapping[str, object]) -> list[str]:
    lines: list[str] = []
    opinions = record.get("opinions")
    if isinstance(opinions, list):
        for entry in opinions:
            if isinstance(entry, Mapping):
                label = " / ".join(str(entry[key]) for key in ("method", "assay_source", "endpoint") if entry.get(key))
                value = entry.get("measurement")
                reading = f" ({entry.get('qualifier', '')}{value} nM)" if value is not None else ""
                lines.append(f"{label} reported {entry.get('state')}{reading}.")
    if axis == "immunogenicity" and record.get("presentation") == "not_configured":
        lines.append(
            "This score scans the whole chain. It stays undecided until a class I table names the presented 9-mers."
        )
    windows = record.get("positive_windows")
    if isinstance(windows, int) and not isinstance(windows, bool):
        lines.append(f"{windows} nine-residue windows scored above 0.")
    if record.get("method") == SUPPLIED_METHOD:
        lines.append("The number came from a table you supplied. A supplied score does not clear or hit.")
    if record.get("method") == _DENSITY_METHOD and _as_float(record.get("measurement")) is not None:
        lines.append("The number is binding peptides per 100 windows.")
    binding = record.get("binding_peptides")
    if isinstance(binding, int) and not isinstance(binding, bool):
        lines.append(f"{binding} peptides in the table were counted as binding.")
    unscored = record.get("unscored_peptides")
    if isinstance(unscored, int) and not isinstance(unscored, bool) and unscored:
        lines.append(f"{unscored} windows were named but never scored.")
    alleles = record.get("alleles")
    if isinstance(alleles, int) and not isinstance(alleles, bool):
        lines.append(f"The DR panel is {alleles} alleles.")
    kmers = record.get("proteome_kmers")
    if isinstance(kmers, int) and not isinstance(kmers, bool):
        lines.append(f"The proteome file held {kmers} distinct 9-mers.")
    cluster = record.get("cluster")
    if isinstance(cluster, str) and cluster:
        lines.append(f"The UniRef50 cluster is {cluster}.")
    # The study behind a plasma_fold number. "source" is where the STATE came
    # from, so the provenance this axis refuses to be measured without has a key
    # of its own; a screen written before that split carries neither.
    study = record.get("fold_source")
    if isinstance(study, str) and study:
        lines.append(f"The fold table names {study}.")
    fold_column = record.get("fold_column")
    raw = _as_float(record.get("measurement"))
    if fold_column in ("fold", "fold_change", "log2_fold_change") and raw is not None:
        lines.append(f"The supplied {fold_column} value is {_num(raw)}.")
    source = record.get("source")
    threshold = _as_float(record.get("threshold"))
    if threshold is not None:
        lines.append(f"The named cut is {_num(threshold)}.")
    if source == "axis_evidence":
        lines.append("A curated call is recorded for this check.")
        lines.extend(_curated_lines(record))
        measured = record.get("measured_state")
        if isinstance(measured, str) and measured and measured != record.get("state"):
            lines.append(f"The measurement said {measured.replace('_', ' ')}.")
    lines.extend(_named_ids("Alleles not covered", record.get("missing_alleles")))
    if axis == "proteome_offtarget":
        lines.extend(_named_ids("Designs with a shared 9-mer", record.get("failed")))
    # A supplied table is one scan, and a run holds many designs. The
    # state above was recorded for all of them; this is the list it was
    # actually taken from.
    lines.extend(_named_ids("Designs this table does not cover", record.get("uncovered_examples")))
    if axis == "paralog":
        lines.extend(_named_ids("Other proteins in the cluster", record.get("paralogs")))
        lines.extend(_coevolution_detail(record))
    if axis == "immunogenicity":
        lines.extend(_named_ids("Designs with a positive window", record.get("positive_examples")))
    if axis == "innate_immune":
        lines.extend(_motif_site_lines(record))
    if axis == "off_target_hybridization":
        lines.extend(_seed_reference_lines(record))
    units = record.get("units")
    if units == "pg/L":
        lines.append("The number is picograms per litre.")
    elif isinstance(units, str) and units:
        lines.append(f"The stored unit is {units}.")
    readings = record.get("n_readings")
    if isinstance(readings, int) and not isinstance(readings, bool) and readings > 1:
        lines.append(f"The reading is the lowest of {readings} cells.")
    lines.extend(_named_ids("Columns that did not parse", record.get("unparsed_columns")))
    return [f"<p>{_esc(line)}</p>" for line in lines]


def _motif_site_lines(record: Mapping[str, object]) -> list[str]:
    """Which published rule fired on the innate-immune axis, and where.

    ``oligo.innate_immune_record`` stores the motif that matched at each
    position rather than merging the two rules, because the murine consensus
    does not match the human hexamer at all. Nothing on this page read it, so
    every hit was presented under the murine formula and the reader had no way
    to find the hexamer in the chain. The CpG count comes along because it is
    the number the undecided case turns on and the measurement is not it.
    """
    found: list[str] = []
    sites = record.get("motif_sites")
    if isinstance(sites, list):
        named: list[str] = []
        for site in sites:
            if not isinstance(site, Mapping):
                continue
            position = site.get("position")
            motif = site.get("motif")
            if isinstance(position, bool) or not isinstance(position, int) or position < 1:
                continue
            if not isinstance(motif, str) or motif not in _CPG_MOTIFS:
                continue
            named.append(f"base {position}, {motif}")
        if named:
            text = "Motif hexamers: " + "; ".join(named[:_SAMPLE_CAP])
            if len(named) > _SAMPLE_CAP:
                text = f"{text}, and {len(named) - _SAMPLE_CAP} more"
            found.append(f"{text}.")
    count = record.get("cpg_count")
    if isinstance(count, int) and not isinstance(count, bool) and count >= 0:
        found.append(f"The design holds {count} CpG.")
    return found


def _seed_reference_lines(record: Mapping[str, object]) -> list[str]:
    """The reference file a seed count was taken against.

    The guide for this axis says the reference's size and digest are recorded
    with the verdict. They were -- in ``screen.json``, which this page's reader
    never opens -- so "a clear against that file" named no file at all and
    could not be checked against anything.
    """
    name = record.get("reference_file")
    if not isinstance(name, str) or not name:
        return []
    line = f"The reference is {name}"
    bases = record.get("reference_bases")
    if isinstance(bases, int) and not isinstance(bases, bool) and bases >= 0:
        line = f"{line}, {bases} bases"
    digest = record.get("reference_digest")
    if isinstance(digest, str) and digest:
        line = f"{line}, digest {digest}"
    return [f"{line}."]


def _marked(chart: str, threshold: float | None, *, count: int = 1) -> str:
    if threshold is None:
        return chart
    if count > 1:
        # Plasma fold is judged on magnitude, so its cut is drawn on both sides.
        return chart + "<p>The dashed lines are the cut.</p>"
    return chart + "<p>The dashed line is the cut.</p>"


def _gauge_block(
    title: str,
    value: float | None,
    *,
    low: str,
    high: str,
    vmin: float,
    vmax: float,
    threshold: float | None,
    better: str,
    slug: str,
) -> str:
    if value is None:
        return ""
    shown = _num(value)
    cuts: tuple[float, ...] = () if threshold is None else (threshold,)
    return (
        "<div>"
        f"<p><strong>{_esc(title)}</strong> {_esc(shown)}</p>"
        f"{_gauge(value, vmin=vmin, vmax=vmax, cuts=cuts, better=better, slug=slug)}"
        f"{_marked('', threshold)}"
        f"<p><strong>Low. </strong>{_esc(low)}</p>"
        f"<p><strong>High. </strong>{_esc(high)}</p>"
        "</div>"
    )


def _gauge(
    value: float | None,
    *,
    vmin: float,
    vmax: float,
    cuts: Sequence[float] = (),
    better: str,
    slug: str,
) -> str:
    """One number on a low-to-high scale, with the favorable side shaded.

    A cut splits the track into a favorable band and a concerning one, which is
    what a reader is looking for. Without a cut there is no boundary to draw, so
    a direction is a gradient and no direction is a plain track: shading a band
    there would invent a line the screen never measured against.
    """
    lo, hi = _bounds(value, vmin, vmax, cuts)
    width = 400.0
    left = 30.0
    track = f'<rect x="{left:.0f}" y="40" width="{width:.0f}" height="14" rx="7"'
    zones = _zones(lo, hi, cuts, better, left, width, slug)
    if zones:
        body = zones
    elif better in {"low", "high"}:
        grad = _esc(f"scale-{slug}")
        start, end = ("#1f7a4d", "#c47b12") if better == "low" else ("#c47b12", "#1f7a4d")
        body = (
            f'<defs><linearGradient id="{grad}"><stop offset="0" stop-color="{start}"/>'
            f'<stop offset="1" stop-color="{end}"/></linearGradient></defs>'
            f'{track} fill="url(#{grad})"/>'
        )
    else:
        body = f'{track} fill="{_TRACK}"/>'
    rule = f'{track} fill="none" stroke="{_RULE}"/>'
    marks: list[str] = []
    for cut in cuts:
        cx = left + _frac(cut, lo, hi) * width
        marks.append(
            f'<line x1="{cx:.1f}" y1="34" x2="{cx:.1f}" y2="60" '
            f'stroke="{_CUT_INK}" stroke-width="1.5" stroke-dasharray="3 2"/>'
            f'<text x="{cx:.1f}" y="30" text-anchor="{_anchor(cx, left, width)}" font-size="10" '
            f'fill="{_CUT_INK}">{_esc(_num(cut))}</text>'
        )
    vx = left + _frac(value, lo, hi) * width
    marks.append(
        f'<line x1="{vx:.1f}" y1="32" x2="{vx:.1f}" y2="62" stroke="#17202a" stroke-width="2"/>'
        f'<circle cx="{vx:.1f}" cy="47" r="4" fill="#17202a"/>'
        f'<text x="{vx:.1f}" y="16" text-anchor="{_anchor(vx, left, width)}" font-size="11" '
        f'font-weight="700">{_esc(_num(value))}</text>'
    )
    # The ends are printed only when a number sits between them. Where nothing
    # was measured the range is this drawing's own padding rather than anything
    # the screen holds, and printing it reads as a measured span of 0 to 1.
    ends = ""
    if value is not None:
        ends = (
            f'<text x="{left:.0f}" y="72" font-size="10" fill="#52616e">{_esc(_num(lo))}</text>'
            f'<text x="{left + width:.0f}" y="72" text-anchor="end" font-size="10" '
            f'fill="#52616e">{_esc(_num(hi))}</text>'
        )
    return (
        '<svg viewBox="0 0 460 88" role="img" class="scale-chart">'
        f"<title>{_esc('Low to high scale')}</title>"
        f"<desc>{_esc(_gauge_words(value, lo, hi, cuts))}</desc>"
        f"{body}{rule}{''.join(marks)}{ends}"
        f'<text x="{left:.0f}" y="85" font-size="11">Low</text>'
        f'<text x="{left + width:.0f}" y="85" text-anchor="end" font-size="11">High</text>'
        "</svg>"
    )


def _zones(
    lo: float,
    hi: float,
    cuts: Sequence[float],
    better: str,
    left: float,
    width: float,
    slug: str,
) -> str:
    """The favorable and concerning bands, clipped to the rounded track."""
    if not cuts or better not in {"low", "high", "middle"}:
        return ""
    edges = [left + _frac(cut, lo, hi) * width for cut in cuts]
    right = left + width
    if better == "middle":
        if len(edges) < 2:
            # A two-sided cut with one line drawn would shade half the scale
            # favorable on the strength of a boundary that is not there.
            return ""
        spans = [
            (left, edges[0], "hit"),
            (edges[0], edges[-1], "clear"),
            (edges[-1], right, "hit"),
        ]
    elif better == "low":
        spans = [(left, edges[0], "clear"), (edges[0], right, "hit")]
    else:
        spans = [(left, edges[0], "hit"), (edges[0], right, "clear")]
    clip = _esc(f"track-{slug}")
    bands = "".join(
        f'<rect x="{start:.1f}" y="40" width="{max(0.0, end - start):.1f}" height="14" fill="{_FILL[kind]}"/>'
        for start, end, kind in spans
    )
    return (
        f'<defs><clipPath id="{clip}">'
        f'<rect x="{left:.0f}" y="40" width="{width:.0f}" height="14" rx="7"/>'
        f"</clipPath></defs>"
        f'<g clip-path="url(#{clip})">{bands}</g>'
    )


def _gauge_words(value: float | None, lo: float, hi: float, cuts: Sequence[float]) -> str:
    """The same chart in a sentence, for a reader who cannot see it."""
    if value is None:
        return ""
    said = f"The number is {_num(value)}. Scale {_num(lo)} to {_num(hi)}."
    if not cuts:
        return said
    named = ", ".join(_num(cut) for cut in cuts)
    return f"{said} The cut is {named}."


def _bars(
    rows: Sequence[tuple[str, float | None, bool]],
    *,
    cut: float | None = None,
    title: str,
    slug: str,
    highlight: str = "#17202a",
) -> str:
    """Horizontal bars with a shared scale, and the cut as a dashed line.

    Horizontal, because the names are accessions and example ids rather than
    short labels, and a vertical chart turns those into overlapping text.
    """
    measured = [(name, value, mark) for name, value, mark in rows if value is not None]
    if not measured:
        return ""
    numbers = [value for _name, value, _mark in measured]
    lo = min([0.0, *numbers, cut if cut is not None else 0.0])
    hi = max([0.0, *numbers, cut if cut is not None else 0.0])
    if hi <= lo:
        hi = lo + 1.0
    left = 150.0
    width = 270.0
    zero = left + _frac(0.0, lo, hi) * width
    height = 10 + 22 * len(measured)
    parts: list[str] = []
    for index, (name, value, mark) in enumerate(measured):
        y = 6 + index * 22
        parts.append(f'<text x="0" y="{y + 12}" font-size="11" fill="#26323d">{_esc(_clip(name))}</text>')
        x = left + _frac(value, lo, hi) * width
        start, end = (zero, x) if x >= zero else (x, zero)
        fill = highlight if mark else _BAR_INK
        parts.append(
            f'<rect x="{start:.1f}" y="{y + 2:.0f}" width="{max(1.0, end - start):.1f}" height="14" rx="3" '
            f'fill="{fill}"/>'
            f'<text x="{end + 5:.1f}" y="{y + 13:.0f}" font-size="10" fill="#26323d">{_esc(_num(value))}</text>'
        )
    if cut is not None:
        cx = left + _frac(cut, lo, hi) * width
        parts.append(
            f'<line x1="{cx:.1f}" y1="2" x2="{cx:.1f}" y2="{height - 4:.0f}" '
            f'stroke="{_CUT_INK}" stroke-width="1.5" stroke-dasharray="3 2"/>'
        )
    return (
        f'<svg viewBox="0 0 460 {height}" role="img" class="scale-chart" data-chart="{_esc(slug)}">'
        f"<title>{_esc(title)}</title>"
        f"{''.join(parts)}</svg>"
    )


def _sample_charts(samples: list[tuple[str, float | None, float | None, object]]) -> str:
    """Every design's instability and GRAVY, against the cut each one is judged on."""
    shown = samples[:_SAMPLE_CAP]
    # Named above each chart, because the two have different scales and the
    # same row names: unlabelled, the second one reads as more of the first.
    instability_rows = [(name, value, passed is False) for name, value, _gravy, passed in shown if value is not None]
    gravy_rows = [(name, value, passed is False) for name, _instability, value, passed in shown if value is not None]
    parts: list[str] = []
    if len(samples) > _SAMPLE_CAP and (instability_rows or gravy_rows):
        parts.append(f"<p>Showing the first {_SAMPLE_CAP} of {len(samples)} designs.</p>")
    if instability_rows:
        parts.append('<p class="chart-name">Instability</p>')
        parts.append(
            _bars(
                instability_rows,
                cut=_INSTABILITY_MAX,
                title="Instability of each design",
                slug="samples-instability",
                highlight=_EDGE["hit"],
            )
        )
    if gravy_rows:
        parts.append('<p class="chart-name">GRAVY</p>')
        parts.append(
            _bars(
                gravy_rows,
                cut=_GRAVY_MAX,
                title="GRAVY of each design",
                slug="samples-gravy",
                highlight=_EDGE["hit"],
            )
        )
    return "".join(parts)


def _style() -> str:
    rules = [
        ".axis-board { margin: 0 0 16px; }",
        ".axis-board > input { position: absolute; opacity: 0; width: 1px; height: 1px; }",
        ".axis-board > label { display: inline-flex; align-items: center; gap: 6px; "
        "margin: 0 6px 6px 0; padding: 4px 10px; "
        "border: 1px solid #b7c3ce; border-radius: 999px; background: #f4f7f9; font-size: 12px; "
        "font-weight: 600; cursor: pointer; }",
        ".axis-board > input:focus + label { outline: 2px solid #0b6e4f; }",
        ".axis-board > input:checked + label { color: #fff; }",
        ".axis-board > input:checked + label .dot { background: #fff; border-color: #fff; }",
        f".axis-board > input:checked + label.tab-hit {{ background: {_TAB_ON['hit']}; "
        f"border-color: {_TAB_ON['hit']}; }}",
        f".axis-board > input:checked + label.tab-clear {{ background: {_TAB_ON['clear']}; "
        f"border-color: {_TAB_ON['clear']}; }}",
        f".axis-board > input:checked + label.tab-untested {{ background: {_TAB_ON['untested']}; "
        f"border-color: {_TAB_ON['untested']}; }}",
        # Unselected tabs carry the state as a ring and a dot, so the row reads
        # green / red / grey at a glance without one tab having to be open.
        f".axis-board > label.tab-untested {{ box-shadow: inset 0 0 0 2px {_EDGE['untested']}; "
        f"background: {_FILL['untested']}; color: {_INK['untested']}; }}",
        f".axis-board > label.tab-hit {{ box-shadow: inset 0 0 0 2px {_EDGE['hit']}; "
        f"background: {_FILL['hit']}; color: {_INK['hit']}; }}",
        f".axis-board > label.tab-clear {{ box-shadow: inset 0 0 0 2px {_EDGE['clear']}; "
        f"background: {_FILL['clear']}; color: {_INK['clear']}; }}",
        ".dot { width: 9px; height: 9px; border-radius: 50%; border: 1px solid; flex: none; }",
        f".dot-clear {{ background: {_DOT['clear']}; border-color: {_EDGE['clear']}; }}",
        f".dot-hit {{ background: {_DOT['hit']}; border-color: {_EDGE['hit']}; }}",
        f".dot-untested {{ background: {_DOT['untested']}; border-color: {_EDGE['untested']}; }}",
        ".status-grid { list-style: none; display: grid; gap: 5px; margin: 10px 0; padding: 0; "
        "grid-template-columns: repeat(auto-fit, minmax(118px, 1fr)); }",
        ".status-grid .chip { border: 1px solid; border-radius: 8px; padding: 4px 7px; font-size: 12px; "
        "line-height: 1.25; }",
        ".status-grid .chip span { display: block; font-weight: 600; }",
        ".status-grid .chip strong { font-weight: 700; }",
        f".chip-clear {{ background: {_FILL['clear']}; border-color: {_EDGE['clear']}; color: {_INK['clear']}; }}",
        f".chip-hit {{ background: {_FILL['hit']}; border-color: {_EDGE['hit']}; color: {_INK['hit']}; }}",
        f".chip-untested {{ background: {_FILL['untested']}; border-color: {_EDGE['untested']}; "
        f"color: {_INK['untested']}; }}",
        ".axis-panel { display: none; }",
        ".axis-board > input:checked ~ .axis-panels { border: 1px solid #d5dee6; border-radius: 10px; "
        "background: #fbfdfe; padding: 10px 12px 2px; }",
        ".axis-panel h3 { margin: 0 0 4px; font-size: 15px; }",
        ".axis-status { font-weight: 700; margin: 0 0 6px; }",
        f".status-hit {{ color: {_INK['hit']}; }} "
        f".status-clear {{ color: {_INK['clear']}; }} "
        f".status-untested {{ color: {_INK['untested']}; }}",
        ".axis-todo { margin: 0 0 8px; padding: 7px 9px; border-left: 3px solid #8aa0b2; "
        "background: #eef3f7; border-radius: 0 7px 7px 0; }",
        ".guide, .metric-row { display: grid; gap: 8px; grid-template-columns: 1fr 1fr; }",
        ".guide p, .metric-row p { margin: 0 0 4px; }",
        ".guide { margin: 6px 0 8px; }",
        ".scale-chart { width: 100%; height: auto; max-width: 460px; display: block; margin: 2px 0 6px; }",
        ".chart-name { margin: 8px 0 0; font-size: 12px; font-weight: 700; color: #3d4a57; }",
        ".plain-metrics { margin: 6px 0 10px; padding-left: 18px; }",
        "details { margin: 8px 0 12px; }",
    ]
    # One rule per tab the widest board can hold, not per tab the protein board
    # holds. A ligand board draws the ten panel proteins and the three plasma
    # checks, and with nine rules its last four tabs opened nothing: clicking
    # one unchecked the first and showed an empty box. A rule past the end of a
    # narrower board matches nothing, because nth-of-type is scoped to each
    # board.
    for index in range(1, len(_TOX_AXES) + len(_CATALOG) + 1):
        rules.append(
            ".axis-board > input:nth-of-type("
            f"{index}):checked ~ .axis-panels > .axis-panel:nth-child({index}) {{ display: block; }}"
        )
    rules.append("@media (max-width: 700px) { .guide, .metric-row { grid-template-columns: 1fr; } }")
    return "<style>" + " ".join(rules) + "</style>"


def _script() -> str:
    """Keep Enter on a tab from submitting the page.

    The tabs are radio inputs inside the panel's one form, and Enter in a form
    control activates a submit button. On this page that button is Close, so
    arrowing to a tab and pressing Enter ended the run instead of opening the
    tab. Space still ticks the radio, which is what opens it.
    """
    return (
        "<script>\n"
        'document.querySelectorAll(".axis-board > input").forEach(function (tab) {\n'
        '  tab.addEventListener("keydown", function (event) {\n'
        '    if (event.key === "Enter") event.preventDefault();\n'
        "  });\n"
        "});\n"
        "</script>"
    )


def _screen_sentence(line: str) -> bool:
    return (
        line.startswith("Screen ")
        or line.startswith("Screen:")
        or line.startswith("Measured ")
        or line.startswith("Measured:")
    )


def _design_title(entry: str, chain: str) -> str:
    if entry and chain:
        return f"{entry} chain {chain}"
    if entry:
        return entry
    return "This design"


def _axis_record(screen: Mapping[str, object], axis: str, bucket: str) -> dict[str, object]:
    group = screen.get(bucket)
    if not isinstance(group, dict):
        return {}
    record = group.get(axis)
    return dict(record) if isinstance(record, dict) else {}


def _state(record: Mapping[str, object]) -> str:
    state = record.get("state")
    if isinstance(state, str) and state in _STATES:
        return state
    return "untested"


def _is_declared(record: Mapping[str, object]) -> bool:
    """Whether a person declared this axis's state in ``axis_evidence.json``.

    ``screen._stamp_source`` stamps that source and clears the blocker, so
    nothing else on the record says the state was not measured.
    """
    return record.get("source") == _CURATED_SOURCE


def _declared_axes(screen: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    """Every axis of this screen whose state a person declared.

    The block names come from ``mcp_api``, which already names the declared
    axes beside the call for the MCP reader, so the page and that verdict
    cannot look in different places.
    """
    found: dict[str, Mapping[str, object]] = {}
    for name in _AXIS_GROUPS:
        group = screen.get(name)
        if not isinstance(group, Mapping):
            continue
        for axis, record in group.items():
            if isinstance(axis, str) and axis and isinstance(record, Mapping) and _is_declared(record):
                found[axis] = record
    return found


def _declared_sentence(screen: Mapping[str, object]) -> str:
    """The axes behind the call that nobody measured, and what they measured as.

    A ``screened_clear`` resting on a person's call reads exactly like one
    every axis was measured clear for. Where the measurement disagreed, that
    is named too: the record keeps it under ``measured_state`` and it is the
    part of a verdict a reader cannot reconstruct.
    """
    found = _declared_axes(screen)
    if not found:
        return ""
    pieces: list[str] = []
    for axis in sorted(found):
        record = found[axis]
        said = axis.replace("_", " ")
        measured = record.get("measured_state")
        if isinstance(measured, str) and measured and measured != record.get("state"):
            said = f"{said}, which measured {measured.replace('_', ' ')}"
        pieces.append(said)
    return "Declared by a person, not measured: " + "; ".join(pieces) + "."


def _state_word(state: str, declared: bool) -> str:
    word = _STATE_WORD.get(state, state)
    return f"{word} (declared)" if declared else word


def _status(state: str, *, declared: bool = False) -> str:
    if state == "hit":
        said = "Hit. This check found a problem."
    elif state == "clear":
        said = "Clear on this check."
    else:
        said = "Not decided."
    if declared:
        return f"{said} Declared by a person, not measured."
    return said


def _call_text(call: object) -> str:
    if call == "axis_hit":
        return "Hit. At least one check found a problem."
    if call == "screened_clear":
        return "Every check in this bar is clear."
    return "Not decided. At least one check in the bar has no result."


def _sequence_sentence(passed: object, gravy: float | None, instability: float | None) -> str:
    if passed is True:
        # ExPASy ProtParam states the stability rule as "smaller than 40", and
        # ``sequence_pass`` applies it strictly, so the page says "below" and
        # not "at or below". The GRAVY half is this project's own bar and is
        # still inclusive.
        return (
            "Sequence check passed. GRAVY is at or below "
            f"{_num(_GRAVY_MAX)} and instability is below {_num(_INSTABILITY_MAX)}."
        )
    if passed is False:
        reasons: list[str] = []
        if gravy is not None and gravy > _GRAVY_MAX:
            reasons.append(f"GRAVY {_num(gravy)} is above {_num(_GRAVY_MAX)}")
        if instability is not None and instability >= _INSTABILITY_MAX:
            reasons.append(f"instability {_num(instability)} is not below {_num(_INSTABILITY_MAX)}")
        if reasons:
            return "Sequence check failed. " + ", and ".join(reasons) + "."
        return "Sequence check failed."
    return ""


def _modality_lines(record: Mapping[str, object]) -> list[str]:
    """The drug-type report. An absent record draws nothing."""
    kind = record.get("binder_kind")
    if kind == "protein":
        heading, rows = "Protein sequence triage", _polypeptide_rows(record)
    elif kind == "peptide":
        heading, rows = "Peptide sequence triage", _polypeptide_rows(record)
    elif kind == "nucleic":
        heading, rows = "Oligonucleotide triage", _oligo_rows(record)
    elif kind == "ligand":
        heading, rows = "Small-molecule composition", _ligand_rows(record)
    else:
        return []
    if not rows:
        return []
    items = "".join(f"<li>{_esc(row)}</li>" for row in rows if row)
    return [f"<h4>{heading}</h4>", f'<ul class="plain-metrics">{items}</ul>']


def _polypeptide_rows(record: Mapping[str, object]) -> list[str]:
    length = record.get("length")
    rows = []
    if isinstance(length, int) and not isinstance(length, bool) and length > 0:
        rows.append(f"Length {length} residues.")
    index = _as_float(record.get("aliphatic_index"))
    if index is not None and index >= 0:
        rows.append(f"Aliphatic index {_num(index)}.")
    else:
        lower_bound = _as_float(record.get("aliphatic_index_lower_bound"))
        if lower_bound is not None and lower_bound >= 0:
            rows.append(
                f"Identified residues contribute {_num(lower_bound)} to the aliphatic index. "
                "The whole-chain value depends on the unresolved residues."
            )
    mass = _as_float(record.get("molecular_weight"))
    if mass is not None and mass > 0:
        origin = ""
        if record.get("molecular_weight_source") == "iupac_conventional":
            origin = ", from the amino-acid formulas"
        rows.append(f"Molecular weight {_num(mass)}{origin}.")
    for row in (
        _extinction_sentence(record),
        _absorbance_sentence(record),
        _half_life_sentence(record),
        _composition_sentence(record),
        _atom_sentence(record),
        *_complexity_rows(record),
        _triage_motif("Furin cleavage motif", record, "furin"),
        _triage_motif("Asp-Pro", record, "acid_cleavage"),
        _glutamine_sentence(record),
        _nonstandard_sentence(record),
        _run_sentence(
            "Longest same-residue run",
            record.get("longest_homopolymer"),
            record.get("homopolymer_start"),
            residue=_one_letter(record.get("homopolymer_residue")),
        ),
        _run_sentence(
            "Longest hydrophobic run",
            record.get("longest_hydrophobic_run"),
            record.get("hydrophobic_run_start"),
        ),
        _transmembrane_sentence(record),
        _aggrescan_sentence(record),
    ):
        if row:
            rows.append(row)
    if not rows:
        return []
    rows.append("These counts are a report. No pass band applies to them.")
    return rows


def _extinction_sentence(record: Mapping[str, object]) -> str:
    reduced = _whole(record.get("extinction_280_reduced"))
    paired = _whole(record.get("extinction_280_cystine"))
    parts: list[str] = []
    if reduced is not None:
        parts.append(f"{reduced} with cystines reduced")
    if paired is not None:
        parts.append(f"{paired} with cystines paired")
    if not parts:
        return ""
    return "Extinction at 280 nm is " + " and ".join(parts) + "."


def _absorbance_sentence(record: Mapping[str, object]) -> str:
    parts: list[str] = []
    reduced = _whole(record.get("extinction_280_reduced"))
    paired = _whole(record.get("extinction_280_cystine"))
    if reduced is not None and reduced > 0:
        number = _as_float(record.get("absorbance_280_reduced"))
        if number is not None and number > 0:
            parts.append(f"{_num(number)} with cystines reduced")
    if paired is not None and paired > 0:
        number = _as_float(record.get("absorbance_280_cystine"))
        if number is not None and number > 0:
            parts.append(f"{_num(number)} with cystines paired")
    if not parts:
        return ""
    return "Absorbance 0.1% (1 g/L) is " + " and ".join(parts) + "."


def _half_life_sentence(record: Mapping[str, object]) -> str:
    clauses: list[str] = []
    for key, label in (
        ("half_life_mammalian", "mammalian"),
        ("half_life_yeast", "yeast"),
        ("half_life_ecoli", "E. coli"),
    ):
        phrase = record.get(key)
        if isinstance(phrase, str) and phrase in HALF_LIFE_PHRASES:
            clauses.append(f"{label} {phrase}")
    if not clauses:
        return ""
    return "N-end rule half-life: " + ", ".join(clauses) + "."


def _composition_sentence(record: Mapping[str, object]) -> str:
    composition = record.get("residue_composition")
    if not isinstance(composition, dict):
        return ""
    parts: list[str] = []
    for letter in "ACDEFGHIKLMNPQRSTVWY":
        count = composition.get(letter)
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            continue
        parts.append(f"{letter} {count}")
    if not parts:
        return ""
    return "Composition " + ", ".join(parts) + "."


def _atom_sentence(record: Mapping[str, object]) -> str:
    atoms = record.get("atom_counts")
    if not isinstance(atoms, dict):
        return ""
    parts: list[str] = []
    for element in ("C", "H", "N", "O", "S"):
        count = atoms.get(element)
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            continue
        parts.append(f"{element} {count}")
    if not parts:
        return ""
    return "Atoms " + ", ".join(parts) + "."


def _complexity_rows(record: Mapping[str, object]) -> list[str]:
    rows: list[str] = []
    entropy = _as_float(record.get("sequence_entropy_bits"))
    if entropy is not None and entropy >= 0:
        shown = _num(entropy)
        word = "bit" if shown == "1" else "bits"
        rows.append(f"Sequence entropy {shown} {word}.")
    residue = record.get("most_common_residue")
    share = _as_float(record.get("most_common_fraction"))
    if (
        isinstance(residue, str)
        and len(residue) == 1
        and residue.isalpha()
        and residue.isupper()
        and share is not None
        and 0.0 <= share <= 1.0
    ):
        rows.append(f"Most common residue {residue}, share {_num(share)}.")
    distinct = record.get("distinct_residues")
    if isinstance(distinct, int) and not isinstance(distinct, bool) and distinct > 0:
        rows.append(f"Distinct residues {distinct}.")
    return rows


def _transmembrane_sentence(record: Mapping[str, object]) -> str:
    count = record.get("transmembrane_count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        return ""
    if count == 0:
        return "Kyte-Doolittle span 0."
    details: list[str] = []
    spans = record.get("transmembrane_spans")
    if isinstance(spans, list):
        for span in spans:
            if not isinstance(span, dict):
                continue
            start = span.get("residue_start")
            end = span.get("residue_end")
            hydropathy = _as_float(span.get("hydropathy"))
            if isinstance(start, bool) or not isinstance(start, int) or start < 1:
                continue
            if isinstance(end, bool) or not isinstance(end, int) or end < start or hydropathy is None:
                continue
            details.append(f"Residues {start}-{end}, hydropathy {_num(hydropathy)}")
    if details:
        return f"Kyte-Doolittle span {count}. " + ". ".join(details) + "."
    return f"Kyte-Doolittle span {count}."


def _aggrescan_sentence(record: Mapping[str, object]) -> str:
    """AGGRESCAN normalized sum. A score that is not that method stays off the page."""
    found = record.get("aggrescan")
    if not isinstance(found, Mapping) or found.get("method") != "aggrescan_2007":
        return ""
    score = _as_float(found.get("na4vss"))
    count = found.get("hotspot_count")
    if score is None or isinstance(count, bool) or not isinstance(count, int) or count < 0:
        return ""
    sentence = f"AGGRESCAN normalized sum {_num(score)}. Hot spots {count}."
    if count == 0:
        return sentence
    details: list[str] = []
    spans = found.get("hotspots")
    if isinstance(spans, list):
        for span in spans:
            if not isinstance(span, dict):
                continue
            start = span.get("residue_start")
            end = span.get("residue_end")
            if isinstance(start, bool) or not isinstance(start, int) or start < 1:
                continue
            if isinstance(end, bool) or not isinstance(end, int) or end < start:
                continue
            details.append(f"Residues {start}-{end}")
    if details:
        return f"{sentence} " + ". ".join(details) + "."
    return sentence


def _whole(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _oligo_rows(record: Mapping[str, object]) -> list[str]:
    length = record.get("length")
    rows: list[str] = []
    if isinstance(length, int) and not isinstance(length, bool) and length > 0:
        rows.append(f"Length {length} nucleotides.")
    fraction = _as_float(record.get("gc_fraction"))
    if fraction is not None and 0.0 <= fraction <= 1.0:
        rows.append(f"GC fraction {_num(fraction)}.")
    for row in (
        _cpg_sentence(record),
        _run_sentence("Longest G run", record.get("longest_g_run"), record.get("g_run_start")),
    ):
        if row:
            rows.append(row)
    if not rows:
        return []
    rows.append("These counts are a report. No pass band applies to them.")
    return rows


def _ligand_rows(record: Mapping[str, object]) -> list[str]:
    if record.get("composition") == "ambiguous":
        count = record.get("component_count")
        if isinstance(count, int) and not isinstance(count, bool) and count > 1:
            return [
                f"The structure names {count} non-polymer components. No single formula weight is reported.",
                "No pass band applies to this report.",
            ]
        return ["No single formula weight is reported.", "No pass band applies to this report."]
    weight = _as_float(record.get("formula_weight"))
    if weight is None:
        return []
    name = component_label(record.get("component_id"))
    origin = ""
    if record.get("formula_weight_source") == "iupac_conventional":
        origin = ", from the chemical formula"
    if name is None:
        lead = f"Formula weight {_num(weight)}{origin}."
    else:
        lead = f"Formula weight {_num(weight)} for {name}{origin}."
    return [lead, "No pass band applies to the formula weight."]


def _triage_motif(label: str, record: Mapping[str, object], prefix: str) -> str:
    count = record.get(f"{prefix}_count")
    if isinstance(count, bool) or not isinstance(count, int):
        return ""
    if count == 0:
        return f"{label} 0."
    sites = record.get(f"{prefix}_sites")
    details: list[str] = []
    if isinstance(sites, list):
        for site in sites:
            if not isinstance(site, dict):
                continue
            residue = site.get("residue")
            motif = site.get("motif")
            if isinstance(residue, bool) or not isinstance(residue, int) or residue < 1:
                continue
            if not isinstance(motif, str) or motif not in TRIAGE_MOTIFS:
                continue
            details.append(f"Residue {residue}, motif {motif}")
    if details:
        return f"{label} {count}. " + ". ".join(details) + "."
    return f"{label} {count}."


def _glutamine_sentence(record: Mapping[str, object]) -> str:
    flag = record.get("n_terminal_glutamine")
    if flag is True:
        return "N-terminal glutamine is present."
    if flag is False:
        return "N-terminal glutamine is absent."
    return ""


def _nonstandard_sentence(record: Mapping[str, object]) -> str:
    count = record.get("nonstandard_count")
    if isinstance(count, bool) or not isinstance(count, int):
        return ""
    if count == 0:
        return "Nonstandard residue 0."
    numbers = _positive_labels(record.get("nonstandard_residues"))
    if not numbers:
        return f"Nonstandard residue {count}."
    word = "Residue" if len(numbers) == 1 else "Residues"
    return f"Nonstandard residue {count}. {word} {', '.join(numbers)}."


def _run_sentence(label: str, length: object, start: object, *, residue: str | None = None) -> str:
    if isinstance(length, bool) or not isinstance(length, int):
        return ""
    if isinstance(start, bool) or not isinstance(start, int) or start < 1:
        if residue is None:
            return f"{label} {length}."
        return f"{label} {length}, residue {residue}."
    if residue is None:
        return f"{label} {length}, starting at residue {start}."
    return f"{label} {length}, residue {residue}, starting at residue {start}."


def _cpg_sentence(record: Mapping[str, object]) -> str:
    count = record.get("cpg_count")
    if isinstance(count, bool) or not isinstance(count, int):
        return ""
    if count == 0:
        return "CpG 0."
    numbers = _positive_labels(record.get("cpg_positions"))
    if not numbers:
        return f"CpG {count}."
    word = "Residue" if len(numbers) == 1 else "Residues"
    return f"CpG {count}. {word} {', '.join(numbers)}."


def _positive_labels(raw: object) -> list[str]:
    if not isinstance(raw, list):
        return []
    labels: list[str] = []
    for item in raw:
        if isinstance(item, bool) or not isinstance(item, int) or item < 1:
            continue
        labels.append(str(item))
    return labels


def _one_letter(raw: object) -> str | None:
    if not isinstance(raw, str) or len(raw) != 1 or not raw.isalpha():
        return None
    return raw


def _chemical_lines(record: Mapping[str, object]) -> list[str]:
    """The chemical report, drawn beside the sequence gauges.

    An absent record draws nothing. Counts of zero stay on the page. The
    measured line is the place that names a hotspot only when its count is
    above zero.
    """
    if not record:
        return []
    rows = [
        row
        for row in (
            _motif_sentence("Deamidation", record, "deamidation"),
            _motif_sentence("Isomerization", record, "isomerization"),
            _motif_sentence("N-glycosylation sequon", record, "glycosylation"),
            _oxidation_sentence(record),
            _free_cysteine_sentence(record),
            _charge_sentence(record),
        )
        if row
    ]
    if not rows:
        return []
    rows.append("These counts are a report. No pass band applies to them.")
    items = "".join(f"<li>{_esc(row)}</li>" for row in rows)
    return ["<h4>Chemical liability</h4>", f'<ul class="plain-metrics">{items}</ul>']


def _motif_sentence(label: str, record: Mapping[str, object], prefix: str) -> str:
    count = record.get(f"{prefix}_count")
    if isinstance(count, bool) or not isinstance(count, int):
        return ""
    if count == 0:
        return f"{label} 0."
    sites = record.get(f"{prefix}_sites")
    details: list[str] = []
    if isinstance(sites, list):
        for site in sites:
            if not isinstance(site, dict):
                continue
            residue = site.get("residue")
            motif = site.get("motif")
            if isinstance(residue, bool) or not isinstance(residue, int):
                continue
            if not isinstance(motif, str) or motif not in MOTIF_NAMES:
                continue
            details.append(f"Residue {residue}, motif {motif}")
    if details:
        return f"{label} {count}. " + ". ".join(details) + "."
    return f"{label} {count}."


def _number_text(raw: object) -> list[str]:
    if not isinstance(raw, list):
        return []
    return [str(item) for item in raw if isinstance(item, int) and not isinstance(item, bool)]


def _oxidation_sentence(record: Mapping[str, object]) -> str:
    parts = (
        _residue_sentence("Methionine", record, "methionine"),
        _residue_sentence("Tryptophan", record, "tryptophan"),
    )
    return " ".join(part for part in parts if part)


def _residue_sentence(label: str, record: Mapping[str, object], prefix: str) -> str:
    count = record.get(f"{prefix}_count")
    if isinstance(count, bool) or not isinstance(count, int):
        return ""
    if count == 0:
        return f"{label} 0."
    numbers = _number_text(record.get(f"{prefix}_residues"))
    if not numbers:
        return f"{label} {count}."
    word = "Residue" if len(numbers) == 1 else "Residues"
    return f"{label} {count}. {word} {', '.join(numbers)}."


def _free_cysteine_sentence(record: Mapping[str, object]) -> str:
    if record.get("free_cysteine") == "untested":
        return "Free cysteine was not counted. The structure file does not name disulfide partners."
    count = record.get("free_cysteine_count")
    if isinstance(count, bool) or not isinstance(count, int):
        return ""
    if count == 0:
        return "Free cysteine 0."
    numbers = _number_text(record.get("free_cysteine_residues"))
    if not numbers:
        return f"Free cysteine {count}."
    word = "Residue" if len(numbers) == 1 else "Residues"
    return f"Free cysteine {count}. {word} {', '.join(numbers)}."


def _charge_sentence(record: Mapping[str, object]) -> str:
    point = _as_float(record.get("isoelectric_point"))
    charge = _as_float(record.get("net_charge_ph74"))
    if point is None or charge is None:
        return ""
    return (
        f"Isoelectric point {_num(point)}. Net charge at pH 7.4 is {_num(charge)}. " "The pKa set is Bjellqvist 1994."
    )


def _flag_sentence(label: str, value: object) -> str:
    if value is True:
        return f"{label}: passed."
    if value is False:
        return f"{label}: failed."
    return ""


def _ppi_sentence(screen: Mapping[str, object]) -> str:
    value = screen.get("ppi_pass")
    blocker = screen.get("ppi_blocker")
    if value is True:
        return "PPI pass: passed."
    if value is False:
        return "PPI pass: failed."
    if isinstance(blocker, str) and blocker:
        said = _BLOCKERS.get(blocker)
        if said is not None:
            return said
        text = blocker.replace("_", " ")
        return text if text.endswith(".") else f"{text}."
    return ""


def _contact_sentence(value: float | None) -> str:
    if value is None:
        return ""
    if value > 0:
        return f"Contact surface {_num(value)}. Above 0 means the design touches the target. Higher is more contact."
    return "Contact surface 0. The design has no measured contact with the target."


def _clash_sentence(value: float | None) -> str:
    if value is None:
        return ""
    return f"Clash fraction {_num(value)}. Lower is the favorable side. 0 means no counted clash."


def _sample_rows(screen: Mapping[str, object]) -> list[tuple[str, float | None, float | None, object]]:
    """Each stored design: its name, instability, GRAVY, and its sequence verdict."""
    samples = screen.get("samples")
    if not isinstance(samples, list):
        return []
    rows: list[tuple[str, float | None, float | None, object]] = []
    for item in samples:
        if not isinstance(item, dict):
            continue
        name = item.get("example_id")
        label = name if isinstance(name, str) and name else "design"
        rows.append(
            (
                label,
                _as_float(item.get("instability_index")),
                _as_float(item.get("gravy")),
                item.get("sequence_pass"),
            )
        )
    return rows


def _coevolution_detail(record: Mapping[str, object]) -> list[str]:
    """The transfer note stored beside the UniRef50 count."""
    lines: list[str] = []
    signal = record.get("coevolution_signal")
    blocker = record.get("coevolution_blocker")
    if not blocker and isinstance(signal, str) and signal in COEVOLUTION_SENTENCES:
        lines.append(COEVOLUTION_SENTENCES[signal])
        lines.append(HUMAN_TOXICITY_SCOPE)
        lines.append("This note is stored beside the UniRef50 count.")
        # Inside the branch, because the count is how many rows agreed on THIS
        # note. Printed outside it, a table refused for naming two different
        # notes still opened with "The same note is on 2 rows." -- a sentence
        # the three below it go on to deny.
        rows = record.get("coevolution_rows")
        if isinstance(rows, int) and not isinstance(rows, bool) and rows > 1:
            lines.append(f"The same note is on {rows} rows.")
    lines.extend(_named_ids("Other species named", record.get("coevolution_species")))
    named = record.get("coevolution_signals")
    if isinstance(named, list):
        sentences = [
            COEVOLUTION_SENTENCES[item] for item in named if isinstance(item, str) and item in COEVOLUTION_SENTENCES
        ]
        if sentences:
            lines.append("The rows name: " + " ".join(sentences))
    if isinstance(blocker, str) and blocker:
        said = _BLOCKERS.get(blocker, blocker.replace("_", " "))
        action = _ACTIONS.get(blocker, "")
        lines.append(f"{said} {action}".strip())
        if record.get("human_toxicity_state") == "untested" and blocker != "coevolution_not_configured":
            lines.append(HUMAN_TOXICITY_SCOPE)
    return lines


def _named_ids(caption: str, value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    names = [str(item) for item in value if isinstance(item, str) and item]
    if not names:
        return []
    shown = names[:_SAMPLE_CAP]
    text = f"{caption}: {', '.join(shown)}"
    if len(names) > _SAMPLE_CAP:
        text = f"{text}, and {len(names) - _SAMPLE_CAP} more"
    return [text + "."]


def _dict(value: object) -> dict[str, object]:
    return dict(value) if isinstance(value, dict) else {}


def _as_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


def _num(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.4f}".rstrip("0").rstrip(".")


def _clip(name: str, limit: int = 24) -> str:
    """A row label short enough for the chart's name column."""
    return name if len(name) <= limit else name[: limit - 1] + "…"


def _anchor(x: float, left: float, width: float) -> str:
    """Keep a chart label inside the drawing rather than centred off the edge."""
    if x < left + 24:
        return "start"
    if x > left + width - 24:
        return "end"
    return "middle"


def _bounds(
    value: float | None,
    vmin: float,
    vmax: float,
    cuts: Sequence[float] = (),
) -> tuple[float, float]:
    """Round scale ends, wide enough for the number and every cut.

    Round, because the ends are printed: a scale reading -1.08 to 1.08 is the
    padding talking, not the measurement. The ends step outward instead, so a
    value never sits exactly on the edge where its own label would be clipped.
    """
    lo, hi = vmin, vmax
    for number in (value, *cuts):
        if number is None:
            continue
        lo = min(lo, number)
        hi = max(hi, number)
    if hi <= lo:
        hi = lo + 1.0
    step = _nice_step((hi - lo) / 4.0)
    lo = math.floor(lo / step) * step
    hi = math.ceil(hi / step) * step
    if value is not None:
        if value >= hi:
            hi += step
        if value <= lo:
            lo -= step
    if hi <= lo:
        hi = lo + step
    return round(lo, 6), round(hi, 6)


def _nice_step(raw: float) -> float:
    """1, 2, 2.5 or 5 times a power of ten, at or above ``raw``."""
    if not math.isfinite(raw) or raw <= 0:
        return 1.0
    power = 10.0 ** math.floor(math.log10(raw))
    for multiple in (1.0, 2.0, 2.5, 5.0):
        if raw <= multiple * power:
            return multiple * power
    return 10.0 * power


def _frac(value: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return min(1.0, max(0.0, (value - lo) / (hi - lo)))


def _esc(value: object) -> str:
    return escape(str(value), quote=True)
