"""The training-corpus registry says what every embedded model saw, and stays honest.

Read-only: no network, no weights, nothing written.

The guards follow the house pattern from
``tests/test_disease_threshold_provenance.py``: set equality in **both**
directions against the runtime source of truth, imported lazily inside the test
body, so a deleted entry fails as loudly as an unregistered one.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest

from rfproteina.disease import corpus_provenance as cp

REPO = Path(__file__).resolve().parents[1]
VENDORED_PROPERTY_NAMES = (
    REPO / "screening subpackage" / "crossabsense" / "CrossAbSense" / "src" / "utils" / "property_names.py"
)


def _slugs_the_package_names() -> set[str]:
    """Every method slug a record can carry, read from the modules that define them.

    Imported here rather than listed, so adding a model to the package without
    declaring its corpus fails this test instead of passing silently.
    """
    from rfproteina.disease import (
        badapple2,
        canagliflozin,
        cellprofiler_stain,
        class_ii,
        crest,
        cypstrate,
        deepsp,
        deepviscosity,
        delphi_psr,
        developability,
        espsim,
        fastsolv,
        httk,
        humanness,
        jazzy,
        ligand_ames,
        ligand_caco2,
        ligand_dili,
        ligand_metabolites,
        ligand_microsomal_clearance,
        ligand_regressor,
        ligand_safety,
        ligand_solubility,
        metfrag,
        mhc_tables,
        molprobity_geometry,
        octant,
        openantigens,
        p2rank,
        peptide_rbc,
        pksim,
        prodigy,
        propka,
        qupkake,
        scscore,
        solubility,
        spyrmsd_pose,
        stability,
        tnp,
        toxscan,
        toxscan_ames,
        vina_pose,
    )
    # Importing the CATH metric initializes the training framework. Read only its
    # published method constants and require both names, also on a CPU-only install.
    source = REPO / "src" / "rfproteina" / "metrics" / "cath_metric.py"
    cath_ids = {
        target.id: ast.literal_eval(node.value)
        for node in ast.parse(source.read_text(encoding="utf-8")).body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name) and target.id in {"FOLDSEEK_METHOD", "PROTEINA_CLASSIFIER"}
    }

    return {
        badapple2.METHOD,
        canagliflozin.METHOD,
        toxscan.METHOD,
        toxscan_ames.METHOD_ID,
        crest.METHOD,
        espsim.METHOD,
        fastsolv.METHOD,
        jazzy.METHOD,
        cellprofiler_stain.METHOD_ID,
        prodigy.METHOD,
        peptide_rbc.METHOD,
        pksim.METHOD_ID,
        *p2rank.MODELS.values(),
        openantigens.METHOD,
        metfrag.METHOD,
        propka.METHOD,
        qupkake.METHOD,
        ligand_metabolites.METHOD,
        vina_pose.METHOD,
        spyrmsd_pose.METHOD,
        scscore.METHOD,
        *cypstrate.MODES.values(),
        tnp.METHOD,
        solubility.PROTEIN_SOL_METHOD,
        ligand_safety.BAYESHERG_METHOD,
        cath_ids["FOLDSEEK_METHOD"],
        cath_ids["PROTEINA_CLASSIFIER"],
        developability.CROSSABSENSE_METHOD,
        developability.DEEPSP_READER_METHOD,
        developability.NUMBERING_METHOD,
        deepsp.DEEPSP_METHOD,
        delphi_psr.METHOD,
        deepviscosity.VISCOSITY_METHOD,
        ligand_safety.ADMET_AI_METHOD,
        ligand_ames.METHOD,
        ligand_caco2.METHOD,
        ligand_dili.METHOD,
        ligand_microsomal_clearance.METHOD,
        ligand_solubility.METHOD,
        ligand_regressor.METHOD_FOREST,
        ligand_regressor.METHOD_CHEMPROP,
        class_ii.CLASS_II_METHOD,
        mhc_tables.METHOD,
        molprobity_geometry.METHOD,
        httk.METHOD,
        stability.METHOD,
        octant.OCTANT_METHOD,
        humanness.OASIS_METHOD,
        humanness.ABLANG2_METHOD,
    }


def test_every_model_the_package_names_declares_its_corpus():
    """A model that can put a number on a record must say what it was fitted on."""
    missing = _slugs_the_package_names() - set(cp.TRAINED_ON)
    assert missing == set(), f"these models produce numbers with no corpus declared: {sorted(missing)}"


def test_the_registry_declares_no_model_the_package_does_not_have():
    """Stale entries fail too: a declaration for a model nobody runs is noise.

    The three encoders and the Open Targets lookup are declared on purpose and
    are not method slugs, so they are named here rather than silently allowed.
    """
    extra = set(cp.TRAINED_ON) - _slugs_the_package_names()
    assert extra == {"esmc_6b", "esmc_300m", "prott5", "opentargets_associations"}


def test_every_corpus_a_model_names_is_defined_and_every_corpus_is_named():
    """No orphan in either direction."""
    assert cp.unresolved_corpora() == []
    named = {name for found in cp.TRAINED_ON.values() for name in found.corpora}
    assert set(cp.CORPORA) - named == set()


def test_every_head_is_a_label_of_a_corpus_the_model_saw():
    """A head predicting something its corpus never measured would break the derivation."""
    for slug, found in cp.TRAINED_ON.items():
        labels = {label for name in found.corpora for label in cp.CORPORA[name].labels}
        assert set(found.heads) <= labels, f"{slug} has heads outside its corpora: {found.heads}"


def test_every_kind_is_in_the_closed_vocabulary():
    assert set(cp.OVERLAP_CAVEAT) == cp.OVERLAP_KINDS
    for found in cp.TRAINED_ON.values():
        assert found.unaligned_kind in cp.OVERLAP_KINDS


def test_toxscan_local_ames_model_shares_the_author_export_label_corpus():
    from rfproteina.disease import toxscan, toxscan_ames

    local = cp.trained_on(toxscan_ames.METHOD_ID)
    assert local.corpora == cp.trained_on(toxscan.METHOD).corpora == ("ToxScan-study-partitions",)
    assert cp.overlap_with(toxscan_ames.METHOD_ID, "AMES") == cp.BENCHMARK_LABEL_CORPUS
    assert cp.overlap_with(toxscan.METHOD, "Ames Mutagenicity") == cp.BENCHMARK_LABEL_CORPUS


def test_delphi_psr_names_only_the_private_ipi_training_set():
    model = "ipi_delphi_onehot_psr_v4"
    assert cp.trained_on(model).corpora == ("IPI-DELPHI-PSR",)
    assert cp.overlap_with(model, "psr_filter") == cp.BENCHMARK_LABEL_CORPUS
    assert cp.overlap_with(model, "PR_CHO") == cp.PRIVATE_CORPUS_OVERLAP_UNKNOWN


def test_the_gdpa1_labels_are_read_from_the_vendored_tree_not_asserted_here():
    """The vendored CrossAbSense source is the authority on what GDPa1 measured.

    Loaded by path because ``screening subpackage`` has a space in its name and
    is deliberately not an importable package -- the technique
    ``tests/helpers/disease.py:load_mcp_server`` already uses for the MCP script.

    This is the guard that keeps the contamination finding true. A vendor bump
    that adds an assay column fails here rather than drifting, which is exactly
    how the prose table in docs/ddd-benchmark-readiness.md went stale.
    """
    spec = importlib.util.spec_from_file_location("_vendored_property_names", VENDORED_PROPERTY_NAMES)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    upstream = set(module.PRIMARY_PROPERTIES.values()) | set(module.ADDITIONAL_PROPERTIES.values())
    assert set(cp.CORPORA["GDPa1"].labels) == upstream


def test_crossabsense_against_the_developability_assays_is_a_label_corpus():
    """The correction this registry exists to make.

    ``docs/ddd-benchmark-readiness.md`` rated CrossAbSense's contamination risk
    LOW on 2026-11-16. It is fitted on GDPa1, which is the measured assay panel
    the benchmark's Antibody Developability column scores, so against those
    assays it is the highest-risk model in the program, not the lowest.
    """
    assert cp.overlap_with("crossabsense_v0.9", "HIC") == cp.BENCHMARK_LABEL_CORPUS
    assert cp.overlap_with("crossabsense_v0.9", "PR_CHO") == cp.BENCHMARK_LABEL_CORPUS
    assert cp.overlap_with("crossabsense_v0.9", "Tm2") == cp.BENCHMARK_LABEL_CORPUS


def test_a_label_the_corpus_has_and_the_model_cannot_predict_is_declared_not_scored():
    """HAC is a GDPa1 column with no CrossAbSense head. No number is at risk; the corpus still is."""
    assert cp.overlap_with("crossabsense_v0.9", "HAC") == cp.BENCHMARK_LABEL_CORPUS_UNMODELLED


def test_a_sibling_readout_of_the_same_campaign_is_weaker_but_not_nothing():
    """Tm1 and Tonset are the same thermostability measurement read differently from Tm2."""
    assert cp.overlap_with("crossabsense_v0.9", "Tm1") == cp.BENCHMARK_LABEL_CORPUS_UNMODELLED
    found = cp.CORPORA["GDPa1"]
    assert set(found.family_of("Tm1")) == {"Tm1", "Tm2", "Tonset"}


def test_hic_and_hac_are_not_treated_as_one_quantity():
    """Both are hydrophobic chromatography on different phases. Conflating them would
    manufacture an overlap finding, which is the mirror of the error being fixed."""
    assert set(cp.CORPORA["GDPa1"].family_of("HAC")) == {"HAC"}


def test_risk_is_a_property_of_the_pair_and_not_of_the_model():
    """The same model is a label corpus against one dataset and unaligned against another."""
    assert cp.overlap_with("crossabsense_v0.9", "HIC") == cp.BENCHMARK_LABEL_CORPUS
    assert cp.overlap_with("crossabsense_v0.9", "ddG") == cp.PUBLIC_CORPUS_UNALIGNED
    assert cp.overlap_with("ablang2_pseudo_log_likelihood", "HIC") == cp.PUBLIC_CORPUS_UNALIGNED


def test_simulated_labels_are_distinguished_from_measured_ones():
    """DeepSP's labels come from molecular dynamics, so recalling them leaks no wet-lab answer."""
    assert cp.overlap_with("deepsp_conv1d_torch_v1", "SAP_pos") == cp.BENCHMARK_LABEL_CORPUS
    assert cp.overlap_with("deepsp_conv1d_torch_v1", "HIC") == cp.SIMULATED_LABELS


def test_a_runtime_lookup_is_its_own_kind():
    """Open Targets answers the target-prediction items by reading a public table live.

    That overlaps exactly when the benchmark's gold set is drawn from the same
    table, which is a sharper risk than a training corpus, not a milder one.
    """
    kind = cp.overlap_with("opentargets_associations", "disease_target_association")
    assert kind == cp.BENCHMARK_LABEL_CORPUS
    assert cp.overlap_with("opentargets_associations", "HIC") == cp.RUNTIME_LOOKUP


def test_an_undeclared_model_is_reported_rather_than_left_bare():
    """Mirrors threshold_provenance: the gap shows in the record instead of hiding."""
    assert cp.overlap_with("no_such_model", "HIC") == cp.CORPUS_UNDECLARED
    assert cp.overlap_with(None, None) == cp.CORPUS_UNDECLARED
    assert cp.undeclared(["crossabsense_v0.9", "no_such_model", ""]) == ["no_such_model"]


def test_a_declaration_rides_on_an_answer_even_with_no_model():
    """A declined item with no producer still gets a block, so nothing hides behind silence."""
    block = cp.declare(None)
    assert block["overlap"] == cp.CORPUS_UNDECLARED
    assert block["corpora"] == []
    assert block["caveat"]


def test_a_declaration_names_what_it_was_scored_against():
    block = cp.declare("crossabsense_v0.9", against="HIC")
    assert block["overlap_against"] == "HIC"
    assert block["overlap"] == cp.BENCHMARK_LABEL_CORPUS
    assert any(entry["name"] == "GDPa1" for entry in block["corpora"])
    assert block["corpora"][0]["rows"] == 242


@pytest.mark.parametrize("slug", sorted(cp.TRAINED_ON))
def test_every_entry_says_where_its_claim_was_read_from(slug):
    """No claim here is inferred from a model's name; each cites a file or a sentence."""
    assert cp.TRAINED_ON[slug].declared_from.strip()


def test_the_summary_line_reads_as_a_sentence():
    line = cp.summary_line("crossabsense_v0.9", "HIC")
    assert "benchmark_label_corpus" in line and "GDPa1" in line
