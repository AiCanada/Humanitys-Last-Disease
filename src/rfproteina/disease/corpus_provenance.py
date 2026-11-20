"""What each embedded model was trained on, and whether that overlaps what it is scored against.

Third sibling to :mod:`rfproteina.disease.provenance`, which answers *which
code* produced a record, and :mod:`rfproteina.disease.threshold_provenance`,
which answers *where a number's bar came from*. This one answers *what the
producing model already saw*.

The gap this closes is one an audit found by hand and could not keep closed.
``docs/ddd-benchmark-readiness.md`` carried a table of model, corpus and
"overlap risk" as prose. Nothing read it and no test pinned it, and it was
wrong where it mattered most: it rated CrossAbSense **LOW** while CrossAbSense
is fitted on GDPa1, the measured assay panel the benchmark's Antibody
Developability column scores. A model that has seen the answer key can recall
rather than predict, and the number it returns looks like its best work.

**Risk is a property of a pair, never of a model.** That is the whole design.
Asking "how contaminated is CrossAbSense" has no answer; asking "how
contaminated is CrossAbSense *against HIC*" has a precise one. AbLang-2 on the
Observed Antibody Space is genuinely unaligned with a developability assay and
would be a label corpus against a humanness benchmark. Storing a severity is
what let the old table go stale, so no severity is stored here:
:func:`overlap_with` derives one for a (model, dataset) pair, and the kinds are
**mechanisms**, which do not drift, rather than severities, which do.

The kinds are a closed vocabulary, so a consumer can branch on them instead of
matching strings:

``benchmark_label_corpus``
    Fitted on measured labels for the very quantity being scored. The strongest
    form. Treat the number as potentially recalled.
``benchmark_label_corpus_sibling``
    Fitted on a different column of the same measurement campaign, measuring
    the same quantity another way. Weaker, and not nothing.
``benchmark_label_corpus_unmodelled``
    The corpus carries labels for this quantity and the model has no head for
    it. No answer is at risk, because no answer exists -- but the corpus is
    still declared, because the next version of the model might grow one.
``public_corpus_unaligned``
    A large public corpus with no known per-item overlap with the thing being
    scored.
``simulated_labels``
    The labels are computed, not measured. Memorising them cannot leak a
    wet-lab answer.
``fitted_in_repo``
    Fitted here, from a named corpus, under this program's own split. The split
    is ours, so the reach of the fit is knowable and is already reported per
    prediction.
``runtime_lookup``
    Not a trained model: an answer read from a public table at run time. It
    contaminates exactly when the benchmark's gold set is drawn from that same
    table, which is a sharper risk than the training-corpus kind, not a milder
    one.
``corpus_undeclared``
    No entry here. A defect, not a kind.

Every entry records where its claim was read from, with a date, the way
``immunogenicity.ClassIIOption.licence_read_on`` and ``tables.Source`` already
do. Nothing here is inferred from a model's name.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

#: Fitted on measured labels for the quantity being scored.
BENCHMARK_LABEL_CORPUS = "benchmark_label_corpus"
#: Fitted on another column of the same campaign, measuring the same quantity.
BENCHMARK_LABEL_CORPUS_SIBLING = "benchmark_label_corpus_sibling"
#: The corpus has the labels; this model has no head that consumes them.
BENCHMARK_LABEL_CORPUS_UNMODELLED = "benchmark_label_corpus_unmodelled"
#: A large public corpus with no known per-item overlap with what is scored.
PUBLIC_CORPUS_UNALIGNED = "public_corpus_unaligned"
#: Labels are computed rather than measured.
SIMULATED_LABELS = "simulated_labels"
#: Fitted in this repository, from a named corpus, under this program's split.
FITTED_IN_REPO = "fitted_in_repo"
#: Read from a public table at run time rather than learned.
RUNTIME_LOOKUP = "runtime_lookup"
#: No entry in :data:`TRAINED_ON`. A defect, not a kind.
CORPUS_UNDECLARED = "corpus_undeclared"

#: The closed vocabulary. A kind outside this set is a bug in this file.
OVERLAP_KINDS = frozenset(
    {
        BENCHMARK_LABEL_CORPUS,
        BENCHMARK_LABEL_CORPUS_SIBLING,
        BENCHMARK_LABEL_CORPUS_UNMODELLED,
        PUBLIC_CORPUS_UNALIGNED,
        SIMULATED_LABELS,
        FITTED_IN_REPO,
        RUNTIME_LOOKUP,
        CORPUS_UNDECLARED,
    }
)


@dataclass(frozen=True)
class Corpus:
    """A body of data a model was fitted on, named so an overlap can be checked.

    ``labels`` is the load-bearing field: it is what lets a benchmark's name be
    matched against what the corpus actually measured. A corpus with an empty
    ``labels`` can still be declared -- that is the honest state for a large
    unlabelled sequence set -- but it can never produce a label-overlap finding.

    ``label_families`` groups labels that measure the same quantity a different
    way, which is what :data:`BENCHMARK_LABEL_CORPUS_SIBLING` is derived from.
    Membership is deliberately conservative: two assays go in a family only when
    they differ in protocol rather than in what they measure.

    ``rows`` is ``None`` unless this repository can show the number. A count
    read off a blog post is not a count.
    """

    name: str
    publisher: str
    locator: str
    read_from: str
    labels: tuple[str, ...] = ()
    label_families: tuple[tuple[str, ...], ...] = ()
    rows: int | None = None
    unit: str = ""
    licence: str = ""

    def family_of(self, label: str) -> tuple[str, ...]:
        """The labels measuring the same quantity as ``label``, including it."""
        for family in self.label_families:
            if label in family:
                return family
        return (label,)

    def as_record(self) -> dict[str, object]:
        """The flat form stored on an answer. Empty fields are dropped."""
        record: dict[str, object] = {
            "name": self.name,
            "publisher": self.publisher,
            "locator": self.locator,
            "read_from": self.read_from,
        }
        if self.rows is not None:
            record["rows"] = self.rows
        if self.unit:
            record["unit"] = self.unit
        if self.licence:
            record["licence"] = self.licence
        return record


@dataclass(frozen=True)
class TrainedOn:
    """What one model saw, keyed by the method slug its records already carry.

    ``heads`` is the subset of its corpora's labels the model actually predicts.
    The distinction between a label it has a head for and one it merely saw is
    the difference between :data:`BENCHMARK_LABEL_CORPUS` and
    :data:`BENCHMARK_LABEL_CORPUS_UNMODELLED`, and collapsing the two would
    either overstate the risk or hide a declaration duty.

    ``unaligned_kind`` is what this model is when no label matches: a fitted
    in-repo regressor and a vendor's sequence model are both "no overlap here"
    for different reasons, and a reader deserves to know which.

    ``declared_from`` is the sentence or file the claim was read from, so a
    reviewer can check it without trusting this file.
    """

    model: str
    corpora: tuple[str, ...]
    role: str
    declared_from: str
    heads: tuple[str, ...] = ()
    unaligned_kind: str = PUBLIC_CORPUS_UNALIGNED
    weights_installed: bool = False


#: The GDPa1 assay columns. Not restated from the model card: the guard test
#: reads them out of the vendored tree's own ``property_names.py`` and fails if
#: this tuple drifts from it.
_GDPA1_LABELS = (
    "HIC",
    "AC-SINS_pH7.4",
    "Titer",
    "Tm2",
    "PR_CHO",
    "Purity",
    "SEC_%Monomer",
    "Tm1",
    "HAC",
    "Tonset",
    "AC-SINS_pH6.0",
)

CORPORA: dict[str, Corpus] = {
    "GDPa1": Corpus(
        name="GDPa1",
        publisher="Ginkgo Bioworks",
        locator="ginkgo-datapoints/GDPa1",
        read_from=(
            "screening subpackage/crossabsense/CrossAbSense/MODEL_CARD.md, which names the "
            "dataset and '242 IgGs', and src/utils/property_names.py for the columns, 2026-11-18"
        ),
        labels=_GDPA1_LABELS,
        # Three melting readouts of one thermostability measurement, and the two
        # AC-SINS pH points, are siblings. HIC and HAC deliberately are NOT: both
        # are hydrophobic chromatography but on different stationary phases, and
        # treating them as one quantity would manufacture an overlap finding.
        label_families=(("Tm1", "Tm2", "Tonset"), ("AC-SINS_pH6.0", "AC-SINS_pH7.4")),
        rows=242,
        unit="IgG",
    ),
    "TDC-ADMET": Corpus(
        name="Therapeutics Data Commons, ADMET suite",
        publisher="Therapeutics Data Commons",
        locator="https://tdcommons.ai/single_pred_tasks/adme/",
        read_from="docs/screening-implementation-handoff.md and the ADMET-AI model card, 2026-11-18",
        labels=("Caco2", "HIA", "Pgp", "BBB", "PPBR", "VDss", "CYP", "Half_Life", "CL", "hERG", "AMES", "DILI"),
    ),
    "BindingDB": Corpus(
        name="BindingDB",
        publisher="BindingDB",
        locator="https://www.bindingdb.org/",
        read_from="ligand_regressor.CorpusReport.licence and its saved manifest notes, 2026-11-18",
        labels=("nav1_5", "bcrp", "bsep"),
        licence="BindingDB; cite BindingDB and the per-row reference",
    ),
    "IEDB": Corpus(
        name="IEDB",
        publisher="Immune Epitope Database",
        locator="https://www.iedb.org/",
        read_from="class_ii.CorpusReport.licence, and tables.SOURCES for the downloaded tables, 2026-11-18",
        labels=("mhc_class_i_binding", "mhc_class_ii_binding", "tcell_response"),
        licence="CC BY 4.0",
    ),
    "mass-spec-ligands": Corpus(
        name="Mass-spectrometry eluted-ligand sets",
        publisher="various, as bundled by MHCflurry",
        locator="mhcflurry models_class1_presentation",
        read_from="mhc_tables.MODELS_RELEASE and the MHCflurry 2.x publication, 2026-11-18",
        labels=("mhc_class_i_presentation",),
    ),
    "Megascale": Corpus(
        name="Megascale (Tsuboyama 2023)",
        publisher="Tsuboyama et al.",
        locator="Tsuboyama2023_Dataset2_Dataset3_20230416.csv",
        read_from=(
            "the vendored ThermoMPNN tree's local.yaml data_loc entries and "
            "screening subpackage/thermompnn/manifest.toml, which excludes data_all/, 2026-11-18"
        ),
        labels=("ddG",),
    ),
    "DeepSP-MD": Corpus(
        name="DeepSP molecular-dynamics descriptor set",
        publisher="Lai lab",
        locator="https://github.com/Lailabcode/DeepSP",
        read_from="developability.DEEPSP_SOURCE and the DeepSP publication, 2026-11-18",
        labels=("SAP_pos", "SCM_neg", "SCM_pos"),
    ),
    "DeepViscosity-mAbs": Corpus(
        name="DeepViscosity training antibodies",
        publisher="Lai lab",
        locator="https://github.com/Lailabcode/DeepViscosity",
        read_from="deepviscosity.TRAINING_ANTIBODIES and deepviscosity.CAVEATS, 2026-11-18",
        labels=("viscosity_class",),
        rows=229,
        unit="mAb",
    ),
    "OAS": Corpus(
        name="Observed Antibody Space",
        publisher="Oxford Protein Informatics Group",
        locator="https://opig.stats.ox.ac.uk/webapps/oas/",
        read_from="humanness.ABLANG2_SOURCE, 2026-11-18",
        labels=("antibody_sequence",),
    ),
    "OASis-peptides": Corpus(
        name="OASis 9-mer peptide set",
        publisher="BioPhi",
        locator="OASis v1, packaged",
        read_from="humanness.OASIS_METHOD and its packaged table, 2026-11-18",
        labels=("humanness_identity",),
    ),
    "UniRef": Corpus(
        name="UniRef-family protein sequences",
        publisher="UniProt consortium",
        locator="https://www.uniprot.org/uniref",
        read_from="the ESM-C and ProtT5 model cards, 2026-11-18",
        labels=("protein_sequence",),
    ),
    "OpenTargets": Corpus(
        name="Open Targets Platform associations",
        publisher="Open Targets",
        locator="https://api.platform.opentargets.org/api/v4/graphql",
        read_from="sources.OPENTARGETS_URL and sources.ASSOCIATED_TARGETS, 2026-11-18",
        labels=("disease_target_association",),
    ),
    "Octant-CYP": Corpus(
        name="OpenADMET Octant CYP inhibition release",
        publisher="OpenADMET",
        locator="openadmet/Octant_CYP_inhibition_reactivity_blog_release",
        read_from="octant.py's release constants, 2026-11-18",
        labels=("cyp3a4",),
        licence="CC BY 4.0",
    ),
}


#: Every model or lookup that can put a number on an answer, and what it saw.
#: Keyed by the method slug the record already carries, so a stored screen and a
#: declaration join on a key that exists on disk rather than on a second name
#: invented here.
TRAINED_ON: dict[str, TrainedOn] = {
    "crossabsense_v0.9": TrainedOn(
        model="crossabsense_v0.9",
        corpora=("GDPa1",),
        role="Five antibody developability heads, read from the oracle's own CSV.",
        declared_from="MODEL_CARD.md: 'datasets: ginkgo-datapoints/GDPa1' and '242 IgGs, Ginkgo Bioworks'",
        heads=("HIC", "PR_CHO", "AC-SINS_pH7.4", "Tm2", "Titer"),
        weights_installed=False,
    ),
    "deepsp_conv1d_torch_v1": TrainedOn(
        model="deepsp_conv1d_torch_v1",
        corpora=("DeepSP-MD",),
        role="Spatial aggregation and charge descriptors, run here as a torch port.",
        declared_from="deepsp.py's module docstring and developability.DEEPSP_SOURCE",
        heads=("SAP_pos", "SCM_neg", "SCM_pos"),
        unaligned_kind=SIMULATED_LABELS,
        weights_installed=True,
    ),
    "deepsp": TrainedOn(
        model="deepsp",
        corpora=("DeepSP-MD",),
        role="The same descriptors, read from DeepSP's own CSV rather than run.",
        declared_from="developability.DEEPSP_READER_METHOD and DEEPSP_SOURCE",
        heads=("SAP_pos", "SCM_neg", "SCM_pos"),
        unaligned_kind=SIMULATED_LABELS,
    ),
    "deepviscosity_ann_ensemble_torch_v1": TrainedOn(
        model="deepviscosity_ann_ensemble_torch_v1",
        corpora=("DeepViscosity-mAbs",),
        role="High/low viscosity class from DeepSP descriptors.",
        declared_from="deepviscosity.CAVEATS: '229 training antibodies, 162 low and 67 high'",
        heads=("viscosity_class",),
        weights_installed=True,
    ),
    "admet_ai_classification_v1": TrainedOn(
        model="admet_ai_classification_v1",
        corpora=("TDC-ADMET",),
        role="Ten-axis small-molecule ADMET probabilities.",
        declared_from="the ADMET-AI publication and docs/ddd-benchmark-readiness.md's contamination audit",
        heads=("Caco2", "HIA", "Pgp", "BBB", "PPBR", "VDss", "CYP", "Half_Life", "CL", "hERG", "AMES", "DILI"),
    ),
    "bindingdb_morgan_extratrees_v1": TrainedOn(
        model="bindingdb_morgan_extratrees_v1",
        corpora=("BindingDB",),
        role="In-repo regressor for the three axes ADMET-AI does not cover.",
        declared_from="ligand_regressor.train()'s manifest notes: 'trained on N pairs from BindingDB'",
        heads=("nav1_5", "bcrp", "bsep"),
        unaligned_kind=FITTED_IN_REPO,
    ),
    "bindingdb_chemprop_dmpnn_v1": TrainedOn(
        model="bindingdb_chemprop_dmpnn_v1",
        corpora=("BindingDB",),
        role="The same three axes, D-MPNN backend.",
        declared_from="ligand_regressor.train()'s manifest notes: 'trained on N pairs from BindingDB'",
        heads=("nav1_5", "bcrp", "bsep"),
        unaligned_kind=FITTED_IN_REPO,
    ),
    "iedb_pssm_dr_panel_v1": TrainedOn(
        model="iedb_pssm_dr_panel_v1",
        corpora=("IEDB",),
        role="In-repo class II DR panel PSSM.",
        declared_from="class_ii.save()'s notes: 'trained on N (allele, peptide) pairs from IEDB, CC BY 4.0'",
        heads=("mhc_class_ii_binding",),
        unaligned_kind=FITTED_IN_REPO,
    ),
    "mhcflurry": TrainedOn(
        model="mhcflurry",
        corpora=("IEDB", "mass-spec-ligands"),
        role="Class I binding and presentation prediction.",
        declared_from="mhc_tables.MODELS_RELEASE '2.3.0' and the MHCflurry 2.x publication",
        heads=("mhc_class_i_binding", "mhc_class_i_presentation"),
        weights_installed=True,
    ),
    "thermompnn_point_stability_v1": TrainedOn(
        model="thermompnn_point_stability_v1",
        corpora=("Megascale",),
        role="Site-saturation ddG for every point mutation of a structure.",
        declared_from=(
            "the vendored tree's local.yaml, whose data_loc names the Megascale CSV, and "
            "screening subpackage/thermompnn/manifest.toml, which excludes data_all/"
        ),
        heads=("ddG",),
        weights_installed=True,
    ),
    "ablang2_pseudo_log_likelihood": TrainedOn(
        model="ablang2_pseudo_log_likelihood",
        corpora=("OAS",),
        role="Antibody language-model likelihood, reported but not a humanness score.",
        declared_from="humanness.ABLANG2_SOURCE and ABLANG2_NOT_HUMANNESS",
        heads=("antibody_sequence",),
    ),
    "oasis_identity_10pct_subjects": TrainedOn(
        model="oasis_identity_10pct_subjects",
        corpora=("OASis-peptides",),
        role="Humanness as the fraction of 9-mers seen in 10% of subjects.",
        declared_from="humanness.OASIS_METHOD and its packaged peptide table",
        heads=("humanness_identity",),
    ),
    "anarcii": TrainedOn(
        model="anarcii",
        corpora=("OAS",),
        role="Antibody numbering, not a score.",
        declared_from="developability.NUMBERING_METHOD and NUMBERING_LICENCE",
        heads=(),
    ),
    "octant_cyp3a4_pic50_to_nm_v1": TrainedOn(
        model="octant_cyp3a4_pic50_to_nm_v1",
        corpora=("Octant-CYP",),
        role="Measured CYP3A4 pIC50 converted to nM. A table read, not a fit.",
        declared_from="octant.py's release constants and CC BY 4.0 licence",
        heads=("cyp3a4",),
        unaligned_kind=RUNTIME_LOOKUP,
    ),
    "opentargets_associations": TrainedOn(
        model="opentargets_associations",
        corpora=("OpenTargets",),
        role="Disease-to-target ranking, read live from the Open Targets GraphQL API.",
        declared_from="sources.OPENTARGETS_URL, sources.ASSOCIATED_TARGETS and MAX_ASSOCIATIONS",
        heads=("disease_target_association",),
        unaligned_kind=RUNTIME_LOOKUP,
    ),
    "esmc_6b": TrainedOn(
        model="esmc_6b",
        corpora=("UniRef",),
        role="Sequence encoder four CrossAbSense heads load.",
        declared_from="crossabsense.ORACLE_ENCODERS and the ESM-C model card",
        heads=(),
    ),
    "esmc_300m": TrainedOn(
        model="esmc_300m",
        corpora=("UniRef",),
        role="Sequence encoder the PR_CHO head loads.",
        declared_from="crossabsense.ORACLE_ENCODERS and the ESM-C model card",
        heads=(),
    ),
    "prott5": TrainedOn(
        model="prott5",
        corpora=("UniRef",),
        role="Sequence encoder the AC-SINS and Tm2 heads load.",
        declared_from="crossabsense.ORACLE_ENCODERS and the ProtT5 model card",
        heads=(),
    ),
}

#: What a model with no entry gets, so the gap shows on the answer rather than hiding.
UNDECLARED_CORPUS = TrainedOn(
    model="",
    corpora=(),
    role="",
    declared_from="",
    unaligned_kind=CORPUS_UNDECLARED,
)


def trained_on(model: str | None) -> TrainedOn:
    """What ``model`` saw. Never raises.

    A model nobody registered comes back ``corpus_undeclared``, which is the
    honest answer and a visible one, exactly as ``threshold_provenance`` reports
    an unregistered cut rather than omitting it.
    """
    if not model:
        return UNDECLARED_CORPUS
    return TRAINED_ON.get(model, UNDECLARED_CORPUS)


def overlap_with(model: str | None, dataset: str | None) -> str:
    """How ``model``'s training data overlaps ``dataset``. One of :data:`OVERLAP_KINDS`.

    Derived for the pair and never stored, because "how contaminated is this
    model" is not a question with an answer. Resolution order is strongest
    first: a label the model predicts, then a label it merely saw, then a
    sibling measurement of one it predicts, then whatever the model is when
    nothing matches.
    """
    found = trained_on(model)
    if found.unaligned_kind == CORPUS_UNDECLARED:
        return CORPUS_UNDECLARED
    if dataset:
        corpora = [CORPORA[name] for name in found.corpora if name in CORPORA]
        for corpus in corpora:
            if dataset in corpus.labels:
                return BENCHMARK_LABEL_CORPUS if dataset in found.heads else BENCHMARK_LABEL_CORPUS_UNMODELLED
        for corpus in corpora:
            family = set(corpus.family_of(dataset))
            if family & set(found.heads):
                return BENCHMARK_LABEL_CORPUS_SIBLING
    return found.unaligned_kind


#: One sentence per kind, for a reader who gets the record and not this file.
OVERLAP_CAVEAT: dict[str, str] = {
    BENCHMARK_LABEL_CORPUS: (
        "This model was fitted on measured values for the quantity being scored. Treat any "
        "number it produces here as potentially recalled rather than predicted."
    ),
    BENCHMARK_LABEL_CORPUS_SIBLING: (
        "This model was fitted on a different readout of the same measurement campaign. The "
        "quantity is not identical, and the overlap is not nothing."
    ),
    BENCHMARK_LABEL_CORPUS_UNMODELLED: (
        "The corpus behind this model carries measured values for the quantity being scored, "
        "but the model has no head for it, so there is no number at risk here."
    ),
    PUBLIC_CORPUS_UNALIGNED: ("A public corpus with no known per-item overlap with the quantity being scored."),
    SIMULATED_LABELS: (
        "The labels were computed rather than measured, so recalling them cannot leak a " "measured answer."
    ),
    FITTED_IN_REPO: (
        "Fitted in this repository from a named public corpus under this program's own split. "
        "Reach beyond the training set is reported per prediction."
    ),
    RUNTIME_LOOKUP: (
        "Read from a public table at run time rather than learned. This overlaps exactly when "
        "the thing being scored was itself drawn from that table."
    ),
    CORPUS_UNDECLARED: (
        "This model has no entry in corpus_provenance.TRAINED_ON. Treat any number it produces "
        "as of unknown provenance."
    ),
}


def declare(model: str | None, *, against: str | None = None) -> dict[str, object]:
    """The corpus block that rides on an answer, for ``model`` scored on ``against``.

    Populated even when the answer was declined and even when ``model`` is
    ``None``: an absent block on the declines would let exactly the contaminated
    items hide, which is the failure this module exists to prevent.
    """
    found = trained_on(model)
    kind = overlap_with(model, against)
    record: dict[str, object] = {
        "model": model or "",
        "corpora": [CORPORA[name].as_record() for name in found.corpora if name in CORPORA],
        "overlap": kind,
        "caveat": OVERLAP_CAVEAT[kind],
    }
    if against:
        record["overlap_against"] = against
    if found.role:
        record["role"] = found.role
    if found.declared_from:
        record["declared_from"] = found.declared_from
    return record


def undeclared(models: Iterable[str]) -> list[str]:
    """Models with no entry here. Empty is healthy."""
    return sorted({model for model in models if model and model not in TRAINED_ON})


def describe_corpora() -> dict[str, dict[str, object]]:
    """Every corpus, keyed by name, for a reader asking what the program was built on."""
    return {name: corpus.as_record() for name, corpus in sorted(CORPORA.items())}


def summary_line(model: str, dataset: str) -> str:
    """One line a human can read, for a report or a console summary."""
    kind = overlap_with(model, dataset)
    names = ", ".join(trained_on(model).corpora) or "nothing recorded"
    return f"{model} vs {dataset}: {kind} (trained on {names})"


def unresolved_corpora() -> list[str]:
    """Corpus names a model claims that :data:`CORPORA` does not define. Empty is healthy."""
    named = {name for found in TRAINED_ON.values() for name in found.corpora}
    return sorted(named - set(CORPORA))
