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
``private_corpus_overlap_unknown``
    Training rows are not public, so independence for a different assay or
    dataset cannot be verified.
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
``user_supplied_measurements``
    Observations are transcribed from a user-supplied report, not a model or a
    public reference. The recorded source and assay scope remain authoritative.
``corpus_undeclared``
    No entry here. A defect, not a kind.
``no_training_corpus``
    A deterministic measurement on the supplied inputs, with no training set.

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
#: A private corpus whose per-item overlap cannot be verified.
PRIVATE_CORPUS_OVERLAP_UNKNOWN = "private_corpus_overlap_unknown"
#: Labels are computed rather than measured.
SIMULATED_LABELS = "simulated_labels"
#: Fitted in this repository, from a named corpus, under this program's split.
FITTED_IN_REPO = "fitted_in_repo"
#: Read from a public table at run time rather than learned.
RUNTIME_LOOKUP = "runtime_lookup"
#: Measured observations supplied by the user, without model inference.
USER_SUPPLIED_MEASUREMENTS = "user_supplied_measurements"
#: Geometry computed directly from supplied coordinates.
NO_TRAINING_CORPUS = "no_training_corpus"
#: No entry in :data:`TRAINED_ON`. A defect, not a kind.
CORPUS_UNDECLARED = "corpus_undeclared"

#: The closed vocabulary. A kind outside this set is a bug in this file.
OVERLAP_KINDS = frozenset(
    {
        BENCHMARK_LABEL_CORPUS,
        BENCHMARK_LABEL_CORPUS_SIBLING,
        BENCHMARK_LABEL_CORPUS_UNMODELLED,
        PUBLIC_CORPUS_UNALIGNED,
        PRIVATE_CORPUS_OVERLAP_UNKNOWN,
        SIMULATED_LABELS,
        FITTED_IN_REPO,
        RUNTIME_LOOKUP,
        USER_SUPPLIED_MEASUREMENTS,
        NO_TRAINING_CORPUS,
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
    #: The corpus this one was built from, when it was. Two corpora with
    #: different names are not thereby different data: the OASis 9-mer set is
    #: cut from Observed Antibody Space, so ``ablang2_pseudo_log_likelihood``
    #: declaring ``OAS`` and ``oasis_identity_10pct_subjects`` declaring
    #: ``OASis-peptides`` shared nothing by name while resting on one body of
    #: antibodies -- and the pair read as two independent humanness opinions,
    #: which is the case the house rule names outright. Declared here so the
    #: file stops implying an independence it cannot support. **Nothing reads
    #: this yet**: ``overlap_with`` compares a model against a dataset, not a
    #: model against a model, so the pairwise check this would feed does not
    #: exist. A reader of the record can see it; a guard cannot.
    derived_from: str = ""

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
        if self.derived_from:
            # On the record, not only in the table: a reader weighing two
            # opinions needs to see that one corpus was cut from the other.
            record["derived_from"] = self.derived_from
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

CYPSTRATE_ISOFORMS = ("CYP1A2", "CYP2A6", "CYP2B6", "CYP2C8", "CYP2C9", "CYP2C19", "CYP2D6", "CYP2E1", "CYP3A4")
_CYPSTRATE_LABELS = ("human_cyp_substrate", *(isoform.lower() + "_substrate" for isoform in CYPSTRATE_ISOFORMS))

CORPORA: dict[str, Corpus] = {
    "CYPstrate-Hunt-Tian": Corpus(
        name="CYPstrate curated human CYP substrate and non-substrate data",
        publisher="Holmer et al.",
        locator="https://doi.org/10.3390/molecules26154678",
        read_from="The authors curated a core development set from Hunt et al. and Tian et al. "
                  "Individual isoforms used separate training and holdout subsets.",
        rows=1831, unit="compound in the core development set", labels=_CYPSTRATE_LABELS,
    ),
    "Reaxys-SCScore": Corpus(
        name="Reaxys reaction examples used to train SCScore",
        publisher="Elsevier Reaxys",
        locator="https://doi.org/10.1021/acs.jcim.7b00622",
        read_from=(
            "Coley et al. (2018) and the authors' SCScore repository describe approximately 12 million "
            "reactions. Product-over-reactant ordering supplied the training signal."
        ),
        unit="reaction",
    ),
    "SyGMa-Metabolite-Database": Corpus(
        name="Human biotransformations in the Metabolite Database used by Ridder and Wagener",
        publisher="Ridder and Wagener",
        locator="https://pubmed.ncbi.nlm.nih.gov/18311745/",
        read_from="The authors derived the rules and empirical success rates from reported human reactions. "
        "The public release contains the per-rule weights. Compound-level membership is unavailable.",
        labels=("human_metabolite_hypotheses",),
    ),
    "TNP-clinical-VHH-36": Corpus(
        name="TNP clinical-stage VHH descriptor reference",
        publisher="Oxford Protein Informatics Group and Twist Bioscience",
        locator="https://doi.org/10.1038/s42003-026-09594-y",
        read_from="Six descriptor ranges calibrated on 36 clinical domains. The 108-domain assay study included "
                  "36 clinical domains and 72 additional proprietary domains, assayed as Fc fusions.",
        rows=36,
        unit="VHH domain",
    ),
    "NanoBodyBuilder2-SAbDab": Corpus(
        name="SAbDab nanobody structures used in NanoBodyBuilder2 development",
        publisher="Oxford Protein Informatics Group",
        locator="https://doi.org/10.1038/s42003-023-04927-7",
        read_from="ImmuneBuilder paper and Supplementary Note 1. Per-domain training overlap is unverified.",
        unit="structure",
    ),
    "PDBbind-refined-Vina": Corpus(
        name="PDBbind refined set used to fit the original Vina function",
        publisher="PDBbind",
        locator="https://doi.org/10.1002/jcc.21334",
        read_from="Trott and Olson describe fitting their empirical function to structures and affinity measurements.",
        labels=("protein_ligand_binding_affinity",),
        unit="complex",
    ),
    "PRODIGY-affinity-benchmark": Corpus(
        name="Kastritis protein-protein binding affinity benchmark, curated PRODIGY subset",
        publisher="Vangone and Bonvin",
        locator="https://doi.org/10.7554/eLife.07454",
        read_from="The 2015 contact and non-interacting surface regression used the 81-complex subset.",
        labels=("protein_protein_binding_delta_g",),
        rows=81,
        unit="complex",
    ),
    "Niwa-Protein-Sol": Corpus(
        name="Niwa cell-free solubility reference subset",
        publisher="University of Manchester",
        locator="https://protein-sol.manchester.ac.uk/software",
        read_from="October 2017 release, seq_reference_data.txt. Population statistics use 2,396 E. coli proteins.",
        labels=("cell_free_solubility",),
        rows=2396,
        unit="protein",
    ),
    "CATH-S95-v4.4": Corpus(
        name="CATH S95 v4.4.0 labelled structural domains",
        publisher="CATH",
        locator="https://github.com/UCLOrengoGroup/domain-annotation-pipeline",
        read_from="CathDomainList.S95.v4.4.0 paired with the authors' Foldseek S95 index.",
        rows=73535,
        unit="domain",
    ),
    "GearNet-PDB-CATH": Corpus(
        name="PDB chains with CATH fold labels used for the Proteina GearNet classifier",
        publisher="NVIDIA",
        locator="https://arxiv.org/html/2503.00710v1#A7.SS3",
        read_from="Appendix G.3.1 reports 214,564 structures before the train, validation and test split.",
        labels=("cath_class", "cath_architecture", "cath_topology"),
        rows=214564,
        unit="structure",
    ),
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
    "EPA-httk-human-in-vitro": Corpus(
        name="EPA httk bundled human hepatocyte clearance and plasma protein binding measurements",
        publisher="EPA and collaborators",
        locator="https://cran.r-project.org/web/packages/httk/refman/httk.html#chem.physical_and_invitro.data",
        read_from=(
            "The httk 2.7.4 built-in chemical data supply assay-derived human intrinsic clearance and plasma "
            "unbound fraction for nominated reference substances. No optional prediction tables are loaded."
        ),
        labels=("human_hepatocyte_intrinsic_clearance", "human_plasma_unbound_fraction"),
    ),
    "EPA-httkpop-NHANES": Corpus(
        name="EPA HTTK-Pop human physiological resampling data from NHANES",
        publisher="EPA and the US National Center for Health Statistics",
        locator="https://cran.r-project.org/web/packages/httk/refman/httk.html#httkpop_generate",
        read_from=(
            "EPA httk 2.7.4 HTTK-Pop uses NHANES biometrics to simulate variation. "
            "This adapter uses direct resampling and limits the modeled cohort to ages 18-79."
        ),
    ),
    "MolProbity-Top8000": Corpus(
        name="MolProbity quality-filtered protein torsion reference distributions",
        publisher="Richardson Laboratory and collaborators",
        locator="https://pmc.ncbi.nlm.nih.gov/articles/PMC5734394/",
        read_from=(
            "Williams et al. describe the quality-filtered structures behind the empirical MolProbity "
            "Ramachandran and side-chain rotamer reference distributions. CCTBX 2025.11 uses the 8000 "
            "rotamer tables. Probe2 contacts are computed from the submitted atom positions, not this corpus."
        ),
        labels=("ramachandran", "rotamers"),
    ),
    "IPI-DELPHI-PSR": Corpus(
        name="Institute for Protein Innovation in-house DELPHI PSR training set",
        publisher="Institute for Protein Innovation",
        locator="https://zenodo.org/records/21823887",
        read_from=(
            "DELPHI v4 release and provider model registry identify the internal IPI PSR training set. "
            "Individual identities and overlap with other assay corpora cannot be verified. "
            "The separately released Chen et al. DS1 is not this model's training set."
        ),
        labels=("psr_filter",),
    ),
    "TDC-ADMET": Corpus(
        name="Therapeutics Data Commons, ADMET suite",
        publisher="Therapeutics Data Commons",
        locator="https://tdcommons.ai/single_pred_tasks/adme/",
        read_from="docs/screening-implementation-handoff.md and the ADMET-AI model card, 2026-11-18",
        labels=("Caco2", "HIA", "Pgp", "BBB", "PPBR", "VDss", "CYP", "Half_Life", "CL", "hERG", "AMES", "DILI"),
    ),
    "TDC-AqSolDB": Corpus(
        name="AqSolDB aqueous solubility in Therapeutics Data Commons",
        publisher="Sorkun, Khetan and Er. Distributed by Therapeutics Data Commons.",
        locator="https://doi.org/10.1038/s41597-019-0151-1",
        read_from="ADMET-AI 2.0.1 resources/data/admet.csv lists 9,980 compounds for this task. "
        "The original AqSolDB publication reports 9,982 compounds from nine sources. "
        "Individual training split membership is unverified.",
        labels=("Solubility_AqSolDB",),
        rows=9980,
        unit="compounds in the provider task metadata",
    ),
    "TDC-AMES": Corpus(
        name="Ames mutagenicity in Therapeutics Data Commons",
        publisher="Therapeutics Data Commons",
        locator="https://tdcommons.ai/single_pred_tasks/tox/#ames-mutagenicity",
        read_from="ADMET-AI v2.0.1 resources/data/admet.csv lists 7,255 compounds for this task. "
        "Individual training split membership is unverified.",
        labels=("AMES",),
        rows=7255,
        unit="compounds in the provider task metadata",
    ),
    "TDC-Clearance-Microsome-AZ": Corpus(
        name="AstraZeneca human liver microsomal intrinsic clearance in Therapeutics Data Commons",
        publisher="AstraZeneca, distributed by Therapeutics Data Commons",
        locator="https://tdcommons.ai/single_pred_tasks/adme/#clearance-astrazeneca",
        read_from=(
            "ADMET-AI v2.0.1 resources/data/admet.csv lists 1,102 compounds for this human microsomal task. "
            "Individual model training split membership is unverified. The source assay is CHEMBL3301370."
        ),
        labels=("Clearance_Microsome_AZ",),
        rows=1102,
        unit="compounds in the provider task metadata",
    ),
    "TDC-DILI": Corpus(
        name="Xu et al. drug-induced liver injury classification in Therapeutics Data Commons",
        publisher="Therapeutics Data Commons, from a combined historical set described by Xu et al.",
        locator="https://tdcommons.ai/single_pred_tasks/tox/#dili-drug-induced-liver-injury",
        read_from=(
            "ADMET-AI v2.0.1 resources/data/admet.csv and TDC document 475 drugs for this binary task. "
            "The historical set incorporates FDA National Center for Toxicological Research evidence. "
            "Individual training split membership is unverified. This corpus is not FDA DILIrank 2.0."
        ),
        labels=("DILI",),
        rows=475,
        unit="drugs in the provider task metadata",
    ),
    "AstraZeneca-permeability-efflux": Corpus(
        name="AstraZeneca internal permeability and efflux assays",
        publisher="AstraZeneca",
        locator="https://doi.org/10.1021/acsomega.5c04861",
        read_from=(
            "Ivers Ohlsson et al., Data Availability: the released checkpoint was retrained using internal data "
            "collected through May 5, 2024. Training rows are proprietary; exact compound overlap cannot be checked."
        ),
        labels=("Caco2ER", "Caco2Papp", "MDCKER", "NIHMDCKER"),
    ),
    "QupKake-ChEMBL-CREST": Corpus(
        name="QupKake ChEMBL structures with CREST site assignments and ChemAxon pKa estimates",
        publisher="Abarbanel and Hutchison",
        locator="https://github.com/Shualdon/QupKake/tree/f7a294e91929e4e4f0a7100ad5cf5a64a34a7d4f/data",
        read_from="The released compressed CSV contains 1,551,870 rows with CREST site indices and ChemAxon "
        "macro-pKa estimates attributed to the lowest-energy protomer or deprotomer. The assignment is "
        "ambiguous for some multisite compounds.",
        labels=("protonation_site", "deprotonation_site", "simulated_pKa"),
        rows=1551870,
        unit="released rows",
    ),
    "QupKake-experimental-pKa": Corpus(
        name="QupKake experimental pKa transfer-learning corpus",
        publisher="Abarbanel and Hutchison, adapted from Machine-learning-meets-pKa",
        locator="https://doi.org/10.1021/acs.jctc.4c00328",
        read_from="The release contains 5,637 curated experimental records before the 80:20 split. Marvin "
        "assigned reaction sites in the source data. Literature, Novartis and SAMPL were separate evaluations.",
        labels=("aqueous_microscopic_pKa",),
        rows=5637,
        unit="records before splitting",
    ),
    "BayeshERG-release": Corpus(
        name="BayeshERG MLSMR pretraining and hERG fine-tuning partitions",
        publisher="Kim, Park and Nam",
        locator="https://github.com/GIST-CSBL/BayeshERG/tree/25e9466499905a952f9d41cc6bc6886c3f247acb/data",
        read_from="Published Pretraining and Finetuning folders, including 11,457 fine-tuning training rows.",
        labels=("hERG",),
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
        derived_from="OAS",
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
    "P2Rank-Chen11-fpocket": Corpus(
        name="P2Rank Chen11 binding-site training set",
        publisher="P2Rank authors",
        locator="https://github.com/rdk/p2rank/blob/2.5.1/distro/models/readme.md",
        read_from="P2Rank 2.5.1 model notes, chen11-fpocket.ds.",
        labels=("receptor_ligand_binding_pockets",),
    ),
    "Badapple2-PubChem-HTS": Corpus(
        name="Badapple 2 NIH PubChem high-throughput assay snapshot",
        publisher="University of New Mexico Badapple team",
        locator="https://unmtid-dbs.net/download/Badapple2/badapple2.pgdump",
        read_from="Publisher summary archive created 2025-08-08 and the Badapple 2 database documentation.",
        labels=("historical_scaffold_promiscuity",),
    ),
    "PROPKA3-empirical-pKa": Corpus(
        name="Published PROPKA3 empirical pKa parameterization",
        publisher="Olsson, Søndergaard, Rostkowski and Jensen",
        locator="https://doi.org/10.1021/ct100578z",
        read_from=(
            "Experimental protein pKa parameterization and the ligand and coupling extension "
            "(https://doi.org/10.1021/ct200133y). This declaration does not enumerate individual structures."
        ),
        labels=("structure_conditioned_group_pka",),
    ),
    "GFN2-xTB-reference": Corpus(
        name="GFN2-xTB molecular reference properties",
        publisher="Bannwarth, Ehlert and Grimme",
        locator="https://doi.org/10.1021/acs.jctc.8b01176",
        read_from="The GFN2-xTB semiempirical method paper and supporting information.",
        labels=("relative_conformer_energy",),
    ),
    "OpenAntigens-human-2026-11-01": Corpus(
        name="OpenAntigens human portal index, 2026-11-01",
        publisher="Institute for Protein Innovation",
        locator="https://openantigens.org/downloads.html",
        read_from="The publisher portal index and canonical accession membership table. Source terms vary by field.",
        labels=("human_surface_target_evidence",),
    ),
    "BigSolDB-original-organic": Corpus(
        name="Original BigSolDB organic-solvent solubility observations",
        publisher="Krasnov, Mikhaylov, Fedorov and Sosnin",
        locator="https://doi.org/10.5281/zenodo.6984601",
        read_from="Attia et al. converted the organic-solvent measurements to log10 mol/L after excluding water.",
        labels=("organic_solid_solubility_log10_mol_l",),
    ),
    "Gerber-Jazzy-hydration": Corpus(
        name="Gerber experimental hydration free energies used by Jazzy",
        publisher="P. R. Gerber",
        locator="https://doi.org/10.1038/s41598-023-30089-x",
        read_from="Ghiandoni and Caldeweyher (2023) describe the 292 experimental hydration observations used to fit Jazzy.",
        labels=("gas_to_water_hydration_free_energy_kj_mol",),
    ),
}


#: Every model or lookup that can put a number on an answer, and what it saw.
#: Keyed by the method slug the record already carries, so a stored screen and a
#: declaration join on a key that exists on disk rather than on a second name
#: invented here.
TRAINED_ON: dict[str, TrainedOn] = {
    "espsim_gaussian_gasteiger_0_0_1": TrainedOn(
        model="espsim_gaussian_gasteiger_0_0_1",
        corpora=(),
        role="Compares heavy-atom shape and analytic electrostatic overlap for a supplied prealigned ligand pair.",
        declared_from="Bolcato, Heid and Boström (2022), https://doi.org/10.1021/acs.jcim.1c01535, and ESP-Sim 0.0.1.",
        unaligned_kind=NO_TRAINING_CORPUS,
    ),
    "jazzy_neutral_hydration_0_1_4": TrainedOn(
        model="jazzy_neutral_hydration_0_1_4",
        corpora=("Gerber-Jazzy-hydration",),
        heads=("gas_to_water_hydration_free_energy_kj_mol",),
        role="Estimates gas-to-water hydration free energy and donor and acceptor strengths for a nominated neutral form.",
        declared_from="Ghiandoni and Caldeweyher (2023), https://doi.org/10.1038/s41598-023-30089-x, and Jazzy 0.1.4.",
    ),
    "fastsolv_bigsol_ensemble_1_0_1": TrainedOn(
        model="fastsolv_bigsol_ensemble_1_0_1",
        corpora=("BigSolDB-original-organic",),
        heads=("organic_solid_solubility_log10_mol_l",),
        role="Predicts log10 molar solubility for a nominated solid in one pure organic solvent at a declared temperature.",
        declared_from="Attia et al. (2025), https://doi.org/10.1038/s41467-025-62717-7, and Zenodo ensemble 13943074.",
        weights_installed=True,
    ),
    "metfrag_local_fragmenter_2_6_11": TrainedOn(
        model="metfrag_local_fragmenter_2_6_11",
        corpora=(),
        role="Scores in-silico fragments against one supplied measured tandem mass spectrum.",
        declared_from="Ruttkies et al. (2016), https://doi.org/10.1186/s13321-016-0115-9, and MetFrag 2.6.11.",
        unaligned_kind=NO_TRAINING_CORPUS,
    ),
    "openantigens_human_target_index": TrainedOn(
        model="openantigens_human_target_index",
        corpora=("OpenAntigens-human-2026-11-01",),
        heads=("human_surface_target_evidence",),
        role="Exact accession lookup of public target topology and coverage flags.",
        declared_from=(
            "The 2026-11-01 portal snapshot reuses UniProt, PDB, AlphaFold and InterPro evidence. "
            "These sources and the portal share underlying observations."
        ),
    ),
    "crest_gfn2_xtb_imtd_gc_quick": TrainedOn(
        model="crest_gfn2_xtb_imtd_gc_quick",
        corpora=("GFN2-xTB-reference",),
        heads=("relative_conformer_energy",),
        role="Samples ligand conformers and ranks them by gas-phase GFN2-xTB electronic energy.",
        declared_from="The published GFN2-xTB parameterization bundled with CREST 3.0.2.",
    ),
    "propka_empirical_structure_pka_3_5_1": TrainedOn(
        model="propka_empirical_structure_pka_3_5_1",
        corpora=("PROPKA3-empirical-pKa",),
        heads=("structure_conditioned_group_pka",),
        role="Uses the published empirical parameters to estimate group pKa from supplied coordinates.",
        declared_from="PROPKA3 publications and the parameter files bundled in PROPKA 3.5.1.",
    ),
    "badapple2_hiers_pubchem_pscore": TrainedOn(
        model="badapple2_hiers_pubchem_pscore",
        corpora=("Badapple2-PubChem-HTS",),
        role="Read the original per-scaffold pScore and historical assay counters on exact HierS lookup keys.",
        declared_from="Badapple 2.0 paper (10.1021/acs.jcim.5c02297) and pinned scaffold summary dump.",
        heads=("historical_scaffold_promiscuity",),
    ),
    **{
        f"p2rank_{profile}_2_5_1": TrainedOn(
            model=f"p2rank_{profile}_2_5_1",
            corpora=("P2Rank-Chen11-fpocket",),
            role=role,
            declared_from="The P2Rank 2.5.1 bundled model notes and profile definitions.",
            heads=("receptor_ligand_binding_pockets",),
        )
        for profile, role in (
            ("default", "Ranks candidate X-ray receptor pockets using a B-factor feature."),
            ("alphafold", "Ranks candidate receptor pockets with the B-factor feature excluded."),
        )
    },
    **{
        method: TrainedOn(
            model=method, corpora=("CYPstrate-Hunt-Tian",), heads=_CYPSTRATE_LABELS, role=role,
            declared_from="Holmer et al. (2021), DOI 10.3390/molecules26154678, and CYPstrate v0.0.25.",
        )
        for method, role in (
            ("cypstrate_best_performance", "Per-isoform hard-voting substrate classifiers that can abstain."),
            ("cypstrate_full_coverage", "One substrate classifier per isoform, fitted on the same development corpus."),
        )
    },
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
    "sygma_1_1_0_empirical_rules": TrainedOn(
        model="sygma_1_1_0_empirical_rules",
        corpora=("SyGMa-Metabolite-Database",),
        role="Rank individual metabolite structures using empirically weighted human biotransformation rules.",
        declared_from="Ridder and Wagener (2008) and the phase I and II tables in the SyGMa 1.1.0 release.",
        heads=("human_metabolite_hypotheses",),
    ),
    "qupkake_gfn2_xtb_micro_pka": TrainedOn(
        model="qupkake_gfn2_xtb_micro_pka",
        corpora=("QupKake-ChEMBL-CREST", "QupKake-experimental-pKa"),
        role="Site enumeration and aqueous microscopic pKa regression for a single proton transfer.",
        declared_from="Pinned QupKake release datasets and Abarbanel and Hutchison, JCTC (2024).",
        heads=("protonation_site", "deprotonation_site", "simulated_pKa", "aqueous_microscopic_pKa"),
        weights_installed=True,
    ),
    "admet_ai_aqsoldb_v2": TrainedOn(
        model="admet_ai_aqsoldb_v2",
        corpora=("TDC-AqSolDB",),
        role="Small-molecule aqueous log10 molar solubility regression.",
        declared_from="ADMET-AI v2 release checkpoint head and TDC AqSolDB task, reviewed 2026-11-22.",
        heads=("Solubility_AqSolDB",),
        weights_installed=True,
    ),
    "epa_httk_human_css_v2": TrainedOn(
        model="epa_httk_human_css_v2",
        corpora=("EPA-httk-human-in-vitro", "EPA-httkpop-NHANES"),
        role="Adult population oral steady-state simulation using measured human clearance and plasma binding.",
        declared_from="EPA httk 2.7.4 calc_mc_css and HTTK-Pop. Substance identity from the 2025-12-09 DSSTox release.",
        unaligned_kind=SIMULATED_LABELS,
    ),
    "cctbx_molprobity_geometry_v1": TrainedOn(
        model="cctbx_molprobity_geometry_v1",
        corpora=("MolProbity-Top8000",),
        role="Empirical backbone and side-chain torsion references plus separately computed all-atom contacts.",
        declared_from="CCTBX 2025.11 native Ramalyze, Rotalyze and Reduce2/Probe2.",
        heads=("ramachandran", "rotamers"),
        unaligned_kind=RUNTIME_LOOKUP,
    ),
    "supplied_peptide_rbc_assay_v1": TrainedOn(
        model="supplied_peptide_rbc_assay_v1",
        corpora=(),
        role="Review of user-supplied therapeutic peptide red-cell assay observations.",
        declared_from="The supplied peptide red-cell evidence TSV contract. No model is fitted or used.",
        unaligned_kind=USER_SUPPLIED_MEASUREMENTS,
    ),
    "ipi_delphi_onehot_psr_v4": TrainedOn(
        model="ipi_delphi_onehot_psr_v4",
        corpora=("IPI-DELPHI-PSR",),
        role="Paired antibody PSR pass-class prediction for experimental triage.",
        declared_from="DELPHI Zenodo v4 official paired one-hot IPI PSR checkpoint.",
        heads=("psr_filter",),
        unaligned_kind=PRIVATE_CORPUS_OVERLAP_UNKNOWN,
        weights_installed=True,
    ),
    "admet_ai_ames_v2": TrainedOn(
        model="admet_ai_ames_v2",
        corpora=("TDC-AMES",),
        role="Small-molecule bacterial reverse-mutation classification for experimental triage.",
        declared_from="ADMET-AI v2 classification ensemble and the TDC AMES task.",
        heads=("AMES",),
        weights_installed=True,
    ),
    "admet_ai_microsomal_clearance_v2": TrainedOn(
        model="admet_ai_microsomal_clearance_v2",
        corpora=("TDC-Clearance-Microsome-AZ",),
        role="Human liver microsomal intrinsic clearance regression for experimental triage, in uL/min/mg.",
        declared_from="ADMET-AI v2 regression ensemble and the TDC Clearance_Microsome_AZ task.",
        heads=("Clearance_Microsome_AZ",),
        weights_installed=True,
    ),
    "az_gnn_mtl_caco2_v1": TrainedOn(
        model="az_gnn_mtl_caco2_v1",
        corpora=("AstraZeneca-permeability-efflux",),
        role="Native Caco-2 efflux and permeability model scores for experimental triage.",
        declared_from="AstraZeneca GNN-MTL descriptor-free checkpoint, Zenodo 16948542 version 1.",
        heads=("Caco2ER", "Caco2Papp"),
        weights_installed=True,
    ),
    "admet_ai_dili_v2": TrainedOn(
        model="admet_ai_dili_v2",
        corpora=("TDC-DILI",),
        role="Small-molecule DILI label classification for experimental triage, not clinical risk estimation.",
        declared_from="ADMET-AI v2 classification ensemble and the binary TDC DILI task.",
        heads=("DILI",),
        weights_installed=True,
    ),
    "admet_ai_classification_v1": TrainedOn(
        model="admet_ai_classification_v1",
        corpora=("TDC-ADMET",),
        role="Ten-axis small-molecule ADMET probabilities.",
        declared_from="the ADMET-AI publication and docs/ddd-benchmark-readiness.md's contamination audit",
        heads=("Caco2", "HIA", "Pgp", "BBB", "PPBR", "VDss", "CYP", "Half_Life", "CL", "hERG", "AMES", "DILI"),
    ),
    "therapeutic_nanobody_profiler_0.1.0": TrainedOn(
        model="therapeutic_nanobody_profiler_0.1.0",
        corpora=("TNP-clinical-VHH-36", "NanoBodyBuilder2-SAbDab"),
        role="Clinical reference flags for six computed descriptors on the submitted VHH and its modeled structure.",
        declared_from="Gordon et al. (2026) and TNP 0.1.0, with the NanoBodyBuilder2 structure-model corpus.",
        unaligned_kind=SIMULATED_LABELS,
    ),
    "scscore_reaxys_1024bool": TrainedOn(
        model="scscore_reaxys_1024bool",
        corpora=("Reaxys-SCScore",),
        role="Relative synthetic complexity from reaction-derived ordering on a scale of 1 to 5.",
        declared_from="Coley et al. (2018), DOI 10.1021/acs.jcim.7b00622, and the full-Reaxys Boolean release.",
        unaligned_kind=SIMULATED_LABELS,
    ),
    "cellprofiler": TrainedOn(
        model="cellprofiler",
        corpora=(),
        role="Regional intensity measurements after separating DAB and hematoxylin in supplied brightfield images.",
        declared_from="CellProfiler UnmixColors, https://cellprofiler-manual.s3.amazonaws.com/CPmanual/UnmixColors.html",
        unaligned_kind=NO_TRAINING_CORPUS,
    ),
    "pksim": TrainedOn(
        model="pksim",
        corpora=(),
        role="Mechanistic human PBPK simulation of concentration over time for a nominated compound and scenario.",
        declared_from="Open Systems Pharmacology PK-Sim, https://github.com/Open-Systems-Pharmacology/PK-Sim",
        unaligned_kind=NO_TRAINING_CORPUS,
    ),
    "spyrmsd": TrainedOn(
        model="spyrmsd",
        corpora=(),
        role="Symmetry-corrected heavy-atom RMSD against a nominated experimental redocking reference.",
        declared_from="Meli and Biggin (2020), https://doi.org/10.1186/s13321-020-00455-2",
        unaligned_kind=NO_TRAINING_CORPUS,
    ),
    "autodock_vina_fixed_pose": TrainedOn(
        model="autodock_vina_fixed_pose",
        corpora=("PDBbind-refined-Vina",),
        role="Empirical score of one supplied rigid-receptor protein-ligand pose in kcal/mol.",
        declared_from="Trott and Olson (2010), original Vina scoring function",
        heads=("protein_ligand_binding_affinity",),
    ),
    "prodigy_ic_nis": TrainedOn(
        model="prodigy_ic_nis",
        corpora=("PRODIGY-affinity-benchmark",),
        role="Empirical estimate of binding free energy from a protein-protein complex.",
        declared_from="Vangone and Bonvin (2015), contact and non-interacting surface model",
        heads=("protein_protein_binding_delta_g",),
    ),
    "Protein-Sol": TrainedOn(
        model="Protein-Sol",
        corpora=("Niwa-Protein-Sol",),
        role="Sequence-based estimation of cell-free solubility.",
        declared_from="Hebditch et al. 2017 and the authors' October 2017 reference file",
        heads=("cell_free_solubility",),
    ),
    "cath_s95_foldseek": TrainedOn(
        model="cath_s95_foldseek",
        corpora=("CATH-S95-v4.4",),
        role="Reports the classification depth supported by a CATH structural-domain match.",
        declared_from="CATH S95 domain list and UCL Orengo Group domain-annotation-pipeline",
        unaligned_kind=RUNTIME_LOOKUP,
    ),
    "proteina_classifier": TrainedOn(
        model="proteina_classifier",
        corpora=("GearNet-PDB-CATH",),
        role="CATH fold classification with separate learned heads.",
        declared_from="Proteina Appendix G.3.1 and NVIDIA NGC proteina_additional_files 1.0",
        heads=("cath_class", "cath_architecture", "cath_topology"),
    ),
    "bayesherg_mc30_cpu_v1": TrainedOn(
        model="bayesherg_mc30_cpu_v1",
        corpora=("BayeshERG-release",),
        role="Small-molecule hERG blocker classifier with Monte Carlo uncertainty components.",
        declared_from="GIST-CSBL/BayeshERG checkpoint and released training partitions, doi:10.1093/bib/bbac211",
        heads=("hERG",),
    ),
    "bindingdb_morgan_extratrees_v1": TrainedOn(
        model="bindingdb_morgan_extratrees_v1",
        corpora=("BindingDB",),
        role="In-repo regressor for the three axes ADMET-AI does not cover.",
        declared_from="ligand_regressor.load_corpus and the BindingDB subset recorded for training",
        heads=("nav1_5", "bcrp", "bsep"),
        unaligned_kind=FITTED_IN_REPO,
    ),
    "bindingdb_chemprop_dmpnn_v1": TrainedOn(
        model="bindingdb_chemprop_dmpnn_v1",
        corpora=("BindingDB",),
        role="The same three axes, D-MPNN backend.",
        declared_from="ligand_regressor.load_corpus and the BindingDB subset recorded for training",
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
        # Nothing is fitted: humanness.oasis_identity counts a design's 9-mers against a
        # packaged table, which is the RUNTIME_LOOKUP mechanism octant already declares.
        # The public-corpus default denied exactly the risk this kind warns about -- a
        # humanness gold set drawn from the same table this screen reads.
        unaligned_kind=RUNTIME_LOOKUP,
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
            # A sibling requires another label from the same declared assay family.
            family = set(corpus.family_of(dataset)) - {dataset}
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
    PRIVATE_CORPUS_OVERLAP_UNKNOWN: (
        "The model uses non-public training rows, so per-item overlap and independence cannot be verified."
    ),
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
    NO_TRAINING_CORPUS: "The method computes its result from the supplied inputs and has no training set.",
    USER_SUPPLIED_MEASUREMENTS: (
        "The screen uses only supplied measurements. It does not authenticate the lab report or predict new results."
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
    if kind == NO_TRAINING_CORPUS:
        return f"{model} vs {dataset}: {kind} (computed from supplied inputs)"
    names = ", ".join(trained_on(model).corpora) or "nothing recorded"
    return f"{model} vs {dataset}: {kind} (trained on {names})"


def unresolved_corpora() -> list[str]:
    """Corpus names a model claims that :data:`CORPORA` does not define. Empty is healthy."""
    named = {name for found in TRAINED_ON.values() for name in found.corpora}
    return sorted(named - set(CORPORA))
