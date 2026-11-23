"""Where each screening threshold came from, in a form a reader can act on.

Sibling to :mod:`rfproteina.disease.provenance`, which answers *which code*
produced a record. This one answers *where its numbers came from*. A verdict
is reproducible against its inputs, its cuts, its bar and its code; recording
the cuts makes it reproducible, and recording what kind of thing each cut is
makes it interpretable.

The gap this closes is small and specific. Twelve cuts are written into every
screen. Two of them named a source. From the values alone a reader cannot tell
that ``instability_max`` of 40 is a published prediction rule while
``gravy_max`` of 0.1 is a number this program chose, and a
``sequence_pass: false`` decided by the second reads exactly like one decided
by the first.

The kinds are a closed vocabulary, so a consumer can branch on them instead of
matching strings:

``published``
    A paper states this threshold for this purpose. It may still be the wrong
    threshold for your design, but someone defended it in print.
``recommended``
    An external body documents it as recommended practice without a paper
    behind the specific number. IEDB's class I selection rule is the example.
``convention``
    This program chose it. No publication and no external recommendation. A
    pass against one of these is a statement about this program's strictness,
    not about the literature.
``caller_supplied``
    The operator passes it, and it is ``None`` until they do. An axis with one
    of these unset reports ``threshold_not_set`` rather than guessing.
``borrowed``
    Another project's default, carried across. Not a published value, and the
    definition or the scale it was fitted against may differ from the one it is
    applied to here. Every such entry says which.
``reported_only``
    Measured and recorded with no cut adopted, so a reader does not go looking
    for a pass that was deliberately not taken.
``measurement_parameter``
    Not a pass bar at all: a constant that shapes a measurement. Recorded so
    the number stays reproducible, and classified so nobody reads it as a cut.
``unrecorded``
    No entry in :data:`ALL_PROVENANCE`. A defect, not a kind.

Every classification is taken from what the defining constant already says
about itself in ``liability.py``, ``screen.py`` or ``combinations.py``. None is
inferred, and where this file and a constant disagree the constant is right.

A cut that reaches :func:`describe_cuts` with no entry is reported as
``unrecorded`` rather than left bare, and ``tests/test_disease_threshold_provenance.py``
fails on it, so a new threshold cannot be added without saying where it came from.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

#: A paper states this threshold for this purpose.
PUBLISHED = "published"
#: An external body recommends it; no paper defends the specific number.
RECOMMENDED = "recommended"
#: This program's own choice. Nothing external behind it.
CONVENTION = "convention"
#: The operator supplies it; ``None`` until they do.
CALLER_SUPPLIED = "caller_supplied"
#: Another project's default, carried across. Not a published value, and the
#: definition or the scale it was fitted against may differ from the one it is
#: applied to here. Every such entry says which.
BORROWED = "borrowed"
#: Measured and recorded, with no cut adopted. Distinct from
#: ``measurement_parameter``: this is a quantity that could carry a bar and
#: deliberately does not, so a reader does not go looking for the pass.
REPORTED_ONLY = "reported_only"
#: Shapes a measurement rather than deciding a pass.
MEASUREMENT_PARAMETER = "measurement_parameter"
#: No entry in :data:`ALL_PROVENANCE`. A defect, not a kind.
UNRECORDED = "unrecorded"

#: The closed vocabulary. A kind outside this set is a bug in this file.
KINDS = frozenset(
    {
        PUBLISHED,
        RECOMMENDED,
        CONVENTION,
        CALLER_SUPPLIED,
        BORROWED,
        REPORTED_ONLY,
        MEASUREMENT_PARAMETER,
        UNRECORDED,
    }
)


@dataclass(frozen=True)
class Provenance:
    """What kind of threshold this is, and the shortest honest account of it.

    ``source`` is the stable slug already written beside two of the cuts, kept
    unchanged so existing readers and tests do not move. ``citation`` is empty
    for anything that is not ``published``: an empty citation is the point, not
    an omission to fill in later.

    ``caveat`` carries the condition under which the number stops applying, and
    is the field most worth reading. ``ada_median_rank_max`` is a genuinely
    published threshold that is still refused on most machines, because it is
    defined on a rank scale the installed predictor does not produce.
    """

    kind: str
    source: str
    citation: str = ""
    caveat: str = ""

    def as_record(self) -> dict[str, str]:
        """The flat form stored in a screen. Empty fields are dropped."""
        record = {"kind": self.kind, "source": self.source}
        if self.citation:
            record["citation"] = self.citation
        if self.caveat:
            record["caveat"] = self.caveat
        return record


#: Every cut ``screen._applied_cuts`` writes, and where it came from.
CUT_PROVENANCE: dict[str, Provenance] = {
    "gravy_max": Provenance(
        kind=CONVENTION,
        source="program_convention",
        caveat=(
            "No published bar says where a developable protein stops: ProtParam's GRAVY "
            "section is one definitional sentence and names no cutoff, the dedicated "
            "benchmark of hydrophobicity scales against a measured antibody property "
            "proposes none, and the canonical developability threshold tables are wet-lab "
            "readouts that do not mention GRAVY at all. The bar sits just above neutral: on "
            "the Kyte-Doolittle scale positive is net hydrophobic, so 0.1 asks that the "
            "chain not be APPRECIABLY net hydrophobic and admits a slightly hydrophobic "
            "one. The only community touchstone is the sign split at 0.0, which is stricter "
            "than this. It is this program's strictness, not a literature value."
        ),
    ),
    "instability_max": Provenance(
        kind=PUBLISHED,
        source="expasy_protparam_guruprasad_1990",
        citation=(
            "Guruprasad, Reddy and Pandit, Protein Engineering 4(2):155-161 (1990), as applied "
            "by ExPASy ProtParam: a protein whose instability index is below 40 is predicted "
            "as stable."
        ),
    ),
    "clash_distance_angstrom": Provenance(
        kind=MEASUREMENT_PARAMETER,
        source="non_bonded_clash_distance",
        caveat="Defines what counts as a clash for the reported fraction. No pass cut is adopted.",
    ),
    "mhc_i_rank_max": Provenance(
        kind=RECOMMENDED,
        source="iedb_class_i_selection",
        caveat=(
            "IEDB's recommended class I selection: percentile rank <= 1 for each (allele, "
            "length) combination, or IC50 <= 500 nM. IEDB states the two as alternatives, "
            "and this program applies them as a disjunction. IEDB itself notes the absolute "
            "IC50 correlates with immunogenicity better than the rank, and allele-specific "
            "cuts better still: Paul et al. 2013 fitted 60 nM to 944 nM across 38 HLA-A/B "
            "alleles, a sixteenfold spread this flat cut does not carry."
        ),
    ),
    "mhc_i_ic50_nm_max": Provenance(
        kind=RECOMMENDED,
        source="iedb_class_i_selection",
        caveat=(
            "The IC50 half of the same IEDB rule. Either condition selects a peptide. The "
            "number is not IEDB's own: it originates with Sette et al., Journal of "
            "Immunology 153(12):5586-92 (1994), which says an affinity threshold of "
            "APPROXIMATELY 500 nM, preferably 50 nM or less, determines the capacity of a "
            "peptide to elicit a CTL response. IEDB inherits it rather than deriving it, so "
            "the exact 500.0 here is this program's sharpening of an approximate figure."
        ),
    ),
    "mhc_ii_rank_max": Provenance(
        kind=RECOMMENDED,
        source="iedb_class_ii_recommended_rank",
        caveat=(
            "IEDB's recommended class II percentile rank. It assumes a continuous rank: the "
            "installed TEPITOPE scorer emits only the integer bins 1 through 10, plus 100 for a "
            "score under every published threshold, so at a cut of 10 ten of its eleven bins "
            "pass and the reference density is refused as implausible."
        ),
    ),
    "mhc_ii_ic50_nm_max": Provenance(
        kind=RECOMMENDED,
        source="iedb_class_ii_binding_cutoff",
        caveat=(
            "The IC50 binding cutoff for class II, 1,000 nM and not the 500 nM of class I. "
            "Unused by a scorer that reports no IC50, which IEDB's current recommended "
            "class II method is. IEDB cites Southwood et al. 1998 for it, but the same "
            "group's later benchmark calls 1,000 nM chosen on an expert recommendation and "
            "'still somewhat arbitrary', so it is nearer this program's own convention kind "
            "than a fitted value. Do not read the 500 nM in IEDB's class II affinity BANDS "
            "as this cutoff: those bands label high, intermediate and low affinity."
        ),
    ),
    "ada_median_rank_max": Provenance(
        kind=PUBLISHED,
        source="paul_2015_dr_panel_breadth",
        citation=(
            "Paul et al., Journal of Immunological Methods 422:28-34 (2015), "
            "doi:10.1016/j.jim.2015.03.022: a blind-validated universal rule, a median "
            "percentile rank of 20 or better across the seven-allele DR panel."
        ),
        caveat=(
            "Defined on the IEDB consensus percentile rank. A table produced by any other "
            "method is on a different scale, and the axis reports rank_scale_unknown rather "
            "than applying the number anyway. A published threshold is not a portable one."
        ),
    ),
    "ligand_panel_nm_max": Provenance(
        kind=CONVENTION,
        source="program_convention_10uM",
        caveat=(
            "10 uM for every panel axis, this program's screening convention. No entry is from "
            "a published rule: hERG is normally judged as a margin over free Cmax rather than "
            "at a fixed concentration, and BSEP inhibition in DILI work is commonly discussed "
            "nearer 25-50 uM."
        ),
    ),
    "plasma_abundance_max": Provenance(
        kind=CALLER_SUPPLIED,
        source="caller_supplied",
        caveat="No published value to fall back on. Unset until the operator passes one.",
    ),
    "plasma_hub_max": Provenance(
        kind=CALLER_SUPPLIED,
        source="caller_supplied",
        caveat="No published value to fall back on. Unset until the operator passes one.",
    ),
    "plasma_fold_max": Provenance(
        kind=CALLER_SUPPLIED,
        source="caller_supplied",
        caveat="No published value to fall back on. Unset until the operator passes one.",
    ),
}

#: The interface-quality bars in ``interface_quality.REFERENCE_THRESHOLDS``.
#:
#: These are the bars most in need of a kind. Three are BindCraft's defaults
#: carried across to proxies BindCraft never fitted, and each is slack against
#: this program's own measured distribution by a wide margin: a reader seeing
#: ``interface_polar_contacts >= 3`` cleared has learned almost nothing, since
#: the measured median is 18. They are floors that catch a catastrophe, not
#: discriminators, and the caveat says so rather than leaving the bar to imply
#: otherwise. The two with no cut say that too, so nobody hunts for a pass that
#: was deliberately not adopted.
#:
#: Every figure below is quoted from ``REFERENCE_THRESHOLDS`` in
#: ``interface_quality.py``, which is the runtime source of truth.
INTERFACE_PROVENANCE: dict[str, Provenance] = {
    "interface_residues_binder": Provenance(
        kind=BORROWED,
        source="bindcraft_average_n_interface_residues",
        caveat=(
            "BindCraft's interface-residue floor of 7 is a project default, not a published value. "
            "BindCraft counts contacting residues. The local count uses per-residue surface burial."
        ),
    ),
    "interface_polar_contacts": Provenance(
        kind=BORROWED,
        source="bindcraft_average_n_interface_hbonds",
        caveat=(
            "BindCraft's floor of 3 applies to Rosetta hydrogen bonds. DEFINITION DIFFERS. The local "
            "measurement counts cross-interface N/O pairs within 3.5 A."
        ),
    ),
    "buried_unsatisfied_polar_atoms": Provenance(
        kind=BORROWED,
        source="bindcraft_average_n_interface_unsat_hbonds",
        caveat=(
            "BindCraft uses a Rosetta buried-unsatisfied hydrogen-bond filter. The local heavy-atom "
            "proxy uses different burial and satisfaction rules. The Worker A calibration sample "
            "contained 23 unlabeled designs."
        ),
    ),
    "shape_complementarity": Provenance(
        kind=REPORTED_ONLY,
        source="lawrence_colman_1993_scale_differs",
        caveat=(
            "Lawrence & Colman 1993. SCALE DIFFERS because this proxy uses contact patches. The "
            "historical median of 0.166 used the earlier burial calculation."
        ),
    ),
    "dsasa_total_a2": Provenance(
        kind=REPORTED_ONLY,
        source="no_cut_adopted",
        caveat=(
            "Area in square angstroms. BindCraft reports Rosetta delta SASA and a separate interface "
            "SASA percentage. This Biotite measurement has no calibrated cutoff."
        ),
    ),
}

#: Bars that live inside an individual screen, keyed ``<module>.<bar>``.
#:
#: Dotted because this family does collide: ``evalue_max`` is a bar in both
#: novelty and offtarget, with the same value and different reasoning, and a
#: flat key would silently give one of them the other's provenance.
#:
#: Each entry is classified from the ``status`` or ``source`` string the
#: defining module already carries, not from this file's own judgement.
SCREEN_BAR_PROVENANCE: dict[str, Provenance] = {
    # --- novelty.NOVELTY_THRESHOLDS -------------------------------------
    "novelty.same_fold_tmscore": Provenance(
        kind=PUBLISHED,
        source="xu_zhang_2010_same_fold",
        citation=(
            "Xu and Zhang, Bioinformatics 26(7):889-895 (2010): two structures with a "
            "TM-score above 0.5 are in the same fold."
        ),
        caveat=(
            "The polarity is the opposite of most axes here: a hit means the fold is "
            "already known, which is to say NOT novel. The boundary point is the "
            "program's rather than the paper's: Xu and Zhang say a TM-score ABOVE 0.5, "
            "and the verdict here is inclusive at exactly 0.5. One point on a "
            "continuous measure, and no stored verdict has turned on it, but a "
            "published cut applied with a different boundary is not quite the "
            "published cut."
        ),
    ),
    "novelty.evalue_max": Provenance(
        kind=CONVENTION,
        source="program_default_pending_calibration",
        caveat=(
            "Deliberately permissive at 1e6, and not calibrated. With --alignment-type 1 Foldseek "
            "applies -e at its structurealign stage, which runs before TMalign, so a tight cut "
            "decides this axis instead of the TM-score. At the former 1e-3 an atom-exact copy of a "
            "PDB entry returned no rows and read novel, and Foldseek's own default of 10 is still "
            "too tight for that case. The off-target axis keeps 1e-3: there the E-value is the "
            "reading, not a prefilter in front of one."
        ),
    ),
    # --- offtarget.OFFTARGET_THRESHOLDS ---------------------------------
    "offtarget.identity_min": Provenance(
        kind=CONVENTION,
        source="program_default_pending_calibration",
        caveat="No published cross-reactivity identity cut exists to adopt.",
    ),
    "offtarget.coverage_min": Provenance(
        kind=CONVENTION,
        source="program_default_pending_calibration",
        caveat="The minimum of query and target coverage. A program default, not a literature value.",
    ),
    "offtarget.evalue_max": Provenance(
        kind=CONVENTION,
        source="program_default_pending_calibration",
        caveat=(
            "Not calibrated. This was the same number as the novelty E-value cut until 2026-11-20, "
            "when novelty moved to 1e6; the two now differ by nine orders of magnitude and are "
            "separate decisions. Here the E-value is the reading itself, so a tight cut is the "
            "measurement. In novelty it sat in front of TMalign as a prefilter and decided that "
            "axis instead of the TM-score, which is why only that side was relaxed."
        ),
    ),
    # --- stability.STABILITY_BARS ---------------------------------------
    "stability.fraction_positions_improvable": Provenance(
        kind=CONVENTION,
        source="program_default_pending_calibration",
        caveat=(
            "ThermoMPNN publishes no design-level pass mark for whether a protein is stable "
            "enough, and none is invented here. Do not gate on this until it is calibrated."
        ),
    ),
    "stability.n_strongly_stabilizing": Provenance(
        kind=REPORTED_ONLY,
        source="informational_no_gate",
        caveat="A count of mutations below the strongly-stabilising ddG. No bar is adopted.",
    ),
    "stability.most_stabilizing_ddg": Provenance(
        kind=REPORTED_ONLY,
        source="informational_no_gate",
        caveat="The largest stability headroom available. Reported for triage, not gated.",
    ),
    # --- ligand_alerts.PROPERTY_BARS ------------------------------------
    "ligand_alerts.mol_weight": Provenance(
        kind=PUBLISHED,
        source="lipinski_1997_rule_of_five",
        citation="Lipinski et al. 1997, rule of five: molecular weight at or below 500 Da.",
        caveat="A rule for oral small-molecule likeness. It says nothing about a biologic.",
    ),
    "ligand_alerts.clogp": Provenance(
        kind=PUBLISHED,
        source="lipinski_1997_rule_of_five",
        citation="Lipinski et al. 1997, rule of five: ClogP at or below 5.",
        caveat=(
            "METHOD DIFFERS: the measured number is RDKit's Wildman-Crippen atom-based logP "
            "(``Crippen.MolLogP``, JCICS 39:868-873 (1999)), not CLOGP. The rule of five quotes "
            "a different cut for a different estimator -- MLogP > 4.15 against CLogP > 5 -- so "
            "the 5 is not estimator-independent. No cut on a Wildman-Crippen logP is published."
        ),
    ),
    "ligand_alerts.h_bond_donors": Provenance(
        kind=PUBLISHED,
        source="lipinski_1997_rule_of_five",
        citation="Lipinski et al. 1997, rule of five: at most 5 hydrogen-bond donors.",
    ),
    "ligand_alerts.h_bond_acceptors": Provenance(
        kind=PUBLISHED,
        source="lipinski_1997_rule_of_five",
        citation="Lipinski et al. 1997, rule of five: at most 10 hydrogen-bond acceptors.",
    ),
    "ligand_alerts.sa_score": Provenance(
        kind=CONVENTION,
        source="program_default_pending_calibration",
        caveat=(
            "Ertl and Schuffenhauer 2009 define the scale, 1 easy to 10 hard, but no cut on "
            "it. The 6.0 here is this program's, and synthetic accessibility is a cost "
            "signal rather than a safety finding."
        ),
    ),
    "ligand_alerts.qed": Provenance(
        kind=CONVENTION,
        source="program_default_pending_calibration",
        caveat=(
            "Bickerton et al. 2012 define QED but publish no pass mark. The 0.5 here is this "
            "program's, and a QED bar is a ranking preference rather than a safety finding."
        ),
    ),
}

#: Every threshold this module can classify. The cut and interface families use
#: bare names and do not overlap; the per-screen family is dotted because it
#: does.
ALL_PROVENANCE: dict[str, Provenance] = {
    **CUT_PROVENANCE,
    **INTERFACE_PROVENANCE,
    **SCREEN_BAR_PROVENANCE,
}

#: Where the per-cut records are stored inside ``cuts``.
PROVENANCE_KEY = "provenance"

#: What a cut with no entry gets, so the gap shows in the verdict rather than hiding.
UNRECORDED_PROVENANCE = Provenance(
    kind=UNRECORDED,
    source="unrecorded",
    caveat=(
        "This threshold has no entry in threshold_provenance.ALL_PROVENANCE, which is the "
        "three families together. Treat the pass it decides as unsourced."
    ),
)


def provenance_of(cut: str) -> Provenance:
    """Where ``cut`` came from, from either family. Never raises.

    A verdict always says something: a threshold nobody registered comes back
    ``unrecorded`` with a caveat saying to treat the pass it decides as
    unsourced, which is the honest answer and a visible one.
    """
    return ALL_PROVENANCE.get(cut, UNRECORDED_PROVENANCE)


def cut_names(cuts: Mapping[str, object]) -> list[str]:
    """The cut names in ``cuts``: not the ``*_source`` strings, not the provenance block."""
    return sorted(name for name in cuts if not name.endswith("_source") and name != PROVENANCE_KEY)


def describe_cuts(cuts: Mapping[str, object]) -> dict[str, dict[str, str]]:
    """One provenance record per cut, keyed by cut name.

    Keyed rather than flattened into ``<cut>_kind`` and ``<cut>_citation``
    pairs so the verdict gains one key instead of three dozen, which matters to
    every consumer holding an allowlist of record keys.
    """
    return {name: provenance_of(name).as_record() for name in cut_names(cuts)}


def unsourced(cuts: Mapping[str, object]) -> list[str]:
    """Thresholds with no provenance entry, from either family. Empty is healthy."""
    return [name for name in cut_names(cuts) if name not in ALL_PROVENANCE]


def describe_interface_thresholds() -> dict[str, dict[str, str]]:
    """One provenance record per interface-quality bar, keyed by bar name.

    Read straight from :data:`INTERFACE_PROVENANCE` rather than from a measured
    record, because these bars are fixed for the program: unlike the target
    cuts, no caller supplies them per run.
    """
    return {name: found.as_record() for name, found in sorted(INTERFACE_PROVENANCE.items())}


def summary_line(cut: str, value: object) -> str:
    """One line a human can read, for a report or a console summary."""
    found = provenance_of(cut)
    shown = "unset" if value is None else value
    if found.kind == PUBLISHED:
        return f"{cut} = {shown} ({found.kind}: {found.citation})"
    if found.caveat:
        return f"{cut} = {shown} ({found.kind}: {found.caveat})"
    return f"{cut} = {shown} ({found.kind})"
